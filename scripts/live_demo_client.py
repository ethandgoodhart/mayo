"""Mac-side client for the Alpamayo-R1 live-camera demo.

Captures webcam frames with cv2, streams JPEG-encoded frames to the Modal
server over WebSocket (wss), and renders a side-by-side display:
  - left: live webcam preview
  - right: top-down BEV plot of the latest predicted trajectory

Tracks round-trip and GPU Hz. Use ESC or Ctrl-C to quit.

Usage:
  pip install opencv-python websockets msgpack numpy matplotlib
  python3 scripts/live_demo_client.py <wss://...modal.run/ws>

You can get the WS URL from `modal deploy scripts/live_demo_server.py` (look
for the "https://...modal.run" URL and append /ws, swapping https -> wss).
"""

import argparse
import asyncio
import io
import sys
import time
from collections import deque

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import urllib.request
import urllib.error
import websockets


def encode_jpeg(frame_bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


CAM_LAYOUT = [  # (row, col, name) — 2x2 dashcam-style grid on left half
    (0, 0, "front_wide"),
    (0, 1, "front_tele"),
    (1, 0, "cross_left"),
    (1, 1, "cross_right"),
]


TITLE_STRIP_H = 38  # pixel rows reserved at top of BEV for cv2-overlaid title


def render_bev_panel(pred_xy):
    """Render just the BEV (matplotlib) with NO title — the title is drawn live
    via cv2 at 60 fps over a cached copy of this image. Returns BGR numpy."""
    fig = plt.figure(figsize=(7, 6), dpi=110)
    ax_bev = fig.add_subplot(1, 1, 1)
    if pred_xy is not None and len(pred_xy) >= 2:
        T = pred_xy.shape[0]
        ax_bev.plot(-pred_xy[:T, 1], pred_xy[:T, 0], "b-", lw=2, label="Pred")
        ax_bev.scatter(-pred_xy[T - 1, 1], pred_xy[T - 1, 0], c="b", s=30, marker="o")
        ax_bev.scatter([0], [0], c="k", s=60, marker="^", label="ego")
        all_x = np.concatenate([pred_xy[:T, 0], [0]])
        all_y = np.concatenate([-pred_xy[:T, 1], [0]])
        pad = 2.0
        span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 10.0)
        cx, cy = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        ax_bev.set_xlim(cy - span / 2 - pad, cy + span / 2 + pad)
        ax_bev.set_ylim(cx - span / 2 - pad, cx + span / 2 + pad)
        ax_bev.legend(loc="lower right", fontsize=9)
    else:
        ax_bev.scatter([0], [0], c="k", s=60, marker="^")
        ax_bev.set_xlim(-10, 10)
        ax_bev.set_ylim(-5, 15)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("y (left+, m)")
    ax_bev.set_ylabel("x (forward+, m)")
    # Leave vertical space at the top for cv2 to draw the live title.
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.97)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3]
    bgr = rgb[..., ::-1].copy()
    plt.close(fig)
    # Blank out top strip so cv2 overlay sits on clean white.
    bgr[:TITLE_STRIP_H] = 255
    return bgr


def overlay_title(bev_bgr, label, hz, stale_s, warming):
    """Draw live title over the reserved top strip of a cached BEV."""
    out = bev_bgr  # mutate in place; caller already owns a ref
    if warming:
        text = f"{label}  |  awaiting first prediction ..."
    else:
        text = f"{label}  |  {hz:.2f} Hz  |  {stale_s*1000:.0f}ms stale"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x = max(10, (out.shape[1] - tw) // 2)
    out[:TITLE_STRIP_H] = 255
    cv2.putText(out, text, (x, TITLE_STRIP_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_cam_panel(frame_bgr, sim_t, panel_h=660):
    """Fast cv2-native 2x2 dashcam grid from one BGR frame (duplicated 4x).
    Returns BGR image. Runs at camera fps (no matplotlib)."""
    suptitle_h = 30     # top strip for sim_t
    row_label_h = 20    # per-tile label strip
    row_gap = 10
    col_gap = 10
    side_pad = 10
    usable_h = panel_h - suptitle_h - 2 * row_label_h - row_gap
    tile_h = usable_h // 2
    h, w = frame_bgr.shape[:2]
    tile_w = int(w * tile_h / h)
    panel_w = 2 * tile_w + col_gap + 2 * side_pad
    canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, f"sim_t={sim_t:.2f}s",
                (panel_w // 2 - 70, suptitle_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    tile = cv2.resize(frame_bgr, (tile_w, tile_h))
    for r, c, name in CAM_LAYOUT:
        x0 = side_pad + c * (tile_w + col_gap)
        y0 = suptitle_h + r * (tile_h + row_label_h + row_gap)
        cv2.putText(canvas, name.replace("_", " "),
                    (x0 + 4, y0 + row_label_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        canvas[y0 + row_label_h:y0 + row_label_h + tile_h, x0:x0 + tile_w] = tile
    return canvas


def compose_panels(cam_panel_bgr, bev_bgr):
    ch, cw = cam_panel_bgr.shape[:2]
    bh, bw = bev_bgr.shape[:2]
    target_h = max(ch, bh)
    if ch != target_h:
        scale = target_h / ch
        cam_panel_bgr = cv2.resize(cam_panel_bgr,
                                   (int(cw * scale), target_h))
    if bh != target_h:
        scale = target_h / bh
        bev_bgr = cv2.resize(bev_bgr, (int(bw * scale), target_h))
    return np.concatenate([cam_panel_bgr, bev_bgr], axis=1)


async def run(ws_url, cam_index, target_fps, replicas):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open camera {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[client] webcam opened: {int(cap.get(3))}x{int(cap.get(4))}")

    start_t = time.perf_counter()
    state = {
        "latest_pred": None,
        "latest_pred_t": None,
        "latest_raw": None,
        "gpu_ms": 0.0,
        "total_ms": 0.0,
        "recv_count": 0,
        "send_count": 0,
        "last_recv_t": start_t,
        "hz": 0.0,
        "warming": True,
        "server_hw": None,
        "stop": False,
        "start_t": start_t,
        # Rolling 10s window of (arrival_time, stale_ms_at_arrival).
        # stale_ms_at_arrival = interarrival = time since previous pred.
        "stale_samples": deque(),
        "avg_stale_ms_10s": 0.0,
    }
    WINDOW_S = 10.0

    min_frame_interval = 1.0 / max(target_fps, 0.1)

    http_url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    if http_url.endswith("/ws"):
        http_url = http_url[:-3] + "/"
    print(f"[client] pre-warming container via {http_url} (blocks until model is loaded) ...")
    t_warm = time.perf_counter()
    for attempt in range(60):
        try:
            req = urllib.request.Request(http_url)
            resp = urllib.request.urlopen(req, timeout=600)
            body = resp.read(200)
            print(f"[client] container ready after {time.perf_counter()-t_warm:.1f}s: {body[:120]!r}")
            break
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[client]   attempt {attempt+1}: {type(e).__name__} ({e}), retrying ...")
            await asyncio.sleep(5)
    else:
        raise RuntimeError("container never became ready")

    print(f"[client] opening {replicas} WebSocket connection(s) to {ws_url} ...")
    ws_list = []
    # Pre-warm multiple containers in parallel by firing extra HTTP GETs that
    # Modal's @concurrent(max_inputs=1) will route to distinct replicas.
    async def _prewarm():
        def _get():
            try:
                urllib.request.urlopen(http_url, timeout=600).read(1)
            except Exception:
                pass
        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(None, _get)
            for _ in range(replicas)
        ])
    await _prewarm()
    for i in range(replicas):
        w = await websockets.connect(
            ws_url, max_size=8 * 1024 * 1024,
            ping_interval=20, ping_timeout=60, open_timeout=120,
        )
        ws_list.append(w)
        print(f"[client]   ws #{i} connected")
        # Give Modal's scheduler time to commit this connection to one container
        # before the next connect lands (otherwise both can land on the only
        # fully-warm container).
        if i < replicas - 1:
            await asyncio.sleep(3)
    print(f"[client] all {replicas} connections up; waiting for hello ...")

    try:
        async def capture():
            """Pull frames from the webcam as fast as possible, store latest."""
            loop = asyncio.get_event_loop()
            while not state["stop"]:
                ok, frame = await loop.run_in_executor(None, cap.read)
                if not ok:
                    await asyncio.sleep(0.005)
                    continue
                state["latest_raw"] = frame
                await asyncio.sleep(0)  # yield

        async def sender():
            next_send = time.perf_counter()
            idx = 0
            while not state["stop"]:
                now = time.perf_counter()
                if now < next_send:
                    await asyncio.sleep(next_send - now)
                next_send = now + min_frame_interval
                frame = state.get("latest_raw")
                if frame is None:
                    await asyncio.sleep(0.005)
                    continue
                if state["server_hw"] is not None:
                    H, W = state["server_hw"]
                    frame_send = cv2.resize(frame, (W, H))
                else:
                    frame_send = frame
                jpeg = encode_jpeg(frame_send, quality=80)
                # Round-robin across replicas so each server keeps its own 4-frame
                # temporal buffer + each GPU does half the work.
                target = ws_list[idx % replicas]
                idx += 1
                try:
                    await target.send(msgpack.packb({"jpegs": [jpeg, jpeg, jpeg, jpeg]}))
                    state["send_count"] += 1
                except websockets.ConnectionClosed:
                    state["stop"] = True
                    return

        async def receiver(ws, replica_idx):
            while not state["stop"]:
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    state["stop"] = True
                    return
                msg = msgpack.unpackb(raw, raw=False)
                if msg.get("hello"):
                    state["server_hw"] = (int(msg["H"]), int(msg["W"]))
                    print(f"[client] ws#{replica_idx} target resolution: {msg['W']}x{msg['H']}")
                    continue
                if msg.get("warming"):
                    # Wait for ALL replicas to have filled their temporal buffers.
                    continue
                state["warming"] = False
                shape = tuple(msg["pred_shape"])
                pred = np.frombuffer(msg["pred_xy"], dtype=np.float32).reshape(shape)
                state["latest_pred"] = pred
                now = time.perf_counter()
                prev_pred_t = state["latest_pred_t"]
                state["latest_pred_t"] = now
                state["gpu_ms"] = float(msg["gpu_ms"])
                state["total_ms"] = float(msg["total_ms"])
                state["recv_count"] += 1
                state["last_recv_t"] = now
                # Record this prediction's "stale at arrival" = time since previous
                # prediction, same notion as the title's "XXXms stale" displayed
                # between preds. First pred has no previous — skip.
                if prev_pred_t is not None:
                    stale_ms = (now - prev_pred_t) * 1000.0
                    state["stale_samples"].append((now, stale_ms))
                # Drop samples older than WINDOW_S.
                cutoff = now - WINDOW_S
                while state["stale_samples"] and state["stale_samples"][0][0] < cutoff:
                    state["stale_samples"].popleft()
                if state["stale_samples"]:
                    mean_stale_ms = sum(s for _, s in state["stale_samples"]) / len(
                        state["stale_samples"]
                    )
                    state["avg_stale_ms_10s"] = mean_stale_ms
                    state["hz"] = 1000.0 / mean_stale_ms if mean_stale_ms > 0 else 0.0

        async def logger():
            t0 = time.perf_counter()
            last_send = 0
            last_recv = 0
            last_t = t0
            while not state["stop"]:
                await asyncio.sleep(1.0)
                now = time.perf_counter()
                dt = now - last_t
                send_rate = (state["send_count"] - last_send) / dt
                recv_rate = (state["recv_count"] - last_recv) / dt
                last_send, last_recv, last_t = state["send_count"], state["recv_count"], now
                status = "WARMING" if state["warming"] else "ok"
                print(f"[{now-t0:6.1f}s] send={send_rate:4.1f}Hz recv={recv_rate:4.1f}Hz "
                      f"ema_hz={state['hz']:4.2f}  gpu={state['gpu_ms']:6.1f}ms "
                      f"rtt={state['total_ms']:6.1f}ms  n_sent={state['send_count']} "
                      f"n_recv={state['recv_count']}  [{status}]", flush=True)

        bev_state = {
            "bgr": None,           # last rendered BEV panel
            "rendering": False,    # in-flight render?
            "last_pred_id": -1,    # recv_count of the pred used
        }
        label = "4cam_4fr_10diff"

        async def bev_renderer():
            """Re-render the BEV in a worker thread only when a new prediction
            arrives. Caches the rendered BGR; display loop blits it at 60 fps."""
            loop = asyncio.get_event_loop()
            while not state["stop"]:
                await asyncio.sleep(0.01)
                rid = state["recv_count"]
                if rid == bev_state["last_pred_id"]:
                    continue
                if bev_state["rendering"]:
                    continue
                bev_state["rendering"] = True
                pred = state["latest_pred"]
                bev = await loop.run_in_executor(None, render_bev_panel, pred)
                bev_state["bgr"] = bev
                bev_state["last_pred_id"] = rid
                bev_state["rendering"] = False

        async def display():
            window = "Alpamayo live demo  (ESC to quit)"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1540, 720)
            placeholder = np.full((660, 770, 3), 240, dtype=np.uint8)
            cv2.putText(placeholder, "awaiting first prediction ...",
                        (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            target_interval = 1.0 / 60
            while not state["stop"]:
                t_iter = time.perf_counter()
                frame_bgr = state.get("latest_raw")
                if frame_bgr is not None:
                    sim_t = t_iter - state["start_t"]
                    cam_panel = render_cam_panel(frame_bgr, sim_t, panel_h=660)
                    bev = bev_state["bgr"]
                    if bev is None:
                        bev = placeholder.copy()
                    else:
                        bev = bev.copy()  # don't mutate the cache
                    # Show the 10s average stale (same window as Hz) so the
                    # number is readable instead of flashing 0->300 every frame.
                    stale_s = state["avg_stale_ms_10s"] / 1000.0
                    overlay_title(bev, label, state["hz"], stale_s, state["warming"])
                    combo = compose_panels(cam_panel, bev)
                    cv2.imshow(window, combo)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        state["stop"] = True
                        break
                elapsed = time.perf_counter() - t_iter
                await asyncio.sleep(max(0.001, target_interval - elapsed))

        tasks = [capture(), sender(), display(), bev_renderer(), logger()]
        for i, w in enumerate(ws_list):
            tasks.append(receiver(w, i))
        await asyncio.gather(*tasks)
    finally:
        for w in ws_list:
            try:
                await w.send(msgpack.packb({"bye": True}))
                await w.close()
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws_url", help="wss://...modal.run/ws")
    ap.add_argument("--cam", type=int, default=0, help="cv2 camera index")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="target send rate (with --replicas=2, ~12Hz = 6Hz per replica)")
    ap.add_argument("--replicas", type=int, default=2,
                    help="number of parallel server replicas to distribute frames across")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.ws_url, args.cam, args.fps, args.replicas))
    except KeyboardInterrupt:
        print("\n[client] stopped")


if __name__ == "__main__":
    main()
