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


def render_bev_panel(pred_xy, label, hz_so_far, pred_age_s):
    """Render just the BEV (matplotlib), returns BGR numpy. Expensive (~50-80ms),
    so we cache this and only re-render when a new prediction arrives."""
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
        ax_bev.set_title(
            f"{label}  |  {hz_so_far:.2f} Hz  |  {pred_age_s*1000:.0f}ms stale",
            fontsize=11,
        )
    else:
        ax_bev.scatter([0], [0], c="k", s=60, marker="^")
        ax_bev.set_xlim(-10, 10)
        ax_bev.set_ylim(-5, 15)
        ax_bev.set_title(f"{label}  |  awaiting first prediction ...", fontsize=11)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("y (left+, m)")
    ax_bev.set_ylabel("x (forward+, m)")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3]
    bgr = rgb[..., ::-1].copy()
    plt.close(fig)
    return bgr


async def run(ws_url, cam_index, target_fps):
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

    print(f"[client] connecting WebSocket {ws_url} ...")
    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024,
                                  ping_interval=20, ping_timeout=60,
                                  open_timeout=120) as ws:
        print("[client] connected; waiting for hello ...")

        async def sender():
            next_send = time.perf_counter()
            while not state["stop"]:
                now = time.perf_counter()
                if now < next_send:
                    await asyncio.sleep(next_send - now)
                next_send = now + min_frame_interval
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.01)
                    continue
                # Resize to server-target size if known, else send original.
                if state["server_hw"] is not None:
                    H, W = state["server_hw"]
                    frame_send = cv2.resize(frame, (W, H))
                else:
                    frame_send = frame
                jpeg = encode_jpeg(frame_send, quality=80)
                # Simulate 4-camera car: duplicate webcam into all 4 cam slots.
                try:
                    await ws.send(msgpack.packb({"jpegs": [jpeg, jpeg, jpeg, jpeg]}))
                    state["send_count"] += 1
                except websockets.ConnectionClosed:
                    state["stop"] = True
                    return
                state["latest_raw"] = frame  # for local display

        async def receiver():
            while not state["stop"]:
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    state["stop"] = True
                    return
                msg = msgpack.unpackb(raw, raw=False)
                if msg.get("hello"):
                    state["server_hw"] = (int(msg["H"]), int(msg["W"]))
                    print(f"[client] server target resolution: {msg['W']}x{msg['H']}")
                    continue
                if msg.get("warming"):
                    state["warming"] = True
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

        async def display():
            window = "Alpamayo live demo  (ESC to quit)"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1540, 660)
            label = "4cam_4fr_10diff"
            while not state["stop"]:
                await asyncio.sleep(1.0 / 10)  # matplotlib render ~50-100ms; 10fps plenty
                frame_bgr = state.get("latest_raw")
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Simulate 4-camera car: duplicate the webcam into each slot.
                cam_rgbs = {name: frame_rgb for _, _, name in CAM_LAYOUT}
                now = time.perf_counter()
                sim_t = now - state["start_t"]
                pred_age_s = (now - state["latest_pred_t"]) if state["latest_pred_t"] else 0.0
                loop = asyncio.get_event_loop()
                bgr = await loop.run_in_executor(
                    None, render_composite,
                    cam_rgbs, state["latest_pred"], label, sim_t, pred_age_s, state["hz"],
                )
                cv2.imshow(window, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    state["stop"] = True
                    break

        await asyncio.gather(sender(), receiver(), display(), logger())
        try:
            await ws.send(msgpack.packb({"bye": True}))
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws_url", help="wss://...modal.run/ws")
    ap.add_argument("--cam", type=int, default=0, help="cv2 camera index")
    ap.add_argument("--fps", type=float, default=6.0,
                    help="target send rate (server tops out near ~5Hz on B200)")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.ws_url, args.cam, args.fps))
    except KeyboardInterrupt:
        print("\n[client] stopped")


if __name__ == "__main__":
    main()
