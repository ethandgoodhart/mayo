# Live-camera Alpamayo-R1 demo: server side.
#
# Run:
#   modal deploy scripts/live_demo_server.py     # persistent URL
#   modal serve  scripts/live_demo_server.py     # ephemeral URL, good for dev
#
# The URL printed by modal is the HTTPS base; the WebSocket lives at .../ws.
# Client: scripts/live_demo_client.py <wss://...your-url.../ws>
#
# The model expects 4 cameras x 4 temporal frames per call. We broadcast the
# single webcam into cam-slot 0 (front_wide) and reuse the reference clip's
# other 3 cams (cross_left, cross_right, tele) so the VLM still sees a
# plausible multi-view scene. Ego history is taken from the reference clip
# (stationary fallback), so predicted trajectories are best interpreted as
# "what the model thinks given your current front-view".

import modal

REPO_ROOT = "/root/alpamayo"
CLIP_ID = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
REF_T0_US = 5_100_000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision>=0.23.0")
    .pip_install(
        "transformers==4.57.1",
        "accelerate>=1.12.0",
        "einops>=0.8.1",
        "av>=16.0.1",
        "pillow>=12.0.0",
        "opencv-python>=4.10.0",
        "scipy>=1.14",
        "pandas>=2.3.3",
        "mediapy>=1.2.4",
        "matplotlib>=3.8",
        "hydra-core>=1.3.2",
        "hydra-colorlog>=1.2.0",
        "colorlog>=6.8",
        "rich>=14.3.3",
        "torchmetrics==1.8.2",
        "physical_ai_av>=0.2.0",
        "huggingface_hub[hf_transfer]>=0.26",
        "msgpack>=1.0",
        "fastapi[standard]",
    )
    .pip_install("wheel", "packaging", "ninja")
    .pip_install("flash-attn>=2.8.3", extra_options="--no-build-isolation")
    .pip_install("git+https://github.com/NVlabs/alpamayo1.5.git")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{REPO_ROOT}/src",
    })
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__",
                "alpamayo_videos"],
    )
)

app = modal.App("alpamayo-live-demo", image=image)

hf_cache = modal.Volume.from_name("alpamayo-hf-cache", create_if_missing=True)
pai_cache = modal.Volume.from_name("alpamayo-pai-cache", create_if_missing=True)


@app.cls(
    gpu="B200",
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
    scaledown_window=600,
    min_containers=2,
    max_containers=2,
)
@modal.concurrent(max_inputs=1)  # 1 WS connection per container; forces 2 conns onto 2 boxes
class LiveInference:

    @modal.enter()
    def startup(self):
        import os
        os.environ["HF_HOME"] = "/cache/hf"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
        os.environ["PAI_CACHE"] = "/cache/pai"
        if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

        import time
        import torch
        import physical_ai_av  # noqa: F401 (required to register PAI codecs)

        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo_r1 import helper

        self.helper = helper
        self.torch = torch

        print("[server] loading reference clip for ego-history scaffolding ...")
        self.ref_data = load_physical_aiavdataset(CLIP_ID, t0_us=REF_T0_US)
        img = self.ref_data["image_frames"]
        print(f"[server] ref image_frames: shape={tuple(img.shape)} dtype={img.dtype} "
              f"min={float(img.min()):.3f} max={float(img.max()):.3f}")
        print(f"[server] ref ego_history_xyz: shape={tuple(self.ref_data['ego_history_xyz'].shape)}")

        print("[server] loading Alpamayo-R1-10B ...")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
        ).to("cuda")
        model.eval()
        # Flash-attn fails on the expert's 4D float mask — keep SDPA on expert.
        model.expert.config._attn_implementation = "sdpa"
        for m in model.expert.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation_internal"):
                c._attn_implementation_internal = "sdpa"

        print("[server] torch.compile(expert, reduce-overhead) ...")
        model.expert = torch.compile(
            model.expert, mode="reduce-overhead", dynamic=False,
        )
        self.model = model
        self.processor = helper.get_processor(model.tokenizer)

        print("[server] warmup x3 ...")
        for i in range(3):
            t = time.perf_counter()
            _ = self._infer(self.ref_data["image_frames"])
            print(f"[server]   warmup {i+1}: {(time.perf_counter()-t)*1000:.1f} ms")
        print("[server] ready")

    def _infer(self, image_frames):
        """image_frames: torch.Tensor same shape/dtype as self.ref_data['image_frames'].
        Returns pred_xy np.ndarray of shape [T, 2]."""
        import numpy as np
        torch = self.torch
        messages = self.helper.create_message(image_frames.flatten(0, 1))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": self.ref_data["ego_history_xyz"],
            "ego_history_rot": self.ref_data["ego_history_rot"],
        }
        model_inputs = self.helper.to_device(model_inputs, "cuda")
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot, _extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=1,
                return_extra=True,
                diffusion_kwargs={"inference_step": 3},
            )
        pred_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        return pred_xy.astype(np.float32)

    @modal.asgi_app()
    def web(self):
        import asyncio
        import io
        import time
        import numpy as np
        import torch
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from PIL import Image as PILImage
        import msgpack

        api = FastAPI()

        ref = self.ref_data["image_frames"]
        ref_shape = tuple(ref.shape)
        # Expect [1, n_cams*n_frames, 3, H, W] or [1, n_cams, n_frames, 3, H, W].
        # Collapse to [1, n_cams, n_frames, 3, H, W] for slot-wise editing.
        if ref.dim() == 5:
            total = ref.shape[1]
            # Assume n_cams=4, n_frames=4 layout (cam-major).
            ref_chw = ref.view(1, 4, 4, *ref.shape[-3:])
        elif ref.dim() == 6:
            ref_chw = ref
        else:
            raise RuntimeError(f"unexpected ref shape {ref_shape}")
        H, W = ref_chw.shape[-2], ref_chw.shape[-1]
        ref_dtype = ref.dtype
        print(f"[server] serving: per-call tensor shape {tuple(ref.shape)}, H={H} W={W} dtype={ref_dtype}")

        @api.get("/")
        async def root():
            return {"status": "ok", "model": "Alpamayo-R1-10B", "ws": "/ws",
                    "ref_shape": list(ref_shape), "H": H, "W": W}

        @api.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            print("[ws] client connected")
            # 4 temporal buffers, one per cam slot. Each holds up to 4 frames.
            cam_bufs = [[] for _ in range(4)]
            try:
                # Send the target image size + cam layout so the client can pre-resize.
                await websocket.send_bytes(msgpack.packb({
                    "hello": True, "H": H, "W": W, "horizon_s": 6.4,
                    "n_cams": 4,
                    "cam_names": ["front_wide", "front_tele", "cross_left", "cross_right"],
                }))

                def _decode(jpeg_bytes):
                    img = PILImage.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    if img.size != (W, H):
                        img = img.resize((W, H))
                    arr = np.array(img)
                    return torch.from_numpy(arr).permute(2, 0, 1).to(ref_dtype)

                while True:
                    t_recv_start = time.perf_counter()
                    raw = await websocket.receive_bytes()
                    t_recv = time.perf_counter()
                    msg = msgpack.unpackb(raw, raw=False)
                    if msg.get("bye"):
                        break
                    if "jpegs" in msg:
                        jpegs = msg["jpegs"]
                        if len(jpegs) != 4:
                            raise RuntimeError(f"expected 4 jpegs, got {len(jpegs)}")
                    elif "jpeg" in msg:
                        # Back-compat single-cam: duplicate across all 4 slots.
                        jpegs = [msg["jpeg"]] * 4
                    else:
                        raise RuntimeError("msg missing 'jpegs' and 'jpeg'")

                    for ci, jb in enumerate(jpegs):
                        cam_bufs[ci].append(_decode(jb))
                        if len(cam_bufs[ci]) > 4:
                            cam_bufs[ci] = cam_bufs[ci][-4:]

                    min_len = min(len(b) for b in cam_bufs)
                    if min_len < 4:
                        await websocket.send_bytes(msgpack.packb({
                            "warming": True, "buffer": min_len,
                        }))
                        continue

                    # Compose: 4 cams x 4 frames x 3 x H x W, each cam from its own buffer.
                    per_cam = torch.stack([torch.stack(b, dim=0) for b in cam_bufs], dim=0)  # [4, 4, 3, H, W]
                    full = per_cam.unsqueeze(0)  # [1, 4, 4, 3, H, W]
                    img_in = full.view(*ref.shape)

                    t_inf0 = time.perf_counter()
                    pred_xy = await asyncio.get_event_loop().run_in_executor(
                        None, self._infer, img_in
                    )
                    t_inf1 = time.perf_counter()
                    total_ms = (t_inf1 - t_recv_start) * 1000.0
                    gpu_ms = (t_inf1 - t_inf0) * 1000.0
                    recv_ms = (t_recv - t_recv_start) * 1000.0

                    payload = msgpack.packb({
                        "pred_shape": list(pred_xy.shape),
                        "pred_xy": pred_xy.tobytes(),
                        "gpu_ms": gpu_ms,
                        "recv_ms": recv_ms,
                        "total_ms": total_ms,
                    })
                    await websocket.send_bytes(payload)
            except WebSocketDisconnect:
                print("[ws] client disconnected")
            except Exception as e:
                print(f"[ws] error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        return api
