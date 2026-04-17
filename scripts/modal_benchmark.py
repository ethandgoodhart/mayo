"""Modal benchmark for Alpamayo-R1-10B throughput.

Usage (from repo root):
    modal run scripts/modal_benchmark.py::benchmark
    modal run scripts/modal_benchmark.py::benchmark --skip-cot --cot-max-tokens 0
    modal run scripts/modal_benchmark.py::benchmark --cot-max-tokens 48
"""

import modal

REPO_ROOT = "/repo"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    # Install torch from default PyPI (bundles CUDA libs); cu124 channel lacks 2.8.0.
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
    )
    .pip_install(
        "flash-attn>=2.8.3",
        extra_options="--no-build-isolation",
    )
    # Alpamayo-1.5 ships as a git-only package (uv_build backend, no PyPI wheel).
    # Installed after flash-attn so its own flash-attn requirement is satisfied.
    .pip_install("git+https://github.com/NVlabs/alpamayo1.5.git")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        # Import alpamayo_r1 and rl directly from the mounted repo.
        "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/finetune/rl",
    })
    # add_local_dir must be the LAST image step (no run_commands after).
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
    )
)

app = modal.App("alpamayo-r1-bench", image=image)

hf_cache = modal.Volume.from_name("alpamayo-hf-cache", create_if_missing=True)
pai_cache = modal.Volume.from_name("alpamayo-pai-cache", create_if_missing=True)
videos_vol = modal.Volume.from_name("alpamayo-videos", create_if_missing=True)

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"

# (num_cameras, num_frames, diffusion_steps, label)
VIDEO_CONFIGS = [
    (4, 4, 10, "4cam_4fr_10diff"),
    (4, 4,  4, "4cam_4fr_4diff"),
    (4, 1,  4, "4cam_1fr_4diff"),
    (2, 1,  2, "2cam_1fr_2diff"),
    (1, 1,  1, "1cam_1fr_1diff"),
    (1, 4,  4, "1cam_4fr_4diff"),
    (1, 4, 10, "1cam_4fr_10diff"),
]


@app.function(
    gpu="A100-40GB",
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def benchmark(
    cot_max_tokens: int = 256,
    skip_cot: bool = False,
    num_traj_samples: int = 1,
    num_timestamps: int = 15,
    attn_impl: str = "flash_attention_2",
    duration_s: float = 30.0,
    warmup_iters: int = 3,
    dtype: str = "bfloat16",
    compile_vlm: bool = False,
    diffusion_steps: int = 0,
    num_frames: int = 4,
    num_cameras: int = 4,
):
    """Run throughput benchmark for Alpamayo-R1-10B.

    Preloads `num_timestamps` samples across a 30s window of one PAI clip,
    runs a warmup, then loops inference for `duration_s` wall-clock seconds,
    cycling through the preloaded samples. Reports Hz and per-stage timings.
    """
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    # Propagate secret token under both conventional names
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HUGGING_FACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import time
    import torch
    import numpy as np

    # AlpamayoR1's action expert uses a custom 4D float attention mask (to skip
    # padding between the VLM's valid_token_pos_id and the diffusion tokens) —
    # FA2 can't consume that mask directly, so the expert forward crashes with
    # "cu_seqlens_q must have shape (batch_size + 1)". We keep FA2 for the VLM
    # (vision + text decoder, which is where all the time is spent) and force
    # SDPA for just the expert. The switch is done after load, via the
    # _attn_implementation setter which propagates through sub-configs.

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper
    import physical_ai_av

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    print(f"[bench] config: cot_max_tokens={cot_max_tokens} skip_cot={skip_cot} "
          f"num_traj_samples={num_traj_samples} attn_impl={attn_impl} dtype={dtype} "
          f"compile_vlm={compile_vlm} diffusion_steps={diffusion_steps} "
          f"num_frames={num_frames} num_cameras={num_cameras}")
    print(f"[bench] loading model nvidia/Alpamayo-R1-10B ...")
    t_load = time.perf_counter()
    # attn_implementation is propagated via the nested config (not a top-level kwarg)
    # because HF validates top-level attn_impl against AlpamayoR1's _supports_flash_attn_2.
    from alpamayo_r1.config import AlpamayoR1Config
    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = attn_impl
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        config=cfg,
        dtype=torch_dtype,
    ).to("cuda")
    model.eval()
    # Force SDPA for the expert only (see comment above the monkey-patch section).
    model.expert.config._attn_implementation = "sdpa"
    for _m in model.expert.modules():
        _cfg = getattr(_m, "config", None)
        if _cfg is not None and hasattr(_cfg, "_attn_implementation_internal"):
            _cfg._attn_implementation_internal = "sdpa"
    print(f"[bench] model loaded in {time.perf_counter()-t_load:.1f}s")

    if compile_vlm:
        print("[bench] compiling vlm forward ...")
        model.vlm.model = torch.compile(model.vlm.model, mode="reduce-overhead", dynamic=True)

    processor = helper.get_processor(model.tokenizer)

    def build_messages(frames):
        msgs = helper.create_message(frames)
        if skip_cot:
            # End the assistant prefix right at <|traj_future_start|> so the VLM
            # doesn't have to decode a CoC trace. We still generate 1 token so
            # vlm.generate returns a valid GenerateOutput with past_key_values.
            msgs[-1]["content"][0]["text"] = (
                "<|cot_start|><|cot_end|><|traj_future_start|>"
            )
        return msgs

    print(f"[bench] prefetching {num_timestamps} samples from clip {CLIP_ID} ...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    # Spread t0 across ~15s window starting at 5.1s (clip only has ~15s usable)
    t0_start_us = 5_100_000
    window_s = 15.0
    step_us = int(window_s * 1_000_000 / num_timestamps)
    # Camera subset: default 4 cams, user can request fewer (front-wide always first).
    cam_features = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ][:num_cameras]
    preloaded = []
    for i in range(num_timestamps):
        t0 = t0_start_us + i * step_us
        try:
            data = load_physical_aiavdataset(
                CLIP_ID, t0_us=t0, avdi=avdi,
                num_frames=num_frames, camera_features=cam_features,
            )
        except Exception as e:
            print(f"[bench] skip t0={t0} ({type(e).__name__}: {e})")
            continue
        messages = build_messages(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        preloaded.append(model_inputs)

    if not preloaded:
        raise RuntimeError("no samples preloaded")
    print(f"[bench] preloaded {len(preloaded)} samples")

    # Per-stage timing via monkey-patch. `stage_times` accumulates ms for the
    # last run_one call; we reset it each iteration.
    stage_times = {"vlm_ms": 0.0, "vision_ms": 0.0, "diff_ms": 0.0}
    _orig_vlm_generate = model.vlm.generate
    _orig_diff_sample = model.diffusion.sample
    # Vision encoder is model.vlm.visual (Qwen3-VL visual tower).
    _vision = getattr(model.vlm, "visual", None)
    _orig_vision_forward = _vision.forward if _vision is not None else None

    def _timed_vlm_generate(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig_vlm_generate(*a, **kw)
        torch.cuda.synchronize()
        stage_times["vlm_ms"] = (time.perf_counter() - t) * 1000
        return out

    def _timed_diff_sample(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig_diff_sample(*a, **kw)
        torch.cuda.synchronize()
        stage_times["diff_ms"] = (time.perf_counter() - t) * 1000
        return out

    def _timed_vision_forward(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig_vision_forward(*a, **kw)
        torch.cuda.synchronize()
        stage_times["vision_ms"] = (time.perf_counter() - t) * 1000
        return out

    model.vlm.generate = _timed_vlm_generate
    model.diffusion.sample = _timed_diff_sample
    if _vision is not None:
        _vision.forward = _timed_vision_forward

    diffusion_kwargs = {"inference_step": diffusion_steps} if diffusion_steps > 0 else None

    def run_one(sample):
        # sample's tokenized_data gets mutated by sample_traj (input_ids popped),
        # so we pass a shallow copy to keep the preloaded dict reusable.
        td = sample["tokenized_data"]
        td_copy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dict(td).items()}
        s = {
            "tokenized_data": td_copy,
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
        }
        # When skip_cot, prompt already ends at <|traj_future_start|>; we still
        # generate 1 token so vlm.generate returns past_key_values.
        max_gen = 1 if skip_cot else cot_max_tokens
        with torch.autocast("cuda", dtype=torch_dtype):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=s,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_gen,
                return_extra=True,
                diffusion_kwargs=diffusion_kwargs,
            )
        return pred_xyz

    # Warmup
    print(f"[bench] warmup x{warmup_iters} ...")
    torch.cuda.manual_seed_all(42)
    for i in range(warmup_iters):
        _ = run_one(preloaded[i % len(preloaded)])
    torch.cuda.synchronize()
    print(f"[bench] warmup peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Main loop
    print(f"[bench] running for {duration_s}s ...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    latencies = []
    vlm_mss = []
    diff_mss = []
    vision_mss = []
    while True:
        t_iter = time.perf_counter()
        _ = run_one(preloaded[count % len(preloaded)])
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t_iter)
        vlm_mss.append(stage_times["vlm_ms"])
        diff_mss.append(stage_times["diff_ms"])
        vision_mss.append(stage_times["vision_ms"])
        count += 1
        if time.perf_counter() - t0 >= duration_s:
            break
    elapsed = time.perf_counter() - t0

    hz = count / elapsed
    lat_ms = np.array(latencies) * 1000
    print("\n" + "=" * 60)
    print(f"[bench] predictions: {count} in {elapsed:.2f}s  →  {hz:.2f} Hz")
    print(f"[bench] latency ms: mean={lat_ms.mean():.1f} p50={np.percentile(lat_ms,50):.1f} "
          f"p95={np.percentile(lat_ms,95):.1f} max={lat_ms.max():.1f}")
    vlm_arr = np.array(vlm_mss)
    diff_arr = np.array(diff_mss)
    vis_arr = np.array(vision_mss)
    print(f"[bench] stage ms: vlm={vlm_arr.mean():.1f} (vision={vis_arr.mean():.1f} "
          f"text_prefill={vlm_arr.mean()-vis_arr.mean():.1f}) "
          f"diff={diff_arr.mean():.1f} "
          f"other={lat_ms.mean()-vlm_arr.mean()-diff_arr.mean():.1f}")
    print(f"[bench] peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=" * 60)
    return {
        "hz": hz,
        "count": count,
        "elapsed_s": elapsed,
        "latency_ms_mean": float(lat_ms.mean()),
        "latency_ms_p50": float(np.percentile(lat_ms, 50)),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)),
        "peak_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        "config": {
            "cot_max_tokens": cot_max_tokens,
            "skip_cot": skip_cot,
            "num_traj_samples": num_traj_samples,
            "attn_impl": attn_impl,
            "dtype": dtype,
            "compile_vlm": compile_vlm,
        },
    }


@app.local_entrypoint()
def main(
    cot_max_tokens: int = 256,
    skip_cot: bool = False,
    num_traj_samples: int = 1,
    num_timestamps: int = 15,
    attn_impl: str = "flash_attention_2",
    duration_s: float = 30.0,
    dtype: str = "bfloat16",
    compile_vlm: bool = False,
    diffusion_steps: int = 0,
    num_frames: int = 4,
    num_cameras: int = 4,
):
    result = benchmark.remote(
        cot_max_tokens=cot_max_tokens,
        skip_cot=skip_cot,
        num_traj_samples=num_traj_samples,
        num_timestamps=num_timestamps,
        attn_impl=attn_impl,
        duration_s=duration_s,
        dtype=dtype,
        compile_vlm=compile_vlm,
        diffusion_steps=diffusion_steps,
        num_frames=num_frames,
        num_cameras=num_cameras,
    )
    print("[local] result:", result)


@app.function(
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 15,
)
def probe_clip_length(clip_id: str = CLIP_ID) -> dict:
    """CPU-only probe: reports egomotion AND front-wide camera time bounds.
    Camera coverage is usually narrower than egomotion and is the real
    constraint on how long we can run a sim without running out of frames."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import numpy as np
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    info = {"clip_id": clip_id}

    def bounds(feat, name):
        obj = avdi.get_clip_feature(clip_id, feat, maybe_stream=True)
        for attr in ("timestamps", "times", "t_us", "t"):
            arr = getattr(obj, attr, None)
            if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                a = np.asarray(arr)
                info[f"{name}_t_min_us"] = int(a.min())
                info[f"{name}_t_max_us"] = int(a.max())
                info[f"{name}_duration_s"] = float((a.max() - a.min()) / 1e6)
                info[f"{name}_n"] = int(len(a))
                return
        info[f"{name}_t_max_us"] = None

    bounds(avdi.features.LABELS.EGOMOTION, "ego")
    bounds(avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, "camfw")
    print(f"[probe] clip {clip_id}:")
    print(f"  ego   [{info.get('ego_t_min_us',0)/1e6:.2f}s, "
          f"{info.get('ego_t_max_us',0)/1e6:.2f}s]  dur={info.get('ego_duration_s',0):.2f}s")
    print(f"  camfw [{info.get('camfw_t_min_us',0)/1e6:.2f}s, "
          f"{info.get('camfw_t_max_us',0)/1e6:.2f}s]  dur={info.get('camfw_duration_s',0):.2f}s")
    return info


@app.function(
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def find_long_clip(min_camera_s: float = 50.0, max_check: int = 200) -> dict:
    """Scan clips in the PAI index, returning the first one whose front-wide
    camera covers at least `min_camera_s` seconds. This is purely metadata
    access (no frame decode), so it's fast — ~ms per clip."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import numpy as np
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    # Different PAI interfaces expose clip listing differently; try several.
    clip_ids = None
    for attr in ("get_all_clip_ids", "clip_ids", "list_clips", "clips"):
        v = getattr(avdi, attr, None)
        if callable(v):
            try:
                clip_ids = list(v())
                break
            except Exception:
                pass
        elif v is not None:
            try:
                clip_ids = list(v)
                break
            except Exception:
                pass
    if clip_ids is None:
        # Fall back: walk the pandas clip_index if present
        ci = getattr(avdi, "clip_index", None)
        if ci is not None:
            clip_ids = list(ci.index.tolist()) if hasattr(ci, "index") else None
    if clip_ids is None:
        # Last resort: introspect public attrs
        print("[scan] couldn't find clip list; avdi public attrs:",
              [x for x in dir(avdi) if not x.startswith("_")][:40])
        return {"error": "no clip list"}
    print(f"[scan] {len(clip_ids)} clips in index; scanning up to {max_check}")
    best = {"clip_id": None, "camfw_duration_s": 0.0}
    for i, cid in enumerate(clip_ids[:max_check]):
        try:
            camfw = avdi.get_clip_feature(
                cid, avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, maybe_stream=True
            )
            arr = None
            for attr in ("timestamps", "times", "t_us", "t"):
                v = getattr(camfw, attr, None)
                if v is not None and hasattr(v, "__len__") and len(v) > 0:
                    arr = np.asarray(v)
                    break
            if arr is None:
                continue
            dur = float((arr.max() - arr.min()) / 1e6)
            if dur > best["camfw_duration_s"]:
                best = {
                    "clip_id": cid,
                    "camfw_duration_s": dur,
                    "camfw_t_min_us": int(arr.min()),
                    "camfw_t_max_us": int(arr.max()),
                    "index": i,
                }
                print(f"[scan] new best at idx {i}: {cid}  camfw_dur={dur:.1f}s")
            if dur >= min_camera_s:
                print(f"[scan] found clip with >= {min_camera_s:.0f}s at idx {i}: {cid}")
                return best
        except Exception as e:
            continue
    print(f"[scan] no clip >= {min_camera_s:.0f}s in first {max_check}; best = {best}")
    return best


@app.local_entrypoint()
def probe(clip_id: str = CLIP_ID):
    info = probe_clip_length.remote(clip_id=clip_id)
    print("[local] probe:", info)


@app.local_entrypoint()
def scan(min_camera_s: float = 50.0, max_check: int = 200):
    info = find_long_clip.remote(min_camera_s=min_camera_s, max_check=max_check)
    print("[local] best:", info)


@app.function(
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 15,
)
def inspect_clip_index() -> dict:
    """Dump columns + a row of avdi.clip_index so we can see what metadata
    is available (e.g. end_timestamp) without scanning each clip's camera."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    ci = getattr(avdi, "clip_index", None)
    if ci is None:
        print("[inspect] no clip_index attr; public attrs:",
              [x for x in dir(avdi) if not x.startswith("_")][:40])
        return {"error": "no clip_index"}
    print(f"[inspect] clip_index: {len(ci)} rows, columns: {list(ci.columns)}")
    print(f"[inspect] dtypes:\n{ci.dtypes}")
    print(f"[inspect] first row:\n{ci.iloc[0].to_dict()}")
    # If there's duration-like column, show its distribution
    for col in ci.columns:
        if any(k in col.lower() for k in ("duration", "end", "length", "time")):
            try:
                print(f"[inspect] col {col}: "
                      f"min={ci[col].min()} max={ci[col].max()} mean={ci[col].mean()}")
            except Exception:
                pass
    return {"n_clips": int(len(ci)), "columns": list(ci.columns)}


@app.local_entrypoint()
def inspect():
    inspect_clip_index.remote()


PRECACHE_GRID_HZ = 10  # pre-decode frames at 10 Hz (100 ms step)


def _precache_dir(clip_id: str) -> str:
    return f"/cache/pai/precache/{clip_id}"


@app.function(
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def precache_clip(
    clip_id: str = CLIP_ID,
    t0_start_s: float = 1.7,
    duration_s: float = 18.0,
) -> dict:
    """Pre-decode (4cam × 4fr) + ego trajectories at every 100ms step across
    the window and write pickles to the pai_cache volume. Re-runs can skip
    this step entirely."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import pickle
    import time
    import physical_ai_av

    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    all_cams = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]

    cache_dir = _precache_dir(clip_id)
    os.makedirs(cache_dir, exist_ok=True)

    step_us = int(1_000_000 / PRECACHE_GRID_HZ)
    t0_start_us = int(t0_start_s * 1_000_000)
    n = int(duration_s * PRECACHE_GRID_HZ) + 1

    print(f"[precache] clip={clip_id}  range=[{t0_start_s:.2f}s, "
          f"{t0_start_s + duration_s:.2f}s]  n={n} @ {PRECACHE_GRID_HZ}Hz")
    t_start = time.perf_counter()
    ok_count = 0
    skip_count = 0
    fail_count = 0
    for i in range(n):
        t_us = t0_start_us + i * step_us
        out_path = f"{cache_dir}/{t_us}.pkl"
        if os.path.exists(out_path):
            skip_count += 1
            continue
        try:
            data = load_physical_aiavdataset(
                clip_id, t0_us=t_us, avdi=avdi,
                num_frames=4, camera_features=all_cams,
            )
        except Exception as e:
            fail_count += 1
            print(f"[precache] skip t={t_us/1e6:.2f}s ({type(e).__name__})")
            continue
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        ok_count += 1
        if ok_count % 20 == 0:
            el = time.perf_counter() - t_start
            print(f"[precache] {ok_count}/{n}  elapsed={el:.1f}s  "
                  f"rate={ok_count/el:.2f}/s")

    manifest = {
        "clip_id": clip_id, "t0_start_us": t0_start_us,
        "duration_s": duration_s, "grid_hz": PRECACHE_GRID_HZ, "n": n,
        "ok": ok_count, "skip": skip_count, "fail": fail_count,
    }
    with open(f"{cache_dir}/manifest.pkl", "wb") as f:
        pickle.dump(manifest, f)
    pai_cache.commit()
    total = time.perf_counter() - t_start
    print(f"[precache] done: ok={ok_count} skip={skip_count} fail={fail_count} "
          f"in {total:.1f}s")
    return manifest


@app.local_entrypoint()
def precache(
    clip_id: str = CLIP_ID,
    t0_start_s: float = 1.7,
    duration_s: float = 18.0,
):
    info = precache_clip.remote(
        clip_id=clip_id, t0_start_s=t0_start_s, duration_s=duration_s
    )
    print("[local] precache:", info)


@app.function(
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 10,
)
def resolve_clip_id(prefix: str) -> str:
    """Given a clip-id prefix (e.g. '25cd4769'), return the full UUID. Raises if
    no match or ambiguous match."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    ci = getattr(avdi, "clip_index", None)
    if ci is None:
        raise RuntimeError("avdi has no clip_index")
    all_ids = ci.index.tolist() if hasattr(ci, "index") else list(ci)
    matches = [c for c in all_ids if str(c).startswith(prefix)]
    if not matches:
        raise RuntimeError(f"no clip matches prefix {prefix!r}")
    if len(matches) > 1:
        raise RuntimeError(f"prefix {prefix!r} matches {len(matches)} clips: {matches[:5]}")
    print(f"[resolve] {prefix!r} -> {matches[0]}")
    return matches[0]


@app.function(
    gpu="H100",
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache, "/outputs": videos_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def render_videos(
    t0_start_s: float = 5.1,
    duration_s: float = 8.0,
    display_fps: int = 10,
    skip_cot: bool = True,
    config_indices: str = "",
    clip_id: str = CLIP_ID,
    model_variant: str = "r1",
):
    """Render one real-time BEV MP4 per (num_cams, num_frames, diff_steps) config.

    Real-time simulation: per config, we run a sim-clock starting at t=0 that
    advances by *measured GPU inference latency* after each prediction. Each
    prediction becomes "available" for display starting at the sim-time it
    finished computing. We stop when sim-clock reaches `duration_s` (simulates
    a car running for that many real seconds). The video is played back at
    1× speed at `display_fps` — so a slow model (0.6 Hz) will visibly hold a
    stale trajectory for seconds while a fast model (7 Hz) updates fluidly.
    """
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HUGGING_FACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import time
    import pickle
    import numpy as np
    import torch

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mediapy

    if model_variant == "1.5":
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5 as ModelCls
        from alpamayo1_5.config import Alpamayo1_5Config as ModelConfig
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo1_5 import helper
        model_hf_id = "nvidia/Alpamayo-1.5-10B"
        needs_cam_indices_in_msg = True
    else:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1 as ModelCls
        from alpamayo_r1.config import AlpamayoR1Config as ModelConfig
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo_r1 import helper
        model_hf_id = "nvidia/Alpamayo-R1-10B"
        needs_cam_indices_in_msg = False
    import physical_ai_av

    # Load precache manifest if present.
    cache_dir = _precache_dir(clip_id)
    manifest_path = f"{cache_dir}/manifest.pkl"
    precache_hits = 0
    precache_misses = 0
    if os.path.exists(manifest_path):
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        print(f"[render] found precache: {manifest}")
    else:
        print(f"[render] no precache at {cache_dir} — will stream each call")
    grid_step_us = int(1_000_000 / PRECACHE_GRID_HZ)

    cfg = ModelConfig.from_pretrained(model_hf_id)
    cfg.attn_implementation = "flash_attention_2"
    print(f"[render] loading model {model_hf_id} (variant={model_variant}) ...")
    t_load = time.perf_counter()
    model = ModelCls.from_pretrained(
        model_hf_id, config=cfg, dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    # Expert uses a custom 4D float mask; FA2 can't consume it — fall back to SDPA there.
    model.expert.config._attn_implementation = "sdpa"
    for _m in model.expert.modules():
        _c = getattr(_m, "config", None)
        if _c is not None and hasattr(_c, "_attn_implementation_internal"):
            _c._attn_implementation_internal = "sdpa"
    print(f"[render] model loaded in {time.perf_counter()-t_load:.1f}s")

    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    t0_start_us = int(t0_start_s * 1_000_000)
    n_display = int(duration_s * display_fps)
    display_step_us = int(1_000_000 / display_fps)
    # Display timestamps (sim_t grid): one camera frame per display tick.
    display_t_us = [t0_start_us + i * display_step_us for i in range(n_display)]

    all_cams = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]
    # camera_indices the live call would produce for each n_cams (after its internal
    # sort): n_cams=1 -> [front_wide=1]; n_cams=2 -> [front_wide=1, front_tele=6];
    # n_cams=4 -> [cross_left=0, front_wide=1, cross_right=2, front_tele=6].
    live_cam_indices = {
        1: [1],
        2: [1, 6],
        3: [1, 2, 6],
        4: [0, 1, 2, 6],
    }

    def build_messages(frames, camera_indices=None, num_frames_per_camera=4):
        # Pass num_frames_per_camera so Alpamayo-1.5's prompt annotations
        # ("frame 0", "frame 1", ...) match the actual frame count. Default
        # 4 is correct only for 4-frame configs; with 1 frame per cam it
        # would falsely tell the model to expect 4 frames.
        if needs_cam_indices_in_msg:
            msgs = helper.create_message(
                frames,
                camera_indices=camera_indices,
                num_frames_per_camera=num_frames_per_camera,
            )
        else:
            msgs = helper.create_message(frames)
        if skip_cot:
            msgs[-1]["content"][0]["text"] = (
                "<|cot_start|><|cot_end|><|traj_future_start|>"
            )
        return msgs

    def load_data(t_us, n_cams, n_frames):
        """Return a `load_physical_aiavdataset`-shaped dict. Uses precache if
        available (snap to 100ms grid, then subset to requested cam/frame
        counts). Falls back to live streaming if the pickle is missing."""
        nonlocal precache_hits, precache_misses
        t_snap = (t_us // grid_step_us) * grid_step_us
        pkl = f"{cache_dir}/{t_snap}.pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            precache_hits += 1
            # precache holds 4cam × 4fr; subset to requested cams by matching the
            # camera_indices that the live call would have produced, so downstream
            # inference sees the same cam set.
            data = dict(data)
            want = live_cam_indices.get(n_cams, list(range(n_cams)))
            full_idx = data["camera_indices"].tolist()
            row_sel = [full_idx.index(ci) for ci in want if ci in full_idx]
            sel = torch.tensor(row_sel, dtype=torch.long)
            data["image_frames"] = data["image_frames"][sel][:, -n_frames:]
            data["camera_indices"] = data["camera_indices"][sel]
            if "relative_timestamps" in data:
                data["relative_timestamps"] = data["relative_timestamps"][sel][:, -n_frames:]
            if "absolute_timestamps" in data:
                data["absolute_timestamps"] = data["absolute_timestamps"][sel][:, -n_frames:]
            return data
        precache_misses += 1
        return load_physical_aiavdataset(
            clip_id, t0_us=t_us, avdi=avdi,
            num_frames=n_frames, camera_features=all_cams[:n_cams],
        )

    def run_inference(t_us, n_cams, n_frames, diff_steps):
        """Load data at absolute timestamp t_us, run inference, return
        (pred_xy, gt_xy, fw_frame, {data_load_ms, preprocess_ms, gpu_infer_ms}).
        data_load_ms = CPU dataset/pickle load. preprocess_ms = tokenize +
        apply_chat_template + host->device. gpu_infer_ms = sync-bracketed model
        forward (the only timer the car's steady-state Hz depends on)."""
        t0 = time.perf_counter()
        data = load_data(t_us, n_cams, n_frames)
        fw_idx_list = (data["camera_indices"] == 1).nonzero(as_tuple=True)[0]
        fw_idx = int(fw_idx_list[0]) if fw_idx_list.numel() > 0 else 0
        fw_frame = data["image_frames"][fw_idx, -1].permute(1, 2, 0).cpu().numpy()
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        data_load_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        messages = build_messages(
            data["image_frames"].flatten(0, 1),
            camera_indices=data.get("camera_indices"),
            num_frames_per_camera=n_frames,
        )
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        max_gen = 1 if skip_cot else 256
        diffusion_kwargs = {"inference_step": diff_steps}

        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98, temperature=0.6,
                num_traj_samples=1,
                max_generation_length=max_gen,
                return_extra=True,
                diffusion_kwargs=diffusion_kwargs,
            )
        torch.cuda.synchronize()
        gpu_infer_ms = (time.perf_counter() - t_infer) * 1000.0

        pred_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        return pred_xy, gt_xy, fw_frame, {
            "data_load_ms": data_load_ms,
            "preprocess_ms": preprocess_ms,
            "gpu_infer_ms": gpu_infer_ms,
        }

    def render_composite(fw_frame, pred_xy, gt_xy, label, ade, sim_t, pred_age_s, hz_so_far):
        T = min(pred_xy.shape[0], gt_xy.shape[0]) if pred_xy is not None else 0
        fig = plt.figure(figsize=(14, 6), dpi=110)
        ax_img = fig.add_subplot(1, 2, 1)
        ax_bev = fig.add_subplot(1, 2, 2)
        ax_img.imshow(fw_frame)
        ax_img.axis("off")
        ax_img.set_title(f"front-wide  sim_t={sim_t:.2f}s", fontsize=11)
        if pred_xy is not None:
            ax_bev.plot(-gt_xy[:T, 1], gt_xy[:T, 0], "r-", lw=2, label="GT")
            ax_bev.scatter(-gt_xy[T-1, 1], gt_xy[T-1, 0], c="r", s=30, marker="o")
            ax_bev.plot(-pred_xy[:T, 1], pred_xy[:T, 0], "b-", lw=2, label="Pred")
            ax_bev.scatter(-pred_xy[T-1, 1], pred_xy[T-1, 0], c="b", s=30, marker="o")
            ax_bev.scatter([0], [0], c="k", s=60, marker="^", label="ego")
            all_x = np.concatenate([gt_xy[:T, 0], pred_xy[:T, 0], [0]])
            all_y = np.concatenate([-gt_xy[:T, 1], -pred_xy[:T, 1], [0]])
            pad = 2.0
            span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 10.0)
            cx, cy = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2
            ax_bev.set_xlim(cy - span/2 - pad, cy + span/2 + pad)
            ax_bev.set_ylim(cx - span/2 - pad, cx + span/2 + pad)
            ax_bev.legend(loc="lower right", fontsize=9)
            ax_bev.set_title(
                f"{label}  |  minADE={ade:.2f}m  |  {hz_so_far:.2f} Hz  |  {pred_age_s*1000:.0f}ms stale",
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
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    # 1) Preload display camera frames at display_fps (shared across all configs).
    # Uses load_data so it hits the precache when available.
    print(f"[render] preloading {n_display} display frames @ {display_fps}fps ...")
    display_cache = {}  # idx -> front_wide_rgb uint8
    for i, t_us in enumerate(display_t_us):
        try:
            data = load_data(t_us, n_cams=1, n_frames=1)
        except Exception as e:
            print(f"[render]   skip display t={i*display_step_us/1e6:.2f}s "
                  f"({type(e).__name__}: {e})")
            continue
        display_cache[i] = data["image_frames"][0, -1].permute(1, 2, 0).cpu().numpy()
    if not display_cache:
        raise RuntimeError("no display frames preloaded")
    last_idx = max(display_cache.keys())
    # Fill any gaps by reusing the prior successful frame.
    for i in range(n_display):
        if i not in display_cache:
            j = max((k for k in display_cache if k < i), default=last_idx)
            display_cache[i] = display_cache[j]
    print(f"[render] preloaded {n_display} display frames")

    os.makedirs("/outputs", exist_ok=True)
    results = []

    # Config subset filter.
    if config_indices.strip():
        keep_idx = {int(s) for s in config_indices.split(",") if s.strip()}
        configs_to_run = [
            (i, c) for i, c in enumerate(VIDEO_CONFIGS) if i in keep_idx
        ]
        print(f"[render] running subset: {[c[-1] for _, c in configs_to_run]}")
    else:
        configs_to_run = list(enumerate(VIDEO_CONFIGS))

    # 2) For each config: real-time sim (sim_t advances by GPU latency), then
    # render video at display_fps, replaying 1× speed.
    for run_i, (cfg_idx, (n_cams, n_frames, diff_steps, label)) in enumerate(configs_to_run):
        print(f"\n[render] ({run_i+1}/{len(configs_to_run)}) config={label}")

        # Warmup one prediction so first measurement isn't distorted by cudnn
        # autotune / cold caches.
        _ = run_inference(t0_start_us, n_cams, n_frames, diff_steps)

        # Real-time simulation loop. sim_t advances only by GPU inference latency
        # (the only stage that exists on a real car in steady state — see
        # comments above the timings dict). data_load / preprocess are tracked
        # separately for the summary table but don't shift sim_t.
        preds = []  # list of dicts: {start_t, end_t, pred_xy, gt_xy, ade}
        sim_t = 0.0
        ades = []
        data_load_ms_list, preprocess_ms_list, gpu_infer_ms_list = [], [], []
        while sim_t < duration_s:
            t_us = t0_start_us + int(sim_t * 1_000_000)
            try:
                pred_xy, gt_xy, _fw, timings = run_inference(
                    t_us, n_cams, n_frames, diff_steps
                )
            except Exception as e:
                print(f"[render]   sim_t={sim_t:.2f}s load/infer failed "
                      f"({type(e).__name__}: {e})")
                break
            T = min(pred_xy.shape[0], gt_xy.shape[0])
            ade = float(np.linalg.norm(pred_xy[:T] - gt_xy[:T], axis=1).mean())
            gpu_infer_s = timings["gpu_infer_ms"] / 1000.0
            end_t = sim_t + gpu_infer_s
            preds.append({
                "start_t": sim_t, "end_t": end_t,
                "pred_xy": pred_xy, "gt_xy": gt_xy, "ade": ade,
            })
            ades.append(ade)
            data_load_ms_list.append(timings["data_load_ms"])
            preprocess_ms_list.append(timings["preprocess_ms"])
            gpu_infer_ms_list.append(timings["gpu_infer_ms"])
            print(f"[render]   sim_t={sim_t:5.2f}s  "
                  f"load={timings['data_load_ms']:5.1f}ms  "
                  f"prep={timings['preprocess_ms']:5.1f}ms  "
                  f"gpu={timings['gpu_infer_ms']:6.1f}ms  "
                  f"ADE={ade:5.2f}m")
            sim_t = end_t

        n_preds = len(preds)
        if n_preds == 0:
            print(f"[render] no preds for {label}, skipping")
            continue
        hz = n_preds / duration_s

        # 3) Render video at display_fps: for each display tick, find the most
        # recent prediction whose end_t <= tick_t; hold previous if none.
        print(f"[render]   rendering {n_display} display frames ...")
        frames_out = []
        bev_render_ms_list = []
        pred_idx = -1  # index of latest prediction that has finished
        for i in range(n_display):
            tick_t = i / display_fps
            while pred_idx + 1 < n_preds and preds[pred_idx + 1]["end_t"] <= tick_t:
                pred_idx += 1
            t_r = time.perf_counter()
            if pred_idx < 0:
                frames_out.append(render_composite(
                    display_cache[i], None, None, label, float("nan"),
                    tick_t, 0.0, 0.0,
                ))
            else:
                p = preds[pred_idx]
                age = tick_t - p["end_t"]
                frames_out.append(render_composite(
                    display_cache[i], p["pred_xy"], p["gt_xy"],
                    label, p["ade"], tick_t, age, hz,
                ))
            bev_render_ms_list.append((time.perf_counter() - t_r) * 1000.0)

        # Pad to uniform shape.
        max_h = max(f.shape[0] for f in frames_out)
        max_w = max(f.shape[1] for f in frames_out)
        padded = []
        for f in frames_out:
            h, w = f.shape[:2]
            if h == max_h and w == max_w:
                padded.append(f)
            else:
                canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
                canvas[:h, :w] = f
                padded.append(canvas)

        out_path = f"/outputs/{cfg_idx+1}_{label}.mp4"
        t_enc = time.perf_counter()
        mediapy.write_video(out_path, padded, fps=display_fps)
        video_encode_ms = (time.perf_counter() - t_enc) * 1000.0
        avg_ade = float(np.mean(ades))
        print(f"[render] wrote {out_path}  duration={duration_s:.1f}s @ "
              f"{display_fps}fps  n_preds={n_preds}  Hz={hz:.2f}  "
              f"avg minADE={avg_ade:.2f}m")
        results.append({
            "config": label,
            "num_cams": n_cams, "num_frames": n_frames, "diff_steps": diff_steps,
            "n_preds": n_preds, "hz_sim": hz,
            "avg_min_ade_m": avg_ade, "per_pred_ade": ades,
            "video_path": out_path,
            "data_load_ms_mean": float(np.mean(data_load_ms_list)),
            "preprocess_ms_mean": float(np.mean(preprocess_ms_list)),
            "gpu_infer_ms_mean": float(np.mean(gpu_infer_ms_list)),
            "bev_render_ms_mean": float(np.mean(bev_render_ms_list)),
            "video_encode_ms": video_encode_ms,
        })

    videos_vol.commit()

    # Per-stage timing table. "car steady-state" isolates the stages that will
    # exist on a real vehicle: the GPU forward pass only. data_load (pickle/disk
    # read) and bev_render / video_encode are benchmark artifacts (no BEV plot
    # or MP4 muxer is on the hot path of a live vehicle). preprocess (tokenize
    # + apply_chat_template + host->device) DOES remain on-vehicle, but the live
    # system can pipeline it with the previous GPU forward, so it's shown
    # separately too.
    print("\n" + "=" * 90)
    print("[render] PER-STAGE TIMING (ms, mean per prediction or per frame)")
    print("=" * 90)
    hdr = (
        f"{'config':22s} | {'data_load':>10s} | {'preproc':>8s} | "
        f"{'gpu_inf':>8s} | {'bev_rndr':>9s} | {'vid_enc':>8s} | "
        f"{'car_gpu_hz':>10s} | {'sim_hz':>7s} | {'ADE(m)':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        car_hz = 1000.0 / r["gpu_infer_ms_mean"] if r["gpu_infer_ms_mean"] > 0 else 0.0
        print(
            f"{r['config']:22s} | "
            f"{r['data_load_ms_mean']:10.1f} | "
            f"{r['preprocess_ms_mean']:8.1f} | "
            f"{r['gpu_infer_ms_mean']:8.1f} | "
            f"{r['bev_render_ms_mean']:9.1f} | "
            f"{r['video_encode_ms']:8.0f} | "
            f"{car_hz:10.2f} | "
            f"{r['hz_sim']:7.2f} | "
            f"{r['avg_min_ade_m']:7.2f}"
        )
    print("=" * 90)
    print("Legend:")
    print("  data_load  = CPU: pickle/disk read (benchmark-only, gone on car)")
    print("  preproc    = CPU: tokenize + apply_chat_template + host->device (on car)")
    print("  gpu_inf    = GPU: vision tower + VLM prefill + action expert diffusion (on car)")
    print("  bev_rndr   = CPU: matplotlib BEV composite (benchmark-only, gone on car)")
    print("  vid_enc    = CPU: mediapy MP4 encode for whole clip (benchmark-only)")
    print("  car_gpu_hz = 1000 / gpu_inf — steady-state Hz on A100 if CPU is pipelined")
    print("  sim_hz     = preds made per sim-second in this benchmark")
    print("  Note: Thor (Jetson) is ~3-4× slower than A100 without TRT/FP8 tuning.")
    print("[render] Download videos to laptop:")
    print("    modal volume get alpamayo-videos / ./alpamayo_videos")
    return results


@app.local_entrypoint()
def videos(
    t0_start_s: float = 1.7,
    duration_s: float = 18.0,
    display_fps: int = 10,
    skip_cot: bool = True,
    config_indices: str = "",
    clip_id: str = CLIP_ID,
    skip_precache: bool = False,
    model_variant: str = "r1",
):
    """Precache frames to the PAI volume (idempotent), then render + save one
    real-time simulation MP4 per config. Defaults cover the full usable window
    of one ~20s PAI clip: 1.7s (history safety margin) to 19.7s. `clip_id` may
    be a UUID prefix — it's resolved to the full UUID via the PAI clip index."""
    if len(clip_id) < 36:
        clip_id = resolve_clip_id.remote(clip_id)
        print(f"[local] resolved clip_id: {clip_id}")
    if not skip_precache:
        print(f"[local] precaching clip={clip_id} [{t0_start_s:.2f}s, "
              f"{t0_start_s+duration_s:.2f}s] ...")
        info = precache_clip.remote(
            clip_id=clip_id, t0_start_s=t0_start_s, duration_s=duration_s
        )
        print(f"[local] precache done: {info}")

    results = render_videos.remote(
        t0_start_s=t0_start_s,
        duration_s=duration_s,
        display_fps=display_fps,
        skip_cot=skip_cot,
        config_indices=config_indices,
        clip_id=clip_id,
        model_variant=model_variant,
    )
    print("\n[local] per-config results:")
    for r in results:
        car_hz = 1000.0 / r["gpu_infer_ms_mean"] if r["gpu_infer_ms_mean"] > 0 else 0.0
        print(
            f"  {r['config']:20s}  n_preds={r['n_preds']:3d}  "
            f"sim_hz={r['hz_sim']:.2f}  car_gpu_hz={car_hz:.2f}  "
            f"avg minADE={r['avg_min_ade_m']:.2f}m  "
            f"(load={r['data_load_ms_mean']:.0f}ms  "
            f"prep={r['preprocess_ms_mean']:.0f}ms  "
            f"gpu={r['gpu_infer_ms_mean']:.0f}ms  "
            f"bev={r['bev_render_ms_mean']:.0f}ms/fr)"
        )
