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
    .pip_install("wheel", "packaging", "ninja")
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
)

# FP8 image: base + torchao. torchao ships prebuilt wheels (no CUDA source
# compile) and provides float8_dynamic_activation_float8_weight which applies
# post-training FP8 quantization to nn.Linear modules in-place. This bypasses
# the TransformerEngine source-build chain that previously blocked us on
# cudnn.h, libcudnn.so symlinks, and 15+min nvcc compiles of
# transformer_engine_torch. The runtime speedup path is the same (scaled_mm
# FP8 gemm on H100 tensor cores); only the software stack differs.
image_fp8 = (
    image.pip_install("torchao>=0.8.0")
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
    )
)

# TRT image: base + torch_tensorrt. Since tensorrt_llm's HF auto-converter does
# not support Qwen3-VL (2024-Q4 arch) and the wheel install itself failed on
# Modal's py3.12 sandbox (cp310/cp311-only wheels, sdist needs nvidia-smi at
# build time), we pivot this ablation to torch_tensorrt which compiles the
# torch ViT module directly to a TensorRT engine via ONNX-free Dynamo export.
# This still answers "what is the speedup from TRT kernel fusion on the VLM
# side?" even if not strictly through the TRT-LLM LLM API.
image_trtllm = (
    # Pin torch-tensorrt==2.8.0 — the version that ships against torch 2.8.0 in
    # pytorch's release matrix. Previous attempt used `torch-tensorrt>=2.5.0`
    # which resolved to torch-tensorrt 2.11.0 and dragged torch up to 2.11.0,
    # breaking torchvision 0.23.0 ABI (torchvision::nms missing). No
    # extra-index-url: PyPI ships generic manylinux wheels that link against
    # whichever torch is already present.
    image.pip_install("torch-tensorrt==2.8.0")
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
    )
)

# ParoQuant image: base + paroquant + vLLM (Marlin W4A8 kernels).
# FlashDriveVLA's Alpamayo-R1-10B-finetuned-PARO checkpoint is in vLLM's
# Marlin format (confirmed by smoke test: checkpoint ships g_idx,
# g_idx_sort_indices, workspace, input_global_scale — all Marlin-specific).
# paroquant's transformers backend uses AWQ GEMM (W4A16) which is a different
# on-disk layout, so we import vllm directly for its Marlin kernel bindings.
#
# Alpamayo's own pyproject pins vllm==0.11.0 so we match that. Using
# --no-deps on paroquant itself to avoid pulling an incompatible vllm>=0.15.
image_paro = (
    image
    .pip_install("vllm==0.11.0")
    .pip_install("paroquant>=0.1.12", extra_options="--no-deps")
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
    )
)

# DFlash image: base + dflash[transformers]. Only the Transformers backend
# is relevant here because Alpamayo-R1 runs model.vlm.generate() directly
# (not via a vLLM/SGLang server). DFlash's draft.spec_generate(target=...)
# accepts an AutoModelForCausalLM target — we wrap model.vlm.language_model
# (the Qwen3 text decoder underneath Qwen3VLForConditionalGeneration) after
# feeding in VLM-embedded input tokens.
image_dflash = (
    image.pip_install(
        "git+https://github.com/z-lab/dflash.git",
        extra_options="--no-deps",  # Avoid pulling extras that reinstall torch
    )
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
    image=image_fp8,
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
    image=image_fp8,
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
    use_compile: bool = True,
    pixel_scale: float = 1.0,
    vlm_attn_impl: str = "flash_attention_2",
    reuse_visual: bool = False,
    horizon_waypoints: int = 64,
    diff_steps_override: int = 0,
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
    cfg.attn_implementation = vlm_attn_impl
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

    if horizon_waypoints != 64:
        old_n = model.action_space.n_waypoints
        model.action_space.n_waypoints = horizon_waypoints
        new_dims = list(model.action_space.get_action_space_dims())
        model.diffusion.x_dims = new_dims
        print(f"[render] horizon override: n_waypoints {old_n} -> "
              f"{horizon_waypoints} ({horizon_waypoints*model.action_space.dt:.1f}s); "
              f"x_dims -> {new_dims}")

    if reuse_visual:
        # CEILING measurement for "vision-token reuse across ticks" (ablation
        # #9). Cache the entire ViT output on first real call and return it
        # verbatim thereafter. A deployable version would cache per-camera and
        # gate on egomotion delta; this upper-bounds the Hz you could ever get.
        # ADE is expected to degrade because later predictions see stale
        # visual content (all from the first timestep).
        _vit = model.vlm.visual
        _orig_vit_forward = _vit.forward
        _vit_cache = {"out": None}

        def _cached_vit_forward(*args, **kwargs):
            if _vit_cache["out"] is not None:
                return _vit_cache["out"]
            out = _orig_vit_forward(*args, **kwargs)
            _vit_cache["out"] = out
            return out

        _vit.forward = _cached_vit_forward
        print("[render] vision-token reuse: ViT output will be cached after "
              "first call (ceiling mode — all ticks reuse t0's visual features)")

    if use_compile:
        # Compile only the expert: the VLM (vision + text decoder) uses
        # flash_attn_varlen_func, which registers a custom C++ op whose SymInt
        # args get passed 0-d fake tensors during dynamo tracing → immediate
        # compile failure ("Expected a value of type 'int' for argument
        # 'max_seqlen_q' but instead found FakeTensor"). The expert uses SDPA
        # (4D float mask path) which is compile-friendly, and it runs
        # diff_steps=10 fwds per call, so it's where reduce-overhead / CUDA
        # graphs pay off most. dynamic=False because num_cams/num_frames/
        # diff_steps are constant across iters of one render_videos call.
        print("[render] compiling expert (mode=reduce-overhead, dynamic=False) ...")
        model.expert = torch.compile(
            model.expert, mode="reduce-overhead", dynamic=False
        )

    processor = helper.get_processor(model.tokenizer)
    if pixel_scale != 1.0:
        # Shrink (or grow) the Qwen3-VL image-token budget. Each image is
        # resized so its pixel count lands in [min_pixels, max_pixels]; fewer
        # pixels -> fewer ViT tokens -> smaller VLM prefill.
        ip = processor.image_processor
        old_min, old_max = ip.min_pixels, ip.max_pixels
        ip.min_pixels = int(old_min * pixel_scale)
        ip.max_pixels = int(old_max * pixel_scale)
        print(
            f"[render] pixel budget: min {old_min}->{ip.min_pixels}  "
            f"max {old_max}->{ip.max_pixels}  (scale={pixel_scale})"
        )
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
        if use_compile:
            # The uncompiled VLM mutates its KV cache in-place (cat into
            # self.keys/self.values). The compiled expert still holds refs to
            # those tensors from the prior iter's cudagraph capture, so the
            # next VLM call clobbers them → "accessing tensor output of
            # CUDAGraphs that has been overwritten". mark_step_begin tells
            # the cudagraph manager to start a fresh lifetime.
            torch.compiler.cudagraph_mark_step_begin()
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
        if diff_steps_override > 0:
            diff_steps = diff_steps_override
            label = f"{label}_d{diff_steps}"
        if horizon_waypoints != 64:
            label = f"{label}_h{horizon_waypoints}"
        print(f"\n[render] ({run_i+1}/{len(configs_to_run)}) config={label}")

        # Warmup so first measurement isn't distorted by cudnn autotune / cold
        # caches. With torch.compile + mode="reduce-overhead" we need at least
        # 2 passes to capture CUDA graphs; add a third to absorb the prefill-vs-
        # decode shape split (max_gen=1 means one prefill + one 1-token decode,
        # so the VLM sees two static shapes per call → two separate compilations).
        n_warmup = 3 if use_compile else 1
        print(f"[render]   warmup x{n_warmup} ...")
        for _ in range(n_warmup):
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
    use_compile: bool = True,
    pixel_scale: float = 1.0,
    vlm_attn_impl: str = "flash_attention_2",
    reuse_visual: bool = False,
    horizon_waypoints: int = 64,
    diff_steps_override: int = 0,
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
        use_compile=use_compile,
        pixel_scale=pixel_scale,
        vlm_attn_impl=vlm_attn_impl,
        reuse_visual=reuse_visual,
        horizon_waypoints=horizon_waypoints,
        diff_steps_override=diff_steps_override,
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


# -----------------------------------------------------------------------------
# ABLATION MEASUREMENT HELPER
# -----------------------------------------------------------------------------
#
# Shared sim-loop body that matches the baseline methodology (render_videos)
# exactly: 3 warmup iters, then while sim_t < duration_s advance by gpu_inf_s.
# Each ablation function sets up its model, then calls this. Returns the dict
# that gets printed as a CSV row.
#
# IMPORTANT: uses clip 25cd4769 (the results.csv clip), t0_start_s=5.1,
# duration_s=18 → targets ~45 preds for a 400ms model to match baseline row.
# Config is fixed to 4cam_4fr_10diff (the baseline config).

ABL_CLIP = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
ABL_T0_S = 5.1
ABL_DURATION_S = 18.0
ABL_N_CAMS = 4
ABL_N_FRAMES = 4
ABL_DIFF_STEPS = 10


def _run_ablation_sim(model, processor, helper, load_physical_aiavdataset,
                     avdi, all_cams, needs_cam_indices_in_msg,
                     warmup_iters=3, label="ablation", pre_iter_hook=None,
                     n_cams=None, n_frames=None, diff_steps=None,
                     duration_s=None, skip_cot=True, cot_max_tokens=256):
    """Run baseline-methodology sim loop for one ablation.

    Assumes caller has already applied the optimization to `model`. Loads data
    via physical_ai_av directly (no precache lookup — the ablations test clips
    that the baseline precache may not cover). Returns dict with gpu_inf_ms,
    hz_sim, avg_ade_m, n_preds plus per-iter arrays for logging.
    """
    import time
    import numpy as np
    import torch

    t0_start_us = int(ABL_T0_S * 1_000_000)
    _n_cams = ABL_N_CAMS if n_cams is None else n_cams
    _n_frames = ABL_N_FRAMES if n_frames is None else n_frames
    _diff_steps = ABL_DIFF_STEPS if diff_steps is None else diff_steps
    _duration_s = ABL_DURATION_S if duration_s is None else duration_s

    def _build_messages(data):
        frames = data["image_frames"].flatten(0, 1)
        camera_indices = data.get("camera_indices")
        if needs_cam_indices_in_msg:
            msgs = helper.create_message(
                frames, camera_indices=camera_indices,
                num_frames_per_camera=_n_frames,
            )
        else:
            msgs = helper.create_message(frames)
        if skip_cot:
            # End prompt at <|traj_future_start|> so no CoC decode.
            msgs[-1]["content"][0]["text"] = (
                "<|cot_start|><|cot_end|><|traj_future_start|>"
            )
        return msgs

    def _run_one(t_us):
        if pre_iter_hook is not None:
            pre_iter_hook()
        t0 = time.perf_counter()
        data = load_physical_aiavdataset(
            ABL_CLIP, t0_us=t_us, avdi=avdi,
            num_frames=_n_frames, camera_features=all_cams[:_n_cams],
        )
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        data_load_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        messages = _build_messages(data)
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

        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98, temperature=0.6,
                num_traj_samples=1,
                max_generation_length=(1 if skip_cot else cot_max_tokens),
                return_extra=True,
                diffusion_kwargs={"inference_step": _diff_steps},
            )
        torch.cuda.synchronize()
        gpu_infer_ms = (time.perf_counter() - t_infer) * 1000.0

        pred_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        T = min(pred_xy.shape[0], gt_xy.shape[0])
        ade = float(np.linalg.norm(pred_xy[:T] - gt_xy[:T], axis=1).mean())
        return gpu_infer_ms, ade, data_load_ms, preprocess_ms

    print(f"[{label}] warmup x{warmup_iters} ...")
    for _ in range(warmup_iters):
        _ = _run_one(t0_start_us)

    print(f"[{label}] sim loop (duration={_duration_s}s) ...")
    sim_t = 0.0
    gpu_ms_list, ade_list = [], []
    while sim_t < _duration_s:
        t_us = t0_start_us + int(sim_t * 1_000_000)
        try:
            gpu_ms, ade, _, _ = _run_one(t_us)
        except Exception as e:
            print(f"[{label}]   sim_t={sim_t:.2f}s failed "
                  f"({type(e).__name__}: {e})")
            break
        gpu_ms_list.append(gpu_ms)
        ade_list.append(ade)
        print(f"[{label}]   sim_t={sim_t:5.2f}s  gpu={gpu_ms:6.1f}ms  ADE={ade:5.2f}m")
        sim_t += gpu_ms / 1000.0

    n_preds = len(gpu_ms_list)
    if n_preds == 0:
        return {"error": "no predictions completed"}

    gpu_inf_ms = float(np.mean(gpu_ms_list))
    hz_sim = n_preds / _duration_s
    avg_ade = float(np.mean(ade_list))
    print(f"\n[{label}] RESULT: gpu_inf_ms={gpu_inf_ms:.1f}  "
          f"hz={hz_sim:.2f}  ade={avg_ade:.2f}m  n_preds={n_preds}")
    return {
        "gpu_inf_ms": gpu_inf_ms,
        "hz": hz_sim,
        "ade_m": avg_ade,
        "n_preds": n_preds,
    }


def _setup_model_common():
    """Shared boilerplate: env, imports, model load, SDPA-for-expert fix."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HUGGING_FACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import time
    import torch

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper
    import physical_ai_av

    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "flash_attention_2"
    t_load = time.perf_counter()
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    # Expert uses 4D float mask → FA2 incompat → force SDPA.
    model.expert.config._attn_implementation = "sdpa"
    for _m in model.expert.modules():
        _c = getattr(_m, "config", None)
        if _c is not None and hasattr(_c, "_attn_implementation_internal"):
            _c._attn_implementation_internal = "sdpa"
    print(f"[setup] model loaded in {time.perf_counter()-t_load:.1f}s")

    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    all_cams = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]
    return {
        "model": model, "processor": processor, "helper": helper,
        "load_physical_aiavdataset": load_physical_aiavdataset,
        "avdi": avdi, "all_cams": all_cams,
        "needs_cam_indices_in_msg": False,
    }


# -----------------------------------------------------------------------------
# ABLATION #3: FP8 VLM via TransformerEngine
# -----------------------------------------------------------------------------
#
# Strategy: walk model.vlm modules, swap every nn.Linear for te.Linear
# (preserving weights). Wrap VLM forward in te.fp8_autocast with delayed
# scaling recipe. Do 1 calibration pass (amax=history) before measurement.
#
# Risks: (1) TE requires torch CUDA build; may fail. (2) fp8_autocast may
# not interact cleanly with flash_attn inside Qwen3-VL's attention modules.
# (3) Delayed scaling needs a few fwds to converge amax history.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 45,
)
def ablation_fp8_vlm():
    """FP8 weights+activations on VLM via torchao dynamic quantization.

    Uses torchao.quantization.quantize_ with
    Float8DynamicActivationFloat8WeightConfig to swap every eligible nn.Linear
    in model.vlm to an FP8 kernel (scaled_mm on H100). Post-training, no
    calibration, no autocast wrapping needed. Bounded by reuse_visual_ceiling
    (+7% Hz max on 4cam_4fr_10diff) since VLM is <~10% of gpu_inf.
    """
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_
    # Try weight-only FP8 first: activations stay bf16, weights become FP8
    # tensor subclasses and _scaled_mm decompresses on the fly. This avoids
    # the per-activation dynamic quant overhead that made the first attempt
    # (Float8DynamicActivationFloat8WeightConfig) end up SLOWER than bf16 on
    # Qwen3-VL's relatively small attention matmuls. Weight-only still gives
    # memory-bandwidth wins which dominate on the ViT.
    try:
        from torchao.quantization import Float8WeightOnlyConfig
        cfg = Float8WeightOnlyConfig()
        cfg_name = "Float8WeightOnlyConfig"
    except Exception:
        # Older torchao API name fallback.
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        cfg = Float8DynamicActivationFloat8WeightConfig()
        cfg_name = "Float8DynamicActivationFloat8WeightConfig(fallback)"

    # Some kernels need this flag to allow reduced-precision FP8 reductions.
    try:
        torch.backends.cuda.matmul.allow_fp8_reduced_precision_reduction = True
    except Exception:
        pass

    # Count Linears first (for the note).
    n_linear_before = sum(
        1 for m in model.vlm.modules() if isinstance(m, nn.Linear)
    )
    print(f"[fp8_vlm] model.vlm has {n_linear_before} nn.Linear modules; "
          f"applying torchao FP8 ({cfg_name}) ...")

    # Apply FP8 only to the VLM (expert stays bf16). torchao's quantize_ walks
    # the module tree and wraps each eligible Linear's weight in a Float8Tensor
    # subclass (module type stays nn.Linear; only the weight storage changes).
    # Filter skips Linears whose dims aren't multiples of 16 (FP8 alignment).
    def _filter_fn(module, fqn):
        if not isinstance(module, nn.Linear):
            return False
        return module.in_features % 16 == 0 and module.out_features % 16 == 0

    try:
        quantize_(
            model.vlm,
            cfg,
            filter_fn=_filter_fn,
        )
    except Exception as e:
        return {
            "optimization": "fp8_vlm_torchao",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
            "note": f"torchao FP8 quantize_ failed: {type(e).__name__}: {e}",
            "error": str(e),
        }

    # Count again to confirm swap. torchao subclasses the *weight tensor* in
    # recent versions (the module remains nn.Linear), so we check the tensor
    # type not the module type.
    n_fp8_linear = 0
    for m in model.vlm.modules():
        if isinstance(m, nn.Linear) and hasattr(m, "weight"):
            tname = type(m.weight).__name__
            if "Float8" in tname or "float8" in tname.lower():
                n_fp8_linear += 1
    print(f"[fp8_vlm] torchao swapped {n_fp8_linear} Linear weights to FP8 "
          f"({n_linear_before - n_fp8_linear} left in bf16); cfg={cfg_name}.")

    print("[fp8_vlm] starting measurement ...")
    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3,
        label="fp8_vlm",
    )
    result["optimization"] = "fp8_vlm_torchao"
    result["note"] = (f"torchao {cfg_name}; "
                      f"swapped {n_fp8_linear}/{n_linear_before} nn.Linear "
                      f"weights in model.vlm to FP8.")
    return result


@app.local_entrypoint()
def fp8_vlm():
    r = ablation_fp8_vlm.remote()
    print("[local] FP8-VLM result:", r)


# -----------------------------------------------------------------------------
# ABLATION #7: CUDA graphs + static KV cache for VLM
# -----------------------------------------------------------------------------
#
# Strategy: capture model.vlm.visual (ViT) as a CUDA graph with preallocated
# input buffers. The ViT has fixed input shape per config (num_cams * num_frames
# images of constant pixel count), so a single graph works. Replay on each call
# via copy-in of real pixel values. This eliminates Python/CUDA launch overhead
# for the ViT while keeping compute unchanged (unlike reuse_visual which
# returned stale output).
#
# Note: the "static KV cache" part of the ablation name refers to preallocating
# the KV tensors for the text decoder. Since skip_cot=True and max_gen=1, the
# text decoder does exactly one prefill (no autoregressive loop), so there IS
# no KV growth to optimize. The win is all from the ViT graph replay.
#
# Risks: (1) CUDAGraph capture may fail if the ViT uses cudnn benchmark or
# other non-capturable ops. (2) Output tensors live inside capture memory, so
# we must clone them out each replay.

# Dedicated app + image so this ablation is not blocked by unrelated image
# builds (fp8 transformer-engine, trtllm tensorrt_llm) that other subagents
# own. Modal hydrates every image referenced by any function of an app at
# run start; if any image build fails, the whole app aborts. Isolating this
# function into its own app (with an image that only adds add_local_dir on
# top of the already-built base image) sidesteps that coupling so broken
# sibling image builds cannot cascade and block this ablation. add_local_dir
# must be the last image step (Modal disallows run/pip layers after it).
image_cg_vlm = image.add_local_dir(
    ".",
    remote_path=REPO_ROOT,
    ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
)

app_cg_vlm = modal.App("alpamayo-r1-bench-cg-vlm", image=image_cg_vlm)


@app_cg_vlm.function(
    gpu="H100",
    image=image_cg_vlm,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def ablation_cuda_graph_vlm():
    """CUDA graphs on the ViT via torch.compile(mode='reduce-overhead').

    Previous attempts with manual torch.cuda.CUDAGraph capture failed because
    Qwen3-VL's ViT forward contains cudaMalloc-in-forward (from FA2 varlen or
    window-attn kernels) that cudaStreamBeginCapture cannot tolerate, AND any
    failed capture poisons the global CUDA RNG state causing downstream
    torch.multinomial to throw "Offset increment outside graph capture".

    Fix: torch.compile(mode='reduce-overhead') lets dynamo handle graph
    capture via its own mempool + stream management. It's the same pattern
    we already used successfully on the expert (1.35× speedup). The ViT is a
    standard vision transformer so dynamo should trace it cleanly; if there's
    a dynamo failure it reports a traceback rather than silently corrupting
    RNG state.

    Also forces visual.attn_impl to SDPA: Qwen3-VL's ViT uses flash_attn_varlen
    which trips dynamo (FakeTensor vs SymInt), same incompat we hit when trying
    to compile the full VLM for the torch.compile ablation.
    """
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch

    # Force visual tower to SDPA so torch.compile can trace it.
    vit = model.vlm.visual
    try:
        vit.config._attn_implementation = "sdpa"
        if hasattr(vit.config, "_attn_implementation_internal"):
            vit.config._attn_implementation_internal = "sdpa"
        for _m in vit.modules():
            _c = getattr(_m, "config", None)
            if _c is not None and hasattr(_c, "_attn_implementation_internal"):
                _c._attn_implementation_internal = "sdpa"
        print("[cg_vlm] visual tower forced to SDPA (compile-friendly).")
    except Exception as e:
        print(f"[cg_vlm] SDPA force warning: {e}")

    print("[cg_vlm] compiling ViT with mode='default' (dynamo fusion only, no CUDA graphs) ...")
    # Attempt 1 of this retry used mode='reduce-overhead' which REGRESSED
    # gpu_inf_ms from 405->498 (cudagraph capture overhead + mempool friction
    # with expert > the savings). Switching to mode='default' to get the
    # dynamo kernel-fusion benefit without the cudagraph overhead.
    try:
        model.vlm.visual = torch.compile(
            model.vlm.visual, mode="default", dynamic=False
        )
        compile_note = ("torch.compile(ViT, mode='default', "
                        "dynamic=False) applied; dynamo kernel fusion only, "
                        "no CUDA graphs.")
    except Exception as e:
        compile_note = (f"torch.compile failed: {type(e).__name__}: "
                        f"{str(e).splitlines()[0]}")
        print(f"[cg_vlm] {compile_note}")

    # Run measurement. Warmups drive the compile + capture.
    print("[cg_vlm] starting measurement ...")
    # Add mark_step_begin per iter in case cudagraphs' tensor lifetime
    # clashes with the expert's allocator (same pattern we used for the
    # expert compile ablation).
    from functools import wraps
    _orig_run_sim = _run_ablation_sim

    # Wrap rollout to call mark_step_begin per iter.
    _orig_sample = model.sample_trajectories_from_data_with_vlm_rollout

    @wraps(_orig_sample)
    def _sample_with_mark_step(*args, **kwargs):
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        return _orig_sample(*args, **kwargs)

    model.sample_trajectories_from_data_with_vlm_rollout = _sample_with_mark_step

    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=4,  # extra warmup: compile + cudagraph capture needs ≥2 passes.
        label="cg_vlm",
    )
    result["optimization"] = "cuda_graphs_vit"
    result["note"] = (f"ViT compiled with torch.compile(mode='reduce-overhead') "
                      f"(visual forced to SDPA). {compile_note}")
    return result


@app_cg_vlm.local_entrypoint()
def cuda_graph_vlm():
    r = ablation_cuda_graph_vlm.remote()
    print("[local] cuda_graph_vlm result:", r)


# -----------------------------------------------------------------------------
# ABLATION #5: TensorRT-LLM export of VLM
# -----------------------------------------------------------------------------
#
# Strategy: attempt to load Qwen3-VL via tensorrt_llm's HF auto-converter.
# TRT-LLM has first-class support for Qwen2/Qwen2.5 but Qwen3-VL is a
# 2024-Q4 architecture and may not be supported. We try three paths in order:
#   1. tensorrt_llm.LLM(hf_repo="nvidia/Alpamayo-R1-10B")
#   2. tensorrt_llm.LLM(hf_repo="Qwen/Qwen3-VL") for raw VLM
#   3. HF → TRT-LLM checkpoint convert via builder CLI
#
# On success: wrap the VLM forward to dispatch to the TRT engine for both
# ViT and text prefill. On failure: capture the specific error for results.csv
# so we know what architecture support is missing.

@app.function(
    gpu="H100",
    image=image_trtllm,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 90,  # engine build can be long
)
def ablation_trtllm_vlm():
    """TensorRT export of the ViT via torch_tensorrt.

    We pivoted from tensorrt_llm.LLM() (which cannot auto-convert Qwen3-VL:
    2024-Q4 architecture, not in supported models list; install path also
    blocked on Modal py3.12 sandbox) to torch_tensorrt.dynamo on the ViT.
    This still answers "what is the TRT-kernel-fusion speedup on the VLM
    side?" even though it doesn't go through the TRT-LLM LLM API.

    Scope: compiles model.vlm.visual (the ViT) to a TensorRT engine via
    Dynamo-based export. Text decoder stays eager (TRT-LLM would be needed
    for that and is multi-day engineering per the previous row's note).
    Bounded by reuse_visual_ceiling (+7% Hz max) since ViT is <~10% of
    gpu_inf on this 4cam_4fr_10diff config.
    """
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    import traceback

    # Force ViT to SDPA so torch_tensorrt can trace it (same reason as cg_vlm:
    # flash_attn_varlen is not dynamo-traceable).
    vit = model.vlm.visual
    try:
        vit.config._attn_implementation = "sdpa"
        if hasattr(vit.config, "_attn_implementation_internal"):
            vit.config._attn_implementation_internal = "sdpa"
        for _m in vit.modules():
            _c = getattr(_m, "config", None)
            if _c is not None and hasattr(_c, "_attn_implementation_internal"):
                _c._attn_implementation_internal = "sdpa"
    except Exception as e:
        print(f"[trt] SDPA force warning: {e}")

    # Compile ViT with torch_tensorrt via torch.compile(backend='tensorrt').
    # This lazy-compiles on first forward; we warm it up inside the measurement
    # harness. If the compile fails, we record the error as the note and still
    # run the eager measurement so the row has a valid baseline value.
    compile_note = None
    try:
        import torch_tensorrt
        print(f"[trt] torch_tensorrt=={torch_tensorrt.__version__}")
        # Use the torch.compile backend='tensorrt' path — simplest integration,
        # handles dynamic shapes better than torch_tensorrt.compile(..., ir='dynamo').
        model.vlm.visual = torch.compile(
            model.vlm.visual,
            backend="tensorrt",
            options={
                "enabled_precisions": {torch.bfloat16, torch.float16},
                "truncate_long_and_double": True,
                "min_block_size": 1,
            },
        )
        compile_note = ("torch.compile(ViT, backend='tensorrt', "
                        "precisions={bf16,fp16}) applied.")
        print(f"[trt] {compile_note}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[trt] torch_tensorrt compile setup failed: {e}\n{tb}")
        compile_note = (f"torch_tensorrt compile setup failed: "
                        f"{type(e).__name__}: {str(e).splitlines()[0]}")

    # Run the measurement. Warmups will trigger TRT engine compilation (slow
    # first pass, fast afterwards).
    print("[trt] starting measurement (first warmup triggers TRT engine "
          "compile; may take 60-120s) ...")
    try:
        result = _run_ablation_sim(
            model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
            load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
            avdi=ctx["avdi"], all_cams=ctx["all_cams"],
            needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
            warmup_iters=4,  # TRT compile needs ≥1 real pass; +3 for stability.
            label="trt_vit",
        )
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "optimization": "trt_vit",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
            "note": (f"torch_tensorrt runtime failure during measurement: "
                     f"{type(e).__name__}: {str(e).splitlines()[0]}. "
                     f"{compile_note}"),
            "error_trace": tb,
        }

    result["optimization"] = "trt_vit"
    result["note"] = (f"ViT exported to TensorRT via torch_tensorrt "
                      f"(text decoder stays eager — full TRT-LLM LLM path "
                      f"blocked by Qwen3-VL arch support, see prior row). "
                      f"{compile_note}")
    return result


@app.local_entrypoint()
def trtllm_vlm():
    r = ablation_trtllm_vlm.remote()
    print("[local] trtllm_vlm result:", r)


# -----------------------------------------------------------------------------
# ABLATION: FP8 on EXPERT (the actual bottleneck — 60-70% of gpu_inf)
# -----------------------------------------------------------------------------
#
# Expert uses SDPA natively (4D float mask; FA2 incompatible), so no
# tracer/SDPA-penalty issue. Weight-only keeps activations bf16 and decompresses
# FP8 weights to bf16 inside _scaled_mm — wins on memory bandwidth for the
# repeated diffusion-step matmuls.

def _apply_fp8_to_expert(model):
    """Helper: weight-only FP8 on model.expert. Returns (n_swapped, n_total, cfg_name)."""
    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Float8WeightOnlyConfig
        cfg = Float8WeightOnlyConfig()
        cfg_name = "Float8WeightOnlyConfig"
    except Exception:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        cfg = Float8DynamicActivationFloat8WeightConfig()
        cfg_name = "Float8DynamicActivationFloat8WeightConfig(fallback)"
    try:
        torch.backends.cuda.matmul.allow_fp8_reduced_precision_reduction = True
    except Exception:
        pass

    n_total = sum(1 for m in model.expert.modules() if isinstance(m, nn.Linear))

    def _filter_fn(module, fqn):
        if not isinstance(module, nn.Linear):
            return False
        return module.in_features % 16 == 0 and module.out_features % 16 == 0

    quantize_(model.expert, cfg, filter_fn=_filter_fn)

    n_swapped = 0
    for m in model.expert.modules():
        if isinstance(m, nn.Linear) and hasattr(m, "weight"):
            tname = type(m.weight).__name__
            if "Float8" in tname or "float8" in tname.lower():
                n_swapped += 1
    return n_swapped, n_total, cfg_name


@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 45,
)
def ablation_fp8_expert():
    """FP8 weight-only on model.expert. Eager (no compile). Baseline vs 405ms."""
    ctx = _setup_model_common()
    model = ctx["model"]
    n_swapped, n_total, cfg_name = _apply_fp8_to_expert(model)
    print(f"[fp8_expert] swapped {n_swapped}/{n_total} Linear weights in "
          f"model.expert to FP8; cfg={cfg_name}.")
    print("[fp8_expert] starting measurement ...")
    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3, label="fp8_expert",
    )
    result["optimization"] = "fp8_expert_torchao"
    result["note"] = (f"torchao {cfg_name} on model.expert; "
                      f"swapped {n_swapped}/{n_total} Linears to FP8. Eager "
                      f"(no compile). VLM stays bf16+FA2.")
    return result


@app.local_entrypoint()
def fp8_expert():
    r = ablation_fp8_expert.remote()
    print("[local] fp8_expert result:", r)


# -----------------------------------------------------------------------------
# ABLATION: FP8 expert STACKED with torch.compile(expert)
# -----------------------------------------------------------------------------
# Real question: does FP8 add on top of the compile winner (300.5ms, 3.33 Hz)?

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 45,
)
def ablation_fp8_expert_compile():
    """FP8 expert + torch.compile(expert, reduce-overhead). Stacks opts."""
    ctx = _setup_model_common()
    model = ctx["model"]
    n_swapped, n_total, cfg_name = _apply_fp8_to_expert(model)
    print(f"[fp8_expert_compile] FP8 swapped {n_swapped}/{n_total}; "
          f"now torch.compile(expert, mode='reduce-overhead', dynamic=False) ...")

    import torch
    model.expert = torch.compile(
        model.expert, mode="reduce-overhead", dynamic=False,
    )

    # Per-iter cudagraph reset required to avoid KV-cache clobber of cudagraph
    # outputs (same as the render_videos compile path).
    def _pre_iter():
        torch.compiler.cudagraph_mark_step_begin()

    print("[fp8_expert_compile] starting measurement (first warmup is slow: "
          "dynamo trace + cudagraph capture) ...")
    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=4, label="fp8_expert_compile",
        pre_iter_hook=_pre_iter,
    )
    result["optimization"] = "fp8_expert_torchao+compile"
    result["note"] = (f"torchao {cfg_name} on model.expert ({n_swapped}/"
                      f"{n_total} Linears) + torch.compile(expert, "
                      f"mode='reduce-overhead', dynamic=False). Stacked on "
                      f"compile winner (300.5ms/3.33Hz).")
    return result


@app.local_entrypoint()
def fp8_expert_compile():
    r = ablation_fp8_expert_compile.remote()
    print("[local] fp8_expert_compile result:", r)


# -----------------------------------------------------------------------------
# ABLATION A: diff_steps sweep on compile-expert baseline
# -----------------------------------------------------------------------------
# Compile winner = 300.5ms/3.33Hz at diff_steps=10. Expert runs N fwds per call;
# cutting N is a direct multiplier on the 60-70% of gpu_inf that diffusion owns.
# Sweep {8, 6, 5, 4, 3} in one function (shared model load, shared compile
# warmup). torch.compile(dynamic=False) will specialize on the first diff_steps
# value used — we use the same compiled expert for all values since diff_steps
# only changes the Python loop count, not any tensor shape.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def ablation_diff_steps_sweep():
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    print("[diff_sweep] torch.compile(expert, mode='reduce-overhead', "
          "dynamic=False) ...")
    model.expert = torch.compile(
        model.expert, mode="reduce-overhead", dynamic=False,
    )

    def _pre_iter():
        torch.compiler.cudagraph_mark_step_begin()

    results = []
    for ds in [8, 6, 5, 4, 3]:
        print(f"\n[diff_sweep] === diff_steps={ds} ===")
        try:
            r = _run_ablation_sim(
                model=ctx["model"], processor=ctx["processor"],
                helper=ctx["helper"],
                load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
                avdi=ctx["avdi"], all_cams=ctx["all_cams"],
                needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
                warmup_iters=3, label=f"diff{ds}",
                pre_iter_hook=_pre_iter, diff_steps=ds,
            )
            r["diff_steps"] = ds
            r["optimization"] = f"compile+diff_steps_{ds}"
            results.append(r)
        except Exception as e:
            results.append({
                "diff_steps": ds, "error": f"{type(e).__name__}: {e}",
                "optimization": f"compile+diff_steps_{ds}",
            })
    print("\n[diff_sweep] SUMMARY:")
    for r in results:
        print(" ", r)
    return results


@app.local_entrypoint()
def diff_steps_sweep():
    r = ablation_diff_steps_sweep.remote()
    print("[local] diff_steps_sweep:", r)


# -----------------------------------------------------------------------------
# ABLATION B: cam/frame sweep on compile-expert baseline
# -----------------------------------------------------------------------------
# R1 has only 2 data points: 4cam_4fr (1.56m) and 1cam_1fr (4.62m). Fill in
# intermediates: (2,4), (4,2), (1,4). dynamic=False means each (n_cams,n_frames)
# may trigger recompile on first iter — so we recompile per config (expensive
# warmup) but steady-state is correct.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 90,
)
def ablation_cam_frame_sweep():
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    import copy
    # We need a fresh compile per config because ViT input shape changes with
    # n_cams*n_frames and expert KV-cache shape changes with n_frames. Cheapest
    # path: compile once for each config sequentially, reusing the same model
    # (dynamo will cache the specializations).
    model.expert = torch.compile(
        model.expert, mode="reduce-overhead", dynamic=False,
    )

    def _pre_iter():
        torch.compiler.cudagraph_mark_step_begin()

    configs = [(2, 4), (4, 2), (1, 4)]
    results = []
    for n_cams, n_frames in configs:
        cfg_name = f"{n_cams}cam_{n_frames}fr_10diff"
        print(f"\n[cam_sweep] === n_cams={n_cams} n_frames={n_frames} ===")
        try:
            r = _run_ablation_sim(
                model=ctx["model"], processor=ctx["processor"],
                helper=ctx["helper"],
                load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
                avdi=ctx["avdi"], all_cams=ctx["all_cams"],
                needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
                warmup_iters=4, label=f"c{n_cams}f{n_frames}",
                pre_iter_hook=_pre_iter,
                n_cams=n_cams, n_frames=n_frames,
            )
            r["n_cams"] = n_cams
            r["n_frames"] = n_frames
            r["config"] = cfg_name
            r["optimization"] = f"compile+{cfg_name}"
            results.append(r)
        except Exception as e:
            results.append({
                "n_cams": n_cams, "n_frames": n_frames, "config": cfg_name,
                "error": f"{type(e).__name__}: {e}",
                "optimization": f"compile+{cfg_name}",
            })
    print("\n[cam_sweep] SUMMARY:")
    for r in results:
        print(" ", r)
    return results


@app.local_entrypoint()
def cam_frame_sweep():
    r = ablation_cam_frame_sweep.remote()
    print("[local] cam_frame_sweep:", r)


# =============================================================================
# TE FP8-EXPERT ABLATION (isolated app/image — NGC prebuilt TransformerEngine)
# =============================================================================
#
# Prior TE source build (results.csv row 19) dead-ended on Modal's base Ubuntu
# image: missing cudnn.h, libcudnn.so symlinks, dash-vs-bash, multi-15min nvcc
# compiles. Recommendation was NGC pytorch:24.10-py3 which ships TE 1.11
# precompiled. We use a separate modal.App to isolate build failures from
# sibling ablations (diff_steps/cam_frame/sampler) running in parallel.
#
# Goal: wrap `model.expert`'s nn.Linear forward in te.fp8_autocast so
# activations+weights both go through TE's fused _scaled_mm kernels. This is a
# distinct path from torchao (row 24/26): torchao's dequant-before-gemm added
# per-matmul overhead that beat the bandwidth savings even under cudagraph
# (587.9ms eager, 322.3ms compiled vs 300.5ms compile-only). TE's fused FP8
# gemm avoids that extra dequant step.

image_te = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.03-py3", add_python=None,
    )
    # NGC 25.03 already ships git, ffmpeg, libgl1, libglib2 at the OS level.
    # Modal's image builder cannot reach archive.ubuntu.com (confirmed blocked),
    # so we skip apt_install entirely and rely on NGC's preinstalled system pkgs.
    # NGC 25.03 ships Python 3.12, torch 2.7.0a, TE (latest), flash-attn, cudnn.
    # NOTE: 25.03+ adds /etc/pip/constraint.txt which pins package versions; we
    # remove it so our `transformers==4.57.1` pin isn't overridden.
    .run_commands("truncate -s 0 /etc/pip/constraint.txt || true")
    # Install only the extras the repo needs. Do NOT reinstall torch/te/flash-attn.
    .pip_install(
        "transformers==4.57.1",
        "accelerate>=1.12.0",
        "einops>=0.8.1",
        "av>=16.0.1",
        "pillow>=12.0.0",
        "opencv-python>=4.10.0",
        "scipy>=1.15.0",  # 1.15 adds scipy.spatial.transform.RigidTransform (physical_ai_av req)
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
    # Alpamayo-1.5 is git-only; flash-attn already in NGC so its dep is satisfied.
    .pip_install(
        "git+https://github.com/NVlabs/alpamayo1.5.git",
        extra_options="--no-deps",
    )
    # alpamayo1.5's runtime deps (subset — skip torch/flash-attn which NGC ships).
    .pip_install("timm>=1.0.0", "omegaconf>=2.3.0")
    # Force-upgrade scipy: NGC 25.03 ships scipy 1.14 but physical_ai_av uses
    # spt.RigidTransform which only exists in scipy >=1.15.
    # Downgrade numpy: NGC 25.03 ships numpy 2.4.4 but torch 2.7.0a in the same
    # image was compiled against numpy<2, so torch.from_numpy fails at runtime
    # with "Numpy is not available".
    .run_commands("pip install --upgrade 'scipy>=1.15.0' 'numpy<2'")

    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/finetune/rl",
    })
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=["notebooks", ".git", "uv.lock", "finetune/rl", "**/__pycache__"],
    )
)

app_te = modal.App("alpamayo-r1-bench-te")


def _swap_linears_to_te(module, name_prefix=""):
    """Recursively replace nn.Linear in `module` with te.Linear (weights copied).

    Returns (n_swapped, n_total, n_skipped_align). Skips Linears whose in/out
    features aren't %16-aligned (TE FP8 kernel requirement on Hopper).
    """
    import torch
    import torch.nn as nn
    import transformer_engine.pytorch as te

    n_swapped = 0
    n_total = 0
    n_skipped = 0

    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            n_total += 1
            if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                n_skipped += 1
                continue
            has_bias = child.bias is not None
            device = child.weight.device
            dtype = child.weight.dtype
            te_lin = te.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=has_bias,
                params_dtype=dtype,
                device=device,
            )
            with torch.no_grad():
                te_lin.weight.copy_(child.weight)
                if has_bias:
                    te_lin.bias.copy_(child.bias)
            setattr(module, child_name, te_lin)
            n_swapped += 1
        else:
            s, t, k = _swap_linears_to_te(
                child, name_prefix=f"{name_prefix}.{child_name}",
            )
            n_swapped += s
            n_total += t
            n_skipped += k
    return n_swapped, n_total, n_skipped


@app_te.function(
    gpu="H100",
    image=image_te,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def ablation_fp8_expert_te(stack_compile: bool = False):
    """FP8 weight+activation on model.expert via TransformerEngine fp8_autocast.

    Swaps nn.Linear -> te.Linear inside model.expert (weights copied), then
    monkey-patches model.expert.forward so each call is wrapped in
    te.fp8_autocast. This routes every expert matmul through TE's fused FP8
    _scaled_mm kernel (no per-matmul dequant overhead, unlike torchao).
    """
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    # Version sanity check.
    te_version = getattr(te, "__version__", "unknown")
    print(f"[fp8_expert_te] transformer_engine version: {te_version}")

    n_swap, n_tot, n_skip = _swap_linears_to_te(model.expert)
    print(f"[fp8_expert_te] swapped {n_swap}/{n_tot} nn.Linear -> te.Linear "
          f"in model.expert ({n_skip} skipped for %16 alignment)")

    # Delayed scaling: amax history collected over a few warmup forwards, then
    # scale factors reused across subsequent calls. HYBRID = E4M3 fwd, E5M2 bwd
    # (we're inference-only so bwd format doesn't matter).
    fp8_recipe = DelayedScaling(
        margin=0, fp8_format=Format.HYBRID, amax_history_len=16,
        amax_compute_algo="max",
    )

    _orig_expert_forward = model.expert.forward

    def _fp8_forward(*args, **kwargs):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return _orig_expert_forward(*args, **kwargs)

    model.expert.forward = _fp8_forward

    label = "fp8_expert_te"
    pre_iter_hook = None

    if stack_compile:
        print("[fp8_expert_te] stacking with torch.compile(expert, "
              "mode='reduce-overhead', dynamic=False) ...")
        model.expert = torch.compile(
            model.expert, mode="reduce-overhead", dynamic=False,
        )

        def _pre_iter():
            torch.compiler.cudagraph_mark_step_begin()

        pre_iter_hook = _pre_iter
        label = "fp8_expert_te+compile"

    print(f"[{label}] starting measurement (first warmup collects FP8 "
          "amax history; may be slower)...")

    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=5 if stack_compile else 4,
        label=label,
        pre_iter_hook=pre_iter_hook,
    )
    result["optimization"] = label
    result["n_swapped"] = n_swap
    result["n_total"] = n_tot
    result["n_skipped"] = n_skip
    result["te_version"] = te_version
    result["note"] = (
        f"TE fp8_autocast on model.expert ({n_swap}/{n_tot} Linears->te.Linear, "
        f"{n_skip} skipped for %16 align); "
        f"{'stacked with torch.compile(expert)' if stack_compile else 'standalone, no compile'}; "
        f"compile-only baseline=300.5ms/3.33Hz; torchao eager=587.9ms; "
        f"torchao+compile=322.3ms."
    )
    return result


@app_te.local_entrypoint()
def fp8_expert_te():
    r = ablation_fp8_expert_te.remote(stack_compile=False)
    print("[local] fp8_expert_te (standalone) result:", r)


@app_te.local_entrypoint()
def fp8_expert_te_compile():
    r = ablation_fp8_expert_te.remote(stack_compile=True)
    print("[local] fp8_expert_te (stacked w/ compile) result:", r)

# -----------------------------------------------------------------------------
# ABLATION C: higher-order ODE sampler swap (Heun / AB2 = DPM++ 2M analogue)
# -----------------------------------------------------------------------------
# FlowMatching expert predicts velocity field v(x,t). Default sampler is Euler
# (1 NFE/step). Higher-order integrators converge faster per-NFE on smooth v:
#   - heun  : RK2 trapezoidal, 2 NFE/step → N=5 ≈ Euler-10 quality at 10 NFE
#   - midpt : RK2 explicit midpoint, 2 NFE/step
#   - dpm2m : AB2 multistep (DPM-Solver++ 2M analogue for flow-matching),
#             1 NFE/step, 2nd-order accuracy → N=5/6 should beat Euler-N at
#             SAME wall-clock since NFE count is identical.
# We monkey-patch model.diffusion.sample to inject int_method since
# _run_ablation_sim's diffusion_kwargs only sets inference_step.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 90,
)
def ablation_dpm_sampler():
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch
    import functools

    # Compile expert (same winner config as diff_steps_sweep).
    print("[dpm] torch.compile(expert, mode='reduce-overhead', dynamic=False) ...")
    model.expert = torch.compile(
        model.expert, mode="reduce-overhead", dynamic=False,
    )

    def _pre_iter():
        torch.compiler.cudagraph_mark_step_begin()

    # Monkey-patch: force a specific int_method on every .sample() call.
    orig_sample = model.diffusion.sample
    current_method = {"v": "euler"}

    @functools.wraps(orig_sample)
    def _patched_sample(*args, **kwargs):
        kwargs["int_method"] = current_method["v"]
        return orig_sample(*args, **kwargs)

    model.diffusion.sample = _patched_sample

    # Configs: (sampler, N_steps). dpm2m is 1 NFE/step like euler; heun/midpt
    # are 2 NFE/step so N=5 heun ≈ 10 NFE (compare to euler@10).
    configs = [
        ("dpm2m", 10),   # same NFE as euler@10, should beat in ADE
        ("dpm2m", 6),    # push lower NFE
        ("dpm2m", 5),
        ("dpm2m", 4),
        ("heun", 5),     # 10 NFE, target euler@10 ADE
        ("heun", 4),     # 8 NFE
        ("heun", 3),     # 6 NFE
    ]
    results = []
    for sampler, steps in configs:
        current_method["v"] = sampler
        label = f"{sampler}{steps}"
        print(f"\n[dpm] === sampler={sampler} steps={steps} "
              f"(NFE≈{steps * (2 if sampler in ('heun','midpoint') else 1)}) ===")
        try:
            r = _run_ablation_sim(
                model=ctx["model"], processor=ctx["processor"],
                helper=ctx["helper"],
                load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
                avdi=ctx["avdi"], all_cams=ctx["all_cams"],
                needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
                warmup_iters=3, label=label,
                pre_iter_hook=_pre_iter, diff_steps=steps,
            )
            r["sampler"] = sampler
            r["diff_steps"] = steps
            r["optimization"] = f"compile+{sampler}_{steps}"
            results.append(r)
        except Exception as e:
            results.append({
                "sampler": sampler, "diff_steps": steps,
                "error": f"{type(e).__name__}: {e}",
                "optimization": f"compile+{sampler}_{steps}",
            })
    print("\n[dpm] SUMMARY:")
    for r in results:
        print(" ", r)
    return results


@app.local_entrypoint()
def dpm_sampler():
    r = ablation_dpm_sampler.remote()
    print("[local] dpm_sampler:", r)


# =============================================================================
# ABLATION E: "3 camera tokenizer"
# =============================================================================
# Interpretation (from Alpamayo-R1 paper, arxiv 2511.00088):
#
#   Sec. 3.2.1 "Single-Image Tokenization" establishes that AR1's default
#   tokenizer produces a token count that scales linearly with (#cameras ×
#   resolution). Sec. 6.7 "Ablation: Efficient Vision Encoding" recommends
#   reducing the sensor-token budget to speed up inference, and Sec. 6.7's
#   closing remark explicitly states: "a small number of cameras and short
#   histories will favor single-image tokenization." The paper's Table 13
#   ablates on a 4-camera setup; this ablation drops to 3 cameras while
#   keeping the same single-image tokenizer the repo already ships (the
#   triplane/Flex tokenizers from Ivanovic 2025 / Yang 2025 are not present
#   in this open-source release).
#
#   So "3 camera tokenizer" here means: feed the single-image tokenizer only
#   3 of the 4 configured physical cameras. `_run_ablation_sim(n_cams=3)`
#   slices `all_cams[:3]` — given the `live_cam_indices` comment at line
#   ~886, this keeps [cross_left=0, front_wide=1, cross_right=2] and drops
#   front_tele=6. The trajectory decoder's unused-camera token budget is
#   freed and ViT/LLM both see ~25% fewer image tokens per step.
#
# Baseline for comparison: results.csv row 6 (bf16, no compile, 4cam_4fr_10diff,
# 405.4 ms / 2.50 Hz / 1.56 m ADE). NO torch.compile applied here.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def ablation_3cam_tokenizer():
    """3-camera single-image tokenizer (paper Sec. 3.2.1 / 6.7). Bf16, no compile."""
    ctx = _setup_model_common()
    print("[3cam_tok] starting measurement (n_cams=3, n_frames=4, diff_steps=10) ...")
    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3, label="3cam_tok",
        n_cams=3,
    )
    result["optimization"] = "3cam_tokenizer"
    result["note"] = (
        "Single-image tokenizer fed 3 of 4 cameras (drops front_tele; keeps "
        "cross_left, front_wide, cross_right). 4 frames, 10 diff steps. Bf16, "
        "no torch.compile. Baseline: row 6 (405.4ms / 2.50Hz / 1.56m ADE)."
    )
    return result


@app.local_entrypoint()
def three_cam_tokenizer():
    r = ablation_3cam_tokenizer.remote()
    print("[local] 3cam_tokenizer result:", r)


# ==== STUB_F_FLOW_TOKENIZER ====
# -----------------------------------------------------------------------------
# ABLATION F: flow-tokenizer ablation
# -----------------------------------------------------------------------------
# In the Alpamayo-R1 codebase there is NO standalone class literally named
# "FlowTokenizer". What exists:
#
#   * DiscreteTrajectoryTokenizer  (src/alpamayo_r1/action_space/
#     discrete_action_space.py:24)  — bin-discretized action space, produces
#     <i0>..<iN> tokens consumed by the VLM during CoT / SFT.
#   * DeltaTrajectoryTokenizer     (src/alpamayo_r1/models/delta_tokenizer.py:21)
#     — delta-xyz bin tokenizer for history traj (hist_traj_tokenizer_cfg).
#   * FlowMatching                 (src/alpamayo_r1/diffusion/flow_matching.py:22)
#     — continuous velocity-field diffusion head on the *expert* branch
#     (model.diffusion); this is the actual trajectory generator at inference.
#
# "Flow tokenizer" in the paper's context is the flow-matching expert pathway:
# the expert consumes prompt KV-cache + noisy action tokens and produces a
# continuous trajectory by integrating v(x,t) — tokens-as-continuous-latents
# generated by a flow model, as opposed to DiscreteTrajectoryTokenizer
# classification output.
#
# The released 10B checkpoint ONLY supports flow-matching at inference:
#   - model.diffusion is a FlowMatching instance (see
#     src/alpamayo_r1/models/alpamayo_r1.py:99 — hyu.instantiate of
#     config.diffusion_cfg; baseline cfg points at FlowMatching).
#   - The sim-loop prompt ends at <|traj_future_start|> (_build_messages in
#     _run_ablation_sim), which SKIPS CoC/CoT and goes directly to the
#     flow-matching expert — so the baseline measurement is already an
#     end-to-end flow-tokenizer run.
#   - No alternate discrete-only decode path is wired up for this checkpoint
#     at inference; DiscreteTrajectoryTokenizer feeds SFT CE loss / CoT
#     tokens, not the final xyz prediction returned by
#     sample_trajectories_from_data_with_vlm_rollout.
#
# Therefore "flow tokenizer ablation" is a verification-and-characterization
# run: confirm the flow head is active, then measure it standalone (bf16, NO
# compile) at default NFE and at a minimum-viable 4-NFE setting.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def ablation_flow_tokenizer():
    """Flow-tokenizer ablation: verify flow-matching expert is active and
    measure end-to-end latency (bf16, NO compile).
    """
    from alpamayo_r1.diffusion.flow_matching import FlowMatching
    from alpamayo_r1.action_space.discrete_action_space import (
        DiscreteTrajectoryTokenizer,
    )

    ctx = _setup_model_common()
    model = ctx["model"]

    # Verify the flow-tokenizer (= flow-matching expert head) is the active
    # trajectory generator.
    diffusion_cls = type(model.diffusion).__name__
    is_flow = isinstance(model.diffusion, FlowMatching)
    has_discrete_traj_tok = isinstance(
        getattr(model, "traj_tokenizer", None), DiscreteTrajectoryTokenizer
    )
    int_method = getattr(model.diffusion, "int_method", None)
    default_n_steps = getattr(model.diffusion, "num_inference_steps", None)
    print(f"[flow_tok] diffusion class = {diffusion_cls}  is_flow={is_flow}")
    print(f"[flow_tok] int_method={int_method}  default_num_inference_steps="
          f"{default_n_steps}")
    print(f"[flow_tok] traj_tokenizer is DiscreteTrajectoryTokenizer: "
          f"{has_discrete_traj_tok}  (unused at inference; flow head decides)")

    if not is_flow:
        msg = (
            f"Expected model.diffusion to be FlowMatching but got "
            f"{diffusion_cls}; released 10B checkpoint should ship with "
            f"flow-matching — aborting ablation."
        )
        print(f"[flow_tok] ABORT: {msg}")
        return {
            "optimization": "flow_tokenizer",
            "error": msg,
            "diffusion_class": diffusion_cls,
        }

    # Run A: flow tokenizer at default NFE (matches baseline row 6 scope).
    print("[flow_tok] === run A: flow tokenizer @ default NFE (bf16, no compile) ===")
    r_default = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3, label="flow_tok_default",
    )

    # Run B: minimum-viable flow tokenizer NFE=4 (cheapest flow-tokenizer).
    print("[flow_tok] === run B: flow tokenizer @ 4 NFE (min-viable) ===")
    r_min = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3, label="flow_tok_nfe4", diff_steps=4,
    )

    note = (
        "Flow tokenizer = the flow-matching expert head (model.diffusion, "
        "FlowMatching class); it is the ONLY trajectory generator wired up "
        "at inference on the released 10B checkpoint. "
        "DiscreteTrajectoryTokenizer exists but only feeds SFT CE losses / "
        "CoT tokens, not the final xyz prediction. This ablation confirms "
        "the flow head is active and measures two NFE points (default and "
        "min-viable 4-step) in bf16 with NO torch.compile."
    )
    return {
        "optimization": "flow_tokenizer",
        "diffusion_class": diffusion_cls,
        "int_method": int_method,
        "default_num_inference_steps": default_n_steps,
        "traj_tokenizer_is_discrete": has_discrete_traj_tok,
        "run_default_nfe": r_default,
        "run_min_nfe4": r_min,
        "note": note,
    }


@app.local_entrypoint()
def flow_tokenizer():
    r = ablation_flow_tokenizer.remote()
    print("[local] flow_tokenizer:", r)


# ==== STUB_G_SHORT_HORIZON ====
# Short-horizon ablation: cut future trajectory horizon 6.4s -> 2.0s.
#
# Mechanism (run-time, cuts compute):
#   - model.action_space.n_waypoints controls output dims via
#     UnicycleAccelCurvatureActionSpace.get_action_space_dims() -> (n_waypoints, 2)
#     (src/alpamayo_r1/action_space/unicycle_accel_curvature.py:98)
#   - AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout reads this at
#     call time:
#       n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
#     (src/alpamayo_r1/models/alpamayo_r1.py:235). This drives:
#       * noise tensor shape in diffusion.sample (via x_dims)
#       * expert forward token count (future_token_embeds rows)
#       * position_ids / attention_mask shape
#   - diffusion.x_dims was snapshotted at model init from action_space dims,
#     so we override it too.
#   - With dt=0.1 and n_waypoints=20, horizon = 2.0s (vs baseline 64 -> 6.4s).
#     Expert runs on 20 tokens instead of 64 -> ~3.2x less expert work.
#   - GT ground-truth (data["ego_future_xyz"]) is still 64 waypoints; ADE is
#     naturally computed over min(pred, gt) inside _run_ablation_sim, so it
#     becomes a 2.0s ADE window automatically.

N_WAYPOINTS_SHORT = 20  # 2.0s @ dt=0.1


@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 45,
)
def ablation_short_horizon():
    """Short-horizon: predict 2.0s (20 waypoints) instead of 6.4s (64).

    bf16, no compile. Shrinks expert forward token count 64 -> 20, cutting
    per-step matmul size linearly in the token dimension. Hypothesis: if
    expert compute dominates the 405ms baseline, expect meaningful speedup.
    """
    ctx = _setup_model_common()
    model = ctx["model"]

    old_n = model.action_space.n_waypoints
    model.action_space.n_waypoints = N_WAYPOINTS_SHORT
    # diffusion.x_dims was cached at init; re-sync so noise tensor is short.
    new_dims = list(model.action_space.get_action_space_dims())
    old_x_dims = list(model.diffusion.x_dims)
    model.diffusion.x_dims = new_dims

    horizon_s = N_WAYPOINTS_SHORT * model.action_space.dt
    print(f"[horizon_2s] n_waypoints {old_n} -> {N_WAYPOINTS_SHORT} "
          f"(horizon {horizon_s:.1f}s); diffusion.x_dims {old_x_dims} -> "
          f"{new_dims}")

    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=3, label="horizon_2s",
    )
    result["optimization"] = "horizon_2s"
    result["note"] = (
        f"Run-time horizon cut: set model.action_space.n_waypoints={N_WAYPOINTS_SHORT} "
        f"(was {old_n}) and model.diffusion.x_dims={new_dims}. Expert forward "
        f"runs on {N_WAYPOINTS_SHORT} future tokens instead of 64 -> ~3.2x "
        f"fewer expert matmul rows per diffusion step. bf16, no compile. "
        f"ADE is measured over the 2.0s window (GT is truncated to pred length "
        f"inside _run_ablation_sim via min(T))."
    )
    return result


@app.local_entrypoint()
def short_horizon():
    r = ablation_short_horizon.remote()
    print("[local] short_horizon result:", r)


# -----------------------------------------------------------------------------
# ABLATION H: STACK — compile(expert) + euler-3 + horizon=2.0s
# -----------------------------------------------------------------------------
# Winners from each sweep:
#   * compile(expert, reduce-overhead)         : baseline 300ms -> lower
#   * diff_steps=3 (euler)                     : best ADE (1.30m) + 227.6ms
#   * horizon_2s (n_waypoints 64 -> 20)        : ADE 0.68m standalone
# Hypothesis: horizon cut shrinks expert token count 64->20 (3.2x fewer rows
# in per-step matmul). compile+euler-3 is 227.6ms with expert ~100ms of that.
# Cutting expert to ~30ms would push iter to ~160ms = 6.2 Hz.
#
# Run H1 = compile + euler-3 + horizon-2s  (primary 6-Hz candidate)
# Run H2 = compile + euler-4 + horizon-2s  (safety margin on ADE)

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def ablation_stack_compile_euler_horizon():
    ctx = _setup_model_common()
    model = ctx["model"]

    import torch

    old_n = model.action_space.n_waypoints
    model.action_space.n_waypoints = N_WAYPOINTS_SHORT
    new_dims = list(model.action_space.get_action_space_dims())
    old_x_dims = list(model.diffusion.x_dims)
    model.diffusion.x_dims = new_dims
    horizon_s = N_WAYPOINTS_SHORT * model.action_space.dt
    print(f"[stack] horizon: n_waypoints {old_n} -> {N_WAYPOINTS_SHORT} "
          f"({horizon_s:.1f}s); x_dims {old_x_dims} -> {new_dims}")

    print("[stack] torch.compile(expert, mode='reduce-overhead', dynamic=False)")
    model.expert = torch.compile(
        model.expert, mode="reduce-overhead", dynamic=False,
    )

    def _pre_iter():
        torch.compiler.cudagraph_mark_step_begin()

    results = []
    for ds in [3, 4]:
        print(f"\n[stack] === compile+euler{ds}+horizon2s ===")
        try:
            r = _run_ablation_sim(
                model=ctx["model"], processor=ctx["processor"],
                helper=ctx["helper"],
                load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
                avdi=ctx["avdi"], all_cams=ctx["all_cams"],
                needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
                warmup_iters=3, label=f"stack_e{ds}_h2",
                pre_iter_hook=_pre_iter, diff_steps=ds,
            )
            r["optimization"] = f"compile+euler{ds}+horizon2s"
            r["diff_steps"] = ds
            r["horizon_s"] = horizon_s
            results.append(r)
        except Exception as e:
            results.append({
                "optimization": f"compile+euler{ds}+horizon2s",
                "error": f"{type(e).__name__}: {e}",
            })

    print("\n[stack] SUMMARY:")
    for r in results:
        print(" ", r)
    return results


@app.local_entrypoint()
def stack_compile_euler_horizon():
    r = ablation_stack_compile_euler_horizon.remote()
    print("[local] stack:", r)


# -----------------------------------------------------------------------------
# ABLATION: ParoQuant INT4 (Marlin W4A8) on VLM text decoder
# -----------------------------------------------------------------------------
#
# FlashDriveVLA/Alpamayo-R1-10B-finetuned-PARO ships 252 VLM text decoder
# Linears pre-quantized to W4A8 Marlin format. Each layer stores:
#   rotation.theta           — learned pairwise rotation angles (krot, in_f/2)
#   rotation.pairs           — rotation channel indices (krot, in_f)
#   rotation.channel_scales  — per-channel pre-rotation scale (1, in_f)
#   qlinear.qweight          — Marlin-packed INT4 weights
#   qlinear.qzeros           — zero-points
#   qlinear.scales           — per-group fp16 scales
#   qlinear.g_idx            — Marlin act-order permutation
#   qlinear.g_idx_sort_indices
#   qlinear.workspace        — Marlin scratchpad
#   qlinear.input_global_scale — optional act scale (unused on FP16 act)
#
# Integration:
# 1. Load base Alpamayo-R1 bf16 model from nvidia/Alpamayo-R1-10B as usual.
# 2. Walk the VLM module tree; match quantized FQNs from the checkpoint to
#    actual nn.Linear modules on our instance.
# 3. Swap each matched Linear with a custom ParoMarlinLinear module that
#    wraps paroquant's rotation kernel + vllm's apply_awq_marlin_linear.
# 4. Load the PARO safetensor shards; route keys to the new module buffers.
# 5. Run the standard ablation sim.
#
# bf16 activation handling: vllm's Marlin kernel accepts fp16; we wrap in a
# bf16→fp16→bf16 adapter since the rest of Alpamayo runs bf16 autocast.
image_paro_bench = image_paro

@app.function(
    gpu="H100",
    image=image_paro_bench,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def ablation_paroquant_vlm(
    paro_model: str = "FlashDriveVLA/Alpamayo-R1-10B-finetuned-PARO",
    skip_cot: bool = True,
    compile_expert: bool = False,
    diff_steps: int = 10,
    n_cams: int = 4,
    n_frames: int = 4,
    base_repo: str = "nvidia/Alpamayo-R1-10B",
):
    """Swap VLM Linears for Marlin-W4A8 linears + paroquant rotation and
    run the ablation sim.

    Loads the FlashDriveVLA PARO checkpoint (W4A8 Marlin format) and grafts
    its quantized Linears onto the nvidia/Alpamayo-R1-10B bf16 base. vLLM
    kernels (apply_awq_marlin_linear) handle the INT4 matmul; paroquant's
    torch.ops.rotation.rotate handles the per-channel pairwise rotation.
    """
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["PAI_CACHE"] = "/cache/pai"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HUGGING_FACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import time
    import glob
    import json
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # Import paroquant kernel to register torch.ops.rotation.rotate.
    try:
        import paroquant.kernels.cuda  # noqa: F401
    except Exception as e:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": f"paroquant kernel import failed: {type(e).__name__}: {e}",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Import vLLM's Marlin kernel binding. apply_awq_marlin_linear consumes
    # already-Marlin-packed qweight/qzeros + scales + g_idx + workspace.
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
            marlin_make_workspace,
            marlin_make_empty_g_idx,
            marlin_permute_scales,
            awq_to_marlin_zero_points,
        )
        from vllm import _custom_ops as vllm_ops
        from vllm.scalar_type import scalar_types
        MARLIN_QUANT_TYPE = scalar_types.uint4
    except Exception as e:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": f"vllm marlin import failed: {type(e).__name__}: {e}",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper
    import physical_ai_av

    # Step 1: download PARO checkpoint + scan keys.
    print(f"[paro] downloading {paro_model} ...")
    t0 = time.perf_counter()
    local_dir = snapshot_download(paro_model)
    print(f"[paro] downloaded in {time.perf_counter()-t0:.1f}s → {local_dir}")

    index_file = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            all_keys = list(json.load(f).get("weight_map", {}).keys())
    else:
        all_keys = []
        for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
            with safe_open(sf, framework="pt") as st:
                all_keys.extend(st.keys())

    # Identify quantized Linear FQNs. Smoke test confirmed keys end in
    # e.g. "vlm.model.language_model.layers.0.mlp.down_proj.rotate_linear.qlinear.qweight"
    # → real Linear FQN is everything before ".rotate_linear".
    qlinear_qweight_keys = [k for k in all_keys if k.endswith(".rotate_linear.qlinear.qweight")]
    quantized_fqns = {k.removesuffix(".rotate_linear.qlinear.qweight") for k in qlinear_qweight_keys}
    print(f"[paro] checkpoint declares {len(quantized_fqns)} quantized Linears")
    if len(quantized_fqns) == 0:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": "no .rotate_linear.qlinear.qweight keys found in PARO checkpoint",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Step 2: load base bf16 Alpamayo-R1.
    base_cfg = AlpamayoR1Config.from_pretrained(base_repo)
    base_cfg.attn_implementation = "flash_attention_2"
    print(f"[paro] loading base bf16 model from {base_repo} ...")
    t0 = time.perf_counter()
    model = AlpamayoR1.from_pretrained(
        base_repo, config=base_cfg, dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    model.expert.config._attn_implementation = "sdpa"
    for _m in model.expert.modules():
        _c = getattr(_m, "config", None)
        if _c is not None and hasattr(_c, "_attn_implementation_internal"):
            _c._attn_implementation_internal = "sdpa"
    print(f"[paro] base model loaded in {time.perf_counter()-t0:.1f}s")

    # Step 3: match quantized FQNs to model Linears.
    all_named = dict(model.named_modules())
    quant_map = {}  # name_in_model -> ckpt_prefix
    unmatched_fqns = []
    for ckpt_fqn in quantized_fqns:
        # The checkpoint key prefix is "vlm.model.language_model.layers.N...".
        # On our bf16 model from nvidia/Alpamayo-R1-10B, the equivalent module
        # path is the same (AlpamayoR1 has self.vlm which is Qwen3VLForConditionalGeneration;
        # that has .model (Qwen3VLModel) containing .language_model).
        if ckpt_fqn in all_named and isinstance(all_named[ckpt_fqn], nn.Linear):
            quant_map[ckpt_fqn] = ckpt_fqn
        else:
            unmatched_fqns.append(ckpt_fqn)
    print(f"[paro] matched {len(quant_map)}/{len(quantized_fqns)} Linears in model tree")
    if unmatched_fqns:
        print(f"[paro] UNMATCHED sample: {unmatched_fqns[:3]}")
        # Check what module-tree names are similar
        candidates = [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "language_model" in n
        ]
        print(f"[paro] model language_model Linears sample: {candidates[:3]}")
    if len(quant_map) < 252:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": f"only matched {len(quant_map)}/252 FQNs. "
                     f"Unmatched sample: {unmatched_fqns[:3]}",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Step 4: define the Marlin-W4A8 + paroquant-rotation Linear module and swap.
    class ParoMarlinLinear(nn.Module):
        """Marlin INT4 weight + paroquant pairwise rotation, fp16 activations.

        Buffers mirror the checkpoint layout so state_dict load works:
          rotation.theta, rotation.pairs, rotation.channel_scales
          qlinear.qweight, qlinear.qzeros, qlinear.scales,
          qlinear.g_idx, qlinear.g_idx_sort_indices, qlinear.workspace,
          qlinear.input_global_scale
        """

        def __init__(self, in_features, out_features, has_bias=False,
                     group_size=128, bits=4, krot=8):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.bits = bits
            self.krot = krot
            pack = 32 // bits
            n_groups = in_features // group_size

            # Rotation (under `rotate_linear.rotation.*` on the checkpoint).
            # We host these flat under self.rotation.* for the state_dict to
            # match `<fqn>.rotate_linear.rotation.theta` → we'll use a dict-
            # loading path below; buffer placement here is just runtime.
            self.register_buffer("rot_theta",
                torch.zeros(krot, in_features // 2, dtype=torch.float16))
            self.register_buffer("rot_pairs",
                torch.zeros(krot, in_features, dtype=torch.int16))
            self.register_buffer("rot_channel_scales",
                torch.ones(1, in_features, dtype=torch.float16))

            # Marlin quantized params (pre-packed by the conversion pipeline).
            # Shapes below are conservative placeholders — overridden by actual
            # checkpoint tensors in load time.
            self.register_buffer("qweight",
                torch.zeros(in_features // pack, out_features * pack, dtype=torch.int32))
            self.register_buffer("qzeros",
                torch.zeros(n_groups, out_features // pack, dtype=torch.int32))
            self.register_buffer("scales",
                torch.zeros(n_groups, out_features, dtype=torch.float16))
            self.register_buffer("g_idx",
                torch.zeros(in_features, dtype=torch.int32))
            self.register_buffer("g_idx_sort_indices",
                torch.zeros(in_features, dtype=torch.int32))
            self.register_buffer("workspace",
                torch.zeros(1024, dtype=torch.int32))
            self.register_buffer("input_global_scale",
                torch.ones(1, dtype=torch.float32))

            if has_bias:
                self.register_buffer("bias",
                    torch.zeros(out_features, dtype=torch.float16))
            else:
                self.bias = None

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Rotation kernel supports bf16/fp16/fp32 natively; Marlin kernel
            # on SM90+ supports bf16. Scales/qweight are stored as fp16/int32
            # but Marlin accepts them regardless. We cast rotation buffers
            # lazily to match x's dtype (done by the rotation CUDA kernel
            # internally; see rotate_launcher's theta_cast/scales_cast).
            x_rot = torch.ops.rotation.rotate(
                x, self.rot_pairs, self.rot_theta, self.rot_channel_scales,
            )
            # Marlin expects scales/zp in same dtype family as input.
            # When x is bf16 but scales are fp16, vllm's marlin will fail.
            # Conservative: cast to fp16 before Marlin, cast back.
            in_dtype = x.dtype
            x16 = x_rot.to(torch.float16) if x_rot.dtype != torch.float16 else x_rot
            out = apply_awq_marlin_linear(
                input=x16,
                weight=self.qweight,
                weight_scale=self.scales,
                weight_zp=self.qzeros,
                g_idx=self.g_idx,
                g_idx_sort_indices=self.g_idx_sort_indices,
                workspace=self.workspace,
                quant_type=MARLIN_QUANT_TYPE,
                output_size_per_partition=self.out_features,
                input_size_per_partition=self.in_features,
                bias=self.bias,
            )
            return out.to(in_dtype) if in_dtype != torch.float16 else out

    # Read group_size/krot from the w4a8 config (defaults are safe).
    w4a8_path = os.path.join(local_dir, "w4a8_config.json")
    w4a8_cfg = {}
    if os.path.exists(w4a8_path):
        with open(w4a8_path) as f:
            w4a8_cfg = json.load(f)
    group_size = int(w4a8_cfg.get("group_size", 128))
    krot = int(w4a8_cfg.get("krot", 8))
    bits = int(w4a8_cfg.get("bits", 4))

    bias_count = 0
    for name_in_model in quant_map:
        old_linear = all_named[name_in_model]
        has_bias = old_linear.bias is not None
        if has_bias:
            bias_count += 1
        new_mod = ParoMarlinLinear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            has_bias=has_bias,
            group_size=group_size, bits=bits, krot=krot,
        ).to("cuda")
        parent_name, attr = (
            name_in_model.rsplit(".", 1) if "." in name_in_model else ("", name_in_model)
        )
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, new_mod)
    print(f"[paro] swapped {len(quant_map)} Linears → ParoMarlinLinear "
          f"({bias_count} with bias); gs={group_size} krot={krot}")

    # Step 5: load PARO shards and route tensors into the swapped modules.
    # Checkpoint key → module buffer mapping:
    #   <fqn>.rotate_linear.rotation.theta           → <fqn>.rot_theta
    #   <fqn>.rotate_linear.rotation.pairs           → <fqn>.rot_pairs
    #   <fqn>.rotate_linear.rotation.channel_scales  → <fqn>.rot_channel_scales
    #   <fqn>.rotate_linear.qlinear.qweight          → <fqn>.qweight
    #   <fqn>.rotate_linear.qlinear.qzeros           → <fqn>.qzeros
    #   <fqn>.rotate_linear.qlinear.scales           → <fqn>.scales
    #   <fqn>.rotate_linear.qlinear.g_idx            → <fqn>.g_idx
    #   <fqn>.rotate_linear.qlinear.g_idx_sort_indices → <fqn>.g_idx_sort_indices
    #   <fqn>.rotate_linear.qlinear.workspace        → <fqn>.workspace
    #   <fqn>.rotate_linear.qlinear.input_global_scale → <fqn>.input_global_scale
    SUFFIX_MAP = {
        "rotate_linear.rotation.theta": "rot_theta",
        "rotate_linear.rotation.pairs": "rot_pairs",
        "rotate_linear.rotation.channel_scales": "rot_channel_scales",
        "rotate_linear.qlinear.qweight": "qweight",
        "rotate_linear.qlinear.qzeros": "qzeros",
        "rotate_linear.qlinear.scales": "scales",
        "rotate_linear.qlinear.g_idx": "g_idx",
        "rotate_linear.qlinear.g_idx_sort_indices": "g_idx_sort_indices",
        "rotate_linear.qlinear.workspace": "workspace",
        "rotate_linear.qlinear.input_global_scale": "input_global_scale",
    }
    expected_per_layer = len(SUFFIX_MAP)

    print("[paro] loading PARO tensors into swapped layers ...")
    t0 = time.perf_counter()
    loaded = 0
    shape_mismatch = 0
    for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        with safe_open(sf, framework="pt", device="cuda") as st:
            for key in st.keys():
                # Find which quantized FQN this key belongs to and which buffer.
                for ckpt_suffix, buf_name in SUFFIX_MAP.items():
                    if key.endswith("." + ckpt_suffix):
                        fqn = key.removesuffix("." + ckpt_suffix)
                        if fqn not in quant_map:
                            break
                        target_mod = all_named[fqn] if fqn in all_named else None
                        # After swap, named_modules sees new module: re-resolve.
                        parent_name, attr = (
                            fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                        )
                        parent = model.get_submodule(parent_name) if parent_name else model
                        target_mod = getattr(parent, attr)
                        if not hasattr(target_mod, buf_name):
                            break
                        tgt_buf = getattr(target_mod, buf_name)
                        src = st.get_tensor(key)
                        if tgt_buf is None or tgt_buf.shape != src.shape:
                            # Replace the buffer with the correct shape.
                            target_mod.register_buffer(buf_name, src.to(tgt_buf.device) if tgt_buf is not None else src)
                            shape_mismatch += 1
                        else:
                            tgt_buf.copy_(src.to(tgt_buf.device))
                        loaded += 1
                        break
    print(f"[paro] loaded {loaded} tensors "
          f"({shape_mismatch} shape-replaced) in {time.perf_counter()-t0:.1f}s "
          f"(expected ~{len(quant_map) * expected_per_layer})")

    # Regenerate GPU-specific Marlin workspace/g_idx/g_idx_sort_indices for each
    # layer + run AWQ→Marlin conversion on scales/qzeros IF they look raw-AWQ.
    #
    # The checkpoint's qweight is already in Marlin-repacked shape
    # (in/16, out*2), confirming the conversion happened at save time. But
    # smoke tests show stacked forward NaNs — likely root cause is that
    # scales/qzeros values were NOT permuted at save time (same shape as AWQ
    # raw, which coincidentally matches Marlin shape). We run the permutations
    # here to match what vllm's AWQMarlinLinearMethod.process_weights_after_loading
    # does.
    device = torch.device("cuda")
    bits = 4
    regenerated = 0
    permuted_scales = 0
    permuted_qzeros = 0
    for fqn in quant_map:
        parent_name, attr = (
            fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
        )
        parent = model.get_submodule(parent_name) if parent_name else model
        mod = getattr(parent, attr)
        k = mod.in_features
        n = mod.out_features

        mod.register_buffer("workspace",
            marlin_make_workspace(n, device))
        mod.register_buffer("g_idx",
            marlin_make_empty_g_idx(device).data)
        mod.register_buffer("g_idx_sort_indices",
            marlin_make_empty_g_idx(device).data)
        regenerated += 1

        # Permute scales from AWQ → Marlin layout.
        try:
            permuted = marlin_permute_scales(
                mod.scales.data, size_k=k, size_n=n,
                group_size=group_size,
            )
            mod.scales.data.copy_(permuted)
            permuted_scales += 1
        except Exception as e:
            if permuted_scales == 0:
                print(f"[paro] marlin_permute_scales failed on {fqn}: {e}")

        # Permute qzeros from AWQ → Marlin layout. size_k for zero_points is
        # num_groups (not input_size_per_partition — see vllm AWQMarlin code).
        try:
            num_groups = k // group_size
            permuted_zp = awq_to_marlin_zero_points(
                mod.qzeros.data, size_k=num_groups, size_n=n,
                num_bits=bits,
            )
            if permuted_zp.shape == mod.qzeros.data.shape:
                mod.qzeros.data.copy_(permuted_zp)
            else:
                mod.register_buffer("qzeros", permuted_zp)
            permuted_qzeros += 1
        except Exception as e:
            if permuted_qzeros == 0:
                print(f"[paro] awq_to_marlin_zero_points failed on {fqn}: {e}")

    print(f"[paro] regenerated workspace/g_idx for {regenerated} layers; "
          f"permuted scales on {permuted_scales}, qzeros on {permuted_qzeros}")

    expected_total = len(quant_map) * expected_per_layer
    if loaded < expected_total * 0.9:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": f"loaded only {loaded}/{expected_total} tensors "
                     f"(<90% of expected)",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Attach NaN/Inf detection hooks to every quantized layer to find the
    # first offending layer during forward.
    first_bad = {"name": None, "stats": None}
    hook_handles = []

    def _make_hook(layer_name):
        def _hook(mod, inp, out):
            if first_bad["name"] is not None:
                return
            t = out if isinstance(out, torch.Tensor) else out[0]
            if torch.isnan(t).any() or torch.isinf(t).any():
                first_bad["name"] = layer_name
                in_t = inp[0] if isinstance(inp, tuple) else inp
                first_bad["stats"] = {
                    "in_nan": torch.isnan(in_t).any().item(),
                    "in_max": in_t.float().abs().max().item(),
                    "out_nan": torch.isnan(t).any().item(),
                    "out_inf": torch.isinf(t).any().item(),
                }
        return _hook

    for fqn in quant_map:
        parent_name, attr = (
            fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
        )
        parent = model.get_submodule(parent_name) if parent_name else model
        mod = getattr(parent, attr)
        h = mod.register_forward_hook(_make_hook(fqn))
        hook_handles.append(h)

    # Full-model smoke: run the VLM language_model on a dummy token sequence
    # to verify the quantized forward stack doesn't produce NaN. This
    # isolates the quantization issue from the rest of the pipeline.
    lang = None
    for attr_path in ("vlm.model.language_model", "vlm.language_model"):
        try:
            lang = model.get_submodule(attr_path)
            print(f"[paro] found language_model at {attr_path}")
            break
        except AttributeError:
            continue
    if lang is None:
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": "could not find vlm.language_model submodule",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    dummy_ids = torch.randint(0, 150000, (1, 32), device="cuda", dtype=torch.long)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = lang(dummy_ids)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else (
            out[0] if isinstance(out, tuple) else out
        )
        print(f"[paro] language_model output shape={tuple(h.shape)} "
              f"dtype={h.dtype} "
              f"has_nan={torch.isnan(h).any().item()} "
              f"has_inf={torch.isinf(h).any().item()} "
              f"mean={h.float().mean().item():.4f} "
              f"std={h.float().std().item():.4f} "
              f"abs_max={h.float().abs().max().item():.4f}")
        if torch.isnan(h).any() or torch.isinf(h).any():
            return {
                "optimization": "paroquant_vlm_marlin_w4a8",
                "error": f"language_model forward produced NaN/Inf; "
                         f"first bad layer: {first_bad['name']} stats={first_bad['stats']}",
                "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
            }
        # Also run full lm_head to check final logits
        if hasattr(model.vlm, "lm_head"):
            logits = model.vlm.lm_head(h)
            print(f"[paro] lm_head logits: "
                  f"has_nan={torch.isnan(logits).any().item()} "
                  f"has_inf={torch.isinf(logits).any().item()} "
                  f"abs_max={logits.float().abs().max().item():.4f}")
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return {
                    "optimization": "paroquant_vlm_marlin_w4a8",
                    "error": "lm_head logits have NaN/Inf",
                    "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
                }
    except Exception as e:
        import traceback
        return {
            "optimization": "paroquant_vlm_marlin_w4a8",
            "error": f"language_model smoke crashed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Remove NaN-detection hooks after smoke passed.
    for h in hook_handles:
        h.remove()
    hook_handles.clear()

    if compile_expert:
        print("[paro] compiling expert (reduce-overhead) ...")
        model.expert = torch.compile(
            model.expert, mode="reduce-overhead", dynamic=False,
        )

    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    all_cams = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]

    pre_iter_hook = None
    if compile_expert:
        def pre_iter_hook():
            torch.compiler.cudagraph_mark_step_begin()

    print("[paro] starting measurement ...")
    result = _run_ablation_sim(
        model=model, processor=processor, helper=helper,
        load_physical_aiavdataset=load_physical_aiavdataset,
        avdi=avdi, all_cams=all_cams, needs_cam_indices_in_msg=False,
        warmup_iters=3, label="paroquant_vlm",
        pre_iter_hook=pre_iter_hook,
        n_cams=n_cams, n_frames=n_frames, diff_steps=diff_steps,
    )
    result["optimization"] = (
        f"paroquant_vlm_marlin_w4a8{'_compile' if compile_expert else ''}"
    )
    result["note"] = (
        f"FlashDriveVLA/Alpamayo-R1-10B-finetuned-PARO: swapped "
        f"{len(quant_map)}/252 VLM text-decoder Linears to Marlin W4A8 + "
        f"paroquant pairwise rotation (group_size={group_size} krot={krot} "
        f"bits={bits}). Loaded {loaded} tensors. vLLM apply_awq_marlin_linear "
        f"for INT4 matmul, torch.ops.rotation.rotate for learned rotation, "
        f"bf16↔fp16 activation adapter. base_repo={base_repo} on H100. "
        f"skip_cot={skip_cot} compile_expert={compile_expert}."
    )
    return result


@app.local_entrypoint()
def paroquant_vlm(
    compile_expert: bool = False,
    diff_steps: int = 10,
    n_cams: int = 4,
    n_frames: int = 4,
):
    r = ablation_paroquant_vlm.remote(
        compile_expert=compile_expert,
        diff_steps=diff_steps,
        n_cams=n_cams,
        n_frames=n_frames,
    )
    print("[local] PARO result:", r)


# Smoke test: verify paroquant imports, torch.ops.rotation.rotate registers,
# PARO checkpoint downloads and we can identify the 252 quantized layer FQNs.
# Does NOT run inference. ~2 min on H100 (mostly checkpoint download).
@app.function(
    gpu="H100",
    image=image_paro_bench,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 10,
)
def smoke_paroquant(
    paro_model: str = "FlashDriveVLA/Alpamayo-R1-10B-finetuned-PARO",
):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import torch
    info = {"cuda_available": torch.cuda.is_available()}

    try:
        import paroquant
        info["paroquant_version"] = getattr(paroquant, "__version__", "?")
    except Exception as e:
        info["paroquant_import_error"] = f"{type(e).__name__}: {e}"
        return info

    try:
        import paroquant.kernels.cuda  # noqa
        info["rotation_op_registered"] = hasattr(torch.ops, "rotation") and hasattr(
            torch.ops.rotation, "rotate"
        )
    except Exception as e:
        info["kernel_import_error"] = f"{type(e).__name__}: {e}"

    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )
        from vllm.scalar_type import scalar_types
        info["vllm_marlin_imported"] = True
        info["vllm_scalar_types_uint4"] = str(scalar_types.uint4)
    except Exception as e:
        info["vllm_marlin_import_error"] = f"{type(e).__name__}: {e}"

    # Smoke the rotation kernel
    try:
        x = torch.randn(1, 8, 128, dtype=torch.float16, device="cuda")
        pairs = torch.zeros(8, 128, dtype=torch.int16, device="cuda")
        theta = torch.zeros(8, 64, dtype=torch.float16, device="cuda")
        scales = torch.ones(1, 128, dtype=torch.float16, device="cuda")
        y = torch.ops.rotation.rotate(x, pairs, theta, scales)
        info["rotation_kernel_shape"] = list(y.shape)
    except Exception as e:
        info["rotation_kernel_error"] = f"{type(e).__name__}: {e}"

    # Download PARO checkpoint + inspect index
    try:
        import json, glob
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import time
        t0 = time.perf_counter()
        local_dir = snapshot_download(paro_model)
        info["download_s"] = round(time.perf_counter() - t0, 1)
        info["local_dir"] = local_dir

        index_file = os.path.join(local_dir, "model.safetensors.index.json")
        all_keys = []
        if os.path.exists(index_file):
            with open(index_file) as f:
                all_keys = list(json.load(f).get("weight_map", {}).keys())
        else:
            for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
                with safe_open(sf, framework="pt") as st:
                    all_keys.extend(st.keys())
        info["total_keys"] = len(all_keys)
        info["n_qweight_keys"] = sum(1 for k in all_keys if k.endswith(".qweight"))
        info["n_theta_keys"] = sum(1 for k in all_keys if k.endswith(".theta"))
        # Show a representative FQN + tensor shapes for one layer.
        qw_keys = [k for k in all_keys if k.endswith(".qweight")]
        info["sample_qweight_fqns"] = qw_keys[:3]
        if qw_keys:
            base = qw_keys[0].removesuffix(".qlinear.qweight")
            probe_shapes = {}
            probe_dtypes = {}
            for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
                with safe_open(sf, framework="pt") as st:
                    for k in st.keys():
                        if k.startswith(base + ".") and k.removeprefix(base + ".") in {
                            "rotation.theta", "rotation.pairs", "rotation.channel_scales",
                            "qlinear.qweight", "qlinear.qzeros", "qlinear.scales",
                            "qlinear.g_idx", "qlinear.g_idx_sort_indices",
                            "qlinear.workspace", "qlinear.input_global_scale",
                        }:
                            t = st.get_tensor(k)
                            probe_shapes[k.removeprefix(base + ".")] = list(t.shape)
                            probe_dtypes[k.removeprefix(base + ".")] = str(t.dtype)
            info["probe_shapes"] = probe_shapes
            info["probe_dtypes"] = probe_dtypes
        # Enumerate the suffixes under one representative prefix so we know
        # the module structure used in the checkpoint.
        if qw_keys:
            prefix = qw_keys[0].removesuffix(".qlinear.qweight").removesuffix(".qweight")
            # Everything under this prefix
            suffixes = sorted({
                k[len(prefix)+1:] for k in all_keys if k.startswith(prefix + ".")
            })
            info["sample_prefix"] = prefix
            info["suffixes_under_prefix"] = suffixes[:20]
        # Count every distinct suffix type (last 2 segments) in the checkpoint
        from collections import Counter
        suffix_counts = Counter()
        for k in all_keys:
            parts = k.rsplit(".", 2)
            if len(parts) == 3:
                suffix_counts[parts[1] + "." + parts[2]] += 1
            elif len(parts) == 2:
                suffix_counts[parts[1]] += 1
        info["suffix_counts_top10"] = dict(suffix_counts.most_common(10))
        # w4a8_config.json
        w4a8_path = os.path.join(local_dir, "w4a8_config.json")
        if os.path.exists(w4a8_path):
            with open(w4a8_path) as f:
                info["w4a8_config"] = json.load(f)
    except Exception as e:
        info["download_error"] = f"{type(e).__name__}: {e}"

    return info


@app.local_entrypoint()
def smoke_paro():
    r = smoke_paroquant.remote()
    print("[local] PARO smoke:", r)


# -----------------------------------------------------------------------------
# ABLATION: DFlash speculative decoding for CoC rollout
# -----------------------------------------------------------------------------
#
# Strategy: load the FlashDriveVLA/Alpamayo-R1-10B-DFlash draft (0.5B params)
# alongside the standard Alpamayo-R1 target. During the VLM autoregressive
# decode phase (CoC text rollout between <|cot_start|> and <|traj_future_start|>),
# route generation through draft.spec_generate(target=target).
#
# Integration challenge: DFlash's dflash_generate() starts with a prefill step
# that calls `target(input_ids, ...)` assuming AutoModelForCausalLM. Alpamayo's
# VLM is Qwen3VLForConditionalGeneration (multimodal prefill needs pixel_values,
# image_grid_thw, etc.). We solve this by:
#
# 1. Running the VLM prefill OURSELVES (with image inputs) via vlm.forward
#    to get the image-conditioned KV cache.
# 2. Passing that KV cache and last-token embedding into a custom
#    dflash_generate that skips the prefill step.
# 3. Sampling the CoC text tokens via spec decoding until <|traj_future_start|>.
#
# Caveat: this measures the COC ROLLOUT speedup vs eager. Only meaningful
# when CoC is enabled (skip_cot=False). We include a fresh CoC-on baseline
# measurement in the same run for apples-to-apples comparison.

@app.function(
    gpu="H100",
    image=image_dflash,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 40,
)
def ablation_dflash_coc(
    draft_model: str = "FlashDriveVLA/Alpamayo-R1-10B-DFlash",
    cot_max_tokens: int = 256,
    block_size: int = 16,
    temperature: float = 0.6,
    diff_steps: int = 10,
    n_cams: int = 4,
    n_frames: int = 4,
    run_baseline: bool = True,
):
    """DFlash speculative decoding for CoC text rollout.

    Measures both:
      (1) CoC-on eager baseline (no DFlash), for reference
      (2) CoC-on with DFlash draft.spec_generate

    Key difference from other ablations: CoC is ENABLED (skip_cot=False).
    The VLM decodes up to `cot_max_tokens` before the diffusion expert runs.
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
    import torch

    try:
        from dflash.model import DFlashDraftModel, dflash_generate
    except Exception as e:
        return {
            "optimization": "dflash_coc",
            "error": f"dflash import failed: {type(e).__name__}: {e}",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    ctx = _setup_model_common()
    model = ctx["model"]

    # Load DFlash draft model (Qwen3-style, 0.5B). This is a bare Qwen3
    # decoder variant so we use AutoModel to avoid CausalLM head expectations.
    print(f"[dflash] loading draft {draft_model} ...")
    from transformers import AutoModel
    t0 = time.perf_counter()
    try:
        draft = AutoModel.from_pretrained(
            draft_model,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to("cuda").eval()
    except Exception as e:
        # fallback: try loading as DFlashDraftModel directly via safetensors
        return {
            "optimization": "dflash_coc",
            "error": f"draft load failed: {type(e).__name__}: {e}",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }
    print(f"[dflash] draft loaded in {time.perf_counter()-t0:.1f}s "
          f"(type={type(draft).__name__})")

    # Find the target model adapter: dflash_generate calls
    #   target(input_ids, position_ids, past_key_values, use_cache,
    #          logits_to_keep, output_hidden_states)
    # and returns output.logits + output.hidden_states. It also uses
    # target.model.embed_tokens and target.lm_head.
    #
    # For Alpamayo: the VLM is Qwen3VLForConditionalGeneration which exposes
    # .language_model (the Qwen3-style text decoder). Its forward takes
    # input_ids and returns standard CausalLMOutputWithPast when called
    # directly on token inputs (no pixel_values needed if we provide
    # past_key_values already populated).
    #
    # However, the PREFILL step in dflash_generate invokes target(input_ids,
    # ...) with NO past_key_values yet. We can't run the VLM prefill through
    # the plain text path because the first prompt contains image tokens that
    # need Qwen3VL's visual encoding to be replaced.
    #
    # Minimal safe approach: the CoC decode happens AFTER the first token.
    # So we:
    #   (1) Run the FULL VLM prefill (vision + text) once to produce
    #       past_key_values and first decoded token.
    #   (2) Construct a "pre-primed" target wrapper that, on first call,
    #       returns cached prefill outputs; on subsequent calls, runs the
    #       text-only language_model forward.
    # That's invasive and brittle. Simpler fallback: we just REPORT that the
    # DFlash Transformers backend doesn't cleanly support VLMs without custom
    # prefill handoff, and record the integration status.

    # Attempt direct wrap — may fail on prefill step. We catch and fall back
    # to reporting.
    vlm = model.vlm
    lang = getattr(vlm, "language_model", None) or getattr(vlm, "model", None)
    if lang is None:
        return {
            "optimization": "dflash_coc",
            "error": f"couldn't find language_model on VLM (type={type(vlm).__name__})",
            "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        }

    # Probe the VLM module tree for debugging what we have available.
    print(f"[dflash] VLM type: {type(vlm).__name__}")
    print(f"[dflash] VLM children: {[n for n,_ in vlm.named_children()]}")
    print(f"[dflash] lm_head: {hasattr(vlm, 'lm_head')}, "
          f"get_input_embeddings: {vlm.get_input_embeddings() is not None}")

    # Configure target-side hidden layers to match DFlash expectations.
    # DFlash's decoder needs target hidden states from specific layers
    # (build_target_layer_ids). If the draft config has `target_layer_ids`
    # attribute, we use it; otherwise we compute.
    draft_cfg = draft.config
    num_target_layers = getattr(lang.config, "num_hidden_layers", 36)
    num_draft_layers = getattr(draft_cfg, "num_hidden_layers", 1)
    if not hasattr(draft, "target_layer_ids"):
        from dflash.model import build_target_layer_ids
        draft.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
    print(f"[dflash] target_layer_ids = {draft.target_layer_ids}")

    # Because implementing the VLM handoff cleanly requires ~100 lines of
    # Qwen3VL-specific prefill+decode splitting (and this is a first-pass
    # attempt), record the integration state and move on. A follow-up would
    # subclass Qwen3VLForConditionalGeneration to expose a
    # `prepare_for_spec_decode` method returning (embeds_prefix, cache).
    return {
        "optimization": "dflash_coc",
        "gpu_inf_ms": None, "hz": None, "ade_m": None, "n_preds": 0,
        "error": "DFlash Transformers backend prefill is incompatible with "
                 "Qwen3VLForConditionalGeneration without custom VLM→LM "
                 "handoff. Draft loaded ({:.1f}M params, target_layer_ids={}). "
                 "Follow-up: subclass Qwen3VL to expose prefill cache + "
                 "override dflash_generate() to skip its own prefill step."
                 .format(sum(p.numel() for p in draft.parameters()) / 1e6,
                         draft.target_layer_ids),
        "note": "PARTIAL: draft model loads successfully but VLM prefill "
                 "cannot be routed through DFlash's text-only target(input_ids) "
                 "interface in-place. Need custom integration layer.",
    }


@app.local_entrypoint()
def dflash_coc():
    r = ablation_dflash_coc.remote()
    print("[local] DFlash result:", r)


# Smoke test for DFlash: verify dflash imports and draft model loads.
@app.function(
    gpu="H100",
    image=image_dflash,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 10,
)
def smoke_dflash(
    draft_model: str = "FlashDriveVLA/Alpamayo-R1-10B-DFlash",
):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/hf/hub"
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    import torch
    info = {"cuda_available": torch.cuda.is_available()}

    try:
        import dflash
        info["dflash_version"] = getattr(dflash, "__version__", "?")
    except Exception as e:
        info["dflash_import_error"] = f"{type(e).__name__}: {e}"
        return info

    try:
        from dflash.model import DFlashDraftModel, dflash_generate, build_target_layer_ids
        info["dflash_model_api_imported"] = True
    except Exception as e:
        info["dflash_model_import_error"] = f"{type(e).__name__}: {e}"

    # Try loading draft
    try:
        import time
        from transformers import AutoModel, AutoConfig
        t0 = time.perf_counter()
        cfg = AutoConfig.from_pretrained(draft_model, trust_remote_code=True)
        info["draft_config_type"] = type(cfg).__name__
        info["draft_num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
        info["draft_hidden_size"] = getattr(cfg, "hidden_size", None)
        info["draft_vocab_size"] = getattr(cfg, "vocab_size", None)

        draft = AutoModel.from_pretrained(
            draft_model, trust_remote_code=True, dtype=torch.bfloat16,
        ).to("cuda").eval()
        info["draft_load_s"] = round(time.perf_counter() - t0, 1)
        info["draft_type"] = type(draft).__name__
        info["draft_param_count_M"] = round(
            sum(p.numel() for p in draft.parameters()) / 1e6, 1
        )
    except Exception as e:
        info["draft_load_error"] = f"{type(e).__name__}: {e}"

    return info


@app.local_entrypoint()
def smoke_dflash_local():
    r = smoke_dflash.remote()
    print("[local] DFlash smoke:", r)


# -----------------------------------------------------------------------------
# CoC-ON H100 baseline
# -----------------------------------------------------------------------------
# Apples-to-apples reference for any future DFlash integration. CoC is the
# autoregressive Chain-of-Causation text rollout the VLM decodes between
# <|cot_start|> and <|traj_future_start|>. Every other ablation in results.csv
# runs skip_cot=True (decodes 1 token) — so this is the ONLY row where the VLM
# autoregressive decode path is exercised.
#
# Config: 4cam / 4fr / 10diff (same as existing baseline row). Expected: ~900ms
# (baseline 405.4ms + ~500ms CoC decode overhead per README notes). DFlash's
# speedup target is the CoC portion; measure here, measure DFlash later, take
# ratio on same H100.

@app.function(
    gpu="H100",
    image=image_fp8,
    volumes={"/cache/hf": hf_cache, "/cache/pai": pai_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def ablation_coc_on_baseline(
    cot_max_tokens: int = 256,
    diff_steps: int = 10,
    n_cams: int = 4,
    n_frames: int = 4,
    duration_s: float = 18.0,
):
    """CoC-ON H100 baseline: identical to existing 4cam_4fr_10diff row but
    with skip_cot=False so the VLM runs the full CoC autoregressive decode.
    """
    ctx = _setup_model_common()
    result = _run_ablation_sim(
        model=ctx["model"], processor=ctx["processor"], helper=ctx["helper"],
        load_physical_aiavdataset=ctx["load_physical_aiavdataset"],
        avdi=ctx["avdi"], all_cams=ctx["all_cams"],
        needs_cam_indices_in_msg=ctx["needs_cam_indices_in_msg"],
        warmup_iters=2, label="coc_on_baseline",
        n_cams=n_cams, n_frames=n_frames, diff_steps=diff_steps,
        duration_s=duration_s, skip_cot=False, cot_max_tokens=cot_max_tokens,
    )
    result["optimization"] = "coc_on_baseline"
    result["note"] = (
        f"Baseline with CoC ENABLED (skip_cot=False, max_gen={cot_max_tokens}). "
        f"H100 / nvidia/Alpamayo-R1-10B / bf16 / flash_attn_2 / SDPA expert / "
        f"no compile. Reference for DFlash speculative-decode comparison — the "
        f"CoC decode is the only autoregressive text path in the pipeline, "
        f"and is where DFlash's 2-6x speedup applies."
    )
    return result


@app.local_entrypoint()
def coc_on_baseline(
    cot_max_tokens: int = 256,
    diff_steps: int = 10,
    n_cams: int = 4,
    n_frames: int = 4,
    duration_s: float = 18.0,
):
    r = ablation_coc_on_baseline.remote(
        cot_max_tokens=cot_max_tokens,
        diff_steps=diff_steps,
        n_cams=n_cams,
        n_frames=n_frames,
        duration_s=duration_s,
    )
    print("[local] CoC-on baseline result:", r)
