# WEDA-CV-VLM — Troubleshooting Guide

Reference for the operator on call. Each section is a real failure pattern,
its root cause, the diagnostic command that confirms it, and the fix.

When in doubt, start with **§1 First-Response Checklist** — five commands
that triage 80 % of issues in under a minute.

---
## Swap memory (recommended)

Thor has 128 GB of unified memory, but during long VLM queries the KV
cache can briefly push the working set past the available RAM. A 64 GB
swap file smooths those spikes so the allocator pages out cold memory
instead of OOM-killing the pipeline.

### Create or resize the swap file to 64 GB

```bash
# Disable the default small zram swap shipped with JetPack
sudo systemctl disable --now nvzramconfig

# (Re)create /swapfile at 64 GB
sudo swapoff /swapfile 2>/dev/null || true
sudo rm -f /swapfile
sudo fallocate -l 64G /swapfile \
  || sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=progress
sudo chmod 600 /swapfile
sudo mkswap   /swapfile
sudo swapon   /swapfile

# Lower swappiness so swap kicks in only under real pressure
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf
sudo sysctl --system

# Persist across reboots (skip if already present)
grep -q '^/swapfile' /etc/fstab \
  || echo '/swapfile  none  swap  sw  0  0' | sudo tee -a /etc/fstab

# Verify
free -h
swapon --show
```
---

## 1. First-Response Checklist

Run all five before reading any other section:

```bash
# A. Is the pipeline process alive?
pgrep -fa "python3 -m app.main"

# B. Is the API listening on the right interface?
ss -tlnp | grep 9180

# C. Is the GPU healthy?
nvidia-smi || sudo tegrastats --interval 1000 | head -3

# D. Recent VRAM / kernel events?
sudo dmesg -T | tail -50 | grep -iE 'oom|xid|cuda|nvidia'

# E. Free UMA on Thor?
free -g
```

| If you see… | Jump to |
|---|---|
| No process | §2 (pipeline won't start) |
| `127.0.0.1:9180` instead of `0.0.0.0:9180` | §5.1 (LAN bind) |
| `Xid` errors in dmesg | §3 (GPU faults) |
| < 5 GB free in `free -g` | §4 (UMA exhaustion) |
| `tegrastats` shows `RAM` near full | §4 |
| `wise-bench` reports WARN or FAIL | §8 (per-test guidance) |


---

## 2. Pipeline Won't Start

### 2.1 Models load but pipeline exits with no error

**Symptom**: `[4/4] VLM ready` then process exits cleanly.

**Cause**: `display.headless: false` with no `DISPLAY` set. cv2 raises silently when window create fails. The orchestrator's `_display_loop` catches and falls through to clean shutdown.

**Diagnose**:
```bash
echo "DISPLAY=$DISPLAY"
ls -la /tmp/.X11-unix/
```

**Fix**: either export `DISPLAY=:0` and check `xhost +local:` permissions, or set `display.headless: true` in `config/pipeline.yaml`.

### 2.2 ImportError on first launch

**Symptom**: `ModuleNotFoundError: No module named 'X'` at startup.

**Cause**: a base-image package was clobbered by a transitive `pip install` in an earlier run. `--no-deps` in `entrypoint.sh` should prevent new occurrences but won't undo past damage.

**Diagnose**:
```bash
python3 -c "import torch, transformers, cv2, numpy, PIL"
pip show torch transformers numpy
```

**Fix**: rebuild the container from the base image:
```bash
./build.sh --stop
docker rmi $(docker images -q weda-cv-vlm)
./build.sh
```

### 2.3 First VLM load takes 5–10 min

**Symptom**: `[4/4] Loading VLM...` hangs for several minutes.

**Cause**: gptqmodel JIT-compiles AWQ kernels for sm_110 the first time it sees the architecture. One-time cost, cached on disk.

**Diagnose**:
```bash
# Caches that should grow on first AWQ load:
ls -la ~/.cache/gptqmodel ~/.triton/cache /root/.cache/torch_extensions 2>/dev/null
```

**Fix**: wait it out on first run. Subsequent runs load in 90–150 s. If second run is also slow, your cache is being wiped — check that `model_cache/` is on a persistent volume mount, not a tmpfs.

---

## 3. GPU / CUDA Errors

### 3.1 `Tensor.item() cannot be called on meta tensors`

**Symptom**: VLM load fails with this error mid-`from_pretrained()`.

**Cause**: gptqmodel ≥ 7.0 + transformers ≥ 5.6 normally avoids this, but older stacks or accelerate's meta-device dispatch can trip it on Qwen2.5-VL/Qwen3-VL AWQ checkpoints.

**Diagnose**: log line right before the crash shows `[VLM] PHASE 1 — fast path`.

**Fix**: the loader auto-falls-back to PHASE 2 (CPU staging + `init_empty_weights` no-op patch). If even PHASE 2 fails:
```bash
pip install --no-deps --upgrade gptqmodel
```
Or pin transformers to 5.6.0 in the base image.

### 3.2 `'Qwen3VLMoeTextAttention' object has no attribute 'q_bmm_quantizer'`

**Symptom**: VLM crashes on every auto-caption forward, ~50 s after the first physical reasoner query fires.

**Cause**: ModelOpt's CUDA-FP8 extension JIT-compiles when Cosmos-Reason2 (Qwen3-VL-based) runs its first FP8 forward, then patches `Qwen3VLMoe*Attention` classes globally. The VLM (also Qwen3-VL-based, not modelopt-quantized because it's MoE) then crashes because its instances don't have the `q_bmm_quantizer` attribute the patched forward expects.

**Diagnose**: log shows `Loading extension modelopt_cuda_ext_fp8...` shortly before the AttributeError storm.

**Fix**: `config/pipeline.yaml` already defaults to `physical_strategy: "fp16"` to avoid this. If you bumped it to `fp8` or `nvfp4`, revert:
```yaml
quantization:
  physical_strategy: "fp16"
  physical_kv_cache: "fp16"
```
Cost: +3 GB UMA on Cosmos-Reason2-2B. Only safe to use `fp8` if your VLM is NOT a Qwen3-VL derivative (LLaVA, InternVL, Phi-VL).

### 3.3 `The size of tensor a (4304) must match the size of tensor b (4352)`

**Symptom**: Per-track auto-caption logs `VLM cache error: The size of tensor a ...` repeatedly. SCENE row in the HUD shows `(vision-encoder mismatch)`.

**Cause**: `qwen_vl_utils.process_vision_info` is missing. The fallback path in `_build_vlm_inputs` passes the raw PIL image, and the AWQ Qwen3-VL processor's `smart_resize` produces a different token count than the chat template expanded — so the merged sequence length doesn't match.

**Diagnose**:
```bash
docker exec weda-cv-vlm python3 -c "from qwen_vl_utils import process_vision_info; print('OK')"
```

**Fix**:
```bash
docker exec weda-cv-vlm pip install --no-deps 'qwen-vl-utils>=0.0.14'
```
Then restart the pipeline (Ctrl+C, rerun `python3 -m app.main`). The entrypoint installs this eagerly on container boot — if you see this in production, the install failed silently, check `/tmp/qvl.log`.

### 3.4 `CUDA out of memory` / Xid errors / display freezes

**Symptom**: process dies, `dmesg` shows `Xid` faults, display server stalls.

**Cause**: UMA exhaustion. On Jetson Thor 122.8 GB UMA is shared between OS, GPU, and CPU. With Qwen3-VL-30B BF16 (~60 GB) + KV cache spike + auto-caption + new user query landing simultaneously, you can hit 92 GB+ used and the allocator falls back to paging.

**Diagnose**:
```bash
free -g
sudo tegrastats --interval 500 | head -5     # look for "RAM" near total
```

**Fix**:
1. Switch to AWQ checkpoint to drop weights from 60 GB → 17 GB:
   ```yaml
   # config/pipeline.yaml
   vlm:
     model: "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
   ```
2. Reduce `vlm.max_tokens` from 500 to 200 to cap KV growth.
3. Lower `physical_reasoning.max_tokens` from 512 to 256.

### 3.5 `fatal error: Python.h: No such file or directory`

**Symptom**: gcc compile fails inside Triton's JIT path.

**Cause**: missing `python3.<minor>-dev` — Triton needs `Python.h` to JIT-compile CUDA helpers (`cuda_utils.c`).

**Diagnose**:
```bash
ls /usr/include/python3.12/Python.h
```

**Fix**: handled by entrypoint `[2] System Dependencies` — runs `apt-get install python3.12-dev` if header is missing. If you see this error in production, the entrypoint apt step failed; rerun manually:
```bash
apt-get update && apt-get install -y python3.12-dev
```

---

## 4. Memory / Performance

### 4.1 FPS drops to 5 and stays there

**Symptom**: Pipeline runs, but FPS counter shows 5 instead of 25–30.

**Cause**: UMA pressure → torch allocator paging GPU↔CPU. Usually triggered by:
- Per-crop VLM forwards leaking due to a forward error (see §3.3)
- Too many concurrent SSE streams
- KV cache not being released between queries

**Diagnose**:
```bash
sudo tegrastats --interval 1000 | head -10
# Look for: RAM near full + GR3D_FREQ low (GPU starved)
```

**Fix**:
1. Confirm `qwen-vl-utils` is installed (§3.3).
2. Check the orchestrator log for repeated `VLM cache error` lines.
3. Restart the pipeline (Ctrl+C + rerun) — long sessions accumulate state.

### 4.2 Model load 444 s instead of 120 s

**Symptom**: `All models loaded in 444s` instead of the expected ~127 s.

**Cause**: VLM AWQ load took the slow CPU-staged PHASE 2 path because PHASE 1 (device_map streaming) hit the meta-tensor crash.

**Diagnose**: log shows `[VLM] PHASE 2 succeeded in 250.0s`.

**Fix**: this is a fallback, not a failure — the model loaded correctly, just slowly. To restore PHASE 1:
- Confirm gptqmodel ≥ 7.0: `pip show gptqmodel`
- Confirm transformers ≥ 5.6: `pip show transformers`
- Wipe the VLM cache so the next load tries fresh:
  ```bash
  rm -rf /wise-edge/weda-cv-vlm/model_cache/quantized/vlm_*
  ```

### 4.3 Container UMA pinned at 121 / 123 GB

**Symptom**: `nvidia-smi` (or tegrastats) shows ~99 % UMA used at idle.

**Cause**: previous run leaked allocations from cascading per-crop forward errors (§3.3) and the OS page cache holds onto them.

**Fix**: drop the page cache + restart container:
```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
./build.sh --stop && ./build.sh
```

---

## 5. HTTP API

### 5.1 `Connection refused` from another LAN device

**Symptom**: `curl http://<thor-ip>:9180/v1/info` from your laptop returns `Connection refused`.

**Cause**: API bound to `127.0.0.1` instead of `0.0.0.0`.

**Diagnose**:
```bash
ss -tlnp | grep 9180
# 127.0.0.1:9180   → loopback only (cause)
# 0.0.0.0:9180     → all interfaces (fixed)
```

**Fix**: there are three precedence levels (env wins):
1. `WEDA_API_BIND` env var in `.env` (highest priority)
2. `api.bind` in `config/pipeline.yaml`
3. Default `127.0.0.1`

Set in `.env`:
```
WEDA_API_BIND=0.0.0.0
```
Then **`docker compose down && up`** — `restart` does NOT re-read `.env`.

⚠ Once on `0.0.0.0`, set `WEDA_API_TOKEN` to a strong value or write endpoints are exposed unauthenticated.

### 5.2 Swagger UI loads but `Try it out` returns empty

**Symptom**: `http://<thor-ip>:9180/docs` loads, you click Execute, "Server response" stays empty.

**Cause**: Swagger UI's JS makes the actual request from your browser context. If the OpenAPI spec advertises `servers: [{url: "http://127.0.0.1:9180"}]` but you loaded the page from `<thor-ip>`, the JS fetches `127.0.0.1` from your laptop's loopback — which has nothing.

**Diagnose**: open browser DevTools → Network → click Execute → look at the failed request URL.

**Fix**: same as §5.1 — bind to `0.0.0.0` and load Swagger from the Thor's LAN IP.

### 5.3 401 Unauthorized on every endpoint

**Symptom**: `{"type":"about:blank","title":"Unauthorized","status":401,...}` on every call.

**Cause**: `WEDA_API_TOKEN` is set, and you're not sending `Authorization: Bearer <token>`.

**Fix** (in Swagger UI): click the green "Authorize" button top-right, paste the token (no `Bearer ` prefix — Swagger adds it).

**Fix** (curl):
```bash
TOKEN=$(grep ^WEDA_API_TOKEN .env | cut -d= -f2)
curl -fsS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9180/v1/vlm/query -d '...'
```

### 5.4 `/v1/info` shows 200 OK but `features.vlm: false`

**Symptom**: API alive, but VLM feature flag reads false.

**Cause**: VLM didn't finish loading or crashed at load time.

**Diagnose**:
```bash
curl -fsS http://127.0.0.1:9180/v1/health | jq
# look for vlm_loaded, embed_loaded, physical_loaded
```

**Fix**: check pipeline stdout for `[VLM] PHASE 1 succeeded` or any `[VLM] ... failed`. Common causes:
- Missing `HF_TOKEN` for gated repos (Cosmos-Reason2-2B requires login)
- Disk full at `model_cache/`
- Missing `timm` / `open_clip_torch` (entrypoint should install these — check logs)

### 5.5 `/v1/scene/at?ts=...` returns 404

**Symptom**: passing a small integer (e.g. `ts=6`) returns `no entry close to that timestamp`.

**Cause**: `ts` is **Unix epoch seconds** (e.g. `1747094572`), not a relative offset. The scene memory only holds the last `scene_memory.window_min` minutes (default 30 min), so any value outside that window has no match.

**Fix**: use the relative `ago_s` parameter for "N seconds ago":
```bash
curl -fsS "http://<thor>:9180/v1/scene/at?ago_s=30" | jq
```
Or pass an epoch `ts` you got from a previous call:
```bash
TS=$(curl -fsS http://<thor>:9180/v1/scene/now | jq -r .ts)
curl -fsS "http://<thor>:9180/v1/scene/at?ts=$TS" | jq
```
Omitting both returns the most recent buffered entry.

### 5.6 VLM responds `Cannot answer — object not in scene`

**Symptom**: `/v1/vlm/query`, `/v1/vlm/describe`, or `/v1/physical/query` returns a refusal instead of running the model, e.g.
```
Cannot answer — object(s) not in scene: forklift. Currently detected: person, tv.
```

**Cause**: by design. Before the model runs, the query is scanned via `fact_checker.missing_classes_in_query` against the live detection classes. If the query asks about a class the detector doesn't currently see, the model is skipped — this prevents the VLM from hallucinating about objects it can't observe.

**Fix**: rephrase the query to match a currently-detected class, wait for the object to enter the frame, or search the past with `/v1/scene/find`.

### 5.7 Raw camera MJPEG (no detection overlay)

**Symptom**: you want a clean camera feed without bounding boxes on `/v1/stream/rgb.mjpg`.

**Fix**: append `annotate=false`:
```
http://<thor>:9180/v1/stream/rgb.mjpg?annotate=false
```
Or set the deployment default in `config/pipeline.yaml`:
```yaml
streaming:
  draw_detections: false
```
The query parameter overrides the config per-request. YOLO keeps running for tracking / alarms / fact-check — only the drawing on this one stream is skipped.

---

## 6. Camera / Input

### 6.1 Center display panel is black, FPS=0

**Symptom**: HUD chrome renders but the camera frame area stays black.

**Cause**: Orbbec USB hang, USB cable issue, or wrong device index.

**Diagnose**:
```bash
# Inside the container:
ls /dev/video*
v4l2-ctl --list-devices
# Find the python process and inspect its stack:
sudo py-spy dump --pid $(pgrep -f "python3 -m app.main" | head -1)
# Look for `wait_for_frames` blocking or USB I/O blocked
```

**Fix**:
1. Replug the camera USB
2. Set `input.source` to a different camera index in `pipeline.yaml`
3. Force `input.backend: "v4l2"` to bypass Orbbec SDK if the SDK is the hang source



### 6.2 `getXu failed, error code=-7` in logs

**Symptom**: Orbbec error in pipeline log. Camera may stop delivering frames.

**Cause**: USB control transfer failure — typically a flaky USB cable, USB hub overload, or the camera firmware desyncing.

**Fix**:
- Plug camera directly into the Thor (no hub)
- Use a USB 3.0 cable rated for Gen 2 speeds
- Power-cycle the camera if the host doesn't bring it back

---

### 6.3. Wrong camera node: grayscale/depth image, or the index stops working after a replug

**Symptom:** The demo runs but the picture is grayscale or looks like a depth map instead of color video; or a `source` index that worked before returns `Cannot open video source: N` / `can't open camera by index` after the camera was unplugged and reconnected.

**Root Cause:** Multi-sensor cameras (for example the Advantech CAM-5530) expose several `/dev/video*` nodes — RGB, depth and IR streams plus metadata nodes — and only some of them deliver frames. Picking the first node that opens can select the depth or IR sensor. Additionally, V4L2 node numbering is assigned at enumeration time: after a replug the same physical camera can come back under different indices.

**Diagnostic:** Identify the RGB node by its advertised pixel formats, not by trial and error:

```bash
# Inside container
for d in /dev/video*; do echo "== $d"; v4l2-ctl -d $d --list-formats 2>/dev/null; done
```

The RGB stream advertises `MJPG` and/or `YUYV`. Depth nodes advertise `Z16`; IR nodes advertise `GREY`/`Y8`. Nodes listing no capture formats are metadata nodes.

**Resolution:** Set `source` in the config to the node that lists `MJPG`/`YUYV`. After any replug, re-run the diagnostic and update `source` if the index moved — the numbering is not stable across reconnects.

## 7. Display / HUD

### 7.1 VLM responses truncated at ~150 characters

**Symptom**: SCENE / QUERY RESPONSE sections cut off mid-sentence.

**Cause**: per-section line cap. By design — the bottom panel has finite vertical budget.

**Fix**:
- Press `[X]` to open the full-response modal — scrollable, untruncated.
- Or pull full text via REST: `curl http://127.0.0.1:9180/v1/scene/now | jq -r .scene_description`

### 7.2 Stale responses linger on screen

**Symptom**: a query response from 10 minutes ago is still showing.

**Cause**: not stale — it just hasn't aged out yet.

**Fix**: responses age out after **60 s** of no updates. Once out, the section vanishes and other rows expand to fill its space. Adjust by editing `STALENESS_S` in `app/display/renderer.py:_render_bottom_bar`.

---

## 8. Diagnostic / Verification (`wise-bench`)

`wise-bench` is a compiled binary — run it directly, not through `python3`:

```bash
./scripts/wise-bench               # fast (skips model loads)
./scripts/wise-bench --heavy       # also exercises model init
```

Each row reports `PASS`, `WARN`, `FAIL`, `SKIP`, or `INFO`. `PASS` is green; `WARN` is yellow (degraded but workable); `FAIL` is red (a real defect — investigate); `SKIP` means the test was deliberately skipped (e.g. `--heavy` not set); `INFO` is purely informational (test value, not a verdict).

The remainder of §8 groups guidance by test prefix.

### 8.1 SYS — System information (`SYS-001` … `SYS-006`)

Tests: Python version, platform, CPU & RAM, L4T release, CUDA / cuDNN, `nvidia-smi`.

| Test | Common cause of FAIL/WARN | Fix |
|---|---|---|
| `SYS-001 System Python` | Python below 3.10 | base image is mis-tagged; pull a current image |
| `SYS-004 L4T (Jetpack)` | `/etc/nv_tegra_release` missing or `R<38` | container started without `--runtime=nvidia` or on a non-Jetson host |
| `SYS-005 System CUDA/cuDNN` | `nvcc` not on PATH, or `/usr/local/cuda` missing | base image lacks the CUDA toolkit; verify `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64` |
| `SYS-006 NVIDIA-SMI` | `command not found`, or output empty | `--gpus all` (or `--runtime nvidia`) was not passed to `docker run`; on host check `dpkg -l | grep nvidia-l4t-core` |

### 8.2 HW — Hardware-accelerated video (`HW-001`)

`HW-001 Unified GPU Video (NVENC)` checks `/dev/nvidia0` is reachable for encode. FAIL means the NVIDIA container runtime didn't expose the device — set `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics,display` in compose, or run with `--gpus all`. Pipeline doesn't strictly need NVENC for headless inference, but RTSP / file recording does.

### 8.3 FWK — PyTorch + ONNX (`FWK-001` … `FWK-007`)

Tests: PyTorch core / FP32 / BF16 / FP8, TorchVision, ONNX core, ONNXRuntime-GPU.

- `FWK-001 PyTorch Core | FAIL` — `import torch` failed or `torch.cuda.is_available()` returned False inside the container. Confirm the base image has a CUDA-enabled torch wheel: `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"`. If `cuda is None`, you have a CPU-only build.
- `FWK-003/004 BF16 / FP8 | FAIL` — torch doesn't expose the dtype on this device. BF16 requires sm_80+; FP8 requires sm_89+ (Hopper) or sm_90/sm_110 (Blackwell/Thor). On Thor both should pass; if either fails, you're running on the wrong GPU node or torch is too old.
- `FWK-006 ONNXRuntime (GPU) | WARN` — `CUDAExecutionProvider` not in `ort.get_available_providers()`. Cause: PyPI's CPU-only `onnxruntime` overrode the base image's source-built `onnxruntime-gpu`. The pipeline still works without GPU-ORT (TRT-direct detection doesn't need it). See [microsoft/onnxruntime#27098](https://github.com/microsoft/onnxruntime/issues/27098) for Thor + CUDA 13 status.
- `FWK-007 ONNX Core | FAIL` — `onnx` package missing; check the pacakges you recently installed whether its dependecies require paticular version of onnx? FIx: install pacakges with --no-deps flag (rare; usually preinstalled).

### 8.4 MM — OpenCV + GStreamer (`MM-001` … `MM-006`)

- `MM-001 OpenCV (GStreamer) | FAIL` — OpenCV built without GStreamer support. Confirm with `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer` — `YES` expected.
- `MM-002 OpenCV (CUDA) | FAIL` — `cv2.cuda` missing or `getCudaEnabledDeviceCount() == 0`. The pipeline's depth/detector paths don't need cv2-CUDA, but a few utilities do; rebuild OpenCV with CUDA support if you rely on those.
- `MM-003 nvv4l2decoder` / `MM-004 nvv4l2h264enc` / `MM-005 nvvidconv | FAIL` — the L4T GStreamer plugins aren't visible. Check `gst-inspect-1.0 nvvidconv` from inside the container. Usually a missing `--runtime=nvidia` or a base image that didn't ship `nvidia-l4t-gstreamer`.

### 8.5 TRT — TensorRT (`TRT-001`, `TRT-002`)

- `TRT-001 TensorRT (Native) | FAIL` — `import tensorrt` failed or `trt.Builder` couldn't construct. Inspect: `python3 -c "import tensorrt; print(tensorrt.__version__)"`. Likely cause: TRT version doesn't match the host driver (e.g. TRT 10.x against an older L4T). Re-pull a matching base image.
- `TRT-002 TensorRT (Caps) | INFO` — informational only; reports FP16 / INT8 builder capabilities.

### 8.6 DAT — Scientific data stack (`DAT-001` … `DAT-005`)

`numpy`, `pandas`, `scipy`, `PIL`, `shapely`. These are part of the base image and rarely fail on their own.

Pitfall: numpy 2.x ABI break. The image pins `numpy 1.26.x` because TensorRT bindings, onnxruntime-gpu, and some torch wheels are compiled against the numpy-1.x C ABI. If `DAT-001 numpy` reports a 2.x version, expect cascading FAILs in `FWK-001`, `TRT-001`, and possibly `MM-002`. Pin back: `pip install --force-reinstall 'numpy<2'` and restart.

### 8.7 NLP — Transformers stack (`NLP-001` … `NLP-004`)

`transformers`, `huggingface_hub`, `tokenizers`, `accelerate`.

- All four are required for VLM / embedder / reasoner. FAIL on any → corresponding model load will fail.
- transformers 5.x removed several internal helpers (`no_init_weights`, the older `PretrainedConfig` kwarg permissiveness, etc.) that older packages still import. If a heavy test (`AI-002` / `AI-003` / `AI-004`) crashes with an `ImportError` referencing a missing transformers symbol, the package depending on it needs a compatible version. See [transformers#38078](https://github.com/huggingface/transformers/issues/38078) and the Qwen2.5-VL discussions ([QwenLM/Qwen3-VL#1033](https://github.com/QwenLM/Qwen3-VL/issues/1033)) for the AWQ-route's compatibility matrix.

### 8.8 CTR — Container environment (`CTR-001` … `CTR-008`)

| Test | What it checks | If WARN/FAIL |
|---|---|---|
| `CTR-001 Working directory` | cwd is the project root | container started with `WORKDIR` elsewhere; `cd /wise-edge/weda-cv-vlm` |
| `CTR-002 .ready sentinel` | entrypoint completed at least once | first boot still running — wait for `[INFO] Environment ready.`; if it never appears, check `docker logs weda-cv-vlm` for entrypoint errors |
| `CTR-003 HuggingFace cache` | `~/.cache/huggingface/` exists & writable | volume mount missing — see compose `volumes:` |
| `CTR-004 Model cache` | `model_cache/` exists & has quantized dirs | same as CTR-003; first run hasn't quantized anything yet |
| `CTR-005 HF_TOKEN` | env var set + starts with `hf_` | gated repos (Cosmos-Reason2, some VLMs) won't download; export `HF_TOKEN=hf_...` in `.env` |
| `CTR-007 PYTHONDONTWRITEBYTECODE` | env var set to `1` | not strictly required, but stale `.pyc` can shadow edits — `export PYTHONDONTWRITEBYTECODE=1` |
| `CTR-008 Pipeline config path` | `config/pipeline.yaml` exists | missing volume mount or wrong path; verify `docker-compose.yml` |

### 8.9 GPU — Passthrough integrity (`GPU-001` … `GPU-006`)

Tests Container ↔ host GPU exposure.

- `GPU-001 /dev/nvidia* nodes | FAIL` — fewer than 5 nodes visible. Container wasn't started with `--gpus all` / `--runtime=nvidia`.
- `GPU-002 NVIDIA runtime | WARN` — `NVIDIA_VISIBLE_DEVICES` or `NVIDIA_DRIVER_CAPABILITIES` not set. Set both in `docker-compose.yml` under `environment:`.
- `GPU-003 torch.cuda.is_available | FAIL` — torch sees no CUDA device despite nodes existing. Driver / library mismatch — check `nvidia-smi` output against `torch.version.cuda`.
- `GPU-005 Blackwell sm_110 | FAIL` — CUDA capability is reported as something other than `(11, 0)`. Either you're not on Thor, or torch is too old to recognise sm_110. Update torch.
- `GPU-006 UMA detected | FAIL` — GPU memory total doesn't match system memory total within tolerance. Indicates the GPU isn't using unified memory (Thor should always show UMA). Probably running on the wrong host.

### 8.10 NVT — nvitop telemetry (`NVT-001` … `NVT-007`)

`nvitop` reads NVML. On Jetson UMA, several device-level NVML fields return `N/A` because the kernel doesn't populate them — this is by design, not a bug.

- `NVT-003 GPU utilization | PASS | 0% SM` — utilization reads 0 at idle because NVML doesn't expose GA10B compute counters on Jetson. The actual utilization comes from `tegrastats GR3D_FREQ%` (the pipeline's runtime monitor reads that automatically). If you want a live percent, run `tegrastats --interval 1000` in another shell.
- `NVT-004 memory | WARN` — NVML returned `N/A` for memory. The pipeline's GpuMonitor falls back to `/proc/meminfo` (UMA total). Informational only.
- `NVT-005 temperature | WARN` — NVML reports no temperature; the monitor falls back to `/sys/class/thermal/thermal_zone*/temp` automatically. WARN here just means the bench used the sysfs path.
- `NVT-001 module import | FAIL` — `nvitop` isn't installed. `pip install nvitop`.

### 8.11 SDK — `weda_optim` (`SDK-001` … `SDK-010`)

- `SDK-005 GpuMonitor backend | PASS | NvitopDeviceMonitor` — preferred backend. If it reports `TegrastatsMonitor`, NVML is unhealthy on this boot; functional but slightly higher latency.
- `SDK-008 VIC availability | WARN` — VIC hardware offload isn't engaged. `nvvidconv` and its Python binding need to be present together; if they aren't, the pipeline falls back to `cv2.resize` (sub-millisecond for the resolutions in use). Informational on most images.
- `SDK-009 quantization | PASS | detect_backend('auto') → <backend>` — reports which quantizer was selected (`bitsandbytes` / `modelopt`). For MoE VLMs (Qwen3-VL-MoE family) the selection is `bitsandbytes` regardless because modelopt can't handle 3-D expert weights — see §3.2.

### 8.12 APP — Application modules (`APP-001` … `APP-012`)

Each test imports one runtime module (`orchestrator`, `pipeline`, `detector`, `bytetrack`, `depth`, `embedder`, `reasoning`, `physical_reasoner`, `query_engine`, `camera_factory`, `renderer`, `config`).

FAIL means a syntax error, missing transitive dep, or stale `.pyc` shadowing source. Diagnose with the printed error text. Common fixes:
- Clear stale bytecode: `find /wise-edge/weda-cv-vlm -name __pycache__ -type d -exec rm -rf {} +`
- Confirm the imported module's transitives are present (use `pip show`)
- Reinstall the missing transitive with `--no-deps`

### 8.13 DEP — Dependency matrix (`DEP-001` … `DEP-014`)

Tier 1 (hard requirement — FAIL stops the pipeline): `pyyaml`.
Tier 2 (recommended — WARN if missing; degrades a feature): `einops`, `timm`, `open_clip_torch`, `qwen_vl_utils`, `gptqmodel`, `presidio_analyzer`.
Tier 3 (optional — INFO if missing): `bitsandbytes`, `nvidia-modelopt`, `triton`, `pyorbbecsdk`, `pypcre`, `logbar`.
Special: `ultralytics` is BYOL-gated — INFO if not installed unless `ACCEPT_ULTRALYTICS_EULA=true`, in which case it becomes FAIL.

If a Tier-2 package is WARN, the entrypoint's install for it failed silently. Inspect `/tmp/<pkg>.log` for the pip output.

### 8.14 CFG — Pipeline configuration (`CFG-001` … `CFG-007`)

Validates `config/pipeline.yaml` loads and the seven expected sections (`input`, `detection`, `embedding`, `physical_reasoning`, `vlm`, `quantization`) are present and well-formed.

FAIL on `CFG-001` is a YAML parse error — the printed line/column tells you where. FAIL on a specific section means it's missing or malformed. Compare against the shipped `config/pipeline.yaml` for the expected schema.

### 8.15 API — HTTP API stack (`API-001` … `API-030`)

- `API-001/002/003 fastapi / uvicorn / sse-starlette | FAIL` — `pip install` for the API stack failed during entrypoint. Inspect `/tmp/fastapi.log`, `/tmp/uvicorn.log`, `/tmp/sse.log`.
- `API-01x router <name> | FAIL` — that router's module import raised. Inspect the printed traceback; usually a transitive dep is missing.
- `API-030 OpenAPI build | FAIL` — `app.api.main` couldn't build the OpenAPI schema. Almost always due to a Pydantic schema validation error in one of the routers — the traceback names the offending field. If `--live` mode also can't reach `/v1/openapi.json`, the API process isn't actually running; check `ss -tlnp | grep 9180`.

### 8.16 MOD — Innovation modules (`MOD-001` … `MOD-008`)

Imports each module and verifies its expected public symbols exist (e.g. `fact_checker.verify`, `scene_memory.SceneMemory`).

FAIL → import error. Diagnose by importing manually:
```bash
docker exec weda-cv-vlm python3 -c "from app.modules.fact_checker import verify; print('ok')"
```

### 8.17 QE — Query engine routing (`QE-001` … `QE-004`)

- `QE-001 tier1` — 5 probe queries classified as Tier-1 (structured answer, no model).
- `QE-002 tier25` — 3 probes classified as Tier-2.5 (Cosmos-Reason2 physical reasoner).
- `QE-003 tier3` — 3 open-ended probes routed to OPEN_ENDED (heavy VLM).
- `QE-004 COCO80 stoplist` — class-name stopwords filtered for the fact-checker.

FAIL here means the classifier in `app/modules/query_engine.py` no longer recognises the probe patterns — usually because the patterns were edited. Inspect the probe list at the bottom of the wise-bench source.

### 8.18 AI — Functional smoke tests (`AI-001` … `AI-007`)

Run only with `--heavy`. Without `--heavy`, the four model-load tests SKIP.

- `AI-001 detector init` — downloads a small YOLO model and verifies it loads. FAIL on first run usually means Ultralytics couldn't reach github.com / their CDN; check connectivity and Ultralytics EULA acceptance.
- `AI-002 embedder init` — verifies `AutoConfig.from_pretrained` succeeds for Cosmos-Embed1 under the same signature-filter the runtime uses. FAIL means a different config schema problem; capture the traceback.
- `AI-003 reasoner init` — `AutoConfig` for Cosmos-Reason2-2B. FAIL usually means `HF_TOKEN` missing for the gated repo.
- `AI-004 VLM init` — `AutoConfig` for the configured VLM (Qwen3-VL-30B-A3B by default). FAIL usually means HF Hub rate-limited the metadata fetch, or `model_cache/` is on a read-only mount.
- `AI-005/006/007` — pure-Python checks (safety distance, query classification, depth projection). FAIL here means a regression in `app/modules/spatial_tools.py`, `query_engine.py`, or `depth.py`.

---

## 9. Edge cases

Issues that don't fit a wise-bench row but bite in production.

### 9.1 Disk fills up mid-session

**Symptom**: pipeline hangs, log shows `OSError: [Errno 28] No space left on device`, or HF Hub downloads stall.

**Cause**: `model_cache/` accumulates checkpoints, `output/audit/` accumulates JSONL, `/tmp` accumulates pip logs and HF lock files.

**Diagnose**:
```bash
df -h /wise-edge /tmp
du -h --max-depth=1 /wise-edge/weda-cv-vlm/model_cache /wise-edge/weda-cv-vlm/output 2>/dev/null
```

**Fix**: rotate audit logs (`find output/audit -mtime +30 -delete`); trim model cache to only the active checkpoints; allocate a larger NVMe partition for `/wise-edge/`. Triton's JIT cache (`~/.triton/cache`) also grows over time — safe to wipe if no inference is running.

### 9.2 HF Hub: `401 Unauthorized` or `403 Forbidden` on model download

**Symptom**: model load fails with `HfHubHTTPError: 401` or `Cannot access gated repo`.

**Cause**: `HF_TOKEN` missing, expired, or lacks access to the repo (Cosmos-Reason2-2B requires gated access acceptance on the HF Hub page).

**Fix**:
1. Check `HF_TOKEN` is set: `echo "$HF_TOKEN" | head -c 8`. Must start with `hf_`.
2. Accept the model's terms on huggingface.co (browser, while logged in).
3. Restart the container so the env var propagates.

### 9.3 HF Hub: rate-limit (`429 Too Many Requests`)

**Symptom**: model loads work intermittently but periodically fail with `HfHubHTTPError: 429`.

**Cause**: anonymous downloads are rate-limited; or your token is shared across many containers.

**Fix**: set `HF_TOKEN` (raises the cap), and pre-cache via `python3 scripts/prepare_models.py` once. Subsequent boots read from the local cache; HF Hub is only re-hit when a checkpoint changes.

### 9.4 First-launch hang during model prepare (no progress bar)

**Symptom**: `prepare_models.py` prints "Downloading…" and stalls for minutes with no output.

**Cause**: HF Hub progress-bar uses `tqdm`, which buffers on `dumb` terminals (e.g. `docker exec` without `-t`).

**Fix**: re-attach with a TTY:
```bash
docker exec -it weda-cv-vlm python3 scripts/prepare_models.py
```
If it still appears stalled, follow network activity: `docker exec weda-cv-vlm bash -c 'ss -tn | head'`.

### 9.5 Container UMA exhausts after multiple restarts

**Symptom**: `free -g` shows < 5 GB free even at idle; allocator paging kicks in.

**Cause**: Linux page cache holds onto large model files mapped during a previous run. Restarting the container doesn't drop the host's page cache.

**Fix**:
```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### 9.6 Multiple Thor GPU nodes (`/dev/nvidia0` and `/dev/nvidia1`)

**Symptom**: torch sees two CUDA devices; the pipeline appears confused about device placement.

**Cause**: Thor exposes two device nodes for the iGPU's two compute partitions on some L4T builds; both map to the same physical SM array.

**Fix**: pin to one with `CUDA_VISIBLE_DEVICES=0` in `.env`, and verify with `python3 -c "import torch; print(torch.cuda.device_count())"`.

### 9.7 TRT engine cache invalidated on every restart

**Symptom**: `yolo26n-seg.engine` rebuilds (~90 s) on each container start.

**Cause**: the engine cache lives next to the `.pt` file inside the container's overlay layer, not on a persistent volume.

**Fix**: confirm `model_cache/yolo/yolo26n-seg.engine` exists and is on a mounted volume. If the engine is rebuilt despite that, the file's hash didn't match — re-run `python3 scripts/prepare_models.py` to repopulate the cache.

### 9.8 Container clock skew breaks TLS to HF Hub

**Symptom**: `huggingface_hub` raises `SSLError: certificate verify failed: certificate has expired or is not yet valid`.

**Cause**: container clock differs from real time by minutes, certificates fail validity check.

**Fix**: ensure the host clock is set (NTP). Mount `/etc/localtime` and `/etc/timezone` into the container (already in `docker-compose.yml`). If the host is fine, `docker exec weda-cv-vlm date` to confirm.

### 9.9 `Permission denied` on `model_cache/` or `output/`

**Symptom**: pipeline can't write to `output/audit/` or `model_cache/`.

**Cause**: the host directory was created as root and is bind-mounted to a non-root container, or vice versa.

**Fix**: align ownership:
```bash
sudo chown -R $(id -u):$(id -g) /wise-edge/weda-cv-vlm/model_cache
sudo chown -R $(id -u):$(id -g) /wise-edge/weda-cv-vlm/output
```

### 9.10 Long-running session degrades over hours

**Symptom**: After many hours of uptime, FPS slowly drops and UMA creeps up.

**Cause**: Cumulative state in long-lived threadpools (Triton JIT cache, KV-cache fragmentation, accumulated SSE subscribers that didn't disconnect cleanly).

**Fix**: restart the pipeline on a schedule (cron'd `docker restart weda-cv-vlm` once a week is sufficient for booth-grade reliability). For 24/7 deployments, add an external watchdog that restarts on FPS drop.

### 9.11 NVIDIA GStreamer plugins missing on Jetson Thor (AIR-75) — `gst-inspect …/libgstnvvideo4linux2.so` → "No such element or plugin"

**Symptom**: inside the container the NVIDIA GStreamer elements are gone. The
entrypoint logs `nvvidconv element missing — frame resize uses cv2 fallback`, and:
```bash
docker exec weda-cv-vlm gst-inspect-1.0 \
  /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so
# No such element or plugin '/usr/lib/.../libgstnvvideo4linux2.so'
```
The file simply does not exist in the container. The pipeline still runs but
decode/resize fall back to CPU (cv2) — no NVDEC/NVENC/VIC.

**Cause**: on Jetson **Thor** (L4T R38 / JetPack 7) the GPU is exposed with a
dGPU-style driver model — `/dev/nvidia0`, `/dev/nvidiactl`, **no `/dev/nvgpu`**.
`nvidia-container-runtime` runs `mode = "auto"` and therefore selects **CDI**
injection (`/etc/cdi/nvidia.yaml`). That CDI spec ships CUDA/EGL/Vulkan but
**zero GStreamer entries** — so the `libgstnv*.so` plugins are never mounted in.
The CSV mount list that *does* contain them (`drivers.csv`) only applies in **CSV**
mode, which Thor's driver model never triggers.

> The AIMB-294 twin box has no `/etc/cdi/nvidia.yaml`, so it falls back to CSV
> injection and the plugins appear normally. The presence of that one file is the
> entire difference between a working and a broken box.

**Diagnose**:
```bash
# gpu spec present == CDI mode (broken for gst)
nvidia-ctk cdi list | grep nvidia.com/gpu           # listed  → CDI (no gst plugins)
ls /etc/cdi/nvidia.yaml                             # exists  → forces CDI
grep -c gstreamer /etc/cdi/nvidia.yaml              # 0       → confirms none injected

# how many gstreamer mounts did the runtime actually make into the container?
docker exec weda-cv-vlm sh -c \
  'awk "{print \$5}" /proc/self/mountinfo | grep -c gstreamer-1.0'   # 0 == broken
```
Note: `privileged: true` does **not** help — privilege grants capabilities, not
file visibility. A plugin that was never mounted still isn't on the filesystem.

**Fix A — host-level, permanent (preferred, needs sudo)**
Remove the CDI GPU spec so the runtime falls back to CSV, matching AIMB-294:
```bash
sudo mv /etc/cdi/nvidia.yaml /etc/cdi/nvidia.yaml.disabled
./build.sh
```
Relaunch via `./build.sh` rather than calling `docker compose` directly: the
compose file expects `DISPLAY`, `XAUTHORITY` and `X11_SOCKET`, and `build.sh` is
what auto-detects and exports them. Invoking compose on its own leaves those
unset and the X11 bind-mounts resolve to empty paths.
Reversible (`mv` back to restore). Affects every GPU container on the box, but the
CSV path is the validated Jetson default.

**Fix B — per-container, no sudo (compose bind-mount)**
Inject the host plugins + their dep libs into the container. Add to
`docker-compose.yml`:
```yaml
environment:
  - LD_LIBRARY_PATH=/opt/nvhostlibs:${LD_LIBRARY_PATH}   # plugin deps
  - GST_PLUGIN_PATH=/opt/nvgst                            # nv plugins
volumes:
  - /usr/lib/aarch64-linux-gnu/gstreamer-1.0:/opt/nvgst:ro
  - /usr/lib/aarch64-linux-gnu/nvidia:/opt/nvhostlibs:ro
```
Then recreate. `GST_PLUGIN_PATH` adds the plugins; `/opt/nvhostlibs` supplies the
two deps CDI omits (`libgstnvcustomhelper`, `libgstnvdsseimeta`). This repo ships
Fix B in `docker-compose.yml` by default.

⚠ Setting `NVIDIA_CONTAINER_RUNTIME_MODE=csv` on the container does **not** work —
compose's `deploy.resources.reservations.devices: driver: nvidia` requests a CDI
device and forces the CDI path regardless of the env var.

**Verify** (either fix):
```bash
docker exec weda-cv-vlm sh -c \
  'for e in nvv4l2decoder nvv4l2h264enc nvvidconv; do \
     printf "%-16s " $e; gst-inspect-1.0 $e >/dev/null 2>&1 && echo OK || echo MISSING; done'
docker logs weda-cv-vlm 2>&1 | grep -i nvvidconv     # want: "nvvidconv element available"
```
Known non-blocker: `nvjpegdec` stays MISSING — `libgstnvjpeg.so` needs
`libgstnvexifmeta.so`, absent on the host. Not used by this pipeline (H.264/H.265
via `nvv4l2decoder`).

### 9.12 Container won't start after fresh checkout: `entrypoint.sh: permission denied`

**Symptom**: after a fresh `git clone` / checkout the container fails to start, or
`./build.sh` reports `Permission denied`.

**Cause**: the repo stores the shell scripts without the execute bit, so the
checkout lands them as `-rw-rw-r--`. `docker-compose.yml` calls the entrypoint in
exec form (`entrypoint: ["/wise-edge/weda-cv-vlm/entrypoint.sh"]`), which requires
`+x` on the host file.

**Fix**:
```bash
chmod +x build.sh entrypoint.sh scripts/wise-bench
```
Re-apply after every fresh checkout until the +x bit is committed upstream
(`git update-index --chmod=+x …`).

---

## 10. Standard Debug Workflow

When nothing in §1–10 matches, run this in order:

```bash
# 1. Get the pipeline PID
PID=$(pgrep -f "python3 -m app.main" | head -1)
echo "PID=$PID, started: $(ps -o lstart= -p $PID), uptime: $(ps -o etime= -p $PID)"

# 2. Live thread stacks (install py-spy if absent: `pip install py-spy`)
sudo py-spy dump --pid $PID

# 3. Process environment (auth/config sanity)
sudo cat /proc/$PID/environ | tr '\0' '\n' | grep -E 'WEDA|HF|DISPLAY'

# 4. Recent kernel events
sudo dmesg -T | tail -100 | grep -iE 'oom|xid|cuda|nvidia|usb'

# 5. Container vs host bind mismatch
docker inspect weda-cv-vlm --format '{{.HostConfig.NetworkMode}}'
ss -tlnp | grep 9180

# 6. GPU state
sudo tegrastats --interval 1000 | head -5

# 7. Recent VLM activity
docker logs weda-cv-vlm 2>&1 | tail -200 | grep -iE 'vlm|reasoner|cache error|phase'

# 8. API health round-trip
curl -fsS http://127.0.0.1:9180/v1/health | jq
curl -fsS http://127.0.0.1:9180/v1/info | jq '.features'
```

If steps 1–8 all look normal but the pipeline misbehaves, capture a 30-second
profile and attach to the issue report:

```bash
sudo py-spy record -o /tmp/profile.svg --duration 30 --pid $PID
```

---

## 11. References

| Topic | Source |
|---|---|
| AWQ deprecation + migration to gptqmodel | [transformers#38078](https://github.com/huggingface/transformers/issues/38078) |
| Qwen-VL canonical chat-template recipe | [qwen-vl-utils](https://github.com/QwenLM/Qwen-VL/blob/main/qwen-vl-utils) |
| Jetson Thor onnxruntime-gpu source build | [NVIDIA forum 347563](https://forums.developer.nvidia.com/t/method-to-compile-onnxruntime-gpu-on-thor/347563) |
| Sm_110 + CUDA 13 ORT support | [microsoft/onnxruntime#27098](https://github.com/microsoft/onnxruntime/issues/27098) |
| Jetson Thor MoE perf reference | [NVIDIA forum 364663](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance) |
| RFC 6750 (bearer auth) | [datatracker.ietf.org/doc/html/rfc6750](https://datatracker.ietf.org/doc/html/rfc6750) |
| RFC 9457 (Problem Details) | [datatracker.ietf.org/doc/html/rfc9457](https://datatracker.ietf.org/doc/html/rfc9457) |
| WHATWG SSE | [html.spec.whatwg.org/multipage/server-sent-events.html](https://html.spec.whatwg.org/multipage/server-sent-events.html) |

---

## 12. When To Escalate

Open a ticket against the integration team if you observe any of:

1. CUDA Xid errors (`dmesg | grep Xid`) recurring after a clean restart
2. Container UMA pinned > 95 % at idle for > 10 min after a restart
3. Models load but `/v1/health` reports any of `vlm_loaded / embed_loaded / physical_loaded` as `false` despite the load step printing "ready"
4. wise-bench reports any **FAIL** (not WARN) row
5. VLM produces gibberish or repeats a single token (kernel quantization corruption)

Attach to the ticket:
- Output of §11 steps 1–8
- Last 500 lines of `docker logs weda-cv-vlm`
- The `pipeline.yaml` and `.env` (with `HF_TOKEN` redacted)
- `bench-report.md` if it exists
