#!/bin/bash
set -e

# ==============================================================================
# WEDA-CV VLM Pipeline — Container Entrypoint
#
# _pip_add       installs with --no-deps (default, protects base-image pins).
# _pip_add_full  installs with full deps (use when the base image lacks
#                both the package and its transitive deps).
# ==============================================================================

PIP_FLAGS="--no-cache-dir --no-deps"

# ── [1] Display ──────────────────────────────────────────────────────────────
echo "[INFO] DISPLAY=${DISPLAY:-<not set>}"
echo "[INFO] XAUTHORITY=${XAUTHORITY:-<not set>}"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[WARN] DISPLAY unset — GUI unavailable. Check build.sh X11 detection."
else
    DISPLAY_NUM="${DISPLAY##*:}"
    X11_SOCKET_PATH="/tmp/.X11-unix/X${DISPLAY_NUM}"
    if [[ ! -S "${X11_SOCKET_PATH}" ]]; then
        echo "[WARN] X11 socket missing: ${X11_SOCKET_PATH} — set headless: true in config."
    else
        echo "[INFO] X11 socket OK: ${X11_SOCKET_PATH}"
    fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
    echo "[WARN] XAUTHORITY unset — display auth will fail."
elif [[ ! -f "${XAUTHORITY}" ]]; then
    echo "[WARN] XAUTHORITY missing: ${XAUTHORITY} — volume mount likely broken."
else
    echo "[INFO] Xauthority OK: ${XAUTHORITY}"
fi

# ── [2] System Dependencies ──────────────────────────────────────────────────
PY_DEV_PKG="python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')-dev"
HEADER_DIR="/usr/include/$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"

if ! command -v gst-inspect-1.0 &> /dev/null \
   || ! gst-inspect-1.0 v4l2src &> /dev/null 2>&1 \
   || [[ ! -f "/usr/lib/aarch64-linux-gnu/gtk-2.0/modules/libcanberra-gtk-module.so" ]] \
   || [[ ! -f "${HEADER_DIR}/Python.h" ]]; then
    echo "[INFO] Hydrating system deps..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -q \
        libx11-6 libxext6 x11-utils x11-apps v4l-utils kmod \
        libcanberra-gtk-module libcanberra-gtk3-module at-spi2-core \
        gstreamer1.0-plugins-good gstreamer1.0-tools gstreamer1.0-plugins-base \
        gcc build-essential "${PY_DEV_PKG}" \
        > /dev/null 2>&1 \
        || apt-get install -y -q python3-dev > /dev/null 2>&1 || true
    echo "[INFO] System deps ready."
else
    echo "[INFO] System deps already present."
fi

if [[ -f "${HEADER_DIR}/Python.h" ]]; then
    echo "[INFO] Python.h present — Triton JIT will compile."
else
    echo "[WARN] Python.h missing — modelopt-quantized layers will error."
fi

# ── [3] Python Packages ──────────────────────────────────────────────────────
_pip_add() {
    # _pip_add <import_name> <pip_spec> [log_path] [warn_text]
    local mod="$1" spec="$2" log="${3:-/dev/null}" warn="${4:-}"
    if python3 -c "import ${mod}" &> /dev/null; then
        echo "[INFO] ${mod} already installed."
        return 0
    fi
    echo "[INFO] Installing ${spec}..."
    if pip install $PIP_FLAGS "${spec}" > "${log}" 2>&1; then
        echo "[INFO] ${spec} installed."
    else
        [[ -n "$warn" ]] && echo "[WARN] ${warn}"
        return 1
    fi
}

_pip_add_full() {
    local mod="$1" spec="$2" log="${3:-/dev/null}" warn="${4:-}"
    if python3 -c "import ${mod}" &> /dev/null; then
        echo "[INFO] ${mod} already installed."
        return 0
    fi
    echo "[INFO] Installing ${spec} (with deps)..."
    if pip install --no-cache-dir "${spec}" > "${log}" 2>&1; then
        echo "[INFO] ${spec} installed."
    else
        [[ -n "$warn" ]] && echo "[WARN] ${warn}"
        return 1
    fi
}

_pip_add modelopt 'nvidia-modelopt>=0.27' /tmp/modelopt.log \
    "modelopt install failed — bitsandbytes fallback active. See /tmp/modelopt.log"

# No modelopt HF-plugin patch step: the script it called was deleted from the
# bundle, and it only ever addressed a modelopt 0.21.x / transformers 5.5
# signature clash that modelopt >=0.27 does not have.

_pip_add timm timm /tmp/timm.log \
    "timm install failed — Nemotron-Nano-VL will fail to load. See /tmp/timm.log"

_pip_add_full open_clip open_clip_torch /tmp/open_clip.log \
    "open_clip_torch install failed — Nemotron-Nano-VL will fail to load. See /tmp/open_clip.log"

# einops: two-stage fallback for air-gapped or stale-index environments.
if ! python3 -c "import einops" &> /dev/null; then
    echo "[INFO] Installing einops..."
    if pip install $PIP_FLAGS "einops>=0.7,<0.9" > /tmp/einops.log 2>&1 \
       || pip install $PIP_FLAGS --index-url https://pypi.org/simple "einops>=0.7,<0.9" > /tmp/einops.log 2>&1; then
        echo "[INFO] einops installed."
    else
        echo "[WARN] einops install failed — Cosmos-Embed1 DISABLED. See /tmp/einops.log"
    fi
else
    echo "[INFO] einops already installed."
fi

_pip_add triton triton /tmp/triton.log \
    "triton install failed — ModelOpt will use slower path."

_pip_add bitsandbytes bitsandbytes /tmp/bnb.log \
    "bitsandbytes install failed — VLM will load in FP16."

if python3 -c "from gptqmodel import GPTQModel" &> /dev/null; then
    echo "[INFO] gptqmodel already importable."
else
    echo "[INFO] Trying gptqmodel==7.0.0 (--no-deps; numpy untouched)..."
    apt-get install -y -q libpcre3-dev > /dev/null 2>&1 || true
    pip install --no-cache-dir --no-deps gptqmodel==7.0.0 > /tmp/gptqmodel.log 2>&1 || true
    for _t in device-smi tokenicer logbar pypcre threadpoolctl defuser; do
        pip install --no-cache-dir --no-deps "$_t" >> /tmp/gptqmodel.log 2>&1 || true
    done
    echo "--- import check ---" >> /tmp/gptqmodel.log
    if python3 -c "from gptqmodel import GPTQModel" >> /tmp/gptqmodel.log 2>&1; then
        echo "[INFO] gptqmodel installed and importable."
    else
        pip uninstall -y gptqmodel > /dev/null 2>&1 || true
        echo "[WARN] gptqmodel not usable on this image — removed. AWQ checkpoints unavailable. See /tmp/gptqmodel.log"
    fi
fi

if ! python3 -c "import qwen_vl_utils" &> /dev/null; then
    echo "[INFO] Installing qwen-vl-utils..."
    if pip install $PIP_FLAGS 'qwen-vl-utils[decord]>=0.0.14' > /tmp/qvl.log 2>&1; then
        echo "[INFO] qwen-vl-utils installed."
    elif pip install $PIP_FLAGS 'qwen-vl-utils>=0.0.14' > /tmp/qvl.log 2>&1; then
        echo "[INFO] qwen-vl-utils installed (no decord — image-only OK)."
    else
        echo "[WARN] qwen-vl-utils install failed — Qwen3-VL crops will hit token-mismatch fallback. See /tmp/qvl.log"
    fi
else
    echo "[INFO] qwen-vl-utils already installed."
fi

_pip_add pyorbbecsdk pyorbbecsdk2 /tmp/orbbec.log \
    "pyorbbecsdk2 install failed — Orbbec depth unavailable."

_pip_add_full nvitop nvitop /tmp/nvitop.log \
    "nvitop install failed — GPU HUD will show N/A."

# ── [3a] HTTP API stack ──────────────────────────────────────────────────────
_pip_add_full fastapi fastapi /tmp/fastapi.log "HTTP API DISABLED."
_pip_add_full uvicorn 'uvicorn[standard]' /tmp/uvicorn.log "HTTP API DISABLED."
_pip_add_full sse_starlette sse-starlette /tmp/sse.log "VLM streaming DEGRADED."

# ── [3b] PII redaction (optional) ────────────────────────────────────────────
_pip_add_full presidio_analyzer presidio-analyzer /tmp/presidio.log \
    "presidio-analyzer install failed — falling back to regex set."

# ── [3c] ONNX Runtime GPU restore guard ──────────────────────────────────────
echo "[INFO] Verifying onnxruntime GPU providers..."
if python3 -c "
import onnxruntime as ort, sys
ps = ort.get_available_providers()
print(f'[ORT] version={ort.__version__} providers={ps}')
sys.exit(0 if 'CUDAExecutionProvider' in ps else 1)
" 2>/dev/null; then
    echo "[INFO] onnxruntime-gpu intact."
else
    echo "[WARN] onnxruntime CPU-only detected — searching for source-built GPU wheel..."
    _ORT_WHEEL=""
    if [[ -n "${WEDA_ORT_GPU_WHEEL:-}" && -f "${WEDA_ORT_GPU_WHEEL}" ]]; then
        _ORT_WHEEL="${WEDA_ORT_GPU_WHEEL}"
    else
        for _dir in /opt/wheels /wise-edge/wheels /root/wheels /workspace/wheels; do
            [[ -d "$_dir" ]] || continue
            for _candidate in "$_dir"/onnxruntime_gpu-*.whl; do
                [[ -f "$_candidate" ]] && { _ORT_WHEEL="$_candidate"; break 2; }
            done
        done
    fi
    if [[ -n "$_ORT_WHEEL" ]]; then
        echo "[INFO]   wheel: $_ORT_WHEEL"
        pip uninstall -y onnxruntime onnxruntime-gpu > /dev/null 2>&1 || true
        if pip install $PIP_FLAGS --force-reinstall "$_ORT_WHEEL" > /tmp/ort-gpu.log 2>&1 \
           && python3 -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
            echo "[INFO] onnxruntime-gpu restored."
        else
            echo "[WARN] Restore failed — see /tmp/ort-gpu.log"
        fi
    else
        echo "[WARN] No onnxruntime-gpu wheel in /opt/wheels, /wise-edge/wheels, /root/wheels, /workspace/wheels."
        echo "[WARN] Pipeline still works (TRT-direct path doesn't need ORT-GPU)."
        echo "[WARN] To restore: drop the wheel in one of those dirs, or set WEDA_ORT_GPU_WHEEL=<abs-path>."
    fi
fi

# ── [4] Ultralytics EULA Gate ────────────────────────────────────────────────
if ! python3 -c "import ultralytics" &> /dev/null; then
    echo "[INFO] Ultralytics EULA gate — ACCEPT_ULTRALYTICS_EULA=${ACCEPT_ULTRALYTICS_EULA:-<not set>}"
    if [[ "${ACCEPT_ULTRALYTICS_EULA:-}" != "true" ]]; then
        if [[ -t 0 ]]; then
            echo "----------------------------------------------------------------------"
            echo "  Ultralytics components required (BYOL)."
            echo "  Skip prompt: set ACCEPT_ULTRALYTICS_EULA=true in .env"
            echo "----------------------------------------------------------------------"
            printf "  Accept EULA and BYOL terms? [yes/no]: "
            read -r USER_ACCEPT
            USER_ACCEPT_LOWER=$(echo "$USER_ACCEPT" | tr '[:upper:]' '[:lower:]')
            [[ "$USER_ACCEPT_LOWER" == "yes" || "$USER_ACCEPT_LOWER" == "y" ]] \
                && ACCEPT_ULTRALYTICS_EULA=true \
                || { echo "[WARN] EULA declined — detection runs in mock mode."; ACCEPT_ULTRALYTICS_EULA=false; }
        else
            echo "[WARN] Non-interactive shell — set ACCEPT_ULTRALYTICS_EULA=true in .env."
            ACCEPT_ULTRALYTICS_EULA=false
        fi
    fi
    if [[ "${ACCEPT_ULTRALYTICS_EULA}" == "true" ]]; then
        echo "[INFO] Installing Ultralytics..."
        pip install $PIP_FLAGS "ultralytics>=8.4.0" "ultralytics-thop>=2.0.18"
        echo "[INFO] Ultralytics installed."
    fi
else
    echo "[INFO] Ultralytics already installed."
fi

# ── [5] Ultralytics auto-update guard ────────────────────────────────────────
if python3 -c "import ultralytics" &> /dev/null; then
    python3 -c "
import ultralytics.utils.checks as c, inspect
path = inspect.getfile(c)
with open(path) as f: content = f.read()
if 'WEDA-CV Protection' not in content:
    with open(path, 'a') as f:
        f.write('\n\n# WEDA-CV Protection: disable auto-update of dependencies\n')
        f.write('check_requirements = lambda *args, **kwargs: True\n')
    print('[INFO] Ultralytics auto-update guard installed.')
else:
    print('[INFO] Ultralytics auto-update guard already present.')
" || echo "[WARN] Auto-update guard failed — Ultralytics may overwrite system packages."
fi

# ── [6] PyYAML ───────────────────────────────────────────────────────────────
_pip_add yaml pyyaml /tmp/pyyaml.log "PyYAML install failed — config loader will fail."

# ── [7] VIC / GStreamer + PyGObject ──────────────────────────────────────────
if python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst" &> /dev/null; then
    echo "[INFO] PyGObject + GStreamer GIR already present."
else
    echo "[INFO] Installing PyGObject + GStreamer GIR..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /dev/null 2>&1 || true
    for _pkg in libgirepository-2.0-dev libgirepository1.0-dev \
                gobject-introspection libcairo2-dev pkg-config \
                gir1.2-glib-2.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0; do
        apt-get install -y -q "$_pkg" > /dev/null 2>&1 || true
    done
    if pip install --no-cache-dir PyGObject > /tmp/pygobject.log 2>&1 \
       && python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst" &> /dev/null; then
        echo "[INFO] PyGObject installed."
    else
        echo "[WARN] PyGObject unavailable — frame resize uses cv2 fallback. See /tmp/pygobject.log"
    fi
fi

if command -v gst-inspect-1.0 &> /dev/null && gst-inspect-1.0 nvvidconv &> /dev/null; then
    echo "[INFO] nvvidconv element available."
else
    echo "[WARN] nvvidconv element missing — frame resize uses cv2 fallback."
fi

if python3 -c "from weda_optim import Scheduler, Profiler, BufferPool" &> /dev/null; then
    echo "[INFO] weda_optim SDK OK."
else
    echo "[WARN] weda_optim SDK import failed — check volume mounts."
fi

# ── [8] Smoke Test ───────────────────────────────────────────────────────────
if python3 -c "from app.core.orchestrator import Orchestrator; print('[INFO] Pipeline imports OK')"; then
    echo "[INFO] Smoke test passed."
else
    echo "[WARN] Smoke test failed — check dependencies."
fi

# ── [9] Ready Sentinel ───────────────────────────────────────────────────────
touch /wise-edge/weda-cv-vlm/.ready
echo "[INFO] Environment ready."
echo "----------------------------------------------------------------------"

# ── [10] HuggingFace Offline ─────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$@"
