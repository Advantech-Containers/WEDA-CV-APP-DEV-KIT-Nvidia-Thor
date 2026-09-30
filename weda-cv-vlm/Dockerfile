# syntax=docker/dockerfile:1.4
# ══════════════════════════════════════════════════════════════════════
# Component : WEDA Edge CV-VLM Pipeline (L3 Application)
# Target    : NVIDIA Jetson Thor (sm_110, Blackwell, 128 GB UMA)
# Base      : WEDA GPU-Passthrough Runtime
#
# Reference template: build the pipeline into a self-contained image, or
# use it as the starting point for your own application. Day-to-day the
# bundle runs from the published image via ./build.sh — this file is for
# when you want to bake the code in rather than bind-mount it.
# ══════════════════════════════════════════════════════════════════════

ARG BASE_IMAGE=harbor.edgesync.cloud/weda-ai/weda-gpu-passthrough:1.0.0-thor

# ══════════════════════════════════════════════════════════════════════
# STAGE 1: TEST
# ══════════════════════════════════════════════════════════════════════
FROM ${BASE_IMAGE} AS test

WORKDIR /tmp/test-harness

COPY weda_optim/ weda_optim/
COPY config/ config/
COPY app/ app/
COPY tests/ tests/

RUN --mount=type=cache,target=/root/.cache/pip \
    set -e; \
    echo "=== SMOKE TEST ==="; \
    python3 -c "\
import torch, cv2, numpy; \
from weda_optim import BufferPool, Scheduler, Profiler; \
from app.core.orchestrator import Orchestrator; \
print(f'Pipeline OK: PyTorch {torch.__version__}, OpenCV {cv2.__version__}, CUDA {torch.version.cuda}'); \
print(f'SDK OK: BufferPool={BufferPool.__name__}, Scheduler={Scheduler.__name__}')"; \
    echo "=== PYTEST ==="; \
    pip install -q pytest pyyaml 2>/dev/null; \
    python3 -m pytest tests/ -q --tb=short; \
    echo "=== ALL TESTS PASSED ==="

# ══════════════════════════════════════════════════════════════════════
# STAGE 2: Final
# ══════════════════════════════════════════════════════════════════════
FROM ${BASE_IMAGE} AS Final

SHELL ["/bin/bash", "-c"]

# ── Environment (single merged layer) ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:${PATH}" \
    PIPELINE_CONFIG="/wise-edge/weda-cv-vlm/config/pipeline.yaml" \
    HF_HOME="/root/.cache/huggingface"

WORKDIR /wise-edge/weda-cv-vlm

# ── X11 + GStreamer runtime deps ──
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libx11-6 libxext6 x11-utils \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-tools \
        python3-gi gir1.2-gst-plugins-base-1.0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages installed at runtime via entrypoint.sh ──
# ── Entrypoint ──
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# ── SDK (optimization layer) ──
COPY weda_optim/ weda_optim/

# ── Config ──
COPY config/ config/

# ── Application code ──
# Replace or extend this with your own application to build on top of
# the pipeline; keep the WORKDIR so PIPELINE_CONFIG still resolves.
COPY app/ app/

# ── Cleanup ──
RUN find /wise-edge/weda-cv-vlm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /wise-edge/weda-cv-vlm -name "*.pyc" -delete 2>/dev/null; \
    true

# ── Healthcheck ──
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD test -f /wise-edge/weda-cv-vlm/.ready || exit 1

ENTRYPOINT ["/wise-edge/weda-cv-vlm/entrypoint.sh"]
CMD ["/bin/bash"]

# ── Metadata ──
LABEL org.opencontainers.image.title="WEDA Edge CV-VLM Pipeline" \
      org.opencontainers.image.description="Jetson Thor optimized CV pipeline with UMA memory optimization (sm_110)" \
      com.advantech.layer="WEDA-CV-VLM" \
      com.advantech.base="weda-gpu-passthrough:1.0.0-thor" \
      com.advantech.target="jetson-thor"
