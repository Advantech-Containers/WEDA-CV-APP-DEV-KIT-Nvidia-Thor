# ==============================================================================
# WEDA-CV Instance Segmentation Demo
# Base: WEDA-CV Platform (Thor)
# ==============================================================================

# Overridable so the pipeline can point the build at the branch's own
# Harbor project (weda-ai on develop, weda-release on release/*).
ARG BASE_IMAGE=harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="WEDA-CV Instance Segmentation Demo" \
      org.opencontainers.image.description="Ultralytics YOLO Instance Segmentation and Tracking" \
      com.advantech.layer="demo" \
      com.advantech.type="application"

WORKDIR /wise-edge/instance-segmentation
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Install X11 client libs for display support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    x11-utils \
  && rm -rf /var/lib/apt/lists/*

# Copy application source, config, and assets and wise-bench
COPY src/    /wise-edge/instance-segmentation/src/
COPY config/ /wise-edge/instance-segmentation/config/
COPY data/ /wise-edge/instance-segmentation/data/
COPY wise-bench /wise-edge/instance-segmentation/wise-bench

# Create output directory
RUN mkdir -p /wise-edge/instance-segmentation/output

# Default: run the object counter
CMD ["python3", "src/instance_segmentation.py", "--config", "config/config.yaml"]
