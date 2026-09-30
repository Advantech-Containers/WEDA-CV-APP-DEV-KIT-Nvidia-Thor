"""Camera backend factory — instantiate by config name.

Usage:
    backend = create_camera_backend(config, pool)
    backend.open()
    packet = backend.read()  # FramePacket with RGB + optional depth
    backend.close()

Supported backends:
    auto       — tries orbbec, then gstreamer, then v4l2
    v4l2       — USB/CSI via OpenCV V4L2 (proven 60 FPS)
    orbbec     — Orbbec SDK (RGB + depth unified)
    gstreamer  — GStreamer NVDEC hardware decode
    rtsp       — RTSP via PyAV with reconnect
    realsense  — Intel RealSense (stub — add realsense_backend.py)
    zed        — Stereolabs ZED (stub — add zed_backend.py)
"""

from __future__ import annotations

from typing import Optional

from app.io.camera_backend import CameraBackend
from app.utils.data_structures import PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()


def create_camera_backend(config: PipelineConfig, pool=None) -> Optional[CameraBackend]:
    """Create camera backend from config. Returns None on failure."""
    backend_name = getattr(config, "input_backend", "auto").lower()
    source = config.source.lower() if isinstance(config.source, str) else str(config.source)


    # Force RTSP if source is rtsp://
    if source.startswith("rtsp://"):
        from app.io.rtsp_backend import RTSPBackend
        return RTSPBackend(config, pool)

    # Explicit backend selection
    if backend_name == "v4l2":
        return _create_v4l2(config, pool)
    elif backend_name == "orbbec":
        return _create_orbbec(config, pool)
    elif backend_name == "gstreamer":
        return _create_gstreamer(config, pool)
    elif backend_name == "rtsp":
        from app.io.rtsp_backend import RTSPBackend
        return RTSPBackend(config, pool)
    elif backend_name == "realsense":
        return _create_stub("realsense", "pyrealsense2", "app.io.realsense_backend.RealSenseBackend")
    elif backend_name == "zed":
        return _create_stub("zed", "pyzed", "app.io.zed_backend.ZEDBackend")
    elif backend_name == "auto":
        return _create_auto(config, pool)
    else:
        logger.error("Unknown camera backend: %s", backend_name)
        return None


def _probe_orbbec() -> bool:
    """Return True only if an Orbbec device is physically connected and enumerable.

    Calls get_stream_profile_list() — the same call that fails at open() time —
    so we catch the SDK error here before committing to the Orbbec backend.

    IMPORTANT: the Pipeline object is explicitly deleted after the probe so the
    SDK releases its device handle.  Without del, the probe Pipeline holds a lock
    and the OrbbecBackend's own Pipeline() call fails with the same error.
    """
    try:
        from pyorbbecsdk import Pipeline, OBSensorType
        p = Pipeline()
        profiles = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        count = profiles.get_count() if profiles is not None else 0
        del p  # release device handle before OrbbecBackend opens its own Pipeline
        return count > 0
    except Exception:
        return False


def _create_auto(config: PipelineConfig, pool) -> Optional[CameraBackend]:
    """Auto-detect best available backend.

    For integer camera sources (USB):
      - If pyorbbecsdk2 installed AND an Orbbec device responds → Orbbec SDK
        (RGB + real depth + hardware-calibrated intrinsics)
      - Otherwise → V4L2 (RGB only, or depth via V4L2 sibling probe)
    For RTSP/file sources: try GStreamer NVDEC hardware decode, then V4L2.
    """
    source = config.source
    try:
        int(source)
        is_camera = True
    except (ValueError, TypeError):
        is_camera = False

    if is_camera:
        # Try Orbbec SDK first — but only if a device is actually connected
        try:
            from app.io.orbbec_handler import is_orbbec_available
            if is_orbbec_available():
                if _probe_orbbec():
                    logger.info("Auto-detect: Orbbec device found — using Orbbec SDK backend")
                    orbbec = _create_orbbec(config, pool)
                    if orbbec is not None:
                        return orbbec
                else:
                    logger.info(
                        "Auto-detect: pyorbbecsdk2 installed but no Orbbec device detected "
                        "— falling back to V4L2"
                    )
        except Exception:
            pass
        # V4L2 fallback: RGB + optional V4L2 depth sibling probe
        return _create_v4l2(config, pool)

    # RTSP/file: try GStreamer NVDEC hardware decode first
    try:
        from app.io.gstreamer_backend import is_nvdec_available
        if is_nvdec_available():
            backend = _create_gstreamer(config, pool)
            if backend:
                return backend
    except Exception:
        pass

    # Fall back to V4L2
    return _create_v4l2(config, pool)


def _create_v4l2(config, pool) -> CameraBackend:
    from app.io.v4l2_backend import V4L2Backend
    return V4L2Backend(config, pool)


def _create_orbbec(config, pool) -> Optional[CameraBackend]:
    try:
        from app.io.orbbec_backend import OrbbecBackend
        from app.io.orbbec_handler import is_orbbec_available
        if not is_orbbec_available():
            logger.error("Orbbec backend requested but pyorbbecsdk2 not installed")
            return None
        return OrbbecBackend(config, pool)
    except ImportError:
        logger.error("Orbbec backend not available")
        return None


def _create_gstreamer(config, pool) -> Optional[CameraBackend]:
    try:
        from app.io.gstreamer_backend import GStreamerBackend
        return GStreamerBackend(config, pool)
    except ImportError:
        logger.error("GStreamer backend not available")
        return None


def _create_stub(name: str, package: str, class_path: str) -> Optional[CameraBackend]:
    logger.error(
        "Camera backend '%s' requires '%s'. Install it and create %s",
        name, package, class_path,
    )
    return None
