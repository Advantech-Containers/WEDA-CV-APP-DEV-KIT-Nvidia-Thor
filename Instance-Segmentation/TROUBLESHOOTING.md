# WEDA-CV Instance Segmentation + Tracking — Troubleshooting Guide

**Engine Version:** 3.0.0  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Document Classification:** Production Support Reference  

---

## How to Use This Guide

This document is organized by failure domain. Each section covers a specific subsystem of the project (container lifecycle, display, camera, inference, output, memory). Within each section, issues are presented with four fields: the observable symptom, the root cause, a diagnostic command sequence to confirm the cause, and the resolution. All diagnostic commands are intended to be run inside the container unless explicitly noted as host-side.

---

## 1. Container Build and Lifecycle

### 1.1. `.env` File Missing or Incomplete

**Symptom:** `build.sh` exits immediately with `Missing required vars in .env: WEDA_CV_IMAGE CONTAINER_NAME ...`.

**Root Cause:** The `.env` file does not exist in the project root, or one or more required variables are missing. `build.sh` validates the presence of `WEDA_CV_IMAGE`, `CONTAINER_NAME`, `SERVICE_NAME`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`, and `ACCEPT_ULTRALYTICS_EULA` before proceeding.

**Diagnostic:**
```bash
# Host-side
cat .env
```

Verify that all six required variables are present and non-empty.

**Resolution:** Create or repair the `.env` file. A minimal valid example:

```env
WEDA_CV_IMAGE=harbor.edgesync.cloud/weda-ai/weda-cv:thor-1.0.0
CONTAINER_NAME=weda-cv-iseg
SERVICE_NAME=instance-segmentation
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
LD_LIBRARY_PATH=/usr/local/lib
QT_X11_NO_MITSHM=1
NO_AT_SPI=1
DBUS_SESSION_BUS_ADDRESS=/dev/null
ACCEPT_ULTRALYTICS_EULA=true
LOG_MAX_SIZE=10m
LOG_MAX_FILE=3
```

### 1.2. Docker Daemon Not Running

**Symptom:** `build.sh` exits with `Docker daemon not running`.

**Root Cause:** The Docker Engine service is not active on the host.

**Diagnostic:**
```bash
# Host-side
sudo systemctl status docker
```

**Resolution:**
```bash
# Host-side
sudo systemctl start docker
sudo systemctl enable docker
```

If Docker is installed but the NVIDIA Container Runtime is not configured, the container will fail at launch with GPU-related errors. Verify the runtime:

```bash
# Host-side
docker info | grep -i runtime
```

The output must include `nvidia`. If not, install `nvidia-container-toolkit` per the NVIDIA Container Runtime documentation for JetPack 7.x.

### 1.3. Image Pull Failure from Harbor Registry

**Symptom:** `build.sh` step "Synchronizing container image" fails with a connection timeout or authentication error referencing `harbor.edgesync.cloud`.

**Root Cause:** The host cannot reach the Advantech Harbor registry. This may be due to network configuration, DNS resolution failure, proxy settings, or expired registry credentials.

**Diagnostic:**
```bash
# Host-side
ping -c 3 harbor.edgesync.cloud
docker login harbor.edgesync.cloud
docker pull harbor.edgesync.cloud/weda-ai/weda-cv:thor-1.0.0
```

**Resolution:** Verify outbound HTTPS (port 443) access to `harbor.edgesync.cloud`. If behind a corporate proxy, configure Docker's proxy settings in `/etc/systemd/system/docker.service.d/http-proxy.conf`. If credentials are expired, contact Advantech support for renewed registry access.

### 1.4. Hydration Timeout During Container Startup

**Symptom:** `build.sh` progress bar reaches the timeout threshold and displays `TIMEOUT`. The container is running but `docker logs` shows `apt-get update` or `pip install` still in progress.

**Root Cause:** Hydration (`apt-get` plus the BYOL `pip install`) legitimately takes longer than the wait budget. On a cold Jetson Thor / JP7 device the full sequence measures ~70 s, so the former 60-second budget expired on essentially every first launch.

**Diagnostic:**
```bash
# Host-side
docker logs -f weda-cv-instance-segmentation
```

Watch for `[INFO] System dependencies ready` and `[INFO] Ultralytics verified`. The `.ready` sentinel is created only after every hydration step succeeds; if hydration cannot succeed, `.hydration-failed` is created instead and `build.sh` reports the failure immediately rather than waiting out the budget.

**Resolution:** The budget is now 300 s and is configurable — raise `HYDRATION_TIMEOUT` in `.env` for very slow links. Hydration is also cached: the licensed Ultralytics install lives in the `BYOL_VOLUME` named volume and the container is no longer force-removed on each launch, so a repeat launch reuses the running container (~1 s) and even a full container recreate skips the download (~70 s, apt only).

> `build.sh` no longer opens the demo shell after a failed hydration. Previously it printed `Run demo:` regardless, and running that command mid-hydration produced `ModuleNotFoundError: No module named 'ultralytics'` — see §4.1.

If hydration is genuinely broken, reset the cache and retry:

```bash
# Host-side
./build.sh --clean && ./build.sh
```

Do **not** pre-bake Ultralytics into the image: it is BYOL-licensed and must be installed only after the EULA gate. Note also that `docker-compose.yml` declares `image:` with no `build:` section, so `docker compose build` never applied this `Dockerfile` — the demo runs from the pulled base image plus the mounted sources.

### 1.5. Entrypoint Permission Denied

**Symptom:** Container exits immediately with `exec format error` or `permission denied` on `entrypoint.sh`.

**Root Cause:** The `entrypoint.sh` file lacks execute permission or has Windows-style line endings (CRLF).

**Diagnostic:**
```bash
# Host-side
file entrypoint.sh
ls -la entrypoint.sh
```

The output should show `Bourne-Again shell script` (not `with CRLF line terminators`) and the `x` execute bit.

**Resolution:**
```bash
# Host-side
chmod +x entrypoint.sh
dos2unix entrypoint.sh   # if line endings are CRLF
```

### 1.6. Volume Mount Conflicts After Host-Side File Changes

**Symptom:** Changes to `src/`, `config/`, or `assets/` on the host are not reflected inside the running container, or the container shows stale file contents.

**Root Cause:** The `docker-compose.yml` mounts `./src`, `./config`, and `./assets` as read-only bind mounts. However, if the container was started before the host-side files were updated and the filesystem cache has not flushed, the container may serve stale inodes. Additionally, if files are replaced (moved and overwritten rather than edited in-place), Docker's bind mount may still point to the old inode.

**Diagnostic:**
```bash
# Inside container
cat /wise-edge/instance-segmentation/src/instance_segmentation.py | head -5
# Compare against host-side version
```

**Resolution:** Restart the container to force a fresh bind mount:

```bash
# Host-side
./build.sh --stop
./build.sh --run
```

Editing files in-place (using a text editor that writes to the same file) rather than replacing them (which creates a new inode) avoids this issue entirely.

---

## 2. X11 Display and GUI

### 2.1. `cannot open display` Error

**Symptom:** `cv2.imshow()` raises `error: (-2:Unspecified error) The function is not implemented` or the process exits with `cannot open display ""`.

**Root Cause:** The `DISPLAY` environment variable is not set inside the container, or the X11 socket is not mounted. This happens when `build.sh` was not used to launch the container (so X11 auto-detection did not run), or when the host has no active display session.

**Diagnostic:**
```bash
# Inside container
echo $DISPLAY
ls -la /tmp/.X11-unix/
```

`DISPLAY` should contain a value such as `:0` or `:1`. The `/tmp/.X11-unix/` directory should contain a socket file matching the display number (e.g., `X0`).

**Resolution:** If running headless (no physical display), set `visualization.show: False` in `config.yaml`. The engine will process frames and write output without attempting GUI rendering.

If a display is connected but `DISPLAY` is empty inside the container, re-launch using `build.sh` which auto-detects X11:

```bash
# Host-side
./build.sh --stop
./build.sh
```

If the host display session uses a non-standard socket or auth path, export them before running `build.sh`:

```bash
# Host-side
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
./build.sh
```

### 2.2. Xauthority Cookie Mismatch

**Symptom:** The engine starts, `DISPLAY` is set, the X11 socket exists, but `cv2.imshow()` produces `Xlib: connection to ":0" refused by server. Xlib: No protocol specified`.

**Root Cause:** The Xauthority cookie file mounted inside the container does not match the active display session's cookie. This occurs when the display manager regenerates cookies (e.g., after a reboot or GDM restart) but the container still mounts the old file.

**Diagnostic:**
```bash
# Inside container
xdpyinfo 2>&1 | head -5
```

If the output shows "No protocol specified" or "unable to open display", the cookie is stale.

```bash
# Host-side — verify which Xauthority the display manager is using
ps aux | grep -E '[Xx]org' | grep -oP '(?<=-auth )\S+'
```

**Resolution:** Restart the container to trigger fresh Xauthority detection by `build.sh`:

```bash
# Host-side
./build.sh --stop
./build.sh
```

If detection continues to fail, grant unrestricted local display access as a workaround:

```bash
# Host-side (less secure, suitable for development)
xhost +local:docker
```

### 2.3. Multiple X11 Sockets — Wrong Display Selected

**Symptom:** The engine opens a window but it appears on the wrong monitor, or `build.sh` selects a socket that belongs to an inactive VNC session.

**Root Cause:** `build.sh` uses the first X11 socket found in `/tmp/.X11-unix/` (sorted lexicographically). If multiple sockets exist (e.g., `X0` from GDM and `X1` from a VNC server), it may select the wrong one.

**Diagnostic:**
```bash
# Host-side
ls -la /tmp/.X11-unix/
```

**Resolution:** Override the display before launching:

```bash
# Host-side
export DISPLAY=:0
./build.sh
```

### 2.4. `dbind-WARNING: Couldn't connect to accessibility bus`

**Symptom:** The warning `dbind-WARNING **: Couldn't connect to accessibility bus: Failed to connect to socket /run/user/1000/at-spi/bus_1` appears in the log when the display window opens.

**Root Cause:** The AT-SPI2 accessibility daemon socket is not available inside the container. This is a harmless warning from the GTK library used by OpenCV's highgui module. It does not affect display rendering, video output, or inference.

**Diagnostic:** No action required. This is cosmetic.

**Resolution:** The warning can be suppressed by setting `NO_AT_SPI=1` in the `.env` file (already pre-configured in the default `.env`). If the warning persists, ensure the environment variable is being passed through `docker-compose.yml`:

```yaml
environment:
  - NO_AT_SPI=${NO_AT_SPI}
```

### 2.5. `QObject::moveToThread` or Segfault in `cv2.imshow`

**Symptom:** The engine crashes with a Qt-related error or segfault when `visualization.show: True`, particularly when multiple threads are active (threaded camera, async writer).

**Root Cause:** OpenCV's highgui module (Qt backend) requires all GUI calls to originate from the main thread. The engine already calls `cv2.imshow()` from the main thread, but if the environment has a non-standard Qt build or conflicting Qt environment variables, threading interactions may cause instability.

**Diagnostic:**
```bash
# Inside container
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gui
```

Check whether the build uses GTK or Qt for the highgui backend.

**Resolution:** Set `QT_X11_NO_MITSHM=1` in `.env` (already pre-configured). If crashes persist, set `visualization.show: False` and rely on the output video file for visual inspection. This is the recommended approach for production deployments.

---

## 3. Camera and Video Source

### 3.1. `Cannot open video source: 0`

**Symptom:** The engine exits with `Cannot open video source: 0` when configured to use a USB camera.

**Root Cause:** The camera device node (`/dev/video0`) does not exist inside the container, or the V4L2 driver cannot initialize the device.

**Diagnostic:**
```bash
# Inside container
ls -la /dev/video*
v4l2-ctl -d /dev/video0 --all
```

If `/dev/video*` does not exist, the camera was not connected before container startup, or the device mount failed.

If the camera nodes are discoverable and still not able to get the camera inference 
```bash
# On host
sudo systemctl restart nvargus-daemon
```
This should clear the gstreamer blockage and the camera should be availabe for inference.

**Resolution:** Ensure the camera is physically connected before starting the container. The `docker-compose.yml` mounts `/dev:/dev:rw` and uses `privileged: true`, which provides access to all host device nodes. If the camera was hot-plugged after container start, restart the container:

```bash
# Host-side
./build.sh --stop
./build.sh --run
```

If the device exists but OpenCV still cannot open it, the V4L2 driver may be in use by another process:

```bash
# Host-side
fuser /dev/video0
```

Kill any competing process or use a different device index.

### 3.2. GStreamer Backend Throttles Camera to 20 FPS

**Symptom:** A USB camera that supports 30 FPS according to `v4l2-ctl --list-formats-ext` only delivers 20 FPS inside the container. The log shows `cap_gstreamer.cpp` warnings.

**Root Cause:** OpenCV defaulted to the GStreamer capture backend rather than V4L2. GStreamer adds pipeline setup, format negotiation, and internal buffering that reduces effective throughput on USB cameras. This is a known behavior on JetPack OpenCV builds where GStreamer is the compiled default.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
backend = cap.getBackendName()
print(f'Backend: {backend}')
cap.release()
"
```

If the output shows `GSTREAMER`, the GStreamer backend is being used.

**Resolution:** Set `camera.backend: "v4l2"` in `config.yaml`. This forces `cv2.VideoCapture(source, cv2.CAP_V4L2)`, bypassing GStreamer entirely. Also set `camera.fourcc: "MJPG"` to use the camera's hardware MJPG encoder, which reduces USB bandwidth consumption and allows full frame-rate delivery over USB 2.0.

### 3.3. Camera FPS Lower Than Expected Despite V4L2 Backend

**Symptom:** With `camera.backend: "v4l2"` configured, the camera still delivers fewer FPS than the hardware specification.

**Root Cause:** The camera may be in YUYV (uncompressed) mode, which saturates USB 2.0 bandwidth at higher resolutions. At 640×480 YUYV, each frame is ~614 KB. USB 2.0's effective throughput of ~35 MB/s supports only ~57 FPS at this resolution, but USB hub sharing, cable quality, and controller overhead can reduce this below 30 FPS.

**Diagnostic:**
```bash
# Inside container
v4l2-ctl -d /dev/video0 --get-fmt-video
```

Check the `Pixel Format` field. If it shows `YUYV`, the camera is transmitting uncompressed frames.

**Resolution:** Set `camera.fourcc: "MJPG"` in `config.yaml`. MJPG frames are typically 30–50 KB each, reducing USB bandwidth by 10–20× and allowing full frame-rate delivery. Verify that the camera supports MJPG:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext | grep MJPG
```

### 3.4. RTSP Stream Disconnects Mid-Inference

**Symptom:** The engine logs `Empty frame from source` and exits after running successfully for some period on an RTSP stream.

**Root Cause:** The RTSP connection was dropped by the camera, the network, or a timeout. The current engine version (3.0.0) does not implement reconnection logic. When `cap.read()` returns `False`, the engine exits gracefully.

**Diagnostic:** Check network stability to the camera:

```bash
# Inside container
ping -c 10 <camera-ip>
```

**Resolution:** This is a known limitation of v3.0.0 (documented in the Release Notes). For production RTSP deployments, implement a wrapper script that restarts the engine on exit:

```bash
while true; do
    python3 src/instance_segmentation.py --config config/rtsp_app.yaml
    echo "[INFO] Engine exited. Reconnecting in 5s..."
    sleep 5
done
```

Automatic reconnection with exponential backoff is planned for a future release.

### 3.5. Video File Not Found

**Symptom:** The engine logs `Cannot open video source: assets/test.mp4` and exits.

**Root Cause:** The video file does not exist at the specified path. The engine resolves `source` paths in this order: (1) exact path, (2) relative to `PROJECT_ROOT` (the parent of `src/`), (3) inside `assets/`. If the file is not found at any of these locations, `cv2.VideoCapture` receives a nonexistent path and fails to open.

**Diagnostic:**
```bash
# Inside container
ls -la assets/
ls -la /wise-edge/instance-segmentation/assets/
```

**Resolution:** Place the video file in the `assets/` directory on the host. The `docker-compose.yml` mounts `./assets` into the container at `/wise-edge/instance-segmentation/assets/`. Verify the file name in `config.yaml` matches exactly (case-sensitive on Linux).

---

### 3.6. Wrong camera node: grayscale/depth image, or the index stops working after a replug

**Symptom:** The demo runs but the picture is grayscale or looks like a depth map instead of color video; or a `source` index that worked before returns `Cannot open video source: N` / `can't open camera by index` after the camera was unplugged and reconnected.

**Root Cause:** Multi-sensor cameras (for example the Advantech CAM-5530) expose several `/dev/video*` nodes — RGB, depth and IR streams plus metadata nodes — and only some of them deliver frames. Picking the first node that opens can select the depth or IR sensor. Additionally, V4L2 node numbering is assigned at enumeration time: after a replug the same physical camera can come back under different indices.

**Diagnostic:** Identify the RGB node by its advertised pixel formats, not by trial and error:

```bash
# Inside container
for d in /dev/video*; do echo "== $d"; v4l2-ctl -d $d --list-formats 2>/dev/null; done
```

The RGB stream advertises `MJPG` and/or `YUYV`. Depth nodes advertise `Z16`; IR nodes advertise `GREY`/`Y8`. Nodes listing no capture formats are metadata nodes.

**Resolution:** Set `source` in the config to the node that lists `MJPG`/`YUYV`. After any replug, re-run the diagnostic and update `source` if the index moved — the numbering is not stable across reconnects.

## 4. Model and Inference

### 4.1. `ModuleNotFoundError: No module named 'ultralytics'`

**Symptom:** The engine fails at import time with `ModuleNotFoundError: No module named 'ultralytics'`.

**Root Cause:** One of two things:

1. **The demo was started before hydration finished.** This was the common case: `build.sh` timed out at 60 s but opened the shell anyway and advertised `Run demo:`, while `pip install ultralytics` was still running inside the container. Whether the demo failed depended purely on how quickly the command was typed, which is why the error looked intermittent. `build.sh` now refuses to open the shell until `.ready` exists.
2. **The EULA was not accepted**, so `entrypoint.sh` never installed Ultralytics — it installs only when `ACCEPT_ULTRALYTICS_EULA=true`. Hydration now treats this as a hard failure instead of silently signalling ready.

**Diagnostic:**
```bash
# Inside container
python3 -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
echo $ACCEPT_ULTRALYTICS_EULA
ls /wise-edge/instance-segmentation/.ready /wise-edge/instance-segmentation/.hydration-failed
```

Ultralytics is expected to resolve from the BYOL volume at `/opt/weda-cv/byol`, which is on `PYTHONPATH`.

**Resolution:** Set `ACCEPT_ULTRALYTICS_EULA=true` in `.env` on the host and relaunch. Wait for `[  OK]  Environment hydrated.` before running the demo:

```bash
# Host-side
./build.sh --stop
./build.sh
```

If hydration left a broken cache, purge it and start clean:

```bash
# Host-side
./build.sh --clean && ./build.sh
```

To install by hand inside a running container, target the BYOL directory so the install persists and matches the interpreter used by the demo:

```bash
# Inside container
python3 -m pip install --no-deps --target /opt/weda-cv/byol \
    "ultralytics==8.4.0" "ultralytics-thop==2.0.18"
```

### 4.2. Model Auto-Download Fails (No Internet)

**Symptom:** `ConnectionError` or `urllib3.exceptions.MaxRetryError` when the engine attempts to download a model file that is not present locally.

**Root Cause:** The container has no outbound internet access, and the model file specified in `model.path` was not pre-loaded into the project directory or `assets/`.

**Diagnostic:**
```bash
# Inside container
ping -c 3 github.com
ls -la /wise-edge/instance-segmentation/*.pt
ls -la /wise-edge/instance-segmentation/assets/*.pt
```

**Resolution:** Pre-download the model file on a machine with internet access and place it in the project root or `assets/` directory on the host. The volume mount makes it immediately available inside the container.

```bash
# Host-side (on a machine with internet)
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt
cp yolo26n-seg.pt /path/to/instance-segmentation/assets/
```

### 4.3. CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory. Tried to allocate X MiB` during model loading, warmup, or inference.

**Root Cause:** The model's memory footprint exceeds available GPU/unified memory. On Jetson Thor, GPU and CPU share the same LPDDR5X pool. Competing system processes, other containers, or desktop compositors consume shared memory.

**Diagnostic:**
```bash
# Host-side
tegrastats
# Or inside container
python3 -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()>>20}MB, Reserved: {torch.cuda.memory_reserved()>>20}MB')"
```

**Resolution (in order of impact):**

Reduce `model.imgsz` in `config.yaml`. Dropping from `640` to `480` reduces memory consumption by approximately 44%. Dropping to `320` reduces it by approximately 75%.

Set `model.retina_masks: False`. Retina masks require full-resolution mask tensors that scale with frame dimensions.

Filter classes to reduce tracked objects. Set `classes: [0, 2]` rather than `null` to limit detections.

Stop other GPU-consuming processes on the host. Desktop compositors, other inference containers, or Jupyter kernels may be consuming shared memory.

If using TensorRT, export with a workspace limit: `model.export(format='engine', half=True, workspace=4)` (workspace in GB).

### 4.4. `half` Enabled on CPU Device

**Symptom:** `RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'` or similar error when `model.half: True` and `model.device: "cpu"`.

**Root Cause:** FP16 (half-precision) inference is only supported on CUDA devices. The PyTorch CPU backend does not implement FP16 convolution kernels.

**Resolution:** Set `model.half: False` when using `model.device: "cpu"`. FP16 should only be enabled with `cuda` or `cuda:N` devices.

### 4.5. Tracker File Not Found

**Symptom:** `FileNotFoundError` referencing `botsort.yaml` or `bytetrack.yaml`.

**Root Cause:** The tracker configuration file is not found in the Ultralytics package's default tracker directory. This can happen if the Ultralytics installation is incomplete or a non-standard tracker name is specified.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import ultralytics
from pathlib import Path
tracker_dir = Path(ultralytics.__file__).parent / 'cfg' / 'trackers'
print(list(tracker_dir.glob('*.yaml')))
"
```

**Resolution:** Use either `botsort.yaml` or `bytetrack.yaml` — these are the two trackers shipped with Ultralytics. If a custom tracker configuration is needed, provide the full path to the YAML file in the `tracker:` config key.

### 4.6. TensorRT Engine Incompatibility

**Symptom:** `RuntimeError` or `TensorRT: engine file version mismatch` when loading a `.engine` file.

**Root Cause:** TensorRT engine files are not portable across TensorRT versions, CUDA versions, or GPU architectures. An engine exported on one Jetson device or JetPack version cannot be loaded on another.

**Resolution:** Always export the TensorRT engine on the target device inside the same container:

```bash
# Inside container
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n-seg.pt')
model.export(format='engine', half=True, imgsz=640)
"
```

This produces a `yolo26n-seg.engine` file compatible with the current TensorRT and CUDA versions. Set `model.path: "yolo26n-seg.engine"` in `config.yaml`.

---

## 5. Video Output

### 5.1. `Failed to open video writer`

**Symptom:** The engine exits with `Failed to open video writer: /wise-edge/instance-segmentation/output/segmented_output.avi`.

**Root Cause:** The output directory does not exist, is not writable, or the specified codec is not available in the container's OpenCV build.

**Diagnostic:**
```bash
# Inside container
ls -la output/
python3 -c "
import cv2
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
w = cv2.VideoWriter('/tmp/test.avi', fourcc, 30, (640, 480))
print(f'Opened: {w.isOpened()}')
w.release()
"
```

**Resolution:** Ensure the `output/` directory exists and is writable. The `docker-compose.yml` mounts `./output` from the host as read-write. If the directory was deleted on the host, recreate it:

```bash
# Host-side
mkdir -p output
```

If the codec test fails, use `MJPG` which is universally available in all OpenCV builds. The `mp4v` and `XVID` codecs require FFmpeg integration which may not be present in all JetPack OpenCV builds.

### 5.2. Output Video Is Shorter Than Input or Has Jumps

**Symptom:** The output video file has fewer frames than the source, or playback shows visible discontinuities.

**Root Cause:** The async video writer's queue filled up and frames were silently dropped. This happens when the MJPG encoder thread cannot keep pace with the inference rate, particularly at high FPS or high resolution.

**Diagnostic:** Check for a discrepancy between `Frames processed` in the log and the output video's frame count:

```bash
# Inside container
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames output/segmented_output.avi
```

Compare with the `Frames processed` value in the engine's summary log.

**Resolution:** Increase `output.queue_size` in `config.yaml` (e.g., `128` or `256`). If drops persist, switch to synchronous writing by setting `output.async_writer: False`. This guarantees all frames are written but will reduce inference FPS because encoding blocks the main thread. For benchmarking or live demo scenarios where recording fidelity is not critical, the async writer with occasional drops is acceptable.

### 5.3. Output File Size Is Unexpectedly Large

**Symptom:** A 30-second video at 1280×720 produces a 100+ MB output file.

**Root Cause:** MJPG is a per-frame JPEG codec with no inter-frame compression. File sizes are approximately `(frame_width × frame_height × 0.15) × frames_per_second × duration_seconds` bytes.

**Resolution:** This is expected behavior for MJPG. For smaller files, use `output.codec: "mp4v"` which applies inter-frame compression (if available in the OpenCV build). Note that `mp4v` encoding is slower and may cause async writer queue pressure. For production recording, consider disabling in-engine recording and using an external pipeline (GStreamer, FFmpeg) to encode from the display output or a shared memory buffer.

---

## 6. UMA Memory Manager

### 6.1. `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`

**Symptom:** The engine crashes at startup during `MemoryManager` initialization.

**Root Cause:** The PyTorch version on the target device uses `total_memory` instead of `total_mem` for the CUDA device properties attribute. This attribute was renamed across PyTorch versions.

**Resolution:** This was fixed in engine v3.0.0 (the attribute is now `total_memory`). If the error persists, update `memory.py` line 81 to match the attribute name in the installed PyTorch version. Verify the correct name:

```bash
python3 -c "print(dir(torch.cuda.get_device_properties(0)))" | tr ',' '\n' | grep total
```

### 6.2. Memory Manager Logs `disabled (no CUDA or config off)`

**Symptom:** The log shows `Memory manager: disabled` even though the device has a CUDA GPU.

**Root Cause:** Either `performance.memory.enabled` is set to `False` in the config, or `torch.cuda.is_available()` returns `False` inside the container.

**Diagnostic:**
```bash
# Inside container
python3 -c "import torch; print(torch.cuda.is_available())"
```

If this returns `False`, the NVIDIA Container Runtime is not correctly configured, or the `runtime: nvidia` directive in `docker-compose.yml` is not being applied.

**Resolution:** Verify that `docker-compose.yml` includes both `runtime: nvidia` and the `deploy.resources.reservations.devices` block. Verify that `nvidia-smi` works inside the container:

```bash
# Inside container
nvidia-smi
```

If `nvidia-smi` fails, the NVIDIA Container Runtime is not injecting the GPU driver into the container. Refer to the NVIDIA Container Toolkit installation guide for JetPack 7.x.

### 6.3. Pinned Memory Allocation Failure

**Symptom:** `RuntimeError: CUDA error: out of memory` during `PinnedFrameBuffer` creation at startup, before inference begins.

**Root Cause:** CUDA pinned memory reserves physical pages that cannot be swapped. On memory-constrained systems, the pinned buffer allocation (which requires `height × width × 3` bytes of page-locked memory plus a matching GPU tensor) may fail.

**Diagnostic:**
```bash
# Host-side
free -h
tegrastats
```

**Resolution:** Disable pinned buffers:

```yaml
performance:
  memory:
    enabled: True
    pinned_buffers: False       # Fall back to standard numpy buffers
    prealloc_output: False
```

The engine will function normally without pinned buffers, using standard pageable memory. The performance benefit of pinned buffers is primarily in latency consistency (P99 frame time) rather than peak throughput.

---

## 7. Ultralytics and Environment Patches

### 7.1. Ultralytics Overwrites System Packages

**Symptom:** After running the engine for the first time, other Python packages (e.g., NumPy, OpenCV, Pillow) are upgraded or downgraded unexpectedly, causing compatibility failures.

**Root Cause:** Ultralytics includes a `check_requirements()` function that auto-installs or upgrades dependencies to match its pinned versions. This can overwrite the WEDA-CV base image's pre-installed packages which are specifically tested against JetPack's CUDA, cuDNN, and TensorRT libraries.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import ultralytics.utils.checks as c
import inspect
path = inspect.getfile(c)
with open(path) as f:
    content = f.read()
print('Protection patch present' if 'WEDA-CV Protection' in content else 'PATCH MISSING')
"
```

**Resolution:** The `entrypoint.sh` applies an environment protection patch that replaces `check_requirements` with a no-op. If the patch is not present (e.g., the entrypoint did not run, or Ultralytics was reinstalled), apply it manually:

```bash
# Inside container
python3 -c "
import ultralytics.utils.checks as c
import inspect
path = inspect.getfile(c)
with open(path, 'a') as f:
    f.write('\n\n# WEDA-CV Protection: Disable auto-update of dependencies\n')
    f.write('check_requirements = lambda *args, **kwargs: True\n')
print('Patch applied:', path)
"
```

### 7.2. Solutions Logger Spam

**Symptom:** Every frame produces a verbose `Ultralytics Solutions: ✅ {CFG}` log line, flooding the console.

**Root Cause:** Ultralytics' solutions module logs its full configuration on every invocation. The engine v3.0.0 does not use the solutions API (it uses `model.track()` directly), so this issue should not occur. However, if the engine is extended or if the solutions module is imported indirectly, the logging may reappear.

**Resolution:** The `entrypoint.sh` patches the solutions logging line with `pass`. If the patch did not apply (the path `/opt/venv/lib/python3.12/site-packages/ultralytics/solutions/solutions.py` does not exist in the container), suppress it manually:

```bash
# Inside container
SOLUTIONS_PATH=$(python3 -c "import ultralytics.solutions.solutions as s; import inspect; print(inspect.getfile(s))")
sed -i 's/self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")/pass/g' "$SOLUTIONS_PATH"
```

---

## 8. Performance Issues

### 8.1. FPS Is Lower Than Expected on Video Files

**Symptom:** The engine achieves significantly less than 34 FPS on 720p video files on Jetson Thor.

**Root Cause:** Multiple factors can contribute. The most common are: `retina_masks: True` (adds ~8 ms per frame), synchronous video writer (adds ~13 ms per frame), large `imgsz` (higher resolution = longer inference), and thermal throttling.

**Diagnostic:** Enable `debug.timing: True` and examine the component breakdown:

```
┣ Track: 15.4ms | Annotate: 11.3ms | Write: 0.0ms | Display: 1.5ms
```

If `Track` dominates, the bottleneck is inference. If `Write` is non-zero and `output.async_writer: True` is set, the async queue is full. If the overall FPS degrades over time, thermal throttling is likely.

**Resolution by component:**

For `Track` > 20 ms: reduce `model.imgsz` (e.g., `480`), use a smaller model (e.g., `yolo26n-seg.pt` instead of `yolo26s-seg.pt`), or increase `model.vid_stride` to skip frames.

For `Annotate` > 5 ms: set `model.retina_masks: False`, or disable masks entirely with `visualization.show_masks: False`.

For `Write` > 0 ms with async enabled: increase `output.queue_size` or set `output.enabled: False`.

For thermal throttling: monitor with `tegrastats` on the host. If clock speeds are dropping, improve the device's cooling or reduce workload.

### 8.2. Inference FPS Varies Widely (Jitter)

**Symptom:** The timing breakdown shows `Track` latency varying between 12 ms and 40 ms across reporting intervals.

**Root Cause:** CUDA kernel launch overhead and memory allocation jitter. Without warmup, the first inference pass includes CUDA context initialization (~2–3 seconds). Without pinned memory, per-frame `cudaMalloc`/`cudaFree` introduces variable latency. Garbage collection pauses from Python's cyclic GC can also cause sporadic spikes.

**Resolution:** Ensure `performance.warmup: True` and `performance.memory.enabled: True` in the config. For lowest jitter in production, disable Python GC during the inference loop by adding `gc.disable()` before the loop and `gc.enable()` after (this is an advanced customization).

---

## 9. Diagnostic Commands Reference

The following commands are useful for debugging inside the container.

```bash
# GPU status and memory
nvidia-smi
tegrastats                              # Host-side only — run in a separate terminal

# CUDA verification
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# OpenCV build info (codecs, backends, CUDA support)
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -E "CUDA|GStreamer|V4L|FFMPEG|GTK|QT"

# Camera capabilities
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --all

# X11 display verification
xdpyinfo | head -20
echo $DISPLAY
ls -la /tmp/.X11-unix/

# Ultralytics installation and version
python3 -c "import ultralytics; print(ultralytics.__version__)"

# Check environment protection patch
python3 -c "
import ultralytics.utils.checks as c; import inspect
path = inspect.getfile(c)
with open(path) as f: content = f.read()
print('PROTECTED' if 'WEDA-CV Protection' in content else 'UNPROTECTED')
"

# Network connectivity (for model downloads, registry access)
ping -c 3 github.com
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/simple/

# Container resource usage
docker stats weda-cv-iseg --no-stream

# Check for device permission issues
ls -la /dev/video*
ls -la /dev/nvidia*
```

---

## 10. Reporting Issues

When reporting issues to the WISE-Edge Platform Division, include the following information:

The complete engine log from startup through the failure, including the banner, configuration path, model resolution, video properties, memory manager status, and the full traceback if applicable. The output of `nvidia-smi` and `tegrastats` at the time of failure. The `config.yaml` file in use. The JetPack version (`cat /etc/nv_tegra_release`), the container image tag, and the engine version from the startup banner.

---

© 2026 Advantech Corporation. All rights reserved.
