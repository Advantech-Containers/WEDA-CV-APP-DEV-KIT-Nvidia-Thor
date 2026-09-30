# WEDA-CV Object Counting Demo — Troubleshooting Guide

**Version:** 1.0.0  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Document Classification:** Production Support Reference

---

## How to Use This Guide

This document is organized by failure domain. Each section covers a specific subsystem (container lifecycle, display, camera/source, model/inference, configuration, output, environment patches). Issues are presented with four fields: symptom, root cause, diagnostic command, and resolution. All diagnostic commands are intended to be run inside the container unless noted as host-side.

---

## 1. Container Build and Lifecycle

### 1.1. `.env` File Missing or Incomplete

**Symptom:** `build.sh` exits with `Missing required vars in .env`.

**Root Cause:** The `.env` file does not exist in the project root, or one or more required variables (`WEDA_CV_IMAGE`, `CONTAINER_NAME`, `SERVICE_NAME`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`, `ACCEPT_ULTRALYTICS_EULA`) are missing or empty.

**Diagnostic:**
```bash
# Host-side
cat .env
```

**Resolution:** Create or repair the `.env` file using the template provided in the project. All six required variables must be present and non-empty.

### 1.2. Docker Daemon Not Running

**Symptom:** `build.sh` exits with `Docker daemon not running`.

**Diagnostic:**
```bash
# Host-side
sudo systemctl status docker
```

**Resolution:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

If the NVIDIA Container Runtime is not configured, the container will fail at launch with GPU errors. Verify: `docker info | grep -i runtime` — the output must include `nvidia`.

### 1.3. Image Pull Failure from Harbor Registry

**Symptom:** The "Synchronizing container image" step fails with a connection timeout or authentication error.

**Root Cause:** The host cannot reach `harbor.edgesync.cloud` due to network configuration, DNS failure, proxy settings, or expired credentials.

**Diagnostic:**
```bash
# Host-side
ping -c 3 harbor.edgesync.cloud
docker pull harbor.edgesync.cloud/weda-ai/weda-cv:thor-1.0.0
```

**Resolution:** Verify outbound HTTPS (port 443) access. If behind a corporate proxy, configure Docker's proxy settings in `/etc/systemd/system/docker.service.d/http-proxy.conf`. Contact Advantech support for renewed registry credentials if needed.

### 1.4. Hydration Timeout

**Symptom:** `build.sh` progress bar shows `TIMEOUT`. Container is running but not ready.

**Root Cause:** Hydration (`apt-get` plus the BYOL `pip install`) legitimately takes longer than the wait budget — ~70 s on a cold Jetson Thor / JP7 device, so the former 60-second budget expired on essentially every first launch.

**Diagnostic:**
```bash
# Host-side
docker logs -f weda-cv-object-counter
```

Watch for `[INFO] System dependencies ready` and `[INFO] Ultralytics verified`. The `.ready` sentinel appears only after every hydration step succeeds; on unrecoverable failure `.hydration-failed` appears instead and `build.sh` reports it immediately.

**Resolution:** The budget is now 300 s and configurable via `HYDRATION_TIMEOUT` in `.env`. Hydration is also cached — the licensed Ultralytics install lives in the `BYOL_VOLUME` named volume and the container is no longer force-removed each launch, so a repeat launch reuses the container (~1 s) and a full recreate skips the download (~70 s, apt only).

`build.sh` no longer opens the demo shell after failed hydration; running the demo mid-hydration was the source of the intermittent `ModuleNotFoundError` in §5.1. To reset a broken cache: `./build.sh --clean && ./build.sh`.

Do **not** pre-bake Ultralytics into the image — it is BYOL-licensed and must install only after the EULA gate.

### 1.5. Entrypoint Permission Denied

**Symptom:** Container exits immediately with `exec format error` or `permission denied`.

**Root Cause:** `entrypoint.sh` lacks execute permission or has Windows-style line endings (CRLF).

**Diagnostic:**
```bash
# Host-side
file entrypoint.sh
```

**Resolution:**
```bash
chmod +x entrypoint.sh
dos2unix entrypoint.sh
```

### 1.6. Volume Mount Conflicts After Host-Side File Changes

**Symptom:** Changes to `src/` or `config/` on the host are not reflected inside the container.

**Root Cause:** If files are replaced (moved and overwritten) rather than edited in-place, Docker's bind mount may point to the old inode.

**Resolution:** Restart the container: `./build.sh --stop && ./build.sh --run`. Editing files in-place avoids this issue.

---

## 2. X11 Display and GUI

### 2.1. `cannot open display` Error

**Symptom:** `cv2.imshow()` fails with `cannot open display ""` or the function is not implemented.

**Root Cause:** `DISPLAY` is not set inside the container, or the X11 socket is not mounted. This happens when `build.sh` was not used to launch the container or when the host has no active display session.

**Diagnostic:**
```bash
# Inside container
echo $DISPLAY
ls -la /tmp/.X11-unix/
```

**Resolution:** Set `show: false` in config for headless operation. To use GUI, re-launch via `build.sh` which auto-detects X11. For manual override:

```bash
# Host-side
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
./build.sh
```

### 2.2. Xauthority Cookie Mismatch

**Symptom:** Display is set and socket exists, but `cv2.imshow()` produces `Xlib: No protocol specified`.

**Root Cause:** The Xauthority cookie inside the container does not match the active display session (e.g., after a reboot or display manager restart).

**Diagnostic:**
```bash
# Inside container
xdpyinfo 2>&1 | head -5
```

**Resolution:** Restart the container to trigger fresh Xauthority detection: `./build.sh --stop && ./build.sh`. As a workaround: `xhost +local:docker` on the host.

### 2.3. Multiple X11 Sockets — Wrong Display Selected

**Symptom:** Window appears on the wrong monitor, or `build.sh` selects an inactive VNC session socket.

**Root Cause:** `build.sh` uses the first X11 socket found in `/tmp/.X11-unix/` (sorted lexicographically).

**Resolution:** Override before launching: `export DISPLAY=:0 && ./build.sh`.

### 2.4. `dbind-WARNING: Couldn't connect to accessibility bus`

**Symptom:** Harmless warning from GTK when the display window opens.

**Root Cause:** AT-SPI2 accessibility daemon socket is unavailable inside the container. Does not affect rendering or inference.

**Resolution:** No action required. The warning is cosmetic. Suppressed by `NO_AT_SPI=1` in `.env`.

---

## 3. Camera and Video Source

### 3.1. `Cannot open video source: 0`

**Symptom:** Application exits when configured to use a USB camera.

**Root Cause:** `/dev/video0` does not exist inside the container — camera was not connected before container startup, or the device mount failed.

**Diagnostic:**
```bash
# Inside container
ls -la /dev/video*
v4l2-ctl -d /dev/video0 --all
```
If the camera nodes are discoverable and still not able to get the camera inference 
```bash
# On host
sudo systemctl restart nvargus-daemon
```
This should clear the gstreamer blockage and the camera should be availabe for inference.

**Resolution:** Connect the camera before starting the container. The `docker-compose.yml` mounts `/dev:/dev:rw` with `privileged: true`. If the camera was hot-plugged, restart the container. If the device exists but cannot open, check for competing processes: `fuser /dev/video0` on the host.

### 3.2. Video File Not Found

**Symptom:** `Cannot open video source: assets/test.mp4`.

**Root Cause:** The video file does not exist at the specified path. The application checks: (1) exact path, (2) `assets/` subdirectory.

**Diagnostic:**
```bash
# Inside container
ls -la assets/
```

**Resolution:** Place the video file in `assets/` on the host. The volume mount makes it available inside the container. Verify the file name in `config.yaml` matches exactly (case-sensitive).

### 3.3. RTSP Stream Disconnects

**Symptom:** `Empty frame received from source` and graceful exit after running for some time.

**Root Cause:** RTSP connection dropped. Reconnection logic is not implemented in v1.0.0.

**Resolution:** Implement a wrapper script that restarts the application on exit:

```bash
while true; do
    python3 src/object_counter.py --config config/config.yaml
    echo "[INFO] Restarting in 5s..."
    sleep 5
done
```

### 3.4. Webcam Runs at Lower FPS Than Expected

**Symptom:** USB camera delivers fewer FPS than its hardware specification.

**Root Cause:** OpenCV may default to the GStreamer backend on JetPack builds, which adds buffering overhead. The object counting application uses the default `cv2.VideoCapture(source)` without specifying a backend.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print(f'Backend: {cap.getBackendName()}')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
cap.release()
"
```

**Resolution:** If the backend shows `GSTREAMER`, force V4L2 by modifying the `cv2.VideoCapture` call in `object_counter.py` to `cv2.VideoCapture(source, cv2.CAP_V4L2)`. This is a known limitation of the v1.0.0 application — a config-driven backend selection is available in the Instance Segmentation Demo engine (v3.0.0) and may be backported in a future release.

---


### 3.5. Wrong camera node: grayscale/depth image, or the index stops working after a replug

**Symptom:** The demo runs but the picture is grayscale or looks like a depth map instead of color video; or a `source` index that worked before returns `Cannot open video source: N` / `can't open camera by index` after the camera was unplugged and reconnected.

**Root Cause:** Multi-sensor cameras (for example the Advantech CAM-5530) expose several `/dev/video*` nodes — RGB, depth and IR streams plus metadata nodes — and only some of them deliver frames. Picking the first node that opens can select the depth or IR sensor. Additionally, V4L2 node numbering is assigned at enumeration time: after a replug the same physical camera can come back under different indices.

**Diagnostic:** Identify the RGB node by its advertised pixel formats, not by trial and error:

```bash
# Inside container
for d in /dev/video*; do echo "== $d"; v4l2-ctl -d $d --list-formats 2>/dev/null; done
```

The RGB stream advertises `MJPG` and/or `YUYV`. Depth nodes advertise `Z16`; IR nodes advertise `GREY`/`Y8`. Nodes listing no capture formats are metadata nodes.

**Resolution:** Set `source` in the config to the node that lists `MJPG`/`YUYV`. After any replug, re-run the diagnostic and update `source` if the index moved — the numbering is not stable across reconnects.

## 4. Configuration Validation

### 4.1. Invalid Task Name

**Symptom:** `Invalid task 'X'. Must be one of: line_count, polygon_count, region_count`.

**Root Cause:** The `task` key in `config.yaml` contains a value not in the valid set.

**Resolution:** Set `task` to one of: `line_count`, `region_count`, `polygon_count`.

### 4.2. Insufficient ROI Points

**Symptom:** `Task 'region_count' requires at least 4 ROI points, but 2 were provided`.

**Root Cause:** The `roi.points` list has fewer entries than the selected task requires. Line counting requires 2, region counting requires 4, polygon counting requires 3.

**Resolution:** Add the required number of points to `roi.points` in `config.yaml`.

### 4.3. Missing ROI Points

**Symptom:** `ROI points not defined in config. At least 2 points are required`.

**Root Cause:** The `roi.points` key is missing or empty in `config.yaml`.

**Resolution:** Add a `roi:` section with `points:` to the config. See `config/config.schema.md` for examples.

### 4.4. Empty or Missing Configuration File

**Symptom:** `Configuration file not found` or `Configuration file is empty`.

**Root Cause:** The path passed to `--config` does not exist, or the file contains no valid YAML content.

**Diagnostic:**
```bash
# Inside container
cat config/config.yaml
```

**Resolution:** Verify the file path and contents. The default path is `config/config.yaml`. If using a custom path, provide the full relative or absolute path: `python3 src/object_counter.py --config config/my_config.yaml`.

---

## 5. Model and Inference

### 5.1. `ModuleNotFoundError: No module named 'ultralytics'`

**Symptom:** Application fails at import.

**Root Cause:** One of two things:

1. **The demo was started before hydration finished** — `build.sh` timed out but opened the shell anyway and advertised `Run demo:` while `pip install` was still running. Whether it failed depended on how fast the command was typed, which made the error look intermittent. `build.sh` now refuses to open the shell until `.ready` exists.
2. **The EULA was not accepted**, so Ultralytics was never installed. Hydration now treats this as a hard failure instead of silently signalling ready.

**Diagnostic:**
```bash
# Inside container
python3 -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
echo $ACCEPT_ULTRALYTICS_EULA
ls /wise-edge/object-counting/.ready /wise-edge/object-counting/.hydration-failed
```

Ultralytics is expected to resolve from the BYOL volume at `/opt/weda-cv/byol`, which is on `PYTHONPATH`.

**Resolution:** Set `ACCEPT_ULTRALYTICS_EULA=true` in `.env`, relaunch, and wait for `[  OK]  Environment hydrated.` before running the demo. If the cache is broken: `./build.sh --clean && ./build.sh`. To install by hand inside a running container, target the BYOL directory so it persists and matches the demo's interpreter:

```bash
# Inside container
python3 -m pip install --no-deps --target /opt/weda-cv/byol \
    "ultralytics==8.4.24" "ultralytics-thop==2.0.18"
```

### 5.2. Model Auto-Download Fails

**Symptom:** `ConnectionError` when the application attempts to download a `.pt` model.

**Root Cause:** No outbound internet access from the container.

**Resolution:** Pre-download the model on a machine with internet and place it in `assets/` on the host.

### 5.3. Non-PT Model File Not Found

**Symptom:** `Model file 'model.onnx' not found. Non-PT formats must be provided as a local file`.

**Root Cause:** ONNX, TensorRT, and TorchScript models cannot be auto-downloaded. The application requires them as local files.

**Resolution:** Place the model file in the project root or `assets/` directory. Set `model.path` to the correct filename.

### 5.4. CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory` during model loading or inference.

**Root Cause:** Model memory footprint exceeds available unified memory. On Jetson Thor, GPU and CPU share LPDDR5X.

**Resolution (in order of impact):** Reduce `model.imgsz` (e.g., `480` or `320`). Set `model.half: true` for FP16. Filter classes to reduce tracked objects. Stop competing GPU processes. Monitor with `tegrastats` on the host.

### 5.5. FP16 on CPU Device

**Symptom:** `RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'`.

**Root Cause:** `model.half: true` is set but inference is running on CPU. The `model.to("cuda")` call in `object_counter.py` should place the model on GPU, but if CUDA is unavailable, PyTorch falls back to CPU.

**Diagnostic:**
```bash
# Inside container
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Resolution:** Ensure `runtime: nvidia` is in `docker-compose.yml` and the NVIDIA Container Toolkit is installed. If running on CPU intentionally, set `model.half: false`.

### 5.6. Tracker File Not Found

**Symptom:** `FileNotFoundError` for `botsort.yaml` or `bytetrack.yaml`.

**Root Cause:** Tracker config not found in Ultralytics' default directory.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import ultralytics; from pathlib import Path
print(list((Path(ultralytics.__file__).parent / 'cfg' / 'trackers').glob('*.yaml')))
"
```

**Resolution:** Use `botsort.yaml` or `bytetrack.yaml` — both ship with Ultralytics.

---

## 6. Video Output

### 6.1. Video Writer Warning — Output Not Saved

**Symptom:** `Could not open video writer — output will not be saved`.

**Root Cause:** The output directory does not exist, is not writable, or the `mp4v` codec is unavailable.

**Diagnostic:**
```bash
# Inside container
ls -la output/
python3 -c "
import cv2
w = cv2.VideoWriter('/tmp/test.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
print(f'Opened: {w.isOpened()}')
w.release()
"
```

**Resolution:** Ensure `output/` exists and is writable. On the host: `mkdir -p output`. The `docker-compose.yml` mounts `./output` as read-write. If the codec test fails, the OpenCV build may lack FFmpeg support — this is atypical for the WEDA-CV base image but can occur if OpenCV was rebuilt.

### 6.2. Output Video Sync Issues

**Symptom:** Output video appears to have timing inconsistencies — some frames are duplicated or the playback speed does not match the source.

**Root Cause:** The application uses `vid_stride` to skip frames. When a frame is skipped, the raw (unannotated) frame is written directly to the output. The video writer uses the source's reported FPS for timing, which may not match the actual processing rate.

**Resolution:** This is expected behavior when `vid_stride > 1`. For frame-accurate output, set `vid_stride: 1`.

---

## 7. Ultralytics Environment Patches

### 7.1. Ultralytics Overwrites System Packages

**Symptom:** After running the application, other Python packages (NumPy, OpenCV, Pillow) are unexpectedly upgraded or downgraded.

**Root Cause:** Ultralytics' `check_requirements()` auto-installs or upgrades dependencies to its pinned versions, potentially overwriting the WEDA-CV base image's tested packages.

**Diagnostic:**
```bash
# Inside container
python3 -c "
import ultralytics.utils.checks as c; import inspect
path = inspect.getfile(c)
with open(path) as f: content = f.read()
print('PROTECTED' if 'WEDA-CV Protection' in content else 'UNPROTECTED')
"
```

**Resolution:** The `entrypoint.sh` applies a protection patch that replaces `check_requirements` with a no-op. If the patch is missing (e.g., Ultralytics was reinstalled), apply manually:

```bash
python3 -c "
import ultralytics.utils.checks as c; import inspect
path = inspect.getfile(c)
with open(path, 'a') as f:
    f.write('\n\n# WEDA-CV Protection: Disable auto-update of dependencies\n')
    f.write('check_requirements = lambda *args, **kwargs: True\n')
"
```

### 7.2. Solutions Logger Spam

**Symptom:** Every frame produces `Ultralytics Solutions: ✅ {CFG}` in the console.

**Root Cause:** The `solutions.ObjectCounter` logs its full configuration on every invocation. The solutions logging patch in `entrypoint.sh` targets the specific `sed` pattern in the solutions source file.

**Resolution:** The entrypoint applies the patch automatically. If it did not apply (file path changed with an Ultralytics update), locate and patch manually:

```bash
SOLUTIONS_PATH=$(python3 -c "import ultralytics.solutions.solutions as s; import inspect; print(inspect.getfile(s))")
sed -i 's/self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")/pass/g' "$SOLUTIONS_PATH"
```

The solutions logging patch path in `entrypoint.sh` is hardcoded to `/opt/venv/lib/python3.12/site-packages/ultralytics/solutions/solutions.py`. If the venv path or Python version differs on a future base image, update this path in `entrypoint.sh`.

---

## 8. Performance Issues

### 8.1. FPS Is Lower Than Expected

**Symptom:** Application FPS is significantly below the `wise-bench` inference baseline.

**Root Cause:** The ObjectCounter includes annotation, video encoding, and display overhead on top of raw inference. The most common causes of low FPS are: `model.half: false` (FP32 is ~2x slower), large `imgsz`, synchronous video writer blocking, and `show: true` display overhead.

**Resolution:** Set `model.half: true`. Reduce `model.imgsz` (e.g., `480`). Set `show: false` for headless. Increase `model.vid_stride` to skip frames. Filter classes to reduce tracker overhead.

### 8.2. First Frame Is Slow

**Symptom:** The first 1–3 seconds of processing show very low FPS, then performance stabilizes.

**Root Cause:** CUDA context initialization occurs on the first inference pass. PyTorch's CUDA allocator, cuDNN autotuner, and TensorRT engine builder all run initialization routines on the first forward pass.

**Resolution:** This is expected behavior. The first-frame latency is a one-time cost and does not affect steady-state throughput. The application's FPS counter averages over all frames, so the first-frame penalty diminishes as more frames are processed.

---

## 9. Diagnostic Commands Reference

```bash
# GPU status
nvidia-smi
tegrastats                              # Host-side

# CUDA verification
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# OpenCV build info
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -E "CUDA|GStreamer|V4L|FFMPEG"

# Camera capabilities
v4l2-ctl -d /dev/video0 --list-formats-ext

# X11 display verification
echo $DISPLAY
ls -la /tmp/.X11-unix/
xdpyinfo | head -20

# Ultralytics version and protection patch status
python3 -c "import ultralytics; print(ultralytics.__version__)"
python3 -c "
import ultralytics.utils.checks as c; import inspect
with open(inspect.getfile(c)) as f: content = f.read()
print('PROTECTED' if 'WEDA-CV Protection' in content else 'UNPROTECTED')
"

# Configuration validation (dry run)
python3 -c "
import yaml
cfg = yaml.safe_load(open('config/config.yaml'))
print(f'Task: {cfg.get(\"task\")}')
print(f'ROI points: {len(cfg.get(\"roi\", {}).get(\"points\", []))}')
print(f'Model: {cfg.get(\"model\", {}).get(\"path\")}')
print(f'Source: {cfg.get(\"source\")}')
"

# Network connectivity
ping -c 3 github.com
```

---

## 10. Reporting Issues

When reporting issues to the WISE-Edge Platform Division, include: the complete application log from startup through the failure (including the banner, config validation output, and full traceback), the output of `nvidia-smi` and `tegrastats`, the `config/config.yaml` file in use, the JetPack version (`cat /etc/nv_tegra_release`), and the container image tag.

---

© 2026 Advantech Corporation. All rights reserved.
