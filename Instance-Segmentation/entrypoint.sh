#!/bin/bash
set -uo pipefail

# ==============================================================================
# WEDA-CV Instance Segmentation Demo — Self-Hydrating Entrypoint
#
# Hydration is a best-effort, fail-safe operation: a failure here must never
# kill the container, or the operator loses the shell needed to diagnose it.
# Failures are recorded and reported; the .ready sentinel is created only when
# the environment can genuinely run the demo.
# ==============================================================================

readonly APP_DIR="/wise-edge/instance-segmentation"
readonly READY_FILE="${APP_DIR}/.ready"
# Written when hydration definitively cannot succeed, so build.sh can report the
# failure at once instead of polling out its whole timeout budget.
readonly FAILED_FILE="${APP_DIR}/.hydration-failed"

# BYOL packages install here rather than into the image venv. The path is backed
# by a named volume (see docker-compose.yml), so a licensed install survives
# container recreation and hydration runs once per device, not once per launch.
readonly BYOL_DIR="${WEDA_CV_BYOL_DIR:-/opt/weda-cv/byol}"
readonly ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.4.0}"
readonly ULTRALYTICS_THOP_VERSION="${ULTRALYTICS_THOP_VERSION:-2.0.18}"

# Clear any sentinel inherited from a previous run of this container. Doing this
# here — before hydration — is race-free; doing it from the host after the
# container starts can delete a sentinel this script has already written.
rm -f "${READY_FILE}" "${FAILED_FILE}"

HYDRATION_OK=true

# ── [1] Display Validation ────────────────────────────────────────────────────
echo "[INFO] DISPLAY=${DISPLAY:-<not set>}"
echo "[INFO] XAUTHORITY=${XAUTHORITY:-<not set>}"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[WARN] DISPLAY is not set — GUI output will be unavailable"
    echo "[WARN] Check that build.sh detected X11 correctly and DISPLAY was exported"
else
    DISPLAY_NUM="${DISPLAY##*:}"
    X11_SOCKET_PATH="/tmp/.X11-unix/X${DISPLAY_NUM}"

    if [[ ! -S "${X11_SOCKET_PATH}" ]]; then
        echo "[WARN] X11 socket not found: ${X11_SOCKET_PATH}"
        echo "[WARN] Available sockets: $(ls /tmp/.X11-unix/ 2>/dev/null || echo 'none')"
        echo "[WARN] GUI output will be unavailable — set show: false in config"
    else
        echo "[INFO] X11 socket confirmed: ${X11_SOCKET_PATH}"
    fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
    echo "[WARN] XAUTHORITY is not set — display auth will fail"
elif [[ ! -f "${XAUTHORITY}" ]]; then
    echo "[WARN] XAUTHORITY file not found: ${XAUTHORITY}"
    echo "[WARN] Volume mount may have failed — check build.sh detection output"
else
    echo "[INFO] Xauthority confirmed: ${XAUTHORITY}"
fi

# ── [2] System Dependencies ───────────────────────────────────────────────────
if ! command -v gst-inspect-1.0 &> /dev/null || \
   ! gst-inspect-1.0 v4l2src &> /dev/null 2>&1 || \
   [[ ! -f "/usr/lib/aarch64-linux-gnu/gtk-2.0/modules/libcanberra-gtk-module.so" ]]; then
    echo "[INFO] System dependencies missing — hydrating (this may take several minutes)..."
    export DEBIAN_FRONTEND=noninteractive

    APT_LOG="/tmp/weda-cv-apt.log"
    if apt-get update -qq > "${APT_LOG}" 2>&1 && \
       apt-get install -y -q \
            libx11-6 libxext6 x11-utils x11-apps v4l-utils kmod \
            libcanberra-gtk-module libcanberra-gtk3-module at-spi2-core \
            gstreamer1.0-plugins-good gstreamer1.0-tools gstreamer1.0-plugins-base \
            >> "${APT_LOG}" 2>&1; then
        echo "[INFO] System dependencies ready."
    else
        HYDRATION_OK=false
        echo "[FAIL] System dependency installation failed. Last 20 lines:"
        tail -20 "${APT_LOG}" 2>/dev/null | sed 's/^/         /'
        echo "[FAIL] Full log inside the container: ${APT_LOG}"
    fi
else
    echo "[INFO] System dependencies already present."
fi

# ── [3] Ultralytics EULA Gate ─────────────────────────────────────────────────
# Resolve BYOL packages through the same interpreter that runs the demo.
export PYTHONPATH="${BYOL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# pip --target's console-scripts (e.g. `yolo`) land in BYOL_DIR/bin, which is
# not on the image's default PATH — without this, the yolo CLI is unreachable
# even though the ultralytics package itself imports fine.
export PATH="${BYOL_DIR}/bin${PATH:+:${PATH}}"
printf 'export PATH="%s/bin${PATH:+:${PATH}}"\n' "${BYOL_DIR}" > /etc/profile.d/weda-byol.sh 2>/dev/null || true
[ -x "${BYOL_DIR}/bin/yolo" ] && ln -sf "${BYOL_DIR}/bin/yolo" /usr/local/bin/yolo 2>/dev/null || true

UL_INSTALLED="$(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null | tail -1 || true)"
if [[ "${UL_INSTALLED}" != "${ULTRALYTICS_VERSION}" ]]; then
    if [[ -n "${UL_INSTALLED}" ]]; then
        echo "[INFO] Ultralytics ${UL_INSTALLED} installed but ${ULTRALYTICS_VERSION} is pinned — reinstalling..."
    else
        echo "[INFO] Ultralytics not installed — checking EULA acceptance..."
    fi
    echo "[INFO] ACCEPT_ULTRALYTICS_EULA=${ACCEPT_ULTRALYTICS_EULA:-<not set>}"

    if [[ "${ACCEPT_ULTRALYTICS_EULA:-}" != "true" ]]; then
        if [[ -t 0 ]]; then
            echo "----------------------------------------------------------------------"
            echo "  Ultralytics components required (BYOL)."
            echo "  To skip this prompt: set ACCEPT_ULTRALYTICS_EULA=true in .env"
            echo "----------------------------------------------------------------------"
            printf "  Accept EULA and BYOL terms? [yes/no]: "
            read -r USER_ACCEPT
            USER_ACCEPT_LOWER=$(echo "$USER_ACCEPT" | tr '[:upper:]' '[:lower:]')
            if [[ "$USER_ACCEPT_LOWER" == "yes" || "$USER_ACCEPT_LOWER" == "y" ]]; then
                ACCEPT_ULTRALYTICS_EULA=true
            else
                echo "[WARN] EULA declined — Ultralytics not installed. Demo will fail."
                ACCEPT_ULTRALYTICS_EULA=false
            fi
        else
            echo "[WARN] Non-interactive shell and ACCEPT_ULTRALYTICS_EULA != true."
            echo "[WARN] Set ACCEPT_ULTRALYTICS_EULA=true in .env to enable auto-install."
            ACCEPT_ULTRALYTICS_EULA=false
        fi
    fi

    if [[ "${ACCEPT_ULTRALYTICS_EULA}" == "true" ]]; then
        echo "[INFO] Installing Ultralytics ${ULTRALYTICS_VERSION} into ${BYOL_DIR}..."
        mkdir -p "${BYOL_DIR}"

        # python3 -m pip (not bare pip) guarantees the install lands in the same
        # interpreter the import check above and the demo below actually use.
        PIP_LOG="/tmp/weda-cv-pip.log"
        if python3 -m pip install \
                --no-cache-dir \
                --retries 5 \
                --timeout 60 \
                --no-deps \
                --target "${BYOL_DIR}" \
                --upgrade \
                "ultralytics==${ULTRALYTICS_VERSION}" \
                "ultralytics-thop==${ULTRALYTICS_THOP_VERSION}" \
                > "${PIP_LOG}" 2>&1; then
            echo "[INFO] Ultralytics install completed."
        else
            echo "[FAIL] Ultralytics install failed. Last 20 lines:"
            tail -20 "${PIP_LOG}" 2>/dev/null | sed 's/^/         /'
            echo "[FAIL] Full log inside the container: ${PIP_LOG}"
        fi

        # Trust the import, not the installer's exit code — a partially unpacked
        # target directory can leave pip happy and the demo broken.
        UL_NOW="$(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null | tail -1 || true)"
        if [[ "${UL_NOW}" == "${ULTRALYTICS_VERSION}" ]]; then
            echo "[INFO] Ultralytics verified: ${UL_NOW}"
        else
            HYDRATION_OK=false
            echo "[FAIL] Ultralytics is '${UL_NOW:-not importable}' after install; pinned: ${ULTRALYTICS_VERSION}."
        fi
    else
        HYDRATION_OK=false
        echo "[FAIL] Ultralytics unavailable — the demo cannot run until the EULA is accepted."
    fi
else
    echo "[INFO] Ultralytics already installed at the pinned version: ${UL_INSTALLED}"
fi

# ── [4] Environment Protection Patch ─────────────────────────────────────────
if python3 -c "import ultralytics" &> /dev/null; then
    echo "[INFO] Applying environment protection patch..."
    python3 -c "
try:
    import ultralytics.utils.checks as c
    import inspect
    path = inspect.getfile(c)
    with open(path, 'r') as f:
        content = f.read()
    if 'WEDA-CV Protection' not in content:
        with open(path, 'a') as f:
            f.write('\n\n# WEDA-CV Protection: Disable auto-update of dependencies\n')
            f.write('check_requirements = lambda *args, **kwargs: True\n')
        print('[INFO] Protection patch applied.')
    else:
        print('[INFO] Protection patch already present — skipping.')
except Exception as e:
    print(f'[WARN] Protection patch failed: {e}')
    print('[WARN] Ultralytics may attempt to overwrite system packages.')
"
fi

# ── [5] Solutions Logging Patch ───────────────────────────────────────────────
# Resolve the installed location instead of assuming a venv path — Ultralytics
# now lives in BYOL_DIR, and the venv's Python version is not ours to hardcode.
SOLUTIONS_PATH=$(python3 -c \
    "import ultralytics.solutions.solutions as s, inspect; print(inspect.getfile(s))" \
    2>/dev/null || true)

if [[ -n "${SOLUTIONS_PATH}" && -f "${SOLUTIONS_PATH}" ]]; then
    echo "[INFO] Applying solutions logging patch..."
    sed -i \
        's/self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")/pass/g' \
        "${SOLUTIONS_PATH}"
fi

# ── [6] Signal Ready ──────────────────────────────────────────────────────────
if [[ "${HYDRATION_OK}" == "true" ]]; then
    touch "${READY_FILE}"
    echo "[INFO] Environment ready. Launching..."
else
    touch "${FAILED_FILE}"
    echo "[FAIL] Hydration incomplete — not signalling ready."
    echo "[FAIL] The container stays up so the environment can be inspected."
fi
echo "----------------------------------------------------------------------"

exec "$@"
