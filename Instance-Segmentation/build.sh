#!/usr/bin/env bash
# ==============================================================================
# WEDA-CV Instance Segmentation Demo — Build & Launch Script
# Usage:
#   ./build.sh              # Pull, start, and enter the container
#   ./build.sh --run        # Start only (skip pull, use existing image)
#   ./build.sh --shell      # Attach to already-running container
#   ./build.sh --stop       # Stop and remove the container
#   ./build.sh --clean      # Stop and also purge the cached BYOL install
#   ./build.sh --help
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Path Anchoring ────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────
log_info()    { echo -e "\e[1;36m[INFO]\e[0m  $1"; }
log_success() { echo -e "\e[1;32m[  OK]\e[0m  $1"; }
log_warn()    { echo -e "\e[1;33m[WARN]\e[0m  $1"; }
log_error()   { echo -e "\e[1;31m[FAIL]\e[0m  $1" >&2; exit 1; }

# ── Parse Flags ───────────────────────────────────────────────────────────────
MODE="build"
for arg in "$@"; do
    case "$arg" in
        --run)    MODE="run" ;;
        --shell)  MODE="shell" ;;
        --stop)   MODE="stop" ;;
        --clean)  MODE="clean" ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  (no flag)   Pull the container image, start it, and open a shell"
            echo "  --run       Start the container (skip pull, use existing image)"
            echo "  --shell     Attach a shell to the already-running container"
            echo "  --stop      Stop and remove the container"
            echo "  --clean     Stop, remove, and purge the cached BYOL install"
            echo "              (forces a full re-download on the next launch)"
            echo "  --help      Show this help message"
            exit 0 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╔═╗╦  ╦   ╔═╗╔╗ ╦╔═╗╔═╗╔╦╗  ╔═╗╔═╗╦ ╦╔╗╔╔╦╗╔═╗╦═╗
   ║║║║╣  ║║╠═╣───║  ╚╗╔╝   ║ ║╠╩╗║║╣ ║   ║   ║  ║ ║║ ║║║║ ║ ║╣ ╠╦╝
   ╚╩╝╚═╝═╩╝╩ ╩   ╚═╝ ╚╝    ╚═╝╚═╚╝╚═╝╚═╝ ╩   ╚═╝╚═╝╚═╝╝╚╝ ╩ ╚═╝╩╚═
                         Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

# ── Load Static Config from .env ──────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

[[ -f "${ENV_FILE}" ]] || log_error ".env not found at ${ENV_FILE}"

set -a
source "${ENV_FILE}"
set +a

# ── Static Variable Validation ────────────────────────────────────────────────
REQUIRED_STATIC=(
    WEDA_CV_IMAGE
    CONTAINER_NAME
    SERVICE_NAME
    NVIDIA_VISIBLE_DEVICES
    NVIDIA_DRIVER_CAPABILITIES
    ACCEPT_ULTRALYTICS_EULA
    BYOL_VOLUME
    ULTRALYTICS_VERSION
    ULTRALYTICS_THOP_VERSION
)

MISSING=()
for var in "${REQUIRED_STATIC[@]}"; do
    [[ -z "${!var:-}" ]] && MISSING+=("${var}")
done
[[ ${#MISSING[@]} -gt 0 ]] && log_error "Missing required vars in .env: ${MISSING[*]}"

# ── Handle --stop / --clean ───────────────────────────────────────────────────
if [[ "${MODE}" == "stop" || "${MODE}" == "clean" ]]; then
    echo -e "\033[1;34m[1/1] Stopping container...\033[0m"
    if docker ps -a -q -f name="${CONTAINER_NAME}" | grep -q .; then
        export X11_SOCKET=${X11_SOCKET:-/tmp/.X11-unix}
        export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}
        export DISPLAY=${DISPLAY:-:0}
        docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down --remove-orphans
        log_success "Container stopped and removed."
    else
        log_warn "Container '${CONTAINER_NAME}' is not running."
    fi

    if [[ "${MODE}" == "clean" ]]; then
        if docker volume inspect "${BYOL_VOLUME}" &> /dev/null; then
            docker volume rm "${BYOL_VOLUME}" > /dev/null \
                && log_success "BYOL cache purged: ${BYOL_VOLUME}" \
                || log_warn "Could not remove volume '${BYOL_VOLUME}' — still in use?"
            log_info "The next launch will re-download Ultralytics."
        else
            log_warn "BYOL cache '${BYOL_VOLUME}' does not exist — nothing to purge."
        fi
    fi
    exit 0
fi

# ── Handle --shell ────────────────────────────────────────────────────────────
if [[ "${MODE}" == "shell" ]]; then
    echo -e "\033[1;34m[1/1] Attaching to running container...\033[0m"
    if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
        log_success "Entering container '${CONTAINER_NAME}'..."
        docker exec -it "${CONTAINER_NAME}" /bin/bash || true
    else
        log_error "Container '${CONTAINER_NAME}' is not running. Start it first with: ./build.sh"
    fi
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: preflight → X11 → EULA → build/pull → launch → shell
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS=5
[[ "${MODE}" == "run" ]] && TOTAL_STEPS=4

# ── [1/N] Pre-flight Checks ───────────────────────────────────────────────────
echo -e "\033[1;34m[1/${TOTAL_STEPS}] Pre-flight checks...\033[0m"

for cmd in docker jq; do
    command -v "${cmd}" &> /dev/null || log_error "${cmd} not found."
done
docker compose version &> /dev/null || log_error "docker compose not found."
log_success "System utilities (Docker, Compose, JQ) ready."

docker info &> /dev/null || log_error "Docker daemon not running."
log_success "Docker daemon running."

mkdir -p "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/test_logs"
log_success "Output directories ready."

# ── [2/N] X11 Display — Auto-Detection ───────────────────────────────────────
echo -e "\n\033[1;34m[2/${TOTAL_STEPS}] Detecting X11 Display...\033[0m"

export X11_SOCKET=/tmp/.X11-unix

# ── DISPLAY detection ─────────────────────────────────────────────────────────
if [[ -n "${DISPLAY:-}" ]]; then
    log_info "DISPLAY already set in shell: ${DISPLAY} — skipping auto-detect"
else
    if [[ ! -d "${X11_SOCKET}" ]]; then
        log_error "X11 socket directory not found: ${X11_SOCKET}
         Xorg may not be running. Check: systemctl status gdm"
    fi
    mapfile -t X11_SOCKETS < <(find "${X11_SOCKET}" -name "X[0-9]*" -type s 2>/dev/null | sort)
    if [[ ${#X11_SOCKETS[@]} -eq 0 ]]; then
        log_error "No X11 sockets found in ${X11_SOCKET}
         Available: $(ls ${X11_SOCKET}/ 2>/dev/null || echo 'none')
         Xorg may not be running."
    fi

    DETECTED_SOCKET="${X11_SOCKETS[0]}"
    DETECTED_NUM="${DETECTED_SOCKET##*X}"
    export DISPLAY=":${DETECTED_NUM}"

    log_info "Available X11 sockets: $(ls ${X11_SOCKET}/)"
    log_info "Selected: ${DETECTED_SOCKET} → DISPLAY=${DISPLAY}"

    if [[ ${#X11_SOCKETS[@]} -gt 1 ]]; then
        log_warn "Multiple X11 sockets found — using first: ${DETECTED_SOCKET}"
        log_warn "To override: export DISPLAY=:N before running build.sh"
    fi
fi

X11_SOCKET_PATH="${X11_SOCKET}/X${DISPLAY##*:}"
if [[ ! -S "${X11_SOCKET_PATH}" ]]; then
    log_error "X11 socket does not exist: ${X11_SOCKET_PATH}
       DISPLAY=${DISPLAY} but no matching socket found.
       Available: $(ls ${X11_SOCKET}/ 2>/dev/null || echo 'none')"
fi

if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    log_info "XAUTHORITY already set in shell: ${XAUTHORITY} — skipping auto-detect"
else
    DETECTED_XAUTH=$(ps aux 2>/dev/null \
        | grep -E '[Xx]org|[Xx]11' \
        | grep -oP '(?<=-auth )[^ ]+' \
        | head -1 || true)

    if [[ -z "${DETECTED_XAUTH:-}" || ! -f "${DETECTED_XAUTH}" ]]; then
        log_warn "Could not detect XAUTHORITY from Xorg process — trying known locations..."
        for candidate in \
            "/run/user/$(id -u)/gdm/Xauthority" \
            "/run/user/$(id -u)/Xauthority" \
            "${HOME}/.Xauthority" \
            "/tmp/.Xauthority-${USER}" \
            "/tmp/.Xauthority"; do
            if [[ -f "${candidate}" && -s "${candidate}" ]]; then
                DETECTED_XAUTH="${candidate}"
                log_info "Found XAUTHORITY at: ${candidate}"
                break
            fi
        done
    fi

    if [[ -z "${DETECTED_XAUTH:-}" || ! -f "${DETECTED_XAUTH}" ]]; then
        log_error "Could not detect XAUTHORITY.
         Xorg may not be running an active display session.
         Manual override: export XAUTHORITY=/path/to/Xauthority before running build.sh"
    fi

    export XAUTHORITY="${DETECTED_XAUTH}"
fi

if [[ ! -s "${XAUTHORITY}" ]]; then
    log_error "XAUTHORITY file is empty: ${XAUTHORITY}"
fi

log_info "XAUTHORITY=${XAUTHORITY}"
XAUTH_MOUNT=$(docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config 2>/dev/null \
    | grep "Xauthority" | head -2)

log_info "Resolved Xauthority volume mounts:"
echo "${XAUTH_MOUNT}" | while IFS= read -r line; do echo "           ${line}"; done

if echo "${XAUTH_MOUNT}" | grep -qE "^\s*-\s*:/"; then
    log_error "XAUTHORITY volume mount source is empty in compose config.
       XAUTHORITY was not exported to shell before compose ran.
       Run: docker compose --env-file .env config | grep Xauthority to debug."
fi

if command -v xhost &> /dev/null; then
    xhost +local:docker > /dev/null 2>&1 || true
fi

log_success "X11 ready — DISPLAY=${DISPLAY} | XAUTHORITY=${XAUTHORITY}"

# ── [3/N] EULA / BYOL License Gate ───────────────────────────────────────────
echo -e "\n\033[1;34m[3/${TOTAL_STEPS}] BYOL License Acceptance...\033[0m"

if [[ "${ACCEPT_ULTRALYTICS_EULA}" == "true" ]]; then
    log_success "ACCEPT_ULTRALYTICS_EULA=true (set in .env) — skipping prompt."
    EULA_ACCEPTED=true
else
    echo -e "\033[1;36m╔══════════════════════════════════════════════════════════════════════╗\033[0m"
    echo -e "\033[1;36m║      WEDA-CV Component License Agreement                             ║\033[0m"
    echo -e "\033[1;36m╚══════════════════════════════════════════════════════════════════════╝\033[0m"
    echo ""
    echo "  [1] Display the full EULA / License text"
    echo "  [2] Skip display and proceed to acceptance"
    echo "  [3] Decline — skip Ultralytics installation"
    echo ""
    printf "  Enter choice [1/2/3]: "
    # Under `set -e` a bare `read` at EOF aborts the script with no diagnostic —
    # which is what a piped or unattended invocation always hits.
    if ! read -r DISPLAY_CHOICE; then
        echo ""
        log_error "No input available — this shell is not interactive.
         Set ACCEPT_ULTRALYTICS_EULA=true in .env to accept the terms unattended."
    fi

    case "$DISPLAY_CHOICE" in
        1)
            PREVIEW_IMG=$(docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
                config --format json | jq -r ".services.\"${SERVICE_NAME}\".image // empty")
            [[ -z "${PREVIEW_IMG}" ]] && PREVIEW_IMG=$(grep -m1 "^FROM" Dockerfile | awk '{print $2}')
            echo "══════════════════════════════════════════════════════════════"
            docker run --rm --entrypoint cat "${PREVIEW_IMG}" \
                /usr/local/share/doc/nvidia/EULA.txt 2>/dev/null \
                || echo "  (License text not available for preview.)"
            echo "══════════════════════════════════════════════════════════════"
            ;;
        3) EULA_ACCEPTED=false ;;
    esac

    if [[ "${EULA_ACCEPTED:-}" != "false" ]]; then
        printf "\n  Accept EULA and BYOL terms? [yes/no]: "
        if ! read -r USER_ACCEPT; then
            echo ""
            log_error "No input available — this shell is not interactive.
         Set ACCEPT_ULTRALYTICS_EULA=true in .env to accept the terms unattended."
        fi
        USER_ACCEPT_LOWER=$(echo "$USER_ACCEPT" | tr '[:upper:]' '[:lower:]')
        if [[ "$USER_ACCEPT_LOWER" == "yes" || "$USER_ACCEPT_LOWER" == "y" ]]; then
            EULA_ACCEPTED=true
            log_success "EULA accepted."
        else
            EULA_ACCEPTED=false
        fi
    fi
fi

[[ "${EULA_ACCEPTED:-}" == "true" ]] \
    && export ACCEPT_ULTRALYTICS_EULA=true \
    || export ACCEPT_ULTRALYTICS_EULA=false

# ── [4/N] Pull ────────────────────────────────────────────────────────────────
STEP=4
if [[ "${MODE}" == "build" ]]; then
    echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Synchronizing container image...\033[0m"
    if docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
            pull "${SERVICE_NAME}" 2>/dev/null; then
        log_success "Image synchronization complete."
    elif docker image inspect "${WEDA_CV_IMAGE}" &> /dev/null; then
        log_warn "Pull failed (offline or registry unreachable) — using the local image."
    else
        log_error "Pull failed and no local copy of '${WEDA_CV_IMAGE}' exists.
         Check network access to the registry and 'docker login' credentials."
    fi
    STEP=$((STEP + 1))
fi

# ── [N/N] Launch ──────────────────────────────────────────────────────────────
echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Launching container...\033[0m"

# Let Compose decide whether to reuse or recreate. Force-removing the container
# on every run discards the hydrated layer, forcing a full apt + BYOL re-install
# each launch — the reason hydration used to overrun its budget every time.
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
    up -d --remove-orphans "${SERVICE_NAME}"

log_success "Container '${CONTAINER_NAME}' is live."

# ── Wait for Hydration ────────────────────────────────────────────────────────
# The .ready sentinel is cleared by entrypoint.sh before it hydrates. Clearing
# it from the host here would race with the entrypoint that just wrote it.
echo "  Waiting for environment hydration (first run downloads dependencies)..."

TIMEOUT="${HYDRATION_TIMEOUT:-300}"
READY=false
DEAD=false
FAILED=false
elapsed=0

while [[ $elapsed -lt $TIMEOUT ]]; do
    if docker exec "${CONTAINER_NAME}" \
        ls /wise-edge/instance-segmentation/.ready &> /dev/null; then
        READY=true
        printf "\r  [\033[1;32m%s\033[0m] 100%%\n" "$(printf "%*s" 50 | tr ' ' '#')"
        break
    fi

    # The entrypoint reports an unrecoverable failure explicitly — surface it now
    # rather than making the operator sit through the rest of the budget.
    if docker exec "${CONTAINER_NAME}" \
        ls /wise-edge/instance-segmentation/.hydration-failed &> /dev/null; then
        FAILED=true
        printf "\r%*s\r" 70 ""
        break
    fi

    # Fail fast instead of burning the full budget on a container that has died.
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" != "true" ]]; then
        DEAD=true
        printf "\r%*s\r" 70 ""
        break
    fi

    pct=$(( elapsed * 100 / TIMEOUT ))
    hashes=$(( (pct * 50) / 100 ))
    spaces=$(( 50 - hashes ))
    bar="$(printf "%*s" $hashes "" | tr ' ' '#')"
    empty="$(printf "%*s" $spaces "")"
    printf "\r  [\033[1;36m%s\033[0m%s] %3d%% " "$bar" "$empty" "$pct"
    sleep 2
    elapsed=$(( elapsed + 2 ))
done

# ── Hydration Failure — Report, Do Not Hand Over a Broken Shell ───────────────
if [[ "${READY}" != "true" ]]; then
    if [[ "${FAILED}" == "true" ]]; then
        echo -e "  [\033[1;31mFAILED\033[0m]"
        log_warn "Hydration reported an unrecoverable failure after ${elapsed}s."
    elif [[ "${DEAD}" == "true" ]]; then
        echo -e "  [\033[1;31mEXITED\033[0m]"
        log_warn "The container stopped during hydration."
    else
        echo -e "\n  [\033[1;31mTIMEOUT\033[0m]"
        log_warn "Hydration did not complete within ${TIMEOUT}s."
        log_warn "Raise HYDRATION_TIMEOUT in .env if this device is on a slow link."
    fi

    echo ""
    log_info "Last 30 lines of container log:"
    docker logs --tail 30 "${CONTAINER_NAME}" 2>&1 | sed 's/^/           /' || true
    echo ""
    log_warn "The demo is NOT ready — running it now would fail on missing dependencies."
    log_warn "Full log:  docker logs ${CONTAINER_NAME}"
    log_warn "Inspect:   ./build.sh --shell"
    log_warn "Reset:     ./build.sh --clean && ./build.sh"
    exit 1
fi

log_success "Environment hydrated."

# ── Shell ─────────────────────────────────────────────────────────────────────
echo -e "\n\033[1;33m╔══════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;33m║       WEDA-CV Instance Segmentation Demo — Shell         ║\033[0m"
echo -e "\033[1;33m╚══════════════════════════════════════════════════════════╝\033[0m"
echo -e "\n💡 \033[1mRun demo:\033[0m    \033[1;32mpython3 src/instance_segmentation.py\033[0m"
echo -e "🤖 \033[1mBenchmark:\033[0m  \033[1;36m./wise-bench\033[0m"

docker exec -it "${CONTAINER_NAME}" /bin/bash || true