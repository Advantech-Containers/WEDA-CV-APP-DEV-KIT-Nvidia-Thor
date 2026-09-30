#!/usr/bin/env bash
# ==============================================================================
# WEDA-CV VLM Pipeline — Build & Launch Script
#
# Usage:
#   ./build.sh              # Build/Pull, start, and enter the container
#   ./build.sh --run        # Start only (skip build/pull, use existing image)
#   ./build.sh --shell      # Attach to already-running container
#   ./build.sh --stop       # Stop and remove the container
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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  (no flag)   Build/Pull the container, start it, and open a shell"
            echo "  --run       Start the container (skip build/pull, use existing image)"
            echo "  --shell     Attach a shell to the already-running container"
            echo "  --stop      Stop and remove the container"
            echo "  --help      Show this help message"
            exit 0 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╔═╗╦  ╦   ╦  ╦╦  ╔╦╗  ╔═╗╦╔═╗╔═╗╦  ╦╔╗╔╔═╗
   ║║║║╣  ║║╠═╣───║  ╚╗╔╝   ╚╗╔╝║  ║║║  ╠═╝║╠═╝║╣ ║  ║║║║║╣
   ╚╩╝╚═╝═╩╝╩ ╩   ╚═╝ ╚╝     ╚╝ ╩═╝╩ ╩  ╩  ╩╩  ╚═╝╩═╝╩╝╚╝╚═╝
          Jetson Thor Optimized  |  Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

# ── Load Static Config from .env ──────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

[[ -f "${ENV_FILE}" ]] || log_error ".env not found at ${ENV_FILE}"

set -a
source "${ENV_FILE}"
set +a

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    COMPOSE=()
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    log_warn "HF_TOKEN is empty in .env — gated repos (e.g. Cosmos-Reason2-2B) will 401 during model pulls."
fi

# ── Static Variable Validation ────────────────────────────────────────────────
REQUIRED_STATIC=(
    WEDA_CV_IMAGE
    CONTAINER_NAME
    SERVICE_NAME
    NVIDIA_VISIBLE_DEVICES
    NVIDIA_DRIVER_CAPABILITIES
    ACCEPT_ULTRALYTICS_EULA
)

MISSING=()
for var in "${REQUIRED_STATIC[@]}"; do
    [[ -z "${!var:-}" ]] && MISSING+=("${var}")
done
[[ ${#MISSING[@]} -gt 0 ]] && log_error "Missing required vars in .env: ${MISSING[*]}"

# ── Handle --stop ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "stop" ]]; then
    echo -e "\033[1;34m[1/1] Stopping container...\033[0m"
    if docker ps -a -q -f name="${CONTAINER_NAME}" | grep -q .; then
        export X11_SOCKET=${X11_SOCKET:-/tmp/.X11-unix}
        export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}
        export DISPLAY=${DISPLAY:-:0}
        "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down --remove-orphans
        log_success "Container stopped and removed."
    else
        log_warn "Container '${CONTAINER_NAME}' is not running."
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

# ── [1/N] Pre-flight Checks ──────────────────────────────────────────────────
echo -e "\033[1;34m[1/${TOTAL_STEPS}] Pre-flight checks...\033[0m"

command -v docker &> /dev/null || log_error "docker not found."
if [[ ${#COMPOSE[@]} -eq 0 ]]; then
    log_error "Docker Compose not found. Install the Compose v2 plugin (docker-compose-plugin) or the legacy docker-compose binary."
fi
log_success "System utilities ready — using '${COMPOSE[*]}'."

if ! docker info &> /dev/null; then
    if docker info 2>&1 | grep -q -i "permission denied"; then
        log_error "Permission denied connecting to Docker socket.\n         Please add your user to the docker group:\n           sudo usermod -aG docker \${USER}\n         Then restart your session or run:\n           newgrp docker"
    else
        log_error "Docker daemon not running or not accessible."
    fi
fi
log_success "Docker daemon running."

mkdir -p "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/test_logs"
log_success "Output directories ready."

# ── [2/N] X11 Display — Auto-Detection ────────────────────────────────────────
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
type docker || true
declare -f docker || true
XAUTH_MOUNT=$("${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config \
    | grep -i "xauthority" | head -2 || true)

log_info "Resolved Xauthority volume mounts:"
echo "${XAUTH_MOUNT}" | while IFS= read -r line; do echo "           ${line}"; done

if echo "${XAUTH_MOUNT}" | grep -qE "^\s*-\s*:/"; then
    log_error "XAUTHORITY volume mount source is empty in compose config.
       XAUTHORITY was not exported to shell before compose ran.
       Run: ${COMPOSE[*]} --env-file .env config | grep Xauthority to debug."
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
    echo -e "\033[1;36m+----------------------------------------------------------------------+\033[0m"
    echo -e "\033[1;36m|      WEDA-CV Component License Agreement                             |\033[0m"
    echo -e "\033[1;36m+----------------------------------------------------------------------+\033[0m"
    echo ""
    echo "  [1] Display the full EULA / License text"
    echo "  [2] Skip display and proceed to acceptance"
    echo "  [3] Decline — skip Ultralytics installation"
    echo ""
    printf "  Enter choice [1/2/3]: "
    read -r DISPLAY_CHOICE

    case "$DISPLAY_CHOICE" in
        1)
            PREVIEW_IMG=$("${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
                config --format json 2>/dev/null | jq -r ".services.\"${SERVICE_NAME}\".image // empty" 2>/dev/null || echo "")
            [[ -z "${PREVIEW_IMG}" ]] && PREVIEW_IMG="${WEDA_CV_IMAGE}"
            echo "=================================================================="
            docker run --rm --entrypoint cat "${PREVIEW_IMG}" \
                /usr/local/share/doc/nvidia/EULA.txt 2>/dev/null \
                || echo "  (License text not available for preview.)"
            echo "=================================================================="
            ;;
        3) EULA_ACCEPTED=false ;;
    esac

    if [[ "${EULA_ACCEPTED:-}" != "false" ]]; then
        printf "\n  Accept EULA and BYOL terms? [yes/no]: "
        read -r USER_ACCEPT
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

# ── [4/N] Build or Pull ──────────────────────────────────────────────────────
STEP=4
if [[ "${MODE}" == "build" ]]; then
    echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Synchronizing container image...\033[0m"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
        build "${SERVICE_NAME}" 2>/dev/null || true
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
        pull "${SERVICE_NAME}" 2>/dev/null || true
    log_success "Image synchronization complete."
    STEP=$((STEP + 1))
fi

# ── [N/N] Launch ──────────────────────────────────────────────────────────────
echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Launching container...\033[0m"

docker stop "${CONTAINER_NAME}" &> /dev/null || true
docker rm   "${CONTAINER_NAME}" &> /dev/null || true

"${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
    up -d --remove-orphans "${SERVICE_NAME}"

log_success "Container '${CONTAINER_NAME}' is live."

# ── Wait for Hydration ────────────────────────────────────────────────────────
echo "  Waiting for environment hydration..."
docker exec "${CONTAINER_NAME}" rm -f /wise-edge/weda-cv-vlm/.ready 2>/dev/null || true

TIMEOUT=600
elapsed=0
while [[ $elapsed -lt $TIMEOUT ]]; do
    if docker exec "${CONTAINER_NAME}" \
        ls /wise-edge/weda-cv-vlm/.ready &> /dev/null; then
        printf "\r  [\033[1;32m%s\033[0m] 100%%\n" "$(printf "%*s" 50 | tr ' ' '#')"
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

if [[ $elapsed -ge $TIMEOUT ]]; then
    echo -e "\n  [\033[1;31mTIMEOUT\033[0m]"
    log_warn "Hydration timeout. Check: docker logs ${CONTAINER_NAME}"
fi

# ── Shell ─────────────────────────────────────────────────────────────────────
echo -e "\n\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "\033[1;33m|       WEDA-CV VLM Pipeline — Shell                       |\033[0m"
echo -e "\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "\n  Run pipeline:   \033[1;32mpython3 -m app.main --config config/pipeline.yaml\033[0m"
echo -e "  Wise-bench:           \033[1;33m./scripts/wise-bench\033[0m\n"
echo -e "  Wise-bench:           \033[1;33m./scripts/wise-bench --heavy\033[0m\n"

docker exec -it "${CONTAINER_NAME}" /bin/bash || true