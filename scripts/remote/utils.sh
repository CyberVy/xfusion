#!/usr/bin/env bash

# Resolve project root (scripts/remote -> scripts -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 1. Load .env if it exists in project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# 2. Defaults for runtime configuration (can be overridden by .env or shell env)
BACKEND_PORT="${BACKEND_PORT:-8000}"

# Logging helpers
log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $*"
}

log_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $*"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $*"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $*" >&2
}

# Verify rexec installation and agent connectivity
check_rexec() {
    if ! command -v rexec >/dev/null 2>&1; then
        log_error "rexec CLI not found. Please ensure /usr/local/bin/rexec is installed and in PATH."
        exit 1
    fi

    log_info "Checking rexec connection..."
    local status_output
    if ! status_output=$(rexec status 2>&1); then
        log_error "Failed to execute rexec status:\n$status_output"
        exit 1
    fi

    if ! echo "$status_output" | grep -qi "Agent Connected: YES"; then
        log_error "Remote agent is offline or unreachable. Status output:\n$status_output"
        exit 1
    fi

    log_success "Remote agent is online."
}

# Dynamically resolve remote project directory if not explicitly specified
resolve_remote_dir() {
    if [ -n "$REMOTE_DIR" ]; then
        REMOTE_LOG_FILE="${REMOTE_LOG_FILE:-$REMOTE_DIR/.server.log}"
        REMOTE_PID_FILE="${REMOTE_PID_FILE:-$REMOTE_DIR/.server.pid}"
        REMOTE_MODELS_DIR="${REMOTE_MODELS_DIR:-$REMOTE_DIR/models}"
        return
    fi

    log_info "Detecting remote agent workspace..."
    local remote_pwd=""
    for _ in 1 2 3; do
        remote_pwd=$(rexec "pwd" 2>/dev/null | grep -E '^/' | tail -n 1 | tr -d '\r\n')
        if [ -n "$remote_pwd" ]; then
            break
        fi
        sleep 1
    done

    if [ -z "$remote_pwd" ]; then
        log_error "Failed to detect remote agent working directory. Please ensure the agent is connected or set REMOTE_DIR in .env."
        exit 1
    fi

    if [ "$(basename "$remote_pwd")" = "xfusion" ]; then
        REMOTE_DIR="$remote_pwd"
    else
        REMOTE_DIR="$remote_pwd/xfusion"
    fi

    REMOTE_LOG_FILE="${REMOTE_LOG_FILE:-$REMOTE_DIR/.server.log}"
    REMOTE_PID_FILE="${REMOTE_PID_FILE:-$REMOTE_DIR/.server.pid}"
    REMOTE_MODELS_DIR="${REMOTE_MODELS_DIR:-$REMOTE_DIR/models}"
    log_info "Using remote directory: $REMOTE_DIR"
}
