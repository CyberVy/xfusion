#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

# Parse arguments
DO_SYNC=true
DO_RESTART=false

for arg in "$@"; do
    case "$arg" in
        --no-sync)
            DO_SYNC=false
            ;;
        --restart)
            DO_RESTART=true
            ;;
        *)
            ;;
    esac
done

# Step 1: Sync code to remote (unless --no-sync is specified)
if [ "$DO_SYNC" = true ]; then
    log_info "[1/4] Synchronizing code to remote..."
    bash "$SCRIPT_DIR/sync.sh"
else
    log_info "[1/4] Skipping code sync (--no-sync specified)."
fi

# Step 2: Ensure backend is running
if [ "$DO_RESTART" = true ]; then
    log_info "[2/4] Restarting remote backend..."
    bash "$SCRIPT_DIR/backend.sh" restart
else
    log_info "[2/4] Ensuring remote backend is running..."
    bash "$SCRIPT_DIR/backend.sh" start
fi

# Step 3: Establish local port forwarding
log_info "[3/4] Preparing port forwarding on $BACKEND_PORT..."

FORWARD_PID=""
cleanup() {
    log_info "Terminating development session..."
    if [ -n "$FORWARD_PID" ] && kill -0 "$FORWARD_PID" 2>/dev/null; then
        log_info "Closing port forward tunnel (PID $FORWARD_PID)..."
        kill "$FORWARD_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if lsof -i ":$BACKEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -ti ":$BACKEND_PORT" -sTCP:LISTEN | head -n 1)
    # Check if port is already forwarding to a responsive backend
    if curl -s "http://127.0.0.1:$BACKEND_PORT/api/v1/system/health" | grep -q "healthy"; then
        log_success "Port $BACKEND_PORT is already forwarded and responding to health check."
    else
        log_warn "Port $BACKEND_PORT is occupied by PID $EXISTING_PID but not responding to health check."
        log_warn "Stopping stale process on port $BACKEND_PORT..."
        kill -9 "$EXISTING_PID" 2>/dev/null || true
        sleep 1
        rexec forward "$BACKEND_PORT" &
        FORWARD_PID=$!
        sleep 1
    fi
else
    rexec forward "$BACKEND_PORT" &
    FORWARD_PID=$!
    sleep 1
fi

# Step 4: Start local frontend
log_success "[4/4] Remote environment and tunnel are ready! Starting frontend dev server..."
npm run dev
