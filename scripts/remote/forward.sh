#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec

# Check if local port is occupied
if lsof -i ":$BACKEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    OCCUPYING_PID=$(lsof -ti ":$BACKEND_PORT" -sTCP:LISTEN | head -n 1)
    log_warn "Local port $BACKEND_PORT is already in use by process PID $OCCUPYING_PID."
    log_warn "If this is a previous rexec forward tunnel, you may need to kill it first."
fi

log_info "Forwarding localhost:$BACKEND_PORT -> remote:$BACKEND_PORT ..."
exec rexec forward "$BACKEND_PORT"
