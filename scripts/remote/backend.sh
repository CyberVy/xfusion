#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

ACTION="${1:-status}"

case "$ACTION" in
    start)
        log_info "Starting remote backend..."
        rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py start --port $BACKEND_PORT"
        ;;
    stop)
        log_info "Stopping remote backend..."
        rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py stop"
        ;;
    restart)
        log_info "Restarting remote backend..."
        rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py restart --port $BACKEND_PORT"
        ;;
    status)
        rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py status --port $BACKEND_PORT"
        ;;
    logs)
        log_info "Tailing remote log: $REMOTE_LOG_FILE (Press Ctrl+C to stop)..."
        rexec "tail -f -n 50 $REMOTE_LOG_FILE"
        ;;
    check-env)
        rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py check-env"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|check-env}"
        exit 1
        ;;
esac
