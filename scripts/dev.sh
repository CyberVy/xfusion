#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/dev.sh [options] [extra-args...]

Options:
  --remote      (Default) Launch remote GPU dev workflow (sync + daemon + forward + frontend)
  --local       Launch fullstack dev locally (uv run backend + npm run dev frontend)
  -h, --help    Show this help message

Remote Workflow Extra Options:
  --no-sync     Skip syncing code to remote
  --restart     Force restart remote backend daemon

Examples:
  ./scripts/dev.sh
  ./scripts/dev.sh --local
  ./scripts/dev.sh --no-sync
  ./scripts/dev.sh --remote --restart
EOF
}

MODE="remote"
PASS_ARGS=()

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        --remote)
            MODE="remote"
            ;;
        --local)
            MODE="local"
            ;;
        *)
            PASS_ARGS+=("$arg")
            ;;
    esac
done

if [ "$MODE" = "remote" ]; then
    if [ ${#PASS_ARGS[@]} -gt 0 ]; then
        exec bash "$SCRIPT_DIR/remote/dev.sh" "${PASS_ARGS[@]}"
    else
        exec bash "$SCRIPT_DIR/remote/dev.sh"
    fi
else
    echo -e "\033[1;34m[INFO]\033[0m Starting local fullstack development environment..."

    BACKEND_PORT="${BACKEND_PORT:-8000}"
    BACKEND_PID=""

    cleanup() {
        if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "\033[1;34m[INFO]\033[0m Stopping local backend (PID $BACKEND_PID)..."
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    cd "$PROJECT_ROOT"

    echo -e "\033[1;34m[INFO]\033[0m Launching backend API via uv on port $BACKEND_PORT..."
    uv run python -m server &
    BACKEND_PID=$!

    echo -e "\033[1;32m[SUCCESS]\033[0m Backend launched (PID $BACKEND_PID). Starting Vite dev server..."
    npm run dev
fi
