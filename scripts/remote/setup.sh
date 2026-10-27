#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

log_info "Initializing remote environment at $REMOTE_DIR ..."

# 1. Ensure uv is installed on remote
log_info "Checking remote uv installation..."
rexec "command -v uv >/dev/null 2>&1 || pip install -q uv"

# 2. Push code to remote
log_info "Syncing codebase to remote..."
bash "$SCRIPT_DIR/sync.sh"

# 3. Create models directory
rexec "mkdir -p $REMOTE_MODELS_DIR"

# 4. Sync python dependencies into remote .venv
log_info "Running uv sync in $REMOTE_DIR (this may take a minute on initial setup)..."
rexec "cd $REMOTE_DIR && uv sync"

# 5. Verify environment
log_info "Verifying remote Python environment..."
rexec "cd $REMOTE_DIR && uv run python scripts/remote/server_daemon.py check-env"

log_success "Remote setup finished successfully! Ready for development."
