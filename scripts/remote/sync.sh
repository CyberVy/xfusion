#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

log_info "Synchronizing code to remote: $REMOTE_DIR ..."
rexec "mkdir -p $REMOTE_DIR/server/models"

# 1. Sync server code (fast and incremental, ~300 KB)
log_info "Pushing server/ modules..."
rexec push -r "$PROJECT_ROOT/server/api" "$REMOTE_DIR/server/api" || true
rexec push -r "$PROJECT_ROOT/server/engine" "$REMOTE_DIR/server/engine" || true
rexec push -r "$PROJECT_ROOT/server/infra" "$REMOTE_DIR/server/infra" || true

# Push top-level server Python files
for pyfile in "$PROJECT_ROOT"/server/*.py; do
    if [ -f "$pyfile" ]; then
        rexec push "$pyfile" "$REMOTE_DIR/server/$(basename "$pyfile")" || true
    fi
done

# Push models Python logic
for pyfile in "$PROJECT_ROOT"/server/models/*.py; do
    if [ -f "$pyfile" ]; then
        rexec push "$pyfile" "$REMOTE_DIR/server/models/$(basename "$pyfile")" || true
    fi
done
rexec push -r "$PROJECT_ROOT/server/models/loaders" "$REMOTE_DIR/server/models/loaders" || true

# 2. Push static tokenizer configs (17MB) only if missing on remote
if ! rexec "test -d $REMOTE_DIR/server/models/configs" >/dev/null 2>&1; then
    if rexec "test -d $REMOTE_DIR/server/components/CONFIG" >/dev/null 2>&1; then
        log_info "Migrating remote model configs from server/components/CONFIG to server/models/configs..."
        rexec "mkdir -p $REMOTE_DIR/server/models && mv $REMOTE_DIR/server/components/CONFIG $REMOTE_DIR/server/models/configs" || true
    else
        log_info "Pushing static model configs (~17MB, one-time initial transfer)..."
        rexec push -r "$PROJECT_ROOT/server/models/configs" "$REMOTE_DIR/server/models/configs" || true
    fi
fi

# Clean up obsolete remote directories if present
rexec "rm -rf $REMOTE_DIR/server/components $REMOTE_DIR/server/enhancement $REMOTE_DIR/server/message" >/dev/null 2>&1 || true

# 3. Push remote scripts
log_info "Pushing scripts/remote/ ..."
rexec push -r "$PROJECT_ROOT/scripts/remote" "$REMOTE_DIR/scripts/remote" || true

# 4. Push root configurations and entrypoints (README.md is required by pyproject.toml hatchling build)
log_info "Pushing entry files and configurations..."
rexec push "$PROJECT_ROOT/start.py" "$REMOTE_DIR/start.py"
rexec push "$PROJECT_ROOT/pyproject.toml" "$REMOTE_DIR/pyproject.toml"
rexec push "$PROJECT_ROOT/README.md" "$REMOTE_DIR/README.md"

if [ -f "$PROJECT_ROOT/uv.lock" ]; then
    rexec push "$PROJECT_ROOT/uv.lock" "$REMOTE_DIR/uv.lock"
fi

log_success "Sync completed successfully."
