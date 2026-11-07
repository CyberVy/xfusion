#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

log_info "Synchronizing code to remote: $REMOTE_DIR ..."
rexec "mkdir -p $REMOTE_DIR"

# 1. Package and push codebase in a single tarball (~300 KB)
LOCAL_TAR="/tmp/xfusion_sync_$$.tar.gz"
REMOTE_TAR="/tmp/xfusion_sync_$$.tar.gz"
trap 'rm -f "$LOCAL_TAR"' EXIT INT TERM

log_info "Archiving codebase (excluding caches and static assets)..."
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$LOCAL_TAR" \
    -C "$PROJECT_ROOT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    --exclude='server/models/configs' \
    server scripts/remote pyproject.toml README.md $([ -f "$PROJECT_ROOT/uv.lock" ] && echo "uv.lock")

log_info "Transferring codebase archive..."
rexec push "$LOCAL_TAR" "$REMOTE_TAR"
rexec "tar --warning=no-unknown-keyword -xzf '$REMOTE_TAR' -C '$REMOTE_DIR' 2>/dev/null || tar -xzf '$REMOTE_TAR' -C '$REMOTE_DIR' && rm -f '$REMOTE_TAR'"

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

log_success "Sync completed successfully."
