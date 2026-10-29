#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils.sh"

check_rexec
resolve_remote_dir

do_stop() {
    log_info "Stopping remote backend..."
    rexec "
        if [ -f '$REMOTE_PID_FILE' ]; then
            PID=\$(cat '$REMOTE_PID_FILE' 2>/dev/null || true)
            if [ -n \"\$PID\" ] && kill -0 \"\$PID\" 2>/dev/null; then
                kill -15 \"\$PID\" 2>/dev/null || true
                for _ in 1 2 3 4 5; do
                    kill -0 \"\$PID\" 2>/dev/null || break
                    sleep 0.5
                done
                if kill -0 \"\$PID\" 2>/dev/null; then
                    kill -9 \"\$PID\" 2>/dev/null || true
                fi
            fi
            rm -f '$REMOTE_PID_FILE'
        fi
        pkill -f 'python -m server' 2>/dev/null || true
        pkill -f 'uvicorn server.api.server:app' 2>/dev/null || true
    "
    log_success "Remote backend stopped."
}

do_start() {
    # Check if already running and healthy
    if rexec "test -f '$REMOTE_PID_FILE' && kill -0 \$(cat '$REMOTE_PID_FILE' 2>/dev/null) 2>/dev/null && curl -sf 'http://127.0.0.1:$BACKEND_PORT/api/v1/system/health' >/dev/null 2>&1"; then
        log_success "Server already running and healthy (Port: $BACKEND_PORT)."
        return 0
    fi

    # Clean stale process if any
    do_stop >/dev/null 2>&1 || true

    log_info "Starting remote backend on port $BACKEND_PORT..."
    rexec "cd $REMOTE_DIR && mkdir -p \$(dirname '$REMOTE_LOG_FILE') && PORT=$BACKEND_PORT HOST=0.0.0.0 PYTHONUNBUFFERED=1 nohup uv run python -m server > '$REMOTE_LOG_FILE' 2>&1 & echo \$! > '$REMOTE_PID_FILE'"

    log_info "Waiting for backend readiness (timeout: 60s)..."
    local ready=false
    for _ in $(seq 1 60); do
        # Check if process died prematurely
        if ! rexec "test -f '$REMOTE_PID_FILE' && kill -0 \$(cat '$REMOTE_PID_FILE' 2>/dev/null) 2>/dev/null"; then
            log_error "Server process died unexpectedly. Recent logs:"
            rexec "tail -n 30 '$REMOTE_LOG_FILE'" || true
            rexec "rm -f '$REMOTE_PID_FILE'"
            exit 1
        fi

        # Check health endpoint
        if rexec "curl -sf 'http://127.0.0.1:$BACKEND_PORT/api/v1/system/health' >/dev/null 2>&1"; then
            ready=true
            break
        fi
        sleep 1
    done

    if [ "$ready" = true ]; then
        log_success "Server is healthy and listening on http://127.0.0.1:$BACKEND_PORT"
    else
        log_error "Timed out waiting for health endpoint. Recent logs:"
        rexec "tail -n 30 '$REMOTE_LOG_FILE'" || true
        exit 1
    fi
}

do_status() {
    rexec "
        PID=\"\"
        ALIVE=\"NO\"
        HEALTH=\"UNHEALTHY / DOWN\"
        if [ -f '$REMOTE_PID_FILE' ]; then
            PID=\$(cat '$REMOTE_PID_FILE' 2>/dev/null || true)
            if [ -n \"\$PID\" ] && kill -0 \"\$PID\" 2>/dev/null; then
                ALIVE=\"YES\"
                if curl -sf 'http://127.0.0.1:$BACKEND_PORT/api/v1/system/health' >/dev/null 2>&1; then
                    HEALTH=\"HEALTHY (200)\"
                fi
            fi
        fi

        echo '============================================='
        echo 'xfusion Remote Backend Status'
        echo '============================================='
        echo \"Remote Directory: $REMOTE_DIR\"
        echo \"PID File:         $REMOTE_PID_FILE\"
        echo \"Process PID:      \${PID:-None}\"
        echo \"Process Alive:    \$ALIVE\"
        echo \"Health Status:    \$HEALTH\"
        echo \"Backend Port:     $BACKEND_PORT\"
        echo \"Log File:         $REMOTE_LOG_FILE\"

        if command -v nvidia-smi >/dev/null 2>&1; then
            echo '---------------------------------------------'
            echo 'GPU Status (nvidia-smi):'
            nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
        fi
        echo '============================================='
    "
}

do_check_env() {
    rexec "cd $REMOTE_DIR && uv run python -c \"
import sys
print('=' * 45)
print('xfusion Remote Environment Check')
print('=' * 45)
print(f'Python Executable: {sys.executable}')
print(f'Python Version:    {sys.version.split()[0]}')

try:
    import torch
    print(f'PyTorch:           {torch.__version__}')
    print(f'CUDA Available:    {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Device:       {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch:           [NOT INSTALLED]')

for mod_name in ['fastapi', 'uvicorn', 'pydantic', 'diffusers', 'transformers', 'scipy']:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'{mod_name:<18} {ver}')
    except ImportError:
        print(f'{mod_name:<18} [NOT INSTALLED]')
print('=' * 45)
\""
}

ACTION="${1:-status}"

case "$ACTION" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        do_start
        ;;
    status)
        do_status
        ;;
    logs)
        log_info "Tailing remote log: $REMOTE_LOG_FILE (Press Ctrl+C to stop)..."
        rexec "tail -f -n 50 $REMOTE_LOG_FILE"
        ;;
    check-env)
        do_check_env
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|check-env}"
        exit 1
        ;;
esac
