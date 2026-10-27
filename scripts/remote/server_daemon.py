#!/usr/bin/env python3
"""
Remote Server Daemon & Lifecycle Manager for xfusion.
Controls background execution of start.py, PID tracking, health polling, and clean shutdown.
Fully path-agnostic: uses paths relative to project root by default.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_PORT = int(os.environ.get("BACKEND_PORT", "8000"))


def get_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def get_default_pid_file() -> str:
    return os.environ.get("REMOTE_PID_FILE") or os.path.join(get_project_root(), ".server.pid")


def get_default_log_file() -> str:
    return os.environ.get("REMOTE_LOG_FILE") or os.path.join(get_project_root(), ".server.log")


def read_pid(pid_file: Optional[str] = None) -> Optional[int]:
    target_file = pid_file or get_default_pid_file()
    if not os.path.exists(target_file):
        return None
    try:
        with open(target_file, "r") as f:
            content = f.read().strip()
            return int(content) if content else None
    except Exception:
        return None


def write_pid(pid: int, pid_file: Optional[str] = None) -> None:
    target_file = pid_file or get_default_pid_file()
    os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
    with open(target_file, "w") as f:
        f.write(str(pid))


def remove_pid(pid_file: Optional[str] = None) -> None:
    target_file = pid_file or get_default_pid_file()
    if os.path.exists(target_file):
        try:
            os.remove(target_file)
        except OSError:
            pass


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_health(port: int = DEFAULT_PORT, timeout_sec: float = 2.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/v1/system/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "xfusion-Daemon"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as res:
            if res.status == 200:
                data = json.loads(res.read().decode())
                return data.get("status") == "healthy"
    except Exception:
        return False
    return False


def tail_log(log_file: Optional[str] = None, lines: int = 25) -> str:
    target_file = log_file or get_default_log_file()
    if not os.path.exists(target_file):
        return "(Log file does not exist yet)"
    try:
        with open(target_file, "r", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"(Failed to read log: {e})"


def cmd_start(args: argparse.Namespace) -> int:
    port = args.port or DEFAULT_PORT
    pid_file = args.pid_file or get_default_pid_file()
    log_file = args.log_file or get_default_log_file()
    pid = read_pid(pid_file)

    if pid and is_process_running(pid):
        if check_health(port):
            print(f"[INFO] Server already running and healthy (PID: {pid}, Port: {port})")
            return 0
        print(f"[WARN] Stale or unresponsive server detected (PID: {pid}). Restarting...")
        cmd_stop(args)

    project_root = get_project_root()
    start_script = os.path.join(project_root, "start.py")
    if not os.path.exists(start_script):
        print(f"[ERROR] Cannot find start.py at {start_script}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    log_handle = open(log_file, "a")

    env = dict(os.environ)
    env["PORT"] = str(port)
    env["HOST"] = "0.0.0.0"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[INFO] Launching xfusion Backend on port {port}...")
    proc = subprocess.Popen(
        [sys.executable, start_script],
        cwd=project_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    write_pid(proc.pid, pid_file)
    print(f"[INFO] Process spawned with PID {proc.pid}. Waiting for readiness...")

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not is_process_running(proc.pid):
            print(f"[ERROR] Server process died prematurely (code: {proc.poll()}).", file=sys.stderr)
            print("[ERROR] Recent logs:\n" + tail_log(log_file, 30), file=sys.stderr)
            remove_pid(pid_file)
            return 1

        if check_health(port):
            print(f"[SUCCESS] Server is healthy and listening on http://127.0.0.1:{port} (PID: {proc.pid})")
            return 0

        time.sleep(1)

    print(f"[ERROR] Timed out after {args.timeout}s waiting for health endpoint.", file=sys.stderr)
    print("[ERROR] Recent logs:\n" + tail_log(log_file, 30), file=sys.stderr)
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    pid_file = args.pid_file or get_default_pid_file()
    pid = read_pid(pid_file)
    stopped_anything = False

    if pid and is_process_running(pid):
        print(f"[INFO] Stopping process {pid} with SIGTERM...")
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                if not is_process_running(pid):
                    break
                time.sleep(0.3)
            if is_process_running(pid):
                print(f"[WARN] Process {pid} did not terminate. Sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
            stopped_anything = True
        except OSError:
            pass

    # Safety net: clean any remaining start.py or uvicorn processes
    subprocess.run(["pkill", "-f", "start.py"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["pkill", "-f", "uvicorn server.api.server:app"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    remove_pid(pid_file)

    if stopped_anything:
        print("[SUCCESS] Server stopped successfully.")
    else:
        print("[INFO] No running server found.")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cmd_stop(args)
    time.sleep(1)
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    port = args.port or DEFAULT_PORT
    pid_file = args.pid_file or get_default_pid_file()
    log_file = args.log_file or get_default_log_file()
    pid = read_pid(pid_file)
    running = pid is not None and is_process_running(pid)
    healthy = check_health(port) if running else False

    print("=" * 45)
    print("xfusion Backend Status")
    print("=" * 45)
    print(f"Project Root:   {get_project_root()}")
    print(f"PID File:       {pid_file}")
    print(f"Process PID:    {pid or 'None'}")
    print(f"Process Alive:  {'YES' if running else 'NO'}")
    print(f"Health Status:  {'HEALTHY (200)' if healthy else 'UNHEALTHY / DOWN'}")
    print(f"Local Port:     {port}")
    print(f"Log File:       {log_file}")

    # Check CUDA GPU status if available
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch CUDA:   {'Available' if cuda_avail else 'Not Available'}")
        if cuda_avail:
            print(f"GPU Device:     {torch.cuda.get_device_name(0)}")
            mem_alloc = torch.cuda.memory_allocated(0) / (1024**2)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"VRAM Usage:     {mem_alloc:.1f} MB / {mem_total:.1f} MB")
    except Exception:
        pass
    print("=" * 45)
    return 0 if (running and healthy) else 1


def cmd_check_env(_args: argparse.Namespace) -> int:
    print("=" * 45)
    print("xfusion Environment Check")
    print("=" * 45)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")

    try:
        import torch

        print(f"PyTorch:           {torch.__version__}")
        print(f"CUDA Available:    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Device Name:       {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch:           [NOT INSTALLED]")

    for module_name in ["fastapi", "uvicorn", "pydantic", "diffusers", "transformers", "scipy"]:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{module_name:<18} {ver}")
        except ImportError:
            print(f"{module_name:<18} [NOT INSTALLED]")

    print("=" * 45)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="xfusion Server Daemon Manager")
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status", "check-env"],
        help="Action to perform",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Backend port (default: {DEFAULT_PORT})")
    parser.add_argument("--pid-file", default=None, help="Path to PID file (default: <project_root>/.server.pid)")
    parser.add_argument("--log-file", default=None, help="Path to log file (default: <project_root>/.server.log)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Startup timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    if args.action == "start":
        return cmd_start(args)
    elif args.action == "stop":
        return cmd_stop(args)
    elif args.action == "restart":
        return cmd_restart(args)
    elif args.action == "status":
        return cmd_status(args)
    elif args.action == "check-env":
        return cmd_check_env(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
