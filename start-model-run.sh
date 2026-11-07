#!/bin/sh
# Start Model Run on macOS or Linux. The Windows equivalent is Start-ModelRun.ps1.
# With --comfy-dir, start ComfyUI with Model Run inside it (the full experience) and open both.
# With --standalone, or without a ComfyUI folder, start the standalone app in the background and print its PID.
set -eu

usage() {
    echo "Usage: $0 [--comfy-dir DIR [--comfy-python PATH] [--no-browser]] [--standalone] [--models-dir DIR] [--extra-paths FILE]... [--port N] [--python PATH]" >&2
    exit 2
}

port=""
python=python3
comfy_dir=""
comfy_python=""
standalone=""
no_browser=""
set --  "$@" --end-of-options
server_args=""
while [ "$1" != "--end-of-options" ]; do
    case "$1" in
        --comfy-dir)
            [ $# -ge 3 ] || usage
            comfy_dir=$2
            server_args="$server_args $1 $(printf '%s' "$2" | sed "s/'/'\\\\''/g; s/^/'/; s/$/'/")"
            shift 2 ;;
        --comfy-python) [ $# -ge 3 ] || usage; comfy_python=$2; shift 2 ;;
        --standalone) standalone=1; shift ;;
        --no-browser) no_browser=1; shift ;;
        --models-dir|--extra-paths)
            [ $# -ge 3 ] || usage
            server_args="$server_args $1 $(printf '%s' "$2" | sed "s/'/'\\\\''/g; s/^/'/; s/$/'/")"
            shift 2 ;;
        --port) [ $# -ge 3 ] || usage; port=$2; shift 2 ;;
        --python) [ $# -ge 3 ] || usage; python=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

root=$(cd "$(dirname "$0")" && pwd)

if [ -n "$comfy_dir" ] && [ -z "$standalone" ]; then
    cd "$root"
    set -- --comfy-dir "$comfy_dir"
    [ -n "$comfy_python" ] && set -- "$@" --comfy-python "$comfy_python"
    [ -n "$port" ] && set -- "$@" --port "$port"
    [ -n "$no_browser" ] && set -- "$@" --no-browser
    exec "$python" -m model_run.launch "$@"
fi
[ -n "$port" ] || port=8766
logs="$root/.tmp"
mkdir -p "$logs"

if "$python" -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "$port"; then
    echo "Port $port is already in use. Existing processes were left unchanged." >&2
    exit 1
fi

cd "$root"
eval "set -- $server_args"
nohup "$python" -m model_run.server --port "$port" "$@" >"$logs/app-server.stdout.log" 2>"$logs/app-server.stderr.log" </dev/null &
pid=$!

attempt=0
while [ $attempt -lt 30 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Model Run exited. Check .tmp/app-server.stderr.log." >&2
        exit 1
    fi
    if "$python" -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/session' % sys.argv[1], timeout=1)" "$port" 2>/dev/null; then
        echo "Model Run is ready at http://127.0.0.1:$port/ (PID $pid)"
        exit 0
    fi
    attempt=$((attempt + 1))
    sleep 0.2
done
echo "Server did not become reachable. PID $pid; check the logs in .tmp." >&2
exit 1
