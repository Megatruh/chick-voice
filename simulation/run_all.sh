#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO="${1:-outbreak}"
INTERVAL="${2:-2}"

if [[ -f "venv/bin/activate" ]]; then
    source "venv/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
else
    echo "Virtual environment venv atau .venv tidak ditemukan."
    exit 1
fi

mkdir -p simulation/logs
API_PID=""
DASH_PID=""

cleanup() {
    echo
    echo "Menghentikan API dan dashboard..."
    [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
    [[ -n "$DASH_PID" ]] && kill "$DASH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -m uvicorn simulation.api:app \
    --host 127.0.0.1 \
    --port 8000 \
    > simulation/logs/api.log 2>&1 &
API_PID=$!

for _ in {1..30}; do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "API gagal aktif. Periksa simulation/logs/api.log"
    exit 1
fi

python -m streamlit run simulation/dashboard.py \
    --server.headless true \
    --server.port 8501 \
    > simulation/logs/dashboard.log 2>&1 &
DASH_PID=$!

sleep 3

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:8501 >/dev/null 2>&1 || true
fi

echo "API       : http://127.0.0.1:8000"
echo "API docs  : http://127.0.0.1:8000/docs"
echo "Dashboard : http://localhost:8501"
echo "Skenario  : $SCENARIO"
echo "Interval  : $INTERVAL detik"
echo

python -m simulation.audio_simulator \
    --scenario "$SCENARIO" \
    --interval "$INTERVAL"
