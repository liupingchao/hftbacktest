#!/usr/bin/env bash
# Launch collector + connector + live bot in a tmux session.
# Usage: ./run_live.sh <config.toml> <connector_config.toml> [symbol]
#
# Prerequisites:
#   - tmux installed
#   - connector and collector binaries built (cargo build --release)
#   - Python environment with hftbacktest installed
#
# Example:
#   ./run_live.sh ../config_live.toml ./binancefutures.toml BTCUSDT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR/.."

CONFIG="${1:?Usage: $0 <config.toml> <connector_config.toml> [symbol]}"
CONNECTOR_CONFIG="${2:?Usage: $0 <config.toml> <connector_config.toml> [symbol]}"
SYMBOL="${3:-BTCUSDT}"
SESSION="hft_live"

CONNECTOR_BIN="$PROJECT_ROOT/connector/target/release/connector"
COLLECTOR_BIN="$PROJECT_ROOT/collector/target/release/collector"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -f "$CONNECTOR_BIN" ] && [ -f "$PROJECT_ROOT/target/release/connector" ]; then
    CONNECTOR_BIN="$PROJECT_ROOT/target/release/connector"
fi

if [ ! -f "$COLLECTOR_BIN" ] && [ -f "$PROJECT_ROOT/target/release/collector" ]; then
    COLLECTOR_BIN="$PROJECT_ROOT/target/release/collector"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ -x "$PROJECT_ROOT/../venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/../venv/bin/python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

if [ ! -f "$CONNECTOR_BIN" ]; then
    echo "ERROR: connector binary not found at $CONNECTOR_BIN"
    echo "Run: cd $PROJECT_ROOT/connector && cargo build --release"
    exit 1
fi

if [ ! -f "$COLLECTOR_BIN" ]; then
    echo "ERROR: collector binary not found at $COLLECTOR_BIN"
    echo "Run: cd $PROJECT_ROOT/collector && cargo build --release"
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data/collected}"
RUN_DIR="${RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then
    if [ "$(basename "$DATA_DIR")" = "data" ]; then
        RUN_DIR="$(dirname "$DATA_DIR")"
    else
        RUN_DIR="$DATA_DIR"
    fi
fi
PREFLIGHT_MANIFEST="${PREFLIGHT_MANIFEST:-$RUN_DIR/deployment_manifest.json}"
START_MARKER="${START_MARKER:-$RUN_DIR/start_marker.json}"
STOP_MARKER="${STOP_MARKER:-$RUN_DIR/stop_marker.json}"
mkdir -p "$DATA_DIR"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/preflight_live_run.py" \
    --project-root "$PROJECT_ROOT" \
    --config "$CONFIG" \
    --connector-config "$CONNECTOR_CONFIG" \
    --symbol "$SYMBOL" \
    --data-dir "$DATA_DIR" \
    --run-dir "$RUN_DIR" \
    --manifest-out "$PREFLIGHT_MANIFEST" \
    --start-marker-out "$START_MARKER" \
    --stop-marker-out "$STOP_MARKER"

# Kill existing session if any.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create tmux session with 3 panes.
tmux new-session -d -s "$SESSION" -n main

# Pane 0: Collector
tmux send-keys -t "$SESSION:main" \
    "$COLLECTOR_BIN $DATA_DIR binancefuturesum $SYMBOL" Enter

# Pane 1: Connector
tmux split-window -t "$SESSION:main" -v
tmux send-keys -t "$SESSION:main.1" \
    "$CONNECTOR_BIN bf binancefutures $CONNECTOR_CONFIG" Enter

# Pane 2: Live bot
tmux split-window -t "$SESSION:main" -v
tmux send-keys -t "$SESSION:main.2" \
    "trap 'code=\$?; $PYTHON_BIN $SCRIPT_DIR/preflight_live_run.py --write-stop-marker-only --manifest-in $PREFLIGHT_MANIFEST --stop-marker-out $STOP_MARKER --exit-code \$code' EXIT; cd $EXAMPLE_DIR && $PYTHON_BIN live_tick_mm.py --config $CONFIG; exit \$?" Enter

tmux select-layout -t "$SESSION:main" even-vertical

echo "tmux session '$SESSION' started with 3 panes:"
echo "  Pane 0: collector ($SYMBOL)"
echo "  Pane 1: connector (binancefutures)"
echo "  Pane 2: live bot"
echo "  Preflight manifest: $PREFLIGHT_MANIFEST"
echo "  Start marker: $START_MARKER"
echo "  Stop marker: $STOP_MARKER"
echo ""
echo "Attach with: tmux attach -t $SESSION"
