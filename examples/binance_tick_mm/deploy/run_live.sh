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
mkdir -p "$DATA_DIR"

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
    "cd $EXAMPLE_DIR && python live_tick_mm.py --config $CONFIG" Enter

tmux select-layout -t "$SESSION:main" even-vertical

echo "tmux session '$SESSION' started with 3 panes:"
echo "  Pane 0: collector ($SYMBOL)"
echo "  Pane 1: connector (binancefutures)"
echo "  Pane 2: live bot"
echo ""
echo "Attach with: tmux attach -t $SESSION"
