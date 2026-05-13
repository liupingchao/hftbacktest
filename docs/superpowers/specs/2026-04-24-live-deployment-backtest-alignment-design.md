# Live Deployment & Backtest-Live Alignment Design

## Goal

Deploy the existing tick market-making strategy (`backtest_tick_mm.py`) to a Tokyo production server running against Binance Futures (BTCUSDT), producing `audit_live.csv` with the same schema as `audit_bt.csv`. Use the live audit data to calibrate the local backtest environment so that backtest results reliably predict live performance.

## Constraints

- Single symbol: BTCUSDT
- Small position size: `order_notional = 100 USDT` initially
- Tokyo Linux server with Binance API key ready
- Ubuntu backtest server with 1.5 years of Tardis data
- Must reuse existing audit schema and comparison tooling

## Architecture

```
Tokyo Production Server
┌─────────────────────────────────────────────────┐
│  collector (Rust)         connector (Rust)       │
│  ├─ @trade               ├─ @trade              │
│  ├─ @bookTicker          ├─ @bookTicker         │
│  └─ @depth@0ms           ├─ @depth@0ms          │
│     ↓                    ├─ User Data Stream     │
│  btcusdt_YYYYMMDD.gz     └─ REST (submit/cancel)│
│                              ↕ IPC (iceoryx2)    │
│                          live_tick_mm.py          │
│                          ├─ Strategy (reused)    │
│                          ├─ audit_live.csv       │
│                          └─ Risk controls        │
└─────────────────────────────────────────────────┘
          │ rsync / scp
          ↓
Local Ubuntu Backtest Server
┌─────────────────────────────────────────────────┐
│  audit_live.csv                                  │
│     ↓                                            │
│  latency_from_audit.py → order_latency.npz       │
│     ↓                                            │
│  pipeline.py (Tardis data) → manifest.json       │
│     ↓                                            │
│  backtest_tick_mm.py → audit_bt.csv              │
│     ↓                                            │
│  compare_audit.py → alignment_report.json        │
│     ↓                                            │
│  Tune parameters → re-run → iterate              │
└─────────────────────────────────────────────────┘
```

## Phase 1: Collector + Connector Deployment

### 1.1 Build on Tokyo Server

```bash
# Prerequisites: Rust toolchain (rustup), build-essential/gcc
cd connector && cargo build --release
cd collector && cargo build --release
```

### 1.2 Connector Configuration

File: `binancefutures.toml`

```toml
stream_url = "wss://fstream.binance.com/ws"
api_url = "https://fapi.binance.com"
order_prefix = "mm"
api_key = "<BINANCE_API_KEY>"
secret = "<BINANCE_SECRET>"
```

Launch:
```bash
./target/release/connector --name bf --connector binancefutures --config binancefutures.toml
```

Notes:
- Production URLs (not testnet)
- Symbols registered in code must be **lowercase** (`btcusdt`)
- Connector communicates with bot via iceoryx2 shared memory IPC
- Both connector and bot must run on the same machine

### 1.3 Collector Configuration

Launch:
```bash
./target/release/collector /data/collected binancefuturesum BTCUSDT
```

Output: `/data/collected/btcusdt_YYYYMMDD.gz`
Format: `{timestamp_nanos} {json_data}\n` (gzip compressed, daily rotation)

Streams recorded: `@trade`, `@bookTicker`, `@depth@0ms`

### 1.4 Verification Checklist

- [ ] Connector logs show WebSocket market data streaming
- [ ] Connector can query positions via REST (API key permissions OK)
- [ ] Collector produces .gz files with correct `timestamp_ns json` format
- [ ] Collector data vs Tardis same-period data: field comparison validates consistency
- [ ] User data stream connects (listen key obtained)

## Phase 2: live_tick_mm.py

### 2.1 Design Principle

**100% reuse** strategy logic and audit schema from `backtest_tick_mm.py`. Only replace the data interface layer.

### 2.2 Interface Mapping

| Component | backtest_tick_mm.py | live_tick_mm.py |
|-----------|--------------------|--------------------|
| Bot construction | `ROIVectorMarketDepthBacktest([asset])` | `ROIVectorMarketDepthLiveBot([instrument])` |
| Asset config | `BacktestAsset().data().tick_size()...` | `LiveInstrument().connector().symbol()...` |
| Data source | NPZ files + manifest | Connector real-time feed |
| Latency | `IntpOrderLatency` simulation | Real network latency |
| Fill model | `PowerProbQueueModel3` simulation | Exchange real fills |
| Event loop | `wait_next_feed` iterates historical data | `wait_next_feed` waits for live feed |
| Timestamps | Historical local_ts/exch_ts from data | `bot.current_timestamp` + `bot.feed_latency`/`bot.order_latency` |
| Audit output | audit_bt.csv | audit_live.csv |

### 2.3 What Changes

1. **Bot initialization**: `LiveInstrument` replaces `BacktestAsset`
   ```python
   instrument = LiveInstrument() \
       .connector("bf") \
       .symbol("btcusdt") \
       .tick_size(0.1) \
       .lot_size(0.001) \
       .roi_lb(roi_lb) \
       .roi_ub(roi_ub)
   bot = ROIVectorMarketDepthLiveBot([instrument])
   ```

2. **Timestamp collection**:
   - `bot.feed_latency(0)` → returns `(exch_ts, local_ts)` for feed_latency_ns
   - `bot.order_latency(0)` → returns `(req_ts, exch_ts, resp_ts)` for audit

3. **LatencyOracle removed**: Live latency is real, not simulated

4. **TokenBucket retained**: Live trading is still subject to Binance API rate limits

5. **Config extension**: New `[live]` section
   ```toml
   [live]
   connector_name = "bf"
   roi_lb = 50000.0
   roi_ub = 150000.0
   ```

### 2.4 What Stays the Same

- `EwmaSigma` (volatility estimator)
- `GreekOracle` (if using Greeks)
- `WorkingOrders` tracking
- `Action` generation logic
- Fair price calculation: `mid + w_imb*(bid_size-ask_size) + w_spread*spread + w_vol*sigma + greek_adjustment`
- Reservation price: `fair - k_inv * position`
- Half-spread: `base_spread + k_vol*sigma + k_pos*|position| + impact_cost + 0.05*sigma`
- Inventory score: `1 - |position_notional| / max_notional_pos`
- Position limit logic (reduce-only when limit hit)
- Audit CSV writing (same schema, same fields)

### 2.5 Live-Only Safety Features

1. **Graceful shutdown**: SIGINT/SIGTERM handler
   - Cancel all open orders before exit
   - Flush audit CSV
   - Log final position

2. **Position sync on startup**:
   - Query actual position from exchange
   - Reconcile with internal state

3. **Hard limits**:
   - `max_notional_pos` enforced (from config)
   - `order_notional = 100 USDT` for initial small-position testing
   - Reject orders if position sync fails

4. **Heartbeat logging**: Periodic status output (position, PnL, latency stats)

### 2.6 File Structure

```
examples/binance_tick_mm/
├── backtest_tick_mm.py     # Existing backtest engine
├── live_tick_mm.py         # NEW: Live trading engine
├── audit_schema.py         # Shared (unchanged)
├── config.example.toml     # Extended with [live] section
└── ...
```

### 2.7 Shared Code Extraction

To avoid duplication between backtest and live, extract shared components into a module:

```
examples/binance_tick_mm/
├── strategy_core.py        # NEW: EwmaSigma, TokenBucket, GreekOracle,
│                           #       WorkingOrders, Action, fair/reservation/
│                           #       half_spread calculation, audit row building
├── backtest_tick_mm.py     # Imports from strategy_core, backtest-specific wrappers
├── live_tick_mm.py         # Imports from strategy_core, live-specific wrappers
```

This ensures strategy logic changes propagate to both backtest and live simultaneously.

## Phase 3: Calibration Loop

### 3.1 Data Transfer

```bash
# Tokyo → Local (daily cron or manual)
scp tokyo:/path/to/audit_live.csv ./calibration/
scp tokyo:/data/collected/btcusdt_*.gz ./collected/
```

### 3.2 Calibration Workflow

```bash
# 1. Generate latency model from live data
python latency_from_audit.py \
    --audit-csv audit_live.csv \
    --out live_order_latency.npz

# 2. Prepare backtest data for the same date range as live run
python pipeline.py --config config.toml

# 3. Run backtest with calibrated latency
python backtest_tick_mm.py --config config.toml

# 4. Compare
python compare_audit.py --bt audit_bt.csv --live audit_live.csv

# 5. Review alignment_report.json, adjust config, repeat
```

### 3.3 Calibration Priority

| Priority | Metric to Watch | Parameter to Tune | Data Source |
|----------|----------------|-------------------|-------------|
| 1 | entry_latency / resp_latency distribution | `[latency]` entry/resp min/max, spike_prob/ms | audit_live.csv req_ts/exch_ts/resp_ts |
| 2 | action_match_rate | Overall decision consistency | compare_audit output |
| 3 | fair MAE | `[fair]` w_imb, w_spread, w_vol | compare_audit output |
| 4 | position MAE | `[risk]` k_inv, base_spread | compare_audit output |
| 5 | drop_rate difference | latency_guard_ms, api_limit params | compare_audit output |
| 6 | fill count divergence | `[queue]` power_prob_n | Live fill rate vs queue depth |

### 3.4 Success Criteria

- `action_match_rate` > 90%
- `fair` MAE < 0.5 tick
- `position` MAE < 10% of max_notional_pos
- Latency distribution p50/p90/p99 within 10% of live
- Drop rates (latency + API) within 2% absolute

## Deployment Commands Summary

### Tokyo Server Setup

```bash
# 1. Clone repo and build
git clone git@github.com:liupingchao/hftbacktest.git
cd hftbacktest
cd connector && cargo build --release && cd ..
cd collector && cargo build --release && cd ..

# 2. Setup Python environment
conda create -n hft python=3.11
conda activate hft
pip install hftbacktest==2.4.4

# 3. Create config
cp examples/binance_tick_mm/config.example.toml config_live.toml
# Edit config_live.toml: set [live] section, API keys in binancefutures.toml

# 4. Launch (in separate terminals or tmux panes)
./collector/target/release/collector /data/collected binancefuturesum BTCUSDT
./connector/target/release/connector --name bf --connector binancefutures --config binancefutures.toml
python examples/binance_tick_mm/live_tick_mm.py --config config_live.toml
```

### Local Calibration

```bash
# After collecting live data
python examples/binance_tick_mm/latency_from_audit.py --audit-csv audit_live.csv --out live_order_latency.npz
python examples/binance_tick_mm/pipeline.py --config config.toml
python examples/binance_tick_mm/backtest_tick_mm.py --config config.toml
python examples/binance_tick_mm/compare_audit.py --bt audit_bt.csv --live audit_live.csv
```
