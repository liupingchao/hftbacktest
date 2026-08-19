# Live REST Safety Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent live trading on stale connector state by adding Binance REST startup/periodic safety reconciliation, audit visibility, and connector instrumentation.

**Architecture:** Add a small Python REST client and safety state helpers to `live_tick_mm.py`/`strategy_core.py`, wire latest safety state into audit rows, and fail closed when REST position/open orders disagree with local `hbt` state. Add Rust connector logs around position/fill events for root-cause observability without changing connector behavior.

**Tech Stack:** Python stdlib (`urllib`, `hmac`, `hashlib`, `tomllib`), pytest, Rust connector tracing/logging.

---

## File Structure

- Modify `examples/binance_tick_mm/strategy_core.py`
  - Add safety config/status dataclasses and pure comparison helpers.
  - Extend `build_audit_row()` with safety fields.
- Modify `examples/binance_tick_mm/live_tick_mm.py`
  - Add read-only Binance Futures REST client.
  - Add startup and periodic safety reconciliation.
  - Pass safety fields into audit rows.
- Modify `examples/binance_tick_mm/audit_schema.py`
  - Add safety audit fields.
- Modify `examples/binance_tick_mm/config.example.toml`
  - Add `[live_safety]` defaults.
- Modify `examples/binance_tick_mm/test_strategy_core.py`
  - Add tests for safety comparison and audit row fields.
- Modify `connector/src/binancefutures/user_data_stream.rs`
  - Add tracing logs for startup position info, `ACCOUNT_UPDATE`, and `ORDER_TRADE_UPDATE` fills.

---

### Task 1: Add safety comparison helpers and tests

**Files:**
- Modify: `examples/binance_tick_mm/strategy_core.py`
- Modify: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add failing tests for safety comparison**

In `examples/binance_tick_mm/test_strategy_core.py`, import these new symbols:

```python
from strategy_core import (
    LiveSafetyConfig,
    LiveSafetyState,
    evaluate_live_safety,
)
```

Add tests:

```python
def test_evaluate_live_safety_ok_within_position_tolerance() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=0.043,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "ok"
    assert state.position_mismatch == 0.002
    assert state.rest_position == 0.045
    assert state.rest_open_order_count == 1
    assert state.local_open_order_count == 1


def test_evaluate_live_safety_flags_position_mismatch_over_tolerance() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=-0.029,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "position_mismatch"
    assert state.position_mismatch == 0.074


def test_evaluate_live_safety_flags_open_order_mismatch() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "open_order_mismatch"
    assert state.rest_open_order_count == 0
    assert state.local_open_order_count == 1


def test_evaluate_live_safety_reports_rest_error() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="timeout",
    )

    assert state.safety_status == "rest_error"


def test_evaluate_live_safety_disabled() -> None:
    cfg = LiveSafetyConfig(enabled=False, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=-0.029,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "safety_disabled"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -k live_safety -q
```

Expected: fail because symbols do not exist.

- [ ] **Step 3: Implement safety dataclasses and comparison helper**

In `examples/binance_tick_mm/strategy_core.py`, add:

```python
@dataclass
class LiveSafetyConfig:
    enabled: bool = True
    rest_check_interval_sec: float = 5.0
    position_tolerance: float = 0.003
    open_order_check: bool = True
    fail_on_mismatch: bool = True
    connector_config: str = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "LiveSafetyConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            rest_check_interval_sec=max(1.0, float(cfg.get("rest_check_interval_sec", 5.0))),
            position_tolerance=max(0.0, float(cfg.get("position_tolerance", 0.003))),
            open_order_check=bool(cfg.get("open_order_check", True)),
            fail_on_mismatch=bool(cfg.get("fail_on_mismatch", True)),
            connector_config=str(cfg.get("connector_config", "")),
        )


@dataclass
class LiveSafetyState:
    rest_position: float = 0.0
    position_mismatch: float = 0.0
    rest_open_order_count: int = 0
    local_open_order_count: int = 0
    safety_status: str = "safety_disabled"


def evaluate_live_safety(
    *,
    cfg: LiveSafetyConfig,
    rest_position: float,
    local_position: float,
    rest_open_order_count: int,
    local_open_order_count: int,
    rest_error: str,
) -> LiveSafetyState:
    mismatch = round(abs(rest_position - local_position), 12)
    if not cfg.enabled:
        status = "safety_disabled"
    elif rest_error:
        status = "rest_error"
    elif mismatch > cfg.position_tolerance:
        status = "position_mismatch"
    elif cfg.open_order_check and rest_open_order_count != local_open_order_count:
        status = "open_order_mismatch"
    else:
        status = "ok"
    return LiveSafetyState(
        rest_position=rest_position,
        position_mismatch=mismatch,
        rest_open_order_count=rest_open_order_count,
        local_open_order_count=local_open_order_count,
        safety_status=status,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -k live_safety -q
```

Expected: all live safety helper tests pass.

---

### Task 2: Add safety audit fields

**Files:**
- Modify: `examples/binance_tick_mm/audit_schema.py`
- Modify: `examples/binance_tick_mm/strategy_core.py`
- Modify: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add audit schema fields**

In `examples/binance_tick_mm/audit_schema.py`, add after `extra_order_price_ticks` in both `AUDIT_FIELDS` and `REQUIRED_ALIGNMENT_FIELDS`:

```python
    "rest_position",
    "position_mismatch",
    "rest_open_order_count",
    "local_open_order_count",
    "safety_status",
```

- [ ] **Step 2: Extend `build_audit_row()` signature and output**

In `examples/binance_tick_mm/strategy_core.py`, add parameters after `extra_order_price_ticks`:

```python
    rest_position: float,
    position_mismatch: float,
    rest_open_order_count: int,
    local_open_order_count: int,
    safety_status: str,
```

In the returned dict after `"extra_order_price_ticks": extra_order_price_ticks`, add:

```python
        "rest_position": rest_position,
        "position_mismatch": position_mismatch,
        "rest_open_order_count": rest_open_order_count,
        "local_open_order_count": local_open_order_count,
        "safety_status": safety_status,
```

- [ ] **Step 3: Update test base kwargs and add audit test**

In `examples/binance_tick_mm/test_strategy_core.py`, add to `_base_audit_kwargs()`:

```python
        "rest_position": 0.0,
        "position_mismatch": 0.0,
        "rest_open_order_count": 0,
        "local_open_order_count": 0,
        "safety_status": "ok",
```

Add test:

```python
def test_build_audit_row_includes_live_safety_fields() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "rest_position": 0.045,
        "position_mismatch": 0.074,
        "rest_open_order_count": 0,
        "local_open_order_count": 1,
        "safety_status": "position_mismatch",
    })

    row = build_audit_row(**kwargs)

    assert row["rest_position"] == 0.045
    assert row["position_mismatch"] == 0.074
    assert row["rest_open_order_count"] == 0
    assert row["local_open_order_count"] == 1
    assert row["safety_status"] == "position_mismatch"
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_build_audit_row_includes_live_safety_fields -q
```

Expected: pass.

---

### Task 3: Add Python REST client and live reconciliation wiring

**Files:**
- Modify: `examples/binance_tick_mm/live_tick_mm.py`
- Modify: `examples/binance_tick_mm/config.example.toml`

- [ ] **Step 1: Add imports to `live_tick_mm.py`**

Add stdlib imports near the top:

```python
import hashlib
import hmac
import time
import tomllib
import urllib.parse
import urllib.request
```

Import safety helpers:

```python
    LiveSafetyConfig,
    LiveSafetyState,
    evaluate_live_safety,
```

- [ ] **Step 2: Add REST client helpers to `live_tick_mm.py`**

Add above `run_live()`:

```python
class BinanceFuturesRestClient:
    def __init__(self, config_path: str) -> None:
        path = Path(config_path).expanduser().resolve()
        cfg = tomllib.loads(path.read_text())
        self.api_url = str(cfg.get("api_url", "https://fapi.binance.com")).rstrip("/")
        self.api_key = str(cfg.get("api_key", ""))
        self.secret = str(cfg.get("secret", ""))
        if not self.api_key or not self.secret:
            raise ValueError("Binance REST credentials are missing")

    def _signed_get(self, path: str, params: dict[str, Any]) -> Any:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.api_url}{path}?{query}&signature={signature}"
        request = urllib.request.Request(url, headers={"X-MBX-APIKEY": self.api_key})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())

    def position(self, symbol: str) -> float:
        rows = self._signed_get("/fapi/v2/positionRisk", {"symbol": symbol.upper()})
        for row in rows:
            if str(row.get("symbol", "")).upper() == symbol.upper():
                return float(row.get("positionAmt", 0.0))
        return 0.0

    def open_order_count(self, symbol: str) -> int:
        rows = self._signed_get("/fapi/v1/openOrders", {"symbol": symbol.upper()})
        return len(rows)
```

- [ ] **Step 3: Add local open order count helper**

Add above `run_live()`:

```python
def _local_open_order_count(working: WorkingOrders) -> int:
    count = 0
    if working.buy is not None:
        count += 1
    if working.sell is not None:
        count += 1
    count += len(working.extras)
    return count
```

- [ ] **Step 4: Initialize safety config/client/state in `run_live()`**

After config sections are loaded, add:

```python
    safety_cfg = LiveSafetyConfig.from_config(config.get("live_safety", {}))
    rest_client = None
    safety_state = LiveSafetyState(safety_status="safety_disabled")
    next_safety_check_ns = 0
    if safety_cfg.enabled:
        connector_config = safety_cfg.connector_config or str(config.get("live", {}).get("connector_config", ""))
        if not connector_config:
            raise ValueError("live_safety.connector_config must be set when live safety is enabled")
        rest_client = BinanceFuturesRestClient(connector_config)
```

- [ ] **Step 5: Reconcile during live loop before decisions**

After `working = collect_working_orders(hbt.orders(0))` and `working_diagnostics = ...`, add:

```python
                if safety_cfg.enabled and rest_client is not None and ts_local >= next_safety_check_ns:
                    rest_error = ""
                    rest_position = 0.0
                    rest_open_order_count = 0
                    try:
                        rest_position = rest_client.position(symbol)
                        rest_open_order_count = rest_client.open_order_count(symbol)
                    except Exception as exc:
                        rest_error = str(exc)
                    safety_state = evaluate_live_safety(
                        cfg=safety_cfg,
                        rest_position=rest_position,
                        local_position=position,
                        rest_open_order_count=rest_open_order_count,
                        local_open_order_count=_local_open_order_count(working),
                        rest_error=rest_error,
                    )
                    next_safety_check_ns = ts_local + int(safety_cfg.rest_check_interval_sec * 1_000_000_000)
                    if safety_cfg.fail_on_mismatch and safety_state.safety_status not in {"ok", "safety_disabled"}:
                        log.critical(
                            "Live safety mismatch: status=%s rest_position=%.6f local_position=%.6f mismatch=%.6f rest_open_orders=%d local_open_orders=%d",
                            safety_state.safety_status,
                            safety_state.rest_position,
                            position,
                            safety_state.position_mismatch,
                            safety_state.rest_open_order_count,
                            safety_state.local_open_order_count,
                        )
                        break
```

- [ ] **Step 6: Pass safety fields to `build_audit_row()` in live**

In the live `build_audit_row()` call, add:

```python
                    rest_position=safety_state.rest_position,
                    position_mismatch=safety_state.position_mismatch,
                    rest_open_order_count=safety_state.rest_open_order_count,
                    local_open_order_count=safety_state.local_open_order_count,
                    safety_status=safety_state.safety_status,
```

- [ ] **Step 7: Pass neutral safety fields in backtest**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add to `build_audit_row()` call:

```python
                rest_position=0.0,
                position_mismatch=0.0,
                rest_open_order_count=0,
                local_open_order_count=0,
                safety_status="backtest",
```

- [ ] **Step 8: Add config example**

In `examples/binance_tick_mm/config.example.toml`, add:

```toml
[live_safety]
enabled = true
rest_check_interval_sec = 5
# Three BTCUSDT lots by default; avoids failing on one in-flight 0.001 fill.
position_tolerance = 0.003
open_order_check = true
fail_on_mismatch = true
connector_config = "~/hft_live/config/binancefutures.toml"
```

- [ ] **Step 9: Run Python validation**

Run:

```bash
python -m py_compile examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/strategy_core.py
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: compile succeeds and tests pass.

---

### Task 4: Add connector instrumentation

**Files:**
- Modify: `connector/src/binancefutures/user_data_stream.rs`

- [ ] **Step 1: Add logs for `ACCOUNT_UPDATE` positions**

In `process_message()`, inside `EventStream::AccountUpdate(data)`, before sending `LiveEvent::Position`, add:

```rust
tracing::info!(
    symbol = %position.symbol,
    qty = position.position_amount,
    position_side = %position.position_side,
    transaction_time = data.transaction_time,
    "Binance futures ACCOUNT_UPDATE position"
);
```

- [ ] **Step 2: Add logs for order trade updates**

Inside `EventStream::OrderTradeUpdate(data)`, before `update_from_ws`, add:

```rust
tracing::info!(
    symbol = %data.order.symbol,
    client_order_id = %data.order.client_order_id,
    side = ?data.order.side,
    status = ?data.order.order_status,
    last_fill_qty = data.order.order_last_filled_qty,
    accumulated_fill_qty = data.order.order_filled_accumulated_qty,
    transaction_time = data.transaction_time,
    "Binance futures ORDER_TRADE_UPDATE"
);
```

- [ ] **Step 3: Add startup position logs**

In `get_position_information()`, before sending `LiveEvent::Position`, add:

```rust
tracing::info!(
    symbol = %position.symbol,
    qty = position.position_amount,
    position_side = %position.position_side,
    update_time = position.update_time,
    "Binance futures startup REST position"
);
```

- [ ] **Step 4: Run Rust check**

Run:

```bash
cargo check -p connector
```

Expected: check succeeds.

---

### Task 5: Final validation and safety runbook

**Files:**
- Validate all modified files.

- [ ] **Step 1: Run final Python tests**

Run:

```bash
python -m py_compile examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/strategy_core.py
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: pass.

- [ ] **Step 2: Run Rust check**

Run:

```bash
cargo check -p connector
```

Expected: pass.

- [ ] **Step 3: Review diff**

Run:

```bash
git diff -- examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/audit_schema.py examples/binance_tick_mm/config.example.toml examples/binance_tick_mm/test_strategy_core.py connector/src/binancefutures/user_data_stream.rs
```

Expected diff includes only:

```text
REST safety client/checks
audit safety fields
live_safety config
connector instrumentation logs
```

- [ ] **Step 4: Manual live resume precheck**

Before any live restart, run a read-only REST query and confirm current exchange position. If current exchange position is not intended, user must manually flatten or explicitly authorize risk-reduction behavior.

---

## Self-Review

Spec coverage:
- Startup/periodic REST safety is covered by Task 3.
- Configurable tolerance default `0.003` is covered by Task 1 and Task 3.
- Audit safety fields are covered by Task 2 and Task 3.
- Connector instrumentation is covered by Task 4.
- Final verification and live precheck are covered by Task 5.

Placeholder scan: no placeholders.

Type consistency:
- `LiveSafetyConfig`, `LiveSafetyState`, and `evaluate_live_safety` names are used consistently.
- Safety status strings match the design.
- Audit field names match schema, row builder, and tests.
