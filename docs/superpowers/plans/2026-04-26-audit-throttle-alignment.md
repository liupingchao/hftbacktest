# Planned-Action Audit and Quote Throttle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate planned vs executed action audit semantics and add a shared quote throttle so live/backtest can reduce API interval rejects while preserving existing `action` as executed action.

**Architecture:** Add small shared helpers to `examples/binance_tick_mm/strategy_core.py` for formatting actions and deciding quote throttles. Extend `audit_schema.py` and `build_audit_row()` with planned-action fields, then update both `backtest_tick_mm.py` and `live_tick_mm.py` to compute planned actions before local guards and executed actions after guards. Add focused unit tests plus a short audit generation check.

**Tech Stack:** Python 3.11+, pytest, existing hftbacktest Python scripts, TOML config loaded by `tomllib`.

---

## File Structure

- Modify: `examples/binance_tick_mm/strategy_core.py`
  - Add `QuoteThrottleConfig`, `QuoteThrottleState`, `format_actions()`, `should_throttle_quote_update()`.
  - Extend `build_audit_row()` signature and row output with `planned_order_id`, `planned_action`, `throttle_reason`.
- Modify: `examples/binance_tick_mm/audit_schema.py`
  - Add `planned_order_id`, `planned_action`, `throttle_reason` after existing executed `order_id`/`action` fields.
  - Add those fields to `REQUIRED_ALIGNMENT_FIELDS`.
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`
  - Import throttle helpers.
  - Parse `[strategy]` config defaults.
  - Format planned actions immediately after `decide_actions()`.
  - Apply quote throttle before API interval/token bucket guards.
  - Preserve `action`/`order_id` as executed action fields.
  - Pass planned/executed/throttle fields to `build_audit_row()`.
- Modify: `examples/binance_tick_mm/live_tick_mm.py`
  - Same execution/audit flow as backtest.
- Modify: `examples/binance_tick_mm/config.example.toml`
  - Add documented `[strategy]` section with throttle disabled by default.
- Create: `examples/binance_tick_mm/test_strategy_core.py`
  - Focused tests for action formatting, config parsing, and throttle decisions.
- Modify: `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - Add tests for new audit schema/row semantics if importing `build_audit_row` is cheaper than running a simulation.

---

### Task 1: Add shared action formatting tests

**Files:**
- Create: `examples/binance_tick_mm/test_strategy_core.py`
- Modify later: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Write failing tests for `format_actions()`**

Create `examples/binance_tick_mm/test_strategy_core.py` with:

```python
from __future__ import annotations

from strategy_core import Action, format_actions


def test_format_actions_empty_returns_keep() -> None:
    order_id, action = format_actions([])

    assert order_id == ""
    assert action == "keep"


def test_format_actions_joins_order_ids_and_kind_side_names() -> None:
    actions = [
        Action("cancel", "buy", 1, 0.0, 0.0),
        Action("submit", "buy", 2, 100.0, 0.01),
    ]

    order_id, action = format_actions(actions)

    assert order_id == "1|2"
    assert action == "cancel_buy|submit_buy"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: FAIL with import error similar to:

```text
ImportError: cannot import name 'format_actions'
```

- [ ] **Step 3: Implement `format_actions()`**

In `examples/binance_tick_mm/strategy_core.py`, add this function after the `Action` dataclass:

```python
def format_actions(actions: list[Action]) -> tuple[str, str]:
    if not actions:
        return "", "keep"
    order_id = "|".join(str(action.order_id) for action in actions)
    action_name = "|".join(f"{action.kind}_{action.side}" for action in actions)
    return order_id, action_name
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

Only if the user explicitly requested commits in this session, run:

```bash
git add examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/test_strategy_core.py
git commit -m "test: cover audit action formatting"
```

Otherwise, do not commit.

---

### Task 2: Add quote throttle helper tests

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`
- Modify later: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Extend imports**

Change the import in `examples/binance_tick_mm/test_strategy_core.py` to:

```python
from strategy_core import (
    Action,
    QuoteThrottleConfig,
    QuoteThrottleState,
    format_actions,
    should_throttle_quote_update,
)
```

- [ ] **Step 2: Add throttle tests**

Append these tests:

```python
def test_throttle_disabled_never_blocks() -> None:
    cfg = QuoteThrottleConfig(enabled=False, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_blocks_small_update_inside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0), Action("submit", "buy", 2, 10.1, 1.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_050_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == "min_quote_update_interval"


def test_throttle_allows_update_outside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_150_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_allows_large_tick_move_inside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_050_000_000,
        target_bid_tick=103,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_does_not_block_cancel_extra() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "extra", 99, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=100,
        target_ask_tick=110,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_does_not_block_pos_limit_actions() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "sell", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=100,
        target_ask_tick=110,
        planned_actions=actions,
        pos_limit=True,
    )

    assert reason == ""
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: FAIL with import error for `QuoteThrottleConfig` or `should_throttle_quote_update`.

---

### Task 3: Implement quote throttle helpers

**Files:**
- Modify: `examples/binance_tick_mm/strategy_core.py`
- Test: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add throttle dataclasses and helper**

In `examples/binance_tick_mm/strategy_core.py`, add these after `format_actions()`:

```python
@dataclass
class QuoteThrottleConfig:
    enabled: bool = False
    min_interval_ns: int = 100_000_000
    min_move_ticks: int = 2

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "QuoteThrottleConfig":
        cfg = cfg or {}
        enabled = bool(cfg.get("quote_throttle_enabled", False))
        min_interval_ms = max(0.0, float(cfg.get("min_quote_update_interval_ms", 100.0)))
        min_move_ticks = max(0, int(cfg.get("min_quote_move_ticks", 2)))
        return cls(
            enabled=enabled,
            min_interval_ns=int(min_interval_ms * 1_000_000),
            min_move_ticks=min_move_ticks,
        )


@dataclass
class QuoteThrottleState:
    last_sent_api_ts: int | None = None
    last_sent_target_bid_tick: int | None = None
    last_sent_target_ask_tick: int | None = None

    def mark_sent(self, ts_local: int, target_bid_tick: int, target_ask_tick: int) -> None:
        self.last_sent_api_ts = ts_local
        self.last_sent_target_bid_tick = target_bid_tick
        self.last_sent_target_ask_tick = target_ask_tick
```

Then add:

```python
def should_throttle_quote_update(
    *,
    cfg: QuoteThrottleConfig,
    state: QuoteThrottleState,
    ts_local: int,
    target_bid_tick: int,
    target_ask_tick: int,
    planned_actions: list[Action],
    pos_limit: bool,
) -> str:
    if not cfg.enabled or not planned_actions:
        return ""
    if pos_limit:
        return ""
    if any(action.kind == "cancel" and action.side == "extra" for action in planned_actions):
        return ""
    if state.last_sent_api_ts is None:
        return ""
    if state.last_sent_target_bid_tick is None or state.last_sent_target_ask_tick is None:
        return ""

    elapsed_ns = ts_local - state.last_sent_api_ts
    if elapsed_ns < 0 or elapsed_ns >= cfg.min_interval_ns:
        return ""

    bid_move = abs(target_bid_tick - state.last_sent_target_bid_tick)
    ask_move = abs(target_ask_tick - state.last_sent_target_ask_tick)
    if max(bid_move, ask_move) >= cfg.min_move_ticks:
        return ""

    return "min_quote_update_interval"
```

- [ ] **Step 2: Run tests and verify pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected:

```text
8 passed
```

- [ ] **Step 3: Commit**

Only if commits were explicitly requested:

```bash
git add examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/test_strategy_core.py
git commit -m "feat: add shared quote throttle helpers"
```

Otherwise, do not commit.

---

### Task 4: Extend audit schema and build_audit_row tests

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`
- Modify later: `examples/binance_tick_mm/audit_schema.py`
- Modify later: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Extend imports for audit row test**

Change imports in `examples/binance_tick_mm/test_strategy_core.py` to include:

```python
from audit_schema import AUDIT_FIELDS, REQUIRED_ALIGNMENT_FIELDS
from strategy_core import (
    Action,
    GreekValues,
    QuoteThrottleConfig,
    QuoteThrottleState,
    build_audit_row,
    format_actions,
    should_throttle_quote_update,
)
```

- [ ] **Step 2: Add audit schema and row tests**

Append:

```python
def _base_audit_kwargs() -> dict[str, object]:
    return {
        "run_id": "test",
        "symbol": "BTCUSDT",
        "strategy_seq": 1,
        "ts_local": 1_000,
        "ts_exch": 900,
        "action_order_id": "",
        "action_name": "keep",
        "planned_order_id": "1|2",
        "planned_action": "cancel_buy|submit_buy",
        "throttle_reason": "api_interval",
        "reject_reason": "api_interval_guard",
        "req_ts": 0,
        "exch_ts": 0,
        "resp_ts": 0,
        "entry_latency_ns": 0,
        "resp_latency_ns": 0,
        "predicted_entry_ns": 0,
        "best_bid": 100.0,
        "best_ask": 100.1,
        "mid": 100.05,
        "fair": 100.0,
        "reservation": 100.0,
        "half_spread": 1.0,
        "position": 0.0,
        "auditlatency_ms": 0.0,
        "dropped_by_latency": False,
        "dropped_by_api_limit": True,
        "pos_limit": False,
        "impact_cost_val": 0.0,
        "spread_bps": 1.0,
        "vol_bps": 0.0,
        "inventory_score": 1.0,
        "feed_latency_ns": 0,
        "latency_signal_ns": 0,
        "bid_size": 1.0,
        "ask_size": 1.0,
        "greek_values": GreekValues(0.0, 0.0, 0.0, 0.0),
        "greek_adjustment": 0.0,
        "target_bid_tick": 1000,
        "target_ask_tick": 1001,
        "working_bid_tick": 999,
        "working_ask_tick": 1002,
    }


def test_audit_schema_contains_planned_action_fields() -> None:
    assert "planned_order_id" in AUDIT_FIELDS
    assert "planned_action" in AUDIT_FIELDS
    assert "throttle_reason" in AUDIT_FIELDS
    assert "planned_order_id" in REQUIRED_ALIGNMENT_FIELDS
    assert "planned_action" in REQUIRED_ALIGNMENT_FIELDS
    assert "throttle_reason" in REQUIRED_ALIGNMENT_FIELDS


def test_build_audit_row_records_planned_and_executed_actions() -> None:
    row = build_audit_row(**_base_audit_kwargs())

    assert row["planned_order_id"] == "1|2"
    assert row["planned_action"] == "cancel_buy|submit_buy"
    assert row["order_id"] == ""
    assert row["action"] == "keep"
    assert row["reject_reason"] == "api_interval_guard"
    assert row["throttle_reason"] == "api_interval"
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: FAIL because schema fields and `build_audit_row()` params do not exist yet.

---

### Task 5: Implement planned-action audit fields

**Files:**
- Modify: `examples/binance_tick_mm/audit_schema.py`
- Modify: `examples/binance_tick_mm/strategy_core.py`
- Test: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add fields to `AUDIT_FIELDS`**

In `examples/binance_tick_mm/audit_schema.py`, change the top field block from:

```python
    "order_id",
    "action",
    "reject_reason",
```

to:

```python
    "order_id",
    "action",
    "planned_order_id",
    "planned_action",
    "throttle_reason",
    "reject_reason",
```

- [ ] **Step 2: Add fields to `REQUIRED_ALIGNMENT_FIELDS`**

In the same file, change:

```python
    "order_id",
    "action",
    "reject_reason",
```

to:

```python
    "order_id",
    "action",
    "planned_order_id",
    "planned_action",
    "throttle_reason",
    "reject_reason",
```

- [ ] **Step 3: Extend `build_audit_row()` signature**

In `examples/binance_tick_mm/strategy_core.py`, add these keyword-only params immediately after `action_name: str`:

```python
    planned_order_id: str,
    planned_action: str,
    throttle_reason: str,
```

- [ ] **Step 4: Write fields into row**

In `build_audit_row()` return dict, change:

```python
        "order_id": action_order_id,
        "action": action_name,
        "reject_reason": reject_reason,
```

to:

```python
        "order_id": action_order_id,
        "action": action_name,
        "planned_order_id": planned_order_id,
        "planned_action": planned_action,
        "throttle_reason": throttle_reason,
        "reject_reason": reject_reason,
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: all tests in `test_strategy_core.py` pass.

- [ ] **Step 6: Commit**

Only if commits were explicitly requested:

```bash
git add examples/binance_tick_mm/audit_schema.py examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/test_strategy_core.py
git commit -m "feat: add planned action audit fields"
```

Otherwise, do not commit.

---

### Task 6: Update backtest execution flow tests by compiling call sites

**Files:**
- Modify later: `examples/binance_tick_mm/backtest_tick_mm.py`
- Verify: `examples/binance_tick_mm/strategy_core.py`

This task updates production call sites. The verification starts with compile/import failures because `build_audit_row()` now requires new args.

- [ ] **Step 1: Run compile to expose missing arguments**

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py
```

Expected: compile may pass because Python does not check call signatures at compile time. The actual failures will happen in tests or runtime until call sites are updated.

- [ ] **Step 2: Update imports in `backtest_tick_mm.py`**

In `examples/binance_tick_mm/backtest_tick_mm.py`, extend the `strategy_core` import block to include:

```python
    QuoteThrottleConfig,
    QuoteThrottleState,
    format_actions,
    should_throttle_quote_update,
```

- [ ] **Step 3: Initialize throttle config/state in `run_backtest()`**

After API/latency setup near:

```python
    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)
```

add:

```python
    throttle_cfg = QuoteThrottleConfig.from_config(config.get("strategy", {}))
    throttle_state = QuoteThrottleState()
```

- [ ] **Step 4: Replace manual executed action formatting**

Inside the event loop, after `planned_actions, next_order_id = decide_actions(...)`, add:

```python
                    planned_order_id, planned_action = format_actions(planned_actions)
```

Set defaults before the latency branch:

```python
            planned_order_id = ""
            planned_action = "keep"
            throttle_reason = ""
```

- [ ] **Step 5: Apply quote throttle before API interval guard**

Inside `if planned_actions:`, before the existing `if last_api_ts is not None ...` block, add:

```python
                    throttle_reason = should_throttle_quote_update(
                        cfg=throttle_cfg,
                        state=throttle_state,
                        ts_local=ts_local,
                        target_bid_tick=target_bid_tick,
                        target_ask_tick=target_ask_tick,
                        planned_actions=planned_actions,
                        pos_limit=pos_limit,
                    )
                    if throttle_reason:
                        dropped_by_api_limit = True
                        reject_reason = "quote_throttle"
                    elif last_api_ts is not None and (ts_local - last_api_ts) < min_interval_ns:
                        dropped_by_api_limit = True
                        reject_reason = "api_interval_guard"
                        throttle_reason = "api_interval"
                    else:
                        ... existing token bucket / execution loop ...
```

Do not leave the old `if last_api_ts... else:` duplicated. Keep one guard chain.

- [ ] **Step 6: Update successful send state**

Where `last_api_ts = ts_local` is set after executing an action, add:

```python
                                throttle_state.mark_sent(ts_local, target_bid_tick, target_ask_tick)
```

- [ ] **Step 7: Use `format_actions()` for executed actions**

Replace:

```python
                            if executed_actions:
                                action_order_id = "|".join(str(a.order_id) for a in executed_actions)
                                action_name = "|".join(f"{a.kind}_{a.side}" for a in executed_actions)
```

with:

```python
                            action_order_id, action_name = format_actions(executed_actions)
```

Keep defaults as `"", "keep"` when no action is executed.

- [ ] **Step 8: Pass new fields to `build_audit_row()`**

Add args after `action_name=action_name,`:

```python
                planned_order_id=planned_order_id,
                planned_action=planned_action,
                throttle_reason=throttle_reason,
```

- [ ] **Step 9: Run focused tests and compile**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py
```

Expected: tests pass and compile succeeds.

---

### Task 7: Update live execution flow to match backtest

**Files:**
- Modify: `examples/binance_tick_mm/live_tick_mm.py`
- Verify: `examples/binance_tick_mm/live_tick_mm.py`

- [ ] **Step 1: Update imports in `live_tick_mm.py`**

In the `strategy_core` import block, add:

```python
    QuoteThrottleConfig,
    QuoteThrottleState,
    format_actions,
    should_throttle_quote_update,
```

- [ ] **Step 2: Initialize throttle config/state in `run_live()`**

After:

```python
    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)
```

add:

```python
    throttle_cfg = QuoteThrottleConfig.from_config(config.get("strategy", {}))
    throttle_state = QuoteThrottleState()
```

- [ ] **Step 3: Add planned/throttle defaults in event loop**

Near existing defaults:

```python
                action_order_id = ""
                action_name = "keep"
```

add:

```python
                planned_order_id = ""
                planned_action = "keep"
                throttle_reason = ""
```

- [ ] **Step 4: Format planned actions after `decide_actions()`**

After `planned_actions, next_order_id = decide_actions(...)`, add:

```python
                        planned_order_id, planned_action = format_actions(planned_actions)
```

- [ ] **Step 5: Apply same guard chain as backtest**

Inside `if planned_actions:`, use the same order:

```python
                        throttle_reason = should_throttle_quote_update(
                            cfg=throttle_cfg,
                            state=throttle_state,
                            ts_local=ts_local,
                            target_bid_tick=target_bid_tick,
                            target_ask_tick=target_ask_tick,
                            planned_actions=planned_actions,
                            pos_limit=pos_limit,
                        )
                        if throttle_reason:
                            dropped_by_api_limit = True
                            reject_reason = "quote_throttle"
                        elif last_api_ts is not None and (ts_local - last_api_ts) < min_interval_ns:
                            dropped_by_api_limit = True
                            reject_reason = "api_interval_guard"
                            throttle_reason = "api_interval"
                        else:
                            ... existing token bucket / execution loop ...
```

- [ ] **Step 6: Mark throttle state on send**

Where `last_api_ts = ts_local` is set after executing an action, add:

```python
                                    throttle_state.mark_sent(ts_local, target_bid_tick, target_ask_tick)
```

- [ ] **Step 7: Use `format_actions()` for executed actions**

Replace manual executed formatting with:

```python
                            action_order_id, action_name = format_actions(executed_actions)
```

- [ ] **Step 8: Pass new audit fields**

In the `build_audit_row()` call, add after `action_name=action_name,`:

```python
                    planned_order_id=planned_order_id,
                    planned_action=planned_action,
                    throttle_reason=throttle_reason,
```

- [ ] **Step 9: Compile live file**

Run:

```bash
python -m py_compile examples/binance_tick_mm/live_tick_mm.py
```

Expected: no output and exit code 0.

---

### Task 8: Add config defaults and compatibility checks

**Files:**
- Modify: `examples/binance_tick_mm/config.example.toml`
- Test: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add config parsing tests**

Append to `examples/binance_tick_mm/test_strategy_core.py`:

```python
def test_quote_throttle_config_defaults_disabled() -> None:
    cfg = QuoteThrottleConfig.from_config({})

    assert cfg.enabled is False
    assert cfg.min_interval_ns == 100_000_000
    assert cfg.min_move_ticks == 2


def test_quote_throttle_config_clamps_negative_values() -> None:
    cfg = QuoteThrottleConfig.from_config({
        "quote_throttle_enabled": True,
        "min_quote_update_interval_ms": -1,
        "min_quote_move_ticks": -5,
    })

    assert cfg.enabled is True
    assert cfg.min_interval_ns == 0
    assert cfg.min_move_ticks == 0
```

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: PASS.

- [ ] **Step 3: Add `[strategy]` to example config**

In `examples/binance_tick_mm/config.example.toml`, add after `[backtest]`:

```toml
[strategy]
# Disabled by default to preserve historical benchmark behavior.
quote_throttle_enabled = false
# Suggested live starting point: 100ms.
min_quote_update_interval_ms = 100
# Suppress small quote updates inside the interval unless target moved by this many ticks.
min_quote_move_ticks = 2
```

- [ ] **Step 4: Run focused validation**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/strategy_core.py
```

Expected: all tests pass; compile succeeds.

---

### Task 9: Add lightweight audit semantic regression test

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add explicit scenarios for reject rows**

Append:

```python
def test_build_audit_row_for_quote_throttle_reject() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "planned_order_id": "1|2",
        "planned_action": "cancel_buy|submit_buy",
        "action_order_id": "",
        "action_name": "keep",
        "reject_reason": "quote_throttle",
        "throttle_reason": "min_quote_update_interval",
        "dropped_by_api_limit": True,
    })

    row = build_audit_row(**kwargs)

    assert row["planned_action"] == "cancel_buy|submit_buy"
    assert row["action"] == "keep"
    assert row["reject_reason"] == "quote_throttle"
    assert row["throttle_reason"] == "min_quote_update_interval"


def test_build_audit_row_for_executed_action_has_matching_planned_and_action() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "planned_order_id": "1|2",
        "planned_action": "cancel_buy|submit_buy",
        "action_order_id": "1|2",
        "action_name": "cancel_buy|submit_buy",
        "reject_reason": "",
        "throttle_reason": "",
        "dropped_by_api_limit": False,
    })

    row = build_audit_row(**kwargs)

    assert row["planned_action"] == row["action"]
    assert row["planned_order_id"] == row["order_id"]
    assert row["throttle_reason"] == ""
```

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: PASS.

---

### Task 10: Final validation and diff review

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS.

- [ ] **Step 2: Compile changed Python files**

Run:

```bash
python -m py_compile \
  examples/binance_tick_mm/audit_schema.py \
  examples/binance_tick_mm/strategy_core.py \
  examples/binance_tick_mm/backtest_tick_mm.py \
  examples/binance_tick_mm/live_tick_mm.py \
  examples/binance_tick_mm/test_strategy_core.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect diff**

Run:

```bash
git diff -- \
  examples/binance_tick_mm/audit_schema.py \
  examples/binance_tick_mm/strategy_core.py \
  examples/binance_tick_mm/backtest_tick_mm.py \
  examples/binance_tick_mm/live_tick_mm.py \
  examples/binance_tick_mm/config.example.toml \
  examples/binance_tick_mm/test_strategy_core.py
```

Expected:

- `action` remains executed action.
- New fields `planned_action`, `planned_order_id`, `throttle_reason` are always passed to `build_audit_row()`.
- Quote throttle defaults off.
- Live and backtest guard order matches.
- No unrelated strategy alpha changes.

---

## Self-Review

### Spec Coverage

- Planned vs executed audit fields: Tasks 4-5 and 9.
- Backward-compatible `action` semantics: Tasks 5-7 and 10.
- Shared quote throttle logic: Tasks 2-3.
- Live/backtest both use throttle: Tasks 6-7.
- Config defaults: Task 8.
- Tests for helper behavior and audit semantics: Tasks 1-5, 8-9.

### Placeholder Scan

No placeholders remain. Every code-changing step includes concrete code or exact replacement instructions.

### Type Consistency

Planned names are consistent across tasks:

- `QuoteThrottleConfig`
- `QuoteThrottleState`
- `format_actions(actions) -> tuple[str, str]`
- `should_throttle_quote_update(...) -> str`
- `planned_order_id`
- `planned_action`
- `throttle_reason`
