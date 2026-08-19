# Cancel Extra API Interval Bypass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let pure `cancel_extra` cleanup actions bypass the API min-interval guard while still being protected by token bucket limits.

**Architecture:** Add a small shared helper in `strategy_core.py` to identify pure cleanup cancel actions, then use it in both `backtest_tick_mm.py` and `live_tick_mm.py` at the API interval guard. Keep quote throttle and token bucket behavior unchanged.

**Tech Stack:** Python, pytest, existing Binance tick market-making example code.

---

## File Structure

- Modify `examples/binance_tick_mm/strategy_core.py`
  - Add `is_pure_cancel_extra(actions: list[Action]) -> bool` next to `format_actions()`.
- Modify `examples/binance_tick_mm/backtest_tick_mm.py`
  - Import `is_pure_cancel_extra` and use it to bypass only the API interval guard for pure `cancel_extra`.
- Modify `examples/binance_tick_mm/live_tick_mm.py`
  - Apply the same guard behavior as backtest.
- Modify `examples/binance_tick_mm/test_strategy_core.py`
  - Add helper tests and bypass-condition tests.

---

### Task 1: Add shared helper tests and implementation

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`
- Modify: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Write failing tests for pure cancel_extra detection**

Add import in `examples/binance_tick_mm/test_strategy_core.py`:

```python
from strategy_core import (
    Action,
    QuoteThrottleConfig,
    QuoteThrottleState,
    build_audit_row,
    format_actions,
    is_pure_cancel_extra,
    should_throttle_quote_update,
)
```

Add tests near the existing `format_actions` tests:

```python
def test_is_pure_cancel_extra_false_for_empty_actions() -> None:
    assert is_pure_cancel_extra([]) is False


def test_is_pure_cancel_extra_true_for_only_cancel_extra_actions() -> None:
    actions = [
        Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0),
        Action(kind="cancel", side="extra", order_id=2, price=0.0, qty=0.0),
    ]

    assert is_pure_cancel_extra(actions) is True


def test_is_pure_cancel_extra_false_for_mixed_actions() -> None:
    actions = [
        Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0),
        Action(kind="submit", side="buy", order_id=2, price=100.0, qty=0.01),
    ]

    assert is_pure_cancel_extra(actions) is False
```

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_false_for_empty_actions examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_true_for_only_cancel_extra_actions examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_false_for_mixed_actions -q
```

Expected: fail with `ImportError` or `NameError` for `is_pure_cancel_extra`.

- [ ] **Step 3: Implement the helper**

Add after `format_actions()` in `examples/binance_tick_mm/strategy_core.py`:

```python
def is_pure_cancel_extra(actions: list[Action]) -> bool:
    return bool(actions) and all(
        action.kind == "cancel" and action.side == "extra"
        for action in actions
    )
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_false_for_empty_actions examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_true_for_only_cancel_extra_actions examples/binance_tick_mm/test_strategy_core.py::test_is_pure_cancel_extra_false_for_mixed_actions -q
```

Expected: all three tests pass.

---

### Task 2: Apply bypass in backtest and live

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`
- Modify: `examples/binance_tick_mm/live_tick_mm.py`
- Modify: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add a focused bypass-condition test**

Add this test to `examples/binance_tick_mm/test_strategy_core.py`:

```python
def test_pure_cancel_extra_bypasses_interval_condition_but_quote_update_does_not() -> None:
    cancel_extra = [Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0)]
    quote_update = [Action(kind="cancel", side="buy", order_id=2, price=100.0, qty=0.01)]

    assert is_pure_cancel_extra(cancel_extra) is True
    assert is_pure_cancel_extra(quote_update) is False
```

- [ ] **Step 2: Import helper in backtest and live**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add `is_pure_cancel_extra` to the `strategy_core` import block:

```python
    format_actions,
    is_pure_cancel_extra,
    should_throttle_quote_update,
```

In `examples/binance_tick_mm/live_tick_mm.py`, add the same import:

```python
    format_actions,
    is_pure_cancel_extra,
    should_throttle_quote_update,
```

- [ ] **Step 3: Update backtest API interval guard**

In `examples/binance_tick_mm/backtest_tick_mm.py`, replace:

```python
                    elif last_api_ts is not None and (ts_local - last_api_ts) < min_interval_ns:
                        dropped_by_api_limit = True
                        reject_reason = "api_interval_guard"
                        throttle_reason = "api_interval"
```

with:

```python
                    elif (
                        last_api_ts is not None
                        and (ts_local - last_api_ts) < min_interval_ns
                        and not is_pure_cancel_extra(planned_actions)
                    ):
                        dropped_by_api_limit = True
                        reject_reason = "api_interval_guard"
                        throttle_reason = "api_interval"
```

- [ ] **Step 4: Update live API interval guard**

In `examples/binance_tick_mm/live_tick_mm.py`, replace:

```python
                        elif last_api_ts is not None and (ts_local - last_api_ts) < min_interval_ns:
                            dropped_by_api_limit = True
                            reject_reason = "api_interval_guard"
                            throttle_reason = "api_interval"
```

with:

```python
                        elif (
                            last_api_ts is not None
                            and (ts_local - last_api_ts) < min_interval_ns
                            and not is_pure_cancel_extra(planned_actions)
                        ):
                            dropped_by_api_limit = True
                            reject_reason = "api_interval_guard"
                            throttle_reason = "api_interval"
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: all tests pass.

---

### Task 3: Final validation

**Files:**
- Validate: `examples/binance_tick_mm/strategy_core.py`
- Validate: `examples/binance_tick_mm/backtest_tick_mm.py`
- Validate: `examples/binance_tick_mm/live_tick_mm.py`
- Validate: `examples/binance_tick_mm/test_strategy_core.py`
- Validate: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Run Python compile check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/strategy_core.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Review relevant diff**

Run:

```bash
git diff -- examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/test_strategy_core.py
```

Expected: diff only adds `is_pure_cancel_extra`, tests, and API interval guard bypass in backtest/live.

---

## Self-Review

Spec coverage:
- Pure `cancel_extra` detection is covered by Task 1.
- Backtest/live API interval bypass is covered by Task 2.
- Token bucket remains unchanged because the bypass only changes the API interval condition.
- Validation is covered by Task 3.

Placeholder scan: no placeholders.

Type consistency: helper signature is consistently `is_pure_cancel_extra(actions: list[Action]) -> bool`.
