# Cancel Extra Diagnostics and Two-Phase Replace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audit diagnostics for live duplicate orders and an optional two-phase quote replacement policy to reduce `cancel_extra` churn.

**Architecture:** Extend `WorkingOrders` with structured extra-order metadata and shared formatting helpers in `strategy_core.py`. Add a `two_phase_replace_enabled` flag to `decide_actions()` so backtest and live can choose between same-cycle cancel+submit and cancel-then-wait replacement while keeping default behavior unchanged.

**Tech Stack:** Python, pytest, existing `examples/binance_tick_mm` live/backtest strategy code.

---

## File Structure

- Modify `examples/binance_tick_mm/strategy_core.py`
  - Add `ExtraOrder` dataclass.
  - Extend `WorkingOrders` to retain structured extras.
  - Add helper formatting functions for working-order audit diagnostics.
  - Add `two_phase_replace_enabled` parameter to `decide_actions()`.
  - Extend `build_audit_row()` to emit working-order diagnostic fields.
- Modify `examples/binance_tick_mm/audit_schema.py`
  - Add new audit fields to `AUDIT_FIELDS` and `REQUIRED_ALIGNMENT_FIELDS`.
- Modify `examples/binance_tick_mm/backtest_tick_mm.py`
  - Pass `two_phase_replace_enabled` from `[strategy]` config to `decide_actions()`.
  - Pass working-order diagnostic values to `build_audit_row()`.
- Modify `examples/binance_tick_mm/live_tick_mm.py`
  - Same wiring as backtest.
- Modify `examples/binance_tick_mm/config.example.toml`
  - Add default `two_phase_replace_enabled = false`.
- Modify `examples/binance_tick_mm/test_strategy_core.py`
  - Add unit tests for structured extras, diagnostic formatting, and two-phase replace behavior.
- Modify `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - No expected changes unless build-audit helper call sites require signature updates.

---

### Task 1: Add structured extras and diagnostic formatting tests

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`
- Modify: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Add failing tests for extra-order diagnostics**

In `examples/binance_tick_mm/test_strategy_core.py`, add imports:

```python
from strategy_core import (
    Action,
    ExtraOrder,
    QuoteThrottleConfig,
    QuoteThrottleState,
    WorkingOrders,
    build_audit_row,
    format_actions,
    format_working_order_diagnostics,
    is_pure_cancel_extra,
    should_throttle_quote_update,
)
```

Add these tests near the helper tests:

```python
def test_format_working_order_diagnostics_empty() -> None:
    working = WorkingOrders(buy=None, sell=None, extras=[])

    diagnostics = format_working_order_diagnostics(working)

    assert diagnostics == {
        "working_buy_order_id": "",
        "working_sell_order_id": "",
        "extra_order_ids": "",
        "extra_order_sides": "",
        "extra_order_price_ticks": "",
    }


def test_format_working_order_diagnostics_with_primary_and_extras() -> None:
    class Order:
        def __init__(self, order_id: int, price_tick: int) -> None:
            self.order_id = order_id
            self.price_tick = price_tick

    working = WorkingOrders(
        buy=Order(11, 100),
        sell=Order(22, 110),
        extras=[
            ExtraOrder(order_id=33, side="buy", price_tick=101),
            ExtraOrder(order_id=44, side="sell", price_tick=111),
        ],
    )

    diagnostics = format_working_order_diagnostics(working)

    assert diagnostics == {
        "working_buy_order_id": "11",
        "working_sell_order_id": "22",
        "extra_order_ids": "33|44",
        "extra_order_sides": "buy|sell",
        "extra_order_price_ticks": "101|111",
    }
```

- [ ] **Step 2: Run diagnostic tests and verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_format_working_order_diagnostics_empty examples/binance_tick_mm/test_strategy_core.py::test_format_working_order_diagnostics_with_primary_and_extras -q
```

Expected: fail because `ExtraOrder` and `format_working_order_diagnostics` do not exist yet.

- [ ] **Step 3: Implement `ExtraOrder`, extend `WorkingOrders`, and add formatting helper**

In `examples/binance_tick_mm/strategy_core.py`, replace the existing `WorkingOrders` definition with:

```python
@dataclass
class ExtraOrder:
    order_id: int
    side: str
    price_tick: int


@dataclass
class WorkingOrders:
    buy: Any | None
    sell: Any | None
    extras: list[ExtraOrder]

    @property
    def extra_ids(self) -> list[int]:
        return [extra.order_id for extra in self.extras]
```

Add helper after `format_actions()`:

```python
def format_working_order_diagnostics(working: WorkingOrders) -> dict[str, str]:
    return {
        "working_buy_order_id": str(int(working.buy.order_id)) if working.buy is not None else "",
        "working_sell_order_id": str(int(working.sell.order_id)) if working.sell is not None else "",
        "extra_order_ids": "|".join(str(extra.order_id) for extra in working.extras),
        "extra_order_sides": "|".join(extra.side for extra in working.extras),
        "extra_order_price_ticks": "|".join(str(extra.price_tick) for extra in working.extras),
    }
```

- [ ] **Step 4: Update `collect_working_orders()` to populate structured extras**

In `examples/binance_tick_mm/strategy_core.py`, replace:

```python
extra_ids: list[int] = []
```

with:

```python
extras: list[ExtraOrder] = []
```

Replace buy-side extra append:

```python
extra_ids.append(int(order.order_id))
```

with:

```python
extras.append(ExtraOrder(int(order.order_id), "buy", int(order.price_tick)))
```

Replace sell-side extra append:

```python
extra_ids.append(int(order.order_id))
```

with:

```python
extras.append(ExtraOrder(int(order.order_id), "sell", int(order.price_tick)))
```

Replace return:

```python
return WorkingOrders(buy=buy, sell=sell, extra_ids=extra_ids)
```

with:

```python
return WorkingOrders(buy=buy, sell=sell, extras=extras)
```

- [ ] **Step 5: Run diagnostic tests and verify they pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_format_working_order_diagnostics_empty examples/binance_tick_mm/test_strategy_core.py::test_format_working_order_diagnostics_with_primary_and_extras -q
```

Expected: both tests pass.

---

### Task 2: Add two-phase replace behavior to `decide_actions()`

**Files:**
- Modify: `examples/binance_tick_mm/test_strategy_core.py`
- Modify: `examples/binance_tick_mm/strategy_core.py`

- [ ] **Step 1: Add failing tests for two-phase replace behavior**

In `examples/binance_tick_mm/test_strategy_core.py`, add this helper:

```python
class _Order:
    def __init__(self, order_id: int, side: int, price_tick: int, cancellable: bool = True) -> None:
        self.order_id = order_id
        self.side = side
        self.price_tick = price_tick
        self.cancellable = cancellable
        self.status = 0
```

Add tests:

```python
def test_decide_actions_same_cycle_replace_when_two_phase_disabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=False,
    )

    assert format_actions(actions) == ("10|20|21", "cancel_buy|submit_buy|submit_sell")
    assert next_order_id == 22


def test_decide_actions_cancel_only_when_two_phase_enabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("10|20", "cancel_buy|submit_sell")
    assert next_order_id == 21


def test_decide_actions_submits_when_side_absent_with_two_phase_enabled() -> None:
    working = WorkingOrders(buy=None, sell=None, extras=[])

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("20|21", "submit_buy|submit_sell")
    assert next_order_id == 22


def test_decide_actions_extras_keep_priority_with_two_phase_enabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[ExtraOrder(order_id=99, side="buy", price_tick=101)],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("99", "cancel_extra")
    assert next_order_id == 20
```

Make sure `decide_actions` is imported in the existing import block.

- [ ] **Step 2: Run two-phase tests and verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_same_cycle_replace_when_two_phase_disabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_cancel_only_when_two_phase_enabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_submits_when_side_absent_with_two_phase_enabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_extras_keep_priority_with_two_phase_enabled -q
```

Expected: fail because `decide_actions()` does not accept `two_phase_replace_enabled` yet.

- [ ] **Step 3: Extend `decide_actions()` signature**

In `examples/binance_tick_mm/strategy_core.py`, change signature from:

```python
def decide_actions(
    working: WorkingOrders,
    target_bid_tick: int,
    target_ask_tick: int,
    qty: float,
    tick_size: float,
    pos_limit: bool,
    position_notional: float,
    next_order_id: int,
) -> tuple[list[Action], int]:
```

to:

```python
def decide_actions(
    working: WorkingOrders,
    target_bid_tick: int,
    target_ask_tick: int,
    qty: float,
    tick_size: float,
    pos_limit: bool,
    position_notional: float,
    next_order_id: int,
    two_phase_replace_enabled: bool = False,
) -> tuple[list[Action], int]:
```

- [ ] **Step 4: Gate same-cycle replacement submits**

In `decide_actions()`, replace buy-side replacement block:

```python
    if desired_buy and working.buy is not None and buy_diff > 1 and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
```

with:

```python
    if desired_buy and working.buy is not None and buy_diff > 1 and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
        if not two_phase_replace_enabled:
            oid = next_order_id
            next_order_id += 1
            actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
```

Replace sell-side replacement block:

```python
    if desired_sell and working.sell is not None and sell_diff > 1 and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "sell", oid, target_ask_tick * tick_size, qty))
```

with:

```python
    if desired_sell and working.sell is not None and sell_diff > 1 and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))
        if not two_phase_replace_enabled:
            oid = next_order_id
            next_order_id += 1
            actions.append(Action("submit", "sell", oid, target_ask_tick * tick_size, qty))
```

- [ ] **Step 5: Run two-phase tests and verify they pass**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_same_cycle_replace_when_two_phase_disabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_cancel_only_when_two_phase_enabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_submits_when_side_absent_with_two_phase_enabled examples/binance_tick_mm/test_strategy_core.py::test_decide_actions_extras_keep_priority_with_two_phase_enabled -q
```

Expected: all four tests pass.

---

### Task 3: Wire diagnostics and two-phase config into backtest/live

**Files:**
- Modify: `examples/binance_tick_mm/audit_schema.py`
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`
- Modify: `examples/binance_tick_mm/live_tick_mm.py`
- Modify: `examples/binance_tick_mm/config.example.toml`
- Modify: `examples/binance_tick_mm/test_strategy_core.py`

- [ ] **Step 1: Add audit schema fields**

In `examples/binance_tick_mm/audit_schema.py`, add these fields after `working_ask_tick` in both `AUDIT_FIELDS` and `REQUIRED_ALIGNMENT_FIELDS`:

```python
    "working_buy_order_id",
    "working_sell_order_id",
    "extra_order_ids",
    "extra_order_sides",
    "extra_order_price_ticks",
```

- [ ] **Step 2: Extend `build_audit_row()` signature and output**

In `examples/binance_tick_mm/strategy_core.py`, add parameters after `working_ask_tick`:

```python
    working_buy_order_id: str,
    working_sell_order_id: str,
    extra_order_ids: str,
    extra_order_sides: str,
    extra_order_price_ticks: str,
```

In the returned dict, after `"working_ask_tick": working_ask_tick`, add:

```python
        "working_buy_order_id": working_buy_order_id,
        "working_sell_order_id": working_sell_order_id,
        "extra_order_ids": extra_order_ids,
        "extra_order_sides": extra_order_sides,
        "extra_order_price_ticks": extra_order_price_ticks,
```

- [ ] **Step 3: Update audit row test base kwargs**

In `examples/binance_tick_mm/test_strategy_core.py`, update `_base_audit_kwargs()` to include:

```python
        "working_buy_order_id": "",
        "working_sell_order_id": "",
        "extra_order_ids": "",
        "extra_order_sides": "",
        "extra_order_price_ticks": "",
```

Add a test:

```python
def test_build_audit_row_includes_working_order_diagnostics() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "working_buy_order_id": "11",
        "working_sell_order_id": "22",
        "extra_order_ids": "33|44",
        "extra_order_sides": "buy|sell",
        "extra_order_price_ticks": "101|111",
    })

    row = build_audit_row(**kwargs)

    assert row["working_buy_order_id"] == "11"
    assert row["working_sell_order_id"] == "22"
    assert row["extra_order_ids"] == "33|44"
    assert row["extra_order_sides"] == "buy|sell"
    assert row["extra_order_price_ticks"] == "101|111"
```

- [ ] **Step 4: Wire imports and config in backtest/live**

In both `examples/binance_tick_mm/backtest_tick_mm.py` and `examples/binance_tick_mm/live_tick_mm.py`, add `format_working_order_diagnostics` to the import block:

```python
    format_actions,
    format_working_order_diagnostics,
    is_pure_cancel_extra,
```

After throttle state initialization in both files, add:

```python
    strategy_cfg = config.get("strategy", {})
    two_phase_replace_enabled = bool(strategy_cfg.get("two_phase_replace_enabled", False))
```

If a local `strategy_cfg` already exists, reuse it and do not duplicate it.

- [ ] **Step 5: Pass two-phase flag to `decide_actions()`**

In both backtest and live, update the `decide_actions()` call by adding:

```python
                    two_phase_replace_enabled=two_phase_replace_enabled,
```

immediately after `next_order_id=next_order_id`.

- [ ] **Step 6: Pass working-order diagnostics to `build_audit_row()`**

In both backtest and live, after `working = collect_working_orders(...)`, add:

```python
            working_diagnostics = format_working_order_diagnostics(working)
```

In both `build_audit_row()` calls, add:

```python
                working_buy_order_id=working_diagnostics["working_buy_order_id"],
                working_sell_order_id=working_diagnostics["working_sell_order_id"],
                extra_order_ids=working_diagnostics["extra_order_ids"],
                extra_order_sides=working_diagnostics["extra_order_sides"],
                extra_order_price_ticks=working_diagnostics["extra_order_price_ticks"],
```

- [ ] **Step 7: Add config example default**

In `examples/binance_tick_mm/config.example.toml`, under `[strategy]`, add:

```toml
# Disabled by default to preserve historical benchmark behavior.
two_phase_replace_enabled = false
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py -q
```

Expected: all strategy core tests pass.

---

### Task 4: Final validation and diff review

**Files:**
- Validate: `examples/binance_tick_mm/strategy_core.py`
- Validate: `examples/binance_tick_mm/audit_schema.py`
- Validate: `examples/binance_tick_mm/backtest_tick_mm.py`
- Validate: `examples/binance_tick_mm/live_tick_mm.py`
- Validate: `examples/binance_tick_mm/config.example.toml`
- Validate: `examples/binance_tick_mm/test_strategy_core.py`
- Validate: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Run compile check**

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
git diff -- examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/audit_schema.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/config.example.toml examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py
```

Expected diff:

```text
- structured ExtraOrder diagnostics
- working-order diagnostic audit fields
- two_phase_replace_enabled config flag
- decide_actions same-cycle replacement gated by the flag
- backtest/live use the same config and diagnostics
- tests for diagnostics and two-phase behavior
```

---

## Self-Review

Spec coverage:
- Audit diagnostics are implemented by Tasks 1 and 3.
- Two-phase replace is implemented by Task 2 and wired by Task 3.
- Default-off compatibility is covered by Task 3 config and default argument in `decide_actions()`.
- Backtest/live consistency is covered by Task 3.
- Validation is covered by Task 4.

Placeholder scan: no placeholders.

Type consistency:
- `ExtraOrder(order_id: int, side: str, price_tick: int)` is used consistently.
- `format_working_order_diagnostics(working: WorkingOrders) -> dict[str, str]` is used consistently.
- `two_phase_replace_enabled: bool = False` is used consistently in `decide_actions()` and config wiring.
