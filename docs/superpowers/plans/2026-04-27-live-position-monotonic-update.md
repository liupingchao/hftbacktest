# Live Position Monotonic Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale or out-of-order live `Position` events from rolling back `hbt.position()`.

**Architecture:** Track the latest accepted position exchange timestamp per live instrument and apply incoming `LiveEvent::Position` updates only when their `exch_ts` is monotonic. Keep the fix inside the live bot state boundary so all connectors that emit `LiveEvent::Position` benefit without changing strategy or audit code.

**Tech Stack:** Rust, `hftbacktest` crate live feature, existing `LiveEvent::Position` IPC model, Cargo tests.

---

## File Structure

- Modify `hftbacktest/src/live/mod.rs`
  - Add `last_position_exch_ts: i64` to `Instrument<MD>`.
  - Initialize it in `Instrument::new`.
  - Add a small helper method that applies monotonic position updates and returns whether the update was accepted.
  - Add focused unit tests for the helper.
- Modify `hftbacktest/src/live/bot.rs`
  - Use the helper in `LiveEvent::Position` handling.
  - Log ignored stale position events at `debug` level.

No Python strategy files, audit schema, or connector files should change for this core fix.

---

### Task 1: Add monotonic position state and failing tests

**Files:**
- Modify: `hftbacktest/src/live/mod.rs:14-57`

- [ ] **Step 1: Add failing tests to `hftbacktest/src/live/mod.rs`**

Append this test module to the end of `hftbacktest/src/live/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::Instrument;
    use crate::depth::ROIVectorMarketDepth;

    fn instrument() -> Instrument<ROIVectorMarketDepth> {
        Instrument::new(
            "bf",
            "btcusdt",
            0.1,
            0.001,
            ROIVectorMarketDepth::new(60_000.0, 100_000.0, 0.1, 0.001),
            0,
        )
    }

    #[test]
    fn live_position_update_ignores_stale_exchange_timestamp() {
        let mut instrument = instrument();

        assert!(instrument.apply_position_update(-0.001, 200));
        assert_eq!(instrument.state.position, -0.001);
        assert_eq!(instrument.last_position_exch_ts, 200);

        assert!(!instrument.apply_position_update(0.0, 100));
        assert_eq!(instrument.state.position, -0.001);
        assert_eq!(instrument.last_position_exch_ts, 200);

        assert!(instrument.apply_position_update(-0.002, 300));
        assert_eq!(instrument.state.position, -0.002);
        assert_eq!(instrument.last_position_exch_ts, 300);
    }

    #[test]
    fn live_position_update_accepts_startup_zero_timestamp() {
        let mut instrument = instrument();

        assert!(instrument.apply_position_update(0.0, 0));
        assert_eq!(instrument.state.position, 0.0);
        assert_eq!(instrument.last_position_exch_ts, 0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p hftbacktest --features live live_position_update -- --nocapture
```

Expected: FAIL because `Instrument` has no `apply_position_update` method and no `last_position_exch_ts` field.

---

### Task 2: Implement monotonic position updates

**Files:**
- Modify: `hftbacktest/src/live/mod.rs:14-57`

- [ ] **Step 1: Add the timestamp field to `Instrument`**

In `hftbacktest/src/live/mod.rs`, change the struct from:

```rust
pub struct Instrument<MD> {
    connector_name: String,
    symbol: String,
    tick_size: f64,
    lot_size: f64,
    depth: MD,
    last_trades: Vec<Event>,
    orders: HashMap<OrderId, Order>,
    last_feed_latency: Option<(i64, i64)>,
    last_order_latency: Option<(i64, i64, i64)>,
    state: StateValues,
}
```

to:

```rust
pub struct Instrument<MD> {
    connector_name: String,
    symbol: String,
    tick_size: f64,
    lot_size: f64,
    depth: MD,
    last_trades: Vec<Event>,
    orders: HashMap<OrderId, Order>,
    last_feed_latency: Option<(i64, i64)>,
    last_order_latency: Option<(i64, i64, i64)>,
    last_position_exch_ts: i64,
    state: StateValues,
}
```

- [ ] **Step 2: Initialize the field**

In `Instrument::new`, change:

```rust
last_feed_latency: None,
last_order_latency: None,
state: Default::default(),
```

to:

```rust
last_feed_latency: None,
last_order_latency: None,
last_position_exch_ts: i64::MIN,
state: Default::default(),
```

- [ ] **Step 3: Add the helper method**

Inside `impl<MD> Instrument<MD>`, after `new`, add:

```rust
    fn apply_position_update(&mut self, qty: f64, exch_ts: i64) -> bool {
        if exch_ts >= self.last_position_exch_ts {
            self.state.position = qty;
            self.last_position_exch_ts = exch_ts;
            true
        } else {
            false
        }
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
cargo test -p hftbacktest --features live live_position_update -- --nocapture
```

Expected: PASS for both `live_position_update_ignores_stale_exchange_timestamp` and `live_position_update_accepts_startup_zero_timestamp`.

---

### Task 3: Wire live bot Position handling through the helper

**Files:**
- Modify: `hftbacktest/src/live/bot.rs:273-277`

- [ ] **Step 1: Replace unconditional position assignment**

In `hftbacktest/src/live/bot.rs`, replace:

```rust
            LiveEvent::Position { qty, .. } => {
                unsafe { self.instruments.get_unchecked_mut(inst_no) }
                    .state
                    .position = qty;
            }
```

with:

```rust
            LiveEvent::Position { qty, exch_ts, .. } => {
                let instrument = unsafe { self.instruments.get_unchecked_mut(inst_no) };
                if !instrument.apply_position_update(qty, exch_ts) {
                    debug!(
                        %inst_no,
                        qty,
                        exch_ts,
                        last_position_exch_ts = instrument.last_position_exch_ts,
                        "Ignoring stale live position event"
                    );
                }
            }
```

- [ ] **Step 2: Run focused tests again**

Run:

```bash
cargo test -p hftbacktest --features live live_position_update -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Run live package tests**

Run:

```bash
cargo test -p hftbacktest --features live
```

Expected: PASS.

---

### Task 4: Verify diff and prepare handoff

**Files:**
- Inspect: `hftbacktest/src/live/mod.rs`
- Inspect: `hftbacktest/src/live/bot.rs`

- [ ] **Step 1: Review the diff**

Run:

```bash
git diff -- hftbacktest/src/live/mod.rs hftbacktest/src/live/bot.rs
```

Expected: Diff only contains the `last_position_exch_ts` field, helper/tests, and Position event handling change.

- [ ] **Step 2: Check working tree status**

Run:

```bash
git status --short
```

Expected: Modified files include only intended code files plus the generated spec/plan docs if they were created in this session.

- [ ] **Step 3: Report verification evidence**

Report these exact items:

```text
Focused test: cargo test -p hftbacktest --features live live_position_update -- --nocapture
Full live test: cargo test -p hftbacktest --features live
Changed code: hftbacktest/src/live/mod.rs, hftbacktest/src/live/bot.rs
Behavior: stale Position events with older exch_ts no longer overwrite hbt.position()
```

Do not claim the AWS live bug is fully resolved until a new live run verifies that REST and local position remain aligned.
