# Live Position Monotonic Update Design

**Goal:** Prevent stale or out-of-order live position events from rolling back `hbt.position()` after Binance Futures account updates arrive out of order.

## Problem

The Binance Futures connector emits `LiveEvent::Position { qty, exch_ts }` from `ACCOUNT_UPDATE` messages. The live bot currently applies every position event with an unconditional assignment to `state.position`. If a stale `ACCOUNT_UPDATE` with an older exchange timestamp arrives after a newer position event, it can overwrite the local position with an old value such as `0.0`.

This matches the observed AWS run: REST position remained `-0.004`, while `hbt.position()` was `0.0`, causing the 0.003 tolerance safety gate to trigger.

## Scope

Implement the core state fix only:

- Add per-instrument tracking of the latest accepted position exchange timestamp.
- Accept position events only when their `exch_ts` is not older than the latest accepted timestamp.
- Keep startup `exch_ts=0` initialization valid before any later position event.
- Do not change audit schema.
- Do not sync REST position back into internal live bot state.
- Do not change strategy-level safety logic.

## Design

Add `last_position_exch_ts: i64` to `hftbacktest::live::Instrument`. Initialize it to `i64::MIN` so the first position event is always accepted, including startup events with `exch_ts=0`.

Update `LiveBot::process_event` for `LiveEvent::Position`:

```rust
LiveEvent::Position { qty, exch_ts, .. } => {
    let instrument = unsafe { self.instruments.get_unchecked_mut(inst_no) };
    if exch_ts >= instrument.last_position_exch_ts {
        instrument.state.position = qty;
        instrument.last_position_exch_ts = exch_ts;
    } else {
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

This fixes the root cause at the state boundary shared by connectors. Connector-side filtering can still be added later, but it is not required for this bug.

## Tests

Add a focused unit test for the position update rule. The test should exercise a small helper so it does not require constructing a full live IPC channel:

1. Create an `Instrument`.
2. Apply position `qty=-0.001, exch_ts=200` and assert it is accepted.
3. Apply stale position `qty=0.0, exch_ts=100` and assert the instrument remains `-0.001`.
4. Apply newer position `qty=-0.002, exch_ts=300` and assert the instrument updates to `-0.002`.
5. Verify startup `exch_ts=0` is accepted on a fresh instrument.

## Verification

Run the live crate tests with the `live` feature enabled:

```bash
cargo test -p hftbacktest --features live live_position
```

Also run a broader package test if the focused test passes:

```bash
cargo test -p hftbacktest --features live
```
