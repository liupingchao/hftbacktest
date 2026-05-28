# 0527T001 Business Report

执行线程：
- 测试线程

任务状态：
- 待验收

## Scope

This was a planning/diagnosis-only task for the `0526T008` replay audit bloat observed on:

- `local_live_analysis/5-26-active-minmove-control-30min-b`

No repair was made. No live run, strategy change, Stage 6 implementation change, replay lifecycle semantic repair, parameter search, default-on behavior, or promotion claim was made.

## Evidence

Input files inspected:
- `out/backtest_audit_replay/audit_bt_audit_replay.full.csv`
- `out/backtest_audit_replay/audit_bt_audit_replay.csv`
- `out/backtest_audit_replay/summary_audit_replay.json`
- `replay_input_note_0526T008.md`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/strategy_core.py`

Observed counts:
- full replay audit size: about `21GB`
- full replay audit rows: `33,293,775`
- full replay audit `cancel_ack` rows: `32,893,719`
- unique `order_id` with `cancel_ack`: `10,380`
- lifecycle-min canonical Stage 6 input rows: `401,642`
- lifecycle-min `cancel_ack` rows: `10,380`

The top repeated `cancel_ack` orders appeared more than `325k` times each. Example top order counts from the full replay audit:
- order `2659`: `329,084`
- order `2663`: `329,077`
- order `2666`: `329,070`

Sampled rows show the same order alternating between:
- regular `cancel_ack` rows with `local_order_seen=1`
- `forced_live_terminal_constraint` rows with `local_order_seen=0`

Code path evidence:
- `strategy_core.py` `OrderLifecycleTracker.observe()` stores current snapshots from `hbt.orders(0)` and emits a lifecycle event whenever status / request / qty changes.
- `backtest_tick_mm.py` audit replay working-order overlay writes lifecycle rows from `lifecycle_tracker.observe(...)`.
- The same audit replay path also calls `_collect_forced_live_terminal_events(...)`, which writes `forced_live_terminal_constraint` terminal rows for live terminal constraints.
- This explains why canceled orders can be emitted repeatedly as replay decisions advance when terminal snapshots remain visible or are reintroduced by overlay constraints.

## Answers

1. Is this mainly an audit export repetition problem, replay lifecycle state repetition problem, or Stage 6 input-scaling problem?

It is primarily an audit replay lifecycle export / order-state tracking repetition problem. Stage 6 became the visible victim because it expects a manageable lifecycle audit input, but Stage 6 did not create the repeated rows.

2. What exact key best defines duplicate `cancel_ack` rows?

For Stage 6 label semantics, the compact key should be:

```text
event_type=cancel_ack + order_id
```

with first terminal row retained by earliest meaningful terminal timestamp. A stricter forensic key for debugging is:

```text
order_id + cancel_request_ts + min(cancel_ack_ts, ts_exch when stable) + order_status + lifecycle_detail class
```

But Stage 6 only needs the first terminal cancel fact per order, not repeated per-decision terminal snapshots.

3. Are repeated rows semantically redundant for Stage 6 labels?

Yes. Stage 6 needs order submit, cancel request, first cancel ack, fill, expired/rejected and timing facts. The lifecycle-min input retained `10,380` cancel ack rows, matching the unique cancel-ack order count. The extra `32.88M` repeated rows do not add distinct Stage 6 label facts.

4. Should the next task repair the replay audit writer, Stage 6 input layer, or both through a formal compact lifecycle artifact?

Recommended next boundary:

- implement a formal compact lifecycle replay audit artifact in the replay audit output path, or writer-side terminal-order de-dup for lifecycle rows;
- make Stage 6 consume that compact lifecycle artifact by contract;
- preserve full forensic audit only if explicitly requested, and keep it out of the default Stage 6 path.

This is better than only adding a Stage 6 streaming reader, because the 21GB file is itself pathological output and breaks other full-event tooling too.

5. What is the narrow acceptance criterion for a later implementation task?

A later implementation task should pass if, on `5-26-active-minmove-control-30min-b`:

- compact replay lifecycle audit is generated automatically, without manual lifecycle-min workaround;
- compact audit keeps all decision rows plus submit, order_new/order_update if needed, cancel_sent, first cancel_ack per order, fill, expired/rejected;
- `cancel_ack` count is close to unique cancel-ack `order_id` count, not tens of millions;
- Stage 6 runs against the compact artifact without exit `137`;
- existing maker acceptance action/planned/reject/throttle and market-view gates do not regress;
- preserved forensic/full audit behavior is explicit and not the default Stage 6 input.

## Recommendation

Create a narrow implementation task:

```text
0528T002 - compact replay lifecycle audit export for Stage 6 input
```

Suggested implementation boundary:
- modify replay audit output handling in `backtest_tick_mm.py` to support a compact lifecycle audit output or terminal-order de-dup for audit replay lifecycle rows;
- do not change strategy behavior;
- do not change live behavior;
- do not change fill/cancel replay semantics;
- update Stage 6 input selection only to consume the formal compact lifecycle artifact.

## Verification

Commands/evidence used:
- `ls -lh local_live_analysis/5-26-active-minmove-control-30min-b/out/backtest_audit_replay`
- `head -n 1 .../audit_bt_audit_replay.full.csv`
- `head -n 5 .../audit_bt_audit_replay.csv`
- `python -m json.tool .../summary_audit_replay.json`
- `awk` streaming event counts over `audit_bt_audit_replay.csv`
- `awk` streaming `cancel_ack` counts over `audit_bt_audit_replay.full.csv`
- Python `csv.DictReader` streaming count of `cancel_ack` duplicate keys over the full replay audit
- Source inspection of `backtest_tick_mm.py` and `strategy_core.py`

The full 21GB CSV was never loaded into memory as a table; counts were collected by streaming scans.
