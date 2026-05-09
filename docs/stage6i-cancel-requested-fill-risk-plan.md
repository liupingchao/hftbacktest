# Stage 6I Cancel-Requested Fill-Risk Cross-Window Diagnostic Plan

## Goal

Determine whether cancel-requested fill risk is a repeatable cross-window issue before adding any new strategy rule.

Stage 6I is diagnostic only. It must not change maker quoting logic, risk guards, throttles, or sweep ranking. The output is evidence for deciding whether Stage 6J should implement a narrow cancel-requested same-side rule, or whether the issue should remain diagnostic while broader OOS parameter search continues.

## Current Anchor

Use the Stage 6G baseline as the current acceptance anchor:

- baseline: `baselines/5-9-small-stage6g-working-order-action-api-parity/`
- run: `5-9-small`
- action/planned/reject/throttle match: `1.0 / 1.0 / 1.0 / 1.0`
- working semantic mismatch rows: `0`
- working blocking mismatch rows: `0`
- API/throttle mismatch rows: `0`
- strict replay lag breaches: `0`
- post-startup outside dual-gate rows: `0`

The remaining REST/local working-order diagnostic differences are not the blocker for this stage. The blocker under investigation is whether fills after cancel request materially worsen inventory, markout, or peak exposure.

## Scope

In scope:

- extract cancel-requested fill events from live audit lifecycle rows
- detect same-side re-add while same-side cancel-requested orders are still non-terminal
- measure inventory and markout around those events
- compare the same metrics across multiple current-format windows
- decide whether Stage 6J is justified

Out of scope:

- no new strategy rule in Stage 6I
- no parameter promotion from this stage alone
- no use of audit overlays for optimization replay
- no broad cooldown promotion; broad cooldown remains a control comparison only

## Required Samples

Minimum completion set:

- `5-9-small` as the current-format anchor sample
- at least one additional current-format live sample that passes the same Stage 6G acceptance gates

Preferred set:

- `5-9-small`
- one 30min-1H quiet or normal-churn window
- one 30min-1H directional, jumpy, or high-churn window

Historical samples such as `5-7-ver1-livetest`, `5-8-stage3-15m-livetest-v4`, or `5-8-night` can be included only as historical/control evidence unless they are rerun through the current acceptance gates and pass. They must not be mixed into the current-format aggregate without a label.

## Preflight Gates Per Sample

For every sample counted as current-format evidence:

```bash
python examples/binance_tick_mm/align_live_run.py \
  --run-id <RUN_ID> \
  --local-root local_live_analysis \
  --skip-fetch \
  --skip-archive \
  --latency-mode observed
```

```bash
python examples/binance_tick_mm/maker_acceptance.py \
  --alignment-report local_live_analysis/<RUN_ID>/alignment_report_audit_replay.json \
  --backtest-result local_live_analysis/<RUN_ID>/backtest_audit_replay_result.json \
  --out local_live_analysis/<RUN_ID>/maker_acceptance.json
```

Hard requirements:

- `action_match_rate == 1.0`
- `planned_action_match_rate == 1.0`
- `reject_reason_match_rate == 1.0`
- `throttle_reason_match_rate == 1.0`
- `working_order_lifecycle.semantic_mismatch_rows == 0`
- `working_order_lifecycle.blocking_mismatch_rows == 0`
- `api_throttle.mismatch_attribution.mismatch_rows == 0`
- strict replay lag gate passed with `breach_count == 0`
- post-startup dual-gate outside rows are `0`
- BT/live API and latency drop rates remain aligned by the maker acceptance contract

If a sample fails these gates, Stage 6I may still inspect it manually, but it must be labeled `historical_or_failed_gate` and excluded from current-format decision thresholds.

## Extractor Plan

Add a small diagnostic script:

- path: `examples/binance_tick_mm/analyze_cancel_fill_risk.py`
- inputs:
  - one or more `local_live_analysis/<RUN_ID>/audit_live_<RUN_ID>.csv`
  - optional run labels and output directory
- outputs:
  - `local_live_analysis/stage6i_cancel_fill_risk/<RUN_ID>/cancel_fill_events.csv`
  - `local_live_analysis/stage6i_cancel_fill_risk/<RUN_ID>/cancel_fill_summary.json`
  - `local_live_analysis/stage6i_cancel_fill_risk/stage6i_cancel_fill_summary.csv`
  - `local_live_analysis/stage6i_cancel_fill_risk/STAGE6I_CANCEL_FILL_RISK_SUMMARY.md`

The script should use structured CSV fields instead of parsing display strings wherever possible.

## Event Definitions

### Fill After Cancel Request

Count a lifecycle row as `fill_after_cancel_request` when:

- `event_type` is `fill` or `partial_fill`
- and either `fill_after_cancel_request == 1` or `cancel_request_ts > 0`

For each event, capture:

- `run_id`
- `ts_local`
- `order_id`
- `order_side`
- `event_type`
- `status`
- `price`
- `qty`
- `executed_qty`
- `leaves_qty`
- `cancel_request_ts`
- delay from cancel request to fill
- inventory immediately before and after the fill where available
- mid price at fill and at markout horizons where available

### Same-Side Re-Add While Cancel Requested

Maintain a per-run pending-cancel set:

- add order to pending-cancel when a `cancel_sent` or cancel-requested lifecycle row appears
- remove it when the order reaches terminal or effectively terminal lifecycle state:
  - `cancel_ack`
  - `fill`
  - `expired`
  - `rejected`
  - status `filled`
  - status `canceled`
  - status `expired`
  - status `rejected`

Count a same-side re-add when a submit action appears while a pending-cancel order with the same `order_side` still exists.

The diagnostic should separately count:

- any same-side re-add while cancel requested
- same-side re-add that increases current inventory direction
- same-side re-add followed by a fill-after-cancel-request before the pending cancel becomes terminal

## Metrics

Per run:

- `total_fill_count`
- `total_fill_qty`
- `total_fill_notional`
- `fill_after_cancel_request_count`
- `fill_after_cancel_request_qty`
- `fill_after_cancel_request_notional`
- `fill_after_cancel_request_count_rate`
- `fill_after_cancel_request_notional_rate`
- buy/sell split for cancel-requested fills
- `same_side_readd_while_cancel_requested_count`
- `same_side_readd_while_cancel_requested_qty`
- `same_side_readd_then_cancel_fill_count`
- `worsening_fill_after_cancel_request_count`
- `worsening_fill_after_cancel_request_notional`
- cancel-request-to-fill latency p50/p90/max
- inventory before/after cancel-requested fills
- max absolute inventory contribution from cancel-requested fills
- markout around cancel-requested fills at `1s`, `5s`, and `30s` where mid data exists
- approximate PnL or adverse markout contribution

Cross-window:

- repeat count: number of current-format windows with non-trivial cancel-requested fill impact
- median and max notional-rate across windows
- median adverse markout across windows
- whether the issue appears only in one short sample or across regimes

## Markout Direction

Use side-adjusted markout:

- buy fill is adverse when future mid is below fill price
- sell fill is adverse when future mid is above fill price

For a fill with signed direction `+1` for buy and `-1` for sell:

```text
side_adjusted_markout = direction * (future_mid - fill_price)
```

Positive is favorable, negative is adverse. Report both per-unit markout and notional-weighted markout.

## Decision Criteria

Proceed to Stage 6J only if all are true:

- at least two current-format windows pass the Stage 6G alignment gates
- cancel-requested fills are non-trivial in at least two current-format windows
- same-side re-add overlap materially contributes to either peak inventory or adverse markout
- the proposed rule can be narrower than the prior broad cooldown control

Default non-trivial thresholds:

- `fill_after_cancel_request_notional_rate >= 0.10` in at least two windows, or
- cancel-requested fills explain a material share of adverse markout or peak inventory in at least two windows, or
- a small count of events directly explains a large max-position excursion or loss path

Do not proceed to Stage 6J if:

- the issue appears only in one short or failed-gate sample
- same-side re-add does not overlap with harmful cancel-requested fills
- the measured effect is dominated by normal in-cap inventory noise
- the only effective mitigation looks like broad cooldown that cuts profitable paths

If Stage 6J is not justified, continue to Stage 6K broader OOS parameter search with the diagnostic retained.

## Acceptance For Stage 6I

Stage 6I is complete when:

- the extractor exists and has focused unit tests for:
  - fill-after-cancel-request detection
  - pending-cancel lifecycle termination
  - same-side re-add detection
  - side-adjusted markout direction
- at least `5-9-small` is processed successfully
- at least one additional current-format sample is processed, or the report explicitly states that more live data is required
- the cross-window Markdown report is generated
- the report gives one of these decisions:
  - proceed to Stage 6J narrow rule
  - do not implement rule; continue Stage 6K
  - collect more current-format data before deciding

## Execution Order

1. Implement `analyze_cancel_fill_risk.py` and unit tests.
2. Run tests for the extractor.
3. Process `5-9-small`.
4. Re-run acceptance for candidate local samples and classify them as current-format or historical/control.
5. Process all usable samples.
6. Generate cross-window summary.
7. Update `docs/5-8-future-plan.md` with the Stage 6I result and the next-stage decision.

## Expected Next Stage

Likely outcomes:

- if repeated and material: Stage 6J implements a narrow same-side cancel-requested rule
- if isolated or immaterial: Stage 6K resumes broader OOS parameter search
- if data is insufficient: collect another 30min-1H current-format live sample before any strategy change
