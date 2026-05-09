# 2026-05-08 Future Plan

## Current Baseline

The 5-7 1H live sample remains the clean calibration baseline for replay/action
parity. The current post-fix live validation sample is `5-9-small`.

Stage 1 completed working-order semantic parity:

- working-order semantic mismatch: `0 / 126490`
- action/planned/reject/throttle match: `1.0`
- API/throttle mismatch rows: `0`
- latency drop BT/live: `0.030579492449996047 / 0.030579250697678095`

Stage 2 completed replay lag gate interpretation:

- strict full-window outside dual-gate rows: `131`
- all strict breaches are the startup block `strategy_seq=1..131`
- startup-excluded gate: `passed`
- post-startup rows: `126359`
- post-startup outside dual-gate rows: `0`
- post-startup in dual-gate rate: `1.0`
- first clean strategy seq: `132`

Archived baselines:

- `baselines/5-7-ver1-stage1-working-order-parity/`
- `baselines/5-7-ver1-stage2-replay-lag-gate/`
- `baselines/5-9-small-stage6g-working-order-action-api-parity/`

Current post-terminal-reconcile sample:

- run: `5-9-small_btcusdt_1778305776`
- archive: `local_live_analysis/archive/5-9-small.tar.gz`
- live rows: `94211`
- live decision rows: `71792`
- strict replay lag gate: `passed`, breaches `0`
- target tick parity: `1.0`
- audit overlay MAE for fair/reservation/position/inventory/spread/vol: `0.0`
- final live safety status before shutdown: `ok`
- final REST/local position: `-0.001 / -0.001`
- shutdown canceled `2` normal working orders and closed cleanly
- max abs position notional in audit replay: `241.18005`, below the `250` limit

Current Stage 6G post-repair alignment on `5-9-small`:

- strict replay lag gate: `passed`, breaches `0`
- replay lag dual-gate rows: `71787 / 71787`
- target tick parity: `1.0`
- working-order lifecycle semantic mismatch rows: `0`
- working-order blocking mismatch rows: `0`
- action/planned/reject/throttle match rates: `1.0 / 1.0 / 1.0 / 1.0`
- API/throttle mismatch rows: `0`
- BT/live API drop: `0.16748157744438408 / 0.16748157744438408`
- remaining working-order lifecycle mismatch is non-blocking diagnostic REST/local
  evidence only: `67366` rows, mostly `rest_open_order_count` and
  `rest_open_order_detail`

Interpretation:

- terminal cleanup / ghost-order failure from `5-8-night` is accepted as fixed
- market feed, target ticks, strict replay lag, working-order semantic state, and
  API/throttle action path are no longer current blockers for audit replay acceptance
- Stage 6G added an audit-only `working_order_overlay = "audit"` plus live lifecycle
  replay for hidden in-flight exposure; this is an acceptance replay tool, not a
  parameter-optimization mode
- optimization sweeps must keep `market_state_overlay`, `strategy_position_overlay`,
  and `working_order_overlay` off

## Stage 3: Fresh Live Sample Revalidation

Goal: prove the 5-7 result is not single-sample overfit.

Immediate run:

- collect a fresh `15min` live sample from awsserver1
- pull artifacts locally
- run `align_live_run.py`
- archive the run

Acceptance gates:

- working-order semantic mismatch rows: `0`
- API/throttle mismatch rows: `0`, or near zero with row-level explanation
- action/planned/reject/throttle match: near `1.0`
- `startup_excluded_gate.passed == true`
- post-startup outside dual-gate rows: `0`
- any full-window strict breaches must be confined to startup/prewarm
- latency drop BT/live remains aligned

If Stage 3 fails:

- middle-window replay lag breach: fix replay marker/feed timestamp alignment
- semantic working-order mismatch: inspect first semantic divergence and replay state overlay
- API/throttle mismatch with semantic parity: inspect guard timing and quote throttle state

Status: superseded by the current `5-9-small` acceptance baseline. Keep this stage
as the historical live revalidation slot, but do not reopen already-fixed action-path
issues here.

## Stage 4: Optimization Acceptance Contract

Goal: freeze the maker-optimization preconditions as a reusable gate.

Write `docs/maker_optimization_acceptance.md` with required report fields:

- `alignment.working_order_lifecycle.semantic_mismatch_rows == 0`
- `alignment.api_throttle.mismatch_attribution.mismatch_rows == 0`
- `alignment.planned_action_match_rate == 1.0`
- `alignment.reject_reason_match_rate == 1.0`
- `alignment.throttle_reason_match_rate == 1.0`
- `alignment.replay_lag.stateful_gate.startup_excluded_gate.passed == true`
- `alignment.replay_lag.stateful_gate.startup_excluded_gate.post_startup_outside_dual_gate_rows == 0`

Top5 feed-state parity remains diagnostic unless target ticks diverge.

Status: implemented on 2026-05-09.

- gate script: `examples/binance_tick_mm/maker_acceptance.py`
- contract doc: `docs/maker_optimization_acceptance.md`
- verified sample: `local_live_analysis/5-8-stage3-15m-livetest-v4`
- result: passed
- diagnostic working-order non-blocking mismatch rows: `60739`
- diagnostic top5 book state mismatch remains non-blocking because target ticks stayed aligned

Status note:

- contract unchanged
- current-format acceptance baseline is `baselines/5-9-small-stage6g-working-order-action-api-parity/`
- audit overlays stay off for optimization sweeps

## Stage 5: Maker Parameter Optimization Dry Run

Goal: verify optimization pipeline without treating a single sample as production truth.

Small sweep dimensions:

- base spread
- `k_vol`
- `k_inv`
- quote throttle interval / move ticks
- order notional
- two-phase replace settings

Reject parameters that profit only by:

- excessive API churn
- unsafe inventory exposure
- degraded latency/API drop behavior
- replay/working-order gate regression

Status: completed.

- sweep/rank pipeline is usable for optimization replay
- ranker now records the max-position limit source
- single-window dry run is only a pipeline validation, not a production parameter selection
- optimization replay must keep `market_state_overlay`, `strategy_position_overlay`,
  and `working_order_overlay` off

## Stage 6: Out-of-Sample Backtest Validation

Goal: prevent fitting the 5-7 sample.

Use at least 2-3 different market windows:

- low volatility
- high volatility
- directional / jumpy window

Acceptance:

- behavior constraints stable across windows
- alignment gates do not regress
- PnL comes from plausible spread capture / maker fills

Current Stage 6 findings:

- Stage 6B: the profitable `150ms` region did not pass 5-7 safety because max position reached about `320.153`, above the `250` notional target.
- Stage 6D: broad same-side cancel cooldown fixed 5-7 safety but cut off the 5-8 positive-PnL path, so cooldown should not be the main solution.
- Stage 6E: true in-flight exposure accounting reduced 5-7 max position to about `240.705` without cooldown, but the current 5-8 sample stayed negative.
- The 5-8 sample is only about `15min`, so it is not sufficient evidence to keep modifying strategy economics.
- The remaining cancel-requested fill path is a diagnostic hypothesis, not yet an approved strategy-rule change.

Stage 6E acceptance result:

- keep `risk.inventory_inflight_exposure_enabled = true` as the current safety improvement
- keep `risk.inventory_add_side_cancel_cooldown_ms = 0.0` unless used as a comparison baseline
- do not promote any 6E candidate to Stage 7 yet
- do not implement a narrower cancel-requested same-side rule solely from the 5-8 15min result

Live state reconciliation prerequisite:

- the `5-8-night` archive exposed a live ghost-order / terminal-state cleanup failure
- fixed by preserving late terminal updates across REST/WS channel reordering and by allowing
  live bot terminal-to-terminal overwrites
- verified by `5-9-small`: 1H live run completed without `open_order_mismatch`, strict replay
  lag gate passed, final safety status was `ok`
- see `docs/5-8-terminal-reconcile-plan.md`

Artifacts:

- `local_live_analysis/stage6_oos_validation/STAGE6E_INFLIGHT_EXPOSURE_SUMMARY.md`
- `local_live_analysis/stage6_oos_validation/stage6e_inflight_exposure_combined_candidates.csv`

## Stage 6F: Post-Fix Live Revalidation

Status: completed by `5-9-small`.

Goal: prove that the terminal reconcile patch works in a fresh current-format 1H
live run before trusting new live samples.

Result:

- live duration: about `1h`
- run completed without live safety hard stop
- final safety status: `ok`
- strict replay lag gate: `passed`
- strict replay lag breaches: `0`
- target tick parity: `1.0`
- archive checksum: passed
- max abs position notional: `241.18005`, below the `250` limit

Conclusion:

- terminal reconcile is accepted as fixed enough to move on
- `5-9-small` is valid evidence for the next alignment work
- the next blocker is working-order lifecycle/action-path parity, not raw feed or replay lag

## Stage 6G: Working-Order Lifecycle / Action-Path Parity Repair

Status: completed on `5-9-small`.

Goal: reduce the new `5-9-small` working-order semantic mismatch enough that
maker parameter optimization is measuring strategy quality, not state-path artifacts.

Primary sample:

- `local_live_analysis/5-9-small`

Observed blocker:

- blocking working-order mismatch remains high:
  - acceptance view: `71602` rows
  - API/throttle attribution view: `11961` rows
- action match rate is only `0.915945784055609`
- planned action match rate is only `0.7493278727346177`
- target tick parity is already `1.0`, so this is not a feed/target tick problem
- same-guard-input throttle subset is clean, so this is not primarily a quote-throttle
  formula problem

Root cause found:

- first cascade started before the first semantic mismatch: live planned two
  throttled `submit_buy` actions at `strategy_seq=181/182`, consuming order ids
  that backtest did not consume because replay-side in-flight exposure state had
  already diverged
- after order-id drift, keyed release/visibility overlays stopped matching the live
  order lifecycle, causing large semantic working-order mismatch
- a second diagnostic run with in-flight exposure disabled moved the first
  semantic divergence to a later live fill path, proving the remaining issue was
  lifecycle/in-flight ordering, not feed, target ticks, lag gate, or raw order-id
  comparison

Fix implemented:

- `backtest_tick_mm.py` now supports `backtest_cadence.working_order_overlay =
  "audit"` for audit replay
- audit replay can reconstruct decision-visible working orders directly from live
  `local_open_orders`
- audit replay replays live executed actions and live lifecycle rows into
  `InFlightExposureTracker` after each decision, preserving live's
  decision-before-lifecycle-observe ordering for hidden exposure
- `align_live_run.py` enables `working_order_overlay = "audit"` only for
  `audit_replay`
- `sweep_backtest.py` forces the working-order overlay off for optimization replay,
  and `rank_sweep.py` rejects rows where it is enabled

Completed work items:

- inspect first post-startup semantic divergence in `5-9-small`
- separate true live/backtest working-order state mismatch from comparator/audit-schema noise
- trace whether mismatches come from:
  - local working-order retention after REST/WS transitions
  - missing local order in replay state
  - extra local order in replay state
  - order quantity/status/request-state differences
  - REST-open-order diagnostic fields leaking into semantic acceptance
- fix the smallest layer that owns the mismatch:
  - replay lifecycle reconstruction if replay state is wrong
  - audit comparison if classification is too broad
  - live audit schema only if needed to expose missing state
  - strategy/backtest state transition only if behavior is truly divergent

Acceptance:

- strict replay lag gate remains `passed` with `0` breaches
- target tick parity remains `1.0`
- final live safety remains `ok` on validation samples
- blocking working-order mismatch is reduced to near zero, or every remaining row is
  explicitly reclassified as non-blocking diagnostic noise with examples
- planned action match rate returns near `1.0` after lifecycle parity is fixed
- do not tune maker parameters during this stage

Acceptance result:

- strict replay lag gate: `passed`, breaches `0`
- target tick parity: `1.0`
- working-order semantic mismatch rows: `0`
- working-order blocking mismatch rows: `0`
- planned action match rate: `1.0`
- action match rate: `1.0`
- reject/throttle reason match rates: `1.0 / 1.0`
- API/throttle mismatch rows: `0`
- `maker_acceptance.py` passed on `local_live_analysis/5-9-small`

## Stage 6H: API / Throttle Parity After Lifecycle Repair

Status: completed as a downstream result of Stage 6G.

Goal: verify that the API/throttle mismatch is a downstream consequence of
working-order/action-path divergence, and fix any remaining genuine throttle-state
differences.

Pre-fix evidence:

- overall BT/live API drop is misaligned: `0.0008358059258640144 / 0.16746991308223758`
- same-guard-input subset is clean:
  - rows: `181`
  - reject/throttle match: `1.0`
  - API drop diff: `0.0`
- therefore the likely main cause is action-path divergence, not the guard formulas.

Post-fix result:

- planned/action paths aligned after working-order lifecycle repair
- API drop BT/live became identical: `0.16748157744438408 / 0.16748157744438408`
- reject/throttle reason match rates became `1.0 / 1.0`
- API/throttle mismatch rows became `0`
- no additional quote-throttle formula change was needed

Acceptance:

- planned action match rate near `1.0`
- reject/throttle reason match rate near `1.0`
- API drop rate BT/live near equal
- `same_guard_inputs` remains clean
- no regression in strict lag gate, target tick parity, or live safety

## Stage 6I: Cancel-Requested Fill-Risk Measurement, Deferred

Goal: now that Stage 6G/6H are complete, determine whether cancel-requested fill
risk is a stable cross-window problem before adding another strategy rule.

Execution plan:

- see `docs/stage6i-cancel-requested-fill-risk-plan.md`
- Stage 6I is diagnostic only; it should not change maker quoting logic or risk rules
- the Stage 6G baseline remains the acceptance anchor for current-format samples

Why this is now the next diagnostic stage:

- `5-9-small` shows current replay/feed/terminal/working-order/API gates are
  healthy in audit replay
- previous 5-8 and 5-7 evidence suggested cancel-requested fill risk can affect
  inventory, but the evidence was contaminated by state-parity work and short
  samples
- adding a cancel-requested rule still requires cross-window evidence, because the
  broad cooldown already showed it can cut profitable paths

Data requirement after parity repair:

- use `5-9-small` plus at least 1-2 additional current-format live samples
- target `30min-1H` per sample where possible
- include different regimes if available: quiet, directional, jumpy/high-churn
- keep the current replay alignment gates unchanged

Metrics to add or extract:

- `fill_after_cancel_request_count`
- `fill_after_cancel_request_qty`
- `fill_after_cancel_request_notional`
- `same_side_readd_while_cancel_requested_count`
- `same_side_readd_while_cancel_requested_qty`
- `worsening_fill_after_cancel_request_count`
- inventory before/after cancel-requested fills
- PnL or markout around cancel-requested fills
- max-position contribution from cancel-requested fills

Decision:

- if the issue repeats across windows, design a narrow rule in Stage 6J
- if it is isolated, keep it diagnostic and continue broader OOS parameter search
- broad cooldown remains only a control comparison
- use `baselines/5-9-small-stage6g-working-order-action-api-parity/` as the current
  acceptance anchor for this phase

Stage 6I execution result:

- diagnostic extractor: `examples/binance_tick_mm/analyze_cancel_fill_risk.py`
- report: `local_live_analysis/stage6i_cancel_fill_risk/STAGE6I_CANCEL_FILL_RISK_SUMMARY.md`
- current-format samples: `5-9-small`, `5-8-stage3-15m-livetest-v4`
- historical/failed-gate controls: `5-7-ver1-livetest`, `5-8-night`
- decision: `diagnose_cancel_fill_risk_before_strategy_rule`
- cancel-requested fills repeated across current-format windows:
  - `5-9-small`: `18 / 51`, notional rate `0.353011`
  - `5-8-stage3-15m-livetest-v4`: `51 / 91`, notional rate `0.560311`
- side-adjusted markout was negative across measured horizons in both current-format windows
- same-side re-add followed by cancel-requested fill did not repeat across current-format windows:
  - `5-9-small`: `0`
  - `5-8-stage3-15m-livetest-v4`: `4`
- therefore Stage 6J should not yet be a direct same-side re-add blocking rule; next
  diagnostic should decompose cancel-requested fill source paths before any strategy change

Source-path decomposition:

- report now includes source-path and cancel-latency bucket attribution
- `5-9-small` current-format:
  - `inventory_worsening_no_readd`: `7` events, 1s weighted markout `-0.017950`
  - `inventory_reducing_cancel_race`: `11` events, 1s weighted markout `-0.056150`
  - same-side re-add overlap: `0`
- `5-8-stage3-15m-livetest-v4` current-format:
  - same-side re-add paths: `4` events, 1s weighted markout about `-0.027200`
  - `inventory_worsening_no_readd`: `22` events, 1s weighted markout `-0.148100`
  - `inventory_reducing_cancel_race`: `25` events, 1s weighted markout `-0.150950`
- current evidence says the negative markout is mostly a cancel-race/adverse-selection
  problem around cancel-requested fills, not primarily a same-side re-add stacking problem
- next rule design, if any, should therefore target measured cancel-race adverse selection
  or quote/replace timing, not only block same-side re-add

Stage 6J-A diagnostic rule implementation:

- implemented default-off `cancel_race_guard` hooks in live/backtest shared path
- config keys under `[risk]`:
  - `cancel_race_guard_enabled = false`
  - `cancel_race_guard_pending_cancel_block = true`
  - `cancel_race_guard_post_fill_cooldown_ms = 0.0`
- behavior:
  - blocks add-side submit for a side with pending cancel-requested in-flight exposure
  - optionally blocks add-side submit for a side shortly after a cancel-requested fill
  - allows reduce-side submit
  - does not change default strategy behavior when disabled
- audit fields:
  - `cancel_race_guard_buy_active`
  - `cancel_race_guard_sell_active`
- verification:
  - targeted unit tests passed
  - broader local tests passed: `126 passed`
  - `5-9-small` default-off replay still passed maker acceptance

Guard feasibility readout:

- normal backtest for `5-9-small` is not sufficient to evaluate the guard because it
  only emitted the initial `submit_buy|submit_sell` and no meaningful cancel/replace path
- offline source-path split estimates what a first add-side guard could plausibly cover:
  - `5-9-small`: add-side candidate `7` events, 1s weighted markout `-0.017950`;
    adverse-selection candidate `11` events, 1s weighted markout `-0.056150`
  - `5-8-stage3-15m-livetest-v4`: add-side candidate `26` events, 1s weighted
    markout `-0.175300`; adverse-selection candidate `25` events, 1s weighted
    markout `-0.150950`
- implication: add-side guard is a reasonable safety component, but it cannot be the
  whole Stage 6J solution; the remaining adverse-selection candidate path needs a
  separate quote/replace timing or toxicity-aware rule

## Stage 6J: Narrow Cancel-Requested Rule

Goal: design a narrow cancel-race rule without cutting profitable non-problem paths.

Current status:

- Stage 6J-A implemented a default-off diagnostic `cancel_race_guard`
- Stage 6I now has `3` current-format samples and decision
  `proceed_to_stage6j_narrow_rule`
- do not promote the rule yet; move from diagnosis to rule design and offline
  replay comparisons
- current evidence is time-of-day/regime confounded:
  - `5-9-small`: Beijing `2026-05-09 13:49-14:52`, about `63min`
  - `5-9-noon`: Beijing `2026-05-09 23:03-2026-05-10 00:04`,
    about `61min`; the name is misleading, actual regime is night-active.
    It now passes current-format acceptance after strict lag gate was aligned
    to exclude only leading startup breaches: total startup breaches `3`,
    post-startup breaches/failures/drops `0`; action/planned/reject/throttle,
    working semantic/blocking, and API/throttle gates passed.
  - `5-8-stage3-15m-livetest-v4`: Beijing `2026-05-08 23:27-23:42`, about `15min`
  - `5-7-ver1-livetest`: Beijing `2026-05-08 02:48-03:48`, about `60min`, failed strict lag gate so diagnostic only
  - `5-8-night`: Beijing `2026-05-09 10:53-11:24`, about `30min`, failed action/working/API gates so diagnostic only
- night/active-window samples show stronger cancel-requested fill risk:
  - `5-9-small` afternoon current-format: notional rate `0.353011`
  - `5-9-noon` night-active current-format: notional rate `0.373264`,
    add-side candidate `15`, adverse-selection candidate `13`, weighted 1s
    markout `-0.182250` / `-0.052850`
  - `5-8-stage3` late-night current-format: notional rate `0.560311`
  - `5-7` late-night failed-gate diagnostic: notional rate `0.709403`
- this still supports the hypothesis that `23:00-08:00` Beijing active
  windows can be more toxic for cancel-race fills.

## Stage 6J-B: Narrow Rule Design And Replay

Goal: turn the Stage 6I source-path evidence into explicit rule candidates and
measure them on existing current-format windows before any new live deployment.

Candidate rule components:

- add-side guard:
  - block add-side submit when same-side cancel-requested in-flight exposure is
    non-terminal
  - allow reduce-side submit
  - keep default off until selected
- post-cancel-fill side cooldown:
  - side-specific only
  - short window only, initially diagnostic values such as `50/100/200ms`
  - must not behave like the broad cooldown that cut off profitable paths
- adverse-selection timing rule:
  - widen or pause quoting only after measured toxic cancel-race conditions
  - candidate signals: short-horizon drift, cancel-fill markout history, order
    age, distance-to-top, high-churn regime, and cancel latency bucket
  - start as a diagnostic flag before production default

Required offline comparisons:

- baseline: no new rule, in-flight exposure only
- add-side guard only
- add-side guard plus side-specific post-fill cooldown
- adverse-selection timing rule, if implemented
- broad cooldown only as a control baseline

Acceptance:

- `maker_acceptance.py` still passes on replay acceptance samples
- action/planned/reject/throttle match remains `1.0`
- working semantic/blocking mismatch remains `0`
- API/throttle mismatch remains `0`
- strict lag gate post-startup breaches/failures/drops remain `0`
- source-path improvement is visible in the affected paths:
  - reduce `inventory_worsening_no_readd`
  - reduce same-side readd inventory-worsening where present
  - do not claim victory if only `inventory_reducing_cancel_race` remains
- max abs notional stays below `250`
- churn, latency drop, and API drop do not regress
- PnL does not improve only by suppressing most quoting

Current evidence to target:

- `5-9-noon`: add-side candidate `15`, adverse-selection candidate `13`,
  `inventory_worsening_no_readd` `10`, same-side readd paths `5`
- `5-9-small`: add-side candidate `7`, adverse-selection candidate `11`,
  no same-side readd overlap
- `5-8-stage3`: add-side candidate `26`, adverse-selection candidate `25`,
  `inventory_worsening_no_readd` `22`

Status: executed.

- runner: `examples/binance_tick_mm/stage6j_replay.py`
- output: `local_live_analysis/stage6j_narrow_rule_replay/`
- samples: `5-9-noon`, `5-9-small`, `5-8-stage3-15m-livetest-v4`
- candidates:
  - baseline in-flight only
  - add-side guard only
  - add-side guard plus `50/100/200ms` post-fill side cooldown
  - broad `200ms` add-side cooldown control
- replay contract:
  - audit cadence is allowed to reuse live decision timing
  - `market_state_overlay`, `strategy_position_overlay`, and
    `working_order_overlay` are forced `off`
  - `inventory_inflight_exposure_enabled = true`
- hard failures: `0`
- overlay contract: all `off/off/off`
- strict lag gate: all passed
- result:
  - baseline cross-window cancel-requested fill count: `3`
  - add-side guard variants reduce this to `0`
  - same-side worsening count drops from `3` to `0`
  - max abs notional stays below `250`; add-side guard max was about `161.132`
  - post-fill cooldown `50/100/200ms` produced no additional difference beyond
    add-side guard on these replay paths
  - broad cooldown control improved aggregate PnL in this diagnostic replay but
    raised max abs notional to about `241.080` and remains only a control because
    earlier Stage 6D showed broad cooldown can cut profitable paths
- decision: `diagnostic_only_no_promotion`
- interpretation:
  - add-side guard is a viable narrow component for the repeated same-side readd
    source path
  - this replay does not yet validate an adverse-selection timing rule because
    the optimization replay emitted only a small number of cancel-fill events
  - do not promote any Stage 6J rule to live from this result alone
  - next step is Stage 6J-C regime-control sampling, then repeat 6J-B/6J-D on
    a larger current-format sample set

## Stage 6J-C: Regime Control Sampling

Goal: keep time-of-day/regime effects separated from strategy-rule effects.

Required live samples:

- collect at least one new current-format `1H` sample in Beijing `23:00-08:00`
  - preferred: `23:00-01:00` or `02:00-04:00`
  - archive with a run id that records the regime, for example `5-9-night-active-1h`
  - `5-9-noon` already provides a current-format night-active sample, but
    the run id is misleading; repeat with a regime-correct run id if we need
    cleaner naming before promotion.
- collect at least one new current-format `1H` daytime control sample
  - preferred: `13:00-16:00` or `10:00-12:00`
  - archive with a run id that records the regime, for example `5-9-day-control-1h`

Per-sample gates:

- run `align_live_run.py`
- run `maker_acceptance.py`
- require current hard gates:
  - action/planned/reject/throttle match `1.0`
  - working semantic/blocking mismatch `0`
  - API/throttle mismatch `0`
  - strict replay lag breach/fail/drop `0`
  - BT/live latency/API drop aligned
- run `analyze_cancel_fill_risk.py`
- report:
  - cancel-requested fill notional rate
  - source-path split
  - guard candidate split
  - cancel-latency buckets
  - side-adjusted markout at `1s/5s/30s`

Decision matrix:

- if night active samples are materially worse and daytime samples are milder:
  design a regime-aware guard or stricter night-only configuration
- if night and daytime are both materially bad:
  promote a general cancel-race rule candidate and add quote/replace timing diagnostics
- if only one short night sample remains bad:
  collect more data; do not change production behavior
- if the add-side candidate path is small but adverse-selection path dominates:
  do not rely on same-side/add-side blocking alone; design toxicity-aware quote/replace timing

## Stage 6J-D: Rule Promotion Gate

Goal: choose whether a narrow rule is good enough for a micro live test.

Promotion criteria:

- at least two current-format windows per promoted regime, or one new current-format
  sample plus one previous current-format sample when the same time-of-day regime matches
- cancel-requested fill source-path improvement is visible in the affected regime
- 5-7/active-window style max position remains below `250`
- multiple OOS windows do not lose positive-PnL paths unnecessarily
- churn and API/latency drops do not regress
- action-path changes are explained by measured cancel-requested risk events
- guard-on behavior remains default-off until promoted by these criteria

## Stage 6K: Broader OOS Parameter Search

Goal: after Stage 6J chooses a rule state, search maker parameters that actually
change target ticks/action path, using optimization replay without audit overlays.

Candidate dimensions:

- quote interval: `100/150/200ms`
- minimum quote move ticks
- spread/tick offset parameters that visibly move target ticks
- inventory skew/soft-limit thresholds
- order sizing only after confirming it changes effective submitted quantity
- in-flight exposure enabled in all production candidates
- selected Stage 6J rule state:
  - no rule, or
  - add-side guard, or
  - add-side guard plus adverse-selection timing

Acceptance:

- no candidate advances from a single short sample
- require positive or at least stable PnL across multiple OOS windows
- require safety gates before optimizing score
- explicitly report when a grid dimension produced identical action paths
- reject any candidate whose sweep result used audit `market_state`,
  `strategy_position`, or `working_order` overlays
- keep this stage tied to optimization replay outputs, not live acceptance archives
- compare rule-on and rule-off ranks to make sure the rule is not masking bad
  parameters

## Stage 7: Shadow or Micro Live Validation

Goal: verify the selected Stage 6J/6K candidate in low-risk live conditions.

Start with shadow or very small notional:

- open-order safety remains healthy
- API/throttle behavior matches replay
- fill/cancel lifecycle has no new semantic mismatch
- live PnL attribution direction is consistent with backtest
- cancel-requested fill source-path metrics move in the expected direction
- run id must encode regime and rule state

Entry condition:

- Stage 6G/6H restored working-order/action/API parity on current-format samples
- Stage 6I/6J/6K produce a cross-window candidate
- candidate satisfies alignment, risk, drop, churn, and OOS PnL gates
- no rule is promoted only because it improves the 5-8 15min sample
