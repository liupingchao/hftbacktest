# Findings

## Open Findings

- `0510T001` completed the first workflow-run test against `5-10-day-control-1h-06` and is awaiting QA.
- `0510T002` completed cross-sample Stage 6J replay and is awaiting QA.
- `0511T001` should only design adverse-selection timing as a default-off rule using decision-time-visible inputs.
- `0511T002` should perform implementation only after the T001 design contract is accepted.
- `0511T003` should improve dashboard visibility of business results, QA conclusions, key metrics, and next-step decisions.
- `0511T004` completed the first diagnosis of why pure adverse timing candidates matched baseline and is awaiting QA.
- `0512T001` passed QA and established that Stage 6J replay cannot alone prove live adverse-selection source-path improvement.
- `0512T002` passed QA and authorized only the post-design implementation/offline replay task, not live.
- `0512T003` passed QA; `5-11-night-active` is accepted as the main current-format development/diagnostic sample, but does not authorize live.
- `0512T004` passed QA and authorizes `0512T002` only as a design-contract task.
- `0512T005` passed QA. It completed implementation/offline replay and does not authorize live.
- `0512T006` passed QA. It remains planning-only and authorizes only a later read-only T007 attribution implementation, not stricter candidate design, new replay, or live.
- `0512T007` passed QA. It is read-only attribution and does not authorize stricter candidate design, new replay, or live.
- `0512T008` passed QA. It is a read-only market-data / strategy-view quality gate and does not authorize strategy changes, new replay, or live.
- `0513T001` passed QA. It authorizes only bounded `0513T002` implementation for strategy-layer MarketView provenance / top5 audit transparency, not live, replay candidates, core API changes, converter/npz changes, or microprice/OFI/queue work.
- `0513T002` passed QA. It stays within the T001 file boundary and does not authorize live, replay candidates, core API changes, converter/npz changes, or strategy-rule changes.
- `0513T003` passed QA. It validated T002 provenance fields on `5-13-day-control-15min`, but does not prove PnL or full L2 alignment.
- `0513T004` passed QA. It adds a local deployment reproducibility and startup compatibility gate; no live, AWS, or strategy-rule change is authorized by it.
- `0513T005` passed QA as the Step 2 planning task for latency and market-data integrity baseline. It is planning-only and does not authorize code implementation, replay, live, or core/data schema changes.
- `0513T006` QA passed. It generated Step 2 read-only artifacts over existing local samples and does not authorize live, strategy changes, replay sweeps, or core/data schema changes.
- `0513T007` QA failed. It targets Binance raw provenance / top5 sidecar and read-only decision join, but full-run validation exposed a snapshot/bootstrap bug in the reconstructed top5 sidecar.
- `0513T008` QA passed. It collected one fresh no-rule control sample `5-13-day-control-30min` using T004 preflight and T007 full-run sidecar/join checks; it does not authorize new strategy rules or live promotion.
- `0513T009` QA passed. It fixes the T007 snapshot bootstrap / buffered depth replay bug using the existing `5-13-day-control-30min` sample only.
- `0514T001` QA passed. It implements Stage 3 market-view acceptance directly using `5-13-day-control-30min`; a separate planning-only task was not needed because Step 2 already supplied the required facts and artifacts.
- `0514T002` QA passed. It is planning-only for Stage 4 read-only pricing-model research and authorizes `0514T003` as a read-only implementation task.
- `0514T003` QA passed. It generated the Stage 4 read-only pricing-model research artifacts on `5-13-day-control-30min`.
- `0514T004` passed QA as a requirements-only follow-up for maker execution outcome research. It includes the seven added label gaps and per-label statistical method requirements, but does not itself authorize implementation or experiments.
- `0514T005` passed QA. It implemented the T004 requirements as a read-only execution outcome label runner with tests and dataset validation on `5-13-day-control-30min`.
- `0514T006` passed QA. It refines Stage 6 into replay/live fill-cancel lifecycle proxy calibration, defines matched-submit comparison as the primary unit, and does not authorize strategy changes, live, or exact queue proof.
- `0514T007` passed QA. It implemented the read-only Stage 6B replay/live lifecycle calibration runner on `5-13-day-control-30min` and concluded `diagnostic_only_gap_too_large`.
- `0514T008` passed QA. It concludes that the next useful task should diagnose replay fill/cancel lifecycle mismatch before sample-first expansion or quote-adjustment promotion discussion.
- `0515T001` passed QA. It built read-only diagnosis tables for replay-only fills, cancel timeline mismatches, terminal-state mismatches, and strata hot spots before any replay repair or sample expansion task.
- `0515T002` passed QA. It converts the mismatch evidence into a replay repair design contract with hypotheses, minimal scope, and validation gates.
- `0515T003` completed the narrow replay lifecycle repair and reduced the core same-sample mismatch materially, but 2 residual matched-submit cases remain and should be diagnosed before any follow-up repair.
- `0515T004` passed QA. It is now the accepted fact source for residual-case follow-up.

## 0515T001 Findings

- `0515T001` implements a read-only replay lifecycle mismatch diagnosis runner over the same matched submit opportunity universe used by Stage 6B.
- On `5-13-day-control-30min`, replay-only fills are highly concentrated in live-canceled / replay-filled cases: `120` replay-only fill rows and `120` live-cancel / replay-filled rows.
- Replay-only fills are not mainly ultra-short-horizon events: among replay-only fill cases, `100ms=0`, `500ms=4`, `1000ms=6`, `5000ms=31`. This supports a long-horizon persistence bias hypothesis.
- Cancel timeline evidence points toward replay-side terminal / cancel-ack persistence issues rather than submit matching issues: many rows show live cancel-ack already reached while replay keeps the same submit key fill-eligible and later marks it filled.
- Terminal-state mismatch is replay-side dominant: the main transition pattern is `canceled -> filled`, with additional `canceled -> open_or_missing`, rather than symmetric noise.
- Placement hot spots are concentrated in deeper step-back orders: `step_back_gt1` shows replay-only-fill rate about `0.0569`, terminal-state-diff rate about `0.1113`, and large time-to-fill gap.
- Latency buckets `q4` / `q5` and some `q2` buckets also show especially large fill-after-cancel and time-to-fill gaps, indicating the mismatch is not uniform across the sample.
- Markout observability mismatch is likely lifecycle-induced: replay creates more fills first, which then creates more observable markout rows. It does not currently read as an independent future-price sampling bug.
- The evidence is strong enough to justify a separate replay repair task next. Sample expansion should remain later validation work, not the immediate next step.
- For `0512T004` and `0512T002`, `5-11-night-active` is the main development/diagnostic sample; `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small` are cross-sample sanity checks.

## 0515T003 Findings

- `0515T003` implemented a narrow replay lifecycle repair in `audit_replay`, without touching strategy pricing, fair/reservation, quote placement, live collection, or sample policy.
- The repair uses live terminal constraints as an upper bound on replay lifecycle:
  - if live has already terminalized an order as `canceled` / `expired` / `rejected` by the current replay decision time, replay no longer keeps that order fill-eligible
  - replay `fill` / `partial_fill` / `order_update` events that violate that live terminal boundary are rewritten into terminal lifecycle events
  - if replay emits no further lifecycle event but live has already terminalized and the order is no longer visible in live working state, a synthetic terminal event is injected so the mismatch does not merely move from `filled` to `open_or_missing`
- Same-sample regression on `5-13-day-control-30min` shows the core mismatch has been materially reduced:
  - replay filled orders: `172 -> 53`
  - replay fill-after-cancel orders: `133 -> 14`
  - replay-only fill rows: `120 -> 1`
  - live-cancel / replay-filled rows: `120 -> 1`
  - terminal-state diff rows: `230 -> 2`
- Aggregate final-state gaps are now fully aligned on the matched submit universe:
  - `canceled`: `0.09062 -> 0.0`
  - `filled`: `0.04730 -> 0.0`
  - `open_or_missing`: `0.04332 -> 0.0`
- The original hot spots are no longer structural:
  - `step_back_gt1` replay-only-fill rate: `0.05693 -> 0.0`
  - `step_back_gt1` terminal-state-diff rate: `0.11134 -> 0.0`
  - latency `q5` replay-only-fill rate: `0.05347 -> 0.0`
  - latency `q2` terminal-state-diff rate: `0.10736 -> 0.00199`
- Residual mismatch remains in only two matched submits:
  - one `live_filled_replay_canceled`
  - one `live_canceled_replay_filled`
- Stage 6B decision state moves from `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`. This means the dominant replay lifecycle defect is no longer the blocker on this sample; the next decision point should treat remaining issues as residual-case diagnosis or broader-sample validation, not the original large-scale lifecycle mismatch.
- Initial residual interpretation is now split into two mechanisms, but not yet enough for another repair:
  - `live_filled_replay_canceled` looks like a short cancel-race fill miss: live filled about `9.43ms` after cancel request, replay canceled instead.
  - `live_canceled_replay_filled` looks like an optimistic touch fill false positive: replay filled about `262ms` before live cancel request, while live never filled.
- The next useful step is a small read-only residual diagnosis task that classifies whether the remaining explanation is short cancel-race window miss, touch fill optimism, or submit-after-queue exposure approximation bias. Do not open another repair task until that residual trigger evidence is written down.

## 0515T004 Findings

- `0515T004` stayed read-only and only analyzed the 2 residual matched-submit replay/live mismatches left after `0515T003`.
- The two residual cases do not support the same explanation:
  1. `3879|buy` / order `572`
     - live `cancel_request -> fill` delay is about `9.43ms`
     - live fill is preceded by dense supportive trades; `10ms` before the live fill there are `41` supportive raw trades
     - replay instead terminalizes to `cancel_ack`
     - this is strong evidence for `cancel_race_window_too_short`
  2. `28940|sell` / order `4948`
     - replay fill happens about `262.28ms` before live cancel request
     - therefore it is not a cancel-after-fill race case
     - but the current raw-trade check finds `0` supportive trades in the `10/25/50ms` windows before the replay fill
     - this means the current evidence is not strong enough to safely call it `touch_fill_assumption_too_optimistic`
- The second residual is therefore best kept as `residual_replay_fill_trigger_uncertain`, not over-claimed as a known queue/touch bug.
- The practical implication is asymmetric:
  - a very narrow follow-up repair can be justified for the short cancel-race miss class
  - a general residual replay-fill repair is not yet justified for the remaining replay-only fill case without stronger trigger evidence
- A follow-up implementation task should therefore be scoped as a narrow cancel-race residual repair only. It should not include `4948` / `residual_replay_fill_trigger_uncertain`, and it should not be written as a generalized touch/queue repair.
- `0515T005` is not a plan-only task. It is already the next narrow implementation task and should proceed only after keeping that scope restriction intact.

## 0515T005 Findings

- `0515T005` stayed within the narrow repair boundary and only addressed the `cancel_race_window_too_short` residual class.
- The implementation did not introduce a generalized touch-fill or queue-proxy repair. It only maps a replay `cancel_ack` into a replay `fill` when live has already confirmed a short-window fill-after-cancel-request event.
- Same-sample regression outcome:
  - `live_filled_replay_canceled` residual count: `1 -> 0`
  - Stage 6E residual case count: `2 -> 1`
  - remaining residual case is only `28940|sell` / `4948`
- Aggregate metrics moved only slightly and remained aligned:
  - replay filled orders: `53 -> 54`
  - replay fill-after-cancel orders: `14 -> 15`
  - final-state gaps stayed near-zero (`~0.000397` on filled/canceled rates)
- The important controller fact is that `572` has been removed without broadening the replay fill model, while `4948` remains intentionally untouched and still uncertain.
- The next useful follow-up should stay read-only and single-case: diagnose `4948` more deeply before authorizing any touch/queue repair.

## 0515T006 Findings

- `0515T006` stayed read-only and focused only on the single residual case `28940|sell` / `4948`.
- The earlier `4948` uncertainty was partly a diagnosis-limit issue: float price comparison hid same-price supportive trades at `81132.7`.
- After tick-normalized single-case analysis:
  - replay fill occurs about `262.28ms` before live cancel request
  - supportive trades before replay fill are present and dense:
    - `10ms`: `13`
    - `25ms`: `13`
    - `50ms`: `13`
    - `100ms`: `14`
  - nearest supportive trade is only about `0.503ms` before replay fill
  - replay fill happens while the order is still at touch (`ask_top1 = 81132.7`)
- This means `4948` is no longer best described as “unknown trigger”. It is better described as `queue_exposure_proxy_bias_possible`:
  - there is visible market activity that could fill a touch order
  - but live did not fill and later canceled
  - the likely gap is replay-side queue / priority approximation rather than hidden trigger absence
- Even after that narrowing, the evidence is still not enough for an immediate repair task:
  - no exact queue-position proof
  - still only one case
  - not enough basis to safely change generalized touch/queue fill behavior
- `0515T006` has now passed QA. The accepted conclusion is:
  - `4948` is best treated as `queue_exposure_proxy_bias_possible`
  - but no repair task should be opened yet unless stronger queue / repeatability evidence is added
- The next follow-up should therefore be another single-case read-only task, focused specifically on queue / priority evidence for `4948`, not on repair implementation.

## 0516T001 Findings

- `0516T001` stayed read-only and only analyzed queue / priority / exposure evidence for `28940|sell` / `4948`.
- The order remained at touch from submit to the replay fill window:
  - `order_at_touch_share_submit_to_replay_fill = 1.0`
  - order price tick `811327`, side `sell`
- Same-price supportive aggressive trades existed, but their cumulative quantity was below visible touch depth:
  - submit -> replay fill same-price trade count `31`
  - submit -> replay fill same-price trade qty `8.884`
  - submit visible ask qty at the order price `21.143`
  - replay-fill visible ask qty at the order price `15.633`
  - same-price qty / submit visible qty `0.4202`
  - same-price qty / replay-fill visible qty `0.5683`
- The practical diagnosis is now stronger than `queue_exposure_proxy_bias_possible`: `4948` is best classified as `queue_ahead_depth_can_absorb_observed_trades`.
- Interpretation:
  - replay was not filling from a hidden trigger; market trades did hit the order price
  - live could still plausibly remain unfilled because visible queue ahead was large enough to absorb the observed same-price trade quantity
  - replay likely lacks queue-ahead / priority / order-exposure state and treats touch-level supportive trades too optimistically for this case
- This is still not enough for direct repair implementation:
  - no exact queue position
  - no order-id-level queue depletion proof
  - still only a single residual case
  - a future task, if created, should be repair-design first and should not directly change generalized queue/touch fill behavior
- `0516T002` has been created as the next read-only step to test repeatability within the existing `5-13-day-control-30min` sample:
  - find touch no-fill / replay-fill / fill-candidate cases similar to `4948`
  - compute same-price trade qty / visible qty, top1 visible qty decay, unexplained depth shrink, touch duration, quote age, join age, and latency
  - decide whether `4948` is an isolated residual or part of a repeatable queue-ahead proxy mismatch pattern

## 0516T002 Findings

- `0516T002` stayed read-only and scanned the existing `5-13-day-control-30min` matched submit universe for queue-ahead proxy repeatability.
- It found `366` live no-fill / later-canceled touch candidates with same-price aggressive trades:
  - `365` are proxy-only candidates
  - `1` is a replay-fill candidate
  - all `366` satisfy queue-ahead mismatch under the current visible-queue proxy
  - `325` are strong queue-ahead mismatch candidates with high touch share
- The only replay-fill queue-ahead mismatch remains `4948`:
  - `replay_fill_queue_ahead_mismatch_cases = 1`
  - `target_4948_cases = 1`
- Interpretation:
  - queue-ahead no-fill behavior is repeatable in the sample
  - replay usually does not falsely fill those cases after the `0515T003` / `0515T005` repairs
  - `4948` remains the only replay false-positive version of that proxy pattern
- This does not justify generalized queue/touch repair yet:
  - repeatability exists for the proxy-only no-fill phenomenon
  - repeatability does not yet exist for replay-fill false positives
  - the accepted next step should be QA and then a decision on whether to create a repair-design-only task or stop at documented replay limitation
- `0518T001` has been created as a repair-design-only follow-up:
  - describe `4948` and the queue-ahead proxy evidence clearly
  - design a conservative queue proxy gate for future use
  - explicitly defer implementation until more current-format samples or more replay false-positive cases exist
  - do not modify replay, strategy, live collection, or sample policy

## Known Repository Notes

- The repository is a Rust workspace with multiple crates.
- The active user work is under `examples/binance_tick_mm/`.
- Existing planning docs under `docs/` are part of the current project state and should not be ignored.
- `docs/maker_optimization_acceptance.md` defines the current hard gates before maker optimization.
- `docs/5-8-future-plan.md` records the current Stage 6G/5-9-small baseline and next-stage context.

## Strategy Design Principles

- Maker strategy optimization is a system engineering problem. Do not chase a single extreme component, such as mandatory full L2 provenance or exact queue position, while leaving fair price, execution, risk, latency, or replay/live alignment below the acceptance line.
- Current task design should raise every layer above a usable and verifiable baseline: data view, fair price, pricing signal generation, strategy logic, risk guards, execution mechanics, and replay/live alignment.
- A weak layer can dominate the whole strategy PnL. Future tasks should identify and raise the weakest accepted layer before adding complexity to an already adequate layer.
- Top5 provenance is the current practical data boundary for near-term pricing / OFI proxy / microprice proxy work. Full L2 provenance and exact queue position are later enhancements, not current Step 2 blockers, unless top5 evidence proves insufficient.
- Queue-related work under the current boundary should be named as top-of-book/top5 size and age proxy work, not exact queue-position modeling.
- Strategy changes should remain incremental and evidence-layered: first prove data/action-path coverage, then replay-model behavior, then live-derived source-path proof before any live promotion.

## 0513T008 Findings

- `5-13-day-control-30min` was collected with T004 preflight on remote clean worktree commit `f228950`; manifest reports `dirty=false`, schema compatibility passed, start marker exists, stop marker exists, and stop marker exit code is `0`.
- `align_live_run.py` and `maker_acceptance.py` passed. Audit replay common rows: `47496`; action/planned/reject/throttle match rates are all `1.0`; working-order semantic/blocking mismatch `0/0`; strict replay lag breach/drop/fail `0/0/0`.
- T007 full-run sidecar metrics: raw messages `461402`, npz rows `2418372`, final data row mapping coverage `1.0`, depth `pu` mismatch `0`, snapshot alignment status `present`, bookTicker/depth BBO match/mismatch `67306/11`.
- T007 decision join metrics: decision rows `47499`, join coverage `1.0`, future join count `0`, missing join count `0`, stale join count `432`, top5 join age p50/p90/p99 `13.602/24.206/28.134ms`.
- Negative data-quality result: first valid depth update does not satisfy Binance snapshot/update alignment (`first_valid_update_aligned=false`), and every joined decision is marked `gap_crossed` (`47499/47499`).
- Classification: `pricing_research_candidate`, but only for compressed action-path acceptance and BBO/bookTicker/compressed-mid sanity. It is not usable yet for top5 microprice / top5 OFI proxy or queue/fill proxy research.
- Root cause for the T007 sidecar negative result is a framework bug, not run duration. Snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered depth `raw_seq=5` has `U=10537138802913, u=10537138805036` and covers `lastUpdateId+1=10537138804219`; depth `raw_seq=7` has `pu=10537138805036` and should chain after `raw_seq=5`. Current T007 ignores the buffered pre-snapshot update and starts bootstrap at `raw_seq=7`, which makes `first_valid_update_aligned=false` and propagates `sync_gap=true` to every joined decision.

## 0513T009 Findings

- T009 fixes the T007 sidecar bootstrap bug by buffering pre-snapshot depthUpdate messages and replaying the first buffered update satisfying `U <= lastUpdateId + 1 <= u` after the snapshot arrives.
- On `5-13-day-control-30min`, snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered `raw_seq=5` has `U=10537138802913`, `u=10537138805036`, and covers `10537138804219`; future `raw_seq=7` has `pu=10537138805036` and chains from buffered `raw_seq=5`.
- Fixed full-run metrics: raw messages `461402`, npz rows `2418372`, final data row mapping coverage `1.0`, depth `pu` mismatch count `0`, first valid update aligned `true`, top5 rows `67322`, and bookTicker/depth BBO match/mismatch `67307/11`.
- Fixed decision join metrics: decision rows `47499`, join coverage `1.0`, future join count `0`, missing join count `0`, stale join count `432`, gap-crossed join count `0`, and top5 join age p50/p90/p99 `13.602/24.206/28.134ms`.
- Fields now aligned enough for Step 2: `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, `final_data_row_mapping_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `join_missing_count=0`, and `gap_crossed_join_count=0`.
- Fields not exactly aligned: live audit top5 vs T009 sidecar reconstructed top5 per-row exact matches remain partial: best bid tick `30787/47499`, best ask tick `30780/47499`, bid full top5 ticks `8898/47499`, ask full top5 ticks `12393/47499`, bid top5 qtys `8600/47499`, ask top5 qtys `11579/47499`, bid+ask full top5 ticks `3539/47499`, and all bid/ask ticks+qtys `3324/47499`.
- `join_stale` is not zero: stale join count `432`, max top5 join age `251.033249ms`, and max bookTicker join age `603.705003ms`.
- The sample is upgraded for later top5 microprice / top5 OFI proxy / top5 imbalance pricing research candidates, subject to explicit handling of remaining bookTicker BBO mismatches and stale bookTicker-age rows.
- T009 does not start live, recollect data, modify strategy behavior, modify core/connector APIs, or change the standard hftbacktest npz main event schema. It does not prove full L2 equivalence, exact queue position, queue/fill model correctness, strategy PnL, or live promotion readiness.
- Do not keep extending Step 2 to force live audit top5 and sidecar reconstructed top5 into exact equality. The next useful work is Step 3: formalize market-view acceptance thresholds for top5 tick/qty match, BBO drift, source fields, startup exclusion, stale age, and sample classification.

## 0514T001 Findings

- `5-13-day-control-30min` is sufficient for Stage 3 verification because it has T004 manifest/action-path acceptance, T009 fixed sidecar metrics, T009 joined-decision metrics, sidecar provenance CSVs, and known partial top5 tick/qty alignment diagnostics.
- Stage 3 is implemented as an optional market-view gate in `maker_acceptance.py`; existing action/planned/reject/throttle/working-order/replay-lag hard gates remain unchanged when no sidecar metrics are provided.
- Stage 3 required gates: `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, `final_data_row_mapping_coverage>=1.0`, `decision_join_coverage>=1.0`, `future_join_count=0`, `join_missing_count=0`, and `gap_crossed_join_count=0`.
- Stage 3 quality gates: bookTicker/depth BBO mismatch rate `<=0.001`, stale join rate `<=0.02`, top5 join age p99 `<=50ms`, best bid/ask tick match rate `>=0.80`, top5 tick match rate `>=0.80`, and top5 qty match rate `>=0.75`.
- Full-run Stage 3 result on `5-13-day-control-30min`: `passed=true`, classification `passes_pricing_research_market_view`, hard failures `[]`.
- Full-run market-view metrics: BBO mismatch rate `0.00016340354734246414`, stale join rate `0.0090949283142803`, top5 join age p99 `28.13446387999999ms`, bid tick match `0.8236483072258717`, ask tick match `0.8236904160350346`, top5 tick match `0.8232061647296615`, and top5 qty match `0.8014359103924541`.
- Stage 3 still does not prove full L2 equivalence, exact queue position, queue/fill model correctness, strategy PnL, or live promotion readiness.
- Stage 4 preconditions are now satisfied for read-only pricing-model research: the accepted primary sample is `5-13-day-control-30min`, and Stage 3 confirms it is a `passes_pricing_research_market_view` candidate. This does not authorize strategy implementation, live collection, exact queue/fill work, or production promotion.

## 0514T002 Findings

- Stage 4 should start as read-only research, not strategy implementation.
- Primary sample is `5-13-day-control-30min`; optional `5-13-day-control-15min` may be used only for compressed BBO/mid sanity, not primary top5 signal conclusions.
- Candidate signal groups: BBO/bookTicker mid, current mid, weighted mid/top1 microprice, top5 microprice, top1/top5 imbalance, top5 OFI proxy, spread/volatility buckets, lead-lag/fresh-price proxies, and stale-age/freshness buckets.
- Markout horizons should be `100ms`, `500ms`, `1s`, and `5s`; metrics should include raw future mid change, side-adjusted markout, rank/correlation, monotonic quantile buckets, top-vs-bottom quantile spread, and time-split stability.
- Required row filters: exclude or explicitly bucket future joins, missing joins, gap-crossed joins, stale joins, and startup rows; primary conclusion should be based on accepted market-view rows.
- Recommended next task: `0514T003` read-only pricing research runner implementation, outputting summary markdown, candidate metrics CSV/JSON, bucket tables, markout-by-horizon CSV, rejected-signal list, and run manifest.
- A positive Stage 4 research result should only authorize a later design/implementation task for fair/reservation adjustment; it should not directly authorize live or strategy deployment.

## 0514T003 Findings

- `0514T003` implements `examples/binance_tick_mm/pricing_research.py`, a deterministic read-only runner over existing local artifacts only.
- Primary output directory is `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`.
- Full-run row counts: audit decision rows `47499`, accepted-with-stale rows `47499`, primary non-stale rows `47067`, stale rows excluded from primary `432`, future/missing/gap-crossed/startup rows `0`.
- Output artifacts: summary markdown, candidate metrics CSV/JSON, per-signal bucket tables, markout-by-horizon CSV, rejected signals CSV, and run manifest.
- Strongest primary non-stale candidate families are top5/top1 imbalance and microprice-family signals at `500ms`; top5 OFI proxy is weaker but still above the default follow-up threshold.
- Downgraded signals include duplicate reservation/audit-mid fields, weak or unstable spread/volatility/freshness/join-age fields, and weak bookTicker-mid edge under the default threshold.
- QA grouped T003 signals by importance: strongest pricing candidates are top5/top1 imbalance and microprice family; depth-size/liquidity candidates are top5 depth imbalance and top5 side quantities; existing-model/recent-move diagnostics are fair edge, recent mid move, and audit BBO mid edge; weaker follow-up signals are OFI proxies, liquidity concentration, and audit feed latency.
- QA documented duplicates: `reservation_edge_ticks` duplicates `fair_edge_ticks`, `audit_mid_edge_ticks` duplicates `audit_bbo_mid_edge_ticks`, and `book_view_stale_ms` / `latency_signal_ms` duplicate `audit_feed_latency_ms` in this sample.
- This result is research-only. It does not modify strategy behavior, configs, live scripts, core/connector APIs, or the standard npz schema, and it does not prove PnL, full L2 equivalence, exact queue/fill correctness, or live readiness.

## 0514T004 Requirements Findings

- The next research requirement is to connect T003-style pricing signals to maker execution outcomes, not just raw future-mid markout.
- Required outcome categories: fill probability, time-to-fill, adverse selection after fill, spread capture, queue/priority proxies, cancel-to-fill race, post-only/reject/throttle/churn, and inventory impact.
- Required label categories for a later implementation plan: submit-to-fill within `100ms / 500ms / 1s / 5s`, time-to-fill, fill-after side-adjusted markout, spread capture vs future mid, fill-after-cancel-request, reject/throttle/drop/churn bucket, and inventory transition.
- Additional required label categories: quote placement / distance, missed-fill / opportunity cost, realized PnL decomposition, tail risk, partial-fill / order lifecycle, inventory cycle, and sample validity / censoring.
- Later analysis must use statistics appropriate to each label type: Spearman/Pearson plus bucket monotonicity for continuous labels; event-rate/lift/odds-ratio buckets for binary labels; Kaplan-Meier/discrete-hazard or Cox-style treatment for censored time-to-event labels; exposure-normalized rate ratios for count labels; contingency/conditional-probability/mutual-information summaries for lifecycle labels; and tail quantile/CVaR-like summaries for tail labels.
- Any later implementation must stay observed-only unless a separate queue/fill calibration task is authorized. It must not claim counterfactual queue/fill proof or strategy/live readiness.

## 0514T005 Planned Findings

- `0514T005` should implement the T004 contract in a separate task, not inside T004.
- The implementation should be a deterministic read-only runner over existing `5-13-day-control-30min` artifacts.
- Required output directory: `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`.
- Acceptance should focus on label construction coverage and dataset validation: every T004 label class must be either implemented with rows/statistics or explicitly marked `unavailable`, `low_sample`, or `observed_only_proxy` with a reason.
- The task must not modify strategy behavior, live scripts, core/connector, canonical audit schema, standard npz schema, or historical Stage 4/T009 artifacts.

## 0514T005 Findings

- `0514T005` implemented a deterministic read-only execution-outcome label runner plus focused tests, with outputs under `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`. It did not modify strategy behavior, live scripts, core/connector, canonical audit schema, or standard npz schema.
- Coverage result on `5-13-day-control-30min`: `fill_probability`, `time_to_fill`, `adverse_selection_after_fill`, `spread_capture`, `cancel_to_fill_race`, `inventory_impact`, `quote_placement_distance`, `partial_fill_lifecycle`, `inventory_cycle`, and `sample_validity_censoring` are `available`; `queue_priority_proxy`, `post_only_reject_throttle_churn`, `missed_fill_opportunity_cost`, and `realized_pnl_decomposition` are `observed_only_proxy`; `tail_risk` is `low_sample`; no T004 label class is `unavailable`.
- Current sample shape is dominated by high cancel / low fill behavior: submit orders `2516`, filled orders `53`, canceled orders `2452`, expired `9`, open-or-missing `2`, fill-after-cancel orders `16`, fast-cancel-churn `1955`, and partial-fill orders `0`.
- Fill mass is not concentrated only in ultra-short horizons: fill-by-`100/500/1000/5000ms` is `8/22/28/40`. For this sample, optimizing only around `100ms` behavior would miss a large fraction of observed fills.
- The observed lifecycle risk is not mainly tail-only: cancel-to-fill race is material in the realized sample, while tail-risk remains low-sample because only `53` filled orders are available.
- Placement and inventory state are first-order calibration strata for later work: Stage 5 shows meaningful differences across `placement_bucket`, `distance_to_bbo_ticks`, `edge_vs_fair_ticks`, and `inventory_score`, so later replay/live calibration should compare these strata explicitly instead of only reporting aggregate gaps.
- T005 remains observed-only. Queue/priority, missed-opportunity, and realized-PnL decomposition labels are useful for ordering later work, but they do not prove exact queue position, counterfactual fill outcomes, or strategy PnL.
- This result supports refining Stage 6 into read-only replay/live fill-cancel lifecycle proxy calibration. `0514T006` should define the contract first; `0514T007` should implement the calibration runner afterward.

## 0514T006 Findings

- Stage 6 should be framed as replay/live fill-cancel lifecycle proxy calibration, not broad exact-queue calibration. The current evidence base is rich enough for lifecycle comparison but still observed-only for queue priority, missed opportunity, and realized-PnL decomposition.
- The correct Stage 6 comparison unit is not raw cross-domain `order_id`. Live/replay alignment should be built around matched normalized submit opportunities, with submit-key coverage reported explicitly before interpreting lifecycle gaps.
- The required common Stage 6 label schema should reuse the Stage 5 core execution outcomes: fill-by-horizon, time-to-fill, final order state, fill-after-cancel-request, cancel-to-fill delay, fast-cancel-churn, fill markout / spread-retention, and coverage/censoring flags.
- Stage 6 results must be reported both in aggregate and across key strata: `placement_bucket`, `distance_to_bbo_ticks`, `edge_vs_fair_ticks`, `inventory_score`, top-of-book/top5 size-age proxy, and latency regime. Aggregate-only reporting would hide the main replay/live risk concentrations.
- `5-13-day-control-30min` is enough for Stage 6B runner implementation and single-sample methodology validation because it already has Stage 3 acceptance, T009 sidecar alignment, and T005 labels.
- `5-13-day-control-30min` is not enough alone to authorize quote-adjustment promotion: only `53` fills are observed, `partial_fill=0`, and tail-risk remains low-sample. Later promotion-style decisions need additional current-format samples with the same artifact chain.
- `0514T007` should stay strictly read-only. It may classify the result as `methodology_valid_single_sample`, `diagnostic_only_gap_too_large`, or `requires_more_current_format_samples`, but it must not claim exact queue proof, counterfactual fill proof, or live readiness.

## 0514T007 Findings

- `0514T007` implements a read-only replay/live execution outcome calibration runner and uses matched normalized submit opportunities as the primary comparison unit.
- On `5-13-day-control-30min`, submit-key coverage aligns perfectly: live submit orders `2516`, replay submit orders `2516`, matched submit orders `2516`, and matched price tick / qty equality are both `2516/2516`.
- Fill-horizon rates are relatively close on the matched submit universe: absolute gaps are about `0.0012` at `100ms`, `0.0012` at `500ms`, `0.0020` at `1000ms`, and `0.0119` at `5000ms`.
- Fast-cancel-churn is aligned (`0.7770` vs `0.7770`), so the large replay/live difference is not a generic quote-churn mismatch.
- The main replay/live gaps are in lifecycle outcomes rather than submit coverage:
  - replay filled orders `172` vs live `53`
  - replay fill-after-cancel orders `133` vs live `16`
  - final state gaps: `canceled` about `0.0906`, `filled` about `0.0473`, `open_or_missing` about `0.0433`
  - fill-after-cancel-request rate gap about `0.0465`
  - cancel-to-fill delay gap is also large
- Markout observability coverage is materially different between replay and live, even when submit matching is perfect. This means Stage 6 should keep coverage-gap reporting separate from lifecycle-gap reporting.
- Strata output confirms that important gaps concentrate in placement / inventory / latency buckets, especially deeper step-back placements, higher inventory-score buckets, larger same-side size buckets, and some higher join-age / latency buckets.
- Current decision state should remain `diagnostic_only_gap_too_large`: the Stage 6B methodology works on a single accepted sample, but replay lifecycle still deviates too much from live to treat replay fill-side behavior as close enough for promotion-style quote-adjustment experiments.

## 0510T001 Findings

- `5-10-day-control-1h-06` is present under `local_live_analysis/` and has an archive tarball plus sha256.
- `maker_acceptance.py` passed on audit replay with `action/planned/reject/throttle = 1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, and strict replay lag breach/drop/fail `0/0/0`.
- Cancel-requested fills are non-trivial: `21 / 46` fills, notional rate about `0.4564`.
- Risk source is more consistent with adverse-selection / inventory-reducing cancel race than same-side readd: adverse-selection candidate count `14`, add-side candidate count `7`, same-side readd then cancel-fill count `1`.
- Single-sample Stage 6J replay completed with 6 candidates and no hard failures, but decision is `diagnostic_only_no_promotion`.
- Next useful task should add cross-sample validation before any live micro test.

## 0510T002 Findings

- Added `.workflow/runners/run_task.py` and `.workflow/runners/run_0510T002.py`.
- Runner automatically generated missing `maker_acceptance.json` for `5-8-stage3-15m-livetest-v4`.
- Cross-sample Stage 6J replay ran on 4 samples:
  - `5-10-day-control-1h-06`
  - `5-9-small`
  - `5-9-noon`
  - `5-8-stage3-15m-livetest-v4`
- Result: `diagnostic_only_no_promotion`.
- Hard failures: `0`.
- Baseline across 4 runs: pnl sum `-0.8641`, cancel-fill count `3`, same-side worsening `3`.
- Add-side guard candidates: pnl sum `-0.2329`, cancel-fill count `0`, same-side worsening `0`.
- Broad add-side cooldown 200ms control: pnl sum `-0.0535`, cancel-fill count `1`, same-side worsening `1`.
- Do not auto-promote to live. QA and total controller should review whether this warrants a separate adverse-selection timing rule design task.

## 0511T001 / 0511T002 Split Findings

- Current live/backtest structure already shares `strategy_core.py` for action decisions, cancel-race guard helpers, lifecycle tracking, and audit row construction.
- Signal-to-target generation and guard-context assembly are still duplicated in live/backtest loops; this is acceptable for a narrow guard task but should not expand further without a follow-up refactor.
- The existing `cancel_race_guard` blocks add-side exposure while allowing reduce-side actions; adverse-selection timing should preserve this same safety shape.
- Future markout is an evaluation metric only. It must not become a live rule input.
- First adverse-selection timing candidate should target decision-time-visible toxic timing, such as cancel-requested fill timing or target deterioration, and should be validated through Stage 6J source-path metrics rather than PnL alone.
- `0511T001` is design-only and should not modify strategy code.
- `0511T002` is the first task allowed to modify strategy code for this rule, after `0511T001` is accepted.
- `0511T001` completed the design contract and recommends entering `0511T002` only for default-off implementation, unit tests, and offline replay. It does not authorize live micro test or default rule enablement.
- First implementation should use decision-time-visible inputs only: working/target ticks, position, in-flight/cancel-requested state, last cancel-fill timestamps, latency signals, and rolling historical state. Future mid/markout and audit overlays remain forbidden as live decision inputs.
- `0511T002` implemented a default-off adverse timing guard with shared live/backtest helper and audit fields, but cross-sample replay still decided `diagnostic_only_no_promotion`.
- In `0511T002`, pure `adverse_timing_target_deterioration_50/100/200ms` matched baseline across 4 samples: cancel-fill `3`, same-side worsening `3`, pnl sum `-0.8641`.
- In `0511T002`, `add_side_guard_only` and `add_side_guard_plus_adverse_timing_100ms` both reduced cross-sample cancel-fill to `0` and same-side worsening to `0`, with pnl sum `-0.2329`; adverse timing added no incremental replay benefit over add-side guard.
- Do not live-promote the adverse timing candidate from `0511T002`; future work should redesign the toxicity trigger or improve diagnostic replay sensitivity before any live micro test.
- Before redesigning the timing rule, the next diagnostic must distinguish whether `target_deterioration` did not trigger, triggered away from add-side decisions, or exposed a Stage 6J replay/source-path sensitivity gap.

## 0511T003 Dashboard Findings

- The current dashboard is useful as a task status index but weak as a decision board.
- For this project, the dashboard should expose task results such as Stage 6J decision, sample count, candidate count, hard failures, cancel-fill risk, QA conclusion, and live/no-live decision.
- The first dashboard upgrade should remain static and markdown-driven; interactive editing and task execution should stay out of scope.
- `0511T003` upgraded the dashboard into a static decision board with task status, decision summary, key metric chips, business results, QA conclusions, and next steps.
- Metrics are currently extracted from free-text business/QA reports with heuristics. Future reports should add explicit `metrics` or `decision_summary` fields to reduce ambiguity.

## 0511T004 Findings

- `target_deterioration` was not missing. In `0511T002` replay audits, each pure adverse timing candidate produced `buy_active=118429` and `sell_active=160062` across 4 samples.
- Pure adverse timing active rows did not overlap actual add-side submissions: `submit_buy` overlap `0`, `submit_sell` overlap `0`.
- Baseline vs `adverse_timing_target_deterioration_50/100/200ms` had zero row-level differences across action, planned_action, position, target ticks, and working ticks for all 4 samples.
- The first `target_deterioration` trigger is state-misaligned for the intended risk: it detects stale existing working quotes, while many target add-side submit/re-add moments happen when that side has no working order.
- Stage 6J replay currently under-observes the live adverse-selection source-path: live/current-format risk diagnostics show positive adverse-selection candidate counts, but Stage 6J replay summary reports `guard_candidate_adverse_selection_count=0` for baseline and adverse timing candidates.
- Do not implement or live-test another adverse timing rule until the next contract defines add-side submit/re-add coverage and replay/source-path observability.

## 0512 Planned Findings

- The next step is not to tune `target_deterioration`; it is to reconcile replay/source-path observability first.
- `0512T001` must determine whether the replay adverse-selection gap comes from missing lifecycle fields, fill model differences, field semantics, or source-path classification not fitting replay audits.
- `0512T004` must turn the `0512T001` diagnosis into an explicit observability gate before `0512T002` starts.
- `0512T002` must design a new add-side submit/re-add toxic timing rule that proves coverage of actual `submit_buy` / `submit_sell` rows, while treating Stage 6J replay as regression evidence rather than live source-path proof.
- The 4H `5-11-night-active` sample should be used for main diagnosis because it has much more event mass, but success on that sample alone must not authorize live promotion.

## 0512T001 Findings

- The Stage 6J adverse-selection source-path gap is mainly a replay lifecycle / fill-model gap, not a different source-path classifier. Stage 6J calls `run_backtest(...)` to generate a new simulated audit and then calls the same `analyze_audit_csv(...)` classifier.
- Live/current-format risk diagnostics show positive inventory-reducing cancel race counts, but Stage 6J baseline replay generates only `0/1/1/1` cancel-after-request fills across the four samples and zero adverse-selection candidate count.
- Sampled live adverse-selection events do not have matching order ids or same side/price fills in Stage 6J baseline replay audit.
- Stage 6J can compare candidates under its replay fill model, but cannot by itself prove live adverse-selection source-path improvement.
- `0512T002` may proceed as a design-only task, but its contract must use action-path coverage and explicit observability limits; no implementation or live test is authorized by `0512T001`.

## 0512T004 Findings

- `0512T004` defines the required evidence split for later rule work: action-path coverage, replay-model regression, and live-derived source-path proof.
- Action-path coverage must prove that a new guard actually reaches add-side submit/re-add rows: eligible add-side submit count, blocked add-side submit count, blocked reason, row-level action/planned_action diff, reduce-side allowed rows, and blocked reduce-side count.
- Stage 6J replay is a replay-model regression gate only: PnL, position, drop/churn, action churn, replay fill/cancel-fill, and replay source-path metrics can compare candidates within the same simulated lifecycle, but cannot alone prove live inventory-reducing cancel race improvement.
- Live-derived source-path proof must come from current-format live audit analysis: fill-after-cancel-request events, inventory-reducing cancel race, adverse-selection candidate count, add-side candidate count, cancel-to-fill latency, and side-adjusted markout.
- `5-11-night-active` is the main development/diagnostic sample; `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small` are cross-sample sanity checks. A single-sample win on `5-11-night-active` cannot authorize live promotion.
- `0512T002` may start after `0512T004` QA only as a design-contract task. It must not claim live source-path improvement, authorize implementation, or authorize live micro test.

## 0512T002 Findings

- `0512T002` defines a new default-off `add_side_toxic_timing_guard` design contract for add-side submit/re-add path coverage.
- The old `target_deterioration` trigger is no longer the main trigger. It may only be an auxiliary signal when evaluated from submit/re-add eligibility; threshold tuning of the old working-quote-only trigger is explicitly rejected.
- The new rule must first compute add-side submit eligibility, then suppress only the submit leg when a same-side toxic timing signal is active. Reduce-side submit must remain allowed and must be proven by audit with blocked reduce-side count `0`.
- Required audit evidence includes eligible add-side submit count, blocked add-side submit count, blocked reason, target move since last quote/cancel, last cancel request/fill ages, guard-until timestamps, and reduce-side allowed fields.
- Stage 6J replay remains replay-model regression only. It must report action-path coverage and regression metrics across `5-11-night-active` plus sanity checks `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small`.
- Live-derived source-path proof remains outside `0512T002`: historical live counterfactual overlay can show would-block overlap, but true source-path improvement requires a later QA-approved post-rule live micro test and current-format live audit risk analysis.
- `0512T002` does not authorize strategy implementation, default enablement, or live micro test. After QA, it may authorize creation of a separate implementation + offline replay task.

## 0512T005 Planned Findings

- `0512T005` is the implementation/offline replay task created from `0512T002` core conclusions.
- It must implement default-off `add_side_toxic_timing_guard`, shared live/backtest helper, config fields, audit fields, unit tests, Stage 6J replay candidates, and action-path coverage reporting.
- It must prove add-side submit/re-add coverage with eligible rows, blocked rows, submit overlap, action/planned_action diff, and blocked reduce-side rows equal to `0`.
- It must use `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06`, `5-9-noon`, `5-9-small` as cross-sample sanity checks.
- It does not authorize default enablement, live micro test, or claims of live-derived source-path improvement.

## 0512T005 Findings

- Commit `34c954e` implements default-off `add_side_toxic_timing_guard` with shared strategy-core helper, live/backtest config/state integration, audit fields, Stage 6J candidate matrix, and focused tests.
- The rule blocks only add-side submit/re-add legs. Stage 6J cross-sample coverage total: pure toxic timing candidates blocked `594` add-side submits, blocked reduce-side total `0`, baseline action/planned diff `199`, blocked submit overlap `56`, submit removed `56`.
- Per-sample `add_side_toxic_timing_100ms` coverage: `5-11-night-active` blocked `299` rows but no baseline submit removal under replay; `5-10-day-control-1h-06` blocked `6`; `5-9-noon` blocked `46` with `2` submit removals; `5-9-small` blocked `243` with `54` submit removals.
- Stage 6J replay completed 4 samples x 7 candidates with hard failures `0`, overlays `off/off/off`, decision `diagnostic_only_no_promotion`.
- Replay-model result: pure toxic timing 50/100/200ms has action-path coverage but does not improve replay cancel-fill or same-side worsening versus baseline (`cancel-fill 4`, `same-side worsening 3` for both). `add_side_guard_only` and combined guard reduce replay same-side worsening to `0`, but combined toxic timing adds no replay benefit over add-side guard.
- Live-derived source-path proof remains missing. Stage 6J remains a replay-model regression gate only and cannot prove live inventory-reducing cancel race improvement.
- No default enablement and no live micro test are allowed from `0512T005`.

## Signal Design Lessons

- Design trading risk signals around the action path they are meant to change, not only around a plausible market-state condition. The 0511 `target_deterioration` signal was plausible and fired often, but it observed existing working quotes and had zero `submit_buy` / `submit_sell` overlap; it could not change the intended add-side submit/re-add behavior.
- For submit/re-add risk, compute action eligibility first, then evaluate toxicity and suppress only the intended submit leg. T005 gained coverage because `buy_submit_eligible` / `sell_submit_eligible` became explicit inputs to the guard, so the signal was evaluated exactly where add-side submit decisions were made.
- Separate signal domain from signal strength. In T005, recent same-side cancel request / cancel-fill / cancel-requested inflight exposure defines the timing domain, while target move / latency / recent fill defines toxicity. A strong toxicity proxy is still ineffective if it is attached to the wrong decision domain.
- Preserve the intended safety shape in the action layer: allow stale quote cancel and reduce-side submit, but suppress toxic add-side re-add submit. This made blocked reduce-side rows stay at `0` while still producing blocked add-side submit rows.
- Future signal tasks should require action-path coverage evidence before discussing replay risk improvement: eligible rows, blocked rows, blocked reason, baseline action/planned diff, blocked submit overlap, submit removed, and blocked reduce-side rows. Only after those pass should replay-model regression and live-derived source-path proof be interpreted.

## 0512T006 Planned Findings

- T006 is a planning-only follow-up to explain why T005 pure toxic timing candidates had action-path coverage but no Stage 6J replay risk improvement.
- T006 must plan three analyses only: blocked-row stratification, actual submit-removal row analysis, and blocked reason attribution.
- T006 must not design stricter candidates. Candidate redesign belongs to a later task only after the attribution root cause is known.
- T006 must not modify code, run new Stage 6J replay, start live, or claim live-derived source-path proof.
- T007 is reserved for the actual read-only attribution / experiment implementation after T006 defines the inputs, outputs, and decision criteria.

## 0512T006 Findings

- T006 produced a planning-only attribution contract for T007 and did not modify strategy code, run new Stage 6J replay, create stricter candidates, or start live.
- The planned T007 blocked-row stratification must separate true submit removal from eligibility-only coverage, especially `5-11-night-active` where pure toxic timing blocked `299` rows but removed `0` baseline submits.
- The planned T007 submit-removal analysis must focus on the `56` true submit removals and link them to later replay cancel-fill / same-side worsening / source-path events before interpreting replay risk improvement.
- The planned T007 reason attribution must split pending cancel, recent cancel request, recent cancel fill, target move, and latency contributions, and explain why 50/100/200ms pure toxic timing candidates were identical.
- T006 passed QA and now authorizes creation of T007 as a read-only attribution implementation only. It does not authorize rule redesign or live micro test.

## 0512T007 Findings

- T007 explains why T005 pure toxic timing had action-path coverage but no replay risk improvement: most blocked rows did not remove a baseline submit, and the submits that were removed did not overlap the replay cancel-fill risk orders.
- For representative `add_side_toxic_timing_100ms`, blocked rows were `594`, but only `56` row-level submits were actually removed (`9.43%`), representing `42` unique submit-order keys after de-duplication.
- `5-11-night-active` is the clearest case: blocked rows `299`, true submit removed `0`, so coverage was eligibility-only on the main sample.
- True submit removals had `0 / 56` overlap with baseline Stage 6J cancel-fill risk event order ids. Baseline and candidate 100ms risk order ids stayed the same: `5-11-night-active` `2|1`, `5-9-noon` `3`, `5-9-small` `13`.
- 50/100/200ms pure toxic candidates were equivalent because blocked rows were dominated by `pending_cancel+target_move`; the window parameter did not change the blocked set in these replay samples.
- T007 remains replay-model attribution only. It does not provide live-derived source-path proof and does not authorize live micro test.

### 0512T007 Excluded Directions

- Excluded: `pure toxic timing has no action-path coverage`. It does have coverage: representative 100ms blocked rows `594`.
- Excluded: `reduce-side was harmed or the result came from reduce-side suppression`. Blocked reduce-side count remained `0`.
- Excluded: `50/100/200ms window tuning is likely to change this result`. The three windows had `100%` unique blocked-key equivalence across all four replay samples.
- Cause of window equivalence: blocked reason was dominated by `pending_cancel+target_move`; in these replay samples, the window-ms parameter did not become the active differentiator.
- Excluded: `the main sample lacked trigger mass`. `5-11-night-active` had `299` blocked rows, but `0` true submit removals.
- Excluded: `pure toxic timing filtered the Stage 6J replay adverse-selection / cancel-fill risk orders`. True submit removals had `0 / 56` overlap with baseline cancel-fill risk event order ids, and baseline/candidate 100ms risk order ids were unchanged.
- Excluded: `more blocked rows alone should improve replay risk`. The problem was not blocked-row count; `538 / 594` 100ms blocked rows did not actually remove a baseline submit.
- Excluded: `T007 can be used as live source-path improvement proof`. T007 is replay-model attribution only and does not supply live-derived source-path proof.
- Excluded design direction: continuing the same `pending_cancel+target_move` pure toxic timing rule with only window-ms tuning. Future work must redefine what makes a submit a risk-source submit before proposing a new rule.

## 0512T008 Findings

- `strategy_core.decide_actions()` does not receive full depth directly. The live/backtest outer loops read `hbt.depth(0)`, compress it into best bid/ask, mid, bid/ask size, top5 strings, fair/reservation/half_spread, and target ticks, then pass only target/action state into the shared action core.
- Shared `hbt.depth(0)` API does not imply identical live/replay view. Live reads connector-maintained local book at decision time; Stage 6J no-overlay reads replay-reconstructed depth.
- `market_state_overlay=audit` forces live audit compressed market/fair/target fields into audit replay, but it does not overlay top5 tick/qty strings. Top5 strings remain replay-depth derived and can still mismatch.
- On `5-11-night-active`, audit replay overlay had compressed market mismatch `0` and target tick mismatch `0`, but top5 mismatch `11.0159%`.
- On `5-11-night-active`, Stage 6J no-overlay with lag<=250ms had best bid mismatch `9.5306%`, best ask mismatch `9.5246%`, target bid tick mismatch `10.2907%`, and target ask tick mismatch `10.0460%` versus live audit at matched decision timestamps.
- Sanity samples also show Stage 6J no-overlay live/replay view differences: best bid/ask mismatch about `2.5%-4.2%`, and target tick mismatch about `3.1%-5.1%`.
- Raw gzip contains Binance depth sequence data and bookTicker events. Across four samples, depthUpdate `pu` mismatch count was `0`; `5-9-noon` raw gzip has an EOF trailer issue that was recorded as data-quality metadata.
- Converted npz schema is `ev|exch_ts|local_ts|px|qty|order_id|ival|fval` and does not retain Binance `U/u/pu` or `lastUpdateId`.
- Current audit/action alignment is sufficient for the existing simple compressed strategy view, but it is not sufficient to prove full L2 / queue / OFI / microprice equivalence.
- Before microprice / OFI / queue research or strategy changes, extend data-layer/audit schema with top-N book, update ids, exchange/local timestamps, bookTicker-vs-depth consistency, and per-decision book provenance.
- After the date change, this follow-up is `0513T001` for planning only and a later `0513T002` for implementation if QA accepts the plan.

## 0513T001 Findings

- Do not modify `hbt.depth(0)` core API first. The immediate problem is transparency at the Binance maker strategy layer, not proof that the core depth API itself is wrong.
- The next implementation should introduce an explicit strategy-layer `MarketView` / `BookViewSnapshot` wrapper and a shared `build_market_view_from_depth(...)` helper used by both live and backtest loops.
- `strategy_core.decide_actions()` should continue receiving compressed action inputs rather than full depth. The transparency layer should live before fair/target/action construction.
- Audit rows should carry provenance fields such as `market_view_source`, `top5_source`, `market_overlay_source`, `top5_overlay_source`, `book_view_ts_local`, `book_view_ts_exch`, `book_view_feed_latency_ns`, and `book_view_stale_ms`.
- top5 alignment is necessary for current decision-view transparency because current fair uses top5-derived `bid_size - ask_size`, but top5 alignment is not full L2 / queue / OFI proof.
- Overlay top5, if implemented, must be labeled as `audit_overlay` and treated only as audit replay decision-view alignment. It must not be used to claim replay reconstructed book alignment.
- If the Python strategy layer cannot expose Binance update ids, `lastUpdateId`, bookTicker provenance, or decision-row book provenance, create a later core/data task instead of fabricating fields.
- `0513T001` does not implement code. If QA passes, create a separate `0513T002` implementation task for MarketView provenance / top5 audit transparency minimal implementation.
- `0513T002` must keep a hard file boundary: only `strategy_core.py`, `live_tick_mm.py`, `backtest_tick_mm.py`, and `test_backtest_tick_mm.py` are allowed implementation files. Core API, converter/npz schema, connector book management, strategy rules, fair price formula, risk guards, configs, live scripts, and microprice/OFI/queue studies require separate tasks.

## 0513T002 Findings

- `0513T002` implemented a strategy-layer `MarketView` wrapper and `build_market_view_from_depth(...)` helper inside `strategy_core.py`.
- live and backtest loops now build the decision market view through the shared helper, then assign the same best bid/ask, mid, spread, top5, and top5-size values as before.
- Audit rows now include provenance fields: `market_view_source`, `top5_source`, `market_overlay_source`, `top5_overlay_source`, `book_view_ts_local`, `book_view_ts_exch`, `book_view_feed_latency_ns`, `book_view_stale_ms`, `top5_depth_best_bid_tick`, and `top5_depth_best_ask_tick`.
- In audit replay overlay mode, compressed market state is labeled `market_view_source=audit_overlay` and `market_overlay_source=audit`, while top5 remains labeled from replay depth. This preserves T001 option B and avoids pretending replay reconstructed depth has been fixed.
- `0513T002` did not implement Binance `U/u/pu`, `lastUpdateId`, or bookTicker provenance because these are not reliably exposed at the current Python strategy layer. Those require a separate core/data task if needed.
- Focused tests passed for helper output, audit schema fields, build-audit-row provenance, and market-state overlay related tests.
- No live run, no new Stage 6J replay/sweep, no core API change, no converter/npz change, no config change, and no strategy-rule change were performed.

## 0513T004 Findings

- `0513T004` implements a live startup preflight gate in `examples/binance_tick_mm/deploy/preflight_live_run.py` and wires it into `deploy/run_live.sh` before tmux/collector/connector/live bot startup.
- The preflight manifest records git commit, branch, dirty status, config hash, connector config hash, `audit_schema.py` hash, `strategy_core.py` hash, `live_tick_mm.py` hash, `run_live.sh` hash, preflight script hash, symbol, data dir, run dir, configured output paths, Python environment, and start/stop marker paths.
- The compatibility check imports the current `strategy_core`, builds representative decision and lifecycle audit rows, and requires their keys to exactly match the final `AUDIT_FIELDS`. Duplicate fields, missing fields, or extra row keys fail preflight before live starts.
- This directly addresses the `0513T003` failure mode where remote `audit_schema.py` lagged strategy code and live failed later at CSV writer time.
- `run_live.sh` now writes `deployment_manifest.json` and `start_marker.json` before startup, and configures the live bot pane to write `stop_marker.json` on process exit.
- Focused tests passed for current schema compatibility, missing/extra field detection, duplicate field detection, manifest/start marker output, and stop marker output.
- T004 is a deployment reproducibility gate only. It does not start live, modify AWS, change strategy behavior, prove PnL, or prove full L2 / market-view alignment.

## 0513T005 Planned Findings

- Step 2 must first classify sample usability before any pricing, queue, OFI, or microprice research.
- Existing `align_live_run.py` / `compare_audit.py` artifacts already expose useful latency, replay lag, action-path, and top5 mismatch diagnostics, but they do not preserve full Binance raw provenance in the converted npz.
- Current converted npz schema does not retain `U/u/pu`, `lastUpdateId`, or bookTicker provenance. Raw update-id continuity and bookTicker/depth consistency must be computed from raw gzip or moved into a later core/data task.
- Existing samples are useful for Step 2 diagnostics, but most are pre-T004 legacy samples. They cannot prove the new standard run-live deployment flow unless a later fresh T004-standard no-rule sample is collected.
- The Step 2 implementation should output sample-level classification:
  - `compressed_action_path_only`
  - `pricing_research_candidate`
  - `queue_fill_research_candidate`
  - `unusable`
- The likely near-term outcome is that existing samples support compressed action-path diagnostics and some pricing-input sanity checks, but queue/fill research requires stronger raw/top-N/update-id provenance.

## 0513T006 Findings

- T006 generated all required Step 2 read-only artifacts under `local_live_analysis/step2_market_data_baseline_0513T006/`.
- Classification result:
  - `5-13-day-control-15min`: `pricing_research_candidate`, but only for limited live-audit compressed BBO/mid sanity checks.
  - `5-11-night-active`: `compressed_action_path_only`.
  - `5-10-day-control-1h-06`: `compressed_action_path_only`.
  - `5-9-noon`: `compressed_action_path_only`.
  - `5-9-small`: `compressed_action_path_only`.
- No current sample qualifies as `queue_fill_research_candidate`.
- All five samples are legacy/pre-T004 from a deployment reproducibility perspective because they lack `deployment_manifest.json`.
- Only `5-13-day-control-15min` has T002 MarketView provenance fields in live audit; older samples lack strategy-layer source fields.
- Existing action-path acceptance remains useful for the current compressed strategy view, but it is not full L2 / queue / OFI / microprice proof.
- Top5 mismatch remains material across all samples: tick match about `0.9044` to `0.9747`, qty match about `0.8898` to `0.9558`.
- All samples show degraded order-entry tail latency in current artifacts: entry p99 about `4055ms` to `15112ms`.
- Raw gzip depth `pu` continuity was diagnosable in this bounded pass and showed `0` mismatches for the five samples.
- bookTicker/depth consistency in T006 is bounded best-effort only, with `20000` bookTicker checks per sample; it is not production-grade local book proof.
- Converted npz still does not retain Binance `U/u/pu`, snapshot `lastUpdateId`, bookTicker provenance, or per-decision top5 book provenance. The current near-term research boundary is top5-only; full L2 provenance and exact queue-position work are not required for T007.
- T006 did not start live, collect data, run replay/sweep, change strategy/deploy/core/converter code, or authorize live promotion.

## 0513T007 Planned Findings

- The Binance provenance fix should not change the standard hftbacktest `data` npz main event schema. That array should remain the replay-compatible market event stream.
- Raw Binance message provenance belongs in sidecars because `U/u/pu`, snapshot `lastUpdateId`, and bookTicker `u` are message/local-book metadata, not natural price-level event fields.
- Avoid stuffing Binance provenance into reserved `ival/fval` fields. That would be opaque, duplicate message-level data across split price-level rows, and risk corrupting event semantics.
- The sidecar design must make synchronization explicit, not implicit:
  - `raw_seq -> npz row range`
  - `raw_seq -> reconstructed book/top5 row`
  - `reconstructed book/top5 row -> decision row`
- Raw-to-npz mapping must reference final standard `data` array rows after `correct_local_timestamp()` and `correct_event_order()`, because converter ordering can split one input event into exchange/local rows.
- Decision joins must be as-of joins using historical top5 rows only. They must report join key, joined `raw_seq`, joined depth `u`, joined bookTicker `u`, split join age ms, stale status, gap-crossed status, and future join count `0`.
- bookTicker and depth are separate Binance streams; mismatch can come from stream timing and must be measured with age/mismatch buckets before being interpreted.
- Top5 sidecars should include schema/version/manifest metadata: schema version, raw file identity, converter opt, top5 levels, tick_size source, generation time, and input sample id.
- T007 should not modify canonical `audit_schema.py` or live audit CSV schema. Joined decision output should be an independent diagnostic artifact until a later task promotes fields into the formal audit schema.
- T007 is top5-only. It can support later top5 pricing / top5 OFI proxy / top5 microprice proxy research if join-age and mismatch metrics pass, but it cannot prove full L2 equivalence or exact queue position.
- Queue work under this boundary means top-of-book/top5 size and age proxies only.
- Long term, if the strategy must consume update ids/top5 provenance in real time, that should be a separate connector/core API task. T007 is a data-quality and read-only audit/provenance task.

## 0513T007 Findings

- T007 implemented `examples/binance_tick_mm/binance_top5_provenance.py` as a standalone top5 provenance / decision join tool.
- The tool keeps the standard hftbacktest `data` npz main event schema unchanged and writes Binance-specific provenance to sidecars instead.
- Sidecar outputs:
  - `raw_provenance.csv`
  - `raw_to_npz_mapping.csv`
  - `top5_sidecar.csv`
  - `joined_decisions.csv`
  - `sidecar_manifest.json`
  - `metrics.json`
- `raw_to_npz_mapping.csv` maps `raw_seq` to final standard `data` row indices after local timestamp correction and event-order correction.
- Decision joins are as-of joins only; T007 reports `future_join_count=0`.
- Smoke over the first `5000` messages of `5-13-day-control-15min` produced final data row mapping coverage `1.0`, depth `pu` mismatch `0`, and bookTicker/depth BBO match/mismatch `151/1`.
- The same smoke intentionally exposed poor sample-slice usability: `first_valid_update_aligned=false`, stale joins `243/244`, and gap-crossed joins `244/244`. This is expected for a bounded slice and shows the tool reports unusable sync/join states instead of fabricating clean top5 proof.
- T007 does not modify `align_live_run.py`, canonical `audit_schema.py`, live audit CSV schema, strategy behavior, core event schema, py `event_dtype`, connector local book management, or live deployment scripts.
- T007 does not start live, run strategy replay/sweep, prove full L2 equivalence, prove exact queue position, prove strategy PnL, or authorize live promotion.

## 0512T003 Findings

- `5-11-night-active` is a current-format 4H night-active sample with live rows `1,018,503`, decision rows `802,999`, and local archive `local_live_analysis/archive/5-11-night-active.tar.gz`.
- Analysis used the local repository package via conda env `hftbacktest` and `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm`, not the generic site-packages path.
- Maker acceptance passed on audit replay: action/planned/reject/throttle all `1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, strict lag gate breach/drop/fail `0/0/0`, and post-startup outside dual gate rows `0`.
- Audit replay consumed/scheduled `802,996 / 802,999`; the remaining `3` unconsumed rows are tail rows and did not fail the strict lag gate.
- Audit replay PnL for this sample was `-7.6108`, max abs position notional `246.29655`, below the current `250` cap.
- Cancel-requested fill risk repeated strongly in the longer night-active sample: `391 / 915` fills after cancel request, notional rate `0.427300`, same-side readd while cancel-requested `1436`, same-side readd then cancel-fill `13`, worsening cancel-fill `191`.
- Source-path split on `5-11-night-active`: add-side candidates `201`, adverse-selection / inventory-reducing candidates `190`.
- Cross-sample current-format risk summary over `5-9-noon`, `5-10-day-control-1h-06`, and `5-11-night-active` decided `proceed_to_stage6j_narrow_rule`.
- This sample supports narrow cancel-requested fill-risk rule design, but does not authorize direct maker parameter optimization or live micro test. The Stage 6J source-path observability limitation from `0512T001` still applies.
