# Task Plan

## Purpose

This file is the controller-level plan for the Binance maker market-making work.

North star:

- Align live and replay enough that replay decisions, fills, latency, and market views are trustworthy.
- Use that aligned framework to build a positive-PnL high-frequency maker strategy.
- Treat `docs/hft-share.md` as the current strategic reference: first make infra / latency / replay deterministic, then build a stronger current fair-pricing model, then calibrate fill / queue / inventory execution around that pricing model.
- Treat maker strategy as a system engineering problem. The current route is not to make one dimension extreme, such as mandatory full L2 or exact queue position, but to bring each layer above a usable and verifiable baseline: data view, fair price, pricing signals, strategy logic, risk guards, execution mechanics, and replay/live alignment.

Historical task details are kept in `.workflow/tasks/` and `.workflow/reports/`. This file should stay focused on the current operating state and the forward plan.

## Workflow Rules

Reference docs:

- `.workflow/workflow-kit/workflow-manual.md`
- `.workflow/workflow-kit/task-dispatch-template.md`
- `.workflow/workflow-kit/thread-report-template.md`
- `.workflow/workflow-kit/qa-acceptance-template.md`
- `docs/thread-playbook.md`

Roles:

- 总控: scope, sequence, task files, final direction.
- 业务线程: implementation or design inside assigned boundaries.
- 测试线程: focused evidence collection and regression runs.
- QA验收线程: final acceptance result source of truth.

Operating constraints:

- Every formal task needs a `.workflow/tasks/<TASK_ID>.md` file.
- Default to one formal task at a time unless explicitly parallel.
- Do not silently expand scope.
- Business/test reports normally end in `待验收`; QA reports end only in `已通过`, `未通过`, or `阻塞`.
- No direct live promotion. Any live micro test requires replay, acceptance, risk diagnostics, and QA first.
- Strategy changes must distinguish three evidence layers:
  - action-path coverage
  - replay-model regression
  - live-derived source-path proof

## Current Status

Current focus:

- `0513T002`: MarketView provenance / top5 audit transparency implementation is `已通过`.
- `0513T003`: `5-13-day-control-15min` live-data validation for T002 is `已通过`.
- `0513T004`: Deployment reproducibility / startup compatibility gate is `已通过`.
- `0513T005`: Latency and market-data integrity baseline planning is `已通过`.
- `0513T006`: Step 2 read-only latency / market-data integrity analyzer implementation is `已通过`.
- `0513T007`: Binance raw provenance / top5 sidecar and decision join implementation is `未通过`.
- `0513T008`: `5-13-day-control-30min` T004/T007 full-run data-quality collection is `已通过`.
- `0513T009`: T007 Binance snapshot bootstrap / buffered depth replay fix is `已通过`.
- `0514T001`: Stage 3 market-view acceptance gate implementation is `已通过`.
- `0514T002`: Stage 4 pricing-model research plan is `已通过`.
- `0514T003`: Stage 4 read-only pricing-model research runner implementation is `已通过`.
- `0514T004`: Maker execution outcome research requirements is `已通过`.
- `0514T005`: Maker execution outcome label runner implementation is `已通过`.
- `0514T006`: Stage 6A fill/cancel lifecycle proxy calibration plan is `已通过`.
- `0514T007`: Stage 6B replay/live execution outcome calibration runner implementation is `已通过`.
- `0514T008`: Replay fill/cancel lifecycle mismatch diagnosis / repair plan is `已通过`.
- `0515T001`: Read-only replay lifecycle mismatch diagnosis implementation is `已通过`.
- `0515T002`: Replay lifecycle repair design / implementation plan is `已通过`.
- `0515T003`: Replay lifecycle repair implementation is `已通过`.
- `0515T004`: Residual replay fill mismatch diagnosis is `已通过`.
- `0515T005`: Narrow cancel-race residual repair is `已通过`.
- `0515T006`: Replay fill trigger diagnosis for residual case 4948 is `已通过`.
- `0516T001`: Queue/priority evidence diagnosis for residual case 4948 is `已通过`.
- `0516T002`: Queue-ahead proxy repeatability diagnosis is `已通过`.
- `0518T001`: Conservative queue proxy gate repair design is `已通过`.
- `0518T002`: Step 5A BBO quote-anchor / post-only design contract is `已通过`.
- `0518T003`: Step 5B quote-anchor / post-only read-only diagnostic is `已通过`.
- `0518T004`: Step 5C narrow quote-anchor safety layer is `已通过`.
- `0519T001`: Step 6 final lifecycle calibration rerun is `已通过`.
- `0519T002`: Step 6 closure decision and boundary update is `已通过`.
- `0519T003`: Step 7 inventory and execution model redesign contract is `已通过`.
- `0519T004`: Step 8 quote-update mechanics and API-limit hygiene design contract is `已通过`.
- `0519T005`: Step 8B quote-update churn/API/stale-price read-only diagnostic and implementation-planning is `已通过`.
- `0519T006`: Step 8C default-off quote-update helper / instrumentation implementation is `已通过`.
- `0519T007`: Step 9A default-off quote-adjustment replay experiment design contract is `已通过`.
- `0519T008`: Step 9B default-off quote-adjustment offline replay runner implementation is `已通过`.
- `0519T009`: `5-19-day-control-30min` current-format T006 audit collection and T008 rerun is `已通过`.
- `0519T010`: Step 9C multi-sample quote-adjustment validation plan is `已通过`.
- `0519T011`: Current-format no-rule/default-off night-active sample collection is `已通过`.
- `0520T001`: Step 9C read-only multi-sample quote-adjustment validation is `已通过`.
- `0520T002`: Step 9C candidate-bucket decisionability artifact hardening is `已通过`.
- `0521T001`: `5-21-day-control-60min` live test collection is `已通过`.
- `0521T002`: Step 9C candidate x scenario bucket multi-sample determination is `已通过`.
- `0525T001`: Step 9D keep-for-research candidate fine-bucket refinement is `已通过`.
- `0526T001`: Step 9E targeted active control sample collection for min-move sweep seeds is `已通过`.
- `0526T002`: Step 9E targeted active 1H control live test collection is `已通过`.
- `0526T003`: Step 9F narrow min_move_quote_age_churn_guard parameter-sweep design is `已通过`.
- `0526T004`: Step 9G narrow min_move_quote_age_churn_guard read-only parameter-sweep implementation is `已通过`.
- `0526T005`: Maker edge family read-only triage is `已通过`.
- `0526T006`: Focused maker-edge design: inventory-aware quote placement is `已通过`.
- `0526T007`: 180min current-format no-rule/default-off live test collection is `已通过`.
- `0526T008`: Immediate 30min current-format no-rule/default-off live test collection is `已通过`.
- `0527T001`: Replay audit cancel_ack bloat planning/diagnosis for `0526T008` is `已通过`.
- `0528T002`: Compact replay lifecycle audit export for Stage 6 input is `已通过`.
- `0528T001`: Read-only inventory-aware quote placement runner implementation is `已通过`.
- `0529T001`: Fill-quality-first maker edge synthesis and next policy design is `已通过`.
- `0529T002`: Read-only fill-quality bucket synthesis runner implementation is `已通过`.
- `0529T004`: Hyperliquid read-only public market-data evidence hardening is `已通过`.
- `0529T005`: Stage 9L fill-quality rejection decomposition / bucket coarsening read-only analysis is `已通过`.
- `0530T001`: Hyperliquid public market-data research consumer design contract is `已通过`.
- `0530T002`: Stage 9M targeted clean-fill evidence collection / read-only rerun is `已通过`.
- `0531T001`: Hyperliquid public market-data research consumer read-only implementation is `已通过`.
- `0531T002`: Stage 9N clean-fill evidence viability refinement has completed business execution and has passed QA; it remains unblocked by `0530T002` QA.
- `0601T001`: Hyperliquid lag-venue public BTC sample collection and alignment is `已通过`.
- `0602T001`: Binance lead / Hyperliquid lag synchronized public-data collection is `已通过`.
- `0601T002`: Binance lead / Hyperliquid lag read-only as-of joined-feature input is `已通过`.
- `0601T003`: Binance-to-Hyperliquid read-only lead-lag stability analyzer is `已通过`.
- `0601T004`: Binance-led Hyperliquid maker data input contract is `已通过`.
- `0601T005`: Binance-led Hyperliquid read-only pricing-signal runner implementation is `待验收`.

Current QA queue:

| Task ID | Title | Status | Why It Matters |
|---|---|---|---|
| `0512T008` | hbt.depth live/replay view and data-layer quality gate | 已通过 | Established that compressed action-path alignment is not full L2 / queue / OFI proof. |
| `0513T002` | MarketView provenance / top5 audit transparency minimal implementation | 已通过 | Adds explicit live/replay/audit-overlay market-view source fields. |
| `0513T003` | 5-13-day-control-15min T002 live-data validation | 已通过 | Validates T002 fields on a fresh 15-minute no-rule live sample. |
| `0513T004` | Deployment reproducibility / startup compatibility gate | 已通过 | Adds live startup preflight manifest and schema/strategy fail-fast compatibility check. |
| `0513T005` | Latency and market-data integrity baseline plan | 已通过 | Plans Step 2 metrics, samples, outputs, acceptance criteria, and follow-up task split. |
| `0513T006` | Step 2 read-only analyzer implementation | 已通过 | Generated Step 2 artifacts and sample-usability classifications over existing samples; no live, strategy changes, or replay sweeps. |
| `0513T007` | Binance raw provenance / top5 sidecar and decision join | 未通过 | QA found snapshot bootstrap bug: buffered depth before snapshot was not replayed, causing full-run gap-crossed joins. |
| `0513T008` | 5-13-day-control-30min T004/T007 full-run data-quality collection | 已通过 | Collected a fresh no-rule T004-standard 30min sample and classified full-run T007 sidecar/join quality. |
| `0513T009` | Fix T007 snapshot bootstrap / buffered depth replay | 已通过 | Repairs sidecar local-book bootstrap and re-validates on `5-13-day-control-30min`. |
| `0514T001` | Stage 3 market-view acceptance gate implementation | 已通过 | Adds optional market-view quality gate to maker acceptance using T009 fixed sidecar/join metrics. |
| `0514T002` | Stage 4 pricing-model research plan | 已通过 | Defines read-only pricing-model research candidates, markout evaluation, outputs, and next implementation task. |
| `0514T003` | Stage 4 read-only pricing-model research runner implementation | 已通过 | Implements and runs the read-only pricing research runner on `5-13-day-control-30min`. |
| `0514T004` | Maker execution outcome research requirements | 已通过 | Defines execution-outcome labels, the seven added label gaps, and per-label statistical methods for later maker outcome research. |
| `0514T005` | Maker execution outcome label runner implementation | 已通过 | Implements the first read-only execution-outcome label layer and constrains how Stage 6 should be framed. |
| `0514T006` | Stage 6A fill/cancel lifecycle proxy calibration plan | 已通过 | Defines the Stage 6 label schema, comparison unit, strata, sample policy, and Stage 6B boundary. |
| `0514T007` | Stage 6B replay/live execution outcome calibration runner implementation | 已通过 | Proves the methodology on one sample and shows replay lifecycle remains too far from live. |
| `0518T003` | Step 5B quote-anchor / post-only read-only diagnostic | 已通过 | Quantifies BBO drift, quote-distance, reject/throttle/churn, stale/join-age/latency, and rounding/clamp risk on the accepted sample. |

Immediate next controller action:

1. `0521T002` has passed QA.
2. `0525T001` has passed QA.
3. `0526T001` passed QA after completing the first targeted active no-rule / default-off sample; second/third samples were paused per user instruction.
4. `0526T002` passed QA as a caveated 1H active no-rule / default-off live control sample.
5. Do not proceed directly to Step 10 tiny-live design: the 1H sample adds `164` fills but is caveated by `top5_join_age_ms_p99=69.9977ms`, so strict clean-only fill mass still does not meet the `500` fill gate.
6. `0526T004` passed QA. The read-only sweep found `0` promising parameter sets, `80` reject parameter sets, and `676` not-decisionable parameter sets; it does not support live/default-on/tiny-live promotion.
7. `0526T005` passed QA. It found all five maker-edge families have clean evidence strong enough for follow-up, with inventory / quote-distance / size-side ranked highest and fair-price / reservation still promising but currently similar under Stage 5 labels.
8. `0526T006` passed QA. It defines one focused `inventory_aware_quote_placement_request` design: inventory state controls side preference and size pressure, fair/reservation edge is pricing context, quote-distance defines participation frontier, and stale/latency/post-only fields remain safety context only. It authorizes only a later read-only/default-off runner implementation task, not strategy implementation, parameter search, live, default-on, or promotion.
9. `0526T007` passed QA as a 180min current-format no-rule/default-off control collection plus derived diagnostic artifact task. It remains control evidence only and does not authorize candidates, relaxed guards, live/default-on/tiny-live, or promotion.
10. `0526T008` passed QA as a 30min current-format no-rule/default-off control collection plus derived diagnostic artifact task: maker acceptance and market-view gates passed, T009 join quality is clean, Stage 5/5C/6/9B/9D outputs exist, and archive/checksum were generated. The run used a lifecycle-minimized replay audit for Stage 6 because the full replay audit emitted repeated cancel-ack lifecycle rows and was too large for the calibration runner; the original full replay audit remains preserved locally. This remains no-rule/default-off control data only, with no candidate enablement, guard relaxation, strategy changes, parameter sweep, tiny live, or promotion claim.
11. `0527T001` has been refined from placeholder into a small planning/diagnosis-only task using `0526T008` / `5-26-active-minmove-control-30min-b` as input. It must identify whether repeated `cancel_ack` rows come from audit export repetition, replay lifecycle state repetition, or Stage 6 input-scaling assumptions, then recommend the smallest later repair boundary. It does not authorize code repair, live, replay lifecycle semantic changes, Stage 6 implementation changes, strategy changes, parameter search, or promotion.
13. `0527T001` passed QA. It found the 21GB replay audit is primarily caused by audit replay lifecycle export / order-state tracking repetition: full replay audit has `32,893,719` `cancel_ack` rows but only `10,380` unique `order_id`s with cancel_ack, while the lifecycle-min input keeps exactly those first terminal facts for Stage 6. QA next-step recommendations: first create a narrow implementation task for compact replay lifecycle audit export / terminal-order de-dup with Stage 6 consuming that compact artifact by contract; second keep full forensic audit optional/off the default Stage 6 path and avoid strategy/live/promotion changes. This still does not authorize strategy changes, live, parameter search, or promotion.
14. `0528T002` passed QA. It implements a formal compact replay lifecycle audit export, with terminal lifecycle rows de-duplicated inside that compact export by Stage 6 semantics, and wires Stage 6 to consume the compact artifact by contract. Bounded `0526T008` validation over the preserved full replay audit prefix scanned `250,000` rows and wrote `23,345` compact rows after skipping `226,655` duplicate terminal rows; Stage 6 then ran against `audit_bt_audit_replay.compact_lifecycle.csv` in a task-scoped run dir with `decision_state=methodology_valid_single_sample`. It did not change strategy behavior, live behavior, fill/cancel replay semantics, queue/touch logic, parameters, guards, default-on behavior, or promotion state.
12. `0528T001` passed QA. It implemented the read-only/default-off offline runner for the `0526T006` accepted design, generated Stage 9I artifacts over the accepted current-format sample set, and produced a `reject` recommendation: clean request buckets have enough fill mass but worse 5s markout and spread capture than no-change buckets. The short interpretation is that the skeleton found more fills, but they were worse fills because it turned inventory state into quote-placement requests before proving those request buckets had positive fill quality. It did not change strategy behavior, run live, perform parameter search, default-on any behavior, relax guards, or make promotion claims.
15. `0529T001` passed QA. It recommends switching the next Binance maker policy work to fill-quality-first synthesis: existing evidence is enough to reject the fixed inventory skeleton and current min-move grid, but not enough to implement a new strategy policy. The proposed next task is a read-only fill-quality bucket synthesis runner that can decide whether passive quality gating with inventory sizing or reduce-side participation with a spread-capture floor is worth later implementation.
16. `0529T002` passed QA. Stage 9K generated read-only fill-quality bucket synthesis artifacts over nine current-format samples. Clean-only trigger evidence has `35,266` rows, `994` fills, `314` decision-visible trigger buckets, `0` ready-for-policy-design buckets, `68` needs-more-clean-fills buckets, and `246` reject-quality-negative buckets. Shape A and Shape B both have `0` candidate rows, so the result does not support a later policy-design contract yet.
17. `0529T005` passed QA. Stage 9L reconstructed row-level observed submit evidence from the Stage 9K sample manifest, decomposed rejection reasons, tested churn hard-gate sensitivity, and evaluated decision-visible coarsening variants. Final classification is `needs_targeted_clean_fills`; Shape A / Shape B candidates remain `0`, so this still does not authorize policy design, strategy implementation, parameter search, live/default-on, guard relaxation, tiny-live, or promotion.
18. `0529T004` passed QA. It proves a fresh Hyperliquid public-only BTC sample can support public market-data pricing / market-view research (`passes_pricing_research_market_view`), but it does not authorize private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.
19. `0530T001` passed QA. It defines the Hyperliquid public market-data research consumer contract over the accepted `0529T004` fresh sample, rechecks official public docs successfully, and recommends only a later read-only consumer implementation. QA rechecked the official public docs URLs and received HTTP 200. It does not implement consumer code or authorize private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.
20. `0530T002` passed QA after total controller explicitly ratified / accepted the already collected Stage 9M artifact `5-31-stage9m-cleanfill-control-120min-a`. The Stage 9M data/artifact chain reproduced cleanly: existing samples were scanned first; `5-13-day-control-30min` could not add top-gap evidence (`0` rows/fills); one `120min` no-rule/default-off control sample was collected; maker acceptance, market-view, T009, Stage 5, Stage 5C, Stage 6, Stage 9K, and Stage 9L artifacts are present; aggregate Stage 9K clean-only fills moved `994 -> 1102`; the top Stage 9L gap moved only `34 -> 36` fills; coarsened ready buckets and Shape A / Shape B candidates remain `0`. QA caveat: original fixed logs/reports did not prove pre-start approval; acceptance is based on current controller ratification. This still does not authorize policy design, strategy implementation, candidate enablement, guard relaxation, parameter search, tiny-live/default-on, or promotion.
21. `0531T001` passed QA. It implements a read-only local Hyperliquid public market-data research consumer over accepted `0529T004` artifacts, produces deterministic market-view/pricing features, and keeps trade pressure disabled because public trade side semantics remain unverified. It does not authorize fresh collection, private/order endpoints, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.
22. `0531T002` has been created as the next Binance Stage 9N task and is unblocked by `0530T002` QA. It is a read-only evidence refinement over Stage 9M artifacts to decide whether to stop top-gap collection, do only a short `4h-6h` style threshold-crossing collection, or pivot to alternative decision-visible regimes. Business execution has completed and the task has passed QA. It does not authorize new collection, policy design, strategy implementation, candidate enablement, guard relaxation, parameter search, tiny-live/default-on, or promotion.
23. `0601T001` passed QA. It collected and aligned one additional Hyperliquid public-only BTC lag-venue sample with classification `passes_pricing_research_market_view`; this remains lag-venue state / execution-context evidence only.
24. `0602T001` passed QA. It generated a synchronized public-only Binance lead / Hyperliquid lag sample with `1800.105s` overlap and usable Binance sidecar plus Hyperliquid alignment artifacts; it is input for a later `0601T002` read-only join, not a lead-lag statistical conclusion.
25. `0601T002` passed QA. It converts the synchronized public sample into local-observation-time as-of joined features with `future_join_count=0`, `missing_binance_join_count=0`, and `primary_usable_row_count=3596`; it does not establish lead-lag stability or strategy signal proof.
26. `0601T003` passed QA. It finds `18 stable_enough_for_pricing_research`, `6 watch_only`, and `30 unstable` Binance-to-Hyperliquid feature/outcome pairs, with effective future-age audit fields exposing the ~500ms Hyperliquid decision-grid constraint. It supports only later read-only pricing-signal/data-input contract design.
27. `0601T004` passed QA as a read-only Binance-led Hyperliquid maker data input / next-runner contract. It separates Binance lead pricing inputs from Hyperliquid lag venue-state/context inputs and continues to forbid private/order/strategy/live/parameter/default-on/tiny-live/promotion.
28. `0601T005` has completed business execution and is waiting for QA. It implements a read-only pricing-signal runner over accepted local `0601T002/0601T003/0601T004` artifacts, generates `21541` pricing signal rows from `3596` primary rows, enforces the four-feature `0601T004` Binance allowlist, reports nominal horizon plus effective future age, and recommends `keep_for_read_only_research`. It does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## Accepted Facts

These facts should constrain future task design:

- Stage 6J replay regenerates simulated order lifecycle. It is a replay-model regression gate, not proof that live adverse-selection source-path risk improved.
- `0512T005` / `0512T007` ruled out the current pure toxic timing submit-suppression line:
  - It has action-path coverage.
  - It does not remove the replay risk orders.
  - 50/100/200ms windows are equivalent on the tested samples.
  - It does not provide live-derived source-path proof.
- T002/T003 proved provenance transparency, not full book alignment:
  - live decision rows: `market_view_source=live_depth`, `top5_source=live_depth`
  - normal replay decision rows: `market_view_source=replay_depth`, `top5_source=replay_depth`
  - audit replay rows: `market_view_source=audit_overlay`, `market_overlay_source=audit`, `top5_source=replay_depth`
- Top5 is not fully aligned on `5-13-day-control-15min`:
  - top5 tick mismatch rows: about `6.84%`
  - top5 qty mismatch rows: about `9.39%`
  - tail quantity differences are large, especially on ask side.
- Existing audit/replay is enough for compressed action-path acceptance of the current simple strategy.
- Existing audit/replay is not enough to prove full L2 / queue / OFI / microprice equivalence.
- Binance update ids, bookTicker provenance, and per-decision full top-N book provenance remain later data/core tasks if needed.
- `0513T004` closes the first deployment reproducibility gap locally:
  - `run_live.sh` now runs a preflight before tmux/live startup.
  - preflight records commit, git dirty status, config hashes, key code hashes, schema compatibility, start marker, and stop marker paths.
  - stale `audit_schema.py` / strategy mismatch is now a startup failure instead of a live CSV writer failure.
  - This is a deployment gate only; it does not prove strategy PnL or market-view/full L2 alignment.
- `0514T005` adds the first read-only execution-outcome label layer on `5-13-day-control-30min`:
  - coverage status: `10` label classes `available`, `4` `observed_only_proxy`, `1` `low_sample`, `0` `unavailable`
  - sample shape: submit `2516`, filled `53`, canceled `2452`, fill-after-cancel `16`, fast-cancel-churn `1955`, partial-fill `0`
  - fill mass is not only ultra-short-horizon: fill-by-`100/500/1000/5000ms` is `8/22/28/40`
  - queue/priority, missed opportunity, and realized PnL decomposition remain observed-only proxies; tail-risk remains low-sample
  - this supports refining Stage 6 toward replay/live fill-cancel lifecycle proxy calibration instead of exact queue-model language
- `0514T007` constrains the next step:
  - matched submit coverage is complete (`2516/2516`), so comparison-unit instability is no longer the dominant blocker
  - replay lifecycle still materially overfills relative to live (`172` vs `53`), overstates fill-after-cancel-request (`133` vs `16`), and diverges on final states and cancel-to-fill delay
  - therefore the next useful task is replay fill/cancel lifecycle mismatch diagnosis / repair planning, not sample-first expansion
- `0515T003` / `0515T005` materially reduced replay/live lifecycle mismatch on `5-13-day-control-30min`; the remaining structural blocker is no longer broad cancel/fill lifecycle drift, but residual queue/touch fill optimism around `4948`-like cases.
- `0516T001` classifies `4948` as `queue_ahead_depth_can_absorb_observed_trades`, not as a hidden-trigger unknown: same-price trades hit the order price, but observed same-price trade qty is below visible queue proxy.
- `0516T002` shows queue-ahead proxy no-fill behavior repeats in the sample, but replay-fill false-positive evidence remains single-case (`4948`). This supports design-only planning, not generalized repair implementation.
- `0519T002` closes Step 6 for roadmap progression, not for promotion or live readiness:
  - the original broad replay/live lifecycle blocker is no longer `diagnostic_only_gap_too_large`
  - the accepted state is `requires_more_current_format_samples`
  - same-sample event classification is good enough for later default-off offline quote-adjustment replay methodology
  - timing magnitude gaps, the single `4948` queue/touch residual, and single-sample evidence still block live promotion and any generalized queue/touch repair
  - more current-format samples are required before promotion-style claims, but they do not block Step 7 / Step 8 design work
- `0513T006` classifies existing samples for Step 2:
  - `5-13-day-control-15min` is only a limited `pricing_research_candidate` for live-audit compressed BBO/mid sanity checks.
  - `5-11-night-active`, `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small` are `compressed_action_path_only`.
  - No current sample is a `queue_fill_research_candidate`.
  - All five samples are legacy/pre-T004 from deployment-manifest perspective.
  - Current converted npz still lacks Binance `U/u/pu`, `lastUpdateId`, bookTicker provenance, and per-decision top-N provenance.
- The next data-provenance implementation direction is `0513T007`:
  - Keep the standard hftbacktest `data` npz main event array unchanged.
  - Add Binance raw provenance / top5 sidecars.
  - Require explicit `raw_seq -> final npz rows -> reconstructed top5 book -> decision rows` mapping after converter ordering.
  - Require as-of decision join and join-age acceptance before any top5 pricing / top5 OFI proxy / top5 microprice proxy research.
  - T007 is top5-only: it does not require full L2 provenance and cannot prove exact queue position.
  - T007 should remain a bounded standalone sidecar/join implementation; standard `align_live_run.py` integration and canonical audit schema changes are later tasks.
  - Defer connector/core API changes until there is evidence that the live strategy must consume those fields in real time.

## Strategic Interpretation

The current plan is based on `docs/hft-share.md` and the recent T005-T003 evidence.

Main interpretation:

- Do not treat isolated suppress guards as the main path to profitability.
- Treat microprice / OFI / OBI / lead-lag / fresh-price effects as inputs to a stronger `fair/reservation` pricing model, not standalone trigger rules.
- Optimize by raising the weakest layers above the acceptance line, not by overfitting one layer. A profitable maker strategy needs acceptable data integrity, fair-price quality, quote logic, inventory/risk control, execution hygiene, and replay/live comparability at the same time.
- Top5 provenance is the current practical data-view boundary. Full L2 provenance and exact queue position are later enhancements, not current Step 2 blockers, unless top5 evidence proves insufficient.
- Keep the first production track as single-exchange one-way maker.
- Cross-exchange hedging, multi-account scaling, and multi-symbol capital rotation are later scaling work, not the immediate research path.
- BBO / bookTicker anchoring, GTX post-only protection, latency guards, queue/fill calibration, and inventory execution are part of the same maker edge, not separate afterthoughts.

## Ten-Step Plan

The following ten steps are the current roadmap. They are not automatically authorized tasks. Each step must be split into a narrow workflow task before execution.

### 1. Deployment Reproducibility Gate

Goal:

- Make every live/replay run traceable to exact code, schema, config, and raw data.

Scope:

- Standardize AWS updates through git, not scp.
- Record deployed commit, config hash, audit schema hash, start/stop markers, raw manifest, and run-local logs.
- Add or document startup compatibility checks so a stale `audit_schema.py` cannot silently break live collection.

Acceptance:

- A fresh no-rule run can prove which commit and schema produced the artifacts.
- Startup fails fast on schema/strategy mismatch.

Current status:

- `0513T004` implemented the local startup gate and passed QA.
- A later fresh no-rule run should use this gate and verify the manifest artifacts in the collected run directory.

### 2. Latency And Market-Data Integrity Baseline

Goal:

- Know whether the market view is good enough for pricing research.

Scope:

- Measure feed latency, event-to-order latency, strategy compute latency, order entry/response latency, and jitter.
- Validate Binance depth reconstruction, update-id continuity, bookTicker/depth consistency, top-of-book drift, and top5 mismatch buckets.
- Decide whether current Python-layer audit is enough or whether a core/data task must preserve `U/u/pu`, `lastUpdateId`, bookTicker provenance, and per-decision top-N snapshots.

Acceptance:

- Report says whether each sample is usable for compressed action-path alignment only, pricing research, or queue/fill research.

Current planning status:

- `0513T005` produced the Step 2 plan and passed QA.
- `0513T006` generated the read-only analyzer artifacts and passed QA.
- Initial result: existing samples support compressed action-path diagnostics and limited compressed BBO/mid pricing sanity only; they do not support queue/OFI/microprice research.
- `0513T007` has been executed and is waiting for QA.
- The intended near-term research basis is top5 only, not full L2. Queue research at this stage means top-of-book/top5 size and age proxies, not exact queue position.
- A fresh T004-standard no-rule run is a later separate task only if Step 2 requires fresh deployment-provenance evidence.
- `0513T008` collected a fresh T004-standard no-rule sample `5-13-day-control-30min` and classified it as `pricing_research_candidate`, limited to compressed action-path and BBO/bookTicker/compressed-mid sanity. It is not usable yet for top5 microprice / top5 OFI proxy or queue/fill proxy research because first-valid snapshot/update alignment failed and every decision join is gap-crossed.
- `0513T007` failed QA because the T007 sidecar did not replay buffered depth updates captured before the REST snapshot. On `5-13-day-control-30min`, buffered `raw_seq=5` covers `lastUpdateId+1`, but current logic starts with `raw_seq=7`; this is a sidecar bootstrap bug, not a sample-duration issue.
- `0513T008` passed QA. It proves the fresh T004-standard no-rule collection, archive, action-path acceptance, and T007 full-run quality classification; it correctly exposed the original T007 snapshot/bootstrap bug.
- `0513T009` passed QA. It fixed buffered depth replay, regenerated sidecar/join on the existing `5-13-day-control-30min`, and upgraded the sample for top5 microprice / OFI proxy / imbalance pricing research candidates. It still does not authorize full L2 equivalence claims, exact queue/fill proof, strategy changes, or live promotion.
- Step 2 is complete enough to move forward: the remaining non-exact live-audit-vs-sidecar top5 ticks/qtys should become Step 3 market-view acceptance thresholds and classifications, not a reason to keep extending the T009 bootstrap fix.

### 3. Market-View Acceptance Gate

Goal:

- Upgrade `maker_acceptance.py` from action-path acceptance to action-path plus market-view quality acceptance.

Scope:

- Keep action/planned/reject/throttle/working-order/replay-lag gates.
- Add explicit diagnostics or thresholds for best bid/ask drift, top5 tick/qty match, source fields, overlay provenance, startup-excluded book quality, and latency regimes.

Acceptance:

- Acceptance output clearly marks whether a sample passes market-view quality for pricing and fill-model research.

Current status:

- `0514T001` passed QA. It uses existing `5-13-day-control-30min` only; no live, strategy, core, connector, or standard npz schema change was performed.
- Stage 4 preconditions are satisfied for read-only pricing-model research on accepted samples. This authorizes fair-value / markout / top5 microprice / OFI proxy / imbalance studies only; it does not authorize live, strategy-rule implementation, queue/fill proof, or production promotion.

### 4. Pricing-Model Research

Goal:

- Find a stronger current fair-value model.

Scope:

- Build read-only fair-value studies over accepted samples.
- Candidate inputs:
  - BBO/bookTicker anchor
  - mid / weighted mid / microprice
  - top1/top5/top10 imbalance
  - OFI / smoothed OBI bucket
  - spread / realized volatility
  - lead-lag proxies
  - fresh-price reversal buckets
- Evaluate 100ms / 500ms / 1s / 5s markout and side-adjusted markout.

Acceptance:

- Produce a candidate `fair/reservation` adjustment study.
- Do not implement a live strategy rule in this step.

Current readiness:

- Ready to start as a read-only research task after `0514T001` QA passed.
- Primary accepted sample: `5-13-day-control-30min`, classified by Stage 3 as `passes_pricing_research_market_view`.
- Allowed research basis: BBO/bookTicker anchor, mid/weighted-mid, top5 microprice, top5 imbalance, OFI proxy, spread/volatility, lead-lag proxies, and markout evaluation.
- Not authorized: strategy code changes, live collection, live micro test, full L2 equivalence claims, exact queue-position proof, or queue/fill model calibration.

Current task:

- `0525T001` has completed the Step 9D fine-bucket refinement task and is waiting for QA.
- `0521T002` has completed the read-only Step 9C candidate x scenario bucket multi-sample determination task and passed QA.
- `0514T003` passed QA after implementing and running the read-only research runner on `5-13-day-control-30min`.
- Primary output directory for `0514T003`: `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`.
- Planned outputs: `pricing_research_summary.md`, candidate metrics CSV/JSON, bucket tables, markout-by-horizon CSV, rejected-signal list, and run manifest.
- Full-run result: `47499` decision rows, `47067` primary non-stale rows, `432` stale rows excluded from primary, `0` future/missing/gap-crossed/startup rows, and `14` candidate_for_followup signals under the default threshold.
- Strongest primary non-stale candidates are top5/top1 imbalance and microprice-family signals at `500ms` markout; this is research evidence only and does not authorize strategy implementation or live.
- `0514T004` has been created as a requirements-only follow-up. It defines the maker execution outcome questions that must be answered before any fair/reservation or quote-control strategy implementation: fill probability, time-to-fill, adverse selection after fill, spread capture, queue/priority proxies, cancel-to-fill race, post-only/reject/throttle/churn, and inventory impact.
- `0514T004` has been expanded to cover seven additional label classes: quote placement / distance, missed-fill / opportunity cost, realized PnL decomposition, tail risk, partial-fill / order lifecycle, inventory cycle, and sample validity / censoring.
- `0514T004` now requires later implementation plans to choose statistics by label type: Spearman/Pearson plus bucket monotonicity for continuous labels, bucket event rates/lift/odds ratio for binary labels, Kaplan-Meier/discrete hazard or Cox-style methods for censored time-to-event labels, rate ratios or count models for count labels, contingency/mutual-information style summaries for lifecycle labels, and tail quantile/CVaR-like summaries for tail labels.
- `0514T004` passed QA. It remains a requirements-only precursor and does not itself authorize code implementation beyond the separate `0514T005` task.
- `0514T005` passed QA after implementing the read-only execution outcome label runner, focused tests, and full-run validation on `5-13-day-control-30min`.
- `0514T006` has been executed as the planning-only Stage 6A task and is waiting for QA.
- `0514T006` planning result: Stage 6 should compare matched submit opportunities first, then report lifecycle and strata gaps; `5-13-day-control-30min` is sufficient for single-sample methodology and runner validation, but not enough alone for quote-adjustment promotion.

### 5. BBO Quote Anchor And Post-Only Protection Review

Goal:

- Decide whether quote placement should anchor to fast BBO/bookTicker plus pricing adjustment instead of trusting slower or mismatched depth-derived views.

Scope:

- Review GTX/post-only behavior.
- Review `bid <= best_bid`, `ask >= best_ask`, tick rounding, stale quote prevention, latency guard, and reject paths.
- Confirm whether top5 is a pricing input, a risk input, or a quote-anchor input.

Acceptance:

- Produce a design recommendation before implementation.

Current planned tasks:

- `0518T002` is Step 5A: design-only quote-anchor / post-only contract and has passed QA. It recommends fast BBO/bookTicker as the primary hard quote anchor, depth BBO as guarded fallback / consistency check, and top5 as pricing/risk/diagnostic context rather than the final hard post-only anchor. It did not implement strategy behavior.
- `0518T003` is Step 5B: read-only diagnostic implementation and has passed QA. It quantified BBO source drift, quote distance buckets, crossed/post-only-risk candidates, reject/throttle/churn, stale/join-age/latency regimes, fill/markout tradeoffs, current enforcement gaps, and a read-only rounding/clamp counterfactual on `5-13-day-control-30min`. It did not change quote placement or strategy behavior.
- Step 5A/5B are complete as design/diagnostic work. They authorize only a narrow future Step 5C candidate, not production quote-control implementation or live promotion.
- T003 tightens the next boundary: audit_depth is currently clean against its own anchor, but audit_depth vs bookTicker drift is too large to treat fast BBO/bookTicker as an already-backed hard anchor. Do not try to repair this as source-level row-exact drift alignment in the current stage.
- Retain only a narrow Step 5C candidate after `0518T003` QA:
  - goal: add a default-off / diagnostic-first quote-anchor safety layer, not a quote-control strategy
  - allowed scope: anchor arbitration, side-conservative tick rounding, anchor clamp, post-clamp post-only re-check, guarded depth fallback, stale / missing / join-age suppression for fresh add-side submits, and diagnostic counters
  - explicit non-goals: no audit_depth/bookTicker/top5 row-exact drift repair, no top5 hard-anchor promotion, no fair/reservation model change, no quote placement redesign, no replay lifecycle change, no live collection, no live promotion, and no default-on behavior
  - acceptance: default behavior remains unchanged unless explicitly enabled; focused tests cover bid/ask rounding direction, clamp, stale/missing anchor suppression, and post-clamp re-check; replay/diagnostic output proves no new post-only/crossed risk on the accepted sample
- `0518T004` has been created for this Step 5C scope and is now `已通过`.
- Step 5C should not block Step 6 / 7 / 8 planning. It is a safety precondition for later Step 9-style quote-adjustment experiments, not a requirement to finish full market-view source alignment.

### 6. Fill / Cancel Lifecycle Proxy Calibration

Goal:

- Make replay fill/cancel behavior and quote-adjustment diagnostics more trustworthy.

Refinement note:

- Stage 5 showed that the dominant current issues are high cancel churn, non-trivial fill-after-cancel-request, low fill count, and strong placement/inventory stratification in the observed sample.
- It also showed that queue-priority, missed-opportunity, and realized-PnL decomposition are still observed-only proxies, not exact queue proof.
- Therefore Stage 6 is narrowed from broad queue-model language to read-only replay/live lifecycle proxy calibration. Exact queue position remains later work.

Scope:

- Compare live and replay order lifecycle using a common execution-outcome label schema.
- Calibrate fill horizons, time-to-fill, final order state, cancel-to-fill race, fill-after-cancel-request, and fast-cancel churn.
- Compare calibration gaps within key strata such as quote placement, distance-to-BBO, edge bucket, inventory state, top-of-book/top5 size-age proxy, and latency regime.
- Build queue/fill probability proxies only at top-of-book/top5/age level. Binance depth is L2 price-level data, so this cannot be exact MBO queue position.

Acceptance:

- Report whether replay lifecycle proxies are good enough for quote-adjustment PnL experiments on the tested sample set, and which gaps remain too large.

Current planned tasks:

- `0514T006` is the planning-only Stage 6A contract for labels, strata, acceptance metrics, sample requirements, and boundaries.
- `0514T007` is the later read-only Stage 6B implementation task; it must not start live, regenerate replay matrices, or claim exact queue proof.
- `0519T001` has passed QA. It closed the evidence loop by rerunning final lifecycle calibration after the accepted replay lifecycle repairs (`0515T003` and `0515T005`), improved the state to `requires_more_current_format_samples`, and did not repair queue/touch residuals.
- `0519T002` passed QA. Its closure decision is:
  - Step 6 is closed for roadmap progression and later default-off offline experiment methodology.
  - Step 6 is not closed for promotion, live readiness, exact queue proof, or generalized queue/touch repair.
  - `4948` remains parked as a design-only residual until more current-format samples or repeat replay false-positive evidence exist.
  - More current-format samples are required before promotion-style decisions, but they do not block Step 7 / Step 8 design tasks.

### 7. Inventory And Execution Model Redesign

Goal:

- Improve one-way maker inventory behavior around the pricing model.

Scope:

- Review keeping inventory within one order quantity when possible.
- If inventory exceeds one order quantity, evaluate stronger skew back toward small inventory.
- Treat crossing zero as a cycle reset where appropriate.
- Evaluate AS-style dynamic spread and dynamic order amount using volatility and trading intensity.
- Consider TTL / triple-barrier style exit handling only as default-off designs after pricing/fill evidence exists.

Acceptance:

- Produce a design contract for inventory/execution changes, with no live authorization.
- May start after `0519T002` QA as design-only work.
- Must treat replay lifecycle as event-classification usable but not exact queue proof.
- Must not implement strategy behavior or live changes in the design task.

Current planned tasks:

- `0519T003` passed QA. Its Step 7 design contract is:
  - Objective: keep normal inventory close to one order quantity, reduce time spent in larger directional exposure, and make inventory recovery explicit instead of relying on symmetric quote churn.
  - Initial target: define a soft inventory target around `0` and one-order-quantity bands; treat inventory beyond one order quantity as a recovery regime that should skew quoting toward reducing exposure.
  - Quote-side semantics: inventory skew should prefer reservation / fair shift, spread widening, size reduction, and optional add-side suppression before any aggressive exit design. It must not use future markout or audit-overlay fields as live decision inputs.
  - Zero-crossing semantics: crossing through zero can reset inventory cycle diagnostics, so later evaluation should report inventory cycles, cycle duration, max excursion, recovery fills, and adverse markout while reducing inventory.
  - AS-style candidates: dynamic spread and dynamic order amount may be considered as default-off candidates using decision-time-visible volatility, fill intensity, inventory, latency, and lifecycle proxies. They are not authorized as implementation in T003.
  - TTL / triple-barrier: keep as later default-off design candidates only. They require pricing and lifecycle evidence and must be tested offline before any live discussion.
  - Required audit/diagnostic fields before implementation: inventory band, inventory cycle id, skew regime, quote side suppression reason, size multiplier, spread multiplier, reservation shift, TTL state, and recovery-mode markers.
  - Evidence gates: any later implementation must pass replay acceptance, market-view gate, Step 6 lifecycle diagnostics, inventory-cycle metrics, fill-quality/markout diagnostics, API/churn limits, and QA. Single-sample PnL is not enough.
  - Next recommendation after QA: proceed to Step 8 design-only before implementation, so quote-update mechanics and API-limit hygiene constrain the Step 7 candidate set before a Step 9 offline replay experiment.

### 8. Quote Update Mechanics And API-Limit Hygiene

Goal:

- Reduce stale/bad-price exposure without losing useful queue position or breaching API limits.

Scope:

- Prefer replace/modify logic driven by bad-price ticks and time windows over blind cancel+new churn.
- Re-check quote throttle, token bucket, API interval guard, min quote move, cancel/re-add churn, and cancellation-limit risk.

Acceptance:

- Produce either a no-change conclusion or a default-off quote-update design.
- May start after accepted Step 7 design, or as a later design-only workflow task under the `0519T002` QA-passed boundary.
- Must preserve Step 5C boundaries: quote-anchor safety remains default-off / diagnostic-first unless a later task explicitly changes it.
- Must not use GTX rejects as normal control flow and must not start live.

Current planned tasks:

- `0519T004` passed QA. Its Step 8 design contract is:
  - Quote-update mechanics should be driven by bad-price ticks, minimum quote move, stale/missing anchor, join-age / latency, and bounded time-window triggers, not blind cancel+new churn.
  - Preferred future implementation shape is a staged decision: hold quote, modify/replace in place if the venue/API path supports it, or cancel+new only when the quote is materially unsafe, stale, crossed-risky, inventory-worsening, or past a bounded age.
  - GTX/post-only reject remains an exchange backstop and diagnostic bucket, not normal control flow.
  - Step 5C quote-anchor safety stays default-off / diagnostic-first; Step 8 may require future explicit enablement only through a separate implementation task.
  - Step 7 inventory controls may request quote changes only through shared update-intent fields such as reason, priority, min move, side, age, and inventory regime. They must not bypass anti-churn, throttle, token bucket, stale-anchor suppression, or post-only re-check.
  - Anti-churn gates should include per-side min tick move, min quote age, max cancel/re-add rate, in-flight order guard, cancel-pending guard, recent reject/throttle cooldown, and emergency stale/bad-price override.
  - API hygiene must cover token bucket, request spacing, per-action rate budgets, cancellation-limit risk, reject/throttle/drop buckets, and observable degraded modes.
  - Required future audit fields include quote_update_intent, quote_update_action, quote_update_reason, min_move_passed, quote_age_ms, anchor_age_ms, join_age_ms, latency_bucket, throttle_state, token_bucket_state, cancel_readd_bucket, reject/throttle/drop cause, post_only_pre/post_check, and inventory_request_id.
  - Evidence gates before implementation or Step 9: replay acceptance, market-view gate, Step 5C post-only safety diagnostics, Step 6 lifecycle diagnostics, Step 7 inventory-cycle diagnostics, API/churn counters, and QA.
  - Step 9 should not start immediately after T004 QA. The recommended next task is a narrow Step 8B read-only diagnostic / implementation-planning task over existing artifacts to quantify current churn/API/stale/bad-price regimes and decide whether implementation should be no-change, default-off helper, or full default-off replay candidate.
- `0519T005` passed QA. Step 8B result is `default_off_helper_candidate`:
  - Current sample has enough quote-update pressure to justify a later default-off helper / instrumentation task.
  - Key counts: decision rows `47499`; planned submit decision rows `8971`; actual submit decision rows `2256`; planned/action mismatch rows `7186`; latency guard rows `16528`; quote throttle rows `5996`; api interval guard rows `1190`; fast-cancel churn rows `1955/2516`; Stage 5C bid/ask clamped rows `1394/2281`; stale anchor rows `65`; post-only risk after re-check rows `0`.
  - Observable now: action/planned_action, reject_reason/throttle_reason, Stage 5C anchor/clamp/suppress/recheck diagnostics, Stage 5 submit labels, and Stage 6 lifecycle calibration.
  - Missing or proxy-only: `quote_update_intent`, unified `quote_update_reason`, `token_bucket_state`, `inventory_request_id`, `min_move_passed`, `quote_age_ms`, `cancel_readd_bucket`, `latency_bucket`, production `anchor_age_ms`, and post-only pre/post-check fields.
  - Step 9 should not start directly after Step 8B. If continuing Step 8, create a separate default-off helper / instrumentation implementation task that records quote-update intent/action/reason and throttle/token/cancel-readd/post-only/inventory-request fields while keeping behavior unchanged by default.
- `0519T006` passed QA as Step 8C. It implements only a default-off helper / instrumentation layer:
  - centralized quote-update intent/action/reason
  - added audit fields for min move, quote age, join/anchor age, latency bucket, throttle/token state, cancel-readd bucket, reject/throttle/drop cause, post-only pre/post-check, and inventory request id placeholder
  - wired fields through live/backtest audit rows with stable defaults
  - preserved existing action path and throttle/API/latency suppression semantics by recording snapshots after decisions are formed
  - did not run Step 9 replay, live, default-on behavior, Step 5C promotion, or inventory-control implementation.
- `0519T007` passed QA as Step 9A design-only:
  - define the default-off quote-adjustment replay experiment candidate matrix
  - define decision-time-visible inputs, metrics, output artifacts, Step 9B runner boundary, and non-goals
  - do not implement runner, run replay, start live, enable default-on behavior, or make promotion claims.
- `0519T008` passed QA as Step 9B default-off offline runner implementation:
  - implement only a default-off offline runner
  - validate runner mechanics on `5-13-day-control-30min`
  - emit candidate matrix, per-candidate metrics, fill-quality, inventory-cycle, API/churn, post-only safety, action-path/audit coverage and acceptance decision artifacts
  - classify results as `no_effect`, `worse_due_to_churn_or_fill_quality`, `promising_but_single_sample`, `blocked_by_replay_or_market_view`, or `needs_more_instrumentation`
  - do not run live, default-enable candidates, expand samples, or make promotion claims.
- `0519T009` passed QA as the current-format sample collection follow-up:
  - collect `5-19-day-control-30min` as a 30min no-rule / default-off control sample
  - early-check that all 15 T006 quote-update audit fields exist in the live audit header
  - run audit replay, maker acceptance, and the T008 offline runner on the new dataset
  - this is instrumentation/data-quality collection only and does not authorize candidate promotion.
- `0519T010` is the next Step 9C planning-only task:
  - define current-format data scenario coverage and sample/event thresholds
  - define multi-sample replay rerun method for `quote_adjustment_replay.py`
  - define cross-regime candidate stability criteria and reject / keep-for-research / ready-for-tiny-live-design classifications
  - do not implement code, run replay sweeps, collect live data, default-enable behavior, or make promotion claims.
- `0519T011` completed business-thread execution and passed QA:
  - collected 3 separated 30min current-format no-rule/default-off night-active samples
  - run ids start with `5-19-night-active`
  - each sample completed audit replay, maker acceptance, sidecar/join, Stage 5 labels, Step 5C diagnostics, and Step 9B runner output
  - `5-19-night-active-30min-b` second raw gzip was repaired and verified; the accepted local sample is the first 30min slice
  - `5-19-night-active-30min-a` has a strict market-view caveat: `gap_crossed_join_count=28062` and missing Step 5C anchor rows
  - do not perform final read-only multi-sample validation or make candidate conclusions in T011.
- `0520T001` passed QA:
  - total controller accepted the `5-19-night-active-30min-a` caveat for research comparison only
  - accepted-set includes `5-19-day-control-30min` plus all three `5-19-night-active-30min-*` samples
  - clean-only sensitivity must exclude `5-19-night-active-30min-a`
  - accepted-set meets Step 9C research-comparison mass; clean-only is under threshold
  - no candidate is `ready_for_tiny_live_design`
  - `spread_widening_stale_latency` needs runner/artifact work for identifiable submit/fill effects
  - task reused existing `quote_adjustment_replay.py` and did not modify runner, strategy, live behavior, defaults, or promotion status.
- `0520T002` is the narrow follow-on task:
  - objective: harden runner / artifact decisionability so the current 8 Step 9 families become verdictable across existing current-format samples
  - keep sample set fixed; do not expand samples or change strategy / live behavior
  - treat `spread_widening_stale_latency` and similar mixed buckets as the primary hardening target

### 9. Default-Off Quote-Adjustment Replay Experiment

Goal:

- Test quote controls only after data, pricing, fill, and inventory evidence exist.

Scope:

- Candidate controls may include fair shift, reservation shift, spread widening, size reduction, inventory skew, queue-aware join/step-back, or latency-regime no-quote.
- Run multi-sample replay, maker acceptance, market-view gate, fill-quality diagnostics, and cancel-fill diagnostics.

Acceptance:

- Do not use single-sample PnL as evidence.
- Do not promote to live without QA.
- Requires accepted Step 7, Step 8 design, Step 8B diagnostic / implementation-planning, and Step 8C quote-update helper / instrumentation boundaries first.
- May only run as a default-off offline replay experiment.
- Requires more current-format samples before any promotion-style conclusion or live micro-test decision.

Current split:

- `0519T007` Step 9A passed QA and defined the candidate matrix, metrics, artifacts, and Step 9B acceptance gate.
- `0519T008` Step 9B implemented the authorized default-off offline replay runner over the accepted boundary and passed QA.
- `0519T009` is the accepted current-format no-rule control collection task that replaces proxy-only evidence with one T006-field sample before further Step 9B interpretation.
- `0519T010` defines the Step 9C multi-sample validation plan before sample expansion or replay sweep work.
- Sample expansion should come after the Step 9C validation plan is accepted, unless the plan identifies a hard blocker that requires runner changes first.

Step 9A design contract summary:

- Candidate families:
  - baseline/control no-change replay validation
  - fair/reservation shift using decision-time-visible pricing signals
  - inventory reservation shift / recovery-side preference request
  - spread widening in stale, latency, adverse, or inventory-worsening regimes
  - size reduction or add-side suppression in inventory / API / churn pressure regimes
  - stale / latency no-fresh-add quote regime
  - min-move / quote-age / API-churn guard regime
  - Step 5C post-only safety interaction and clamp/suppress audit replay
- Required metrics:
  - net/gross PnL, fee, spread capture
  - fill probability and time-to-fill
  - side-adjusted fill markout / adverse selection
  - cancel-fill and fill-after-cancel
  - inventory cycle, max excursion, zero-crossing and recovery quality
  - API request count, token pressure, throttle/reject/drop and churn
  - stale/bad-price/post-only safety counters
  - action-path and T006 audit-field coverage
  - Step 6 lifecycle boundary and replay/live calibration caveats
- Step 9B minimum scope:
  - implement a default-off offline runner only: complete in `0519T008`
  - validate runner on `5-13-day-control-30min`: complete in `0519T008`
  - emit candidate matrix, per-candidate metrics, action-path/audit coverage, API/churn, fill-quality, inventory-cycle and safety artifacts
  - return a decision classification such as `no_effect`, `worse_due_to_churn_or_fill_quality`, `promising_but_single_sample`, or `blocked_by_replay_or_market_view`
  - do not make live/promotion claims from a single sample.
- `0519T008` classification on the existing sample is `needs_more_instrumentation` because the historical `5-13-day-control-30min` audit predates T006 and lacks all 15 T006 quote-update fields. The runner still validates the artifacts / metrics path using proxy fields and should not be interpreted as candidate performance proof.
- `0519T009` removes the T006 instrumentation blocker on one current-format sample:
  - live audit fields: `159`
  - T006 missing fields: `0`
  - maker acceptance: passed, common rows `96340`
  - T009 sidecar join coverage `1.0`, future join `0`, gap-crossed join `0`
  - Step 9B runner classification: `promising_but_single_sample`
  - this still does not authorize promotion; it only supports later multi-sample validation planning.

Step 9C multi-sample validation plan:

- Core problem:
  - Step 9 is not a single-sample PnL search. It must determine whether default-off quote-adjustment candidates improve maker execution quality across market regimes while preserving replay/live alignment, post-only safety, API hygiene, fill quality, and inventory behavior.
  - The immediate blocker is current-format scenario coverage plus bucket decisionability. `0519T008` proves runner / artifact mechanics; `0519T009` proves one current-format T006 sample is usable. `0520T001` shows the sample set is mass-sufficient but some buckets still are not verdictable. That is enough to justify a narrow runner/artifact hardening task, not enough to accept a strategy.
- Sample policy:
  - Use only current-format no-rule or default-off control samples with the T006 quote-update fields present.
  - Each sample must have preflight manifest, start/stop markers, live audit, raw gzip, archive checksum, audit replay, maker acceptance, sidecar/join metrics, Stage 5 execution labels, Step 5C safety diagnostics, and Step 9B runner outputs.
  - Minimum for research comparison: at least `4` current-format samples including `5-19-day-control-30min`, with at least `120` minutes aggregate duration, `10000` submit orders aggregate, and `250` filled orders aggregate.
  - Minimum before any `ready-for-tiny-live-design` classification: at least `5` current-format samples, `180` minutes aggregate duration, `15000` submit orders aggregate, `500` filled orders aggregate, and at least `2` distinct non-calm regimes.
  - If fill/event thresholds are not met, collect more no-rule/default-off samples rather than lowering thresholds.
- Required scenario buckets:
  - volatility / markout: low / medium / high using realized mid return or markout dispersion.
  - spread / tick distance: one-tick tight, multi-tick, and quote-distance buckets.
  - trade intensity / fill opportunity: low / medium / high trade count or trade notional per minute, plus submit/fill opportunity density.
  - stale / latency / age: join age, anchor age, quote age, latency bucket, stale join, and stale anchor buckets.
  - API / churn: token bucket state, throttle state, reject/drop cause, cancel-readd bucket, min-move pass/fail, quote-update churn.
  - inventory: flat, mild skew, large skew, inventory-worsening add-side, inventory-reducing/recovery-side.
  - post-only / safety: clamp, suppress, guarded fallback, post-only pre/post-check, bad-price or crossed-risk counters.
  - cancel-fill risk: fill-after-cancel, cancel-to-fill delay, cancel race bucket, adverse markout after fill.
  - market-view quality: sidecar join coverage, future/gap/stale join, top5 tick/qty match, BBO anchor quality.
- Replay method:
  - Run `quote_adjustment_replay.py` per sample into a task-scoped `stage9b_quote_adjustment_replay_<task_id>/` output directory.
  - Aggregate by `sample_id x candidate_id x scenario_bucket`; report per-sample metrics, cross-sample median, win rate, worst-sample delta, and hard-gate breaches.
  - Keep candidate decisions limited to decision-time-visible inputs. Future markout, future fill outcome, audit overlay labels, exact queue claims, `4948`-specific logic, and same-sample PnL feedback remain forbidden as decision inputs.
  - If current runner output is sufficient, the next validation task may run existing runner per sample and aggregate artifacts manually. Only create a runner-change task if sample validation exposes missing required artifacts, inconsistent schemas, or inability to stratify the required buckets.
- Hard gates:
  - Data gate: T006 missing fields `0`, maker acceptance passed, action/planned/reject/throttle gates passed, working semantic/blocking mismatch `0`, strict replay lag gate passed, sidecar join coverage acceptable, future/gap-crossed join `0`, and archive/raw integrity documented.
  - Safety gate: no post-only crossed-risk after re-check; no unexplained bad-price regime; no default-on behavior; no production behavior change.
  - Candidate gate: nonzero intended-regime decision coverage in at least `2` samples before research interpretation; no hard safety breach in any accepted sample; no material API/churn/reject degradation versus baseline; no material cancel-fill or adverse markout degradation versus baseline.
- Diagnostic/proxy metrics:
  - PnL proxy, spread capture, opportunity cost, queue/priority, exact fill causality, and PnL decomposition remain diagnostic/proxy unless a later task adds stronger live-derived evidence.
  - Single-sample `promising_but_single_sample` means "eligible for multi-sample validation planning" only.
- Candidate classifications:
  - `reject`: fails data/safety hard gates, has too little coverage, worsens fill quality/adverse markout/cancel-fill/API-churn in multiple samples, or relies on forbidden inputs.
  - `keep_for_research`: has coverage and some favorable regimes, but sample count, event mass, dispersion, or proxy-only evidence is insufficient.
  - `ready_for_tiny_live_design`: passes hard gates across the required sample set, improves or does not worsen execution quality in most eligible regimes, has no catastrophic worst-sample behavior, and has QA acceptance. This still authorizes only a separate live-design task, not live execution.
- Recommended task sequence:
  1. `0520T002` has executed and now awaits QA on the hardened bucket verdict / stability artifacts.
  2. Do not create Step 10 tiny-live-design yet because T001 found no `ready_for_tiny_live_design` candidate.

### 10. Controlled Live Validation And Scaling

Goal:

- Validate a default-off candidate against fresh live controls only after all offline gates pass.

Scope:

- Run no-rule control samples first.
- Then consider tiny live micro tests only after replay, acceptance, risk diagnostics, and QA.
- Compare PnL, fill quality, markout, inventory cycles, queue/fill model error, latency, and API/drop behavior against a fresh control.

Later scaling:

- Multi-symbol selection.
- Multi-subaccount instances.
- Capital rotation to currently profitable tickers.
- Cross-exchange / XEMM.

These are not the current immediate path.

## Standard Live/Replay Loop

Every serious iteration should follow this loop unless a task explicitly says otherwise:

1. Collect a live no-rule or default-off sample with a clear run id.
2. Save live audit, raw gzip, connector logs, bot logs, config, schema hash, and deployed commit.
3. Pull artifacts into `local_live_analysis/<run_id>/`.
4. Generate archive and checksum.
5. Run normal replay and audit replay.
6. Run `maker_acceptance.py`.
7. Run market-view quality diagnostics once Step 3 exists.
8. Run risk/fill diagnostics relevant to the task.
9. Decide whether the issue is data alignment, pricing, fill model, inventory execution, or quote mechanics.
10. Only then create the next implementation or experiment task.

## Stable Rules

- `progress.md` records current operating state.
- `findings.md` records durable risks, failures, and lessons.
- `.workflow/dashboard.html` and `.workflow/dispatch_suggestions.md` are generated by `.workflow/build_dashboard.py`.
- Historical details belong in `.workflow/tasks/` and `.workflow/reports/`, not in this plan.
