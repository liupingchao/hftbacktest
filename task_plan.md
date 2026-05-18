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

1. Treat `0518T003` as QA 已通过 and keep its conclusion diagnostic-only.
2. If continuing Step 5, create only the narrow Step 5C default-off / diagnostic-first quote-anchor safety task described below.
3. Do not repair audit_depth/bookTicker/top5 row-exact drift, do not implement a generic quote-control strategy, and do not start live promotion.

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

- `0514T002` passed QA and authorizes only a read-only Stage 4 pricing-model research runner.
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

### 8. Quote Update Mechanics And API-Limit Hygiene

Goal:

- Reduce stale/bad-price exposure without losing useful queue position or breaching API limits.

Scope:

- Prefer replace/modify logic driven by bad-price ticks and time windows over blind cancel+new churn.
- Re-check quote throttle, token bucket, API interval guard, min quote move, cancel/re-add churn, and cancellation-limit risk.

Acceptance:

- Produce either a no-change conclusion or a default-off quote-update design.

### 9. Default-Off Quote-Adjustment Replay Experiment

Goal:

- Test quote controls only after data, pricing, fill, and inventory evidence exist.

Scope:

- Candidate controls may include fair shift, reservation shift, spread widening, size reduction, inventory skew, queue-aware join/step-back, or latency-regime no-quote.
- Run multi-sample replay, maker acceptance, market-view gate, fill-quality diagnostics, and cancel-fill diagnostics.

Acceptance:

- Do not use single-sample PnL as evidence.
- Do not promote to live without QA.

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
