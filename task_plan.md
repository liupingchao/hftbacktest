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
- `cross-exchange` is the canonical branch for formal Binance-lead / Hyperliquid-lag MVP work. Other branches are temporary or recovery branches until their files are restored onto `cross-exchange`.
- The four MVP milestones in `docs/cross_exchange_maker_mvp_plan.md` are the highest sequencing constraint for this branch.
- Historical task classification is tracked in `docs/cross_exchange_mvp_task_classification.md`.
- Strategy changes must distinguish three evidence layers:
  - action-path coverage
  - replay-model regression
  - live-derived source-path proof

## Current Status

Current formal task:

- `0719T006 / EXACT-TWO-SIDED-MANAGER-EVIDENCE-CONTRACT`
- status: `待执行`
- scope:
  - exact Binance edge-gated exchange-reconciled manager profile
  - two independent buy/sell attempt lifecycles and cancel proofs
  - Task 12 acceptance rejects one-sided or aggregate evidence
  - actual manager writer to full acceptance integration
  - offline-only; no live/private/account/order/cancel/network/remote/service
- Task file: `.workflow/tasks/0719T006.md`.
- No new live task may start before independent QA accepts this repair.

Latest QA result:

- `0719T005 / STRICT-CANCEL-SUCCESS-BOUNDED-ATTEMPT-REPAIR`
  - status: `已通过`
  - implementation commit: `19e4b4a`
  - cancel success semantics and bounded attempt parsing accepted
  - focused `271 passed`; full Hyperliquid `591 passed`
  - historical cancel proof compatibility accepted
  - Principal Task 12 remains open; Task 10 multi-level remains locked
- QA report: `.workflow/reports/0719T005-qa.md`.

Previous QA result:

- `0719T004 / REDACTION-SAFE-REFERENCE-IDENTITY-STRICT-ATTEMPT-REPAIR`
  - status: `未通过`
  - implementation commit: `a739a78`
  - passed: persisted token contract, actual writer exact rebuild, ordinary malformed-attempt matrix, all-token/raw-proof independence, full `526 passed`
  - P1 failure: false/null/empty `success` values are accepted as authoritative cancel proof
  - P2 failure: oversized canonical digit attempts raise rather than fail closed
  - Principal Task 12 remains open; Task 10 multi-level remains locked
- QA report: `.workflow/reports/0719T004-qa.md`.

Previous QA result:

- `0719T003 / RAW-CANCEL-PROOF-INDEPENDENT-RECONCILIATION-REPAIR`
  - status: `未通过`
  - implementation commit: `ba220c5`
  - passed: all-token target conflicts, raw unrelated/ambiguous contradictions, missing raw containers, helper independence, focused `130 passed`
  - P1 failures: persisted producer artifacts do not exact-match rebuilt summaries after redaction; fractional attempts alias through integer truncation
  - Principal Task 12 remains open; Task 10 multi-level remains locked
- QA report: `.workflow/reports/0719T003-qa.md`.

Previous QA result:

- `0719T002 / PER-ATTEMPT-REFERENCE-CANCEL-PROOF-REPAIR`
  - status: `未通过`
  - implementation commit: `7235372`
  - passed: original counterexample, nominal multi-reference producer paths, same-reference redundant generic behavior, focused `123 passed`
  - P1 failures: forged reconciliation rows/raw-proof contradiction pass acceptance; partial conflicting target identity passes producer mapping
  - Principal Task 12 remains open; Task 10 multi-level remains locked
- QA report: `.workflow/reports/0719T002-qa.md`.

Previous QA result:

- `0719T001 / T024-RUNTIME-PROVENANCE-FILL-RECONCILIATION-REPAIR-RERUN`
  - status: `未通过`
  - implementation/live source: `82f4a4d`
  - passed: exact 62-file runtime provenance, pre-start/postrun verification, `65/65` checksum, unclassified blocker rejection, single-window/account/process safety
  - P1 failure: no-fill cancel success is not bound per submitted attempt/reference
  - live result: zero submissions due `fresh_touch_session_gate_no_eligible_candidate`
  - Principal Task 12 remains open; Task 10 multi-level remains locked
- QA report: `.workflow/reports/0719T001-qa.md`.

Previous QA result:

- Previous QA result:
  - `0718T024 / T023-EXACT-ENVELOPE-IDENTITY-REPAIR-RERUN`
  - status: `未通过`
  - scope: T023 replacement window, exact task-scoped envelope/identity, runtime source provenance, fill reconciliation and same-window acceptance
  - live result: one real BTC post-only `Alo` buy `0.005 @ 64770` reached resting and was canceled; final/independent open orders `0`, BTC position `0.0`, estimated loss `0.0`, no fill rows
  - passed facts: exact `0718T024` identity, `0.005 / 1 / 0.01 / 2` runtime caps, child exit/reap, writer health, activation-off state and `61/61` checksum
  - P1 failures: sealed run lacks runtime source commit/critical-file hashes; acceptance ignored producer blocker `fill_reconciliation_required_no_fill_unproven`
  - next route: a new `0719` formal repair/re-run task must seal runtime source provenance and make producer/acceptance fill reconciliation fail-closed
- QA report: `.workflow/reports/0718T024-qa.md`.
- T024 does not close Principal Task 12 and does not unlock Task 10 multi-level.
- Latest QA result:
  - `0718T023 / P3-CUMULATIVE-TINY-LIVE-SAME-WINDOW-ACCEPTANCE`
  - status: `阻塞`
  - scope: Principal Alignment Task 12 bounded single-level tiny-live and same-window acceptance
  - live result: one real post-only `Alo` BTC resting order, cancel, final owned open orders `0`, independent proof `0`, no attributed fill/position transition
  - blocking facts: actual config snapshot used `max_loss_usdc=30.0` and `max_position_btc=0.04`; inner artifacts used stale `0623T007`/`0622T004` identities instead of `0718T023`
  - implementation repair commit: `a57c7da`
  - verification after repair: full `examples/hyperliquid` `455 passed`
  - no second live window or strategy-variable expansion was started after the stop condition
- QA report: `.workflow/reports/0718T023-qa.md`.
- T023 and T024 are failed/blocked diagnostic windows; neither is an accepted Principal Task 12 baseline.
- Latest QA result:
  - `0718T022 / P3-REAL-TIME-STATUS-FILE`
  - status: `已通过`
  - scope: Principal Alignment Task 11 complete real-time status schema and writer failure boundary
  - implementation commit: `b7bca85`
  - verification: watcher `60 passed`; related regression `90 passed`; full `examples/hyperliquid` `453 passed`
  - no live/private/order/cancel/network/remote calls; quote/order/activation behavior unchanged
- QA report: `.workflow/reports/0718T022-qa.md`.
- Principal Alignment Tasks 0-9 are QA accepted; T021 has accepted only the Task 10 default-off prerequisite/ladder gate.
- Current required next task:
  - a new formal offline repair must persist a stable non-sensitive reference identity that remains exact across producer redaction and acceptance rebuild
  - producer and acceptance must strictly reject fractional, boolean and otherwise malformed attempt identities
  - add producer-written artifact acceptance integration and malformed-attempt regressions before any new live window
  - after offline QA, allow at most one new isolated single-level two-sided manager lifecycle window under the exact standing envelope
  - do not reuse T023, T024 or the zero-submit T025 window as an accepted Principal Task 12 baseline
  - preserve exact task-scoped caps, identity, single-level behavior and activation-off state in any later task
- T022 remains QA accepted; `0719T006` is the current formal offline contract task.
- `0718T022` QA report: `.workflow/reports/0718T022-qa.md`.
- Task 10's activation gate remains fail-closed until a controlled tiny-live obtains an accepted single-level two-sided manager lifecycle.
- Accepted formal task:
  - `0717T007 / WINDOW-ATTEMPT-IDENTITY-REPAIR`
  - status: `已通过`
  - scope: `0717T006` Phase 1 only
  - no live/private/order/cancel access
  - implementation commits: `66ba588`, `0271d99`
  - focused verification: `83 passed`
- Standing authorization recorded on `2026-07-17`:
  - serial auto-loop execution is authorized for Principal Alignment Task 0-12
  - task-scoped private reads, Hyperliquid BTC post-only submit/cancel, reduce-only flatten, detached remote jobs, artifact pullback, and conflicting-service isolation are authorized
  - routine authorization prompts are not required while the exact task envelope remains at or below `0.005 BTC` per order, `0.01 BTC` aggregate position delta, `1 USDC` max loss, `2` submissions and `1800s` per bounded window
  - exact task envelopes, QA gates, account/service isolation, reconciliation, checksum, and fail-closed stops remain mandatory
  - source: `docs/superpowers/plans/2026-07-17-principal-alignment-p0-p3.md`
- `0717T006 / LIVE-EVIDENCE-INTEGRITY-REPAIR-PLAN` QA is `已通过`.
- QA report: `.workflow/reports/0717T006-qa.md`.
- The repair route is now authorized to proceed one formal task at a time:
  - window/attempt identity
  - idempotent attempt-bounded fill attribution
  - watcher termination/timeout
  - terminal artifact sealing
  - integrated offline acceptance
- Completed repair phases:
  - `0717T007`: Phase 1 window/attempt identity
  - `0717T008`: Phase 2 idempotent attempt-bounded fill attribution
- Controller scope adjustment on `2026-07-17`:
  - `runtime_risk_envelope_not_enforced` is accepted for the current tiny-live optimization stage.
  - Do not add max-loss/max-position controls in the current repair route.
  - Revisit only when exposure, duration, concurrency, or promotion scope expands.
- Accepted plan task:
  - `0717T006 / LIVE-EVIDENCE-INTEGRITY-REPAIR-PLAN`
  - status: `已通过`
  - plan: `docs/cross_exchange_live_evidence_integrity_repair_plan.md`
  - task: `.workflow/tasks/0717T006.md`
  - business report: `.workflow/reports/0717T006-business.md`
  - remaining repair scope: window/attempt identity, idempotent fill attribution, watcher termination/timeout, and terminal artifact sealing.
  - implementation proceeds as separate sequential tasks after this accepted plan.
- `0717T005` remote-update code review is `未通过`.
- Review range: `d4af427..c9547f2`.
- The later controller scope adjustment supersedes the runtime-risk finding as an immediate repair blocker.
- Focused tests pass, but the current repair route still includes:
  - signal abort not terminating the active watcher process
  - stale failed-run sha256 manifests
  - non-idempotent, non-attempt-bounded fallback fill attribution
  - hardcoded multi-window artifact identity
- Task file: `.workflow/tasks/0717T005.md`
- QA report: `.workflow/reports/0717T005-qa.md`
- Required route: execute the `0717T006` plan as sequential offline repair tasks, add focused regression tests, and obtain QA acceptance before any new live authorization.
- `0717T004` remains `已通过` only for the WTIOIL root-cause audit.
- `0717T004` latest QA is `已通过` for WTIOIL root-cause audit.
- It closes the WTIOIL symbol-mismatch concern:
  - WTIOIL `Open Short 1.14 @ 78.51` came from awsserver1 `xemm.service`, not the 0717T002 Python runner.
  - XEMM config is `CLUSDT -> xyz:CL` with about `90 USDC` order notional.
  - XEMM journal shows Binance `BUY 1.14 @ 78.47`, followed by Hyperliquid `SELL 1.14 xyz:CL`, filled `1.14 @ 78.51`.
- Current live blocker now includes concurrent-service isolation:
  - `xemm.service` was active during evidence collection and can contaminate account-level trade history and rate-limit/fill-source behavior.
  - future live evidence must fail closed if unrelated trading services are active on the same host/account, unless explicitly authorized and isolated.
- `0717T003` latest QA is `已通过` for the safety audit.
- It supersedes the fill interpretation of `0717T002`: the SSM-first collection and open-orders proof remain useful, but `trade_logs/0717trade_history.csv` invalidates the previous 0717T002 no-fill conclusion.
- Current blocker is now account provenance / fill-source identity mismatch:
  - 0717 trade history contains BTC rows that match 0717T002 live intents by timestamp, symbol, side, price, and size.
  - the awsserver1 env account read-only API view returns zero fills/recent fills/open orders/positions for the same interval.
  - the same 0717 trade history contains a WTIOIL short inside window 03 if interpreted as Shanghai local time, but current artifacts/code do not attribute it to the repo runner.
- Latest accepted infra route:
  - `route_to_ssm_first_live_collection_orchestrator_for_future_authorized_live_runs`
- Current evidence blocker:
  - `0717T003 / account_provenance_fill_source_identity_mismatch`
- Controller update:
  - live rerun authorization has now been supplied for `awsserver1` under the 0715T001 envelope, using existing awsserver1 env file `/home/admin/XEMM_rust_latest/.env`, Hyperliquid `BTC`, post-only `Alo`, `3` x `1800s` windows, max size `0.005 BTC`, max submissions `2` per window, max position delta `0.01 BTC`, max loss `1 USDC`, source `cross-exchange/a5431d8b24da7d77671148d316f789b0b25cf3f8`, and real order submit/cancel allowed under that envelope.
- Historical live-evidence follow-up (not the current repair queue):
  - create a repair task for account provenance guard and multi-window artifact id/window id correctness.
  - run only offline regression against 0717T002 artifacts and 0717 trade history before any further live authorization.
  - any future live evidence rerun must launch through `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py` and must include the new account provenance guard.
  - do not create T004-kernel public shadow until account/fill-source identity and role/source-path evidence acquisition are QA accepted or explicitly downgraded by the controller.
  - do not start fee/PnL calibration until future accepted evidence includes liquidity role and exchange-native fill lifecycle attribution.
- Current accepted safety audit:
  - `0717T003 / LIVE-SYMBOL-MISMATCH-WTIOIL-AUDIT`
  - status: `已通过`
  - task file: `.workflow/tasks/0717T003.md`
  - business report: `.workflow/reports/0717T003-business.md`
  - QA report: `.workflow/reports/0717T003-qa.md`
  - result: 0717 BTC trade-history rows strongly match 0717T002 runner intents; WTIOIL short is not attributable to this repo runner from current evidence; awsserver1 env account API view is inconsistent with the downloaded trade history; future live tests are blocked pending account provenance / fill source identity repair.
- Current blocked live evidence task:
  - `0717T002 / T011-SSM-FIRST-CONTROLLED-ROLE-EVIDENCE-RERUN`
  - status: `阻塞`
  - task file: `.workflow/tasks/0717T002.md`
  - business report: `.workflow/reports/0717T002-business.md`
  - QA report: `.workflow/reports/0717T002-qa.md`
  - remote root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z`
  - local root: `local_live_analysis/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z/`
  - result: SSM-first orchestration completed 3 windows, JSON/CSV parse passed, sha manifest `241/241` passed, final open orders `0`.
  - order intents: `4` total; window 01/02 each had one resting buy intent, window 03 had two post-only immediate-match rejects.
  - original artifact fills: `0`; liquidity-role evidence rows: `0`.
  - superseding interpretation: external 0717 trade history has two BTC rows that match window 01/02 intents, so no-fill cannot be claimed.
  - final route: `route_to_controlled_evidence_rerun_or_explicit_downgrade_no_fill_role_evidence`.
  - not supported: maker fill count, fee/PnL calibration, fill-rate calibration, maker viability, T012, promotion, or final MVP pass.
- Current accepted infra task:
  - `0717T001 / LIVE-SSH-RESILIENCE-REMOTE-JOB-ORCHESTRATOR`
  - status: `已通过`
  - task file: `.workflow/tasks/0717T001.md`
  - business report: `.workflow/reports/0717T001-business.md`
  - QA report: `.workflow/reports/0717T001-qa.md`
  - code: `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
  - tests: `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
  - docs: `docs/cross_exchange_live_collection_resilience.md`
  - result: adds a remote live job contract with lock, status, heartbeat, per-window proof, abort manifest, complete manifest, and sha256 manifest.
  - not included: live execution, strategy parameter change, S3 artifact upload, permanent systemd service, AWS security hardening, T004 unlock, or fee/PnL calibration.
- Current blocked task:
  - `0716T006 / T011-CONTROLLED-EVIDENCE-ACQUISITION-WITH-LIQUIDITY-ROLE-CONTRACT`
  - status: `阻塞`
  - task file: `.workflow/tasks/0716T006.md`
  - business report: `.workflow/reports/0716T006-business.md`
  - live rerun business report: `.workflow/reports/0716T006-live-rerun-business.md`
  - output: `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006/`
  - prior result: no complete non-live artifact source or live authorization envelope was supplied; no live execution or endpoint touch occurred.
  - rerun authorization: supplied by controller for the 0715T001 envelope on `awsserver1`.
  - rerun result: awsserver1 connectivity was recovered through SSH/EIC plus SSM role/profile/agent repair; complete artifact root was pulled to `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z_full/`; all three windows parsed and showed final open orders `0`, but all had zero fill rows and zero role-evidence rows.
  - not authorized: live retry without envelope, quote-policy change, threshold/quote-envelope/order-size/max-submission change, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.
- Recent accepted task:
  - `0716T005 / T011-FILL-SOURCE-LIQUIDITY-ROLE-CONTROLLED-EVIDENCE-PREFLIGHT`
  - status: `已通过`
  - task file: `.workflow/tasks/0716T005.md`
  - business report: `.workflow/reports/0716T005-business.md`
  - QA report: `.workflow/reports/0716T005-qa.md`
  - output: `local_live_analysis/cross_exchange_fill_source_liquidity_role_preflight_0716T005/`
  - result: defines the future controlled evidence artifact contract for fill source and maker/taker role capture, including role taxonomy, required artifacts, source-path gates, fee/PnL blocking gates, and a future task template.
  - final route: `route_to_separately_authorized_controlled_evidence_acquisition_with_liquidity_role_contract`
  - not authorized: live retry, quote-policy change, threshold/quote-envelope/order-size/max-submission change, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.
- Current pending task:
  - `0715T001` business execution is complete and status is `待验收`.
  - It replaces the missed `0714T006` gate with a UTC-only scheduled gate.
  - target UTC: `2026-07-15T13:15:00Z`
  - equivalent New York time: `2026-07-15 09:15 EDT`
  - equivalent Shanghai time: `2026-07-15 21:15 CST`
  - automation id: `0715t001-utc-live-test-gate`
  - actual windows: `13:21:13-13:28:04`, `13:30:20-13:31:19`, `13:32:17-13:35:38 UTC`
  - corrected result: `5` total live submissions; artifact fill ledger `0`, but user-provided exchange trade export matches `2` intents / `0.01 BTC` fills. Treat the original no-fill classification as invalid.
- Latest business execution result:
  - `0715T001` collected controlled live evidence under the same conservative envelope.
  - window 1 and 2 reached submitted/resting lifecycle with interval public-flow evidence, but external trade-history reconciliation indicates both filled.
  - window 3 submitted but did not reach resting (`error,error`), so resting interval evidence is correctly empty/not applicable.
  - local artifact validation passed internally, but external trade-history reconciliation failed the no-fill conclusion. No fill probability, maker fill count, fee/PnL, maker viability, T012, promotion, or final MVP claim is supported.
- Required next task:
  - `0716T001 / T011-LIVE-FILL-ATTRIBUTION-REPAIR`
  - status: `待验收`
  - task file: `.workflow/tasks/0716T001.md`
  - business report: `.workflow/reports/0716T001-business.md`
  - scope: offline repair only; preserve raw user fill payloads, reconcile ambiguous cancel responses, and produce attempt-keyed fill attribution for 0715T001.
  - not allowed: live retry, threshold/quote-envelope/size/max-submission changes, quote policy design, fee/PnL calibration.
  - result: repair implemented locally; focused attribution tests `5 passed`, event-driven watcher regression `48 passed`.
  - QA status: `已通过`
- Current pending task:
  - `0716T002 / T011-OFFLINE-QUOTE-FILL-ANALYSIS-WITH-CORRECTED-0715T001-FILLS`
  - status: `已通过`
  - output: `local_live_analysis/cross_exchange_quote_fill_analysis_0716T002/`
  - result: submitted attempts `5`, post-only rejects `3`, corrected filled resting attempts `2`.
  - final route: `route_to_quote_policy_design_prework_and_liquidity_role_evidence_repair`.
  - still not authorized: live retry, threshold/quote-envelope change, fee/PnL calibration, maker viability, T012, promotion, final MVP pass.
- `0716T003 / T011-LIQUIDITY-ROLE-EVIDENCE-REPAIR`
  - status: `已通过`
  - output: `local_live_analysis/cross_exchange_liquidity_role_evidence_repair_0716T003/`
  - result: adds explicit future `fill_liquidity_role_evidence.csv` and fee/PnL role gate.
  - verification: py_compile passed, focused attribution + watcher regression `54 passed`.
  - QA: `已通过`
- `0716T004 / T011-QUOTE-POLICY-DESIGN-PREWORK`
  - status: `已通过`
  - output: `local_live_analysis/cross_exchange_quote_policy_design_prework_0716T004/`
  - result: candidate prework for adverse-flow suppression, post-only reject drift precheck, and fill source-path capture.
  - final route: `route_to_controlled_evidence_design_with_liquidity_role_and_quote_policy_preflight`.
  - not authorized: strategy implementation, parameter change, live retry, fee/PnL calibration.
  - QA: `已通过`
- Previous business execution result:
  - `0714T006` is `阻塞`.
  - The scheduled automation fired at `2026-07-14T21:15:03Z`, which is `2026-07-14 17:15 EDT` / `2026-07-15 05:15 CST`, not the authorized `2026-07-14 09:15 EDT` / `2026-07-14 21:15 CST` pre-open gate.
  - No live window ran, no `awsserver1` live runner started, no credential/private endpoint was touched, and no order was submitted.
- `0715T001 / T011-UTC-SCHEDULED-US-OPEN-CONTROLLED-LIVE-EVIDENCE-WITH-INTERVAL-COVERAGE-REPAIR`
  - status is `待验收`.
  - task file: `.workflow/tasks/0715T001.md`
  - business report: `.workflow/reports/0715T001-business.md`
  - scheduling source: UTC only.
  - scheduled start: `2026-07-15T13:15:00Z`.
  - live envelope: three sequential `1800s` windows, Hyperliquid `BTC`, post-only `Alo`, fast `l2Book`, max size `0.005 BTC`, max submissions `2`, quote hold `3s`, wait `10s`.
  - hard boundaries: no threshold change, quote-envelope change, size/submission expansion, fill-seeking placement, quote policy design, or fee/PnL calibration.
  - output package: `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/`
- Obsolete scheduled live gate:
  - automation id `0714t006-live-test-at-us-open-preflight`
  - should be deleted after recording the missed gate.
  - must not be reused for live execution.
- `0714T006 / T011-SCHEDULED-US-OPEN-CONTROLLED-LIVE-EVIDENCE-WITH-INTERVAL-COVERAGE-REPAIR`
  - status is `阻塞`.
  - task file: `.workflow/tasks/0714T006.md`
  - scheduled start: 2026-07-14 21:15 CST / 09:15 EDT.
  - live envelope: three sequential `1800s` windows, Hyperliquid `BTC`, post-only `Alo`, fast `l2Book`, max size `0.005 BTC`, max submissions `2`, quote hold `3s`, wait `10s`.
  - hard boundaries: no threshold change, quote-envelope change, size/submission expansion, fill-seeking placement, quote policy design, or fee/PnL calibration.
  - blocker: missed scheduled pre-open authorization window; live evidence was not collected.
- `0714T005 / T011-PUBLIC-FLOW-INTERVAL-COVERAGE-CAPTURE-REPAIR`
  - QA status is `已通过`.
  - task file: `.workflow/tasks/0714T005.md`
  - business report: `.workflow/reports/0714T005-business.md`
  - QA report: `.workflow/reports/0714T005-qa.md`
  - code: `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - output package: `local_live_analysis/cross_exchange_public_flow_interval_coverage_capture_repair_0714T005/`
  - repair: adds coverage proof/diagnostic fields and a post-cancel public-stream settling step before future live artifact finalization.
  - mock output proves complete-zero, complete-with-trades, and incomplete-with-diagnostic coverage states.
  - boundaries: no live retry, no thresholds/quote envelope/size/max submissions/strategy behavior change, no fill probability, no fee/PnL.
- `0714T004 / T011-OFFLINE-QUOTE-FILL-EVIDENCE-RERUN-WITH-V2-RESTING-INTERVAL-ARTIFACTS`
  - QA status is `已通过`.
  - task file: `.workflow/tasks/0714T004.md`
  - business report: `.workflow/reports/0714T004-business.md`
  - QA report: `.workflow/reports/0714T004-qa.md`
  - runner: `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py`
  - output package: `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0714T004/`
  - source input: accepted local `0714T003` package `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/`
  - output summary: `71` attempt rows, `1` submitted/resting/no-fill row, `1` coverage evidence row.
  - accepted route: `route_to_public_flow_artifact_repair`.
  - interpretation: v2 live artifact confirms a real resting/no-fill lifecycle, but `public_stream_coverage.csv` is `coverage_not_proven_complete`, so zero captured interval public-trade rows remain `artifact_gap_not_no_exchange_trades`.
  - unsupported: fill probability, quote policy design, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, final MVP pass, or parameter expansion.
- `0714T003 / T011-CONTROLLED-SAME-ENVELOPE-LIVE-EVIDENCE-WITH-V2-RESTING-INTERVAL-CAPTURE`
  - QA status is `已通过`.
  - task file: `.workflow/tasks/0714T003.md`
  - business report: `.workflow/reports/0714T003-business.md`
  - QA report: `.workflow/reports/0714T003-qa.md`
  - remote root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
  - local root: `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
  - windows completed: `3`
  - window 1/2: `no_submit_fail_closed`
  - window 3: `submitted_resting_no_fill`, `buy 0.00036 BTC @ 62650.0`, fills `0`, final open-orders `0`
  - v2 artifacts exist for window 3, including `public_stream_coverage.csv`.
  - zero-row interpretation: `artifact_gap_not_no_exchange_trades`; coverage status `coverage_not_proven_complete`.
  - accepted route: create and execute `0714T004` offline quote/fill evidence rerun, but no fill probability, quote policy design, fee/PnL, maker viability, T012, promotion, or final MVP claim is supported yet.
- `0714T002 / T011-RESTING-INTERVAL-CAPTURE-CONTRACT-REPAIR`
  - QA status is `已通过`.
  - task file: `.workflow/tasks/0714T002.md`
  - business report: `.workflow/reports/0714T002-business.md`
  - QA report: `.workflow/reports/0714T002-qa.md`
  - code: `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - output package: `local_live_analysis/cross_exchange_resting_interval_capture_contract_repair_0714T002/`
  - schema version: `cross_exchange_resting_interval_public_flow_capture_v2`
  - contract version: `cross_exchange_resting_interval_public_flow_capture_contract_v2`
  - key repair: adds stable `attempt_key`, lifecycle interval source/status fields, public-trade quote relation flags, resting-start L2/depth source/status fields, and `public_stream_coverage.csv`.
  - zero-row semantics: generated mock package includes both `zero_public_trades_observed_with_complete_interval_coverage` and `artifact_gap_not_no_exchange_trades`.
  - verification: focused watcher pytest `45 passed`, py_compile/help/artifact parse/deterministic rerun/`git diff --check` passed.
  - next route: create separately authorized controlled same-envelope live evidence task; do not change thresholds, quote envelope, order size, max submissions, or quote policy before accepted live evidence and analysis.
- `0714T001 / T011-PUBLIC-FLOW-INTERVAL-ARTIFACT-REPAIR-DESIGN-V2`
  - QA status is `已通过`.
  - task file: `.workflow/tasks/0714T001.md`
  - business report: `.workflow/reports/0714T001-business.md`
  - QA report: `.workflow/reports/0714T001-qa.md`
  - runner: `examples/hyperliquid/cross_exchange_public_flow_interval_artifact_repair_design_0714T001.py`
  - output package: `local_live_analysis/cross_exchange_public_flow_interval_artifact_repair_design_0714T001/`
  - source input: accepted `0713T003` package `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0713T003/`.
  - output summary: `5` artifact gap rows, `4` instrumentation design rows, `5` acceptance gate rows.
  - final business route: `route_to_resting_interval_capture_contract_repair`.
  - interpretation: the `0` matching attempt-keyed interval public-trade rows remain an artifact observability gap, not proof that no exchange public trades occurred.
  - next task recommendation: implement capture contract / instrumentation repair before any controlled live evidence rerun.
  - unsupported: live retry, threshold/quote-envelope changes, quote policy design, fill probability, queue priority, fee/PnL, maker viability, T012, promotion, or final MVP claim.
- `0713T003 / T011-QUOTE-FILL-PROBABILITY-EVIDENCE-RERUN-WITH-RESTING-INTERVAL-PUBLIC-FLOW`
  - QA status is `已通过`.
  - business report: `.workflow/reports/0713T003-business.md`
  - QA report: `.workflow/reports/0713T003-qa.md`
  - runner: `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0713T003.py`
  - output package: `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0713T003/`
  - source input: accepted local package `local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`.
  - final business route: `route_to_public_flow_artifact_repair`.
  - evidence summary: `18` quote evaluation rows, `1` submitted/resting/no-fill row, `0` matching attempt-keyed interval public-trade rows, proxy-only lifecycle/depth, and `3.125993s` short-horizon censoring.
  - pre-QA repair `c31e6b0` fixes remote provenance, skipped/no-order attempt-id semantics, and repo-relative output paths.
  - amdserver QA reproduction should use `/home/molly/anaconda3/envs/nt-backtest/bin/python`.
  - full `examples/hyperliquid` pytest is currently blocked in that env by missing `requests` and `numba`; focused 0713T003 tests and quote/fill focused regression pass.
  - interpretation: sufficient for the artifact-repair route, not sufficient for quote policy design, fee/inventory/PnL calibration, fill probability, queue priority, maker viability, T012, promotion, or final MVP pass.
  - boundaries held: no live-submit, remote/AWS execution, credential reads, private/account/order/cancel endpoints, new market-data collection, threshold/quote-envelope/order-size/max-submission/strategy changes, live retry, or maker/PnL/T012/MVP claims.
  - Current route: stop this auto-loop and create a separate public-flow interval artifact repair/design task if continuing.
- `0713T002 / T011-CONTROLLED-SAME-ENVELOPE-LIVE-EVIDENCE-WITH-RESTING-INTERVAL-PUBLIC-FLOW`
  - QA status is `已通过`.
  - remote collection ran on `awsserver1` at `/home/admin/hftbacktest-cross-exchange`, commit `a69d7e5361faa537c22ab6ee0d2b53918f76e5ce`.
  - artifact package was pulled back with `scp` from `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/` to `local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`.
  - one controlled same-envelope live window ran: Hyperliquid `BTC`, post-only `Alo`, fast `l2Book`, max size `0.005 BTC`, max submissions `2`, no threshold/quote-envelope/size/max-submission/strategy change.
  - Window 1 classification is `submitted_resting_no_fill`; order status `resting`, fill count `0`, shutdown proof `pass`, runner final open-orders `0`, independent final open-orders `0`.
  - new resting-interval artifacts exist: lifecycle rows `1`, interval public-trade rows `0`, resting-start L2/depth rows `1`, depletion rows `1`.
  - pre-QA repair commit `762e335` fixed future artifact task-id propagation; current pulled-back raw manifests that still show legacy `task_id=0623T007` are covered by `source_attribution_overlay.json` and must be interpreted as writer metadata only.
  - `captured_public_trade_row_count=0` means no matching attempt-keyed interval public-trade rows were captured in the proxy interval; it does not prove that no exchange public trades occurred.
  - pullback validation passed: remote/local raw files `70/70`, sha256 reconciliation `70/70`, JSON parse errors `0/32`, CSV parse errors `0/35`, boundary status `pass`.
  - Current route: Step 4 may be created as a separate formal task. Do not change thresholds, quote envelope, order size, max submissions, or run another live retry before Step 4 analysis.
- `0712T001` QA is also `已通过`; it accepts the offline public-flow interval artifact repair/design result and confirms current accepted artifacts still cannot reconstruct actual resting-interval public trades or actual depletion/trade-through.
- `0713T001 / T011-RESTING-INTERVAL-PUBLIC-FLOW-CAPTURE-INSTRUMENTATION`
  - QA is `已通过`.
  - scope was implementation and offline/mock verification only; no live, remote/AWS, credential, private/account/order/cancel endpoint, market-data collection, threshold, quote-envelope, order-size, max-submission, or strategy behavior change.
  - code: `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - output: `local_live_analysis/cross_exchange_resting_interval_public_flow_capture_instrumentation_0713T001/`
  - accepted route: stop at Step 3 live authorization gate.
- Current evidence interpretation:
  - The accepted set contains `5` attempt rows: one prior `0708T001` QA reference plus four `0709T001` live artifact order attempts.
  - It has `2` post-only rejects that are not fill-probability samples.
  - It has `3` resting/no-fill rows, but two are short-hold censored and the prior reference lacks local quote/fill public-flow artifacts.
  - Current artifacts expose decision-time rolling public-flow/depletion proxies, not full actual resting-interval trade-through/depletion reconstruction.
  - Therefore no fill probability, queue priority, fee/rebate, realized PnL, stable PnL, maker viability, T012 readiness, promotion, or final MVP pass is supported.
- Artifact naming clarification:
  - `accepted_source_row_count=4`
  - `prior_reference_count=1`
  - `live_artifact_attempt_count=4`
- `0712T001` QA is complete:
  - task: `.workflow/tasks/0712T001.md`
  - report: `.workflow/reports/0712T001-business.md`
  - QA report: `.workflow/reports/0712T001-qa.md`
  - output: `local_live_analysis/cross_exchange_public_flow_interval_artifact_repair_0712T001/`
  - code: `examples/hyperliquid/cross_exchange_public_flow_interval_artifact_repair.py`
  - accepted resting/no-fill attempts covered: `3`
  - actual resting-interval public trades are `not_reconstructable_from_current_artifact` for all `3` rows.
  - actual resting-interval depletion/trade-through is `not_reconstructable_from_current_artifact` for all `3` rows.
  - final business route: `route_to_controlled_same_envelope_live_evidence_with_resting_interval_public_flow_artifacts`
  - review-fix commit `65f2461` tightened future route semantics so partial interval trades cannot be misread as offline sufficient unless exact lifecycle/depth evidence is also present.
- Durable route:
  - `0712T001` and `0713T001` are accepted. The auto-loop must stop at the Step 3 live authorization gate until a later formal task records the exact live envelope, `awsserver1` topology, local pullback path, and explicit controller authorization.
- Still not authorized:
  - live retry, T012, live expansion, threshold changes, quote-envelope changes, order-size/max-submission expansion, fee/PnL calibration, stable PnL, maker viability, promotion, or final MVP pass.

- `0709T002` QA is `已通过`. It accepts offline batch same-window replay acceptance over four rows: `0708T001` prior accepted replay reference plus `0709T001` windows 1-3.
- Output package: `local_live_analysis/cross_exchange_t011_batch_same_window_replay_acceptance_0709T002/`.
- Final recommendation: `batch_same_window_replay_acceptance_passed`; all four rows pass market-view, decision-path, lifecycle, economics, optimism, boundary, and overall acceptance.
- Next formal task may be `0709T003 / T011-MULTI-WINDOW-ROBUSTNESS-SYNTHESIS`, offline only. Do not claim PnL, maker viability, T012, promotion, or live expansion.

- `0709T001` QA is `已通过`. It accepts T011 controlled multi-window live evidence acquisition under the explicit T001 envelope.
- Local artifact package: `local_live_analysis/cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z/`.
- Three sequential live windows completed:
  - window 1: `submitted_rejected`, `2` post-only rejects, no fill, final open-orders `0`, independent final open-orders `0`
  - window 2: `submitted_resting_no_fill`, `1` resting order, no fill, final open-orders `0`, independent final open-orders `0`
  - window 3: `submitted_resting_no_fill`, `1` resting order, no fill, final open-orders `0`, independent final open-orders `0`
- No live window exceeded `0.005 BTC` max size or `2` max submissions, all used post-only `Alo`, and all had shutdown proof `pass`.
- Next formal task may be `0709T002 / T011-BATCH-SAME-WINDOW-REPLAY-ACCEPTANCE`, offline only. Do not create T003 before T002 QA accepts.

- `0708T002` QA is `已通过`.
- `0708T002` accepts the single-window `0625T010` same-window replay gate over the accepted `0708T001` fast-L2 live lifecycle.
- Implemented runner: `examples/hyperliquid/cross_exchange_t010_same_window_replay_acceptance.py`.
- Accepted package: `local_live_analysis/cross_exchange_t010_same_window_replay_acceptance_0708T002/`.
- Final recommendation: `same_window_replay_acceptance_passed`.
- Acceptance results:
  - market-view acceptance `pass`, checks `8/8`
  - decision-path acceptance `pass`, checks `10/10`
  - lifecycle acceptance `pass`, checks `12/12`
  - economics/no-fill attribution `pass`, checks `6/6`
  - optimism checks `pass`, checks `8/8`
  - boundary status `pass`
- The replay acceptance conservatively represents:
  - fast L2 source path
  - post-open-orders public-state evidence
  - trigger/guard/edge pass
  - `buy 0.002 BTC @ 63889.0`, post-only `Alo`
  - one real source-artifact order submission
  - `resting` response
  - tracked cancel / shutdown proof
  - no fill
  - final open-orders `0`
  - independent final open-orders `0`
- No-optimism constraints held:
  - no synthetic fill
  - no fill probability / fill horizon
  - no fee/rebate / realized PnL
  - no zero-latency assumption
  - no reject-rate generalization
  - no maker viability claim
- Current route:
  - single-window `0625T010` same-window replay/live acceptance is complete for the `0708T001` no-fill lifecycle.
  - `0625T011`, `0625T012`, stable PnL, maker viability, promotion, and final MVP pass remain blocked.
  - Next task should be a separate `0625T011`-style multi-window evidence/robustness plan or another controlled live evidence acquisition task with explicit envelope.

- `0708T001` QA is `已通过`.
- `0708T001` repaired the T010 live watcher fast Hyperliquid `l2Book` binding and executed a controlled live evidence rerun.
- Code / dispatch commit: `34a77ea / Bind fast Hyperliquid l2Book to T010 watcher`.
- Accepted package: `local_live_analysis/cross_exchange_t010_fast_l2book_controlled_live_evidence_0708T001_20260707T160830Z/`.
- Remote execution:
  - host `awsserver1`
  - repo `/home/admin/hftbacktest-cross-exchange`
  - commit `34a77eaa490daf26584040fbda5522afbf8b6710`
- Fast L2 evidence:
  - `hyperliquid_l2book_fast=true`
  - l2Book messages `799` over `436.021847s`
  - reconnect count `0`
- Trigger / guard evidence:
  - current candidates `1755`
  - anti-drift pass/block `50/16`
  - edge gate pass/block `1/17`
  - trigger found `true`
  - trigger count `1`
  - event-driven guard status `pass`
- Latency evidence:
  - `open_orders_elapsed` median `0.032966s`
  - `open_orders_end_to_public_state` median `0.299699s`
  - `open_orders_end_to_reprice` median `0.299784s`
  - the previous `post_open_orders_handoff_latency_exceeded` blocker did not recur.
- Execution evidence:
  - live submissions `1`
  - real order endpoint called `true`
  - order status type `resting`
  - submitted order `buy 0.002 BTC @ 63889.0`, post-only `Alo`
  - real cancel endpoint called `true`
  - shutdown proof status `pass`
  - fill count `0`
  - final open-orders count `0`
  - independent final open-orders count `0`
- Full `0625T010` remains pending, but is now unblocked for a same-window replay acceptance task over the accepted `0708T001` live artifact.
- Next task should be a narrow `0625T010` same-window replay acceptance over `0708T001`, verifying market view, decision path, submit/resting/cancel/no-fill lifecycle, fee/rebate/no-fill attribution, inventory state, and non-optimistic replay behavior.
- Do not change thresholds, quote envelope, order size, or max submissions before that replay acceptance.

- `0707T007` QA is `已通过`.
- `0707T007` is the handoff-repaired controlled live evidence rerun after accepted `0707T006`.
- It is accepted as a controlled live evidence rerun that safely failed closed before order submission, not as full `0625T010`.
- Authorization / dispatch node: `fc70a55 / Authorize T010 handoff repaired live evidence`.
- Business evidence commit: `69c779b / Record T010 handoff repaired live evidence`.
- Accepted package: `local_live_analysis/cross_exchange_t010_handoff_repaired_controlled_live_evidence_0707T007_20260707T150429Z/`.
- Remote execution:
  - host `awsserver1`
  - repo `/home/admin/hftbacktest-cross-exchange`
  - commit `fc70a55d4af6dcf4168dc8a38ab40c692aac38c9`
- Public stream health:
  - watcher elapsed `1800.078229s`
  - l2Book messages `335`
  - trades messages `4379`
  - trade events `15028`
  - reconnect count `0`
- Trigger / guard evidence:
  - current candidates `4496`
  - anti-drift pass/block `30/311`
  - trigger found `true`
  - trigger count `1`
- Repaired path evidence:
  - `edge_gate_live_compatible_source_available=true`
  - `edge_gate_source_status=decision_time_public_fair_mid_provider`
  - post-open-orders public-state pass/block `15/0`
  - handoff schema active with `handoff_phase=post_open_orders_inline_reprice`
  - guard reason `post_open_orders_handoff_latency_exceeded`
  - trigger/current reprice fields are separated in artifacts.
- Execution evidence:
  - live submissions `0`
  - real order endpoint called `false`
  - real cancel endpoint called `false`
  - fill count `0`
  - final open-orders count `0`
  - independent final open-orders count `0`
- Latency evidence:
  - `open_orders_elapsed` median `0.018696s`
  - `open_orders_end_to_public_state` median `5.046153s`
  - the current blocker is post-open-orders public L2 resync latency exceeding the `1.0s` immediate guard age budget.
- Full `0625T010` remains blocked.
- Next task should narrowly diagnose/repair pre-submit latency budget around post-open-orders public-state resync, without changing thresholds, quote envelope, size, or max submissions.

- `0707T006` QA is `已通过`.
- `0707T006` repaired the inline reprice handoff contract/instrumentation after `0707T005`.
- Pre-task sync aligned local, GitHub `origin/cross-exchange`, and `amdserver:~/workspace/hftbacktest` to `af3bb1ac38783eb18e04da1369dc95eafd3f5f95`.
- Code commit: `b0a1814 / Repair T010 inline reprice handoff contract`.
- The repair:
  - preserves trigger candidate audit fields separately from current inline reprice decision fields.
  - passes the original trigger candidate into the post-open-orders immediate guard.
  - emits `post_open_orders_handoff_latency_exceeded` when private open-orders plus public L2 resync makes the original trigger too old before submit.
  - keeps current reprice failure visible in `current_reprice_*` fields.
- Verification passed: focused pytest `42 passed`, py_compile, CLI help, artifact JSON/CSV parse, and `git diff --check`.
- No anti-drift threshold, touch-stability threshold, edge threshold, quote envelope, size, max-submission, or live-submit behavior was changed.
- Full `0625T010` remains blocked.
- Next task should be a separately authorized controlled live evidence rerun under the same conservative envelope, using the repaired handoff schema.

- `0707T005` QA is `已通过`.
- `0707T005` diagnosed inline reprice candidate handoff drift after the accepted `0707T004` repaired controlled live evidence attempt.
- Source artifact: `local_live_analysis/cross_exchange_t010_repaired_controlled_live_evidence_0707T004_20260707T060126Z/event_driven_edge_gate_live/`.
- Generated diagnosis package: `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_diagnosis_0707T005/`.
- Evidence:
  - post-open-orders public-state rows `8/8` passed and observed L2 after `open_orders_end_ns`.
  - inline reprice attempts `8/8` failed closed.
  - immediate pre-submit guard rows `8/8` failed closed.
  - candidate age at guard was `4.899s` to `5.577s`, median `5.207s`, versus a `1.0s` guard max.
  - source event to post-open-orders L2 delta was `4911ms` to `5254ms`, median `5058ms`.
  - submit-ready intent survived reprice in `1/8` rows; `7/8` rows lost intent fields after current-candidate recomputation.
- Root cause classification:
  - primary: `post_open_orders_handoff_latency_exceeds_immediate_age_guard`.
  - secondary: `inline_reprice_recomputed_candidate_often_no_longer_submit_ready_so_intent_fields_disappear`.
- Next task should be a narrow repair: `T010-INLINE-REPRICE-HANDOFF-CONTRACT-REPAIR`.
- Do not change anti-drift thresholds, touch-stability thresholds, quote envelope, size, or max submissions in that repair.
- Full `0625T010` remains blocked.

- `0707T004` QA is `已通过`.
- `0707T004` is the repaired controlled live evidence task after accepted `0707T001`/`0707T002`/`0707T003`.
- QA accepted it as a repaired controlled live evidence attempt that safely failed closed before order submission, not as full `0625T010`.
- Accepted package: `local_live_analysis/cross_exchange_t010_repaired_controlled_live_evidence_0707T004_20260707T060126Z/`.
- Remote execution:
  - host `awsserver1`
  - repo `/home/admin/hftbacktest-cross-exchange`
  - commit `17de5295e7d7fe2b46eaeccea9d79058c1f65fdf`
- Public stream health:
  - watcher elapsed `1800.001154s`
  - l2Book messages `336`
  - trades messages `1954`
  - trade events `6397`
  - reconnect count `0`
- Trigger / guard evidence:
  - current candidates `2228`
  - anti-drift pass/block `16/79`
  - trigger found `true`
  - trigger count `1`
- Repaired path evidence:
  - `edge_gate_live_compatible_source_available=true`
  - `edge_gate_source_status=decision_time_public_fair_mid_provider`
  - post-open-orders public-state pass/block `8/0`
- Execution evidence:
  - live submissions `0`
  - real order endpoint called `false`
  - real cancel endpoint called `false`
  - fill count `0`
  - final open-orders count `0`
  - independent final open-orders count `0`
- New observed blocker:
  - `outside_quality_a_b_queue_bands;trigger_candidate_stale_before_order;missing_intent_limit_px;missing_or_nonpositive_intent_size;missing_quality_bucket`
- Full `0625T010` remains blocked.
- Next task should diagnose or repair inline reprice candidate handoff drift after post-open-orders resync. Do not change thresholds, quote envelope, size, or max submissions yet.

- `0707T003` QA is `已通过`.
- `0707T003` is Task C from `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- QA accepted the read-only anti-drift / touch-stability live distribution diagnosis.
- Source artifacts:
  - `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/public_shadow_live_1800s/`
  - `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/event_driven_edge_gate_live/`
- Generated artifact package: `local_live_analysis/cross_exchange_t010_anti_drift_distribution_0707T003/`.
- Funnel facts:
  - `0706T008`: candidates `2480`, fresh-touch allowed `125`, anti-drift pass/block `7/118`, edge pass/block `1/6`, would-submit `1`.
  - `0706T010`: candidates `1872`, fresh-touch allowed `112`, anti-drift pass/block `5/107`, trigger `1`, post-open-orders public-state pass/block `0/5`, live submissions `0`.
- Current `250ms` touch-stability-only retention:
  - `0706T008`: `10/125`
  - `0706T010`: `7/112`
- Recommendation: `separately_authorized_controlled_live_evidence_after_a_b_repairs`.
- Do not create a threshold-change task yet.
- Full `0625T010` remains blocked until a separately authorized controlled live evidence task using accepted A+B repairs is created, executed, and QA accepted.

- `0707T002` QA is `已通过`.
- `0707T002` is Task B from `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- QA accepted the post-open-orders public-state resync guard repair.
- The default post-open-orders L2 wait is now bounded and cadence-aware:
  - base timeout `0.2s`
  - max timeout `6.0s`
  - recent L2 cadence can scale the effective timeout up to the bounded max.
- The pass condition remains strict: L2 local receive timestamp must be after `open_orders_end_ns`.
- Local artifact package: `local_live_analysis/cross_exchange_t010_post_open_orders_resync_0707T002/`.
- Artifact positive case shows `post_open_orders_public_state_pass_count=1`; negative case shows fail-closed `public_source_exhausted_before_post_open_orders_l2`.
- Full `0625T010` remains blocked.
- Next auto-loop action: create `0707T003 / T010-ANTI-DRIFT-TOUCH-STABILITY-LIVE-DISTRIBUTION-DIAGNOSIS`.

- `0707T001` QA is `已通过`.
- `0707T001` is Task A from `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- QA accepted the live-compatible edge/source binding repair.
- The `--event-driven-edge-gate-live` CLI path now binds `BinancePublicBookTickerProvider()` as the default decision-time public fair-mid source.
- Focused test coverage verifies the CLI path supplies a `BinancePublicBookTickerProvider` while preserving `anti_drift_gate=True` and `edge_gate=True`.
- Local artifact package: `local_live_analysis/cross_exchange_t010_live_compatible_edge_source_0707T001/`.
- No-submit/block scenario `insufficient_edge_block` shows `edge_gate_source_status=decision_time_public_fair_mid_provider`, `fair_mid_source_pass_count=1`, `edge_gate_block_count=1`, `live_submissions_count=0`, and `mock_order_call_count=0`.
- Full `0625T010` remains blocked.
- The remaining accepted blocker is `post_open_orders_public_state_timeout`.
- Next auto-loop action: create `0707T002 / T010-POST-OPEN-ORDERS-PUBLIC-STATE-RESYNC-REPAIR`.

- `0706T010` QA is `已通过`.
- `0706T010` is the controlled live evidence task after `0706T008`.
- QA accepted the result as `controlled_live_evidence_blocked_before_submit`, not as full `0625T010`.
- Accepted package: `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/`.
- The run completed `1800.001512s` with public stream health: `337` l2Book messages, `1541` trades messages, reconnect count `0`.
- Trigger/pre-submit evidence: `1872` candidates, anti-drift pass/block `5/107`, trigger found `true`, trigger count `1`.
- Blocking reason: `post_open_orders_public_state_timeout`.
- Additional source blocker: `edge_gate_source_status=missing_live_compatible_source`.
- No live submission occurred: `real_order_endpoint_called=false`, `real_cancel_endpoint_called=false`, fill count `0`, final open-orders count `0`, independent final open-orders count `0`.
- Full `0625T010`, T011, T012, stable PnL, maker viability, promotion, and final MVP pass remain blocked.
- Next recommended task: repair/design live-compatible edge/source binding and post-open-orders public-state resync; do not repeat the same live run blindly.

Previous QA result:

- `0706T008` QA is `已通过`.
- `0706T008` created the three-task auto-loop plan at `docs/cross_exchange_t010_candidate_live_auto_loop_plan.md`.
- QA accepted the route recommendation `route_to_controlled_live_evidence_task`.
- Accepted package: `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`.
- The 1800s public-only/no-submit window produced `2480` candidates, `125` fresh-touch allowed candidates, and `1` shadow would-submit path.
- The would-submit path was event `1369`, side `buy`, quote `63019`, quality `quality_a`, fair-mid source age `43ms`, edge `25.5` ticks, action `would_submit_if_real_order_task_authorized`.
- Boundary held: no credential read, no private/account/order/cancel endpoint, no live client, no submit.
- The next auto-loop action is to create `0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE`.
- Do not create `0706T009` repair task from this result.
- Full `0625T010`, T011, T012, stable PnL, maker viability, promotion, and final MVP pass remain blocked.

Earlier QA result:

- `0706T007` QA is `已通过`.
- `0706T007` is the authorized minimal live evidence acquisition attempt for full `0625T010`.
- User authorization was explicit in-session: `继续，授权live evidence任务`.
- Authorization task file was committed before live execution at commit `cbee781`.
- QA accepted the recommendation `full_t010_live_evidence_blocked_no_order_submitted`.
- Accepted package: `local_live_analysis/cross_exchange_mvp_t010_live_evidence_0706T007/`.
- Remote checkout was synced to `cbee781069456bc0fecdddaa1d7297eaf546e7ce` with dirty count `0`.
- Public flow precheck passed with `10` fresh-touch candidates.
- Fresh-touch allowed candidates were `0`.
- Submitted orders were `0`; `real_order_endpoint_called=false`; `real_cancel_endpoint_called=false`; `fill_count=0`.
- Window final open-orders count and independent final open-orders count were both `0`.
- This proves the authorized envelope failed closed before submit when no eligible same-window fresh-touch candidate existed.
- This does not produce submitted-order lifecycle, fill/economics/PnL, or cross-exchange signal/fair-mid/quote-intent live decision evidence.
- Full `0625T010`, `0625T011`, `0625T012`, stable PnL, maker viability, promotion, and final MVP pass remain blocked.

Earlier QA result:

- `0706T006` QA is `已通过`.
- `0706T006` is the no-submit full T010 live evidence acquisition preflight task corresponding to `0625T010-FULL-PREFLIGHT`.
- QA accepted the recommendation `full_t010_live_evidence_acquisition_blocked_pending_authorization`.
- Accepted package: `local_live_analysis/cross_exchange_mvp_t010_full_preflight_0706T006/`.
- This advances full `0625T010` only to preflight/evidence-acquisition planning.
- Current blocking gates:
  - complete live market view missing
  - complete live decision path missing
  - fill/no-fill economics evidence missing
  - latency and ordering evidence missing
  - additional live authorization missing
- This authorizes no live-submit, repeated-window run, fill-seeking run, closer-to-market placement, quote-envelope change, size change, full `0625T010` execution, `0625T011`, `0625T012`, stable PnL claim, maker viability claim, promotion, or final MVP pass.
- The next MVP-forward executable step requires a new formal live evidence acquisition task and explicit authorization of the exact future live envelope.

Earlier QA result:

- `0706T005` QA is `已通过`.
- `0706T005` is the scoped same-window replay acceptance task corresponding to `0625T010-SCOPED`.
- QA accepted the recommendation `scoped_same_window_replay_acceptance_passed`.
- Accepted package: `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/`.
- Source artifacts are only accepted local `0706T002 / 0625T008` live-submit artifacts and `0706T003 / 0625T009` execution outcome calibration artifacts.
- Supported fact comparison passed `12/12`: order intent, `Alo`, submit path, resting status, primary tracked cancel, shutdown proof, and independent final open-orders count `0`.
- Unsupported fail-closed checks passed `11/11`: submit/ack latency, resting duration, cancel latency, cancel-fill race, fill horizon, fill probability, fee/rebate, inventory transition, realized PnL, stable PnL, and maker viability.
- Optimism checks passed `8/8`: no fill probability, fill horizon, fee/rebate, inventory, realized PnL, reject-rate-zero, zero latency, or maker viability assumption was inferred.
- This completes the narrow scoped replay acceptance gate only.
- This does not unlock full `0625T010`, `0625T011`, `0625T012`, another live-submit, repeated-window run, fill-seeking run, stable PnL claim, maker viability claim, promotion, or final MVP pass.
- The next MVP-forward step requires a human/controller decision: create a separately scoped live evidence acquisition/preflight task with explicit authorization, or stop/continue offline tooling without claiming full MVP progress.

Earlier QA result:

- `0706T004` QA is `已通过`.
- `0706T004` is the roadmap refresh task after `0706T003 / 0625T009`.
- QA accepted the updated route in `docs/cross_exchange_maker_mvp_plan.md` and `docs/cross_exchange_mvp_auto_loop_plan.md`.
- The next automatic task is `0706T005 / 0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance`.
- Scoped T010 may consume only accepted local `0706T002 / 0625T008` and `0706T003 / 0625T009` artifacts.
- Scoped T010 must keep fill, fee/rebate, inventory, realized PnL, stable PnL, and maker viability unsupported/fail-closed.
- Scoped T010 passing does not unlock `0625T011`, final MVP validation, stable PnL claims, maker viability claims, or any additional live-submit.
- Full `0625T010` still requires new complete live evidence and explicit authorization for any live-submit/repeated-window/fill-seeking behavior.
- This authorizes no further live-submit, repeated window, fill-seeking run, integrated strategy run, default-on behavior, promotion, or final MVP pass.

Earlier QA result:

- `0706T003` QA is `已通过`.
- `0706T003` is the execution outcome calibration task corresponding to `0625T009`.
- QA accepted the calibration recommendation `execution_outcome_calibration_ready_for_qa`.
- Accepted calibration package: `local_live_analysis/cross_exchange_mvp_t009_execution_outcome_calibration_0706T003/`.
- Calibration scope is one-order artifact only, consuming `0706T002 / 0625T008` pulled-back artifacts.
- Supported replay/live facts are limited to submit endpoint reachable for the exact envelope, post-only `Alo`, order response `resting`, primary tracked cancel success, and independent final open-orders count `0`.
- Post-only reject is recorded only as `not_observed` in this sample, not as a reject-rate estimate.
- Unsupported domains are explicit: submit/ack latency, resting duration, cancel latency, cancel-fill race, fill horizon, fill probability, fee/rebate, inventory transition, realized PnL, stable PnL, and maker viability.
- This authorizes no further live-submit, repeated window, fill-seeking run, integrated strategy run, default-on behavior, promotion, or final MVP pass.

Earlier QA result:

- `0706T002` QA is `已通过`.
- `0706T002` is the live-submit calibration task corresponding to `0625T008`.
- QA accepted the first live-submit calibration result `hyperliquid_tiny_live_real_order_canary_ready_for_qa`.
- User/controller authorization was explicit in-session: `授权 first live-submit calibration` / `授权开始`.
- Remote execution host was `awsserver1`; remote repo was clean `cross-exchange` at commit `25b444e31`; remote Python was `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python` with Hyperliquid SDK available.
- Pulled-back artifact package: `local_live_analysis/cross_exchange_mvp_t008_live_submit_calibration_0706T002/pulled_back_awsserver1/`.
- One BTC post-only `Alo` buy canary order was submitted: size `0.01 BTC`, limit `62146.0`, notional `621.46 USDC`.
- Order status reached `resting`; tracked cancel/cancel-by-cloid ran; independent final open-orders check returned `final_open_orders_count=0`.
- Artifacts passed JSON parse, SHA256 manifest, redaction scan, and `git diff --check`.
- This authorizes no further live-submit, repeated window, fill-seeking run, integrated strategy run, default-on behavior, promotion, or final MVP pass.

- `0706T001` QA is `已通过`.
- `0706T001` is the valid workflow task for `0625T008-PREFLIGHT Edge-Qualified Tiny-Live Calibration Packet`.
- QA accepted the no-submit preflight packet recommendation `live_submit_blocked_pending_controller_authorization`.
- Accepted preflight packet package: `local_live_analysis/cross_exchange_mvp_t008_preflight_packet_0706T001/`.
- Packet prerequisites preserve accepted `0625T005`, `0625T006`, `0625T007`, archived-invalid `0702T001`, and QA-accepted `0702T002`.
- The active preflight risk envelope is zero-submit: max order count, size, notional, position delta, and max loss are all `0`.
- Authorization gate blocks live-submit because no standing live authorization record exists for the exact `0625T008` envelope.
- `0625T008` live-submit task remains not created, not authorized, and not executed.
- Boundary remains local/offline artifact-only/no network/no AWS/no remote/no credentials/no secret values/no live client/no private/account/order/cancel endpoint/no signing/no nonce/no user stream/no order placement/no cancellation/no live bot/no strategy config change/no production config change/no canary/no promotion.

- `0702T002` QA is `已通过`.
- QA accepted the Binance snapshot rate-limit collector fix: default snapshot limit is `100`, bounded retry/backoff records rate-limit evidence, and missing/invalid `lastUpdateId/bids/asks` snapshot is a hard collection failure after manifest evidence is written.
- `0702T001` QA is `已通过` only as a fail-closed invalid dataset archive. It remains `sample_collection_invalid`, with `t003_creation_unlocked=false`; this does not make the data usable for signal acceptance or replay.
- `0702T001` invalid dataset archive record: `local_live_analysis/archive/0702T001_INVALID_DATASET_ARCHIVE.md` and `local_live_analysis/archive/0702T001_invalid_dataset_archive_manifest.json`.
- `0625T007` QA is `已通过`.
- QA accepted the public market-view replay alignment recommendation `public_market_view_replay_alignment_ready_for_qa`.
- Accepted T007 package: `local_live_analysis/cross_exchange_mvp_public_replay_alignment_0625T007/`.
- T007 matched `10704/10704` decision rows with `0` action mismatches, `0` unexplained mismatches, `0` market-view gate failures, `0` cadence/source-age gate failures, and `0` future joins.
- `0625T006` QA is `已通过`.
- QA accepted the MVP audit/replay contract recommendation `audit_replay_contract_ready_for_qa`.
- Accepted T006 package: `local_live_analysis/cross_exchange_mvp_audit_replay_contract_0625T006/`.
- T006 schema hash is `0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5` over `63` fields.
- Synthetic lifecycle fixtures pass/fail as expected.
- Existing T005/M1/M2 artifacts are classified as partial/fail-closed compatibility references.
- T005 median edge caveat and T003 warning bucket remain visible for replay diagnostics.

Current formal task:

- `0707T004 / T010-REPAIRED-CONTROLLED-LIVE-EVIDENCE` is `已通过` as a repaired controlled live evidence attempt.
- It exercised the accepted A+B repairs in the live path and safely failed closed before order submission.
- `0707T003 / T010-ANTI-DRIFT-TOUCH-STABILITY-LIVE-DISTRIBUTION-DIAGNOSIS` is `已通过` as Task C of the post-`0706T010` repair auto-loop.
- It diagnoses trigger/filter distributions and recommends a separately authorized repaired controlled live evidence task before threshold changes.
- `0707T002 / T010-POST-OPEN-ORDERS-PUBLIC-STATE-RESYNC-REPAIR` is `已通过` as Task B of the post-`0706T010` repair auto-loop.
- It fixes the brittle post-open-orders public-state resync guard while preserving stale/no-proof fail-closed behavior.
- `0707T001 / T010-LIVE-COMPATIBLE-EDGE-SOURCE-BINDING` is `已通过` as Task A of the post-`0706T010` repair auto-loop.
- It fixes the live-compatible edge/source binding blocker in the CLI live edge-gate path.
- `0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE` is `已通过` as a controlled live attempt that failed closed before order submission.
- Full `0625T010` remains blocked.
- Next task should diagnose or repair inline reprice candidate handoff drift after post-open-orders resync. Do not change thresholds, quote envelope, size, or max submissions yet.

Recent completed predecessors:

- `0706T008 / 0625T010-LONG-WINDOW-NOSUBMIT-DIAGNOSIS` is `已通过`.
- Route decision: create `0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE`; skip `0706T009` repair.
- The user has granted conditional low-risk live testing authorization for the planned Task 3 envelope, but Task 3 must still be created as a formal task and remain bounded by the plan.
- `0706T007 / 0625T010-LIVE-EVIDENCE-ACQUISITION` is `已通过` as a fail-closed live evidence attempt.
- The authorized envelope produced no eligible candidate and no submitted order.
- Full `0625T010` remains blocked because required submitted lifecycle/economics/PnL and cross-exchange live decision-path evidence are absent.
- `0706T006 / 0625T010-FULL-PREFLIGHT Live Evidence Acquisition Packet` is `已通过`.
- `0706T006` advanced full `0625T010` to no-submit preflight and defined the evidence envelope later used by `0706T007`.
- `0706T005 / 0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance` is `已通过`.
- The narrow scoped replay acceptance gate is complete.
- The next roadmap step is not automatic T011.
- Any next live fill-seeking, repeated-window, closer-to-market placement, quote-envelope change, or size change task requires a new formal task and explicit authorization.
- Full `0625T010`, `0625T011`, and `0625T012` remain blocked until new complete live evidence and explicit authorization exist.
- `0706T004 / MVP Roadmap Refresh After T009` is `已通过`.
- `0706T003 / 0625T009` is `已通过`.
- `0706T002 / 0625T008` is `已通过`.
- No further live-submit is authorized without a new formal task and explicit authorization.
- `0706T001 / 0625T008-PREFLIGHT` is `已通过`.
- `0702T002` is `已通过`.
- It fixes the Binance public collector bug exposed by `0702T001`: HTTP `429` REST depth snapshot failures are now retried with low-frequency backoff, recorded in manifests, and treated as a hard collection failure if no valid `lastUpdateId/bids/asks` snapshot is obtained.
- The Binance snapshot default depth for this collector is now `100`, not `1000`, because current top5 bootstrap does not need a high-weight 1000-level snapshot.
- `0702T001` is `已通过` as an invalid dataset archive, with final recommendation `sample_collection_invalid` and `t003_creation_unlocked=false`.
- `0702T001` successfully collected three raw-only windows and processed them locally, but all three Binance depth snapshots failed with HTTP `429` on `awsserver1` IP `18.182.23.227`, so Binance top5 context was incomplete.
- `0625T003` QA is `已通过`.
- T003 used `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/` as its offline/public-only input package.
- T003 hard-gated nominal `1000ms` signal rows by row-level effective horizon: `1000ms <= effective_future_age_ms <= 1250ms`.
- T003 used leave-one-window-out train/evaluation separation, forbade same-window threshold backfill, and generated `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`.
- T003 QA accepts `signal_contract_accepted_for_shadow`, with accepted contract `binance_lead_composite`, threshold `abs(z) >= 1.0`, side mapping `positive_signal_buy_negative_signal_sell`, and one regime/source-age warning bucket. This does not authorize live/shadow/order/canary/promotion behavior.
- `0627T001` is `已通过`.
- The controller-level MVP sequence is defined in `docs/cross_exchange_maker_mvp_plan.md`.

Cross-exchange maker MVP task queue:

- Milestone M-A Signal Contract: `0625T001` alpha/edge decomposition -> `0625T002` synchronized public sample expansion -> `0625T003` out-of-sample signal acceptance.
- Milestone M-B Production-Equivalent Shadow: `0625T004` shared signal/quote-intent kernel -> `0625T005` multi-window production shadow acceptance.
- Milestone M-C Minimal Hyperliquid Alignment: `0625T006` audit/replay contract -> `0625T007` public market-view replay alignment -> `0625T008` edge-qualified tiny-live calibration -> `0625T009` execution outcome calibration.
- Milestone M-D Integrated MVP: `0625T010` same-window replay acceptance -> `0625T011` multi-sample robustness -> `0625T012` final controlled MVP validation.
- `0625T001` is complete, `0625T002` identified the effective-horizon blocker, `0627T001` QA accepted the corrected fast-HL sample set, `0625T003` QA accepted the signal contract for public-shadow use, `0625T004` QA accepted the shared kernel, `0625T005` QA accepted production shadow for replay-contract work, `0625T006` QA accepted the audit/replay contract, `0625T007` QA accepted public market-view replay alignment, `0706T001` QA accepted the no-submit `0625T008-PREFLIGHT` packet, `0706T002` QA accepted first live-submit calibration, and `0706T003` QA accepted one-order execution outcome calibration. Later roadmap items remain blocked until explicit controller/live authorization and dispatch.

Previous formal task:

- `0624T003` QA is `已通过`. It completed the fresh AWS repaired public-shadow validation with `candidate_count=599`, `fresh_touch_allowed_count=68`, `anti_drift_pass_count=4`, `fair_mid_source_pass_count=3`, `edge_gate_pass_count=0`, and `shadow_would_submit_count=0`.
- It confirmed that repaired BBO history/cache and fresh-touch evidence work in the real public stream. The remaining blocker is alpha/edge decisionability after anti-drift and fair-mid filtering.
- It preserved no-submit/no-private/no-order/no-final-gate boundaries and did not authorize canary, quote-distance/cap/post-only relaxation, M3, stable PnL, default-on behavior, or promotion.

Earlier formal task:

- `0624T002` QA is `已通过`. It repaired decision-time BBO history/cache visibility fields, fresh-touch block reason taxonomy, same-touch queue-reset delta evidence, and local/exchange ordering diagnostics.
- `0624T002` T010 replay repair validation reduced `synthetic_current_event_only` from the original `1257` generic blocker to `6`, with `same_touch_stable_enough=1068` and `same_touch_reset_supported=317`.
- The remaining blocker after repair is classified as `bbo_evidence_repaired_remaining_blocker_is_flow_or_downstream_gate`; the next evidence need is fresh AWS public-only no-submit validation.

- `0623T010` QA is `已通过`. It ran a 600s public-only no-submit candidate funnel diagnosis on `awsserver1` using the existing `m2_live_public_source_shadow_v1` path while preserving no-submit / no-private / no-order / no-final-gate boundaries and without changing quote distance or caps.
- `0623T010` funnel evidence: `current_candidate_count=1259`, `shadow_would_submit_count=0`, `fresh_touch_evidence_pass=2`, `fresh_touch_gate_allowed=0`, `fair_mid_source_pass_count=0`, and `edge_gate_pass_count=0`.
- `0623T010` first blocking stage is the accepted fresh-touch / dynamic-size gate, before Binance freshness, fair-mid source, and edge gate. Dominant blockers are `missing_touch_freshness_or_queue_reset_evidence=1257`, `missing_same_side_strict_through_support=960`, and `missing_recent_same_side_at_or_through_throughput=205`.
- `0623T009` business execution is complete and has been QA-accepted. It merges the AWS public no-submit shadow soak and canary preflight ledger into one `awsserver1` remote execution path.
- `0623T009` remote evidence: system `/usr/bin/python3` lacked WebSocket dependencies, so the task recovered with the existing venv `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`.
- `0623T009` live public shadow evidence: `l2Book=34`, `trades=142`, `subscription_ack=2`, `reconnects=0`, `duration_elapsed`; the shadow path stayed fail-closed with `shadow_would_submit_count=0`.
- `0623T009` canary preflight ledger evidence: `live_public_source_observed=true`, `source_path_exercised=false`, `final_recommendation=hyperliquid_tiny_live_m2_canary_preflight_blocked`, `next_real_canary_authorized=false`.

Previous formal task:

- `0623T007` QA is `已通过`. It adds `m2_live_public_source_shadow_v1`, a no-submit public shadow path for the T006 fair-mid provider and T004 edge gate.
- T007 local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/` cover positive fresh public-shadow would-submit/no-submit plus missing Binance state, stale Binance state, wrong symbol, insufficient edge, anti-drift block, and short real public-shadow attempt cases.
- The accepted mock/public-source-compatible path produced `2` shadow would-submit decisions while proving no credential reads, no private/account/order/cancel endpoints, no live client initialization, no remote refresh, and no final gate rerun.
- The short real public-shadow attempt did not observe Hyperliquid public L2 in this environment and is recorded as blocked by `_ssl.c:1011: The handshake operation timed out` / `no_hyperliquid_public_l2_observed`.
- `0623T007` does not authorize a real maker canary. The next live step must first resolve live public-source observation or be separately scoped with explicit approval; M2 remains blocked on live maker fill / fee / inventory / realized PnL proof.
- `0623T006` QA is `已通过`. It implements / accepts `m2_decision_time_public_fair_mid_provider_v1` for the `0623T004` watcher-local edge gate.
- Accepted T006 provider contract: current in-process Hyperliquid public L2/BBO plus decision-time Binance public state with `symbol`, `signal_ts_ms`, bid/ask or mid, and conservative `lead_move_ticks`; target `BTC`, `horizon_ms=1000`; formula `fair_mid_px = current_hyperliquid_mid + conservative_binance_lead_move_ticks * tick_size`.
- T006 local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/` cover positive fresh fair-mid pass plus fail-closed missing source, missing Binance state, stale source, wrong symbol, wrong horizon, insufficient edge, provider exception, missing Hyperliquid state, future timestamp, missing fair mid, and invalid quote/tick cases.
- `0623T006` stayed no-live/no-private/no-order/no-remote-refresh/no-final-gate and does not authorize quote-distance change, one-tick-back, inside-spread, cap relaxation, default-on behavior, M3, stable PnL, or promotion.
- `0623T005` QA is `已通过`. It synthesized accepted `0623T001`-`0623T004` evidence and made the quote-placement envelope decision without code implementation, live orders, credential reads, private/account/order endpoint calls, remote refresh, final gate rerun, cap relaxation, or quote-distance change.
- `0623T005` decision is `continue_touch_only_with_repaired_gates`. The only currently authorized envelope remains `0 tick` touch-only with post-`open_orders` public L2 freshness, real fresh-touch evidence, current BBO / queue quality, anti-drift taxonomy, fair-value edge gate, Hyperliquid `Alo`, dynamic size hard cap `<=0.005 BTC`, tracked cancel, independent open-orders proof, and T008 fail-closed ledger.
- `0623T005` recommends `0623T006 M2 live-compatible fair-mid source acceptance gate` as the next single task after QA. It should accept or implement a fresh `edge_signal_provider` for the watcher-local edge gate while staying no-live/no-private/no-order and without changing quote distance.
- `0623T004` QA is `已通过`. It adds an explicit fair-value edge gate to the watcher-local inline reprice path: after post-`open_orders()` public L2 freshness, immediate fresh-touch/current BBO guard, and anti-drift, but before order submission. The gate records `fair_mid_px`, `quote_px`, `edge_ticks`, `signal_age_ms`, `fee_buffer_ticks`, `adverse_selection_buffer_ticks`, `edge_gate_status`, and `edge_gate_reason`.
- `0623T004` deliberately does not treat accepted read-only pricing-signal / optimistic proxy artifacts as a live alpha source. No accepted live-compatible decision-time fair-mid provider was found, so `--event-driven-edge-gate-live` enables a fail-closed adapter unless a provider is explicitly injected. Local mock artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_edge_gate_0623T004/` cover positive-edge pass plus missing-source, stale-signal, and insufficient-edge blocks. Focused verification passed with event-driven watcher tests `21 passed`, `py_compile`, watcher CLI help, JSON manifest validation, and `git diff --check`. No live order window was run and no real order endpoint was called by this task. It is the accepted input for `0623T005`.
- `0623T003` QA is `已通过`. It splits anti-drift public flow taxonomy so buy-side sell-at-bid touch flow is classified as fill-support touch / visible queue depletion, while strict-through below limit remains adverse; sell-side handling is symmetric.
- `0623T003` changes anti-drift pressure blocking to use strict-through adverse quantity plus recent adverse BBO evidence. Touch-flow support alone no longer blocks, but it is not sufficient by itself and all other gates still apply. Focused verification passed with event-driven watcher/taxonomy tests `16 passed`, `py_compile`, watcher CLI help, and `git diff --check`. Local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_flow_taxonomy_0623T003/` cover touch-support pass, strict-through + adverse-BBO block, and mixed-flow pass cases. No live order window was run and no real order endpoint was called. It unblocked `0623T004`, which has now passed QA.
- `0623T002` QA is `已通过`. It hardens event-driven fresh-touch evidence so synthetic `quote_aging_status=stayed_touch` is no longer sufficient when `event_driven_inline_candidate=true`; real BBO-history touch stability or top reset evidence is required.
- `0623T002` adds fields such as `freshness_source`, `touch_stability_ms`, `last_touch_change_ms`, `top_reset_status`, `top_reset_reason`, and `fresh_touch_evidence_status`. Local verification passed with event-driven watcher tests `13 passed`, fill-loop tests `22 passed`, `py_compile`, two CLI help checks, and `git diff --check`. Local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fresh_touch_evidence_0623T002/` include synthetic-only block and real BBO-history pass cases. No live order window was run and no real order endpoint was called. The next executable item is `0623T003`.
- `0623T001` QA is `已通过`. It added post-`open_orders()` public L2 freshness gating to the watcher-local inline reprice path: reprice / submit now requires a new `l2Book` observed after `open_orders_end_ns`, records public-state seq evidence, and fails closed before order submission if no fresh post-open L2 arrives within the bounded wait.
- `0623T001` was verified locally only: event-driven watcher tests `11 passed`, public watcher tests `4 passed`, `py_compile` passed, CLI help passed, and `git diff --check` passed. Local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_state_freshness_0623T001/` include one mock pass case and one mock block case proving stale post-open L2 blocks the order endpoint. No live order window was run, no credentials were read, no remote checkout was refreshed, and no real order endpoint was called. The next executable item is `0623T002`.
- `0622T006` QA is `已通过` at task level. It added the anti-drift / touch-stability submit gate to the watcher-local inline reprice path and preserved `Alo`, unchanged dynamic size hard cap `<=0.005 BTC`, max `30` real submissions, no taker/crossing, no one-tick-back, no cap relaxation, public-only waiting, tracked cancel, independent open-orders proof, and T008 fail-closed.
- `0622T006` did not complete M2. Formal run 1 evaluated `914` current candidates, anti-drift gate passed `6` / blocked `93` out of `99` gate evaluations, and submitted `2` valid post-only `Alo` buy attempts (`0.00422 BTC @ 64956.0`, `0.00179 BTC @ 65032.0`). Both were rejected by exchange-side post-only validation after BBO drift to `64954@64955` and `65025@65026`; T008 returned `fail_closed_no_realized_live_pnl`.
- `0623T001`-`0623T005` were created as a sequential repair queue, not parallel work, and all five tasks are now QA accepted. These tasks do not authorize taker/crossing, one-tick-back, inside-spread, cap relaxation, default-on behavior, M3, live execution, or stable PnL claims.
- `0622T005` QA is `已通过` at task level. It implemented the inline reprice / post-only reject repair inside the event-driven watcher and executed one controlled remote live-calibration run. The formal run refreshed remote to `c4faf36b7a60342f195238041d2b711ca300233e`, final gate passed, watcher triggered once after `82.208056s`, and submitted `2` post-only `Alo` buy attempts at `0.00004 BTC`. Both attempts were rejected by Hyperliquid post-only protection after exchange-side BBO drift (`64143@64144`, then `64142@64143`), so no fill occurred and T008 returned `fail_closed_no_realized_live_pnl`.
- `0622T004` QA is `已通过` at task level. It converted the `0622T003` same-process watcher path from fixed-window batch candidate generation to event-driven rolling current-candidate evaluation and added a fast event-driven submit path that defers slow private/account pre-submit reads. The formal rerun reached one real post-only `Alo` submit attempt at `0.005 BTC`, but the exchange rejected it as would-immediately-match after fast BBO drift; no fill occurred and T008 returned `fail_closed_no_realized_live_pnl`.
- M2 remains blocked on live maker fill plus fee/inventory/realized PnL proof. The current evidence does not authorize M3, stable PnL claims, taker/crossing, one-tick-back, cap relaxation, or default-on behavior.
- `0622T003` QA is `已通过` at task level. It implemented the same-process watcher-triggered maker-order repair and ran two formal same-process remote attempts. The final short-iteration rerun found one `quality_a` buy candidate, but the immediate same-process pre-submit guard failed closed before order submission because the candidate was stale, the selected quote was no longer current touch, and the current top-of-book queue had refilled outside the quality band. No live order was submitted, independent open-orders proof was empty, and T008 ledger returned `fail_closed_no_realized_live_pnl`; M2 remains blocked.
- `0622T002` QA is `阻塞`. It implemented and ran a time-boxed public-only L2/trades watcher with gated maker live trigger. The watcher found one `quality_a` buy candidate, but the triggered live window reran current fresh-touch gating and found `0` full quality-gate allowed candidates before order submission. No live order endpoint was called, no fill occurred, independent open-orders proof was empty, and T008 ledger failed closed with `fail_closed_no_realized_live_pnl`; M2 remains blocked on trigger-to-order staleness / no live realized PnL proof.
- `0622T001` QA is `阻塞`. It implemented `fresh_touch_size_by_throughput_session_gate` in `fill_window` / `fill_loop`, passed focused local verification, refreshed remote checkout, reran final gate, ran public precheck/session-gate, and stopped before order submission because the formal micro-window had `0` full quality-gate allowed fresh-touch candidates. No live order endpoint was called, no fill occurred, and T008 ledger failed closed with `fail_closed_no_realized_live_pnl`; M2 remains blocked.

Latest QA result:

- `0623T005` is `已通过`. QA accepted the quote-placement envelope decision gate: continue `0 tick` touch-only with repaired gates, reject immediate one-tick-back / inside-spread as next tasks, and recommend `0623T006 M2 live-compatible fair-mid source acceptance gate` as the next single no-live/no-private/no-order task. M2 remains blocked because no accepted live-compatible decision-time fair-mid / edge source exists yet and no live maker fill / fee / inventory / realized PnL proof exists.
- `0623T004` is `已通过`. QA accepted the fair-value edge gate integration as an additional pre-submit requirement after state freshness, fresh-touch/current BBO guard, and anti-drift. The implementation keeps accepted pricing-signal artifacts as read-only/proxy evidence only; no live-compatible fair-mid source was accepted, so the live edge adapter fails closed without an injected provider. Focused verification passed (`21 passed`), py_compile, watcher CLI help, JSON validation, and `git diff --check`; local artifacts cover positive-edge pass plus missing-source, stale-signal, and insufficient-edge blocks. M2 remains blocked and M3 must not start.
- `0622T004` is `已通过` at task level. QA accepted the event-driven current-candidate / fast-submit repair, focused regression (`40 passed` on the current workspace), artifact completeness (`93` files, `0` empty), public-only waiting boundary, `Alo`, `<=0.005 BTC`, max `2` submission cap, independent open-orders proof, redaction, and T008 fail-closed behavior. This does not complete M2: the single valid post-only submit was rejected after exchange-side BBO drift to `64090@64091`, no order filled, and T008 returned `fail_closed_no_realized_live_pnl`.
- `0622T003` is `已通过` at task level. QA accepted the same-process watcher/live repair, focused regression, artifact completeness (`714` files, `0` empty), public-only waiting boundary, same-process guard fail-closed evidence, independent open-orders proof, redaction, and T008 fail-closed behavior. This does not complete M2: no order was submitted because the selected candidate was stale and no live PnL proof exists.
- `0622T001` is `阻塞`. QA accepted the implementation and safety boundaries, but the formal task outcome is blocked by the expected fail-closed condition: no eligible `quality_a` / `quality_b` fresh-touch candidate appeared, no live order was submitted, and T008 returned `fail_closed_no_realized_live_pnl`.
- `0622T005` is `已通过` at task level. QA accepted the inline watcher-local reprice / submit repair, public-only waiting boundary, same-process submit path, max `2` post-only `Alo` attempts, `<=0.005 BTC` cap, post-only reject retry matrix, final open-orders proof, independent open-orders proof, redaction, and T008 fail-closed behavior. This does not complete M2: both valid attempts were rejected by exchange-side post-only validation after BBO drift, no order rested or filled, and T008 returned `fail_closed_no_realized_live_pnl`. M2 remains blocked and M3 must not start.
- `0622T002` is `阻塞`. QA accepted the implementation and safety boundaries for the time-boxed public watcher and gated maker live trigger: focused regression passed (`25 passed`), artifacts were complete (`163` files, `0` empty), remote refresh and final gate passed, watcher remained public-only, redaction held, independent open-orders proof was empty, and T008 failed closed. The watcher found `1` eligible `quality_a` buy candidate, but the separate live window reran current fresh-touch gating after handoff and found `0` allowed candidates, so `live_submissions_count=0`, `fill_count=0`, and no fee/inventory/realized PnL proof exists. M2 remains blocked and M3 must not start.
- `0619T002` is `已通过`. It accepted the design-only M2 maker quote-placement / size / time-of-day redesign contract. The accepted contract changes the next implementation shape from generic flow-aware touch join to fresh-touch touch-only entry, dynamic size-by-throughput with a smaller `<=0.005 BTC` hard cap, buy-default side discipline, and no fixed time-of-day claim until cross-hour public scorecards exist. It did not change code, place orders, read credentials, refresh remote checkout, or rerun final gate. M2 remains blocked because no live maker fill / fee / inventory / realized PnL proof exists yet.
- `0619T001` is `阻塞`. It implemented and executed the M2 maker-only flow-aware retry repair. Remote refresh used an incremental bundle from `5391b79439a4f0cb24fd40e4e6aa4b8de53f73e3` to `7d0e813704addc5412e974c79b0eee922075a544`, final gate returned go, public-flow precheck passed, and one live window ran with `6` flow-aware attempts. Two post-only `Alo` candidates were submitted, four crowded-touch candidates were skipped, tracked cancel/final open-orders proof passed, and independent remote open-orders check was empty. No fill occurred and T008 ledger returned `fail_closed_no_realized_live_pnl`; M2 remains blocked and M3 must not start.
- `0618T012` is `已通过`. It completed a read-only Hyperliquid public L2/trades flow diagnosis without live orders, credential reads, private/account/order endpoint calls, remote checkout refresh, final gate rerun, taker/crossing behavior, or cap relaxation. Local direct public collection from this machine failed with Hyperliquid public API/WebSocket SSL EOF and was recorded only as a collection-path blocker; a public-only `awsserver1` collection succeeded with `l2Book=56`, `trades=122`, `subscription_ack=2`, `reconnects=0`, and `close_reason=duration_elapsed`. The pulled-back sample produced `56` book events, `356` individual trade events, and `38` passive touch-quote candidates. Hypothesis matrix: `queue_too_deep=supported`, `no_trade_through=rejected_for_sample`, `wrong_time_of_day=inconclusive`, `wrong_side=supported`, and `quote_aging_or_fast_drift=supported`. M2 remains blocked on live maker fills and M3 must not start.
- `0618T011` is `已通过`. It completed the M2 no-fill diagnosis/design decision without network, live orders, credential reads, remote refresh, final gate rerun, or private/order endpoint calls. It consumed local `0618T009` / `0618T010` pulled-back artifacts and analyzed `9` maker-only `Alo` attempts: `6` buy / `3` sell, `9/9` no-fill, `9/9` same-side touch join, median spread proxy `1` tick. Defensible public same-side depth proxy exists for `4/9` attempts only and ranged from `21.83x` to `1921.75x` of the `0.00999 BTC` order; the rest are correctly marked `per_attempt_depth_missing`. The design decision is `do_not_blind_retry; run_read_only_public_flow_diagnosis_next`. M2 remains blocked on live maker fills and M3 must not start.
- `0618T010` is `阻塞`. It attempted the M2 no-fill repair by adding adaptive same-window maker-only cancel/requote under the same caps. The live retry completed one 10-minute-bounded window with `6/6` post-only `Alo` attempts, `side_policy=alternate`, crossing guard pass, tracked cancel, final open orders 0, and independent remote open-orders count 0. No attempt filled; the aggregate fill ledger is empty and the T008 ledger again reports `fail_closed_no_realized_live_pnl`. M2 remains blocked.
- `0618T009` is `阻塞`. It executed the controlled M2B tiny-live fill loop after git-safe remote refresh and final gate go. Three real post-only `Alo` windows reached `resting`, used tracked cancel, ended with `final_open_orders_count=0`, and preserved redaction/safety boundaries. However all three windows had `fill_count=0`; the T008 ledger rerun in `live_pulled_back` mode reports `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`. M2 is not complete and M3 must not start from this evidence.
- `0618T008` is `已通过`. It completed M2A with a no-network Hyperliquid PnL ledger/reconciler and official artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_pnl_ledger_0618T008/`. Against accepted M1 artifacts it correctly reports `live_realized_pnl_proof=false`, `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`, and `m1_no_fill_fail_closed=true`. A local fixture proves ledger arithmetic only (`gross_pnl_usdc=0.26`, `fee_usdc=0.125268`, `net_pnl_usdc=0.134732`, `inventory_delta_btc=0.01`, `slippage_usdc=0.0`). It did not place orders, read credentials, call private/account/order endpoints, start a live bot, or prove stable PnL / maker viability.
- `0618T007` is `已通过`. It completed M1 repeated tiny-live canary windows with one formal task and three independent Hyperliquid post-only `Alo` canary windows. The loop refreshed `/home/admin/hftbacktest-cross-exchange` through git bundle + `merge --ff-only`, reran final gate to `allow_create_0617T008=true`, and produced `hyperliquid_tiny_live_m1_repeated_canary_ready_for_qa`. All three windows reached `resting`, used tracked cancel, ended with `final_open_orders=[]`, did not call `schedule_cancel`, and preserved redaction boundaries. It does not prove realized PnL, fees/rebates, inventory accounting, stable PnL, maker viability, default-on, promotion, or scale-up.
- `0618T006` is `已通过`. It refreshed the M1 precondition remote checkout facts for `/home/admin/hftbacktest-cross-exchange`, confirmed remote state `cross-exchange:d37438e0c:0` with `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`, and reran the final gate into `local_live_analysis/hyperliquid_tiny_live_final_go_no_go_gate_0618T006/`. The gate now reports `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`. It did not place orders, read credentials, call private/account/order endpoints, start a live bot, execute M1, or prove PnL / maker viability.
- `0618T005` is `已通过`. It completed M0 evidence-chain baseline verification with a single read-only business task plus QA. It reran the signal / quote replay and optimistic PnL proxy into `local_live_analysis/m0_evidence_chain_baseline_0618T005/`, re-parsed the accepted `0618T004` canary artifacts, and reran the final gate without placing orders, reading credentials, calling private/account/order endpoints, or starting a live bot. M0 is complete as a baseline verification, but the latest read-only final gate rerun fails closed with `allow_create_0617T008=false` because saved remote state is `cross-exchange:52b5b9541:0` while the current local gate commit is `d37438e`; before M1, remote checkout state must be refreshed/synced and the final gate rerun must pass.
- `0618T004` is `已通过`. It executed and QA-accepted the first real-order Hyperliquid canary on `awsserver1`, using `/home/admin/XEMM_rust_latest/.env` as the credential source without returning secret values. The canary exercised private read, real `Exchange.order`, `Exchange.cancel`, `Exchange.cancel_by_cloid`, `Exchange.schedule_cancel`, and final `Info.open_orders`; the order response was `resting`, tracked cancel succeeded, final open orders were empty, and shutdown proof is `pass`. `schedule_cancel` was reachable but rejected by account traded-volume eligibility, so later live tasks must not rely on scheduled-cancel / dead-man switch unless this changes. QA refreshed the remote final-gate input after report sync; remote state is `cross-exchange:52b5b9541:0`, and T004 final gate reports `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.
- `0618T003` is `已通过`. It checked exactly four `awsserver1` XEMM candidate files and located Hyperliquid credential-shaped fields in `/home/admin/XEMM_rust/.env` and `/home/admin/XEMM_rust_latest/.env` as `HL_WALLET` and `HL_PRIVATE_KEY`. No token/private-key values were returned or written to repo artifacts.
- `0618T002` is `已通过`. It made the official Hyperliquid Python SDK available locally and on `awsserver1`, verified import and required `Exchange` / `Info` method surfaces without credentials or endpoint calls, reran the executor self-test, and reran the final gate. The gate now returns `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`. This only allows total control to create a later live/canary task; `0618T002` did not create or execute `0617T008`.
- `0618T001` is `已通过`. The minimal Hyperliquid tiny-live real-order executor repair is accepted, the no-order self-test artifacts are complete, and the repaired final gate still fails closed on the missing official SDK blocker. This unblocks `0618T002`.

- `0617T007` is `已通过` as a final go/no-go gate, but its gate decision is no-go. QA accepted that the gate correctly consumes accepted local evidence, records `canonical_7`, reconciles the old `0616T008` approval packet against the current `0617T008` instruction, records remote state, and preserves no-live/no-order/no-private boundaries. The gate output is `tiny_live_needs_missing_precondition`, `allow_create_0617T008=false`, with blockers `remote_execution_checkout_not_synced_or_invalid` and `hyperliquid_real_order_executor_missing_or_unproven`. `0617T008` must not be created or executed from this gate result.
- `0617T006` is `已通过`. It accepted the Hyperliquid tiny-live read-only optimistic PnL proxy with `canonical_7` as the official sample set. At `75` ticks / persistence `2` / `1000ms`, official `canonical_7` reports `295.985 USDC`, mean `40.932789` ticks per intent, and `7/7` positive samples under the `unconstrained_all_intents` optimistic public-data upper-bound assumption. This is not real PnL, execution viability, maker viability, or live authorization.
- `0617T005` is `已通过`. It accepted the Hyperliquid tiny-live read-only signal / quote replay over `8` local pricing-signal inputs, selected `75` ticks with persistence `2` as the primary read-only threshold candidate, and preserved the no-live/no-order/no-private/no-real-PnL boundary. Its direct follow-up `0617T006` has now also passed QA; this T005 evidence still does not by itself authorize live execution.
- `0617T003` is `已通过`. It created the final Hyperliquid tiny-live live-capable preflight/operator packet on `awsserver1`, binding `/home/admin/hftbacktest-cross-exchange`, `/usr/bin/python3`, `scp` pullback, and the approved `0616T008` caps into artifacts under `local_live_analysis/hyperliquid_tiny_live_final_live_capable_preflight_0617T003/`. It did not read credentials, call private endpoints, query accounts, place/cancel/amend orders, start a live bot, deploy, promote, prove PnL, or claim maker viability. The next step, if total control requests it, is a separate `0616T008` live execution task; `0617T003` must not be reinterpreted as live execution.
- `0617T002` is `已通过`. It repeated the `awsserver1` cross-exchange python3 preflight on `/home/admin/hftbacktest-cross-exchange`, confirmed the remote state remained `cross-exchange:7642b16:0`, selected remote Python remained `/usr/bin/python3` (`Python 3.13.5`), and artifact pullback/checksum validation with `scp` worked again. No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred. The next task may be a final live-capable preflight/operator task binding the new path, system Python, `scp` pullback, and approved caps; it must still pass QA before any real orders.
- `0617T001` is `已通过`. It created a separate `awsserver1` checkout at `/home/admin/hftbacktest-cross-exchange` for the current local `cross-exchange` branch, preserving the existing Binance maker `master` route at `/home/admin/hft_live/hftbacktest`. The new route is `cross-exchange:7642b16:0`, selected remote Python is `/usr/bin/python3` (`Python 3.13.5`), and artifacts were pulled back with `scp` and validated locally. No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred. This resolves the `0616T007` environment interpretation blocker, but does not by itself authorize `0616T008`; a new live-capable preflight dry-run over the new path should pass QA first.
- Historical blocker: `0616T007` is `阻塞`. It executed the `awsserver1` live-capable preflight dry-run and pulled back dry-run artifacts, but the remote repo at `/home/admin/hft_live/hftbacktest` is on `master` instead of `cross-exchange`, has `29` dirty status rows, and lacks both `conda` and `rsync`. No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred. The auto loop stopped there and `0616T008` was not created.
- Controller clarification after `0616T007`: the remote `/home/admin/hft_live/hftbacktest` `master` checkout is the Binance maker route and should not be repurposed for the Binance-lead / Hyperliquid-lag route. The next repair task may create a separate remote path `/home/admin/hftbacktest-cross-exchange` for the current local `cross-exchange` branch, and remote preflight may use system `python3` when recorded explicitly. Local validation should still use `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python`.

Latest dispatched task:

- `0624T003`: M2 AWS repaired public-shadow funnel live validation is `已通过` after QA. Official artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_aws_repaired_public_shadow_funnel_0624T003_20260624T063826Z/`. The fresh AWS run kept public-only/no-submit boundaries, produced `599` candidates, reduced synthetic-only evidence to `2`, raised `fresh_touch_allowed_count` to `68`, and showed the next blockers are anti-drift and edge/fair-mid. No real canary is authorized.
- `0622T006`: M2 anti-drift maker-only submit gate / live calibration is `已通过` at task level after QA. Implementation commits are `73d473a`, `13ec3cc`, and `5d8a1ec`; official artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006/`, `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun/`, and `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun2/`. The main formal run kept `Alo` and `<=0.005 BTC`, submitted `2` valid maker-only attempts under the `30` call cap, got `2` post-only would-immediately-match rejects after fast BBO drift, and T008 failed closed with no realized live PnL proof. M2 remains blocked.
- `0622T005`: M2 event-driven inline reprice / post-only reject repair is `已通过` after QA. Implementation commits are `741b5b2` and `c4faf36`; official artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_inline_reprice_0622T005/`. Formal path: remote refresh and final gate passed; watcher ran `82.208056s`, evaluated `48` current candidates, triggered once, and submitted `2` post-only `Alo` buy attempts at `64144.0` for `0.00004 BTC`. Post-`open_orders` reprice-to-submit latency was about `0.043ms-0.045ms`, but both orders were rejected by exchange-side post-only validation after BBO moved to `64143@64144` and `64142@64143`. Final open orders were empty; T008 ledger returned `fail_closed_no_realized_live_pnl`. M2 remains blocked.
- `0622T004`: M2 event-driven current-candidate maker submit repair is `已通过` after QA. Implementation commits are `59ec906` and `f6487c0`; official final artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_event_driven_current_candidate_0622T004_rerun_fast_submit/`. Formal rerun refreshed remote to `f6487c063`, final gate passed, watcher triggered after `11.32509s`, outer guard latency was `0.000286s`, inner guard passed at candidate age `0.609s`, one `Alo` buy order of `0.005 BTC` was submitted, and Hyperliquid rejected it as post-only would-immediately-match after BBO moved to `64090@64091`. Final open orders were empty and T008 failed closed with no realized live PnL proof; M2 remains blocked.
- `0622T003`: M2 same-process watcher-triggered maker order repair is `已通过` after QA. Implementation commits are `1b063a7` and `5c58af1`; official final artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter/`. Formal final path: remote refresh and final gate passed; watcher found `1` eligible `quality_a` buy candidate out of `150` candidates after `163.946857s`; same-process guard then failed closed with `trigger_candidate_stale_before_order;selected_quote_not_current_touch;current_top_depth_outside_quality_a_band;current_top_order_count_outside_quality_a_band`. Submitted live orders were `0`; independent remote open orders were empty; T008 ledger returned `fail_closed_no_realized_live_pnl`. M2 remains blocked.
- `0622T002`: M2 time-boxed public watcher with gated maker live trigger is `阻塞` after QA. It produced implementation commit `b1c1ff9` and official artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_timeboxed_watcher_0622T002/`. Formal gate path: remote refresh and final gate passed; public watcher found `1` eligible `quality_a` buy candidate out of `54` candidates after `120.451756s`; live trigger started but the immediate live-window pre-submit gate found `0` full quality-gate allowed candidates, so submitted live orders were `0`; independent remote open orders were empty; T008 ledger returned `fail_closed_no_realized_live_pnl`. QA accepted safety/implementation evidence but M2 remains blocked.
- `0622T001`: M2 fresh-touch dynamic-size session-gate implementation and controlled live micro-test is `阻塞` after QA. It produced implementation commits `81cb084` and `ad6080c`, official artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fresh_touch_live_0622T001/`, and a business report at `.workflow/reports/0622T001-business.md`. Formal gate path: remote refresh and final gate passed; public precheck passed; fresh-touch full quality-gate allowed candidates were `0`; submitted live orders were `0`; independent remote open orders were empty; T008 ledger returned `fail_closed_no_realized_live_pnl`.
- `0618T012`: M2 read-only public L2/trades flow diagnosis is `已通过`. It rejects `no_trade_through` as the primary explanation for the sampled window and points to queue depth, quote aging / fast drift, and side asymmetry as the current public-flow blockers. M2 remains blocked; next M2 work should be a maker-only retry design/repair, not a blind live retry, and must preserve `Alo`, T008 ledger fail-closed, and same or smaller caps.
- `0619T001`: M2 maker-only flow-aware retry repair is `阻塞`. It was the first non-blind retry after T012 and preserved `Alo`, same-or-smaller caps, tracked cancel, final open-orders proof, and T008 ledger fail-closed, but still got no live maker fill.
- `0619T002`: M2 maker quote-placement / size / time-of-day redesign contract is `已通过`. It concludes the next task should not keep the current `500x` crowded-touch tolerance, `15s` hold, fixed `0.00999 BTC` size, or sell-allowed symmetry. The next implementation should instead use fresh-touch queue bands (`<=20x` / `20x-100x`), dynamic size `<=0.005 BTC`, buy-default side gating, and current-window-only execution until a later cross-hour scorecard exists.
- `0618T011`: M2 no-fill diagnosis and fill-acquisition design decision is `已通过`. It rejects another blind same-caps live retry for now and recommends a separately scoped read-only public L2/trades flow diagnosis before any later maker-only retry. M2 remains blocked.
- `0618T010`: M2B fill-acquisition repair with adaptive maker requote is `阻塞`. The repair executed as designed and preserved safety boundaries, but still produced no passive maker fill. Next action should be no-fill diagnosis / design decision rather than another blind same-caps retry.
- `0618T009`: M2B Hyperliquid controlled tiny-live fill loop and PnL reconciliation is `阻塞`. The safety and shutdown path passed, but no passive maker fill occurred in three windows, so no real PnL / fee / inventory reconciliation proof exists. Next action, if continuing M2, should be a separately scoped fill-acquisition retry/repair under the same maker-only caps and T008 ledger gate.
- `0618T008`: M2A Hyperliquid real PnL ledger and reconciliation no-order gate is `已通过`. It establishes the ledger/reconciliation gate required before M2B live fill windows. The next task is `0618T009`, a controlled tiny-live fill loop plus PnL reconciliation using T008; it must fail closed if no fill or incomplete economics/inventory settlement is observed.
- `0618T007`: M1 Hyperliquid repeated tiny-live canary loop with git-safe refresh is `已通过`. It is the latest completed task and the latest QA result. M1 is complete, but only as repeated canary order/cancel/shutdown evidence.
- `0618T006`: Remote checkout refresh and final gate rerun before M1 is `已通过`. It resolved the `0618T005` remote checkout drift by confirming remote `cross-exchange:d37438e0c:0` and rerunning the final gate to `allow_create_0617T008=true`. It is a precondition refresh only, not a live/canary execution task.
- `0618T005`: M0 evidence-chain baseline read-only verification is `已通过`. It completed the minimum-task M0 closure using one read-only business task plus QA, with outputs under `local_live_analysis/m0_evidence_chain_baseline_0618T005/`. It did not place orders, read credentials, call private/account/order endpoints, or start a live bot. The evidence baseline is complete, but the current final gate rerun is no-go until remote checkout state is refreshed/synced to the current local gate commit.
- `0618T001`: Hyperliquid tiny-live minimal real-order executor and final gate repair has completed business execution and is `待验收`. It implemented a no-default-live, SDK-wrapped real-order executor scaffold, cap/post-only/max-loss/cancel-all/redaction tests, local and `awsserver1` no-order self-test artifacts, artifact pullback, and a repaired final gate. The executor blocker from `0617T007` is repaired at wrapper/self-test level, but the repaired gate remains no-go because the official Hyperliquid Python SDK is unavailable locally and on `awsserver1`. Final gate output is `tiny_live_needs_missing_precondition`, `allow_create_0617T008=false`, blocker `hyperliquid_official_sdk_dependency_unavailable`. No real order/private/account endpoint was called and `0617T008` was not created.
- `0617T007`: Hyperliquid tiny-live final go/no-go gate is `已通过` for gate QA, but the accepted gate decision is no-go. It is read-only and reconciles `0617T001-T006`, `0616T006-T007`, the approved caps/operator packet, latest controller instruction, and actual `awsserver1` remote facts before any possible `0617T008`. Final recommendation is `tiny_live_needs_missing_precondition`, `allow_create_0617T008=false`, with blockers `remote_execution_checkout_not_synced_or_invalid` and `hyperliquid_real_order_executor_missing_or_unproven`. It did not create `0617T008`, read credentials, call private endpoints, query accounts, place/cancel/amend orders, or start a live bot.
- `0617T006`: Hyperliquid tiny-live read-only optimistic PnL proxy is `已通过`. After user clarification, `canonical_7` is the formal sample-set口径. It estimates a public-data theoretical upper bound under 100% theoretical intent fill assumptions, using future Hyperliquid mid labels and the `0617T004` / `0617T005` rule sources. It remains no-live/no-order/no-private and must not claim real fills, real PnL, fee/rebate settlement, spread capture, account inventory, execution quality, maker viability, deployment readiness, or live authorization.
- `0617T005`: Hyperliquid tiny-live read-only signal / quote replay is `已通过`. It replays the `0617T004` protocol over accepted local cross-exchange public/read-only datasets and remains no-live/no-order/no-private. It cannot claim real fills, PnL, real inventory, post-only reject behavior, queue priority, deployment readiness, or maker viability.
- `0617T004`: Hyperliquid tiny-live signal / quote policy protocol has been created as the next task after `0617T003` QA passed. It must define the minimal Binance-lead / Hyperliquid-lag maker signal and quote policy before any `0616T008` live execution can be created. It is design/protocol only and must not place/cancel/amend orders, query accounts, call private endpoints, read credentials, start a live bot, deploy, promote, prove PnL, or claim maker viability.
- `0617T003`: Hyperliquid tiny-live final live-capable preflight / operator packet on `awsserver1` has been created as the next task after `0617T002` QA passed. It must bind `/home/admin/hftbacktest-cross-exchange`, `/usr/bin/python3`, `scp` pullback, and the approved `0616T008` caps into final preflight/operator artifacts only. It must not place/cancel/amend orders, query accounts, call private endpoints, read credentials, start a live bot, deploy, promote, prove PnL, or claim maker viability. It is not `0616T008` live execution.
- `0616T006`: Hyperliquid tiny-live live-capable preflight / operator packet has been created, dispatched, and completed by the business thread to `待验收` after `0616T005` QA passed. It prepares the future `awsserver1` operator packet and local artifact validation path only. It does not authorize real orders, cancellation, live bot startup, account query, credential disclosure, signing/nonce/user-stream implementation, deployment, promotion, PnL proof, or maker viability proof. Unapproved live fields remain `pending_controller_approval`.

Prepared but not dispatched:

- `0618T002`: Hyperliquid official SDK dependency readiness and final gate repair has been created as the next dependency-repair task, but it must not execute until `0618T001` QA is `已通过`. It is no-order/no-private/no-account and targets the current `0618T001` repaired-gate blocker `hyperliquid_official_sdk_dependency_unavailable` by making the official Hyperliquid Python SDK available locally and on `awsserver1`, proving SDK surface readiness without credentials or endpoint calls, rerunning executor self-test, and rerunning final gate. It must not create `0617T008` or claim real submit-order API success.
- No live follow-up is authorized by `0617T007` QA because the accepted gate decision is `allow_create_0617T008=false`. `0618T001` is the current repair task for the missing real-order executor and final gate evidence; it is not a live execution task. Any later execution / live follow-up still needs a new passing final gate, explicit dispatch, prerequisites, QA, and controller approval boundaries.
- `0618T004`: Hyperliquid tiny-live real-order canary interface validation is the next task to create after `0618T003` QA passed and `0618T002` raised the gate to `allow_create_0617T008=true`. It is the first task allowed to verify the authenticated real-order interface chain using the `0618T003` credential location results, but it must still remain a minimal canary with strict caps, immediate cancel / shutdown evidence, and artifact pullback. It is not a continuous live strategy task and it must not relax `10min / 0.01 BTC / post-only / max loss cap`.

Execution update:

- `0617T007` business execution completed. The gate consumes accepted local QA/business artifacts and actual remote state, then fails closed with `allow_create_0617T008=false` because the remote execution checkout is stale and no QA-accepted Hyperliquid real-order executor path exists. `0617T008` was not created or executed.
- `0617T007` QA is `已通过` for the gate itself, and confirms the no-go result. Auto loop step 3 is not executed because the gate does not allow `0617T008` creation.
- `0617T004` business execution completed. The protocol now defines the minimum Binance-lead / Hyperliquid-lag maker signal and quote policy, but it fails closed on the live absolute threshold because accepted artifacts do not provide a defensible cutoff. The next step is a separate read-only replay / threshold calibration task before any `0616T008` live execution.
- `0617T005` QA is `已通过`. Full quote replay consumes `8` locally available pricing-signal inputs: `0601T005` plus all `7` historical event-mode `pricing_signal_rows.csv` files referenced by `0609T008`. It replayed `161455` raw pricing rows and `26948` de-duplicated decision rows. The primary read-only calibration candidate is `75` ticks with `2` observations of persistence (`654` intents, `313` buy / `341` sell, `2.4269%` intent rate, `8/8` sample coverage), with `75` ticks and `3` observations as a stricter fallback (`327` intents, `1.2134%` intent rate). This still does not authorize `0616T008` live execution before `0617T006` QA/controller ratification.
- `0617T006` QA is `已通过`. The user selected `canonical_7` as the formal sample-set口径, so the prior exact-six ambiguity no longer blocks acceptance. At `75` ticks / persistence `2` / `1000ms`, official `canonical_7` reports `295.985 USDC` and `7/7` positive samples; diagnostic `0617T005_8_input` reports `299.38 USDC` and `8/8` positive samples. Final recommendation is `hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa`. This is not real PnL or execution viability.

Latest live approval:

- On `2026-06-17`, the controller approved a limited `0616T008` Hyperliquid tiny-live execution window after `0616T006` QA and `0616T007` dry-run QA both pass. Approved caps: `symbol=BTC`, `max_order_size=0.01 BTC`, `max_order_notional=700 USDC`, `max_position=0.04 BTC`, `max_position_notional=2800 USDC`, `max_notional=3000 USDC`, `max_loss=30 USDC`, `duration=10 minutes`, `host_machine=awsserver1`, `account_scope=Hyperliquid account configured on awsserver1`, `maker_only/post_only=true`, `real_orders_allowed=true`. BTC/USD reference at approval time was `65794.035`; this approval is limited to `0616T008` and does not authorize scaling, default-on behavior, deployment, promotion, relaxed caps, or later live windows.

Latest auto-loop result:

- `0616T001` through `0616T005` are `已通过` under the cross-exchange auto-loop protocol. The loop created a controller runbook, accepted branch correction, defined Hyperliquid private/order readiness boundary, implemented a no-trading local Hyperliquid private order artifact validator, implemented a local fake shutdown dry-run proof gate, and defined a Hyperliquid tiny-live protocol design. The loop has intentionally stopped at the human approval gate. No live task, private endpoint call, credential/signing/nonce/user-stream implementation, account query, real order placement, real cancellation, deployment, promotion, or PnL proof is authorized. Any next live-capable task requires explicit controller approval of symbol, max notional, max order size, max position, max loss, duration, host/machine, account scope, and whether real orders are allowed.

Latest correction:

- `0616T001`: Cross-exchange branch correction and Hyperliquid maker execution-readiness gate is `待验收`. It records that the active branch intent is Binance lead / Hyperliquid lag maker research, not Binance single-venue maker live. It stops the planned `0615T009` Binance `BTCUSDT` small-cap live path and stops using `0615T008` as the cross-exchange live predecessor. `0615T001-T007` remain reusable as methodology/templates for source-line readiness, local read-only collectors, source-chain runner-consumption gates, and proof-limited runner mechanics, but they are not sufficient Hyperliquid private/order readiness artifacts. The corrected next direction is a separately scoped Hyperliquid maker private/order execution-readiness boundary using `0601T004` / `0601T005` / `0609T002` as the relevant cross-exchange evidence chain. This does not authorize Hyperliquid private/order endpoints, credentials, signing, nonce, user stream, live order placement, strategy implementation, parameter search, deployment, promotion, or PnL/maker viability proof.

Latest accepted execution:

- `0611T004`: Basis-positive account inventory local artifact skeleton / validator is `已通过`. It implemented only a task-scoped local account/inventory fixture schema, disk-only CSV/JSON parser, fail-closed validator, inventory snapshot / transition / conservation checks, CLI/help, tests, design note, local artifacts, and business report using accepted `0610T008` / `0611T001` inputs and `0611T002` / `0611T003` as context only. Final recommendation is `account_inventory_artifact_skeleton_ready_for_qa`; this means only that the local account inventory artifact skeleton / validator is ready for QA/controller review. It does not authorize endpoint/source collector/runner implementation, account/private/order/live data, user stream, signing/nonce handling, inventory lifecycle proof, realized inventory/exposure proof, economics metrics, PnL proof, strategy/live/default-on/tiny-live readiness, deployment, promotion, or execution-layer maker viability proof.
- `0611T003`: Basis-positive replay lifecycle validation / reconciliation gate is `已通过`. It implemented only a task-scoped local replay lifecycle fixture schema, disk-only parser, fail-closed validator, ordering/reconciliation policy artifacts, CLI/help, tests, design note, local artifacts, and business report using accepted `0610T007` / `0611T001` inputs and `0611T002` as private-order context only. Final recommendation is `replay_lifecycle_validation_gate_ready_for_qa`; this means only that the local validation / reconciliation gate is ready for QA/controller review. It does not authorize endpoint/source collector/runner/replay-live semantics, private/order/account/live data, queue priority or cancel-fill race metrics, exact queue position, PnL, maker viability, live/default-on/tiny-live readiness, deployment, or promotion.
- `0611T002`: Basis-positive private_order_response local artifact skeleton / validator is `已通过`. It implemented only a task-scoped local fixture/artifact schema, disk-only parser, fail-closed validator, CLI/help, tests, design note, local artifacts, and business report using accepted `0610T006` / `0611T001` design inputs. Final recommendation is `private_order_response_artifact_skeleton_ready_for_qa`, meaning only that the local artifact skeleton / validator is ready for controller review; it does not authorize endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T006`: Basis-positive private_order_response_source_line design-only contract is `已通过`. It defines only the private/order response artifact schema, response/reject/lifecycle label taxonomies, timestamp policy, terminal-state consistency, fail-closed validation gates, and overclaim rejection rules for `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle`, with final recommendation `private_order_response_contract_ready_for_qa`; this does not authorize source reader, collector, runner, private/order/account/live endpoint, user stream, signing/nonce handling, real execution metric, strategy/live/default-on/tiny-live, case-library/shadow decision, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T005`: Basis-positive execution source design decomposition / source-line routing contract is `已通过`. It split the seven execution gaps into four source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode. Final recommendation is `private_order_source_design_ready_next`, meaning only that a later separately dispatched design-only task may define the `private_order_response_source_line` contract; it does not authorize source implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T004`: Basis-positive fail-closed execution-evidence runner skeleton implementation is `已通过`. It emits exactly seven unavailable/proof-limited execution-gap status rows, preserves source policy and overclaim rejection, and final recommendation is `fail_closed_runner_skeleton_ready_for_qa`; this does not authorize real execution metrics, private/order/account/live data use, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T003`: Basis-positive execution-evidence source availability / runner implementation gate is `已通过`. It consumed QA-passed `0610T002` artifacts and classified all seven execution gaps as fail-closed placeholder only under current sources. Final recommendation is `runner_skeleton_ready_with_fail_closed_sources`, meaning only that a later separately scoped task may implement a fail-closed/read-only skeleton; it does not authorize execution metric proof claims, runner implementation inside T003, case-library, shadow decisions, strategy/private/order/live/default-on/tiny-live, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T002`: Basis-positive read-only execution-evidence runner contract/design is `已通过`. It defines the future read-only runner input contract, output schema, seven-gap metric mapping, validation plan, fail-closed/overclaim rules, and boundary validation while preserving private/order response as `forbidden_current_task / future_requires_separate_design`, replay/simulation as `supporting_regression_not_execution_proof`, and public proxy artifacts as design context only. Final recommendation is `read_only_execution_evidence_runner_design_ready`, meaning only that current contract/design artifacts are ready for QA/controller review; it does not indicate implementation readiness or authorize runner implementation, case-library, shadow decisions, strategy/private/order/live/default-on/tiny-live, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0609T010`: Basis-positive clean context read-only maker-viability proxy runner implementation is `已通过`. It produced `21270` proxy rows across `6` proxy metrics from `3545` source rows and final recommendation `read_only_proxy_evidence_ready_for_qa`. This remains proxy evidence only and does not authorize strategy, case-library, source-row catalog, shadow decisions, private/order endpoint use, live/default-on/tiny-live, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0609T008`: Basis-positive row-level read-only generator implementation is `已通过`. It produced `3545` row-level read-only research rows across `7` samples with `row_level_read_only_artifacts_ready_for_qa`. This remains observation-layer research only and does not authorize case-library implementation, source-row case catalog, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

Latest newly accepted source-line contracts:

- `0610T009`: Basis-positive economics_fee_rebate_source_line design-only contract is `已通过`. It defines only economics/fee/rebate/spread-capture artifact schema, fee/rebate settlement taxonomy, spread-capture taxonomy, maker/taker classification policy, currency conversion / tick-value policy, settlement timestamp policy, reconciliation boundary, validation gates, and overclaim reject rules for `fees_rebates_spread_capture`, with final recommendation `economics_fee_rebate_contract_ready_for_qa`; it explicitly rejects hypothetical spread, fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone as proof of fees/rebates/spread capture or PnL, and does not authorize economics endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T008`: Basis-positive account_inventory_source_line design-only contract is `已通过`. It defines only account/inventory artifact schema, inventory snapshot / transition taxonomy, conservation checks, reconciliation boundary, fail-closed gates, and overclaim rejection rules for `inventory_lifecycle`, with final recommendation `account_inventory_contract_ready_for_qa`; it explicitly states that order fills alone cannot prove inventory lifecycle and does not authorize account/private/order/live endpoints, source readers, collectors, runners, user streams, signing/nonce handling, real inventory metrics, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- `0610T007`: Basis-positive replay_lifecycle_semantics_source_line design-only contract is `已通过`. It defines only replay/live lifecycle event schema, queue semantics boundary, cancel/fill race event-ordering policy, timestamp policy, replay/live proof-limit rules, fail-closed validation gates, and overclaim rejection rules for `queue_priority` and `cancel_fill_race`, with final recommendation `replay_lifecycle_contract_ready_for_qa`; it does not authorize replay/live semantics, source readers, collectors, runners, private/order/account/live endpoints, user streams, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

Latest accepted synthesis gate:

- `0611T001`: Basis-positive execution source-line synthesis / implementation-readiness gate is `已通过`. It defines only source-line contract registry, implementation-readiness gate matrix, source dependency reconciliation matrix, forbidden overclaim matrix, next-task sequence, manifest, boundary validation, design document, and business report; it recommends future separately scoped implementation tasks only as `future_recommendation_only` and does not authorize endpoint/source reader/collector/runner implementation, real execution/economics metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof. Final recommendation is `source_line_synthesis_gate_ready_for_qa`, meaning only that the synthesis/gate design is ready for QA/controller review.

Prepared next task sequence:

- Superseded by `0616T001` for the `cross-exchange` branch: the `0615T009` Binance `BTCUSDT` small-cap live-test path must not be dispatched as the next cross-exchange task. `0615T008` is retained only as historical Binance live-risk design reference, not as authorization or gate for Binance-led Hyperliquid maker execution.

- `0615T006`: Basis-positive source-chain runner-consumption gate / synthesis design is the next formal task after `0615T005` QA passed. It must reconcile the accepted private order response, account inventory, economics fee/rebate, and replay lifecycle artifacts into a single runner-consumption contract. It may define source dependency matrices, timestamp reconciliation, cross-source identity / redaction policy, fail-closed runner input schema, proof-limit taxonomy, and QA gates. It must not implement a runner, consume live credentials, collect venue data, place/cancel/amend orders, change strategy behavior, run live, compute PnL, or claim maker viability.
- `0615T007`: Basis-positive proof-limited read-only execution evidence runner v1 implementation is allowed only after `0615T006` QA passes. It may consume only QA-accepted local/read-only artifacts and emit proof-limited evidence rows, unavailable rows, validation summaries, and overclaim rejections. It must fail closed when any required source line is missing, stale, inconsistent, or locally synthetic. It must not call endpoints, use credentials, run live, place/cancel/amend orders, default-enable strategy behavior, perform parameter search, or make PnL/promotion claims.
- `0615T008`: Small-cap live-test protocol / risk gate design is allowed only after `0615T007` QA passes and after the read-only runner output is accepted as mechanically valid. It is a design and dry-run gate, not a live run. It must define exact symbol/config, maximum notional, maximum order size, maximum position, maximum loss, duration, maker-only/post-only behavior, kill-switch criteria, cancel-all / shutdown proof, required preflight/replay/acceptance checks, logging/artifact requirements, credential handling boundary, operator approval steps, and rollback plan.
- `0615T009`: Small-cap live test and real-environment data collection is allowed only after `0615T008` QA passes and total control explicitly approves the live window. This is the fourth task in the sequence and is the first task that may open a small live test. Its scope is data collection under the `0615T008` caps and kill-switch rules, not strategy promotion. It must preserve start/stop markers, deployment manifest, raw market data, live audit, private/order/account/economics artifacts where authorized by the protocol, connector/bot logs, archive/checksum, cancel-all/shutdown evidence, and immediate post-run safety summary. It must not expand notional, relax risk caps, change strategy defaults, continue running after a kill condition, or claim PnL/maker viability from the run alone.
- `0615T010`: Post-live evidence analysis and decision gate is allowed only after `0615T009` artifacts are complete. It must analyze the live-derived source-path facts, compare them with replay/read-only runner expectations, classify gaps, and recommend one of: stop/revert, repeat with same caps, repair instrumentation, adjust risk gate, or prepare a later separately scoped experiment. It must not silently promote strategy behavior or scale capital.

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
- `0601T005`: Binance-led Hyperliquid read-only pricing-signal runner implementation is `已通过`.
- `0601T006`: Binance-led Hyperliquid public multi-sample robustness validation is `已通过` as public-only collection / initial synthetic-grid aggregate evidence, with the formal robustness decision superseded by `0604T003` canonical event-mode artifacts.
- `0604T001`: Hyperliquid event-driven horizon alignment and de-aliased verdict repair is `已通过`.
- `0604T002`: Event-driven horizon rerun and comparison for `xemm_0603_quiet_a/b/c` is `已通过`.
- `0604T003`: Canonical event-mode pricing-signal and robustness artifact repair is `已通过`.
- `0604T004`: Canonical event-mode evidence loader / validator foundation is `已通过`.
- `0604T005`: Canonical evidence source lock / guard hardening is `已通过`.
- `0604T006`: Canonical signal quality ranking evidence is accepted for downstream use after `0604T008` repaired the report bucket-consistency QA defect.
- `0604T007`: Canonical horizon / regime diagnostics is `已通过`.
- `0604T008`: T006 signal-ranking report bucket consistency repair is `已通过`; downstream work should use the T008-refreshed T006 artifacts.
- `0604T009`: Canonical signal + horizon/regime synthesis is `已通过`; it produced one read-only Milestone 3 executability candidate regime and did not authorize maker action, strategy, live/default-on/tiny-live, parameter search, or promotion.
- `0604T013`: Shutdown cancel bounded-wait implementation is historical intermediate work and is now `作废` as an open queue item because later proof-semantics tasks superseded its incomplete acknowledgement proof.
- `0604T015`: Shutdown cancel wait-result ambiguity diagnosis is historical intermediate work and is now `作废` as an open queue item; its findings were consumed by `0604T016` / `0605T001` / `0605T003`.
- `0604T016`: Shutdown cancel acknowledgement proof semantics fix is `未通过`; QA found the `PARTIALLY_FILLED` local active order proof bug. The defect was repaired by `0605T001`.
- `0605T001`: Shutdown terminal proof for partially filled local orders is `已通过`.
- `0605T002`: Shutdown local-absent proof and missing exchange open-orders reconciliation diagnosis is `已通过`.
- `0605T003`: Shutdown final proof hardening with exchange reconciliation and proof-level reporting is `已通过`.
- `0605T004`: Shutdown final proof no-order dry-run validation is `已通过` and is the latest accepted shutdown/live-safety fact source. It validates local fake no-network proof/log/audit observability only; it does not authorize live/default-on/tiny-live or promotion.
- `0608T002`: Milestone 3 read-only maker executability assessment for `regime_011_1000_spread_10_20_ticks` is `已通过`; final recommendation is `reject_not_maker_executable`, with non-zero multi-sample public proxy trigger mass but spread/adverse/post-only/latency proxy risk rejecting maker execution.
- `0608T003`: Regime 011 read-only directional momentum viability assessment is `已通过`; final recommendation is `reject_directional_edge_unstable`. It does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, or promotion.
- `0608T004`: Regime 011 feature-conditioned directional signal validity diagnosis is `已通过`; final recommendation is `watch_needs_contract_visibility_clarification` with `0` supported/watch-valid patterns and `21` invalid patterns. It does not authorize case-library, shadow decisions, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0608T005`: Regime 011 basis-context visibility / lineage diagnosis is `已通过`; final contract decision is `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only decision-time context only, retaining execution-PnL caveat and forbidding strategy/private/order/live/shadow/case-library/promotion.
- `0608T006`: Basis-positive independent robustness diagnosis outside Regime 011 is `待验收`. Final recommendation is `needs_more_samples`: broad basis-positive evidence has `2425` rows and strong positive persistence, but it is sample-concentrated (`max_sample_row_share=0.67917526`) and the conservative cost/tail proxy rejects on p95 wrong-way loss. It does not authorize strategy, case-library, shadow decisions, private/order endpoint use, live/default-on/tiny-live, parameter search, or promotion.

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
28. `0601T005` passed QA as the read-only Binance-led Hyperliquid pricing-signal runner implementation. It consumes accepted local `0601T002/0601T003/0601T004` artifacts, generates `21541` pricing signal rows from `3596` primary rows, enforces the four-feature `0601T004` Binance allowlist, reports nominal horizon plus effective future age, and recommends `keep_for_read_only_research` with `single_public_sample_caveat=true`. It does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.
29. `0601T006` passed QA as the public-only collection and initial multi-sample robustness aggregation step. It collected/processed `xemm_0603_quiet_b/c`, incorporated the existing reference and quiet samples, and produced initial synthetic-grid aggregate artifacts, but those ordinary synthetic fixed-grid outputs are no longer the formal robustness decision source.
30. `0604T001` passed QA. It adds Hyperliquid event-driven decision rows and de-aliased future-row-delta diagnostics so `100/250/500ms` nominal horizons are not over-counted as independent evidence.
31. `0604T002` passed QA. It reran `xemm_0603_quiet_a/b/c` through the event-mode local chain and showed event mode fixes artifact-level fixed-grid aliasing while public `l2Book` cadence still leaves short-horizon independence caveats.
32. `0604T003` passed QA and is the current formal evidence source for Binance-led Hyperliquid pricing-signal robustness. Future robustness decisions must use canonical event-mode artifacts; ordinary synthetic fixed-grid artifacts remain backward-compatible diagnostics only.
33. `0604T009` passed QA. It defines exactly one read-only Milestone 3 executability candidate, `regime_011_1000_spread_10_20_ticks`, anchored by `binance_mid_move_ticks_from_prev` at the 1000ms horizon. This candidate is an input for later executability assessment only and does not authorize maker side, quote behavior, order behavior, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
34. `0604T013` is no longer an open queue item. It introduced bounded shutdown cancel waiting, but later review proved bounded wait alone did not establish cancel/final-state proof; the open workflow path moved through `0604T015` / `0604T016` and then `0605T001-0605T004`.
35. `0604T015` is no longer an open queue item. It reproduced the wait-result ambiguity and missing final proof gap; its diagnostic findings have been consumed by the later repair chain.
36. `0604T016` remains a historical `未通过` task. QA accepted the wait-result classification direction but rejected the terminal proof because `PARTIALLY_FILLED` active local orders could be misclassified.
37. `0605T001` passed QA and repaired the `PARTIALLY_FILLED` local active order proof bug.
38. `0605T002` passed QA and established that local-absent proof is only local proof, not exchange-side no-open-order proof.
39. `0605T003` passed QA and added exchange open-orders reconciliation / final proof level / summary and audit-tail proof semantics without changing strategy quote/submit/cancel behavior, bindings, Rust, connector, production config, live/default-on/tiny-live, or promotion boundary.
40. `0605T004` passed QA and is the latest accepted shutdown/live-safety fact source. It ran only local fake HBT / fake REST no-order no-network dry-run validation. It proves proof/log/audit artifact observability, not a real exchange shutdown run.
41. `0608T001` is the workflow housekeeping task that reconciles stale controller state. After this cleanup, total control must not continue from `0604T015` as an active task.
42. `0608T002` passed QA as the read-only maker executability assessment for `regime_011_1000_spread_10_20_ticks`. It rejected maker executability with final recommendation `reject_not_maker_executable`; this does not authorize maker case-library, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, shadow decisions, or promotion.
43. `0608T003` passed QA as the read-only directional momentum viability assessment for the same regime. It consumed T002's accepted maker rejection, resolved row-level canonical sample files from `multi_sample_manifest.json` `samples[].pricing_signal_rows`, and produced final recommendation `reject_directional_edge_unstable`.
44. `0608T004` passed QA as the read-only feature-conditioned validity diagnosis for Regime 011. It found no valid supported/watch pattern: `0` supported, `0` watch-valid, `21` invalid. The only non-tail/non-redundancy-looking strong local pattern was `context_basis_mid_ticks > 0`, which remained blocked in T004 by the prior contract visibility caveat.
45. `0608T005` passed QA as the read-only basis-context visibility / lineage diagnosis. It confirmed `context_basis_mid_ticks > 0` is formula-derived from decision-time Binance and Hyperliquid mids, as-of clean, not single-sample dominated, and persistent at `1000/5000/10000ms`; the field may be treated as context-only supported in later read-only research while retaining execution-PnL caveat.
46. `0608T006` business execution is complete and awaiting QA. It intentionally leaves the Regime 011 shell and evaluates `context_basis_mid_ticks > 0` independently across spread, join-age, volatility, and Hyperliquid book-state strata. Final recommendation is `needs_more_samples` because evidence is sample-concentrated, with additional cost/tail caution from p95 wrong-way loss.

## High-Confidence Regime Maker Research Plan

This is a controller-level research roadmap, not a formal task dispatch. It does not authorize new task files, strategy implementation, private/order endpoints, parameter search, live/default-on/tiny-live, or promotion.

Current development sequencing:

- `0604T004` passed QA and is the shared canonical event-mode evidence loader / validator foundation for parallel Milestone 0 / Milestone 1 workers.
- Its only purpose is to build the shared canonical event-mode evidence loader / validator foundation that later guard, ranking, and horizon/regime diagnostic runners must reuse.
- The parallel worker split is unblocked by `0604T004` QA, while downstream work must keep using the canonical-only sample set and exclude synthetic fixed-grid diagnostics from formal evidence.
- Dispatched parallel read-only workers:
  - `0604T005`: Milestone 0 canonical evidence source lock / guard hardening.
- `0604T006`: Milestone 1 canonical signal quality ranking completed; its original QA defect was report wording only and was fixed by `0604T008`.
- `0604T007`: Milestone 1 canonical horizon / regime diagnostics passed QA.
- `0604T008`: Narrow T006 report-consistency repair passed QA; use refreshed `local_live_analysis/canonical_signal_quality_ranking_0604T006/**` artifacts for later synthesis.
- `0604T009`: Milestone 2 canonical signal + horizon/regime synthesis passed QA. The only promoted read-only candidate for later Milestone 3 assessment is `regime_011_1000_spread_10_20_ticks`.
- `0608T002`: Milestone 3 read-only maker executability assessment passed QA with `reject_not_maker_executable`.
- `0608T003`: Follow-up read-only directional momentum viability assessment passed QA with `reject_directional_edge_unstable`.
- `0608T004`: Feature-conditioned validity diagnosis passed QA with `watch_needs_contract_visibility_clarification` and no supported case-design pattern.
- `0608T005`: Basis context visibility / lineage diagnosis passed QA with `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only context only.
- `0608T006`: Business execution complete; awaiting QA as a read-only basis-positive robustness diagnosis outside Regime 011.

Goal:

- Use `0604T003` canonical event-mode evidence to find a small number of high-confidence, low-trigger, explainable maker regimes.
- Prove first that a regime has stable decision-time-visible pricing signal quality, then separately prove that it is suitable for maker execution.
- Avoid turning broad directional prediction into a maker strategy before fill probability, adverse selection, spread capture, churn, post-only safety, and inventory effects are understood.

Milestone 0 - Lock Evidence Source:

- Formal Binance-led Hyperliquid robustness evidence must use `0604T003` canonical event-mode artifacts.
- Ordinary synthetic fixed-grid artifacts remain diagnostic-only.
- Prioritize `500ms+` horizons, especially `1000ms+`; treat `100/250ms` as weakly independent because public `l2Book` cadence can still alias short horizons.

Acceptance line:

- Every later analysis traces back to event-mode samples.
- No fixed-grid short-horizon result is treated as canonical independent evidence.

Milestone 1 - Signal Quality Ranking:

- Rank the four accepted Binance lead allowlist features by direction consistency, effect size, mean absolute correlation, sample count, independent future-row-delta count, and sample concentration.
- Initial interpretation from `0604T003`:
  - `binance_mid_move_ticks_from_prev` is the most stable global signal candidate.
  - `binance_top5_imbalance` is a strong book-pressure candidate.
  - `binance_top5_bid_qty` is useful as liquidity/context.
  - `binance_microprice_minus_mid_ticks` is more regime-dependent and should not be treated as a simple global signal.

Acceptance line:

- Keep only signals that are stable across multiple canonical samples and do not rely mainly on `100/250ms` evidence.
- Maintain an explicit reject/watch list.

Milestone 2 - High-Confidence Regime Definition:

- Define candidate regimes using only decision-time-visible fields.
- Prefer regimes where Binance lead impulse and book pressure agree, Hyperliquid has not fully reacted, join age is fresh, spread leaves maker capture room, basis/dislocation is not uncontrolled, and volatility is high enough to matter without becoming pure noise.
- Prepared next formal task: `0604T009` should synthesize `0604T006/T008` signal ranking and `0604T007` horizon/regime diagnostics into candidate/watch/reject regime definitions only. It must not output maker side, quote behavior, order behavior, strategy action, live/default-on/tiny-live, or promotion.

Acceptance line:

- Each regime has multi-sample support, enough row count, stable future-move direction, and a clear decision-time-visible definition.

Milestone 3 - Maker Executability Assessment:

- Test whether each high-confidence pricing regime is actually suitable for maker action.
- Current input is limited to `regime_011_1000_spread_10_20_ticks` from `0604T009`.
- Current maker executability task `0608T002` has passed QA with `reject_not_maker_executable`.
- Current follow-up directional task `0608T003` has passed QA with `reject_directional_edge_unstable`.
- Follow-up feature validity task `0608T004` has passed QA with no supported/watch-valid feature-conditioned case pattern.
- Follow-up basis visibility task `0608T005` has passed QA and upgrades `context_basis_mid_ticks > 0` only to read-only context-supported status, not to maker/directional executability.
- Regime 011 should not progress to maker case library, directional case library, shadow decisions, strategy implementation, live/default-on/tiny-live, parameter search, or promotion.
- The next formal task is `0608T006`, a read-only basis-positive robustness diagnosis outside the Regime 011 shell, checking cross spread/join-age/volatility/venue-state stability and cost/tail proxy.
- Required questions:
  - Which side, if any, should be quoted?
  - Is the regime better for joining, stepping back, reducing inventory, or avoiding add-side exposure?
  - Does expected movement exceed adverse-selection risk after fill?
  - Is spread wide enough to matter after fees/rebates and slippage assumptions?
  - Is signal persistence long enough relative to quote/cancel latency?
  - Does the regime create unacceptable quote churn or post-only risk?

Acceptance line:

- A regime cannot progress on future mid prediction alone.
- It must have a plausible maker-side spread capture / adverse-selection / churn / post-only story.

Milestone 4 - Case Library:

- Convert surviving regimes into a small case library rather than a broad parameter grid.
- Each case should include:
  - `case_id`
  - entry conditions
  - maker action hypothesis
  - expected edge source
  - reject conditions
  - required evidence still missing

Acceptance line:

- Prefer 2-4 explainable cases.
- Every case has explicit reject conditions and can explain why it is maker-suitable rather than merely directional.

Milestone 5 - Shadow Decision Verification:

- Generate read-only shadow decisions over existing artifacts: no orders, no private endpoints, no strategy live.
- Measure trigger frequency, side distribution, expected quote churn, stale-decision rate, spread bucket, post-only risk proxy, future markout, and case overlap/conflict.

Acceptance line:

- Trigger frequency is low enough to preserve quality.
- Shadow decisions do not depend on high churn or stale/aliased evidence.
- Case conflicts are explicit and resolvable.

Milestone 6 - Execution Evidence Gap:

- Decide what cannot be proven with public-only data.
- Explicitly list whether later work needs Hyperliquid private/order lifecycle evidence, real maker fill/cancel/reject/post-only data, queue/fill proxy, fee/rebate modeling, or inventory-cycle analysis.

Acceptance line:

- If execution gaps remain large, continue read-only research.
- If pricing and maker-executability evidence are both strong, the next step may only be an execution-evidence design discussion, not strategy implementation or live promotion.

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


## Cross-Exchange T011 Robustness Route

`0709T003 / T011-MULTI-WINDOW-ROBUSTNESS-SYNTHESIS` business execution generated the T011 route decision and is awaiting QA.

Current synthesis facts:

- accepted windows: `4` (`0708T001`, `0709T001_window_01`, `0709T001_window_02`, `0709T001_window_03`)
- lifecycle distribution: `submitted_resting_no_fill=3`, `submitted_rejected=1`
- safety invariant: `pass=4`
- replay overall acceptance: `pass=4`
- economics support: `no_fill_fail_closed=4`
- final recommendation enum: `route_to_quote_fill_probability_evidence`

If QA passes, the T011 auto-loop is complete and the next formal task should target quote/fill probability evidence only. This route explicitly does not claim stable PnL, maker viability, T012 readiness, promotion, final MVP pass, or live expansion.


## Cross-Exchange T011 Accepted Route

`0709T003` QA passed. The T011 auto-loop is complete.

Accepted route:

- `route_to_quote_fill_probability_evidence`

Controller review clarification:

- Treat T011 as multi-window live artifact/lifecycle/safety and non-optimistic consistency evidence, not as full replay-engine regression proof.
- Treat the four-row synthesis as three newly collected `0709T001` live windows plus one prior `0708T001` QA reference.
- `0709T002` did not newly reparse the raw `0708T001` artifact and should not be cited as four equivalent raw-window replay reruns.

Allowed next planning direction:

- Create a separate formal task to gather or analyze quote/fill probability evidence under conservative boundaries.

Still not authorized:

- T012, live expansion, threshold changes, quote-envelope changes, order-size/max-submission expansion, stable PnL, maker viability, promotion, or final MVP pass.


## Cross-Exchange 0710T001 Quote/Fill Evidence

`0710T001 / T011-QUOTE-FILL-PROBABILITY-EVIDENCE` business execution generated the quote/fill evidence route and is awaiting QA.

Current facts:

- attempt rows: `5` (`0708T001` prior QA reference plus four `0709T001` live artifact attempts)
- post-only rejects: `2`, both classified as not fill-probability samples
- resting no-fill rows: `3`, including one prior reference with missing local quote/fill artifact and two short-hold censored T011 rows
- depth proxy: present for `4`, missing for the prior reference
- trade-through/depletion: current T011 rows have rolling decision-time proxies, but no full resting-interval reconstruction
- final recommendation enum: `route_to_public_flow_artifact_repair`

If QA passes, the next formal task should repair/design public-flow interval artifacts before any quote/fill probability claim. Still not authorized: live expansion, threshold/quote-envelope changes, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.


## Cross-Exchange 0710T001 Accepted Route

`0710T001` QA passed.

Accepted route:

- `route_to_public_flow_artifact_repair`

Allowed next planning direction:

- Create a separate formal task to repair/design resting-interval public flow and depletion artifacts for quote/fill probability evidence.
- Formal task file created:
  - `.workflow/tasks/0712T001.md`
  - Status: `待执行`

Still not authorized:

- T012, live expansion, threshold changes, quote-envelope changes, fee/PnL calibration, maker viability, promotion, or final MVP pass.


## 2026-07-19 Principal Alignment T004

`0719T004 / REDACTION-SAFE-REFERENCE-IDENTITY-STRICT-ATTEMPT-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `a739a78cfb4cc58a23644e67644a4289ad5789af`
- cancel reference schema: `per_attempt_reference_cancel_reconciliation_v2`
- persisted oid/cloid identity: type-bound SHA-256 opaque tokens
- strict attempt identity: positive integer or canonical positive-integer string only
- actual standalone and two-sided manager artifacts independently rebuild exact after redaction
- focused regression: `173 passed`
- full Hyperliquid regression: `526 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T004. Multi-level remains default-off until a separately accepted real single-level two-sided lifecycle exists.


## 2026-07-19 Principal Alignment T004 QA

`0719T004` QA is `未通过`.

Accepted sub-results:

- redaction-safe oid/cloid token contract
- producer-written standalone/manager exact rebuild
- ordinary malformed attempt matrix and fractional alias fail-closed
- all-token/raw-proof helper independence

Remaining blockers:

- malformed cancel statuses such as `{"success": false}` are still treated as authoritative success
- extremely long digit-only attempt strings raise rather than fail closed

The next formal task is offline-only. No live task may start until strict cancel-success semantics and bounded attempt parsing are independently QA accepted.


## 2026-07-19 Principal Alignment T005

`0719T005 / STRICT-CANCEL-SUCCESS-BOUNDED-ATTEMPT-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `19e4b4a7e740a01763fcaf67df28ef3283abbabe`
- one cancel action requires exactly one explicit success status
- false/null/zero/empty/container/extra-key/multi-status payloads fail closed
- reference attempts are bounded to `1..2147483647`
- T004 redaction-safe token and actual writer exact rebuild remain intact
- focused regression: `271 passed`
- full Hyperliquid regression: `591 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T005.


## 2026-07-19 Principal Alignment T005 QA Accepted

`0719T005` QA passed.

Accepted facts:

- strict one-status cancel success semantics
- malformed/falsey/container/multi-status response fail-closed
- bounded attempt identity `1..2147483647`
- no exception on oversized canonical digit strings
- T004 redaction-safe token and actual writer exact reconstruction preserved
- focused `271 passed`; full Hyperliquid `591 passed`

The next formal route is an offline exact-orchestrator and two-sided manager evidence-contract task. Principal Task 12 and Task 10 remain open until a later accepted live lifecycle.


## 2026-07-19 Principal Alignment T006

`0719T006 / EXACT-TWO-SIDED-MANAGER-EVIDENCE-CONTRACT` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `338975f53ccdb351912b03a54cb58e246c01f8a1`
- exact runs require an explicit `legacy-single-order` or `two-sided-manager` profile
- the new two-sided profile fixes the Binance-edge manager mode and conservative live envelope
- manager artifacts contain exactly one buy and one sell lifecycle with distinct attempt/status/reference/cancel evidence
- actual exchange order responses are persisted per side after redaction
- Task 12 acceptance requires the exact two-sided profile and rejects one-sided, duplicate, aggregate, stale, mismatched and forged evidence
- actual writer nominal integration passes
- full Hyperliquid regression: `605 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T006. Task 12 and Task 10 remain open until the later bounded live lifecycle is accepted.


## 2026-07-19 Principal Alignment T006 QA

`0719T006` QA is `未通过`.

Remaining blockers:

- raw order response is not independently bound to side/attempt/reference evidence
- fill lifecycle can bypass strict per-reference terminal proof
- exact profile can execute one-sided after inventory-based quote pruning
- duplicate runner flags can override the values acceptance reads
- actual producer integration does not yet run manager watcher output through acceptance

The next formal task is offline-only. No live task may start until all five gaps are repaired and independently QA accepted.

Formal task dispatched:

- `.workflow/tasks/0719T007.md`
- Status: `待执行`
- Scope: raw response binding, unified terminal proof, pre-submit exact side gate, canonical argv and real manager-watcher acceptance integration.


## 2026-07-19 Principal Alignment T007

`0719T007 / RAW-LIFECYCLE-TERMINAL-COMMAND-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `5239af62d67381c5b3584c0873b58e6a0b246cfe`
- raw order responses are independently parsed and bound to side, canonical attempt, intent and terminal reference
- fill and no-fill branches share one per-attempt cancel-or-complete-fill terminal contract
- exact manager mode rejects a non-`{buy,sell}` quote set before any order endpoint call
- runner argv is preflight-sealed, unique and canonical, with explicit task/window run identity
- real manager-watcher output is consumed from canonical `window_01` artifacts by Task 12 acceptance
- focused acceptance/watcher/orchestrator/manager regressions pass
- full Hyperliquid regression: `631 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T007.


## 2026-07-19 Principal Alignment T007 QA

`0719T007` QA is `未通过`.

Remaining blockers:

- full-fill terminal proof is not independently rebuilt from raw user-fill pullbacks
- fill attribution accepts a correct oid plus conflicting cloid instead of enforcing all-token identity
- argparse abbreviations and unbound executable/output paths can change the parsed command while acceptance passes

The next formal task is offline-only. No live task may start until raw fill reconstruction, all-token identity and canonical executable/output binding are independently QA accepted.

Formal task dispatched:

- `.workflow/tasks/0719T008.md`
- Status: `待执行`
- Scope: raw fill reconstruction, all-token fill identity, argparse abbreviation shutdown and runtime-provenance command/path binding.


## 2026-07-19 Principal Alignment T008

`0719T008 / RAW-FILL-ALL-TOKEN-CANONICAL-PATH-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `57c4d9346836c5ae73e4c1358fc51a17b469c7dc`
- raw user-fill pullbacks are versioned, redaction-safe and include the mark/fee context consumed by the producer
- Task 12 acceptance independently rebuilds fill identity, quantity, role and attempt binding from raw pullbacks
- raw reconstruction exact-compares ledger, attribution, role and summary artifacts
- producer and acceptance enforce all-token same-reference semantics
- orchestrator/watcher disable argparse abbreviations
- runtime provenance seals exact command, executable, script, run root and canonical output path
- focused regression: `269 passed`
- full Hyperliquid regression: `651 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T008.


## 2026-07-19 Principal Alignment T008 QA

`0719T008` QA is `未通过`.

Accepted sub-results:

- missing/empty raw pullbacks fail closed
- producer and acceptance enforce oid/cloid all-token identity
- argparse abbreviations and noncanonical argv fail closed
- Python, watcher entrypoint, physical run root and output path are bound
- fresh real manager-watcher zero-fill producer output passes acceptance

Remaining blockers:

- synchronized raw/ledger/role/summary mutation can use impossible buy-above-limit or sell-below-limit fill prices as full-fill terminal proof
- acceptance trusts the checksum verification summary instead of independently recalculating the terminal SHA-256 manifest

The next formal task is offline-only. No live task may start until side-aware fill-price semantics and independent terminal checksum verification are independently QA accepted.

Formal task dispatched:

- `.workflow/tasks/0719T009.md`
- Status: `待执行`
- Scope: side-aware fill-limit semantics and independent terminal SHA-256 manifest verification.


## 2026-07-19 Principal Alignment T009

`0719T009 / FILL-LIMIT-TERMINAL-CHECKSUM-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `6fd4089310a5dbf9694c977808081494ec63e3ac`
- producer reference-bound fills require exact symbol/side and side-aware limit-price semantics
- Task 12 acceptance independently enforces the same raw fill execution contract
- full-fill terminal proof rechecks intent price semantics
- acceptance independently parses and recomputes the current terminal SHA-256 manifest
- stored checksum summary must equal independent recomputation
- synchronized impossible-price and stale-checksum attacks fail closed
- focused regression: `284 passed`
- full Hyperliquid regression: `666 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T009.


## 2026-07-19 Principal Alignment T009 QA

`0719T009` QA is `未通过`.

Accepted sub-results:

- side-aware buy/sell limit checks
- raw and full-fill terminal price validation
- independent current run-root SHA-256 recomputation
- ordinary malformed/path/digest/file-set manifest failures
- fresh real manager-watcher zero-fill producer acceptance

Remaining blockers:

- `Close Long` and `Close Short` fallback direction semantics are reversed and can produce forged full-fill terminal proof
- blank terminal manifest records are ignored instead of rejected as malformed

The next formal task is offline-only. No live task may start until exact Hyperliquid direction mapping and blank-manifest-line fail-closed behavior are independently QA accepted.


## 2026-07-19 Principal Alignment T010 Dispatched

Formal task dispatched:

- `.workflow/tasks/0719T010.md`
- Status: `待执行`
- Scope: exact Hyperliquid fill-direction mapping, explicit side/direction agreement and blank terminal-manifest record rejection.

This task is offline-only. No live task may start before independent QA accepts T010.


## 2026-07-19 Principal Alignment T010

`0719T010 / FILL-DIRECTION-MANIFEST-BLANK-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `542319128b661058fee111a9a7886c534fb12fcb`
- producer uses exact Hyperliquid direction semantics and requires explicit side/direction agreement
- invalid, unknown or conflicting direction evidence creates producer fail-closed reasons
- acceptance independently enforces the same direction contract
- synchronized and resealed conflict evidence remains blocked
- terminal SHA-256 verification rejects explicit blank and whitespace-only records
- focused regression: `319 passed` across fill, acceptance and manager/watcher/orchestrator suites
- fresh actual two-sided zero-fill producer fixture: `1 passed`
- full Hyperliquid regression: `689 passed`
- live/private/order/cancel/network/remote actions: none

No live task may start before independent QA accepts T010.


## 2026-07-19 Principal Alignment T010 QA

`0719T010` QA is `已通过`.

Accepted results:

- exact `Open Long`/`Close Short` buy and `Open Short`/`Close Long` sell semantics
- explicit side/direction agreement and unknown/conflict fail-closed behavior
- synchronized resealed direction attacks remain blocked
- blank, whitespace and CRLF manifest records fail closed
- normal terminal newline and fresh two-sided zero-fill producer remain accepted
- full Hyperliquid regression: `689 passed`

The offline execution-safety and evidence-integrity gate is clear for one new isolated single-level two-sided bounded live task. Multi-level, dynamic spread, fill feedback and inventory skew remain locked.


## 2026-07-19 Principal Alignment T011 Dispatched

Formal task dispatched:

- `.workflow/tasks/0719T011.md`
- Status: `待执行`
- Exact source: `d8e22c2d9288fef86707d9b26f7791d7d8711c09`
- Scope: one isolated `two-sided-manager` Hyperliquid BTC window with Binance edge-gate, at most `900s`, `0.005 BTC/order`, `0.01 BTC` position, `1 USDC` loss and `2` submissions.

No second window or strategy-variable activation is allowed. Multi-level remains locked until independent QA accepts the live lifecycle.


## 2026-07-19 Principal Alignment T011

`0719T011 / SINGLE-LEVEL-TWO-SIDED-BOUNDED-LIVE` business execution is complete and awaiting QA.

Current facts:

- exact source commit: `d8e22c2d9288fef86707d9b26f7791d7d8711c09`
- exactly one `900.045779s` bounded window ran
- source provenance `62/62`, terminal checksum `106/106`
- public evaluations `2564`, edge-gate evaluations `10`, edge-gate passes `0`
- one trigger reached immediate reprice guard, which failed closed
- stop condition: `edge_gate_no_fresh_sufficient_signal`
- submissions/fills/order calls/cancel calls: `0/0/0/0`
- final open orders `0`, post BTC position `0.0`, child reaped cleanly
- same-window acceptance blocked; no lifecycle claim and no multi-level unlock
- no second live window ran

Independent QA is the current node.


## 2026-07-19 Principal Alignment T011 QA

`0719T011` QA is `未通过`.

Accepted sub-results:

- exactly one `900.045779s` live window
- exact source `62/62` and terminal checksum `106/106`
- exact envelope, activation boundary, empty final open orders and zero BTC position
- clean child reap, healthy writer and clear kill-switch
- same-window acceptance correctly failed closed
- remote/local path mismatches are a portability gap, not source mutation

Remaining blockers:

- top-level trigger and stop-condition summaries do not match the `26` trigger rows
- private read-only endpoint activity is conflated with order/cancel endpoint activity
- acceptance does not independently reconstruct those summaries from row evidence
- remote canonical paths are incorrectly compared to the local pullback root
- no submitted two-sided lifecycle was observed

The next formal task is offline-only. No new live window may start until evidence-summary reconstruction and path portability are independently QA accepted. Multi-level and every adaptive controller remain locked.


## 2026-07-19 Principal Alignment T012 Dispatched

Formal task dispatched:

- `.workflow/tasks/0719T012.md`
- Status: `待执行`
- Scope: row-derived trigger/blocker/endpoint summaries, candidate-attempt-submission cardinality and remote/local provenance path portability.

This task is offline-only. T011 must remain blocked for zero submissions and missing lifecycle after path portability is repaired. No new live or multi-level task may start before independent QA accepts T012.


## 2026-07-19 Principal Alignment T012

`0719T012 / LIVE-EVIDENCE-SUMMARY-PATH-PORTABILITY-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `ab8ce0e6e803fd82ded91d054371bd0a2c3f6905`
- producer writes exact row-derived decision, blocker, endpoint and candidate/attempt/submission summaries
- acceptance independently reconstructs strict booleans, statuses, identities, joins and reason atoms
- remote provenance is anchored by an external canonical expected run root
- local pullback source/checksum/current-byte verification remains independent and strict
- synchronized remote-root rewrites and summary mutations fail closed
- T011 re-acceptance provenance is `112/112 pass`
- T011 exact reconstruction is `2564` evaluations, `26` triggers, `5` anti-drift blocks, `11` guard failures and `10` edge blocks
- T011 has `21` candidate attempt rows, `1` manager-attempt identity and `0` submissions
- T011 remains blocked for the real edge/guard/lifecycle failures
- focused regression: `240 passed`
- full Hyperliquid regression: `699 passed`
- live/private/order/cancel/network/remote actions: none

No new live task may start before independent QA accepts T012.


## 2026-07-19 Principal Alignment T012 QA Not Accepted

`0719T012` QA is `未通过`.

Accepted sub-results:

- remote/local path portability is repaired without weakening source or checksum checks
- T011 row-derived counts reconstruct exactly and provenance is `112/112 pass`
- candidate rows, manager-attempt identity, submissions and endpoint classes are separated
- T011 remains correctly blocked for zero submissions and missing lifecycle

Remaining strict-join blockers:

- malformed event or attempt identities can skip validation
- submitted attempts are not required to join an order-authorized trigger
- trigger, anti-drift, immediate-guard and edge status/reason evidence is not exact-joined bidirectionally

The next formal task is offline-only and limited to strict evidence joins. No new live window may start before independent QA accepts that repair.


## 2026-07-20 Principal Alignment T013 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T013.md`
- Status: `待执行`
- Scope: canonical event/attempt parsing, unique causal joins, order-authorization binding and cross-matrix status/reason exact comparison.

The task is offline-only. T011 legacy evidence may use only deterministic ambiguity-rejecting raw-field bridges, and no new live window may start before independent QA accepts T013.


## 2026-07-20 Principal Alignment T013

`0720T013 / DECISION-EVIDENCE-STRICT-JOIN-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `5b482929a09c4402b5682fc9c7c130aafbb25977`
- new guard evidence persists direct event identity and no longer writes anonymous placeholder rows
- all trigger, anti-drift, guard, edge and attempt identities parse fail-closed
- status/reason/phase joins are bidirectional and exact
- every submitted attempt must join one order-authorized trigger
- legacy guard bridging is explicit and bound only to exact T011 task/source
- synchronized malformed identity, schema downgrade, noncanonical key, zero authorization and reason-drift attacks are blocked
- focused regression: `243 passed`
- full Hyperliquid regression: `706 passed`
- T011 replay remains `112/112` provenance pass with `validation_reasons=[]` and zero-submission lifecycle blocked
- live/private/order/cancel/network/remote actions: none

No new live task may start before independent QA accepts T013.


## 2026-07-20 Principal Alignment T013 QA Not Accepted

`0720T013` QA is `未通过`.

Accepted sub-results:

- ordinary malformed/missing event and attempt identities fail closed
- legacy guard bridging is explicit and exact T011 task/source bound
- zero authorization plus two submissions is rejected
- cross-matrix status/reason/phase joins and two-sided shared-event identities pass focused QA

Remaining blocker:

- an attempt-key suffix containing thousands of digits reaches an unbounded `int()` conversion and can crash acceptance instead of returning a blocked manifest

The next formal task is a minimal offline bounded-parser repair. No new live window may start before independent QA accepts it.


## 2026-07-20 Principal Alignment T014 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T014.md`
- Status: `待执行`
- Scope: bounded attempt-key suffix parsing and oversized/out-of-range fail-closed regression only.

The task is offline-only. All accepted T013 identity, authorization, causal-join and legacy-bridge behavior must remain unchanged.


## 2026-07-20 Principal Alignment T014

`0720T014 / ATTEMPT-KEY-BOUNDED-PARSER-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `33975476d629e1abc1f62324276785ac0908093f`
- attempt-key suffixes now reuse the bounded standalone attempt parser
- 5000-digit and out-of-range suffixes return blocked manifests with `attempt_key_mismatch`
- focused regression: `245 passed`
- full Hyperliquid regression: `708 passed`
- T011 replay remains `112/112` provenance pass with `validation_reasons=[]` and lifecycle blocked
- live/private/order/cancel/network/remote actions: none

No new live task may start before independent QA accepts T014.


## 2026-07-20 Principal Alignment T014 QA Accepted

`0720T014 / ATTEMPT-KEY-BOUNDED-PARSER-REPAIR` is `已通过`.

Accepted facts:

- 5000-digit and out-of-range attempt-key suffixes do not raise;
- hostile suffixes produce blocked manifests with `attempt_key_mismatch`;
- exact task/window/attempt equality and T013 strict causal joins remain intact;
- focused QA `245 passed`, full Hyperliquid QA `708 passed`;
- exact T011 legacy replay remains provenance `112/112`, `validation_reasons=[]` and lifecycle blocked.

The parser blocker is closed. Before the next live window, the exact-envelope duration contract must be aligned from its historical `900s` hard limit to the standing authorization maximum `1800s` in a separate offline-only task. That task may change duration validation only; strategy formulas, signal freshness, edge thresholds, risk caps and activation remain unchanged.


## 2026-07-20 Principal Alignment T015 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T015.md`
- Status: `待执行`
- Scope: externally bound exact window duration and standing `1800s` maximum only.

The task is offline-only. It must preserve historical `900s` replay compatibility and cannot change strategy formulas, signal freshness, edge thresholds, risk caps, activation or manager behavior.


## 2026-07-20 Principal Alignment T015

`0720T015 / EXACT-ENVELOPE-DURATION-CONTRACT-ALIGNMENT` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `35ceea2a5ca861a9329e384a2faeca11d9615cf7`
- orchestrator exact-envelope cap is aligned to the standing `1800s` maximum
- acceptance binds preflight/runtime duration to external `expected_window_seconds`
- historical default remains `900s`; future tasks can explicitly select `1800s`
- `1800s` preflight passes with zero watcher/private/order/cancel activity
- `1800.001s` fails before output
- focused `171 passed`; full Hyperliquid `712 passed`
- T011 replay remains provenance `112/112`, `validation_reasons=[]` and lifecycle blocked

No live task may start before independent QA accepts T015.


## 2026-07-20 Principal Alignment T015 QA Accepted

`0720T015 / EXACT-ENVELOPE-DURATION-CONTRACT-ALIGNMENT` is `已通过`.

Accepted facts:

- exact two-sided `1800s` preflight passes without starting watcher/private/order work
- `1800.001s`, zero and non-finite durations fail closed before output
- acceptance binds external expected duration to preflight and runtime argv
- `1800s` positive and bidirectional mismatch fixtures behave correctly
- focused QA `171 passed`; full Hyperliquid QA `712 passed`
- T011 expected `900s` replay remains byte-exact, provenance `112/112`, `validation_reasons=[]` and lifecycle blocked

The next formal task may run one isolated `1800s` single-level two-sided manager tiny-live window under the unchanged standing risk/submission envelope. Duration is the only primary variable changed from T011.


## 2026-07-20 Principal Alignment T016 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T016.md`
- Status: `待执行`
- Exact source: `657bde21f590829fbe08e2fcba3126cec4010e9f`
- Scope: one isolated `1800s` single-level two-sided manager live window.

Only duration changes from T011. Fixed spread, source freshness, edge threshold, quote size, risk caps, submission budget and all adaptive/multi-level activation remain unchanged.


## 2026-07-20 Principal Alignment T016

`0720T016 / SINGLE-LEVEL-TWO-SIDED-1800S-BOUNDED-LIVE` business execution is complete and awaiting QA.

Current facts:

- exact source: `657bde21f590829fbe08e2fcba3126cec4010e9f`
- exactly one window started; it stopped fail-closed after `199.099722s`
- buy and sell each reached a real post-only endpoint with `0.005 BTC`
- submissions/fills/post-only rejects: `2/0/0`
- buy cancel has authoritative success; sell cancel is `cancel_response_status_invalid`
- cancel references proven: `1/2`
- first stop condition: `fill_reconciliation_required_no_fill_unproven`
- independent post-child and post-live proofs both show open orders `0` and BTC position `0.0`
- source provenance `62/62`, terminal checksum `104/104`, child reaped with `rc=0`
- same-window acceptance: provenance `112/112`, config `72/72`, mechanism/evidence integrity `fail`
- final live status retains one unknown sell order while independent terminal proofs show zero
- decision replay records four candidate-attempt rows against the exact-two-primary-attempt contract
- no second live window or adaptive/multi-level activation occurred

Independent QA is the current node. Task 8 and every adaptive/multi-level live action remain locked pending the QA disposition and any required offline repair.


## 2026-07-20 Principal Alignment T016 QA Not Accepted

`0720T016` QA is `未通过`.

Accepted sub-results:

- exact source/root/task/window/duration/envelope and activation boundary
- real buy and sell `0.005 BTC` post-only submissions, both resting
- independent final open orders `0`, BTC position `0.0`, estimated loss `0.0`
- child reap, writer health, source `62/62` and terminal checksum `104/104`
- one window only and conservative first-stop behavior

Remaining blockers:

- attempt 2 has `cancel_response_status_invalid`; reference-bound terminal proof is only `1/2`
- final `live_status` remains sell `unknown` / owned `1` despite independent terminal account proofs of zero
- acceptance checks `len(all candidate-attempt rows) == 2`, producing a false decision failure for `4` candidate rows and `2` submitted rows

The next formal task is offline-only and must repair cancel evidence/query semantics, terminal status reconciliation and submitted-attempt cardinality. T016 can be replayed to remove the decision false-negative, but its historical missing terminal evidence cannot be invented. Task 8 and every adaptive/multi-level live action remain locked until the repair receives independent QA and a new bounded live proves both references terminal.


## 2026-07-20 Principal Alignment T017 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T017.md`
- Status: `待执行`
- Scope: cancel failure response persistence, reference-bound terminal query evidence, final operator-status reconciliation and submitted-attempt exact-two acceptance.

This task is offline-only. It cannot infer terminality from account-wide empty orders, generic cancel errors or historical T016 absence. No new bounded live, Task 8 or adaptive/multi-level activation may start before independent QA accepts T017.


## 2026-07-20 Principal Alignment T017

`0720T017 / CANCEL-TERMINAL-STATUS-EVIDENCE-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `b35ea1e665595d9c3ae54b069cb239be44480256`
- failed cancel validation preserves the redacted exchange response
- canonical per-reference order-status query evidence is persisted and independently reconstructed
- only exact structured canceled statuses plus reference absence can supplement cancel acknowledgement
- unknown, filled, mismatched, duplicate, malformed and keyword-only evidence remains fail-closed
- final operator status is written after terminal manager/account reconciliation
- exact-two decision cardinality now uses submitted rows, while candidate rows remain separately validated
- focused regression: `336 passed`
- full Hyperliquid regression after final end-to-end coverage: `731 passed`
- exact T016 replay is byte-preserving, decision `43/43` pass, lifecycle `49 pass / 12 fail`
- T016 historical attempt 2 remains unproven at `1/2` terminal references
- live/private/order/cancel/network/remote/service actions: none

Independent QA is the current node. No new bounded live, Task 8 or adaptive/multi-level activation may start before T017 is accepted.


## 2026-07-20 Principal Alignment T017 QA Not Accepted

`0720T017` QA is `未通过`.

Accepted sub-results:

- failed cancel response persistence and redaction
- exact valid enum classification and keyword-only rejection
- reference/attempt mismatch, duplicate, error, unknown and filled reconciliation blockers
- foreign-order separation, v2 compatibility and submitted-row cardinality
- byte-preserving T016 replay with decision `43/43` pass and historical lifecycle blocker retained
- no strategy, risk-cap or activation drift

Remaining blockers:

- malformed non-string `orderStatus.status` can raise instead of returning unknown/fail-closed
- finalizer can observe tracked orders that are absent from the stale manager snapshot used by final `live_status`
- query-filled without complete raw fill proof produces inconsistent operator order/fill semantics

The next formal task is offline-only. It must harden malformed status parsing, reconcile the final operator snapshot against the exact final account evidence, and keep query-filled unresolved until complete fill evidence exists. No new bounded live or Task 8 may start before that repair passes independent QA.


## 2026-07-20 Principal Alignment T018 Dispatched

Formal task dispatched:

- `.workflow/tasks/0720T018.md`
- Status: `待执行`
- Scope: malformed status exception safety, supplied final account snapshot reconciliation and query-filled unresolved semantics.

This task is offline-only. It must preserve T017's accepted cancel evidence, v2 compatibility, submitted cardinality and T016 monotonic replay behavior. No new bounded live or Task 8 may start before independent QA accepts T018.


## 2026-07-20 Principal Alignment T018

`0720T018 / ORDER-STATUS-EXCEPTION-FINAL-SNAPSHOT-REPAIR` business execution is complete and awaiting QA.

Current facts:

- implementation commit: `d475fbb261fba912e61a3c39fa2bf95df998ea84`
- non-string status values and classifier exceptions fail closed with persisted redacted evidence
- final manager/operator state is rebuilt from the exact final open-order and user-state snapshot
- reappearing tracked references restore visible working exposure
- query-filled without raw fill proof remains unknown and blocking
- focused regression: `362 passed`
- full Hyperliquid regression: `756 passed`
- exact T016 replay remains byte-preserving with decision `43/43` pass and lifecycle `49 pass / 12 fail`
- no live/private/order/cancel/network/remote/service actions

Independent QA is the current node. No new bounded live or Task 8 may start before T018 is accepted.


## 2026-07-20 Principal Alignment T018 QA Not Accepted

`0720T018` QA is `未通过`.

Accepted sub-results:

- malformed status and classifier exception safety
- query-filled unresolved/no-raw-fill semantics
- T017 v2/v3/cardinality/foreign-order behavior
- byte-preserving T016 monotonic replay
- no strategy, risk-cap or activation drift

Remaining blockers:

- a valid final open-order snapshot is discarded when final user-state retrieval fails, leaving stale operator orders and exposure
- a final open row does not override local `cancel_requested` state to exchange-authoritative resting/partial

The next formal task is offline-only. It must decouple final open-order reconciliation from position snapshot success and make final exchange rows authoritative for visible order state. No new bounded live or Task 8 may start before that repair passes independent QA.
