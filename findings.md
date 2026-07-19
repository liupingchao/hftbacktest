# Findings

## 0719T006 Two-Sided Evidence Contract Boundary

- The next formal task is `0719T006 / EXACT-TWO-SIDED-MANAGER-EVIDENCE-CONTRACT`.
- The current exact orchestrator and Task 12 acceptance still describe the prior one-sided event-driven path.
- The exchange-reconciled manager currently submits two intents but primary attempt/status artifacts aggregate or omit one side.
- A valid single-level two-sided lifecycle needs one buy and one sell with distinct attempt IDs/keys and distinct terminal reference proof.
- The next live profile must use the Binance edge-gated watcher path, not a Hyperliquid-mid fallback.
- Offline acceptance must prove actual manager-writer artifacts pass before any remote live command is authorized.
- Multi-level and all adaptive controllers remain off.

## 0719T005 Cancel Success And Attempt Boundary

- The next formal task is `0719T005 / STRICT-CANCEL-SUCCESS-BOUNDED-ATTEMPT-REPAIR`.
- A single cancel action needs exactly one status row. Presence of a `success` key is not evidence that the action succeeded.
- Accepted success values must be explicit nonempty references; falsey values, containers, floats, extra keys and multi-status responses fail closed.
- Producer and acceptance must implement the protocol independently so a shared helper cannot hide a semantic bug.
- Reference attempts are bounded protocol identities. The accepted range is `1..2147483647`, with length checked before conversion.
- This task is offline-only and preserves the T004 token/redaction contract.
- No live task may be created until independent QA accepts this repair.

## 0719T004 Persisted Identity Repair Boundary

- The next formal task is `0719T004 / REDACTION-SAFE-REFERENCE-IDENTITY-STRICT-ATTEMPT-REPAIR`.
- Raw oid/cloid remain redacted; stable opaque SHA-256 tokens are the persisted join identity.
- Reference keys must be derived from attempt plus opaque tokens so generic recursive redaction cannot change their semantics.
- Producer and acceptance must independently validate token format and raw/token consistency.
- Attempt identity is not a numeric quantity. It must be parsed as a strict positive integer without coercion or truncation.
- Real producer-written standalone and manager artifacts are required integration inputs for acceptance tests.
- This task is offline-only and cannot change strategy behavior, caps or activation state.
- No live task may be created until independent QA accepts this repair.

## 0719T003 Raw-Proof Repair Boundary

- QA status is `未通过`; implementation commit is `ba220c5`.
- Producer target matching must use all-token consistency, not any-token intersection.
- A cancel row carrying both oid and cloid is valid only when each token uniquely resolves to the same attempt/reference.
- Downstream evidence integrity requires independent derivation from raw proof, not agreement between two copied producer summaries.
- Acceptance must parse raw exchange response statuses itself and must not import or call the producer reconciliation helper.
- Forged `matched_reference_key` and `authoritative_success` fields are claims to verify, not primary evidence.
- Producer now applies all-token unique same-reference matching.
- Acceptance now rebuilds the full reconciliation object from raw proof and requires exact agreement with both summaries.
- QA's unrelated-target and ambiguous-only raw-proof contradictions now fail closed in direct regressions.
- Focused regression is `130 passed`; full Hyperliquid regression is `483 passed`.
- The task remains offline-only and cannot change strategy behavior, caps or activation state.
- Independent QA confirmed all-token producer matching and raw-proof contradiction blocking, and confirmed acceptance does not import/call the producer helper.
- Remaining P1: producer artifact writers redact raw `oid/cloid` fields after reconciliation is computed. Persisted raw identity rebuilds to different keys than the stored producer summary, so valid standalone/manager artifacts fail exact equality and downstream acceptance.
- Remaining P1: producer and acceptance attempt parsers use `int(value)`. Fractional attempts such as reference `1.1` and cancel `1.9` alias to attempt `1` and are accepted.
- The next offline repair needs a stable redaction-safe identity token, strict positive-integer attempt parsing, and acceptance integration tests using actual producer-written artifacts.
- No live task may be created until independent QA accepts the repaired persisted evidence contract.

## 0719T002 Per-Reference Cancel Proof Boundary

- QA status is `未通过`; implementation commit is `7235372`.
- A zero-fill terminal claim is valid only when every submitted attempt/reference has its own authoritative successful cancel evidence.
- Aggregate cancel success is structurally unsafe because one order's success can mask another order's ambiguous terminal state.
- Reference identity is deterministic and attempt-scoped: oid and cloid are aliases for the same submitted reference only within the same attempt.
- Cancel evidence with no target, an unknown target, or a target matching multiple references is not usable terminal proof.
- A generic `already canceled, or filled` response is tolerable only after the same reference already has authoritative successful cancel evidence.
- Standalone execution must preserve all submitted refs across requote attempts; replacing the aggregate list loses terminal proof history.
- Two-sided manager evidence must assign distinct attempt identities to both sides before fill/cancel reconciliation.
- Downstream acceptance must scan emitted rows rather than trust summary booleans. It must also verify the same reconciliation object is present in the cancel shutdown proof.
- The QA counterexample, multi-reference success matrix, malformed/ambiguous mappings, standalone multi-attempt artifacts and two-sided manager artifacts now have direct regressions.
- Full Hyperliquid regression is `476 passed`; this task performed no live or private operation.
- Independent QA confirmed the original cross-attempt any-success counterexample is fixed and nominal standalone/manager artifacts are structurally present.
- Remaining P1: acceptance trusts producer `matched_reference_key` and `authoritative_success` rows and only compares copied summary objects. It does not rebuild reconciliation from `cancel_shutdown_proof.tracked_refs/cancel_results` or parse the raw exchange response.
- QA forged an unrelated evidence target while preserving a matching summary key; acceptance returned mechanism pass.
- QA also supplied ambiguous-only raw cancel evidence while both summaries claimed pass; acceptance again returned mechanism pass.
- Remaining P1: producer target matching uses any oid/cloid token intersection. A correct oid plus unknown cloid is accepted instead of fail-closed.
- The next offline repair must enforce all-token consistency and independently derive acceptance from raw proof. Add forged-both-copies, raw-proof contradiction and partial-conflicting-token tests.
- This task does not close Principal Task 12 or unlock multi-level. No new bounded live task may be created until the offline evidence repair is QA accepted.

## 0719T001 Runtime Provenance And No-Submit Boundary

- QA status is `未通过`.
- Runtime source provenance must be evidence from the live run itself, not a remote path relationship. T025 seals the exact source commit and all 62 non-test Hyperliquid Python source hashes before watcher startup, then revalidates before child start and after child exit.
- Offline acceptance now computes expected source hashes from the exact local Git commit archive. Runtime/preflight path equality is no longer accepted as a source proof substitute.
- A generic exchange response containing `already canceled, or filled` is not independently a fill ambiguity when an earlier authoritative tracked cancel succeeded and final orders/fill pullbacks/attribution/account state all reconcile.
- `no_fill_observed` is an economics-only producer blocker only when structured reconciliation status is `no_fill_reconciled`. Every other producer blocker remains mechanism/evidence and must fail downstream acceptance.
- Remaining P1: the implementation checks for any authoritative cancel success across the aggregate cancel list and overwrites `tracked_refs` on each attempt. It does not prove that every submitted attempt/reference has its own authoritative terminal cancel evidence.
- QA reproduced attempt 1 success plus attempt 2 ambiguous-only returning `no_fill_reconciled`; this must instead fail closed. Existing tests cover only a single tracked reference.
- T025 live source/provenance, config/control, process cleanup, terminal checksum and account safety all passed.
- The live window did not submit: the outer public trigger/event guard passed, but both inner attempts were skipped as `outside_quality_a_b_queue_bands`.
- The producer correctly emitted `fresh_touch_session_gate_no_eligible_candidate` as a mechanism/evidence blocker. Acceptance correctly failed rather than interpreting the safe no-submit window as a lifecycle pass.
- T025 is not an accepted Principal Task 12 baseline. Any retry must be a new formal task with a new isolated output path and one new bounded window.
- Multi-level remains locked. The existing one-sided T024 diagnostic lifecycle and zero-submit T025 window do not satisfy the original single-level two-sided manager prerequisite.
- The next formal boundary is offline per-attempt/per-reference cancel-proof repair plus regressions, followed only after QA by one bounded single-level two-sided manager lifecycle window with multi-level/dynamic/fill-feedback/skew activation off.

## 0718T023 Task-Scoped Live Envelope Boundary

- T023 QA status is `阻塞`.
- The live process completed safely at the process/open-orders level, but the window is not an accepted T023 evidence window.
- The task required `1 USDC` max loss and `0.01 BTC` aggregate position delta; the actual approved config snapshot recorded `30.0 USDC` and `0.04 BTC`.
- Root task identity was `0718T023`, while watcher/fill artifacts recorded `0623T007` and `0622T004`; the attempt key was `0622T004:window_01:attempt_1`.
- This is an exact envelope mismatch and a window/attempt identity mismatch. Actual absence of loss or position breach does not repair the contract violation.
- Supported live facts remain: one post-only `Alo` resting order, tracked cancel, final owned open orders `0`, independent private proof `0`, no attributed fill, no maker fill and no BTC position transition.
- Remote/local checksum passed `62/62`; status writer v2 was healthy.
- `a57c7da` repairs task-scoped max-loss/max-position propagation and identity propagation through cloid/ledger/manifests.
- A new formal repair/re-run task is required before any further live window. The current T023 window must not be used as an accepted same-window replay baseline or multi-level unlock.
- No stable PnL, fee/rebate calibration, fill rate, queue priority, maker viability, promotion or final MVP claim is supported.

## 0717T008 Fill Attribution Repair Boundary

- QA status is `已通过`.
- QA report is `.workflow/reports/0717T008-qa.md`.
- Implementation commits are `5d9f6f0`, `cd1b804`, `a23ff91`, `f6b4f84` and `ea7998a`.
- Stable fill identity must not depend on list position or mark price.
- One window owns one fill ledger map across every pullback phase.
- Oid/cloid matches take precedence over time-bounded fallback.
- A fill carrying an untracked oid/cloid must remain unattributed; it cannot fall through to price/time matching.
- Fallback attribution must be unique across attempts and remain under each attempt quantity cap.
- Exchange-native fill/trade ids can be deduplicated across repeated pullbacks.
- A synthesized identity collision inside one pullback is inherently ambiguous and must fail closed instead of silently undercounting distinct fills.
- Cancel acknowledgement timestamps define the terminal attempt boundary and must be recorded after the cancel call returns or fails.
- Ambiguous, conflicting, over-quantity and foreign-reference fills remain evidence, but must not enter attributed quantity or fee totals.
- QA review confirmed that a conflicting same-id payload must also remove any prior attributed row; `cd1b804` closes that accounting edge.
- QA review also confirmed that same-pullback synthesized-id collision detection must work after an earlier pullback already attributed the fill, and that tracked-reference overfills need a quantity-cap reason; `a23ff91` closes both edges.
- A hard identity conflict is window-terminal for that fill id: later pullbacks may update evidence metadata but must never restore attributed quantity or fee; `f6b4f84` enforces this quarantine.
- Cloid priority plus pre-attempt and post-terminal fallback rejection are now directly covered by `ea7998a`.
- Focused verification is `96 passed`; no live/private/order/cancel/remote action was performed.
- Phase 3 may now start as a separate formal watcher termination/timeout repair task; terminal checksum sealing and live remain out of scope.

## 0717T009 Watcher Termination/Timeout Repair Boundary

- QA status is `已通过`.
- Business report is `.workflow/reports/0717T009-business.md`.
- QA report is `.workflow/reports/0717T009-qa.md`.
- Implementation commit is `9b00e4c`.
- Scope is limited to `cross_exchange_live_remote_orchestrator.py`, its focused tests and resilience documentation.
- Current defect: blocking `subprocess.run()` prevents prompt child termination, timeout enforcement and process-group cleanup.
- Required evidence includes child pid/process-group, termination reason/signal, SIGKILL escalation, returncode, reap status, timeout budget and proof-after-child-exit.
- No terminal checksum-seal repair, live execution, private/order/cancel endpoint or strategy change is allowed.
- Focused verification is `103 passed`; no live/private/order/cancel/remote action was performed.
- Phase 4 terminal artifact seal repair is now the only next direction; integrated offline acceptance remains deferred.

## 0717T010 Terminal Artifact Seal Repair Boundary

- QA status is `已通过`.
- Task file is `.workflow/tasks/0717T010.md`.
- Business report is `.workflow/reports/0717T010-business.md`.
- QA report is `.workflow/reports/0717T010-qa.md`.
- Implementation commit is `b5247d9`.
- T009 left terminal checksum sealing deferred.
- Current defect: the manifest uses absolute paths; success rewrites `run_complete.json` and regenerates the manifest; failure generates the manifest before final `run_status.json`; heartbeat can still write while sealing.
- Required contract: finalize window/root evidence and final status/event log, stop and join heartbeat, write one run-root-relative manifest, verify it immediately, then write only the excluded verification summary.
- Implementation verification is `13` focused tests and `109` combined regression tests.
- No live/private/order/cancel/network/remote/service action, strategy change, or Phase 5 integrated acceptance is allowed.

## 0717T011 Integrated Offline Acceptance Boundary

- The next formal task is `0717T011 / LIVE-EVIDENCE-INTEGRATED-OFFLINE-ACCEPTANCE`.
- QA status is `已通过`.
- Task file is `.workflow/tasks/0717T011.md`.
- Business report is `.workflow/reports/0717T011-business.md`.
- QA report is `.workflow/reports/0717T011-qa.md`.
- It may use only offline fake-watcher, synthetic fill-attribution and artifact fixtures to exercise Phases 1-4 together.
- Required evidence: distinct window/attempt identity, idempotent fill totals, ambiguous attribution quarantine, child termination/reap, no next window after abort, final open-orders proof ordering, and final checksum verification.
- No live, private endpoint, order/cancel, network, remote/service action, strategy change, quote-policy change, or promotion is allowed.

## 0718T012 Price Normalization Boundary

- The formal task is `0718T012 / PRICE-NORMALIZATION-REPAIR`, Principal Alignment Task 1.
- QA status is `已通过`.
- Task file is `.workflow/tasks/0718T012.md`.
- Business report is `.workflow/reports/0718T012-business.md`.
- QA report is `.workflow/reports/0718T012-qa.md`.
- Implementation commit is `9252c4b`.
- It must create one authoritative Hyperliquid perp price normalization/post-only helper and wire kernel/executor/window callers to it.
- Required offline evidence includes exact precision rules, buy-floor/sell-ceil behavior, idempotence, no-crossing post-only output, and NaN/inf/nonpositive fail-closed cases.
- No live, private endpoint, order/cancel, network, remote/service action, strategy promotion, or quote-policy expansion is allowed.

## 0718T013 Persistent Kill-Switch Boundary

- The next formal task is `0718T013 / PERSISTENT-KILL-SWITCH-REPAIR`, Principal Alignment Task 2.
- It must implement durable halt state, fail-closed corrupt/missing-state handling, idempotent cancellation/flatten sequencing, and offline mock evidence.
- Real flatten, live order/cancel, private endpoint, remote/service action, strategy promotion, and automatic recovery are not allowed in this task.
- The implementation must keep the halt file in an independent control directory, atomically persist the trigger before any exchange action, and leave the state halted after any cancel/close/proof failure.
- Repeated invocation must be idempotent: an already active halt is observable without issuing duplicate cancel or market-close actions.
- T013 implementation adds explicit `armed`/`reset` state, so a missing control root/file is not treated as a clean start by quote paths; only explicit initialization/reset or a successfully flattened halt after expiry can resume quoting.
- The live path checks the same control state at watcher entry, before each fill-window attempt, and in `run_order_once`; max-loss rejection invokes the kill-switch sequence before the rejected order can be sent.
- `run_order_once` serializes the final halt check and `client.order()` with the kill-switch lock, so the durable trigger wins or waits in a defined order; endpoint evidence is marked only after the final gate.
- Any non-empty account open-order result is treated as ownership-ambiguous and fails closed until a later order-manager ownership classifier can prove otherwise.
- `run_controller` passes the configured control-state directory through the remote watcher command.
- T013 QA is `已通过`; implementation commits are `955cf9e` and `8bf84a7`, with `142` related offline regression tests passing.
- Next task is `0718T014 / AGGREGATE-EXPOSURE-RUNTIME-ENVELOPE`; no live/private/order/cancel/remote/service action is authorized for that offline task.
- Toxicity/stale/orchestrator trigger classification remains an explicit callable contract and observe-only boundary; automatic production trigger wiring beyond max-loss is deferred to the later risk/toxicity tasks as specified by the plan.

## 0718T014 Aggregate Exposure Runtime Envelope Boundary

- The formal task is `0718T014 / AGGREGATE-EXPOSURE-RUNTIME-ENVELOPE`, Principal Alignment Task 3.
- Scope is limited to the executor and focused executor tests.
- `ProjectedExposure` must separately represent worst long and worst short exposure from current signed position plus working, cancel-pending, and inflight leaves.
- Unknown submit state is conservatively treated as possibly resting; cancel-pending quantity remains counted until exchange confirmation.
- Proposed multi-level quotes must be aggregated before position, notional, and submission-cap validation.
- Reduce-side quotes remain eligible when they reduce inventory, but a quote that can sell through or buy through inventory is charged to the resulting opposite-side exposure.
- A task-specific envelope may only tighten the global defaults; the effective runtime cap is the stricter value.
- No live, private endpoint, order/cancel, network, remote/service action, strategy quote-policy change, or promotion is allowed in T014.

## 0718T014 Aggregate Exposure Implementation Ready for QA

- T014 implementation is complete and the business report is `.workflow/reports/0718T014-business.md`.
- `ProjectedExposure` preserves signed current position and all working/inflight directional leaves, then derives separate worst long/short quantities.
- `validate_runtime_envelope()` aggregates all proposed quote sizes before applying effective task-vs-global caps.
- The executor submit boundary validates aggregate exposure before `client.order()`, while max-loss remains the higher-priority kill-switch path.
- Offline verification is `26` focused executor tests, `26` kill-switch tests, and `153` related regression tests; compile and diff checks pass.
- The task is `待验收`; no live/private/order/cancel/network/remote/service action occurred.

## 0718T014 QA Repair Follow-Up

- The first QA review rejected the initial implementation because runtime state and cumulative submissions were optional at real call sites, existing leaves were valued at the next quote price, and reducing quotes could be blocked by simple gross-notional addition.
- Follow-up repair makes `run_order_once(projected=..., submissions_used=...)` required for live calls.
- `runtime_projected_exposure()` reads exchange-confirmed `user_state` and `open_orders`, parses BTC side/size/limit price, and carries the highest existing price into notional validation; malformed or foreign open orders fail closed.
- Canary, event-driven watcher, and fill-window callers now pass the snapshot and cumulative submission count. Fast event-driven position proof is pre-submit; only slow fee pullback remains post-submit.
- Formal default submission cap remains `2`; the existing explicit anti-drift operational mode may use the repository upper bound `30`.
- Related offline regression is now `158 passed`; the task remains `待验收` pending repeat QA.

## 0718T014 QA Accepted Finding

- T014 QA is `已通过`; implementation commits are `b48d7c3` and `4550726`.
- The required runtime projection and cumulative submission count are now mandatory at the submit boundary and wired from real canary/watcher/fill-window callers.
- Existing working leaves use their observed maximum quote price for notional valuation; missing or malformed valuation inputs fail closed.
- Pure inventory-reducing orders remain eligible near the aggregate notional cap, while sell-through/buy-through exposure is still charged to the resulting opposite side.
- Final related offline regression is `158 passed`; no live/private/order/cancel/network/remote/service action occurred.
- Cross-order lifecycle ownership and snapshot/submit serialization remain a later order-manager responsibility, not a Task 3 blocker.
- The only next formal task is `0718T015 / PRICE-TAXONOMY-QUOTE-ELIGIBILITY`, Principal Alignment Task 4.

## 0718T015 Dispatch Finding

- T015 is the sole next task for Principal Alignment Task 4 (C1/C2).
- The plan's `test_cross_exchange_pricing_stack.py` is absent in this checkout; the existing focused test is `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`.
- The current kernel still treats `signal_below_threshold` as `action=block` and emits one signal-selected side, so the task must change that contract while preserving fail-closed market/risk/post-only guards.
- T015 remains offline-only and must not initialize a live client or touch private/order/cancel/network/remote/service paths.

## 0718T015 Implementation Finding

- T015 implements `PricingConfigV1` with exact normalization stats hash and canonical config hash.
- Valid inputs now produce both post-only bid and ask quote intents; positive/negative alpha moves `forecast_mid_px` while threshold only sets `confidence_bucket`.
- Microprice is used only for a positive, coherent, fresh same-snapshot BBO; invalid qty or stale snapshot falls back to mid, while explicit incoherence blocks.
- Production shadow and public replay rows carry and compare `pricing_config_hash` and `normalization_stats_hash`.
- Offline verification is `13` focused shared-kernel/shadow/replay tests plus `14` price-math tests; T015 is `待验收`.

## 0718T015 QA Accepted Finding

- T015 QA is `已通过`; implementation commit is `2fe86f9`.
- The price taxonomy contract is now versioned by `PricingConfigV1`; config and normalization hashes are propagated through kernel, shadow, and replay artifacts.
- Signal threshold is an audit confidence bucket, not a quote eligibility gate; valid inputs produce both post-only sides.
- Microprice remains guarded by positive quantities, same-snapshot coherence, and freshness; fallback/block reasons are explicit.
- The next task is `0718T016 / RESERVATION-INVENTORY-SKEW-C12`, with skew kept disabled until its offline C12 acceptance passes.

## 0718T016 Dispatch Finding

- T016 is the only next task and covers Principal Alignment Task 5 (C10/C11/C12).
- It must preserve hard position caps independently of skew and retain reduce-side participation near/over cap.
- The C12 runner must compare alpha+zero-skew and alpha+bounded-skew on the same decision universe, keep observed fills separate from proxy fills, and label no-fill rows as censored.
- No live/private/order/cancel/network/remote/service action is authorized or planned.

## 0718T016 Implementation Finding

- T016 implements bounded reservation skew with explicit position ratio/notional and hard-cap audit.
- `NEAR_POSITION_CAP_RATIO=0.8` conservatively removes the inventory-worsening side while preserving the reduce side; runtime exposure caps remain authoritative.
- Desired and final quotes, clamp reasons, edge changes, and post-only invariants are recorded.
- C12 same-universe A/B artifacts keep observed fills, proxy fills, and censored no-fill rows separate.
- Current C12 inputs are deterministic fixtures only; real lifecycle fill evidence count is explicitly `0`, quote spread retention is a quote-width proxy, and inventory metrics use the fixture position path.
- Structural acceptance passes, but the enablement recommendation remains `remain_disabled_pending_real_c12_evidence`.
- Related offline regression is `33 passed`; T016 is `待验收`.

## 0718T016 QA Accepted Finding

- T016 QA is `已通过`; implementation commits are `ebced8e` and `36e7be6`.
- Reservation/skew sign, bounded penalty, near-cap reduce-only eligibility, post-only clamp evidence, and same-universe C12 structure are accepted.
- Real lifecycle fill evidence count remains `0`; no skew enablement, fill-rate, PnL, or maker-viability claim is supported.
- The only next task is `0718T017 / EXCHANGE-RECONCILED-SINGLE-LEVEL-ORDER-MANAGER`.

## 0718T017 Dispatch Finding

- The executor already exposes order/cancel/open-orders/user-state/query methods and runtime exposure validation, but it has no persistent strategy-owned quote lifecycle.
- T017 must add a separate manager rather than turning the one-shot executor into a continuous quote loop.
- Ownership must be recognizable from the fixed-length SDK cloid prefix, and ambiguous submit/cancel states must remain counted until exchange reconciliation.
- T017 is offline/mock-only and does not authorize watcher wiring or live endpoints.

## 0718T017 Implementation Finding

- T017 implemented the exchange-reconciled single-level manager in `examples/hyperliquid/hyperliquid_maker_order_manager.py`.
- Active ownership is keyed by `(symbol, side, canonical_price_key)`; same-price re-add after confirmed cancel receives a new generation/cloid.
- Foreign orders are ignored, duplicate owned logical keys fail closed, and partial recovery prefers exchange `remainingSz`.
- Submit ambiguity is resolved by oid/cloid query before retry; query failures leave the order `unknown` and no duplicate submit is attempted.
- Cancel-pending, submit-inflight and unknown leaves remain in aggregate working exposure until exchange reconciliation proves absence.
- Focused manager/executor tests passed `43`; related offline regression passed `30`; no live/private/order/cancel/network/remote/service action occurred.

## 0718T017 QA Accepted Finding

- T017 QA is `已通过`; implementation commit is `a3aed72`.
- Task 6 lifecycle, ownership, reconciliation, anti-churn and exposure-preservation boundaries are accepted.
- Task 7 is the only next task: watcher wiring, atomic throttled status, public-only shadow, and separately bounded first tiny-live. Dynamic spread, fill feedback, multi-level and inventory skew remain disabled.

## 0718T018 Dispatch Finding

- T018 is the only next formal task and covers Principal Alignment Task 7.
- The existing watcher already has public-only event-driven shadow and legacy single-side fill-window paths; the implementation must add typed bid/ask desired quotes and route lifecycle ownership through T017's manager without turning `run_order_once` into the continuous loop.
- A minimum atomic, throttled `live_status.json` is required before the first real order.
- The first tiny-live may proceed after the task-local execution-safety and evidence-integrity gates pass. It must stay single-level, fixed-spread, post-only, skew/dynamic-spread/fill-feedback/multi-level off, and within the standing envelope.
- No profitability, fill-rate, maker-viability or promotion claim is allowed from the first lifecycle window.

## 0717T007 Identity Contract Boundary

- QA status is `已通过`.
- Implementation commits are `66ba588` and `0271d99`.
- Stable attempt namespace is:
  - `<task_id>:window_<zero-padded-window-id>:attempt_<attempt-id>`
- The orchestrator window index is the authority for inline artifact identity.
- Single-window callers default to `artifact_window_id=1`.
- This task does not solve repeated-pullback deduplication or time-bounded fill attribution; those remain Phase 2.

## 2026-07-17 Principal Alignment Standing Authorization

- The user authorized uninterrupted serial auto-loop execution for Principal Alignment Task 0-12.
- The authorization covers task-scoped private reads, Hyperliquid BTC post-only submit/cancel, reduce-only flatten, detached remote jobs, artifact pullback, and temporary isolation of conflicting trading services.
- It removes repeated permission prompts, not task-level risk or evidence gates.
- A new user decision is required only if a task exceeds the recorded symbol, venue, order-size, aggregate-position, max-loss, submission, duration, concurrency, or taker boundaries.

## 0717T006 Repair Plan Scope Finding

- `0717T006` QA is `已通过`.
- The controller explicitly accepts the current runtime max-loss/max-position defaults for the tiny-live optimization stage.
- `runtime_risk_envelope_not_enforced` is no longer part of the immediate repair route.
- The remaining repair route is limited to four evidence/control defects:
  - multi-window and attempt identity
  - idempotent, attempt-bounded fill attribution
  - watcher termination and timeout
  - terminal artifact sealing and checksum verification
- The implementation dependency order is identity -> fill attribution -> process lifecycle -> artifact seal -> integrated offline acceptance.
- Plan document: `docs/cross_exchange_live_evidence_integrity_repair_plan.md`.
- No future implementation task should silently add risk-limit controls or strategy changes.

## 0717T005 Remote Update Code Review Finding

- `0717T005` QA is `未通过`.
- The repository is current at `c9547f2`, and focused tests pass, but test success does not close the live-safety/evidence gaps.
- P1 findings:
  - `0717T002` authorized `max_position_delta=0.01 BTC` and `max_loss=1 USDC`, while the runtime path still uses executor defaults `0.04 BTC` and `30 USDC`; the controller later accepted this risk and removed it from the immediate repair route.
  - orchestrator signal handling records abort intent but does not stop or timeout the active watcher subprocess.
  - failed-run sha manifest is stale by construction; local reproduction produced two checksum mismatches.
  - fallback fill attribution reuses window-wide pullbacks and full-row deduplication, allowing one fill to be counted more than once.
- P2 findings:
  - inline artifacts still hardcode window 1 across orchestrator windows.
  - latest commit tracks `.DS_Store` and overlapping trade-history exports.
  - `0717T002` raw artifacts are gitignored and not reproducible from a fresh checkout.
- Future live runs remain blocked until the remaining evidence/control defects, account provenance, and concurrent-service isolation gates are repaired, accepted, or explicitly downgraded by the controller.

## 0717T004 WTIOIL Root Cause Finding

- `0717T004` QA is `已通过`.
- The WTIOIL trade-history row is now attributed:
  - trade-history row: `2026/7/17 13:41:40`, `WTIOIL (xyz)`, `Open Short`, `78.51`, `1.14`, notional `89.5014`.
  - root cause: awsserver1 `xemm.service`, running `/home/admin/XEMM_rust/target/release/xemm_rust`.
- `xemm.service` configuration:
  - `maker_symbol`: `CLUSDT`
  - `hedge_symbol`: `xyz:CL`
  - `symbol`: `CL`
  - `order_notional_usd`: `90.0`
  - `order_refresh_interval_secs`: `18`
- Exact journal match:
  - `2026-07-17T05:41:39.493870Z`: XEMM recovered Binance fill `BUY 1.140000 @ $78.470000`.
  - `2026-07-17T05:41:39.507464Z`: XEMM began executing `SELL 1.14` on Hyperliquid.
  - `2026-07-17T05:41:39.630198Z`: XEMM sent Hyperliquid market order `SELL 1.14 xyz:CL`.
  - `2026-07-17T05:41:40.295954Z`: hedge executed successfully, filled `1.14 @ $78.51`.
  - `2026-07-17T05:42:20.742519Z`: XEMM trade summary reports `Hyperliquid: SELL 1.1400 xyz:CL @ $78.510000`.
- Conclusion:
  - WTIOIL was not caused by the 0717T002 Python runner and is not evidence of a Python symbol-routing bug.
  - WTIOIL was caused by a concurrent live XEMM CL/WTI hedge service using the same host/account environment.
- Additional source-path finding:
  - XEMM logs also show Hyperliquid REST fill lookup did not find the fill and fell back to fill event data, which aligns with the broader REST fill-source/provenance weakness found in 0717T003.
- Live gate implication:
  - before any future live evidence run, detect and fail closed on active non-task trading services (`xemm.service` or equivalent), or use a clean isolated account/subaccount.
  - account provenance guard remains necessary, but service isolation is now a separate required gate.

## 0717T003 0717 Trade History / WTIOIL Safety Audit Finding

- `0717T003` QA is `已通过`.
- The user-provided `trade_logs/0717trade_history.csv` must be treated as counter-evidence to the previous 0717T002 no-fill conclusion.
- Parsed 0717 trade rows include:
  - `2026/7/17 13:11:50`, `BTC`, `Open Long`, `63422`, `0.00067`
  - `2026/7/17 13:36:03`, `BTC`, `Open Long`, `63150`, `0.005`
  - `2026/7/17 13:41:40`, `WTIOIL (xyz)`, `Open Short`, `78.51`, `1.14`
- If the CSV timestamps are Shanghai local time, the two BTC rows map to 0717T002 window 01 and window 02 and match the runner intents exactly by symbol, side, price, and size.
- 0717T002 window 03 artifacts do not support WTIOIL attribution:
  - intent rows are `BTC buy 0.005 @ 62874.0` and `BTC buy 0.00304 @ 62882.0`.
  - both were post-only immediate-match rejects.
  - Hyperliquid meta confirms `asset=0` is `BTC`.
- Code inspection found no production WTIOIL route in the Hyperliquid live runner path.
- Code inspection found live BTC symbol guards:
  - `SYMBOL = "BTC"`
  - `validate_order_intent()` rejects non-BTC.
  - `SDKHyperliquidClient.order()` passes `intent.symbol`.
  - fill fallback rejects symbol mismatch.
- Read-only SSM account-scope audit command `7b8e64ff-771d-4013-955c-ae990ad9a6a9` found the awsserver1 env account and private-key wallet are the same redacted address/hash, but that address has zero fills/recent fills/open orders/positions for the 0717T002 interval.
- Final safety finding:
  - 0717T002 no-fill classification is invalid.
  - BTC rows are highly consistent with 0717T002 runner activity.
  - WTIOIL short remains unexplained but is not attributable to the current repo runner from available artifacts/code.
  - account provenance / fill-source identity is not closed, so no further live run should proceed until a guard records and verifies the order-submit account, fill-pullback account, and post-state account identity.

## 0717T003 Required Repair Route

- Add account provenance artifacts to the live runner:
  - redacted/hash wallet-from-private-key
  - redacted/hash account/vault used by `Exchange(...)`
  - redacted/hash address passed to `user_fills_by_time`
  - redacted/hash address used by `open_orders`, `user_state`, and `user_fills`
  - fail-closed when these are absent or inconsistent with the configured live envelope.
- Fix multi-window artifact identity:
  - `run_intent_marker.json` should use the orchestrator window id, not hardcoded `1`.
  - inline finalize fill attribution should preserve true window id and attempt id.
- Add offline regression using 0717T002 artifacts and 0717 trade history to prove future reports cannot classify matched external fills as no-fill.

## 0717T002 SSM-First Live Rerun Finding

- `0717T002` QA is `阻塞`.
- Superseded for fill interpretation by `0717T003`: collection and open-orders proof remain useful, but the no-fill interpretation is invalidated by `trade_logs/0717trade_history.csv`.
- The SSM-first remote live orchestrator completed all three windows and avoided the prior long-SSH evidence dependency.
- Run root:
  - remote: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z`
  - local: `local_live_analysis/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z/`
- Safety/evidence closure:
  - window 01, 02, and 03 runner return code `0`
  - each window independent open-orders proof `0`
  - final root open-orders proof `0`
  - JSON/CSV parse passed
  - sha manifest `241/241` matched
- Live evidence facts:
  - total order intents: `4`
  - window 01: one resting buy intent, no fills
  - window 02: one resting buy intent, no fills
  - window 03: two post-only immediate-match rejects, no resting interval rows
  - total fill ledger rows: `0`
  - total liquidity-role rows: `0`
- The current blocker is not collection failure; it is absence of fills under this conservative envelope.
- P0 fill source / maker-taker role evidence remains blocked as `no_fill_role_evidence_absent`.
- Fee/PnL calibration, maker fill count, fill-rate calibration, maker viability, T012, promotion, and final MVP pass remain unsupported.
- A discovered limitation remains: the current watcher/orchestrator CLI supports max order size and max submissions but does not expose runtime flags for `max_loss=1 USDC` or `max_position_delta=0.01 BTC`; those were used as task-envelope and post-run validation boundaries, not newly implemented runtime controls.

## 0717T001 Live Collection SSH Resilience Finding

- `0717T001` QA is `已通过`.
- It addresses the repeated live-test risk where public SSH can timeout or break during live evidence collection.
- The fix adds `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`.
- The orchestrator is SSM-friendly and does not rely on the caller's SSH session as the evidence source of truth.
- It adds:
  - nonblocking live lock
  - `run_status.json`
  - `heartbeat.json`
  - `orchestrator_events.jsonl`
  - per-window `window_status.json`
  - per-window `independent_remote_open_orders_check.json`
  - `abort_manifest.json`
  - `run_complete.json`
  - `remote_sha256_manifest.txt`
- Offline tests prove complete and failed-window paths without reading credentials or touching exchange endpoints.
- Future live evidence runs should be launched through SSM-first orchestration and recovered from remote status/heartbeat/artifact files if SSH/scp disconnects.
- This is an infra collection repair only. It does not change strategy behavior, quote policy, thresholds, order size, max submissions, max loss, live authorization requirements, T004 gating, or fee/PnL calibration.
- Deferred infra work remains S3 artifact upload/download, permanent systemd service/timer hardening, and AWS network/IAM hardening.

## 0716T005 Fill Source Liquidity-Role Preflight QA Finding

- `0716T005` QA is `已通过`.
- It converts accepted 0716T001 fill-attribution repair, 0716T003 liquidity-role contract repair, and 0716T004 quote-policy prework into a future controlled evidence preflight contract.
- Output package:
  - `local_live_analysis/cross_exchange_fill_source_liquidity_role_preflight_0716T005/`
- The preflight requires future evidence to preserve:
  - `fill_liquidity_role_evidence.csv`
  - `user_fills_pullback_audit.json`
  - `live_fill_ledger.csv`
  - `order_intent_audit.csv`
  - `private_order_response_audit.json`
  - `resting_interval_lifecycle_matrix.csv`
  - `public_stream_coverage.csv`
  - `boundary_manifest.json`
- Role statuses remain explicit:
  - `confirmed_maker`
  - `confirmed_taker`
  - `unknown_liquidity_role`
- Fee/PnL remains blocked if any fill has `unknown_liquidity_role` or missing source-path evidence.
- Final route is `route_to_separately_authorized_controlled_evidence_acquisition_with_liquidity_role_contract`.
- QA accepts this as a preflight contract only. It unlocks creation of a separate controlled evidence acquisition task, not live execution itself.
- This does not authorize live retry, quote-policy change, threshold/quote-envelope/order-size/max-submission change, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.

## 0716T006 Controlled Evidence Acquisition Task Finding

- `0716T006` has been created and executed through its authorization/source gate.
- Status is `阻塞`.
- Output package:
  - `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006/`
- Final route:
  - `blocked_missing_live_authorization`
- Controller subsequently supplied a live rerun authorization for the 0715T001 envelope on `awsserver1`.
- `0716T006` was reopened for that authorized rerun only.
- The live rerun connectivity blocker has been recovered; current blocker is absence of fill role evidence.
- No complete non-live artifact source was supplied.
- No complete live authorization envelope was supplied.
- No live execution was run, and no public market-data stream, private user stream, order-submit endpoint, cancel endpoint, or credential path was touched.
- Missing inputs remain exact UTC schedule, host/account scope, symbol/venue, duration/window count, post-only behavior, max order size, max submissions, max position/inventory delta, max loss, credential/source boundary, source branch/commit, and explicit real-order authorization.
- The sequence may not proceed to T004 public shadow unless the controller either obtains accepted role/source-path evidence or explicitly downgrades the route.
- Still not authorized: live retry, quote-policy change, threshold/quote-envelope/order-size/max-submission change, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.
- Authorized live rerun envelope is limited to Hyperliquid `BTC`, post-only `Alo`, three sequential `1800s` windows, max size `0.005 BTC`, max submissions `2` per window, max position delta `0.01 BTC`, max loss `1 USDC`, source `cross-exchange/a5431d8b24da7d77671148d316f789b0b25cf3f8`, and existing awsserver1 env file `/home/admin/XEMM_rust_latest/.env`.
- Window 1 started at `2026-07-16T07:31:33Z`; it was recovered/pulled back and open-orders proof was `0`.
- Window 2 started at `2026-07-16T07:51:37Z`, completed at `2026-07-16T08:00:19Z`, and remote log shows independent open-orders proof `0`.
- Window 3 started at `2026-07-16T08:00:19Z`; after its theoretical completion, SSH/ping checks failed, then later recovered after EC2 Instance Connect plus SSM role/profile/agent repair.
- Post-recovery checks show no remaining `0716T006` / `hyperliquid_tiny_live` process and read-only Hyperliquid `open_orders()` returned `0`.
- Complete artifact package was pulled to `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z_full/`.
- Artifact validation parsed `101` JSON and `117` CSV files with `0` errors.
- All three windows show endpoint submit/cancel activity within the authorized envelope, shutdown proof `pass`, and final open orders `0`.
- All three windows show `fill_count=0`, `ledger_fill_rows=0`, and zero `fill_liquidity_role_evidence.csv` rows.
- Final route is `route_to_controlled_evidence_rerun_or_explicit_downgrade_no_fill_role_evidence`.

## Project Milestones

North star:

- Binance lead / Hyperliquid lag maker strategy that is repeatable, risk-bounded, and net-positive after fees on live data.

Current checkpoint status:

- M0 Evidence chain and gate baseline: complete
- M1 Repeated tiny-live canary windows: complete
- M2 Real PnL and fee / slippage / inventory accounting: blocked on live maker fills
- M3 Cross-day / cross-regime stability: pending
- M4 Expansion or stop decision: pending

## 0715T001 UTC Live Evidence Correction

- `0715T001` replaced the missed `0714T006` gate and business execution is complete.
- The new gate uses UTC as the scheduling source:
  - target UTC: `2026-07-15T13:15:00Z`
  - New York equivalent: `2026-07-15 09:15 EDT`
  - Shanghai equivalent: `2026-07-15 21:15 CST`
- Actual execution ran three sequential controlled live windows under the same conservative envelope:
  - window 1: artifact reported submitted/resting/no-fill, but exchange trade-history reconciliation matches the submitted buy intent exactly.
  - window 2: artifact reported submitted/resting/no-fill, but exchange trade-history reconciliation matches the second submitted buy intent exactly.
  - window 3: submitted but no resting lifecycle (`error,error`), so interval evidence is empty/not applicable.
- Total live submissions: `5`; artifact live fill ledger rows: `0`; external trade-history reconciliation matches `2` intents / `0.01 BTC` fills / `0.098024 USDC` fees.
- This is a fill-attribution defect in the artifact path, not no-fill evidence.
- Current evidence still does not support maker fill count, fill probability, fee/PnL calibration, maker viability, `T012`, promotion, final MVP pass, or parameter expansion.
- Next route must be `0716T001` fill attribution repair before QA acceptance, offline quote/fill rerun, live retry, or fee/PnL calibration.

## 0716T001 Fill Attribution Repair Finding

- `0716T001` QA is `已通过` and repairs the artifact false-negative mechanism exposed by 0715T001.
- Root cause:
  - fill attribution depended too strongly on tracked oid matching.
  - raw/redacted `user_fills_by_time` pullback payloads were not persisted for later artifact-level reconciliation.
  - ambiguous cancel response text that includes `already canceled, or filled` could still be followed by no-fill classification.
- Repair:
  - attempt-level attribution now supports tracked-oid matching and bounded symbol/side/price/size fallback.
  - `live_fill_ledger.csv` includes attribution status/source fields.
  - `user_fills_pullback_audit.json` is emitted.
  - missing liquidity role is `unknown`, not maker.
- Corrected 0715T001 evidence is fill-supported but liquidity-role-unknown:
  - `0.01 BTC` matched external fills across window_01 and window_02.
  - maker fill count remains unsupported.
  - fee/PnL calibration remains unsupported until liquidity role and lifecycle/PnL attribution are repaired/accepted.
- Accepted next route is offline quote/fill analysis using corrected attribution; live retry and fee/PnL calibration remain disallowed.

## 0716T002 Quote-Fill Analysis QA Finding

- `0716T002` QA is `已通过`.
- Corrected 0715T001 evidence does not support the prior low-fill/no-fill interpretation.
- Attempt-level facts:
  - submitted attempts: `5`
  - post-only rejects: `3`
  - corrected filled resting attempts: `2`
  - strict-trade-through filled attempts: `1`
- Interpretation:
  - 0-tick touch placement can fill under this envelope.
  - window_02 fill is aligned with strong adverse public flow / strict trade-through, so quote policy risk is now the main design topic.
  - window_01 fill is externally matched but not fully explained by captured public depletion, so source-path fill lifecycle evidence still needs improvement.
- Durable route:
  - `route_to_quote_policy_design_prework_and_liquidity_role_evidence_repair`
- Still unsupported:
  - maker fill count, fee/PnL calibration, realized PnL, maker viability, T012, promotion, final MVP pass.

## 0716T003 Liquidity Role Evidence Repair Finding

- `0716T003` QA is `已通过` and adds an explicit future artifact contract for maker/taker role evidence.
- Future live artifacts now include `fill_liquidity_role_evidence.csv`.
- Role status values:
  - `confirmed_maker`
  - `confirmed_taker`
  - `unknown_liquidity_role`
- Fee/PnL role gate:
  - role-known fills can pass the role-evidence gate.
  - unknown-role fills remain blocked for fee/PnL calibration.
- This repairs future evidence collection only; it does not retroactively recover maker/taker role for 0715T001 external trade export.

## 0716T004 Quote Policy Design Prework Finding

- `0716T004` QA is `已通过` and accepts design prework only; no strategy behavior or parameter change is authorized.
- Candidate directions:
  - keep touch-only baseline as control.
  - design adverse public-flow suppression for cases like window_02 strict trade-through fill.
  - design post-only reject drift precheck for the 3/5 reject pattern.
  - improve fill explanation/source-path capture for externally matched but public-depletion-unexplained fills.
- Final route:
  - `route_to_controlled_evidence_design_with_liquidity_role_and_quote_policy_preflight`
- Fee/PnL calibration remains blocked until future accepted evidence has maker/taker role and exchange-native fill lifecycle attribution.

## 0714T006 Missed Scheduled Live Gate Finding

- `0714T006` is `阻塞`.
- The task itself was correctly formalized and guarded, but the scheduled automation fired at the wrong absolute time:
  - actual trigger: `2026-07-14T21:15:03Z`
  - actual New York time: `2026-07-14 17:15 EDT`
  - actual Shanghai time: `2026-07-15 05:15 CST`
  - authorized target: `2026-07-14 09:15 EDT` / `2026-07-14 21:15 CST`
- This was a scheduling/timezone failure, not a market-data, strategy, execution-layer, interval-coverage, or QA failure.
- Correct handling was fail-closed:
  - no live windows
  - no credential/private endpoint access
  - no order submission
  - no live artifact
- The one-time automation `0714t006-live-test-at-us-open-preflight` is obsolete and should be deleted.
- Any future live evidence collection must be a fresh formal task with a newly authorized exact live window.

## 0714T005 Interval Coverage Capture Repair QA Finding

- `0714T005` QA is `已通过`.
- Code:
  - `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- Output package:
  - `local_live_analysis/cross_exchange_public_flow_interval_coverage_capture_repair_0714T005/`
- Finding:
  - The artifact contract can now distinguish complete interval coverage with zero public trades from incomplete coverage with a diagnostic reason.
  - Future live artifacts should no longer stop coverage at a pre-resting trade cursor if the public websocket can observe a post-interval public event after cancel.
  - This is still capture repair only: it does not claim fill probability, quote policy design, fee/PnL, maker viability, T012, promotion, or final MVP pass.
- Follow-up gate result:
  - The timed live gate was formalized as `0714T006`, but later ended `阻塞` because the automation fired after the authorized pre-open window.
  - No live evidence was collected from that missed gate.

## 0714T004 V2 Quote-Fill Evidence QA Finding

- `0714T004` QA is `已通过`.
- Runner:
  - `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py`
- Output package:
  - `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0714T004/`
- Source input:
  - accepted `0714T003` v2 live package.
- Attempt-level evidence:
  - `71` attempt rows
  - `70` no-submit/skipped rows
  - `1` submitted/resting/no-fill row
  - fill count `0`
- Coverage evidence:
  - `public_stream_coverage_evidence_matrix.csv` rows `1`
  - coverage status `coverage_not_proven_complete`
  - zero public trade interpretation `artifact_gap_not_no_exchange_trades`
- Accepted route:
  - `route_to_public_flow_artifact_repair`
- Finding:
  - The v2 artifact path is now good enough to show why the quote/fill evidence still cannot be interpreted as low fill probability.
  - The blocker is not a fill-probability model yet; it is still interval public-flow coverage.
  - Next route should be a narrow public-flow artifact/capture repair, not quote policy design, threshold change, quote-envelope change, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.

## 0714T003 V2 Live Evidence QA Finding

- `0714T003` QA is `已通过`.
- It ran three controlled same-envelope live windows on `awsserver1` using the repaired v2 resting-interval capture contract.
- Artifact package:
  - `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/`
- Window results:
  - Window 1 and 2 were no-submit fail-closed windows with final open-orders `0`.
  - Window 3 submitted one post-only `Alo` buy order, status `resting`, no fill, cancel/shutdown proof `pass`, and final open-orders `0`.
- Window 3 v2 public-flow evidence:
  - `public_stream_coverage.csv` exists with `coverage_not_proven_complete`.
  - zero-row interpretation is `artifact_gap_not_no_exchange_trades`.
- Finding:
  - The v2 live artifact path works in a real submitted/resting/no-fill lifecycle.
  - The actual interval public-flow evidence is still not complete enough to claim no exchange public trades, fill probability, quote policy, queue priority, fee/PnL, maker viability, T012, promotion, or final MVP pass.
  - The next task should be `0714T004` offline quote/fill evidence rerun, not a threshold or quote-envelope change.

## 0714T002 Resting-Interval Capture Contract Repair QA Finding

- `0714T002` QA is `已通过`.
- It upgrades the future watcher resting-interval artifact contract to `cross_exchange_resting_interval_public_flow_capture_v2`.
- New/strengthened evidence fields:
  - stable `attempt_key`
  - lifecycle interval source/status/completeness fields
  - public-trade quote relation flags
  - L2/depth source and quote-in-book status fields
  - `public_stream_coverage.csv`
  - zero-row interpretation counts in the manifest
- Mock validation package:
  - `local_live_analysis/cross_exchange_resting_interval_capture_contract_repair_0714T002/`
- Key finding:
  - The schema can now represent `zero_public_trades_observed_with_complete_interval_coverage` separately from `artifact_gap_not_no_exchange_trades`.
  - This is still instrumentation repair only; it does not create new live evidence or support fill probability, quote policy design, queue priority, fee/PnL, maker viability, T012, promotion, or final MVP pass.

## 0714T001 Public-Flow Interval Repair Design QA Finding

- `0714T001` QA is `已通过`.
- Runner:
  - `examples/hyperliquid/cross_exchange_public_flow_interval_artifact_repair_design_0714T001.py`
- Output package:
  - `local_live_analysis/cross_exchange_public_flow_interval_artifact_repair_design_0714T001/`
- Source input:
  - accepted `0713T003` quote/fill probability evidence package.
- The design emits:
  - `5` artifact gaps
  - `4` instrumentation design rows
  - `5` acceptance gates
  - required artifact contract `cross_exchange_resting_interval_public_flow_capture_contract_v2`
- Final business route:
  - `route_to_resting_interval_capture_contract_repair`
- Finding:
  - The current `0` matching attempt-keyed interval public-trade rows means artifact observability is insufficient; it does not prove no exchange public trades occurred.
  - Before another controlled live evidence task, the code/schema path should implement attempt-keyed lifecycle, public-trade coverage, resting-start L2/depth, and interval coverage status fields.
  - Still unsupported: fill probability, quote policy design, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP pass.

## 0713T003 Quote/Fill Probability Evidence QA Finding

- `0713T003` QA is `已通过`.
- Runner:
  - `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0713T003.py`
- Output package:
  - `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0713T003/`
- Source input:
  - accepted local `0713T002` pulled-back package `local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
  - remote source is provenance-only: `awsserver1:/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
- The rerun preserves source attribution:
  - formal source task id `0713T002`
  - raw legacy writer metadata task id `0623T007`
  - raw pulled-back artifacts were not mutated.
- Pre-QA review repair `c31e6b0` fixes remote provenance, skipped/no-order attempt-id semantics, and repo-relative generated paths. amdserver QA reproduction should use `/home/molly/anaconda3/envs/nt-backtest/bin/python`; focused tests pass there, while full `examples/hyperliquid` is blocked by missing `requests` and `numba`.
- Attempt-level evidence:
  - `18` quote evaluation rows
  - `17` no-order/skipped rows
  - `1` submitted/resting/no-fill row: `buy 0.0049 BTC @ 62844.0`, post-only `Alo`, event sequence `2922`
  - skipped/no-order rows have blank `order_attempt_id`; the submitted/resting row keeps `order_attempt_id=1`
  - fill/maker fill count `0/0`
  - post-only rejects `0`
- Resting-interval public-flow/depletion evidence:
  - matching attempt-keyed interval public-trade rows `0`
  - same-side visible qty at or ahead of quote `0.05334 BTC`
  - required depletion qty `0.05824 BTC`
  - queue depletion multiple `0`
  - depletion status `insufficient_interval_trades_or_depth`
- Caveats:
  - lifecycle interval remains `proxy_interval_from_local_order_response_and_cancel_ack`
  - resting timestamp is a local exchange-response-end proxy, not exact exchange resting timestamp
  - cancel ack is a local cancel-ack proxy, not exact exchange cancel ack
  - depth row is `l2_snapshot_proxy_not_after_order_resting`
  - hold horizon is short: `3.125993s`
- Final business route:
  - `route_to_public_flow_artifact_repair`
- QA accepts this as a public-flow artifact repair route only. This finding does not support fill probability, proof that no exchange public trades occurred, exact queue position, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, quote policy design, or final MVP pass.

## 0713T002 Controlled Same-Envelope Live Evidence QA Finding

- `0713T002` QA is `已通过`.
- It executed Step 3 from `docs/cross_exchange_resting_interval_public_flow_auto_loop_plan.md` on `awsserver1`, using the same conservative envelope as accepted T011 live evidence: Hyperliquid `BTC`, post-only `Alo`, fast `l2Book`, max order size `0.005 BTC`, and max submissions `2`.
- Artifact package:
  - remote `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
  - local `local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
- Window 1 produced a `submitted_resting_no_fill` lifecycle:
  - `1` real post-only `Alo` submission
  - order status `resting`
  - submitted order `buy 0.0049 BTC @ 62844.0`
  - fill count `0`
  - post-only rejects `0`
  - shutdown proof `pass`
  - runner final open-orders `0`
  - independent final open-orders `0`
- Resting-interval public-flow artifacts were captured under schema `cross_exchange_resting_interval_public_flow_capture_v1`:
  - lifecycle rows `1`
  - interval public-trade rows `0`
  - resting-start L2/depth rows `1`
  - interval depletion rows `1`
- Pre-QA repair commit `762e335` fixes future formal task-id propagation for inline live artifacts. The current pulled-back raw `0713T002` manifests may still contain legacy writer metadata `task_id=0623T007`; `source_attribution_overlay.json` maps those raw files to formal evidence task `0713T002` without mutating raw files.
- The captured lifecycle/depth fields remain conservatively statused where exact exchange timestamps/depth were not available:
  - resting timestamp is a local exchange-response-end proxy.
  - cancel acknowledgement is a local cancel-ack proxy.
  - L2 depth row is `l2_snapshot_proxy_not_after_order_resting`.
  - depletion estimate is `insufficient_interval_trades_or_depth` because no matching attempt-keyed interval public-trade rows were captured.
- `captured_public_trade_row_count=0` must not be read as proof that no public trades occurred on the exchange; it is only a no-captured-row fact for the proxy interval.
- Pullback and validation passed locally: `scp`, sha256 reconciliation `70/70`, JSON parse errors `0/32`, CSV parse errors `0/35`, true secret-write flags `0`, boundary manifest `pass`.
- This accepts Step 3 and supports creating a separate Step 4 quote/fill probability evidence rerun task using the pulled-back local package. It still does not prove fill probability, queue priority, fee/rebate, realized PnL, stable PnL, maker viability, T012 readiness, promotion, or final MVP pass.

## 0713T001 Resting-Interval Capture Instrumentation Finding

- `0713T001` QA is `已通过`.
- It extends `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` so future same-envelope live watcher artifacts can emit `cross_exchange_resting_interval_public_flow_capture_v1`.
- New future-capture artifact files:
  - `resting_interval_lifecycle_matrix.csv`
  - `resting_interval_public_trades.csv`
  - `resting_start_l2_book_snapshot_at_or_after_order_resting.csv`
  - `resting_interval_depth_depletion_matrix.csv`
  - `resting_interval_capture_manifest.json`
- Task-scoped mock package:
  - `local_live_analysis/cross_exchange_resting_interval_public_flow_capture_instrumentation_0713T001/`
- The implementation preserves conservative status fields: current exchange response timing remains marked as a local response-end proxy unless exact exchange resting timestamp is available, and L2 depth is marked by whether the local receive timestamp is at/after the resting proxy.
- Focused tests prove public trades are keyed to the matching attempt and proxy lifecycle/depth does not authorize `offline_repair_sufficient`.
- This is instrumentation only. It does not run live, call endpoints, read credentials, collect market data, change thresholds, quote envelope, order size, max submissions, or strategy behavior.
- Step 3 live evidence remains blocked until a later formal task records the exact live envelope, `awsserver1` remote artifact root, local pullback path, and explicit controller authorization.

## 0712T001 Public-Flow Interval Artifact Repair QA Finding

- `0712T001` QA is `已通过`.
- QA accepted that the current artifacts expose only proxy lifecycle/depth for the two live resting/no-fill rows and no local interval artifact for the prior reference.
- Actual resting-interval public trades and actual interval depletion/trade-through remain `not_reconstructable_from_current_artifact` for all `3` accepted resting/no-fill attempts.
- The accepted route is `route_to_controlled_same_envelope_live_evidence_with_resting_interval_public_flow_artifacts`.
- No fill probability, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claim is supported.

## 0712T001 Public-Flow Interval Artifact Repair Finding

- `0712T001` business execution completed and was later accepted by QA.
- Output package: `local_live_analysis/cross_exchange_public_flow_interval_artifact_repair_0712T001/`.
- Runner: `examples/hyperliquid/cross_exchange_public_flow_interval_artifact_repair.py`.
- The task defines `cross_exchange_resting_interval_public_flow_contract_v1`.
- Accepted resting/no-fill attempts covered: `3`
  - `0708T001` prior QA reference: no local resting-interval public-flow artifact; row is `not_reconstructable_from_current_artifact`.
  - `0709T001_window_02`: live resting/no-fill row; lifecycle/depth are proxy-only.
  - `0709T001_window_03`: live resting/no-fill row; lifecycle/depth are proxy-only.
- Live proxy bindings:
  - `resting_start_ts` is derived from `event_driven_latency_matrix.exchange_order_response.end_unix_seconds`, so it is a local response-end proxy, not exact exchange resting timestamp.
  - `cancel_or_shutdown_ts` is derived from response-end proxy plus `quote_aging_guard_matrix.hold_elapsed_seconds`, not exact cancel acknowledgement.
  - same-side visible depth comes from pre-submit/post-open-orders inline-reprice L2 proxy, not an exact resting-start L2 snapshot.
- Actual resting-interval public trades are not reconstructable for all `3` rows.
- Actual resting-interval depletion/trade-through estimate is not reconstructable for all `3` rows.
- For the two live rows, `rolling_flow_state.csv` ends before the order-response proxy:
  - `0709T001_window_02`: last public exchange time `1783580109291`, resting-start proxy `1783580110069`.
  - `0709T001_window_03`: last public exchange time `1783580236390`, resting-start proxy `1783580237570`.
- Final route:
  - `route_to_controlled_same_envelope_live_evidence_with_resting_interval_public_flow_artifacts`
- Review-fix commit `65f2461` resolved two pre-QA review risks:
  - route selection no longer treats exact interval public trades alone as `offline_repair_sufficient`; exact lifecycle/depth evidence must also be present.
  - unkeyed future interval public-trade rows are no longer assigned to a specific order attempt.
- The official 0712T001 business route and matrices remained unchanged after deterministic rerun; only manifest `git_commit` and sha256 were refreshed to the tightened runner commit.
- This remains non-optimistic: no fill probability, synthetic fill, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claim is supported.

## 0712 Controller Hygiene Finding

- At the 0712 controller hygiene checkpoint, latest accepted QA was `0710T001`, not `0709T003`.
- The durable route after `0710T001` is `route_to_public_flow_artifact_repair`.
- The quote/fill manifest field formerly named `accepted_reference_count=4` was ambiguous. It is now split into:
  - `accepted_source_row_count=4`
  - `prior_reference_count=1`
  - `live_artifact_attempt_count=4`
- The next formal task file has been created as `.workflow/tasks/0712T001.md`, status `待执行`.
- `0712T001` is a narrow public-flow interval artifact repair/design task. It does not authorize live retry, threshold changes, quote-envelope changes, fee/PnL calibration, maker viability, T012, promotion, or final MVP pass.

## 0710 T011 Controller Review Clarification

- The accepted T011 route remains `route_to_quote_fill_probability_evidence`.
- The durable interpretation is narrower than "full replay model proven":
  - T011 supports multi-window live artifact, lifecycle, safety, and non-optimistic consistency evidence.
  - T011 does not prove full multi-window replay-engine regression.
  - T011 does not prove fill probability, fee/rebate, realized PnL, stable PnL, maker viability, T012 readiness, promotion, or final MVP pass.
- The four-row T011 synthesis combines:
  - one prior `0708T001` QA-accepted reference via `0708T002`;
  - three newly collected `0709T001` live windows.
- The next work should be planned as quote/fill probability evidence under conservative boundaries. A plan exists at `docs/cross_exchange_quote_fill_probability_evidence_plan.md`, but no formal task has been dispatched from it.

## 0709T002 Batch Same-Window Replay Acceptance Finding

- `0709T002` QA is `已通过`.
- It implemented offline batch replay acceptance over prior `0708T002` accepted replay facts plus the three QA-accepted `0709T001` live windows.
- Output package: `local_live_analysis/cross_exchange_t011_batch_same_window_replay_acceptance_0709T002/`.
- Final recommendation: `batch_same_window_replay_acceptance_passed`.
- Batch rows: `0708T001`, `0709T001_window_01`, `0709T001_window_02`, `0709T001_window_03`.
- Classification counts: `submitted_rejected=1`, `submitted_resting_no_fill=3`.
- All four rows pass market-view, decision-path, lifecycle, economics, optimism, boundary, and overall acceptance.
- This should be read as batch artifact/lifecycle/non-optimistic consistency acceptance. It is not full multi-window replay-engine regression.
- The result remains non-optimistic: no synthetic fill, fee/rebate, realized PnL, latency improvement, queue priority, inventory transition, maker viability, promotion, or T012 claim.
- Accepted next route is T011 T003 multi-window robustness synthesis.

## 0709T001 Controlled Multi-Window Live Evidence Finding

- `0709T001` QA is `已通过`.
- It executed the first task from `docs/cross_exchange_t011_multi_window_auto_loop_plan.md`: `T011-CONTROLLED-MULTI-WINDOW-LIVE-EVIDENCE`.
- Remote execution was on `awsserver1` at commit `df94c9c3880cb91d2fe6a43da0ae58c74f5a8e29`.
- Artifact package: `local_live_analysis/cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z/`.
- It ran three sequential controlled live windows under the accepted envelope: fast Hyperliquid `l2Book`, max order size `0.005 BTC`, max submissions per window `2`, post-only `Alo`, tracked cancel/shutdown proof, and independent final open-orders proof.
- Window 1 classified as `submitted_rejected`: `2` post-only rejects, no fill, shutdown proof pass, independent final open-orders `0`.
- Window 2 classified as `submitted_resting_no_fill`: `1` resting order, no fill, shutdown proof pass, independent final open-orders `0`.
- Window 3 classified as `submitted_resting_no_fill`: `1` resting order, no fill, shutdown proof pass, independent final open-orders `0`.
- Boundary validation passed locally: `177` pulled-back files, `87` JSON files parsed with `0` errors, `81` CSV files parsed with `0` errors, `0` empty files, and no true secret-write flags.
- This evidence does not prove fills, fee/rebate, realized PnL, maker profitability, stable PnL, maker viability, T012 readiness, promotion, or final MVP pass.
- Accepted next route is T011 T002 batch same-window replay acceptance, not threshold/quote/size expansion.

## 0708T002 Same-Window Replay Acceptance Finding

- `0708T002` QA is `已通过`.
- It accepts the single-window `0625T010` same-window replay gate over the accepted `0708T001` fast-L2 live lifecycle.
- Runner: `examples/hyperliquid/cross_exchange_t010_same_window_replay_acceptance.py`.
- Output package: `local_live_analysis/cross_exchange_t010_same_window_replay_acceptance_0708T002/`.
- Final recommendation: `same_window_replay_acceptance_passed`.
- Acceptance results:
  - market-view checks `8/8` pass
  - decision-path checks `10/10` pass
  - lifecycle checks `12/12` pass
  - economics/no-fill attribution checks `6/6` pass
  - optimism checks `8/8` pass
  - boundary status `pass`
- Same-window replay can conservatively represent the `0708T001` facts:
  - fast L2 live source path
  - post-open-orders public-state evidence
  - final guard / edge-gate pass
  - `buy 0.002 BTC @ 63889.0`, post-only `Alo`
  - one real source-artifact order submission
  - `resting` response
  - tracked cancel / shutdown proof
  - no fill
  - final open-orders `0`
  - independent final open-orders `0`
- The replay remains non-optimistic:
  - no synthetic fill
  - no fill probability / fill horizon
  - no fee/rebate
  - no realized PnL
  - no zero-latency assumption
  - no reject-rate generalization
  - no maker viability claim
- This closes the single-window T010 replay/live acceptance gate for the no-fill lifecycle.
- It does not prove stable PnL, maker viability, multi-window robustness, `0625T011`, `0625T012`, promotion, or final MVP pass.

## 0708T001 Fast L2 Watcher Binding / Controlled Live Evidence Finding

- `0708T001` QA is `已通过`.
- The previous `0707T007` observation, "Hyperliquid l2Book cadence was roughly 5 seconds in this live window", was not an exchange limitation. It was caused by the T010 live watcher path not binding the already-supported Hyperliquid `l2Book fast=true` option.
- Code commit `34a77ea` adds `--hyperliquid-l2book-fast`, threads it into `live_public_event_source(...)`, and records the setting in live manifests.
- Focused tests prove:
  - fast subscription adds `fast=true` to `l2Book`;
  - `trades` subscription is not changed;
  - `--event-driven-edge-gate-live --hyperliquid-l2book-fast` passes the flag into the inline reprice live runner.
- Controlled live evidence with fast L2:
  - artifact: `local_live_analysis/cross_exchange_t010_fast_l2book_controlled_live_evidence_0708T001_20260707T160830Z/`
  - remote commit: `34a77eaa490daf26584040fbda5522afbf8b6710`
  - `hyperliquid_l2book_fast=true`
  - l2Book messages `799` over `436.021847s`
  - reconnect count `0`
- Latency conclusion:
  - `open_orders_end_to_public_state` median improved from `5.046153s` in `0707T007` to `0.299699s` in `0708T001`.
  - candidate age at guard stayed below the `1.0s` immediate budget, max `0.890775s`.
  - `post_open_orders_handoff_latency_exceeded` did not recur.
- Execution conclusion:
  - the system submitted one real post-only `Alo` order, got `resting`, observed no fill, canceled/tracked shutdown, and ended with final open-orders `0`.
  - submitted order: buy `0.002 BTC @ 63889.0`, notional `127.778 USDC`.
  - fill count `0`, maker fill count `0`, post-only reject count `0`.
- This is the first accepted same-window live lifecycle artifact for the T010 path with real submit/resting/cancel/no-fill evidence.
- It still does not pass full `0625T010` because replay acceptance has not yet verified the same window, and fill economics / realized PnL remain unsupported without fills.
- Next task should be same-window replay acceptance over this `0708T001` artifact, not another threshold/quote/size change.

## 0707T007 Handoff-Repaired Controlled Live Evidence Finding

- `0707T007` QA is `已通过` as a controlled live evidence rerun, not as full `0625T010`.
- It used the `0707T006` repaired handoff schema under the same conservative live envelope.
- Output package: `local_live_analysis/cross_exchange_t010_handoff_repaired_controlled_live_evidence_0707T007_20260707T150429Z/`.
- Remote execution was on `awsserver1` at commit `fc70a55d4af6dcf4168dc8a38ab40c692aac38c9`.
- Public stream was healthy: `335` l2Book messages, `4379` trades messages, `15028` trade events, reconnect count `0`.
- Trigger/pre-submit evidence improved in volume:
  - `4496` current candidates
  - anti-drift pass/block `30/311`
  - trigger found `true`, trigger count `1`
- Repaired path evidence:
  - edge source binding active: `edge_gate_source_status=decision_time_public_fair_mid_provider`
  - post-open-orders public-state pass/block `15/0`
  - handoff schema active: `handoff_phase=post_open_orders_inline_reprice`
  - decisive guard reason: `post_open_orders_handoff_latency_exceeded`
  - trigger candidate fields and current reprice fields are separated.
- Execution stayed safe:
  - live submissions `0`
  - real order endpoint called `false`
  - real cancel endpoint called `false`
  - fill count `0`
  - final open-orders count `0`
  - independent final open-orders count `0`
- Latency finding:
  - private `open_orders` is not the main delay; median `open_orders_elapsed` is `0.018696s`.
  - post-open-orders public L2 resync dominates; median `open_orders_end_to_public_state` is `5.046153s`.
  - This exceeds the `1.0s` immediate guard age budget and causes fail-closed before order submission.
- Next useful task is not a threshold change. It should diagnose/repair the pre-submit latency budget around post-open-orders public-state resync.
- Full `0625T010` remains blocked because there is still no submitted lifecycle, fill/no-fill economics, fee/rebate, inventory, or PnL evidence.

## 0707T006 Inline Reprice Handoff Contract Repair Finding

- `0707T006` QA is `已通过`.
- It repaired the artifact/schema and guard contract identified by `0707T005`.
- Code commit: `b0a1814 / Repair T010 inline reprice handoff contract`.
- Trigger candidate audit and current reprice decision are now separated in immediate guard artifacts.
- The post-open-orders inline reprice path passes the original trigger candidate into the final guard.
- If private open-orders plus public L2 resync makes the trigger candidate older than the immediate max age, the decisive reason is now `post_open_orders_handoff_latency_exceeded`.
- Current reprice failure is still visible in current-reprice fields instead of being mixed into the primary fail-closed reason.
- Validation artifact: `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_contract_repair_0707T006/`.
- Validation summary:
  - `guard_status=fail_closed`
  - `guard_reason=post_open_orders_handoff_latency_exceeded`
  - `trigger_candidate_quality_bucket=quality_a`
  - `current_reprice_skip_reason=outside_quality_a_b_queue_bands`
  - `live_submissions_count=0`
- This repair intentionally does not make submission easier. It does not change anti-drift, touch-stability, edge thresholds, quote envelope, order size, max submissions, open-orders/L2-resync latency, or live-submit authorization.
- Full `0625T010` remains blocked.
- Next useful task is a separately authorized controlled live evidence rerun using this repaired schema.

## 0707T005 Inline Reprice Handoff Drift Finding

- `0707T005` QA is `已通过`.
- It diagnosed the `0707T004` blocker without code, threshold, quote, size, max-submission, or live-submit changes.
- Source artifact: `local_live_analysis/cross_exchange_t010_repaired_controlled_live_evidence_0707T004_20260707T060126Z/event_driven_edge_gate_live/`.
- Generated package: `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_diagnosis_0707T005/`.
- The repaired post-open-orders public-state resync itself passed: `8/8` rows observed L2 after `open_orders_end_ns`.
- The later inline reprice / immediate pre-submit guard failed closed: `8/8` rows.
- The common failure is age drift:
  - candidate age at guard min `4.899s`, median `5.207s`, max `5.577s`
  - immediate guard max `1.0s`
  - source event to post-open-orders L2 delta min `4911ms`, median `5058ms`, max `5254ms`
- Candidate recomputation drift also appears:
  - `1/8` rows still had selected quote, size, and quality bucket but failed stale-age.
  - `7/8` rows also lost submit-ready intent fields after reprice, producing `missing_intent_limit_px`, `missing_or_nonpositive_intent_size`, and `missing_quality_bucket`.
- Root cause classification:
  - primary: `post_open_orders_handoff_latency_exceeds_immediate_age_guard`
  - secondary: `inline_reprice_recomputed_candidate_often_no_longer_submit_ready_so_intent_fields_disappear`
- This is not primarily a missing live-compatible edge source, post-open-orders public-state freshness failure, private order endpoint failure, post-only reject, or lifecycle/PnL issue.
- Next useful task is a narrow handoff-contract repair before another controlled live evidence rerun.
- The repair should preserve trigger candidate audit fields separately from current reprice decision fields and emit an explicit latency/handoff fail-closed reason.
- Do not change anti-drift thresholds, touch-stability thresholds, quote envelope, size, or max submissions inside that repair.
- Full `0625T010` remains blocked.

## 0707T004 Repaired Controlled Live Evidence Finding

- `0707T004` QA is `已通过`.
- It is accepted as a repaired controlled live evidence attempt, not as full `0625T010`.
- Output package: `local_live_analysis/cross_exchange_t010_repaired_controlled_live_evidence_0707T004_20260707T060126Z/`.
- Remote execution was on `awsserver1` at commit `17de5295e7d7fe2b46eaeccea9d79058c1f65fdf`.
- Public stream was healthy: `336` l2Book messages, `1954` trades messages, `6397` trade events, reconnect count `0`.
- Trigger/pre-submit evidence improved:
  - `2228` current candidates
  - anti-drift pass/block `16/79`
  - trigger found `true`, trigger count `1`
- Task A repair was exercised:
  - `edge_gate_live_compatible_source_available=true`
  - `edge_gate_source_status=decision_time_public_fair_mid_provider`
- Task B repair was exercised:
  - post-open-orders public-state pass/block `8/0`
- Execution stayed safe:
  - live submissions `0`
  - real order endpoint called `false`
  - real cancel endpoint called `false`
  - fill count `0`
  - final open-orders count `0`
  - independent final open-orders count `0`
- The new observed blocker is inline reprice / candidate handoff drift after post-open-orders resync:
  - `outside_quality_a_b_queue_bands;trigger_candidate_stale_before_order;missing_intent_limit_px;missing_or_nonpositive_intent_size;missing_quality_bucket`
- Full `0625T010` remains blocked because no submitted lifecycle, fill/no-fill economics, fee/rebate, inventory, or PnL evidence was produced.
- Next useful task should diagnose or repair the handoff from trigger candidate to final inline reprice candidate after post-open-orders resync.
- Do not change anti-drift thresholds, quote envelope, size, or max submissions until the handoff-drift blocker is understood.

## 0707T003 Anti-Drift Touch-Stability Distribution Finding

- `0707T003` QA is `已通过`.
- This is Task C in `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- The diagnosis used accepted `0706T008` public no-submit and `0706T010` controlled live artifacts.
- Generated package: `local_live_analysis/cross_exchange_t010_anti_drift_distribution_0707T003/`.
- Both windows have nonzero candidate/fresh-touch/anti-drift paths:
  - `0706T008`: `2480` candidates, `125` fresh-touch allowed, anti-drift pass/block `7/118`, edge pass/block `1/6`, would-submit `1`.
  - `0706T010`: `1872` candidates, `112` fresh-touch allowed, anti-drift pass/block `5/107`, trigger `1`.
- Anti-drift eval touch-stability is sparse but not zero at the current `250ms` threshold:
  - `0706T008`: `10/125` retained by touch-stability-only proxy at `250ms`; actual anti-drift pass `7`.
  - `0706T010`: `7/112` retained by touch-stability-only proxy at `250ms`; actual anti-drift pass `5`.
- Dominant candidate skip remains `missing_same_side_strict_through_support`; dominant anti-drift block remains `touch_stability_below_minimum`.
- The diagnosis does not justify changing thresholds inside this loop. A/B repaired concrete mechanism blockers first: missing live-compatible edge source and brittle post-open-orders public-state resync.
- Recommendation is `separately_authorized_controlled_live_evidence_after_a_b_repairs`.
- Any live evidence rerun must be a new formal task with explicit envelope and authorization.
- Do not create a threshold-change task unless repaired controlled live evidence still shows insufficient trigger frequency or repeated fail-closed behavior attributable to anti-drift / touch-stability.

## 0707T002 Post-Open-Orders Resync Repair Finding

- `0707T002` QA is `已通过`.
- This is Task B in `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- The accepted `0706T010` blocker `post_open_orders_public_state_timeout` is addressed at the guard-mechanism level.
- The default resync wait is no longer a fixed short `0.2s`; it is bounded and cadence-aware, with base `0.2s` and max `6.0s`.
- The safety invariant is unchanged: a pass requires a Hyperliquid L2 local receive timestamp strictly after `open_orders_end_ns`.
- Local artifact package: `local_live_analysis/cross_exchange_t010_post_open_orders_resync_0707T002/`.
- The positive case proves `post_open_orders_public_state_pass_count=1`; the negative case remains fail-closed with `public_source_exhausted_before_post_open_orders_l2`.
- This does not change anti-drift thresholds, touch-stability thresholds, edge thresholds, quote envelope, order size, or live authorization.
- Full `0625T010` remains blocked.
- Next useful task is Task C: diagnose live anti-drift / touch-stability distributions before any threshold decision.

## 0707T001 Live-Compatible Edge Source Binding Finding

- `0707T001` QA is `已通过`.
- This is Task A in `docs/cross_exchange_t010_execution_handoff_repair_auto_loop_plan.md`.
- The accepted `0706T010` blocker `edge_gate_source_status=missing_live_compatible_source` is repaired for the CLI `--event-driven-edge-gate-live` path.
- The implementation binds `BinancePublicBookTickerProvider()` as the default `binance_public_state_provider` for the event-driven edge-gate live runner.
- Focused coverage verifies that the CLI path supplies the provider while keeping `anti_drift_gate=True` and `edge_gate=True`.
- Local artifact package: `local_live_analysis/cross_exchange_t010_live_compatible_edge_source_0707T001/`.
- The no-submit/block scenario `insufficient_edge_block` proves the source is available and edge rows populate without a mock order call: `edge_gate_source_status=decision_time_public_fair_mid_provider`, `fair_mid_source_pass_count=1`, `edge_gate_block_count=1`, `live_submissions_count=0`, `mock_order_call_count=0`.
- This does not solve `post_open_orders_public_state_timeout`.
- This does not authorize live-submit, threshold changes, quote-envelope changes, size changes, full T010, T011, T012, stable PnL, maker viability, promotion, or final MVP pass.
- Next useful task is Task B: repair the post-open-orders public-state resync guard while keeping stale/no-proof cases fail-closed.

Branch / fact-source rule:

- `cross-exchange` is the canonical branch for formal Binance-lead / Hyperliquid-lag MVP work. Temporary or recovery branches may preserve useful history, but they are not workflow facts until restored into `cross-exchange`.
- The M-A / M-B / M-C / M-D milestone sequence in `docs/cross_exchange_maker_mvp_plan.md` is the highest branch constraint.
- Historical classification is recorded in `docs/cross_exchange_mvp_task_classification.md`.

## 0706T010 Controlled Live Evidence Finding

- `0706T010` QA is `已通过` as a controlled live evidence attempt, not as full `0625T010`.
- The accepted package is `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/`.
- The run completed `1800.001512s` on `awsserver1` at commit `da12034faa5bb787fe94399f445e79db29777c7e`.
- Public stream was healthy: `337` l2Book messages, `1541` trades messages, `4949` trade events, reconnect count `0`.
- Trigger/pre-submit evidence exists: `1872` candidates, anti-drift pass/block `5/107`, trigger found `true`, trigger count `1`.
- The run failed closed before order submission with `post_open_orders_public_state_timeout`.
- The live-compatible edge/source path is also incomplete: `edge_gate_source_status=missing_live_compatible_source`.
- Execution stayed safe: live submissions `0`, `real_order_endpoint_called=false`, `real_cancel_endpoint_called=false`, fill count `0`, final open-orders count `0`, independent final open-orders count `0`.
- Full `0625T010` remains blocked because submitted lifecycle/economics/PnL evidence is still absent.
- Next useful work is not another blind live repeat; it is a repair/design task for live-compatible edge/source binding and post-open-orders public-state resync.

## 0706T008 Long-Window No-Submit Finding

- `0706T008` QA is `已通过`.
- The new three-task auto-loop plan is `docs/cross_exchange_t010_candidate_live_auto_loop_plan.md`.
- The accepted long-window no-submit package is `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`.
- In `1800.001312s`, public streams were healthy: `336` l2Book messages, `2144` trades messages, `6881` trade events, reconnect count `0`.
- The no-submit funnel produced `2480` candidates, `2170` fresh-touch evidence passes, `125` fresh-touch allowed candidates, `7` anti-drift passes, `1` fair-mid source pass, `1` edge pass, and `1` shadow would-submit.
- The would-submit row is event `1369`, `buy`, quote `63019`, quality `quality_a`, fair-mid source age `43ms`, edge `25.5` ticks, action `would_submit_if_real_order_task_authorized`.
- Boundary held: credentials/private/account/order/cancel/live-client all false, `no_submit_enforced=true`, `real_orders_allowed=false`.
- This changes the route: the `0706T007` no-submit result was a short-window miss, not proof that the gate cannot produce candidates.
- The auto-loop should route to `0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE`, not to `0706T009` repair.

## 0706T007 Live Evidence Finding

- `0706T007` QA is `已通过` for the authorized live evidence acquisition attempt.
- User authorization was explicit in-session: `继续，授权live evidence任务`.
- The authorization task file was committed before live execution at commit `cbee781`.
- The live evidence package is `local_live_analysis/cross_exchange_mvp_t010_live_evidence_0706T007/`.
- The accepted recommendation is `full_t010_live_evidence_blocked_no_order_submitted`.
- Remote checkout was synced to `cbee781069456bc0fecdddaa1d7297eaf546e7ce` with dirty count `0`.
- Under the authorized fresh-touch envelope, public precheck produced `10` fresh-touch candidates but `0` eligible candidates. The runner submitted `0` orders.
- `real_order_endpoint_called=false`, `real_cancel_endpoint_called=false`, `fill_count=0`, final open-orders count `0`, and independent final open-orders count `0`.
- This is a useful fail-closed live evidence result: it proves the authorized envelope can stop before submit when same-window evidence is insufficient.
- It does not produce submitted-order lifecycle, fill/economics/PnL, or cross-exchange signal/fair-mid/quote-intent live decision evidence.
- Full `0625T010` remains blocked. Further progress requires either a new evidence route that links the accepted cross-exchange signal kernel to live decisions, or a separately authorized envelope change; neither is authorized by `0706T007`.

## 0706T006 Full T010 Preflight Finding

- `0706T006` QA is `已通过` for `0625T010-FULL-PREFLIGHT Live Evidence Acquisition Packet`.
- The preflight package is `local_live_analysis/cross_exchange_mvp_t010_full_preflight_0706T006/`.
- The accepted recommendation is `full_t010_live_evidence_acquisition_blocked_pending_authorization`.
- Full `0625T010` still lacks complete same-window live evidence: public market view, decision path, linked lifecycle, latency/ordering, fill/no-fill economics, fee/rebate, inventory transition, PnL ledger, shutdown, and independent final open-orders proof.
- The future live evidence task must be separately created and explicitly authorized. `0706T006` does not authorize live-submit, repeated-window, fill-seeking behavior, quote-envelope change, size change, full T010 execution, T011, T012, PnL claim, maker viability claim, promotion, or final MVP pass.
- If a future evidence window has no fill, it may support fail-closed no-fill economics only. It must not create a fill model, fee model, PnL claim, or maker viability claim.
- The next executable MVP-forward step is not full T010 itself; it is a new live evidence acquisition task after explicit authorization of the exact envelope.

## 0706T005 Scoped Replay Acceptance Finding

- `0706T005` QA is `已通过` for `0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance`.
- The scoped acceptance package is `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/`.
- The accepted recommendation is `scoped_same_window_replay_acceptance_passed`.
- Supported facts matched `12/12`: order intent, `Alo`, submit path, resting status, primary tracked cancel, shutdown proof, and independent final open-orders count `0`.
- Unsupported fail-closed checks passed `11/11`: submit/ack latency, resting duration, cancel latency, cancel-fill race, fill horizon, fill probability, fee/rebate, inventory transition, realized PnL, stable PnL, and maker viability.
- Optimism checks passed `8/8`: no fill probability, fill horizon, fee/rebate, inventory, realized PnL, reject-rate-zero, zero-latency, or maker viability assumption was inferred.
- This proves the current replay acceptance contract can honestly represent the one observed submit/resting/cancel/open-orders case without inventing economics.
- This does not prove a general execution model, fill model, fee/PnL model, stable PnL, maker viability, full T010, T011, T012, or final MVP readiness.
- The next MVP-forward task should not keep optimizing this one-order scoped case. It should either stop for human decision or create a separately authorized live evidence acquisition/preflight task to gather complete lifecycle/economics/PnL evidence.

## 0706T004 Roadmap Refresh Finding

- `0706T004` QA is `已通过` for the roadmap refresh after `0706T003 / 0625T009`.
- The MVP target remains unchanged: prove Binance lead signal -> Hyperliquid decision-time fair value -> post-only maker quote intent -> production-equivalent public shadow -> tiny-live lifecycle/economics evidence -> same-window replay -> replay/live acceptance.
- Current accepted evidence does not yet cover the full target. T009 supports only one-order submit/resting/cancel/open-orders facts from `0706T002`.
- The next automatic route is now `0706T005 / 0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance`: local/offline/no-live replay of the supported one-order facts with unsupported fill/cost/PnL/viability fields kept fail-closed.
- A scoped T010 pass would be a replay sanity/contract acceptance only. It must not unlock T011, final MVP validation, stable PnL claims, maker viability claims, or any additional live-submit.
- Full `0625T010` still requires new complete live evidence covering market view, decision path, submit/cancel/reject/fill, fee/rebate, inventory/PnL attribution, shutdown, and open-orders proof.
- Any additional live-submit, repeated-window, fill-seeking, closer-to-market placement, size change, or quote-envelope change requires a new formal task and explicit authorization.

## 0625 Cross-Exchange Maker MVP Sequencing Finding

- The MVP should not begin by rebuilding the full Binance alignment stack for Hyperliquid, and it should not proceed directly from public lead-lag research to parameterized live maker execution.
- The shortest defensible sequence is:
  - alpha/edge decomposition and multi-window signal acceptance
  - production-equivalent public shadow using a shared decision kernel
  - minimal Hyperliquid market-view and lifecycle replay contract
  - edge-qualified tiny-live calibration
  - same-window replay/live acceptance
  - multi-window final MVP validation
- Existing work is reusable: synchronized public joins, lead-lag features, Hyperliquid raw conversion, real `Alo` order/cancel/shutdown mechanics, fail-closed PnL ledger, and the current event-driven public watcher.
- Current first blocker is signal/edge decisionability. `0624T003` reaches fresh-touch allowed rows but produces no edge pass, so a live canary or quote-distance relaxation would mix an unresolved alpha problem with execution risk.
- The formal roadmap is `docs/cross_exchange_maker_mvp_plan.md`. `0625T001` is complete; `0625T002` found and repaired the effective-horizon contract issue; `0627T001` QA accepted the corrected HL fast sample package. `0625T003` business execution is complete and awaits QA; later roadmap tasks remain controller-gated and sequential.

## 0625T003 Signal Acceptance Business Finding

- T003 must be an out-of-sample signal acceptance task, not another sample collection task and not a live/shadow task.
- The accepted `0627T001` package unlocks creation/dispatch only because it provides near-target nominal `1000ms` rows: T003 must still filter rows by effective horizon and report nominal/effective semantics explicitly.
- The main T003 risk is leakage through same-window threshold/side selection. The task draft therefore requires leave-one-window-out or equivalent documented train/evaluation separation, forbids same-window threshold backfill, and rejects candidates whose side mapping flips in held-out windows.
- Candidate features must remain decision-time-visible and allowlisted; T003 should not run unconstrained feature/model/parameter search.
- Passing T003 may only recommend `signal_contract_accepted_for_shadow`, which unlocks a later production-equivalent shadow task. It does not authorize watcher changes, private/order endpoints, live orders, canary, promotion, or Hyperliquid alignment/replay claims.
- Business execution implemented this scope in `examples/hyperliquid/cross_exchange_signal_acceptance.py` and produced `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`.
- QA accepted the business recommendation `signal_contract_accepted_for_shadow`. The accepted candidate is `binance_lead_composite` over Binance top5 imbalance, Binance microprice-minus-mid, and Binance short mid move; selected threshold is `abs(z) >= 1.0`; side mapping is `positive_signal_buy_negative_signal_sell`.
- Held-out adjusted edge proxy is positive in all three windows (`13.52309469 / 1.89788732 / 4.02109181` ticks), with nonzero direction hit `0.76530612 / 0.87234043 / 0.96428571` and max held-out contribution `0.38660714`.
- The main caveat is bucket stability: one source-age bucket in `xemm_0627_t001_hlfast_utc17_b` has adjusted proxy `-0.93820225` over `89` active rows. This does not change the business recommendation, but QA and T004/T005 should keep it visible.
- T003 acceptance freezes the MVP v1 signal contract for the next public-shadow stage only. It does not authorize watcher/live changes, private/order endpoints, live orders, canary, production promotion, or Hyperliquid replay/alignment claims.

## 0625T004 Shared Signal / Quote-Intent Kernel Finding

- T004 QA accepted `examples/hyperliquid/cross_exchange_shared_signal_kernel.py` as the shared pure decision kernel for the next production-equivalent public shadow and later replay path.
- The kernel consumes the QA-accepted T003 signal contract and produces deterministic signal, side, fair-mid, touch quote intent, edge, action, and block reason outputs from explicit public market-view fields.
- Normalization stats are explicit kernel inputs; this avoids hidden same-window recalibration or threshold/side retuning inside T004.
- Fixture artifacts under `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/` cover buy would-submit, sell would-submit, below-threshold block, missing-feature block, and T003 warning-bucket visibility.
- T004 did not modify the existing watcher public-shadow path. T005 should consume this module when implementing production-equivalent public shadow; otherwise the shared-kernel/replay invariant would not yet be exercised by real shadow artifacts.
- The fixture parameter `expected_move_ticks_per_signal_z=4.0` is deterministic infrastructure evidence only, not a production profitability claim. T005 must validate would-submit count and counterfactual edge on fresh public windows before any live boundary can be considered.

## 0625T005 Production Shadow Finding

- T005 QA is `已通过`.
- T005 business execution used the T004 shared kernel over the accepted `0627T001` three-window public package and produced no-submit production shadow artifacts.
- Would-submit sample size is no longer zero/thin: `1098` aggregate, with `429/291/378` per window.
- Mean adjusted counterfactual edge is positive per window (`13.54662005 / 2.02233677 / 4.33333333` ticks) and aggregate (`7.32058288` ticks), and max window contribution is `0.39071038`.
- The T003 warning bucket remains visible: `1231` warning-bucket decisions and `91` warning-bucket would-submit rows.
- QA accepted the final recommendation `production_shadow_accepted_for_replay_contract`.
- Important caveat: median adjusted counterfactual edge is `-1.5` ticks due to zero 1000ms moves after subtracting the buffer. This does not block the accepted recommendation, but T006 should preserve the caveat when defining audit/replay contracts.
- T005 remains public-only/no-submit and does not authorize live execution, private/order endpoints, canary, promotion, or replay/live alignment claims.

## 0625T006 Audit Replay Contract Finding

- T006 QA is `已通过`.
- T006 defines the MVP audit/replay contract and validator.
- The contract schema has `63` fields across identifiers, venues, timestamps, source ages, market view, signal, quote intent, decision, lifecycle, economics, and boundary categories.
- Schema hash is `0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5`.
- Synthetic lifecycle fixtures cover accepted decision/submit/resting/cancel/reject/partial-fill/full-fill/block rows and fail-closed missing/unsafe rows.
- Existing artifacts are classified conservatively:
  - T005 production shadow is accepted only as a partial public-market/signal/quote/markout contract.
  - M1 repeated canary is accepted only as submit/resting/cancel-shutdown lifecycle reference without fill/PnL.
  - M2 ledger is accepted as fail-closed ledger reference without realized PnL proof.
- T006 preserves the T005 median adjusted-edge caveat and T003 warning bucket for later replay diagnostics.
- T006 does not prove public replay alignment, does not authorize live execution, and does not authorize private/order endpoints, canary, promotion, or replay/live alignment claims.
- T006 acceptance unlocks controller creation/dispatch of T007 public market-view replay alignment only.

## 0625T007 Public Replay Alignment Finding

- T007 QA is `已通过`.
- T007 replays the T004 shared kernel over the QA-accepted `0627T001` aligned public context package and compares against T005 production-shadow decisions.
- Replay source is the accepted aligned public context rows; raw `0627T001` WebSocket files are not present in the local repository, and no new public collection was performed.
- Replay matched T005 action path exactly: `10704/10704` decision rows matched, with `0` action mismatches and `0` unexplained mismatches.
- Market-view/source-age/cadence gates passed: market-view fail-closed count `0`, cadence/source-age gate fail count `0`.
- Future-label exclusion gate passed: future decision input field count `0`, future timestamp not-after-decision count `0`.
- QA accepted the recommendation `public_market_view_replay_alignment_ready_for_qa`.
- T007 remains offline/public-only/no-submit and does not authorize live execution, private/order endpoints, canary, promotion, T008 live submit, or final MVP pass.
- Next controller action may prepare `0625T008-PREFLIGHT` only; first live-submit `0625T008` still requires standing live authorization or explicit controller approval.

## 0706T001 / 0625T008-PREFLIGHT Finding

- `0706T001` QA is `已通过` as the valid workflow task for `0625T008-PREFLIGHT Edge-Qualified Tiny-Live Calibration Packet`.
- The accepted packet is under `local_live_analysis/cross_exchange_mvp_t008_preflight_packet_0706T001/`.
- Final recommendation is `live_submit_blocked_pending_controller_authorization`.
- The preflight packet preserves accepted `0625T005`, `0625T006`, and `0625T007` as prerequisites, while explicitly excluding archived-invalid `0702T001` from reusable signal/replay evidence and accepting `0702T002` as the collector fail-fast repair.
- The active preflight envelope is zero-submit: max order count, size, notional, position delta, and max loss are all `0`.
- No standing live authorization record exists for this exact `0625T008` envelope, so live-submit remains blocked.
- `0625T008` live-submit was not created, authorized, or executed.
- Boundary remains local/offline artifact-only with no network, AWS, remote, credentials, live client, private/account/order/cancel endpoints, signing, nonce, user stream, order placement, cancellation, live bot, canary, promotion, or final MVP pass.

## 0706T002 / 0625T008 Live-Submit Calibration Finding

- `0706T002` QA is `已通过` as the first live-submit calibration corresponding to `0625T008`.
- The user/controller explicitly authorized the live-submit in-session with `授权 first live-submit calibration` / `授权开始`.
- The accepted artifact package is `local_live_analysis/cross_exchange_mvp_t008_live_submit_calibration_0706T002/pulled_back_awsserver1/`.
- Remote execution was on clean `awsserver1` `cross-exchange` checkout at commit `25b444e31`, using `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python` with Hyperliquid SDK available.
- One real Hyperliquid order submission was attempted: `BTC` buy, `0.01 BTC`, limit `62146.0`, notional `621.46 USDC`, post-only `Alo`.
- The order reached `resting`, tracked cancel/cancel-by-cloid ran, and independent final open-orders check returned `0`.
- Artifacts report `credentials_written=false`, `secret_values_written=false`, and `raw_signatures_written=false`; redaction scan found no raw credential/private-key/signature values.
- This calibrates first submit/resting/cancel/open-orders evidence only. It does not authorize another live-submit, repeated window, fill-seeking run, integrated strategy run, parameter relaxation, default-on behavior, promotion, final MVP pass, or claims of stable PnL / maker viability.
- Caveat: remote checkout was not current local documentation HEAD; the run used the already-present remote real-order canary executor path and should be treated as one-order live calibration evidence, not as proof of latest local workflow deployment.

## 0706T003 / 0625T009 Execution Outcome Calibration Finding

- `0706T003` QA is `已通过` as the execution outcome calibration corresponding to `0625T009`.
- The accepted package is `local_live_analysis/cross_exchange_mvp_t009_execution_outcome_calibration_0706T003/`.
- The calibration consumes only the one-order `0706T002 / 0625T008` pulled-back artifact.
- Supported facts are limited to submit endpoint reachability for this exact envelope, post-only `Alo`, order response `resting`, primary tracked cancel success, and independent final open-orders count `0`.
- Post-only reject was not observed, but this is not a reject-rate estimate.
- Unsupported domains remain explicit: submit/ack latency, resting duration, cancel latency, cancel-fill race, fill horizon, fill probability, fee/rebate, inventory transition, realized PnL, stable PnL, and maker viability.
- The next replay work may consume only the supported submit/resting/cancel/open-orders facts; it must not invent fill, fee, inventory, PnL, or viability parameters.
- Any next live fill-seeking or repeated-window task still requires a new formal task and explicit authorization.

## 0625T002 Sample Expansion Contract Finding

- `0702T001` found a new collector-side reliability bug: Binance REST depth snapshot can be rate-limited on the shared `awsserver1` public IP before WebSocket collection otherwise succeeds.
- The observed Binance error was HTTP `429` on IP `18.182.23.227`; the `2400 requests per minute` text is the IP-level limit description, not direct evidence that the collector itself issued `2400/min` snapshot requests.
- Prior accepted `0627T001` did not have this problem: all three Binance depth snapshots were HTTP `200` with valid `lastUpdateId` and non-empty bids/asks, and complete/valid context rows were present.
- The defect in the collector was fail-open behavior: HTTP `429` snapshot bodies were written into raw data and the collection process still returned success, causing local alignment to discover `snapshot_alignment_status=missing` too late.
- `0702T001` QA accepted the scheduled collection only as a fail-closed invalid dataset archive. It remains `sample_collection_invalid`, `t003_creation_unlocked=false`, and must not be used for signal acceptance, production shadow, replay alignment, live submit, canary, or promotion.
- The invalid dataset archive record is `local_live_analysis/archive/0702T001_INVALID_DATASET_ARCHIVE.md` plus `local_live_analysis/archive/0702T001_invalid_dataset_archive_manifest.json`; current worktree does not contain 0702T001 data directories, so this is a lightweight committed archive record rather than a raw-data tarball.
- `0702T002` QA is `已通过` and fixes the collector contract: top5 bootstrap snapshot defaults to depth `100`, rate-limit responses are retried with bounded low-frequency backoff, manifests record attempt/rate-limit evidence, and missing valid snapshot now fails collection instead of producing a top5-empty sample.
- `0627T001` was temporarily blocked by remote Binance alignment OOM after first-window collection evidence. The interface fix is implemented in commit `6392d8d`, local focused tests pass, AWS 60s fast smoke observed `l2Book=112`, and the first formal 1800s HL fast manifest shows `l2Book=3335` with `l2book_fast=true` and `reconnect_count=0`. The failing step was remote `binance_top5_provenance.py build-sidecars --buffer-size 10000000` on `bookTicker=1188137`, which was killed at about `3.4G` memory on the small no-swap instance. Disk was not exhausted. The blocker was resolved by making `awsserver1` raw-only and running alignment after copyback.
- New operating rule: `awsserver1` must only collect public raw data for this task. Binance/Hyperliquid alignment and downstream processing must run on macmini or amdserver. AWS collection commands should use `--skip-alignment` and the manifest must mark `raw_collection_only=true`.
- `0627T001` business execution completed after applying the raw-only AWS rule. The final package `cross_exchange_mvp_hl_fast_sample_expansion_0627T001` has `10745` complete symmetric 1000ms contexts and `10704` valid near-target 1000ms signal contexts. Recommendation is `sample_contract_ready_for_signal_acceptance`; `t003_creation_unlocked=true` after QA/controller review.
- `0627T001` QA is now `已通过`. This unlocks controller creation/dispatch of T003 only; it does not freeze signal shape, side mapping, edge formula, freshness limits, watcher behavior, live orders, canary, or promotion.
- Follow-up task `0627T001` is created to test HL `l2Book fast=true` public collection and rerun the synchronized sample expansion. It is not T003 and does not weaken the repaired near-target horizon gate.
- Effective-horizon gate repair: the prior T002 acceptance was too permissive because it reported effective age but did not make near-target label coverage a fail-closed gate.
- The repaired contract separates field completeness from signal-label validity. `complete_context=true` only means fields are present; `valid_for_1000ms_signal_acceptance=true` now requires both complete fields and near-target effective horizon.
- For nominal `1000ms`, the near-target gate is `1000ms <= effective_future_age_ms <= 1250ms`.
- Current repaired T002 has complete context rows `668/666/665`, but valid near-target `1000ms` signal rows only `2/0/1`, aggregate `3`.
- The repaired recommendation is `needs_more_public_samples`, with `t003_creation_unlocked=false`; T003 must not be created from the current sample package.
- Prior QA accepted T002 before the effective-horizon gate was introduced; that acceptance is superseded by the repair and should not be used to create T003.
- Prior QA independently verified raw checksums, JSON/CSV schemas, context completeness, deterministic package reproduction, boundary flags and focused tests; those evidence checks remain useful, but the repaired effective-horizon gate changes the recommendation.
- The only neighboring-suite gap is environmental: an older integration test requires missing local sample `cross_exchange_public_sample_0602T001`; T002 task-scoped and remaining neighboring tests pass.
- T002 business execution collected and locally processed all three required new AWS windows. All raw checksums match and all reconnect counts are zero.
- Synchronized overlaps are `1799.999859s`, `1800.036434s`, and `1800.059208s`; starts are separated by more than 30 minutes.
- Decision-time-only relative regime classification assigns high, normal, and low activity/liquidity. The high bucket has mean Binance rolling RV `9.20857357` ticks and combined public trade-event rate `229.19660223/s`; normal/low are about `6.30155019/6.04992684` ticks and `150.0925175/148.65788796/s`.
- Local joins contain `3596/3596/3595` rows with `671/669/667` primary rows. Future, missing, and stale Binance join counts are zero in every window.
- Complete symmetric 1000ms edge contexts are `668/666/665`, `1999` aggregate. Every complete row retains current HL bid and ask as separate buy-touch/sell-touch alternatives plus dual top5, signal inputs, basis/context, timestamps/source ages and future labels.
- Effective horizon is materially different from nominal horizon because the accepted synthetic HL decision series is sparse after primary filtering: nominal 1000ms labels have effective median `5000ms` in all three windows and means about `5111-5140ms`. T003 must use effective-age controls and must not interpret nominal horizon as exact elapsed time.
- Prior recommendation `sample_contract_ready_for_signal_acceptance` is superseded by repaired recommendation `needs_more_public_samples`. T002 still does not accept a signal, freeze side mapping, or authorize live behavior.
- T001 establishes that existing historical alpha is promising but production evidence remains too thin, with only four edge rows and no production anti-drift future markout.
- T002 therefore expands evidence rather than tuning the signal or execution policy.
- Three new AWS public-only windows are required, not reuse of historical samples as substitutes.
- Each window targets `1800s`, must retain at least `1500s` synchronized overlap, and the accepted set must cover at least two observed public volatility/liquidity regimes.
- To avoid leaking T003's responsibility, T002 does not select a maker side. It preserves both Hyperliquid touch alternatives and counts complete symmetric edge-evaluable contexts.
- Minimum coverage is `100` complete contexts aggregate and `20` per window with dual top5, signal inputs, basis/venue state, timestamps/source ages and future labels.
- Only `sample_contract_ready_for_signal_acceptance` may unlock T003; the repaired package does not meet that condition. This task does not authorize live orders, private/order endpoints, canary, strategy relaxation or promotion.

## 0625T001 Alpha / Edge Decomposition Finding

- Original business execution was rejected by QA; repair commits `27c08dd` / `72f4cb4` are now `已通过`.
- Recommendation is `needs_more_public_samples`, not `signal_contract_candidate`.
- Historical signal evidence is directionally promising:
  - `3596` primary rows, future join `0`, missing Binance join `0`, stale Binance source `0`.
  - all four allowlist features are positive and `stable_across_samples` at `1000ms` across three canonical event-mode samples.
- Production evidence is currently too thin:
  - `599` candidate rows
  - `68` fresh-touch/anti-drift rows
  - `64` anti-drift blocks, `4` passes
  - `3` fair-mid passes, `1` stale block
  - `4` edge rows, `0` edge passes
- Fresh production edge values are `-24.5`, `-24.5`, and `0.5` ticks. Lowering the current seven-tick edge requirement would produce only one pass at a zero-tick diagnostic threshold and still zero passes at one tick or above.
- All edge-evaluated production candidates are buy. The two negative-edge rows use `lead_move_ticks=-25`; the third fresh row uses `0`. This shows the current candidate side is not yet bound to a frozen lead signal contract.
- Production anti-drift rows do not contain same-window future markout. It is not defensible to claim that the gate is over-filtering or correctly filtering from current artifacts.
- The QA defects are repaired:
  - effective-horizon count/min/mean/max/offset and timing status are emitted
  - nominal `100/250ms` labels are materially delayed to about `500.417ms`; `500/1000ms` are aligned
  - basis and Hyperliquid spread/top5 imbalance/microprice/join-age conditioning is emitted with fail-closed bucket coverage
  - full-range `git diff --check b21afff..27c08dd` passes
- The historical sample shows material conditional association at `1000ms`, including basis bucket range `47.0199146` ticks and Hyperliquid top5 imbalance range `34.33546961` ticks. This is diagnostic association only, not causal proof or a frozen production signal.
- Repeat QA independently reproduced the ten artifacts, raw effective-horizon statistics, conditioning ranges, root-cause assessments and safety boundary. The task is accepted.
- T001 acceptance does not invalidate or change the current `needs_more_public_samples` recommendation.
- The next evidence task should collect at least three separated 30-minute public windows with complete dual-top5, signal component/composite, side/quote, gate, source-age and future-label fields. It must remain public-only/no-submit.

## 0624T003 QA / Current Alpha-Edge Finding

- `0624T003` QA is `已通过`. The same QA sweep also accepted `0622T006`, `0623T006`, `0623T007`, and `0623T010`.
- T002 repaired BBO history/cache fields are validated in fresh AWS public live mode, not only in replay: `candidate_count=599`, `repaired_synthetic_current_event_only_count=2`, `repaired_fresh_touch_evidence_pass_count=502`, `same_touch_reset_supported_count=198`, and `fresh_touch_allowed_count=68`.
- The live funnel no longer primarily fails at BBO history/cache or fresh-touch evidence. After `68` fresh-touch allowed rows, `64` are blocked by anti-drift and only `4` reach fair-mid/edge; `edge_gate_pass_count=0`.
- Edge detail remains weak: the rows reaching edge are blocked by `edge_below_required_buffer=3` or `fair_mid_source_stale=1`, so the current lead-lag/fair-mid path does not yet show enough positive tradable edge.
- This is a task-level acceptance only. It does not authorize real canary, live orders, credential reads, private/account/order endpoints, quote-distance/cap/post-only relaxation, M3 readiness, stable PnL, default-on behavior, or promotion.
- The next useful task should be public-only/no-submit alpha/edge decomposition: anti-drift semantics, fair-mid source freshness/timing/direction, lead-lag markout, and edge buffer calibration. It should not start by loosening quote distance, caps, post-only behavior, or private/order boundaries.

## 0623T010 Candidate Funnel Diagnosis Finding

- `0623T010` QA is `已通过`. It ran a 600s AWS public-only no-submit shadow window and produced a row-level funnel diagnosis.
- The first blocking stage is before Binance freshness, fair-mid source, and edge gate: `fresh_touch_gate_allowed=0` out of `1259` public candidate evaluations.
- Public flow was not absent: `at_or_through_trade_seen=1054`, `strict_trade_through_seen=299`, and `visible_top_plus_order_depleted=129`; however accepted fresh-touch / dynamic-size eligibility still never allowed a candidate.
- Dominant blockers are `missing_touch_freshness_or_queue_reset_evidence=1257`, `missing_same_side_strict_through_support=960`, and `missing_recent_same_side_at_or_through_throughput=205`.
- This diagnosis suggests the next repair should focus on fresh-touch evidence generation / BBO-history continuity / queue-reset recognition under the public shadow path, while continuing to reject quote-distance changes, cap relaxation, private/order endpoints, and canary authorization.

## 0624T002 Prepared BBO Evidence-Chain Repair Finding

- `0624T002` QA is `已通过`.
- It repairs the four coupled evidence-chain items: BBO history/cache visibility, fresh-touch block reason taxonomy, same-touch queue-reset evidence, and event ordering / local visibility diagnostics.
- The implementation keeps evidence quality separate from strategy loosening: `synthetic_current_event_only` remains fail-closed, accepted fresh-touch / queue-reset pass criteria are not weakened, and quote distance / cap / post-only / private-order boundaries are unchanged.
- T010 replay validation shows the original generic blocker was mostly BBO-history/cache accounting. `synthetic_current_event_only` drops from `1257` in T001 diagnosis to `6` after repaired local-receive-order reconstruction; `same_touch_stable_enough=1068`, `same_touch_reset_supported=317`, `last_l2_too_old=83`, `same_touch_seen_but_not_stable=102`, and `bbo_history_too_sparse=6`.
- Local receive ordering is clean in this replay (`1259/1259` latest L2 received before or at candidate), while exchange timestamp conflict remains rare (`1` row). The remaining dominant blocker is `bbo_evidence_repaired_remaining_blocker_is_flow_or_downstream_gate`, not Binance freshness or fair-mid/edge itself.

## 0624T003 Prepared AWS Repaired Public-Shadow Live Validation Finding

- `0624T003` QA is `已通过`.
- It verified that T002 repaired BBO history/cache fields populate in a fresh `awsserver1` public stream, not only in T010 replay.
- The fresh 600s public-only no-submit sample produced `599` candidate evaluations from `112` l2Book messages and `487` trade messages (`2077` expanded trade events), with `source_path_exercised=true` and `reconnect_count=0`.
- BBO evidence repair held in live mode: `repaired_synthetic_current_event_only_count=2` (`0.33389%`), `repaired_fresh_touch_evidence_pass_count=502`, `same_touch_stable_enough_count=502`, `same_touch_reset_supported_count=198`, and `fresh_touch_allowed_count=68`.
- The main blocker moved downstream: after `68` fresh-touch allowed rows, `64` were blocked by anti-drift (`touch_stability_below_minimum=59`, `adverse_trade_pressure_with_recent_adverse_bbo=5`); `4` reached fair-mid/edge, `3` passed fair-mid source, and `0` passed edge (`edge_below_required_buffer=3`, `fair_mid_source_stale=1`).
- `shadow_would_submit_count=0`; no real canary is authorized. The result supports focusing next on anti-drift threshold/source semantics and edge/fair-mid live source quality, not BBO history/cache repair or fresh-touch evidence availability.
- This task did not authorize quote-distance changes, cap relaxation, post-only weakening, accepted fresh-touch / queue-reset weakening, credential reads, private/order endpoints, canary execution, M3 readiness, stable PnL, default-on behavior, or promotion.

## 0623T009 AWS Public Shadow Soak Finding

- `0623T009` business execution is `已通过`. It combines the requested AWS live public no-submit shadow soak and canary preflight ledger into one `awsserver1` task.
- The task first hit an environment blocker on `awsserver1`: system `/usr/bin/python3` had no `websockets` and no `websocket-client`, so the public stream path could not start there.
- The task recovered with the existing remote venv `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`, which does have `websocket-client` and Hyperliquid available.
- Remote live public data was observed successfully: `l2Book=34`, `trades=142`, `subscription_ack=2`, `reconnects=0`, `duration_elapsed`.
- Shadow evidence stayed fail-closed: `current_candidate_count=176`, `shadow_evaluation_count=176`, `shadow_would_submit_count=0`, `fair_mid_source_pass_count=0`, `edge_gate_pass_count=0`, and no private/order endpoint was called.
- A controller-requested venv rerun reconfirmed this on `awsserver1`: 180s `duration_elapsed`, `l2Book=35`, `trades=142`, `total_trade_event_count=418`, `subscription_ack=2`, `reconnects=0`, `current_candidate_count=177`, `shadow_evaluation_count=177`, `shadow_would_submit_count=0`, `fair_mid_source_pass_count=0`, `edge_gate_pass_count=0`, and `source_path_exercised=false`.
- The canary preflight ledger is intentionally blocked: `live_public_source_observed=true`, `shadow_would_submit_count=0`, `source_path_exercised=false`, `final_recommendation=hyperliquid_tiny_live_m2_canary_preflight_blocked`, `next_real_canary_authorized=false`, and `live_realized_pnl_proof=false`.
- This finding does not authorize a real canary, credential reads, private/account/order endpoints, or realized PnL claims.

## 0623T007 Public Shadow Source Finding

- `0623T007` QA is `已通过`. It implements a no-submit public shadow source path for the T006 fair-mid provider and T004 edge gate.
- Policy: `m2_live_public_source_shadow_v1`. The path consumes Hyperliquid public L2/trades plus Binance public bookTicker-compatible state, then records fair-mid source rows, edge-gate rows, public source freshness rows, candidate audit rows, and a boundary manifest.
- Local mock/public-source-compatible artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/` show a positive fresh public shadow path with `2` would-submit decisions, fair-mid source pass, edge-gate pass, and `any_private_or_order_endpoint_called=false`.
- Fail-closed evidence covers missing Binance public state, stale Binance public state, wrong symbol, insufficient edge, and anti-drift shadow block. All block scenarios remain no-submit.
- A short real public shadow attempt was made, but this environment did not observe Hyperliquid public L2; blocker evidence records `_ssl.c:1011: The handshake operation timed out`, `no_hyperliquid_public_l2_observed`, and `no_fresh_touch_candidate_reached_fair_mid_source`.
- This finding does not authorize a real maker canary. It does not prove live public source stability, live maker fill, fee/inventory accounting, realized PnL, M3 readiness, maker viability, or stable PnL.
- It also does not authorize credential reads, private/account/order endpoints, live orders, remote refresh, final gate rerun, quote-distance changes, one-tick-back, inside-spread, cap relaxation, default-on behavior, or promotion.

## 0623T006 Fair-Mid Source Finding

- `0623T006` QA is `已通过`. It implements / accepts a task-scoped decision-time fair-mid provider for the watcher-local edge gate.
- Accepted provider policy: `m2_decision_time_public_fair_mid_provider_v1`.
- Source contract: use current in-process Hyperliquid public L2/BBO plus decision-time Binance public state with `symbol`, `signal_ts_ms`, bid/ask or mid, and conservative `lead_move_ticks`; emit `target_symbol=BTC`, `horizon_ms=1000`, `fair_mid_px`, `source`, `hl_mid_px`, `binance_mid_px`, `basis_mid_ticks`, `lead_move_ticks`, `source_age_ms`, and public-state sequence diagnostics.
- Accepted formula: `fair_mid_px = current_hyperliquid_mid + conservative_binance_lead_move_ticks * tick_size`.
- The source is accepted only as a live-compatible decision-time contract. Offline `pricing_signal_rows.csv`, optimistic proxy output, future markout, realized PnL, and oracle horizons remain forbidden as live edge sources.
- Fail-closed evidence covers missing source, missing Binance public state, stale source, wrong symbol, wrong horizon, provider exception, missing Hyperliquid public state, future timestamp, missing fair mid, invalid quote/tick, and insufficient edge.
- Local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/` show one positive fresh fair-mid pass reaching mock post-only `Alo` submit and all block scenarios stopping before mock order calls.
- This finding does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, maker viability, or stable PnL. It does not authorize live execution, credential reads, private/account/order endpoints, remote refresh, final gate rerun, quote-distance changes, one-tick-back, inside-spread, cap relaxation, default-on behavior, or promotion.

## 0623T005 Quote-Placement Envelope Decision Finding

- `0623T005` QA is `已通过`.
- Decision: `continue_touch_only_with_repaired_gates`. The only currently authorized M2 quote-placement envelope remains `0 tick` touch-only with the repaired gate chain from `0623T001`-`0623T004`.
- The accepted gate chain is now coherent at the local/mocked evidence level: post-`open_orders` public L2 freshness, real BBO-history fresh-touch evidence, corrected fill-support vs adverse-flow taxonomy, and a fail-closed fair-value edge gate.
- The current live blocker is not authorization to move quote distance. It is the absence of an accepted live-compatible decision-time fair-mid / edge source. With the edge gate enabled, missing source correctly blocks before order submission.
- One-tick-back and inside-spread protocols are rejected as the next task because both change quote placement and need a separate risk envelope plus fresh positive edge proof.
- The next recommended task is `0623T006 M2 live-compatible fair-mid source acceptance gate`: accept or implement a fresh `edge_signal_provider` with strict schema / symbol / horizon / timestamp validation, no live orders, no credentials, no private/account/order endpoints, and no quote-distance change.
- This finding does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, maker viability, or stable PnL; it does not authorize taker/crossing, one-tick-back, inside-spread, cap relaxation, live execution, default-on behavior, or promotion.

## 0623T004 Fair-Value Edge Gate Finding

- `0623T004` QA is `已通过`. It adds a decision-time fair-value edge gate before watcher-local inline order submission.
- Accepted `0617T005`, `0617T006`, and canonical pricing-signal artifacts remain read-only / proxy evidence. They do not provide an accepted live-compatible fair-mid source at submit time.
- The implemented live adapter is therefore fail-closed unless an explicit `edge_signal_provider` supplies fresh schema-compliant edge. `--event-driven-edge-gate-live` does not synthesize edge from offline CSV artifacts.
- Edge evidence fields are now explicit: `fair_mid_px`, `quote_px`, `edge_ticks`, `signal_age_ms`, `fee_buffer_ticks`, `adverse_selection_buffer_ticks`, `edge_gate_status`, and `edge_gate_reason`.
- Default policy requires `horizon_ms=1000`, `signal_age_ms<=250`, and edge strictly greater than `2.0` fee ticks plus `5.0` adverse-selection ticks. Buy requires `fair_mid_px > quote_px + buffer`; sell is symmetric.
- Local non-live evidence covers positive-edge pass plus missing live source, stale signal, and insufficient edge fail-closed cases. Wrong-symbol and wrong-horizon schema cases are covered by focused tests.
- This finding repairs the missing "worth trading" gate only. It does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, maker viability, or stable PnL; it also does not authorize taker/crossing, `Ioc`, one-tick-back, cap relaxation, live execution, default-on behavior, or promotion.

## 0623T003 Flow Taxonomy Finding

- `0623T003` QA is `已通过`. It repairs the conflict where buy-side sell-at-bid touch flow could be treated as adverse pressure by anti-drift even though it is fill-support / visible queue-depletion evidence.
- Anti-drift flow taxonomy now separates `fill_support_touch`, `fill_support_visible_queue_depletion`, `adverse_strict_through`, and `neutral_or_opposite_flow`; sell-side handling is symmetric.
- Anti-drift pressure blocking now depends on strict-through adverse quantity plus recent adverse BBO evidence. Touch-flow support alone does not block, but it also remains insufficient by itself and must still pass fresh-touch, current BBO, queue, size, state freshness, immediate guard, and maker-only rules.
- Local non-live taxonomy evidence covers touch-support pass, strict-through + adverse-BBO block, and mixed-flow pass cases.
- This finding repairs flow classification only. It does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, or stable PnL; it also does not authorize taker/crossing, one-tick-back, cap relaxation, live execution, or default-on behavior.

## 0623T002 Fresh-Touch Evidence Finding

- `0623T002` QA is `已通过`. It removes event-driven synthetic `stayed_touch` as sufficient fresh-touch evidence and requires real BBO-history evidence for event-driven inline candidates.
- Accepted event-driven freshness sources are now explicit: `real_bbo_history_touch_stability` after at least `250ms` same-touch stability, or `real_bbo_history_top_reset` after same-touch top size/order-count reduction.
- `synthetic_current_event_only`, missing BBO history, ambiguous evidence, or non-pass `fresh_touch_evidence_status` fail closed before order submission.
- Local non-live evidence shows synthetic-only candidates produce no trigger / no mock order intent, while real BBO-history stability can pass the gate and reach one mock `Alo` submit.
- This finding repairs fresh-touch evidence quality only. It does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, or stable PnL; it also does not authorize taker/crossing, one-tick-back, cap relaxation, live execution, or default-on behavior.

## 0623T001 State Freshness Finding

- `0623T001` QA is `已通过`. It repairs the stale-BBO hard flaw by requiring a new public `l2Book` snapshot observed after private `open_orders()` returns before inline reprice / submit can continue.
- If no post-open L2 arrives within the bounded `0.2s` wait, the path fails closed with `post_open_orders_public_state_stale` / precise source reason and does not call the order endpoint.
- Local non-live evidence proves both sides of the gate: a post-open L2 pass case reaches one mock `Alo` submit, while a no-post-open-L2 case records `post_open_orders_public_state_block_count=1`, `live_submissions_count=0`, and zero mock order intents.
- This finding repairs public-state freshness only. It does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, or stable PnL; it also does not authorize taker/crossing, one-tick-back, cap relaxation, live execution, or default-on behavior.

## 0623T001-T005 Repair Sequence Finding

- The current maker path has five ordered repair needs before further quote-distance changes should be considered: stale public BBO after `open_orders`, over-strong synthetic fresh-touch evidence, fill-support vs adverse-drift taxonomy conflict, missing fair-value/alpha edge gate, and quote-placement envelope decision.
- `0623T001` should run first because using stale BBO after `open_orders()` can invalidate any downstream anti-drift or immediate reprice decision.
- `0623T002` should run second because fresh-touch must be proven from real BBO/top-reset history before flow or edge gates can be trusted.
- `0623T003` should run third because anti-drift must distinguish touch-flow fill support from true adverse drift.
- `0623T004` should run fourth because fill-acquisition evidence is not enough; maker submission needs positive fair-value edge or it risks adverse-selection fills.
- `0623T005` should run last as a design/decision gate. It may recommend a future quote-placement risk-envelope task, but it does not authorize one-tick-back, inside-spread, live execution, cap relaxation, M3 readiness, or stable PnL claims.

### M0 Evidence Chain and Gate Baseline

- Goal: keep the accepted read-only replay, optimistic proxy, real-order canary, and final gate artifacts reproducible.
- Checkpoint: `0617T005`, `0617T006`, and `0618T004` remain the hard baseline.
- Pass condition: threshold candidate, optimistic proxy, canary order/cancel, redaction, and final gate can all be rerun from saved artifacts.
- Completion evidence: `0618T005` QA is `已通过`; it reran the read-only replay and optimistic proxy into `local_live_analysis/m0_evidence_chain_baseline_0618T005/`, re-parsed the accepted `0618T004` canary artifacts, and reran the final gate without placing orders or reading credentials.
- M0 result: evidence-chain baseline is complete, but the latest read-only final gate rerun correctly fails closed on `remote_execution_checkout_not_synced_or_invalid` because saved remote state is `cross-exchange:52b5b9541:0` while current local gate commit is `d37438e`.
- Forward guard: before any M1 live/canary task is created, refresh/sync the remote execution checkout and rerun the final gate; do not treat M0 completion as live authorization.
- `0618T006` QA resolved the forward guard: remote checkout is `cross-exchange:d37438e0c:0`, the refreshed final gate returns `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`, and no live/canary order was executed.

### M1 Repeated Tiny-Live Canary Windows

- Goal: run multiple independent tiny-live / canary windows under the same strict caps.
- Pass condition: each window preserves `order -> tracked cancel -> final open_orders=[]`, with no credential leakage and no reliance on scheduled-cancel.
- Stop condition: any window that expands scope, relaxes caps, or loses shutdown proof is a deviation.
- Completion evidence: `0618T007` QA is `已通过`; one formal M1 loop task used git-safe bundle + remote `merge --ff-only`, reran the final gate to `allow_create_0617T008=true`, and completed `3/3` independent Hyperliquid post-only `Alo` canary windows.
- M1 result: all three windows reached `resting`, used tracked cancel, recorded `final_open_orders=[]`, did not call `schedule_cancel`, and preserved credential / secret / raw-signature redaction boundaries.
- Forward guard: M1 proves repeated canary order/cancel/shutdown mechanics only. It does not prove realized PnL, fee/rebate accounting, inventory accounting, stable PnL, maker viability, promotion, default-on, or scale-up.

### M2 Real PnL and Cost Accounting

- Goal: measure live fills, fees / rebates, slippage, inventory change, and realized net PnL.
- Pass condition: every live window has a complete and auditable PnL ladder that can be compared with replay.
- Stop condition: optimistic proxy numbers are never treated as realized PnL.
- M2A completion evidence: `0618T008` QA is `已通过`; it added a no-network ledger/reconciler and validated the accepted M1 artifacts as `fail_closed_no_realized_live_pnl` because the three canary windows rested and canceled without fill/economics settlement/inventory transition evidence.
- M2A fixture evidence: a local maker-fill fixture computes `gross_pnl_usdc=0.26`, `fee_usdc=0.125268`, `net_pnl_usdc=0.134732`, `inventory_delta_btc=0.01`, and `slippage_usdc=0.0`, proving ledger arithmetic only, not live PnL.
- Forward guard: M2B must consume the `0618T008` ledger/reconciler. If live windows produce no fill or incomplete fee/inventory settlement, M2 remains incomplete and must fail closed rather than claim stable PnL.
- M2B blocked evidence: `0618T009` QA is `阻塞`; the loop safely ran `3/3` real post-only `Alo` windows after git-safe refresh and final gate go, but all windows remained `resting` and then canceled with `fill_count=0`, `final_open_orders_count=0`, and `shutdown_proof_status=pass`.
- M2 current result: no realized PnL proof exists yet. The T008 ledger rerun for T009 records `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- Forward guard: do not enter M3, do not claim stable PnL, and do not switch to taker/crossing behavior to force fills. Any M2 retry must stay maker-only/post-only under the same caps unless a new controller-approved risk envelope is created.
- Fill-acquisition repair evidence: `0618T010` QA is `阻塞`; the adaptive repair ran one same-caps live window with `6/6` cancel/requote attempts, `side_policy=alternate`, and all attempts were post-only `Alo`, non-crossing, and reached `resting`.
- T010 result: no attempt filled. The aggregate fill ledger remains empty, final open orders are 0, and T008 ledger again reports `fail_closed_no_realized_live_pnl`.
- No-fill diagnosis evidence: `0618T011` QA is `已通过`; it analyzed T009/T010 artifacts offline with no network, no live order, no credential read, and no private/order endpoint call.
- T011 result: `9/9` maker-only attempts were no-fill and `9/9` joined same-side touch. Defensible public depth proxy exists for `4/9` attempts only, with same-side top-depth `21.83x` to `1921.75x` of the `0.00999 BTC` order. T010 attempts 2-6 have BBO-only evidence and are marked `per_attempt_depth_missing`.
- T011 design decision: do not run another blind same-caps live retry. Next M2 work should be a read-only public L2/trades flow diagnosis that separates queue-depth, trade-through, quote-aging, side/regime, and time-of-day causes before any later maker-only retry.
- Public flow diagnosis evidence: `0618T012` QA is `已通过`; it used public-only Hyperliquid `l2Book` / `trades` data and no live order, credential read, private/account/order endpoint, remote checkout refresh, final gate rerun, taker/crossing behavior, or cap relaxation.
- T012 collection result: local direct public collection from this machine failed with Hyperliquid public API/WebSocket SSL EOF and was recorded only as a collection-path blocker. A public-only `awsserver1` collection succeeded with `l2Book=56`, `trades=122`, `subscription_ack=2`, `reconnects=0`, and `close_reason=duration_elapsed`.
- T012 diagnosis result: the pulled-back sample produced `56` book events, `356` individual trade events, and `38` passive touch-quote candidates. The hypothesis matrix is `queue_too_deep=supported`, `no_trade_through=rejected_for_sample`, `wrong_time_of_day=inconclusive`, `wrong_side=supported`, and `quote_aging_or_fast_drift=supported`.
- T012 interpretation: the sampled no-fill problem is not simply lack of trade-through. Public flow did trade through touch quotes in many candidate windows (`21/38` strict-through, `34/38` touch-trade), but visible same-side top depth plus order-size depletion occurred in only `7/38` candidates, adverse lost-touch occurred in `21/38`, and side asymmetry was material (`buy` public-depletion `6/19`, `sell` `1/19`).
- T012 forward decision: do not continue blind retry. If M2 continues, the next task should be a maker-only retry design/repair that addresses quote aging / fast drift and side selection while preserving `Alo`, T008 ledger fail-closed, and same or smaller caps. Time-of-day remains undecided until longer or cross-hour public samples exist.
- Flow-aware retry evidence: `0619T001` QA is `阻塞`. The task repaired blind retry behavior and safely executed one flow-aware window after remote refresh and final gate go, but still produced `fill_count=0`. It submitted `2` post-only `Alo` candidates, skipped `4` crowded-touch candidates, canceled/requoted one buy after `lost_touch+adverse_drift`, ended with final open orders empty, and T008 ledger again reported `fail_closed_no_realized_live_pnl`.
- `0619T002` redesign finding: the current `0619T001` policy is still too tolerant for fill acquisition. It allowed a sell candidate at about `120.25x` same-side top depth multiple and a buy candidate at about `199.18x`; neither filled, and the buy aged out in about `1.01s`. The next design must stop treating these states as acceptable passive-entry queues.
- `0619T002` QA is `已通过`: QA accepted the design-only contract and confirmed no code implementation, live order, credential read, remote refresh, final gate rerun, or strategy/live artifact was mixed into this task.
- `0622T001` task boundary: the next step should not be another diagnosis-only task. It is authorized as the current implementation plus controlled live micro-test task, but only after focused tests, git-safe remote refresh, final gate go, and public session-gate eligibility. Live execution remains limited to one micro-window, at most `2` post-only `Alo` submissions, dynamic size hard cap `<=0.005 BTC`, tracked cancel, final open-orders proof, and T008 ledger fail-closed.
- `0622T001` QA is `阻塞`: the fresh-touch implementation landed as `m2_fresh_touch_size_by_throughput_session_gate_v1`, local focused verification passed, remote checkout/final gate passed, and a formal public precheck/session gate ran. The formal micro-window had `8` public candidates but `0` full quality-gate allowed fresh-touch candidates, so the task correctly stopped before live order submission. No real order endpoint was called, independent remote open-orders proof was empty, and T008 ledger returned `fail_closed_no_realized_live_pnl`.
- `0622T001` live calibration result: the current sampled micro-window still points to queue-depth / insufficient strict-through quality under the new stricter gate. Public precheck produced buy strict-through `1/4`, sell strict-through `0/4`, public depletion `0/8`, and median top-depth multiples far outside the quality bands for most candidates. M2 remains blocked on live maker fill / fee / inventory / realized PnL evidence.
- `0622T002` forward decision: the next task should wait for a better public micro-window instead of forcing another immediate order attempt. The watcher phase must remain public-only and time-boxed; a live order is allowed only if the existing `0622T001` fresh-touch / dynamic-size / session-gate allows a current-window candidate. This keeps the project aligned with fast live calibration while still preventing taker/crossing, cap relaxation, and blind queue joining.
- `0622T002` QA is `阻塞`: the time-boxed watcher did find one `quality_a` buy candidate after `120.451756s`, with strict-through `0.41541 BTC`, same-side top qty `0.00016 BTC`, one same-side top order, and dynamic size `0.005 BTC`. However the separate live window reran the public pre-submit gate after the handoff; by then the current candidate set had shifted to `10` candidates with `0` allowed, so no real order endpoint was called and no fill/PnL evidence was produced. QA accepted implementation/safety evidence, but M2 remains blocked. The new bottleneck is trigger-to-order staleness between watcher evidence and live window execution, not T008 ledger or open-orders shutdown.
- `0622T002` next repair finding: if M2 continues, the watcher trigger and immediate maker-only submission guard should run in the same remote process to remove controller pullback / second long precheck latency. This must preserve current L2/BBO recheck, Hyperliquid `Alo`, dynamic size hard cap `<=0.005 BTC`, tracked cancel, independent open-orders proof, T008 fail-closed, and no taker/crossing / one-tick-back / cap relaxation.
- `0622T003` task boundary: the next live-calibration attempt is allowed only as a same-process watcher-triggered maker-order repair. It may remove the slow `watcher -> pullback -> new live window` handoff, but it may not weaken the current-candidate guard or risk envelope. A stale candidate before order is a valid fail-closed result, not a reason to cross, use taker, move one tick back, or relax size/caps.
- `0622T003` QA is `已通过`: the same-process repair removed the controller pullback / separate live-window path and kept the watcher waiting phase public-only. The final short-iteration rerun found `1` eligible `quality_a` buy candidate out of `150`, but the immediate same-process guard failed closed before private submit because candidate age was `3.711s` versus the `3.0s` max, quote `64227` was no longer current touch versus current bid/ask `64219/64220`, and current queue quality had refilled to `20.52451 BTC`, `49` top orders, and `4104.902x` top-depth multiple. No order was submitted, final open-orders proof was empty, and T008 ledger returned `fail_closed_no_realized_live_pnl`.
- `0622T003` M2 implication: the latest blocker is no longer controller pullback or a second long live-window precheck. The observed blocker is opportunity half-life inside the public collection / candidate selection / immediate guard path itself. M2 still requires live maker fill plus fee/inventory/realized PnL evidence; do not enter M3, do not claim stable PnL, and do not weaken `Alo`, `<=0.005 BTC`, or T008 fail-closed boundaries.
- `0622T004` task boundary: the next repair should be event-driven current-candidate submit, not another shorter fixed-window batch retry. It must generate quote/side/size from the current in-memory L2/BBO and rolling trade-flow state, target `candidate_event_to_guard_start <= 500ms`, fail closed above `1.0s` candidate age, and preserve `Alo`, `<=0.005 BTC`, no taker/crossing, no one-tick-back, tracked cancel, open-orders proof, and T008 fail-closed.
- `0622T004` QA is `已通过`: event-driven current-candidate evaluation and fast-submit repair were implemented. The rerun reached a live `Alo` order submission attempt under `<=0.005 BTC` after outer guard latency `0.000286s` and inner guard candidate age `0.609s`, but Hyperliquid rejected the post-only buy because BBO moved from the guarded `64107/64108` to `64090@64091` by exchange processing time. No fill occurred, final open-orders proof was empty, and T008 returned `fail_closed_no_realized_live_pnl`.
- `0622T004` M2 implication: the previous blocker, pre-submit staleness from slow private/account preflight, is repaired. The current blocker is sub-second fast-drift / exchange-side post-only rejection before resting plus no live fill. M2 remains blocked until a live maker fill with fee/inventory/mark/realized PnL evidence passes T008; do not solve this by switching taker, crossing, one-tick-back, larger size, or looser caps.
- `0622T005` task boundary: the next repair is inline reprice / post-only reject handling inside the watcher process. It should not call the full `fill_window.run_window` submit path after trigger; instead it must use the same latest in-memory L2/BBO, run only private `open_orders` safety before submit, reprice after that check, and submit `Alo` immediately if strict guard still passes.
- `0622T005` retry boundary: a post-only would-immediately-match rejection may trigger at most one maker-only retry, but only after re-evaluating the latest public L2/BBO/current candidate. It must not use taker, crossing, one-tick-back, larger size, looser queue bands, or stale trigger quotes. Total real order endpoint calls remain capped at `2`.
- `0622T005` business execution is `待验收`: the inline watcher-local submit path was implemented and formally executed after git-safe remote refresh and final gate go. The run evaluated `48` current candidates over `82.208056s`, triggered once, and submitted `2` post-only `Alo` buy attempts at `64144.0` for `0.00004 BTC`, staying below the unchanged `<=0.005 BTC` hard cap.
- `0622T005` live result: both attempts passed local strict guard at current/submit BBO `64144/64145`, top-depth multiple `4.25x`, and `quality_a`; attempt candidate ages were `0.559s` and `0.826s`. Hyperliquid rejected both as post-only would-immediately-match because BBO moved by exchange validation time to `64143@64144` and then `64142@64143`. No order rested or filled, final open orders were empty, and T008 returned `fail_closed_no_realized_live_pnl`.
- `0622T005` M2 implication: the `watcher -> fill_window` overhead is no longer the active blocker. Post-`open_orders` `open_orders_end_to_reprice` was `0.084ms-0.136ms`, and `reprice_to_order_submit` was `0.043ms-0.045ms`; the remaining observed blocker is exchange-side fast BBO drift before post-only validation, with private `open_orders` still taking `0.04255s-0.2836s` before the final reprice. M2 remains blocked on missing live maker fill / fee / inventory / realized PnL proof.
- `0622T005` QA is `已通过`: QA accepted the task-level repair, artifact completeness, public-only waiting boundary, same-process inline path, post-only retry matrix, final open-orders proof, independent open-orders proof, redaction, and T008 fail-closed behavior. This does not change the milestone status: M2 remains blocked and M3 must not start.
- `0622T006` task boundary: the next repair should not try to shorten the already-fast local reprice-to-submit path further. It should add anti-drift / touch-stability gating before maker-only submit so orders are skipped when recent public L2/trades state shows adverse BBO movement likely to trigger exchange-side post-only rejection. Per controller update, it should keep the same dynamic size logic and `<=0.005 BTC` hard cap but expand the data sample to max `30` real submissions; it must still preserve `Alo`, no taker/crossing, no one-tick-back, tracked cancel, independent open-orders proof, and T008 fail-closed.
- `0622T006` QA is `已通过` at task level: anti-drift / touch-stability gating was implemented and formally executed. The main run evaluated `914` current candidates, anti-drift passed `6` / blocked `93` of `99` gate evaluations, and submitted `2` post-only `Alo` buy attempts under the unchanged `<=0.005 BTC` cap (`0.00422 BTC @ 64956.0`, `0.00179 BTC @ 65032.0`).
- `0622T006` live result: both attempts passed local immediate guard and same-process reprice, but Hyperliquid rejected them as post-only would-immediately-match after BBO drifted to `64954@64955` and `65025@65026`. Reruns produced either no submission or an SSH/control-plane interruption that was cleaned up with final open orders proven empty.
- `0622T006` M2 implication: the anti-drift gate is useful as a filter but is not sufficient to prove fill acquisition. M2 remains blocked on no live maker fill / fee / inventory / realized PnL proof, and the next repair should address post-`open_orders` public-state freshness / stale BBO and exchange-side fast drift without switching to taker, crossing, one-tick-back, larger size, or looser caps.
- `0619T002` next-policy contract: move from generic flow-aware touch join to `fresh_touch_size_by_throughput_session_gate`.
  - Quote placement:
    - place only at touch with `quote_offset_ticks=0`
    - no one-tick-back workaround
    - require fresh-touch / queue-reset evidence
    - `quality_a`: same-side top depth multiple `<=20x`, same-side top order count `<=6`, same-side strict-through support, hold `<=3s`
    - `quality_b`: same-side top depth multiple `20x-100x`, same-side top order count `<=12`, same-side strict-through support plus enough recent touch-trade throughput, hold `<=1s`
    - anything worse must skip instead of resting
  - Size:
    - replace fixed `0.00999 BTC` with dynamic size `min(bucket_cap, 0.25 * recent_same_side_at_or_through_trade_qty_btc_last_3s, 0.005 BTC)`
    - bucket caps: `quality_a=0.005 BTC`, `quality_b=0.002 BTC`
    - if recent throughput is too low, skip instead of forcing an order
  - Side:
    - default `buy_only`
    - sell remains disabled unless both same-window precheck and a later hour-by-side scorecard materially favor sell
  - Time-of-day:
    - no fixed allowed UTC hour yet
    - next live execution must stay tied to a current public-precheck micro-window
    - any fixed UTC-hour include/exclude decision needs a later cross-hour public scorecard first
  - Loop:
    - at most `2` live submissions in a window
    - abort the window when no `quality_a` or `quality_b` candidate appears quickly
- Forward guard: do not enter M3, do not claim stable PnL, do not switch to taker/crossing, and do not loosen caps. M2 remains blocked until a live maker fill with fee/inventory/mark evidence passes the T008 ledger.

### M3 Cross-Day / Cross-Regime Stability

- Goal: show the strategy survives more than one market regime instead of one favorable sample.
- Pass condition: positive or at least non-degrading net result across multiple dates / regimes, with drawdown controlled inside the approved risk envelope.
- Stop condition: a single good window is not enough to claim stability.

### M4 Expansion or Stop Decision

- Goal: only after M1 to M3 pass, decide whether to widen size, keep the same envelope, or stop and rework.
- Pass condition: any expansion is justified by repeated live evidence, not by replay alone.
- Stop condition: no silent scale-up, no default-on promotion, and no cap relaxation without new evidence.

Drift guard:

- If a task does not advance one of the milestones above, it is not part of the main line.
- If a task weakens a milestone boundary, it should be treated as scope drift.

## 0624T001 BBO Evidence-Chain Diagnosis Finding

- `0624T001` QA is `已通过`. It added an offline public-only BBO evidence-chain diagnosis mode and replayed the AWS `0623T010` public-shadow row-level artifacts.
- The dominant blocker classification is `public_bbo_density_or_cache_continuity_blocks_bbo_history_visibility`.
- Evidence: T010 had `1259` candidate rows, but only `112` l2Book-triggered candidate evaluations versus `1147` trade-triggered evaluations; public stream totals were `112` book events versus `3449` trade events, a `3.247318%` book/trade event ratio.
- Fresh-touch evidence remained unavailable for almost all rows: `synthetic_current_event_only_count=1257`, `fresh_touch_evidence_pass_count=2`, `fresh_touch_allowed_count=0`, and `queue_reset_supported_count=2`.
- Event ordering was not the dominant observed blocker in this sample: `exchange_time_regression_count=1`, `trade_older_than_latest_l2_count=1`, and `negative_next_l2_delta_count=0`.
- Conclusion: the current public-shadow path mostly evaluates trade-triggered candidates without enough accepted real BBO-history visibility to prove touch stability or queue reset. The next repair should inspect live public book subscription/update handling and BBO history/cache construction, not Binance freshness, fair-mid edge, quote-distance relaxation, cap relaxation, or private/order paths.

## 0618T003 Credential Location Finding

- The Hyperliquid credential-shaped fields are in the XEMM `.env` files on `awsserver1`.
- Candidate credential paths:
  - `/home/admin/XEMM_rust/.env`
  - `/home/admin/XEMM_rust_latest/.env`
- Candidate keys:
  - `HL_WALLET`
  - `HL_PRIVATE_KEY`
- The `config.json` files contain Hyperliquid configuration fields only in this scan.
- Reports and artifacts intentionally record only paths and key names, not credential values.

## 0618T004 Real-Order Canary Boundary

- `0618T004` is the first formal task scoped to validate the authenticated Hyperliquid order/cancel/private-read interface chain with a real canary order attempt.
- It may use the `0618T003` credential locations on `awsserver1`, but it must never print or persist credential values, private keys, raw signatures, or nonces.
- The task must stay inside the tiny-live caps: `BTC`, `duration<=10 minutes`, `max_order_size=0.01 BTC`, `max_loss=30 USDC`, maker-only / post-only `Alo`, immediate cancel / shutdown proof, and artifact pullback.
- This is not a continuous live strategy run, not promotion, and not PnL or maker-viability proof.
- Business execution proved the primary order/cancel/private-read interface path: the post-only canary order reached `resting`, tracked cancel succeeded, and final open orders were empty.
- `schedule_cancel` was called but Hyperliquid rejected it because the account has not met the traded-volume eligibility threshold. The next live task must not depend on scheduled-cancel / dead-man switch unless this account eligibility changes; tracked cancel plus final open-order proof remains the validated shutdown path.
- QA accepted `0618T004`; final gate remains `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true` after refreshing remote state to `cross-exchange:52b5b9541:0`.

## 0618T002 SDK Readiness Finding

- `0618T002` removed the official SDK dependency blocker using `hyperliquid-python-sdk==0.24.0`.
- Local interpreter: `/home/molly/anaconda3/bin/python`.
- Remote interpreter: `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`.
- The readiness checker verifies imports and SDK method surfaces only; it does not construct wallet-backed clients and does not call `/exchange`, `/info`, private/account/order endpoints, websocket, signing, or nonce paths.
- The repaired final gate now reports `tiny_live_ready_for_controller_go` and `allow_create_0617T008=true`.
- This is not proof that the real submit-order API works; that requires a separately approved task that submits a real post-only canary/tiny-live order under caps.

## 0616T007 awsserver1 Preflight Blocker

- `0616T007` correctly stopped the auto loop before live execution.
- SSH to `awsserver1` succeeded, but the remote repo at `/home/admin/hft_live/hftbacktest` is on branch `master`, not `cross-exchange`.
- Controller clarification: that `master` checkout is the Binance maker route, so it should not be modified or repurposed for the Binance-lead / Hyperliquid-lag route.
- The remote repo has dirty changes (`29` status rows), so it is not a clean execution checkout for the approved cross-exchange tiny-live path.
- Remote `conda` is not available and remote `rsync` is not available. The task used `scp` to pull back dry-run evidence, but this does not satisfy the preferred operator packet path without either installing `rsync` or updating the packet to accept `scp`.
- Remote system Python is `/usr/bin/python3` at `Python 3.13.5`; this is acceptable for a future remote preflight if selected and recorded explicitly, while local validation continues to use `.conda-envs/hft-py38`.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- `0616T008` remains blocked despite the earlier limited live approval, because that approval was conditional on `0616T007` QA passing first.

## 0617T001 Cross-Exchange Remote Path Finding

- `0617T001` resolved the route separation issue by creating `/home/admin/hftbacktest-cross-exchange` on `awsserver1` from the local `cross-exchange` branch.
- The old Binance maker route `/home/admin/hft_live/hftbacktest` remains `master:703c149` with dirty count `29` and was not modified.
- The new cross-exchange route is `cross-exchange:7642b16` with dirty count `0`.
- Remote system Python `/usr/bin/python3` at `Python 3.13.5` is the selected Python for this path; conda is not required for the remote preflight path if this fact remains explicit.
- Artifact pullback used `scp` because remote `rsync` is unavailable; future operator packet wording should either accept `scp` as a valid pullback method or install `rsync` before requiring it.
- This does not authorize live execution by itself. A new live-capable preflight dry-run over `/home/admin/hftbacktest-cross-exchange` must pass QA before `0616T008` can be created.

## 0617T002 Repeat Preflight Finding

- `0617T002` confirmed the new cross-exchange remote path is stable across repeated preflight.
- Remote state remained `/home/admin/hftbacktest-cross-exchange` at `cross-exchange:7642b16:0`.
- Remote Python remained `/usr/bin/python3` at `Python 3.13.5`.
- `scp` pullback and checksum validation worked again.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- The next task may be a final live-capable preflight/operator task that binds the new path, system Python, `scp` pullback, and approved caps; it should still stop before real orders until QA accepts it.

## 0617T003 Task Boundary

- `0617T003` is the final live-capable preflight/operator task before any `0616T008` live execution can be created.
- It may materialize the approved caps and operational path into artifacts, but it is not a live order task.
- Any credential read, private endpoint call, account query, signing/nonce/user-stream implementation, order placement, cancellation, amendment, live bot startup, deployment, promotion, PnL proof, or maker viability claim would be out of scope and must fail closed.
- QA passed. The final operator packet binds `/home/admin/hftbacktest-cross-exchange`, `/usr/bin/python3`, `scp` pullback, and the approved `0616T008` caps.
- `real_orders_allowed` is explicitly scoped to a later separately dispatched `0616T008`; `0617T003` itself remains no-order.

## 0617T004 Task Boundary

- `0617T004` is required because the previous preflight/operator tasks prepared execution rails and caps, but did not define a trading signal or quote policy.
- It must define a minimal Binance-lead / Hyperliquid-lag maker policy before live execution can be considered.
- If threshold selection cannot be defended from accepted local artifacts, the task must recommend threshold calibration rather than proceeding to `0616T008`.
- It is design/protocol only and must remain no-live/no-order/no-private.

## 0617T005 Replay Boundary

- `0617T005` was dispatched after `0617T004` QA passed.
- Its purpose is to run a read-only signal / quote replay on existing accepted cross-exchange public/read-only datasets after `0617T004` defines the protocol.
- The replay may compute `basis_mid`, threshold/persistence state, side intent, theoretical quote price, quote distance, post-only/crossing diagnostics, stale/data-gap rejections, and simulated cap triggers.
- It cannot prove real fills, real PnL, real inventory, real post-only reject behavior, queue priority, deployment readiness, or maker viability because the existing datasets do not include complete Hyperliquid private order/account/economics source paths.
- It remains no-live/no-order/no-private and must be QA/controller reviewed before any live task can consume its threshold candidate.

## 0617T004 Threshold Finding

- The protocol evidence is strong enough to define direction, side mapping, and quote/cancel boundaries.
- The accepted artifacts do not provide a defensible absolute live threshold for `basis_mid_ticks`.
- Therefore the live threshold is `blocked_for_live_execution`, and the next step must be a read-only replay / calibration task before any `0616T008` live execution.

## 0617T005 Replay Finding

- The local `pricing_signal_rows.csv` files are complete enough for full read-only multi-sample replay. The earlier missing-file conclusion was caused by manifest absolute-path resolution, not absent data.
- `0617T005` now replays `0601T005` plus all seven historical event-mode pricing-signal artifacts referenced by `0609T008`: `8` files, `161455` raw pricing rows, and `26948` de-duplicated decision rows.
- Source availability records every historical manifest sample as `local_direct_file_available=true` and `replay_source_used=pricing_signal_rows`.
- The most balanced read-only threshold candidate is `75` ticks with `2` observations of persistence: `654` theoretical intents, `313` buy / `341` sell, `2.4269%` intent rate, and `8/8` samples with any intent. A stricter low-activity fallback is `75` ticks with `3` observations: `327` intents and `1.2134%` intent rate.
- `0617T005` QA is `已通过`, so this read-only calibration evidence can feed `0617T006`. It still does not authorize direct `0616T008` live execution without `0617T006` QA/controller ratification.

## 0617T006 Optimistic PnL Proxy Boundary

- `0617T006` exists to answer a narrow question: under the most optimistic assumption that every theoretical maker intent fills at quote, is the public-data future-mid upper bound materially positive?
- This is an optimistic PnL proxy / theoretical upper bound only. It is not real PnL and cannot validate fill probability, queue priority, post-only reject behavior, private/order lifecycle, account inventory, fee/rebate settlement, spread capture, maker viability, or live readiness.
- The task must reconcile the user's "6 datasets" wording against accepted manifests. Current nearby facts include `7` canonical event-mode historical samples in `0609T008` and `8` total `0617T005` replay inputs when `0601T005` is included.
- The useful outputs are fixed-horizon and oracle-best-horizon summaries by sample, side, threshold, persistence, and horizon. Aggregate-only optimistic PnL is not enough because sample concentration can hide instability.

## 0617T006 Optimistic PnL Finding

- `0617T006` QA passed on `2026-06-17 23:16 CST`; latest QA report is `.workflow/reports/0617T006-qa.md`.
- The user selected `canonical_7` as the formal sample-set口径, so the earlier exact `6` dataset wording is superseded and no longer blocks QA.
- The official set is `canonical_7`; `0617T005_8_input` remains a diagnostic comparison set.
- At `75` ticks / persistence `2` / `1000ms`, both sets are materially positive under the optimistic public-data upper-bound assumption: official `canonical_7` is `295.985 USDC` with `7/7` positive samples, and diagnostic `0617T005_8_input` is `299.38 USDC` with `8/8` positive samples.
- The oracle-best-horizon result is much larger (`1098.535 USDC` on `canonical_7`, `1124.89 USDC` on `0617T005_8_input`) but is explicitly non-tradeable because it selects the best future horizon after the fact.
- This does not prove real PnL, fills, fees/rebates, queue priority, execution viability, maker viability, or live readiness.

## 0617T007 Final Gate Finding

- `0617T007` correctly fails closed before `0617T008` tiny-live creation.
- The final recommendation is `tiny_live_needs_missing_precondition` and `allow_create_0617T008=false`.
- The recorded `awsserver1` remote checkout is on the right path and branch, but it is stale: remote `/home/admin/hftbacktest-cross-exchange` is `cross-exchange:7642b16:0`, while the local accepted commit at gate runtime was `1556a85`.
- Current repo scope has no QA-accepted Hyperliquid tiny-live real-order executor, no proven exchange-side post-only enforcement path, no real cancel-all/shutdown implementation beyond local fake/placeholder evidence, and no live-capable private order response source.
- The old approval packet names `0616T008`; the latest controller instruction names `0617T008`, so this migration must remain explicit and cannot be treated as silent live authorization.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- `0617T008` must not be created or executed unless a later repaired gate and QA explicitly produce `allow_create_0617T008=true`.
- QA accepted the gate mechanics on `2026-06-17`; this acceptance does not override the no-go decision.

## 0618T001 Repair Boundary

- The next repair must address the remaining substantive blocker from `0617T007`: the absence of a QA-accepted Hyperliquid tiny-live real-order executor with proven post-only enforcement, max-loss stop, cancel-all shutdown proof, private/order response artifact handling, and artifact pullback.
- Syncing `awsserver1` to the latest `cross-exchange` commit repairs the stale-checkout symptom, but it does not by itself make live execution safe.
- `0618T001` may implement a live-capable executor and update the final gate, but it must not run the live order window or create `0617T008`.
- A passing `0618T001` gate would only let total control decide whether to create a later separate tiny-live execution task under the same strict caps.
- Business execution repaired the executor-wrapper blocker at no-order self-test level: strict caps, Hyperliquid `Alo` intent enforcement, max-loss fail-closed logic, cancel-all control flow, redaction, artifacts, and final gate consumption are implemented.
- The repaired final gate still correctly fails closed because the official Hyperliquid Python SDK is not installed locally or on `awsserver1`; SDK availability is required before a live task can use official signing/order/cancel behavior.
- `0618T001` does not prove the real submit-order API path works. That requires a later separately approved task that actually submits a real post-only canary/tiny-live order and then cancels/shuts down under caps.

## 0618T002 SDK Readiness Boundary

- The next repair should address only the SDK dependency blocker from `0618T001`.
- Installing or making the official Hyperliquid SDK importable is not a live authorization and does not prove order placement works by itself.
- `0618T002` must verify SDK import and method surface without credentials, without constructing secret-backed clients, and without calling private/order/account endpoints.
- If `0618T002` removes the SDK blocker and final gate later allows `0617T008` creation, total control still needs a separate live/canary task before any real order is submitted.

## 0616T008 Live Approval Boundary

- The controller approved a single limited `0616T008` Hyperliquid tiny-live small-notional execution window on `2026-06-17`, conditional on `0616T006` QA and `0616T007` awsserver1 preflight dry-run QA passing first.
- Approved caps: `symbol=BTC`, `max_order_size=0.01 BTC`, `max_order_notional=700 USDC`, `max_position=0.04 BTC`, `max_position_notional=2800 USDC`, `max_notional=3000 USDC`, `max_loss=30 USDC`, `duration=10 minutes`, `host_machine=awsserver1`, `account_scope=Hyperliquid account configured on awsserver1`, `maker_only/post_only=true`, `real_orders_allowed=true`.
- BTC/USD reference at approval time was `65794.035`; the notional caps intentionally round above the `0.01 BTC` and `0.04 BTC` spot equivalents.
- This is not a general live authorization. Any cap mismatch, missing QA prerequisite, private credential disclosure, default-on behavior, deployment/promotion claim, scaling request, or later live window must stop for controller approval.

## 0616T006 Task Boundary

- `0616T006` has been created as the next formal task after `0616T005` QA passed.
- Scope is Hyperliquid tiny-live live-capable preflight / operator packet for future `awsserver1` execution and local artifact validation, not real live execution.
- The task may prepare operator commands, schemas, local validators, dry-run artifacts, artifact pullback/checksum policy, and host preflight requirements.
- The task must leave unapproved live fields as `pending_controller_approval`.
- It must not authorize or perform real order placement, cancellation, amendment, live bot startup, account query, credential disclosure, signing/nonce/user-stream implementation, deployment, promotion, PnL proof, or maker viability proof.
- Business execution produced a local/offline packet generator and validator, official artifacts, documentation, and a `待验收` business report.
- The generated packet names `awsserver1` as the intended host but keeps `host_machine` approval status as `pending_controller_approval`; it also keeps `real_orders_allowed=pending_controller_approval`.

## 0616 Cross-Exchange Auto Loop Stop Point

- The auto loop defined in `docs/cross_exchange_auto_loop_protocol.md` completed through `0616T005` and must now stop.
- `0616T002-T005` improved Hyperliquid readiness only through design, fixture validation, local fake shutdown proof, and protocol design. They do not authorize private endpoint calls, credentials, signing, nonce, user streams, account queries, real order placement, real cancellation, live startup, deployment, promotion, or PnL proof.
- The latest effective QA result is `0616T005` with status `已通过`.
- The next possible live-capable task requires explicit controller approval of all live fields recorded in `local_live_analysis/hyperliquid_tiny_live_protocol_design_0616T005/human_approval_fields.csv`.

## 0616T001 Cross-Exchange Correction Finding

- The active branch is `cross-exchange`; the user clarified the intended direction is Binance lead / Hyperliquid lag maker strategy.
- The previous `0615T009` interpretation as a Binance `BTCUSDT` single-venue small-cap live test is stopped for this branch.
- `0615T008` is not a valid cross-exchange live predecessor because it defines a Binance-specific small-cap protocol, not Hyperliquid maker private/order readiness.
- `0615T001-T007` are not wasted, but they must be treated as reusable source-chain/evidence methodology or migration templates only. They do not authorize Hyperliquid private/order endpoints, credentials/signing/nonce/user stream, order placement, strategy live, parameter search, promotion, PnL proof, or maker viability proof.
- The next correct task should define a Hyperliquid maker private/order execution-readiness boundary using the accepted cross-exchange evidence chain (`0601T004`, `0601T005`, `0609T002`) and must stay no-live/no-order until separate readiness gates pass QA.
- Official `0616T001` artifacts are under `local_live_analysis/cross_exchange_branch_correction_0616T001/`, with final recommendation `cross_exchange_branch_correction_ready_for_qa`.

## Post-0615T005 Forward-Path Finding

- Superseded for `cross-exchange` by `0616T001`; the bullets below describe the prior Binance small-cap path and must not be used as current next-task authority on this branch.
- The user wants the next design to end in a small-cap live test and then use the real-environment data for analysis and decision-making.
- That intent is compatible with the repository only if it is staged behind a runner-consumption gate, a proof-limited read-only runner, and a dedicated live-test risk protocol.
- The first live-capable task must be `0615T009`, not the immediate next task, because the current accepted artifacts still stop at no-trading source implementations.
- The live task must have hard caps, maker-only/post-only behavior, kill-switch criteria, cancel-all/shutdown evidence, and a post-run analysis task before any decision to repeat or scale.
- No current artifact authorizes direct live promotion, default-on behavior, or PnL-based scaling.

## 0615T006 Task Boundary

- `0615T006` has been created, business execution is complete, and QA is `已通过`.
- Scope is source-chain runner-consumption gate / synthesis design, not runner implementation or live execution.
- It defines dependency, timestamp, identity/redaction, runner input, proof-limit, fail-closed, and QA gates for a later `0615T007` proof-limited runner.
- Its final recommendation is `source_chain_runner_consumption_gate_ready_for_qa`; this can only mean the gate design is ready for QA/controller review.
- QA report is `.workflow/reports/0615T006-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T006` as the latest effective QA result.
- It must not be interpreted as authorization for endpoint/source collector implementation, credentials/signing/nonce/user stream, real private/order/account/live/economics data reads, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real metrics, PnL proof, deployment, promotion, or maker viability proof.
- `0615T009` remains the first live-capable task, and only after `0615T007` / `0615T008` pass QA plus explicit total-control approval.

## 0615T007 Task Boundary

- `0615T007` has been created, business execution is complete, and QA is `已通过`.
- Scope is a proof-limited local read-only runner over already accepted artifacts, not endpoint/source collector work or live execution.
- The runner may emit proof-limited, unavailable, or fail-closed rows only.
- Missing-source and PnL/promotion overclaim requests fail closed.
- Its final recommendation is `proof_limited_read_only_runner_ready_for_qa`; this can only mean runner mechanics are ready for QA/controller review.
- QA report is `.workflow/reports/0615T007-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T007` as the latest effective QA result.
- It must not be interpreted as authorization for real execution/economics metrics, PnL, strategy decisions, live readiness, deployment, promotion, or maker viability proof.

## 0615T008 Task Boundary

- `0615T008` has been created, business execution is complete, and QA is `已通过`.
- Scope is small-cap live-test protocol / risk gate design and dry-run acceptance, not live execution.
- The protocol requires `BTCUSDT`, `10` minute duration cap, `25 USDT` gross notional cap, `5 USDT` single-order cap, `10 USDT` position cap, `2 USDT` max loss, maker-only/post-only, default-on forbidden, cancel-all/shutdown proof, and explicit total-control approval before `0615T009`.
- Its final recommendation is `small_cap_live_test_protocol_ready_for_qa`; this can only mean the protocol is ready for QA/controller review.
- QA report is `.workflow/reports/0615T008-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T008` as the latest effective QA result.
- It must not be interpreted as authorization to open live, use credentials, connect endpoints, place/cancel orders, change strategy defaults, deploy, promote, prove PnL, or prove maker viability.

## 0615T005 Task Boundary

- `0615T005` has been created and dispatched after `0615T004` QA passed.
- `0615T005` business execution is complete and QA is `已通过`.
- Scope is no-trading economics fee/rebate read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue economics/account/order collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, fees/rebates/spread-capture proof, PnL proof, or metric proof.
- It may implement local transform/redaction/artifact writing/arithmetic checks/fail-closed checks and validation handoff into `economics_fee_rebate_source.py`.
- It must preserve the accepted rule that fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone cannot prove fees/rebates/spread capture or PnL.
- Its `ready` recommendation can only mean the local no-trading economics fee/rebate read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, economics proof, PnL proof, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `economics_fee_rebate_read_only_source_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005/`.
- QA report is `.workflow/reports/0615T005-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T005` as the latest effective QA result.

## 0615T004 Task Boundary

- `0615T004` business execution is complete and QA is `已通过`.
- Scope is no-trading account inventory read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue account collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, inventory lifecycle proof, or metric proof.
- It may implement local transform/redaction/artifact writing/conservation checks/fail-closed checks and validation handoff into `account_inventory_source.py`.
- It must preserve the accepted rule that order fills alone cannot prove inventory lifecycle.
- Its `ready` recommendation can only mean the local no-trading account inventory read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, inventory lifecycle proof, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `account_inventory_read_only_source_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004/`.
- QA report is `.workflow/reports/0615T004-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T004` as the latest effective QA result.

## 0615T004 Task Boundary

- `0615T004` has been created and dispatched after `0615T003` QA passed.
- Scope is no-trading account inventory read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue account collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, inventory lifecycle proof, or metric proof.
- It may implement local transform/redaction/artifact writing/conservation checks/fail-closed checks and validation handoff into `account_inventory_source.py`.
- It must preserve the accepted rule that order fills alone cannot prove inventory lifecycle.
- Its `ready` recommendation can only mean the local no-trading account inventory read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, inventory lifecycle proof, strategy use, live readiness, deployment, promotion, or maker viability proof.

## 0615T003 Task Boundary

- `0615T003` has been created and dispatched after `0615T002` QA passed.
- `0615T003` business execution is complete and awaiting QA.
- Scope is no-trading private order response read-only collector implementation over task-local fixture inputs only, not endpoint implementation, real venue collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, or metric proof.
- It may implement local transform/redaction/artifact writing/fail-closed checks and validation handoff into `private_order_response_source.py`.
- It must not place, cancel, or amend orders, change strategy behavior, run live/default-on/tiny-live, collect real data, call endpoints, compute fill probability/post-only/real-order-lifecycle metrics, claim execution proof, claim PnL, recommend deployment, or recommend promotion.
- Its `ready` recommendation can only mean the local no-trading read-only collector implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `private_order_response_read_only_collector_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003/`.

## 0615T002 Task Boundary

- `0615T002` has been created and dispatched after `0615T001` QA passed.
- `0615T002` business execution is complete and awaiting QA.
- Scope is private order response read-only collector boundary / implementation design, not endpoint implementation, collector implementation, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, or metric proof.
- It must preserve the `0615T001` convergence policy and must not consume the one allowed local-only exception; it is non-local in direction because it defines a future real source-line / read-only collector boundary.
- It may design endpoint/permission contracts, field handoff into `private_order_response_source.py`, redaction/local-storage policy, no-trading safety gates, future implementation QA gates, next-task sequence, local artifacts, and a business report.
- It must not place or cancel orders, change strategy behavior, run live/default-on/tiny-live, collect real data, call endpoints, compute fill probability/post-only/real-order-lifecycle metrics, claim execution proof, claim PnL, recommend deployment, or recommend promotion.
- Its `ready` recommendation can only mean the private order response read-only collector boundary design is ready for QA/controller review; it cannot authorize implementation, collection, runner consumption, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `private_order_response_read_only_collector_boundary_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_read_only_collector_boundary_0615T002/`.
- The next executable direction, if QA accepts, is a separate no-trading read-only collector implementation that emits local redacted artifacts validated by `private_order_response_source.py`; runner consumption, metrics, strategy/live, deployment, promotion, and maker viability proof remain forbidden until separately scoped and accepted.

## 0615T001 Convergence Point

- `0615T001` has been created as the next formal task after `0612T001` QA passed.
- `0615T001` business execution is complete and QA is `已通过`.
- The purpose is to stop the recent local artifact skeleton / validator chain from expanding indefinitely and to define the next move toward real source-line implementation or a read-only collector.
- Controller convergence policy: after `0615T001` QA, at most `1` additional local-only task may be dispatched.
- That single extra local-only task is allowed only if `0615T001` names a concrete blocker that must be repaired before any real source-line / read-only collector task can be safely scoped.
- If no such named blocker exists, or after that one extra local-only task completes, the next formal execution-proof task must move to real source-line implementation or read-only collector work.
- `0615T001` itself remains design / boundary work only. It must not implement endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live/economics data reads, remote execution, collection, runner consumption, real metrics, PnL proof, strategy/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.
- The expected output is an actionable field-authority / permission-boundary / runner-consumption gate that lets total control dispatch the next non-local task without inventing missing source-line boundaries.
- Final recommendation is `real_source_line_readiness_boundary_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_execution_source_real_readiness_collector_boundary_0615T001/`.
- QA report is `.workflow/reports/0615T001-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T001` as the latest effective QA result.

## 0612T001 Task Boundary

- `0612T001` has been created and dispatched after `0611T004` QA passed.
- `0612T001` business execution is complete and QA is `已通过`.
- Scope is local-only economics fee/rebate settlement artifact skeleton / validator, not economics/private/order/account/live endpoint work, source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T009` economics fee/rebate source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order, `0611T003` replay lifecycle, and `0611T004` account inventory local artifacts may be used only as future cross-check context, not current fees/rebates/spread-capture or PnL proof.
- It may implement local economics/fee/rebate/spread-capture schema constants, fixture loader/parser, fail-closed validation, maker/taker classification checks, settlement checks, currency conversion / tick-value arithmetic checks, spread-capture consistency checks, timestamp policy artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, forbidden endpoint/action fields, missing or conflicting settlement authority, unsupported maker/taker classification, non-conserving fee/rebate arithmetic, currency conversion/tick-value mismatch, settlement timestamp merge/order defects, spread-capture overclaims, account-inventory-alone overclaim, order-fill-alone overclaim, public-markout-alone overclaim, PnL overclaim, and live/deployment/promotion overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, economics/account/private/order/live data, remote execution, collection, runner consumption, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local economics fee/rebate artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `economics_fee_rebate_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001/`.

## 0611T004 Task Boundary

- `0611T004` business execution is complete and QA is `已通过`.
- Scope is local-only account inventory artifact skeleton / validator, not account/private/order/live endpoint work, source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T008` account inventory source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order and `0611T003` replay lifecycle local artifacts may be used only as future cross-check context, not current inventory lifecycle proof.
- It implements local account/inventory schema constants, fixture loader/parser, fail-closed validation, snapshot / transition / conservation checks, reconciliation boundary artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, forbidden endpoint/action fields, missing/stale/partial/conflicting/unsupported snapshots, unknown/ambiguous/conflicting/unsupported transitions, non-conserving quantities, unit/precision/sign inconsistency, duplicate transitions, out-of-order transitions, order-fills-alone inventory proof overclaim, current inventory lifecycle proof overclaim, PnL/economics overclaim, and live/deployment/promotion overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, account/private/order/live data, remote execution, collection, runner consumption, real inventory metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local account inventory artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `account_inventory_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/`.

## 0611T003 Task Boundary

- `0611T003` business execution is complete and QA is `已通过`.
- Scope is local-only replay lifecycle validation / reconciliation gate, not replay/live semantic implementation, exchange endpoint/source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T007` replay lifecycle source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order local skeleton may be used only as future cross-check context, not current queue or race proof.
- It may implement local lifecycle schema constants, fixture loader/parser, fail-closed validation, ordering/reconciliation policy artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, merged or missing timestamp domains, non-monotonic same-order sequence, invalid terminal lifecycle ordering, ambiguous/conflicting/out-of-order events, duplicate lifecycle event identity, duplicate terminal state for the same opaque order reference, cross-order causal overclaim, replay-as-execution-proof overclaim, queue-priority proof overclaim, and cancel-fill-race metric overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, replay/live semantic implementation, private/order/account/live data, remote execution, collection, runner consumption, queue priority metrics, exact queue position proof, cancel-fill race metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local replay lifecycle validation / reconciliation gate is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `replay_lifecycle_validation_gate_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/`.

## 0611T002 Task Boundary

- `0611T002` business execution is complete and QA is `已通过`.
- Scope is local-only `private_order_response` artifact skeleton / validator, not exchange endpoint/source collector work.
- It may use the accepted `0610T006` private-order source-line contract and `0611T001` synthesis gate as design inputs.
- It may implement local schema constants, fixture loader/parser, fail-closed validation, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, conflicting terminal states, duplicate event identity, incomplete lifecycle evidence, unsupported evidence source, and missing/out-of-order timestamps.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `private_order_response_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002/`.

## 0611T001 Task Boundary

- `0611T001` has been created as a design-only source-line synthesis / implementation-readiness gate task after `0610T009` QA passed.
- `0611T001` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_execution_source_line_synthesis_gate_0611T001/`.
- Synthesis gate design is `docs/basis_positive_execution_source_line_synthesis_gate.md`.
- Final recommendation is `source_line_synthesis_gate_ready_for_qa`, meaning only that the synthesis/gate design is ready for QA/controller review.
- QA report is `.workflow/reports/0611T001-qa.md`, and `docs/qa-acceptance-report.md` now records `0611T001` as the latest effective QA result.
- It may consume only QA-passed `0610T009` / `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It must cover exactly four source lines and exactly seven execution gaps from `0610T005`.
- It may define a source-line contract registry, implementation-readiness gate, source dependency reconciliation, forbidden overclaim matrix, and next-task sequence.
- It may recommend future separately scoped implementation tasks, but any such recommendation must explicitly require separate task dispatch and QA before implementation.
- It must not implement or use endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real execution metrics, real economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0611T001` can only mean the synthesis/gate design is ready for QA/controller review; it cannot authorize implementation or metric proof.

## 0610T009 Task Boundary

- `0610T009` has been created as a design-only `economics_fee_rebate_source_line` contract task after `0610T007` / `0610T008` QA passed.
- `0610T009` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_source_line_contract_0610T009/`.
- Source-line contract is `docs/basis_positive_economics_fee_rebate_source_line_contract.md`.
- Final recommendation is `economics_fee_rebate_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- It may consume only QA-passed `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the primary gap assigned to `economics_fee_rebate_source_line`: `fees_rebates_spread_capture`.
- Required contract coverage includes economics artifact schema, fee/rebate settlement taxonomy, spread-capture taxonomy, maker/taker classification policy, currency conversion / tick-value policy, settlement timestamp policy, reconciliation boundary, validation gates, overclaim reject rules, manifest, boundary validation, and business report.
- `0610T006` private-order response artifacts may be used only as future fill dependency context, `0610T007` replay lifecycle artifacts only as future timestamp/order consistency context, and `0610T008` account inventory artifacts only as future reconciliation context.
- The task must explicitly reject hypothetical spread, fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone as proof of fees/rebates/spread capture or PnL.
- It must not implement or use economics/private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T009` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation, source collection, runner implementation, economics proof, PnL proof, or metric proof.

## 0610T007 / 0610T008 Parallel Execution Finding

- `0610T007` and `0610T008` were executed in parallel after total control confirmed the parallel condition.
- Both tasks remained design-only contracts and QA is `已通过`.
- `0610T007` final recommendation is `replay_lifecycle_contract_ready_for_qa`; this means only that the replay lifecycle semantics source-line design is ready for QA/controller review.
- `0610T008` final recommendation is `account_inventory_contract_ready_for_qa`; this means only that the account inventory source-line design is ready for QA/controller review.
- Parallel write-scope check passed at the commit level: `139b76a` / `692f445` touched only T007 task/report/doc/artifact paths, and `66127b7` / `156f6da` touched only T008 task/report/doc/artifact paths.
- Shared tracking files were not modified by the business-thread commits; total control is responsible for this tracking update.
- QA reports were written to `.workflow/reports/0610T007-qa.md` and `.workflow/reports/0610T008-qa.md`; `docs/qa-acceptance-report.md` now contains the latest effective QA result for `0610T008`.
- Neither task authorizes source reader/collector implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T008 Task Boundary

- `0610T008` has been created as a prepared design-only `account_inventory_source_line` contract task.
- It is parallel-eligible with `0610T007` only because both tasks are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`).
- It may consume only QA-passed `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the primary gap assigned to `account_inventory_source_line`: `inventory_lifecycle`.
- Required contract coverage includes account/inventory artifact schema, inventory snapshot taxonomy, inventory transition taxonomy, conservation checks, reconciliation boundary, timestamp policy, fail-closed validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- `0610T006` private order response artifacts may be used only as future transition input / future cross-check context, not current inventory lifecycle proof.
- The task must explicitly state that order fills alone cannot prove inventory lifecycle.
- It must not implement or use account/private/order/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real inventory metrics, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T008` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation, source collection, runner implementation, inventory lifecycle proof, or metric proof.

## 0610T007 Task Boundary

- `0610T007` has been created and dispatched as a design-only `replay_lifecycle_semantics_source_line` contract task after `0610T006` QA.
- It is parallel-eligible with `0610T008` only because both tasks are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`).
- It may consume only QA-passed `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the two primary gaps assigned to `replay_lifecycle_semantics_source_line`: `queue_priority` and `cancel_fill_race`.
- `0610T006` private order response artifacts may be used only as future cross-check / future event-source dependency context, not current proof source for queue priority or cancel-fill race.
- It must define replay/live lifecycle event schema, queue semantics boundary, cancel/fill race event-ordering policy, timestamp policy, replay/live proof-limit rules, validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- It must preserve replay as supporting regression, not execution proof.
- It must not implement replay/live semantics, private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, exact queue position proof, cancel-fill race metric proof, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T007` can only mean the design contract is ready for QA/controller review; it cannot authorize replay/live semantic implementation or metric proof.

## 0610T006 Task Boundary

- `0610T006` has been created and dispatched as a design-only `private_order_response_source_line` contract task after `0610T005` QA.
- `0610T006` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_private_order_response_source_line_contract_0610T006/`.
- Final recommendation is `private_order_response_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- It may consume only QA-passed `0610T005` source decomposition artifacts plus necessary `0610T004` / `0610T003` / `0610T002` QA/business reports and manifests as prior fact sources.
- It may cover only the three primary gaps assigned to `private_order_response_source_line`: `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle`.
- It must define response artifact schema, response label taxonomy, post-only reject taxonomy, lifecycle state taxonomy, timestamp policy, terminal-state consistency, validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- It must not implement or use private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T006` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation or metric proof.

## 0610T005 Task Boundary

- `0610T005` has been created as a prepared design-only source decomposition / source-line routing contract task and is unblocked by `0610T004` QA.
- `0610T005` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Final recommendation is `private_order_source_design_ready_next`, meaning only that a later separately dispatched design-only task may define the `private_order_response_source_line` contract.
- It may consume only QA-passed `0610T004`/`0610T003`/`0610T002` contract, gate, skeleton artifacts plus necessary QA/business reports.
- It must split the seven execution gaps into source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- It must keep `private_order_response_source_line`, `replay_lifecycle_semantics_source_line`, `account_inventory_source_line`, and `economics_fee_rebate_source_line` distinct unless all six split gates match.
- It must not implement source readers, source collectors, runners, private/order/account/live endpoints, user streams, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T004 Task Boundary

- `0610T004` has been created as a prepared fail-closed/read-only runner skeleton implementation task and is unblocked by `0610T003` QA.
- `0610T004` QA is `已通过`.
- `0610T004` business execution implemented the local fail-closed/read-only runner skeleton.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_fail_closed_runner_0610T004/`.
- Final recommendation is `fail_closed_runner_skeleton_ready_for_qa`, meaning only that the skeleton is ready for QA review.
- It may consume only QA-passed `0610T003`/`0610T002` contract and gate artifacts plus necessary QA/business reports.
- It may implement prerequisite/source-policy/schema/overclaim validation and proof-limited unavailable status rows for the seven execution gaps.
- It must not compute real execution metrics or claim fill probability, queue/priority, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion proof.
- It must not read private/order/account/live data or authorize strategy/private/order/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T003 Task Boundary

- `0610T003` completed business execution as a design/gate-only source availability and runner implementation gate task after `0610T002` QA and passed QA.
- It may consume QA-passed `0610T002` artifacts plus necessary `0610T001` / `0609T011` / `0609T010` QA/business reports and manifests only as prior fact sources.
- It classified each of the seven execution gaps as fail-closed placeholder only under current sources and identified blocker/source-design paths before any execution-proof metric can be considered.
- Final recommendation is `runner_skeleton_ready_with_fail_closed_sources`, meaning only that a later separately scoped task may implement a fail-closed/read-only skeleton that validates prerequisites, source policies, output restrictions, gap coverage, and overclaim rejection.
- It must preserve private/order response artifacts as `forbidden_current_task / future_requires_separate_design`.
- It must preserve replay/simulation artifacts as `supporting_regression_not_execution_proof`.
- It must preserve public proxy artifacts as design context only, not execution proof.
- This task must not implement runner behavior or authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T002 Task Boundary

- `0610T002` completed business execution as a design-only read-only execution-evidence runner contract task and passed QA.
- It consumed QA-passed `0610T001` artifacts plus necessary T010/T011/T001 QA/business reports only.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_runner_contract_0610T002/`.
- Runner contract is `docs/basis_positive_execution_evidence_runner_contract.md`.
- Final recommendation is `read_only_execution_evidence_runner_design_ready`, meaning only that the current runner contract/design artifacts are ready for QA/controller review.
- The recommendation does not indicate implementation readiness.
- Private/order response artifacts remain `forbidden_current_task / future_requires_separate_design`.
- Replay/simulation artifacts remain `supporting_regression_not_execution_proof`.
- Public proxy artifacts remain design context only, not execution proof.
- This task does not authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T001 Task Boundary

- `0610T001` completed business execution as a design-only execution-evidence requirements contract task and passed QA.
- It consumed QA-passed `0609T011` artifacts plus necessary T010/T011 QA/business reports only.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_requirements_0610T001/`.
- Design contract is `docs/basis_positive_execution_evidence_requirements.md`.
- Final recommendation is `execution_evidence_runner_contract_ready`, meaning only that a later separately scoped read-only runner contract/design task can be considered after QA.
- Private/order response artifacts are classified as `forbidden_current_task / future_requires_separate_design`.
- Replay/simulation artifacts are classified as `supporting_regression_not_execution_proof`.
- This task does not authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T011 Task Boundary

- `0609T011` completed business execution as a read-only proxy evidence synthesis task and is now `待验收`.
- It consumed only QA-passed T010 local proxy artifacts and T010 QA/business reports.
- Official artifacts are under `local_live_analysis/basis_positive_proxy_evidence_synthesis_0609T011/`.
- Final recommendation is `continue_to_execution_evidence_design`, meaning only that a later separately scoped design task can define execution-layer evidence requirements.
- Metric/sample/proof-class matrices summarize proxy evidence at aggregate level only; no source-row case catalog, case-library behavior, shadow decision, executable trigger, trading instruction, order side, quote price, or quote size is emitted.
- Execution evidence gaps remain unproven: fill probability, queue/priority, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, live readiness, default-on readiness, tiny-live readiness, deployment readiness, promotion, and maker execution viability.
- This task must not be interpreted as strategy/private/order/live/default-on/tiny-live authorization, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T010 Task Boundary

- `0609T010` completed business execution as the read-only maker-viability proxy runner implementation task after `0609T009` QA and has passed QA.
- It implemented only local proxy metrics over T008/T009 allowlisted public/canonical observation-layer artifacts.
- Official artifacts are under `local_live_analysis/basis_positive_maker_viability_proxy_0609T010/`.
- Final recommendation is `read_only_proxy_evidence_ready_for_qa`.
- It keeps all T008 execution-layer gaps unproven and reports proxy results with caveats from the T009 contract.
- It must not implement case-library behavior, source-row case catalogs, shadow decisions, executable triggers, order side, quote price/size, private/account/order endpoints, order lifecycle logic, strategy/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or maker execution viability proof.

## 0609T009 Task Boundary

- `0609T009` completed business execution as a design-only execution-evidence gap planning task after `0609T008` QA and is now `待验收`.
- It may read `0609T008` row-level read-only artifacts and inherited T006/T007 boundary references only as local public/canonical observation-layer inputs.
- It must classify execution-layer questions into proxy-available, proxy-with-caveat, not-provable-without-separate-evidence, or forbidden-for-current-stage categories.
- It may design a later read-only maker-viability proxy runner contract, including input/output schema, validation requirements, and overclaim reject conditions.
- Final recommendation is `read_only_proxy_runner_ready_for_implementation`, meaning only that a later separately dispatched read-only proxy runner implementation task can be considered after QA.
- It must not implement the proxy runner, generate case-library entries, create source-row case catalogs, produce shadow decisions, output executable triggers or trading instructions, set order side/quote price/size, use private/account/order endpoints, touch order lifecycle logic, run live/default-on/tiny-live, run parameter search, recommend deployment, claim promotion, or claim maker execution viability is proven.

## 0609T001 Business Findings

- `0609T001` completed business execution as a read-only basis-positive wrong-way decomposition and targeted sample design task.
- Runner: `examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py`.
- Focused tests: `examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py`.
- Official artifacts: `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/`.
- T006 prerequisite was validated from `basis_positive_robustness_manifest.json`: `final_recommendation=needs_more_samples`, `scope_policy=not_limited_to_regime_011`, and `t005_final_contract_decision=upgrade_to_context_only_supported`.
- Baseline comparison at `1000ms`: `basis > 0` has `2425` rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, wrong-way count `69`, p95 wrong-way loss `138` ticks, and max wrong-way loss `180` ticks; `basis <= 0` has `7564` rows, hit rate `0.33853760`, and mean future move `-16.94407721` ticks.
- Positive-basis magnitude strengthens monotonically by mean future move from small `18.47337278` to medium `29.41516710` to large `90.09975062` ticks, while wrong-way tail is heavier in the small positive-basis bucket.
- Controlled checks classify basis-positive as retaining nontrivial effect within both Binance momentum buckets and Hyperliquid book-state buckets.
- Candidate visible tail hypotheses are `basis_positive_small`, negative Hyperliquid top5 imbalance, and negative Hyperliquid microprice-minus-mid; current sample/time concentration still requires targeted validation rather than direct promotion.
- Final recommendation is `targeted_collection_ready`, meaning only that a future separately dispatched and QA-accepted collection design is now supportable. It does not authorize new collection inside T001, strategy implementation, private/order endpoints, order lifecycle, case-library, shadow decisions, live/default-on/tiny-live, parameter search, or promotion.

## 0608T003-0608T005 Regime 011 / Basis Context QA Findings

- `0608T003` passed QA as the read-only directional momentum viability assessment for `regime_011_1000_spread_10_20_ticks`.
- `0608T003` final recommendation is `reject_directional_edge_unstable`: base regime row count `242`, sample count `3`, direction hit rate `0.51239669`, per-sample signed edge `10.51282051 / -9.04040404 / 2.5` ticks, conservative net edge `-8.07024793` ticks, and tail-risk proxy rejected.
- `0608T004` passed QA as the read-only feature-conditioned validity diagnosis for the same regime.
- `0608T004` final recommendation is `watch_needs_contract_visibility_clarification`, with `valid_supported_pattern_count=0`, `valid_watch_pattern_count=0`, and `invalid_pattern_count=21`.
- In `0608T004`, the strongest non-tail/non-redundancy-looking local pattern was `context_basis_mid_ticks > 0`: `57` rows, `3` samples, hit rate `0.94736842`, net edge proxy `88.54385965` ticks, and `tail_risk_acceptable_proxy`; it was still classified `invalid_not_decision_visible` only because the prior data contract treated basis as diagnostic/caveated context.
- `0608T005` passed QA as the read-only basis-context visibility / lineage diagnosis for `context_basis_mid_ticks > 0`.
- `0608T005` final contract decision is `upgrade_to_context_only_supported`: basis lineage is confirmed as `(binance_mid_px - hyperliquid_mid_px) / 0.1`, `57/57` rows pass formula reconstruction with max error `0.0`, `57/57` rows are timestamp/as-of clean, future input joins and missing input joins are both `0`, all three canonical samples contribute, and max sample row share is `0.38596491`.
- `0608T005` persistence check did not reverse direction at `1000/5000/10000ms`; `100/250ms` remains watch/alias context only.
- Controller interpretation: Regime 011 should not progress to maker case-library, directional case-library, shadow decisions, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion. `context_basis_mid_ticks > 0` may be used only as decision-time context in later read-only research, retaining execution-PnL caveat.
- Contract amendment after `0608T005`: `basis_mid_dislocation` / `context_basis_mid_ticks` is now `allow / context_only_supported` in the `0601T004` data contract; `basis_microprice_dislocation` remains `diagnostic_only / diagnostic_context`.
- `0608T006` business execution is complete and awaiting QA as a read-only basis-positive robustness diagnosis outside the Regime 011 shell.
- `0608T006` final recommendation is `needs_more_samples`: `context_basis_mid_ticks > 0` has `2425` primary rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, and positive persistence at `1000/5000/10000ms`, but max sample row share is `0.67917526`, join-age and volatility coverage are narrow, and the cost/tail proxy classifies `cost_tail_reject` with p95 wrong-way loss `138` ticks.
- `0608T006` collinearity check did not classify basis-positive as solely explained by Hyperliquid top5 imbalance or microprice-minus-mid; both rows are `not_explained_solely_by_hl_book_state`.
- `0608T006` does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, executable trading instructions, or promotion.

## 0604T015 Task Boundary

- `0604T015` has been created to diagnose the post-`0604T013` live shutdown gap where shutdown now calls `wait_order_response()` but still cannot prove cancel acknowledgement.
- The task must reproduce and attribute all reported issues: `0` as `Ok`/timeout and batch-folded received response, `3` as order response but not canceled-state proof, raw `wait_result` logging without classification, misleading `ack_waits` counter semantics, fake tests returning `0` as success, and missing final local/REST/audit tail proof.
- Scope is diagnosis, stable reproduction, range check, root cause, and test coverage gap reporting only.
- It must not repair `live_tick_mm.py`, py bindings, Rust live bot/backtest, connector, production config, audit schema, normal loop cancel semantics, or start live/default-on/tiny-live/promotion.

## 0604T016 Task Boundary

- `0604T016` has been created as the follow-up repair task for live shutdown cancel acknowledgement proof semantics.
- The repair must keep `order_response_received` and `terminal_confirmed` as independent dimensions.
- Hard acceptance line: no code path may set `terminal_confirmed=True` solely from `wait_result == 3` or `order_response_received=True`; terminal confirmation must require independent final order-state proof.
- The task now fixes allowed enum values: `wait_outcome` may only be `order_response_received`, `ok_unknown_or_timeout`, `wait_error`, or `not_requested`; `terminal_confirmation_source` may only be `local_orders`, `rest_open_orders`, or `none`, with `rest_open_orders` unavailable unless a safe local proof path exists.
- Scope is limited to `live_tick_mm.py` shutdown helper result semantics, shutdown summary logging, focused tests, and the task business report.
- It must not modify py bindings, Rust live bot/backtest, connector, production config, audit schema, normal loop cancel semantics, or start live/default-on/tiny-live/promotion.

## Binance Maker MM Test Environment Finding

- For `examples/binance_tick_mm`, prefer `/home/molly/anaconda3/envs/hftbacktest/bin/python` for pytest verification.
- Verified command on 2026-06-04: `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm` -> `282 passed in 2.92s`.
- The generic/base `python -m pytest examples/binance_tick_mm` can import `/home/molly/anaconda3/lib/python3.13/site-packages/hftbacktest/data/utils/tardis.py` during `run_env_test.py` collection and fail before test execution with numba cache locator error: `RuntimeError: cannot cache function '_convert_depth': no locator available`.
- Treat that base-env failure as an environment/collection issue, not as a Binance maker MM regression, when the same suite passes in the project `hftbacktest` conda env.

## 0604T006 Business Findings

- `0604T006` completed business execution; its initial QA found only a report bucket-consistency defect, and `0604T008` has repaired that defect with QA `已通过`.
- New runner: `examples/hyperliquid/canonical_signal_quality_ranking.py`.
- Focused tests: `examples/hyperliquid/test_canonical_signal_quality_ranking.py`.
- Task artifacts: `local_live_analysis/canonical_signal_quality_ranking_0604T006/`.
- The runner consumes `0604T003` canonical event-mode artifacts through the `0604T004` loader path and refuses non-canonical / diagnostic-only synthetic inputs.
- Ranking is limited to the four `0601T004` primary allowlist features and writes `signal_quality_ranking.csv`, `signal_quality_reject_watch_list.csv`, `signal_quality_ranking_manifest.json`, and `signal_quality_ranking_report.md`.
- Ranking result: `binance_mid_move_ticks_from_prev` rank 1 / `keep_for_read_only_research`; `binance_top5_imbalance` rank 2 / `watch_regime_dependent`; `binance_top5_bid_qty` rank 3 / `watch_regime_dependent`; `binance_microprice_minus_mid_ticks` rank 4 / `watch_regime_dependent`; rejects `0`.
- The result matches the current controller interpretation at the ordering level: mid move is the most stable global candidate, top5 imbalance remains the strongest book-pressure candidate but is watch-labeled by the strict concentration/short-horizon caveats, bid qty remains liquidity/context watch, and microprice-minus-mid remains regime-dependent watch.
- Downstream work should use the T008-refreshed T006 artifacts under `local_live_analysis/canonical_signal_quality_ranking_0604T006/`, where the report now mechanically matches each feature's `final_bucket`.
- This is read-only signal quality ranking only. It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

## 0604T007 Business Findings

- `0604T007` completed business execution and passed QA.
- New runner: `examples/hyperliquid/canonical_horizon_regime_diagnostics.py`.
- Focused tests: `examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py`.
- Task artifacts are under `local_live_analysis/canonical_horizon_regime_diagnostics_0604T007/`: `horizon_independence_diagnostics.csv`, `regime_conditioning_diagnostics.csv`, `regime_watch_list.csv`, `horizon_regime_diagnostics_manifest.json`, and `horizon_regime_diagnostics_report.md`.
- The runner consumes the `0604T003` canonical event-mode aggregate through the `0604T004` loader path and refuses diagnostic-only synthetic inputs.
- Horizon findings: `100/250ms` are `watch_needs_more_samples`; `500/1000/5000/10000ms` are `diagnostic_supported`, with `1000ms+` explicitly preferred for interpretation.
- Regime findings remain watch-only diagnostics: support counts are `10 diagnostic_supported`, `6 watch_needs_more_samples`, and `2 reject_aliased_or_concentrated`; no bucket is promoted into a final regime or maker action.
- This task does not authorize new collection, final regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T008 QA Findings

- `0604T008` passed QA as the narrow T006 signal-ranking report bucket-consistency repair.
- It fixed `examples/hyperliquid/canonical_signal_quality_ranking.py` so the `Controller Interpretation Check` is generated from actual `ranking_rows` / `final_bucket` values instead of static text.
- Focused tests now include a regression check that every feature's report interpretation line matches its generated bucket and that `binance_top5_imbalance` cannot be written as kept when it is `watch_regime_dependent`.
- Refreshed T006 artifacts keep the same ranking semantics: `binance_mid_move_ticks_from_prev=keep_for_read_only_research`; `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks=watch_regime_dependent`.
- QA verified temporary and official report/CSV consistency with `checked_features 4` and `missing []`.
- This repair does not change ranking scoring, allowlist, source-lock guard, canonical loader, strategy, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.

## 0604T009 Business Findings

- `0604T009` completed business execution and is awaiting QA.
- New runner: `examples/hyperliquid/canonical_regime_synthesis.py`.
- Focused tests: `examples/hyperliquid/test_canonical_regime_synthesis.py`.
- Task artifacts are under `local_live_analysis/canonical_regime_synthesis_0604T009/`: `candidate_regime_definitions.csv`, `candidate_regime_evidence_summary.csv`, `candidate_regime_watch_reject_list.csv`, `canonical_regime_synthesis_manifest.json`, and `canonical_regime_synthesis_report.md`.
- The runner consumes `0604T003` canonical event-mode evidence through the accepted `0604T004/0604T005` loader/source-lock guard path and refuses diagnostic-only synthetic inputs.
- Classification counts: `candidate_for_milestone3_executability=1`, `watch_needs_more_samples=6`, `reject_unstable_direction=9`, `reject_concentrated_or_aliased=2`.
- The only read-only candidate for later Milestone 3 executability assessment is `regime_011_1000_spread_10_20_ticks`: primary anchor `binance_mid_move_ticks_from_prev`, horizon `1000ms`, context `primary_usable / fresh_0_50ms / spread_10_20_ticks`, row count `242`, sample count `3`, effective future-row-delta support `3`.
- `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks` remain secondary context only and are not allowed as promoted primary anchors.
- This task does not authorize maker side, quote behavior, order behavior, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

## 0604T005 QA Findings

- `0604T005` passed QA as read-only canonical evidence source lock / guard hardening.
- The reusable guard API was added to `examples/hyperliquid/canonical_event_mode_evidence.py`: `guard_canonical_event_mode_evidence`, `validate_canonical_source_lock_manifest`, and `build_canonical_source_lock_artifacts`.
- Task artifacts are under `local_live_analysis/canonical_evidence_source_lock_0604T005/`: `canonical_source_lock_manifest.json`, `canonical_guard_check_report.md`, and `negative_guard_validation_report.csv`.
- The canonical guard accepts `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` as formal event-mode evidence with `canonical_sample_count=3` and names the `0604T004` loader/foundation as the required foundation artifact source.
- The guard rejects `synthetic_diagnostic_comparison` as formal evidence unless explicit negative validation is requested; negative validation reports `canonical_sample_count=0` and `diagnostic_rejection_count=3`.
- Focused pytest now covers accepted canonical source, rejected diagnostic source, missing source-lock metadata, zero-canonical formal-evidence failure, and downstream-worker-style guard consumption.
- QA verification passed: `--help`, `py_compile`, focused pytest (`10 passed`), true canonical source-lock rerun, synthetic diagnostic negative validation, manifest JSON parse, and `git diff --check`.
- This remains read-only canonical evidence source-lock / guard hardening only. It does not authorize signal ranking, regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T005-0604T007 Dispatch Boundary

- `0604T005`, `0604T006`, and `0604T007` have been dispatched as parallel read-only workers after `0604T004` QA.
- `0604T005` owns Milestone 0 canonical evidence source lock / guard hardening.
- `0604T006` owns Milestone 1 canonical signal quality ranking over the four accepted Binance lead allowlist features.
- `0604T007` owns Milestone 1 canonical horizon / regime diagnostics.
- All three workers must consume `0604T003` canonical event-mode evidence through the `0604T004` loader/foundation and must exclude ordinary synthetic fixed-grid diagnostics from formal evidence.
- The dispatch does not authorize new collection, final high-confidence regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T004 QA Findings

- `0604T004` passed QA as a read-only canonical event-mode evidence loader / validator foundation.
- New module: `examples/hyperliquid/canonical_event_mode_evidence.py`.
- Focused tests: `examples/hyperliquid/test_canonical_event_mode_evidence.py`.
- Task artifacts: `local_live_analysis/canonical_event_mode_evidence_0604T004/`.
- The loader validates required `0604T003` aggregate files and columns, requires canonical samples to have `decision_mode=event` plus `canonical_status=canonical_event_mode`, and excludes `diagnostic_only_synthetic_decision_grid` samples from canonical outputs.
- Accepted canonical aggregate result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`.
- Synthetic diagnostic validation result: `canonical_sample_count=0`, `diagnostic_rejection_count=3`.
- Diagnostic-only synthetic comparison has an empty venue-state CSV because there are no canonical venue-conditioning rows; the loader accepts that only for all-diagnostic/no-canonical inputs while keeping canonical inputs strict.
- Required outputs exist: `canonical_sample_manifest.json`, `canonical_sample_quality_summary.csv`, `diagnostic_rejection_report.csv`, and `canonical_evidence_validation_report.md`; a parallel negative-validation output exists under `synthetic_diagnostic_validation/`.
- QA verification passed: `--help`, `py_compile`, focused pytest (`5 passed`), true canonical input rerun, synthetic diagnostic input rerun, and `git diff --check`.
- This is only a read-only loader/validator foundation for later parallel analysis. It does not authorize signal ranking, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T004 Task Boundary

- `0604T004` has been created as the narrow serial foundation before any parallel Milestone 0 / Milestone 1 development.
- It must implement only a read-only canonical event-mode evidence loader / validator over accepted `0604T003` artifacts.
- The loader must admit `decision_mode=event` / `canonical_status=canonical_event_mode` samples and exclude `diagnostic_only_synthetic_decision_grid` samples from the canonical evidence set.
- Required outputs are `canonical_sample_manifest.json`, `canonical_sample_quality_summary.csv`, `diagnostic_rejection_report.csv`, and `canonical_evidence_validation_report.md`.
- This task intentionally does not do signal ranking, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T003 QA Findings

- `0604T003` passed QA and is the formal Binance-led Hyperliquid pricing-signal robustness evidence source.
- Canonical event-mode aggregate under `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` produced `sample_count=3`, `canonical_sample_count=3`, `diagnostic_synthetic_sample_count=0`, and recommendation `continue_read_only_runner_refinement`.
- Synthetic diagnostic comparison under `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison/` produced `sample_count=3`, `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and recommendation `needs_more_public_samples`.
- Future robustness decisions must use event-driven Hyperliquid decision rows plus de-aliased future-row-delta diagnostics as canonical evidence.
- Ordinary synthetic fixed-grid artifacts remain parseable only for backward-compatible diagnostics and must not be interpreted as independent short-horizon stability evidence.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, connector/core API changes, standard npz schema changes, canonical Binance maker audit schema changes, or promotion.

## 0604T003 Task Boundary

- `0604T003` has been created to repair the remaining ordinary synthetic pricing-signal / robustness artifact risk after `0604T001` and `0604T002`.
- The task should make event-driven Hyperliquid decision rows plus de-aliased future-row-delta diagnostics the canonical decision path for Binance-led Hyperliquid pricing-signal and multi-sample robustness research.
- Synthetic fixed-grid artifacts may remain parseable for backward-compatible diagnostics, but must be marked or treated as diagnostic-only when `100/250/500ms` nominal horizons alias to the same future row.
- The task must ensure recommendation logic uses independent effective horizon / future-row-delta evidence rather than nominal horizon count alone.
- Required task-scoped evidence should use existing local event-mode artifacts under `local_live_analysis/event_horizon_comparison_0604T002/**`; no new collection is authorized.
- The task does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, connector/core API changes, standard npz schema changes, canonical Binance maker audit schema changes, or promotion.
- Business execution and QA are complete. Canonical event-mode a/b/c aggregate produced `canonical_sample_count=3` and `continue_read_only_runner_refinement`; ordinary synthetic a/b/c diagnostic comparison produced `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and `needs_more_public_samples`.

## 0601T006 QA Findings

- `0601T006` passed QA as public-only collection / initial synthetic-grid multi-sample aggregate evidence.
- Accepted collection/process evidence includes `xemm_0603_quiet_b` and `xemm_0603_quiet_c` collected on `awsserver1`, copied back locally, and processed through local alignment, as-of join, lead-lag analysis, pricing-signal runner, and aggregate robustness runner.
- The task produced the required initial aggregate artifacts under `local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/` with `sample_count=4` and recommendation `continue_read_only_runner_refinement`.
- Because `0604T001-0604T003` later proved and repaired fixed-grid horizon aliasing, the ordinary synthetic-grid `0601T006` aggregate is accepted only as precursor collection / diagnostic evidence.
- The formal robustness interpretation is superseded by `0604T003` canonical event-mode artifacts; synthetic fixed-grid outputs must not be used as canonical short-horizon independent evidence.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T006 Task Boundary

- `0601T006` has been created as the Binance-led Hyperliquid public multi-sample robustness validation task.
- It is unblocked by `0601T005` QA, which accepted the four-feature read-only pricing-signal runner with a `single_public_sample_caveat=true`.
- The task must validate whether the four primary allowlist features are stable across multiple synchronized public samples: `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Scope is public-only Binance lead / Hyperliquid lag synchronized samples. Target coverage is 2-3 new samples across active/high-vol, quiet/low-vol, and normal-liquidity regimes when feasible.
- Each sample should reuse the accepted chain: synchronized public collection, `0601T002` style as-of join, `0601T003` style lead-lag analysis, and `0601T005` pricing-signal runner.
- Required aggregate outputs are `multi_sample_manifest.json`, `sample_quality_matrix.csv`, `feature_horizon_stability_across_samples.csv`, `effective_horizon_aliasing_by_sample.csv`, `venue_state_conditioning_across_samples.csv`, and `pricing_signal_robustness_recommendation.md`.
- Final recommendation must stay within the task taxonomy: `continue_read_only_runner_refinement`, `needs_more_public_samples`, `narrow_to_specific_venue_state_regime`, or `reject_for_runner_design`.
- The task does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T005 QA Findings

- `0601T005` passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- QA reran help, py_compile, focused pytest, `/tmp` reproduction, official manifest JSON parse, row-count/allowlist/recommendation checks, and `git diff --check`; all passed.
- Official and `/tmp` reproduction row counts matched: input `3599`, primary `3596`, excluded `3`, pricing signal rows `21541`, feature quality rows `4`, horizon label rows `30`, feature/regime rows `540`, and venue-state conditioning rows `54`.
- Primary allowlist enforcement passed: only `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty` appear as primary features.
- Future labels are separated from decision-time input fields; QA found no `input_*future*` fields.
- Recommendation is `keep_for_read_only_research`, with `single_public_sample_caveat=true`.
- Boundary grep found only prohibitions/scope text/manifest flags, not execution paths for private/order/live/strategy/parameter/default-on/tiny-live/promotion.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T005 Business Findings

- `0601T005` completed business execution and then passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- Runner: `examples/hyperliquid/binance_led_pricing_signal_runner.py`.
- Focused tests: `examples/hyperliquid/test_binance_led_pricing_signal_runner.py`.
- Output directory: `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/`.
- Required artifacts exist: `run_manifest.json`, `pricing_signal_rows.csv`, `pricing_signal_feature_quality.csv`, `horizon_label_summary.csv`, `feature_stability_by_regime.csv`, `venue_state_conditioning_summary.csv`, and `pricing_signal_recommendation.md`.
- The runner consumed only accepted local `0601T002/0601T003/0601T004` artifacts and did not perform network collection.
- Primary evidence remains `3596` rows with `3` excluded rows; generated pricing signal rows are `21541`.
- The runner enforces the four `0601T004` primary Binance allowlist features: `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Future labels are separated from decision-time input columns and include nominal horizon plus `effective_future_age_ms`.
- Recommendation is `keep_for_read_only_research` with `single_public_sample_caveat=true`.
- Binance and Hyperliquid trade pressure remain disabled as `disabled_unverified_side_semantics`.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T004 QA Findings

- `0601T004` passed QA as the Binance-led Hyperliquid maker data input / next-runner contract.
- Output contract: `docs/binance_led_hyperliquid_maker_data_input_contract.md`.
- Task artifacts: `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/`.
- The contract separates Binance lead pricing inputs from Hyperliquid lag venue-state/context inputs.
- Primary Binance lead allowlist is `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Diagnostic-only inputs include absolute Binance top5 microprice, rolling volatility/liquidity context, Hyperliquid venue-state conditioning fields, and contract-caveated basis/dislocation fields.
- Binance and Hyperliquid trade pressure remain disabled until side semantics are separately proven by a QA-accepted task.
- The only authorized follow-up is a later read-only pricing-signal runner. No private/order endpoint, order lifecycle, strategy implementation, live/default-on/tiny-live, parameter search, schema/connector/core API change, or promotion is authorized.

## 0601T003 QA Findings

- `0601T003` passed QA as the read-only Binance-to-Hyperliquid lead-lag stability analyzer.
- Accepted output directory: `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/`.
- Primary rows: `3596`; excluded rows: `3`; horizon observations: `129246`.
- Verdict counts: `18 stable_enough_for_pricing_research`, `6 watch_only`, `30 unstable`, `0 insufficient_samples`.
- QA accepted the effective future-age audit fields in the horizon/regime/basis/venue-state summaries.
- Because the current Hyperliquid decision grid is roughly 500ms, nominal `100/250/500ms` horizons may map to the same future row; later tasks must report both nominal horizon and effective future age.
- The result supports only later read-only pricing-signal/data-input contract design and does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.

## 0602T001 QA Findings

- `0602T001` passed QA as a synchronized public-only Binance lead / Hyperliquid lag collection task.
- The accepted output directory is `local_live_analysis/cross_exchange_public_sample_0602T001/`.
- Synchronization overlap is `1800.105259472s`, passing both the `600s` minimum and `1800s` target gates.
- Binance public data is present with `depthUpdate=67210`, `trade=135126`, `bookTicker=816079`, public depth snapshot status `ok`, `top5_row_count=67211`, `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, and final data row mapping coverage `1.0`.
- Hyperliquid public data is present with `l2Book=3332`, `trades=3311`, `trade_event_count=12840`, `topn_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `missing_join_count=0`, and classification `passes_pricing_research_market_view`.
- The sample is accepted only as synchronized public-data input for a later `0601T002` read-only join. It does not establish a Binance-lead / Hyperliquid-lag statistical effect and does not authorize private/order endpoints, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.

## 0601T001 QA Findings

- `0601T001` passed QA as a Hyperliquid lag-venue public-only BTC sample collection and alignment task.
- The accepted output directory is `local_live_analysis/hyperliquid_public_sample_0601T001/`.
- Raw sha256 is `62ebed4f4cc7a5fc9846f9491f9bd3ae0f06ab5b5f1a766d15aa844c50c1bd4c`; collection duration was `1800.093269476s`.
- Collection metrics are `l2Book=3328`, `trades=4939`, `subscriptionResponse=2`, reconnect count `0`, and recovery snapshot count `1`.
- Alignment classification is `passes_pricing_research_market_view`, with `topn_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `missing_join_count=0`, and event order validation `passed`.
- This remains lag-venue state / execution-context evidence only and does not authorize private/order endpoints, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.

## 0531T001 QA Findings

- `0531T001` passed QA as the Hyperliquid public market-data research consumer read-only implementation.
- The consumer reads accepted `0529T004` local public artifacts and writes deterministic outputs under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Final classification is `passes_pricing_research_market_view`; trade pressure remains explicitly disabled with `unverified_side_semantics`.
- QA accepted the implementation as read-only/local/public-only and found no private/order/live/parameter/default-on/tiny-live/promotion boundary crossing.

## 0531T002 Task Boundary

- `0531T002` has been created as the Binance Stage 9N clean-fill evidence viability refinement task.
- It is unblocked by `0530T002` QA, which passed after total controller ratified / accepted the already collected Stage 9M artifact. The task remains read-only and still must not collect new data.
- The task must read only accepted local artifacts from `0529T002`, `0529T005`, and `0530T002`; it must not collect new data, deploy remotely, modify strategy behavior, enable candidates, relax guards, run parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, connector/core API changes, schema changes, or Hyperliquid work.
- Required outputs live under `local_live_analysis/stage9n_clean_fill_refinement_0531T002/` and must include run manifest, fill-flow decomposition, axis fill-rate summary, top-gap viability, candidate-regime triage, collection-time estimate, and recommendation markdown.
- The core decision is whether Stage 9M evidence supports stopping top-gap collection, doing only a short threshold-crossing collection, or pivoting to alternative decision-visible regime refinement. Any policy design or strategy implementation remains a separate later task after QA.

## 0530T002 Findings

- `0530T002` passed QA after total controller explicitly ratified / accepted the already collected Stage 9M artifact `5-31-stage9m-cleanfill-control-120min-a`.
- Existing-sample scan came first. The only usable not-yet-included sample, `5-13-day-control-30min`, had `0` top-gap rows and `0` top-gap fills, so it could not add relevant Stage 9L clean-fill evidence.
- One new current-format no-rule/default-off control sample was collected as `5-31-stage9m-cleanfill-control-120min-a`, from `2026-05-30T16:16:06Z` to `2026-05-30T18:16:06Z`, with deployed commit `4760d481da3a06021ce25f9de4f2f0914662c5e0`, `git.dirty=false`, stop exit code `0`, and archive sha256 `f25ff59f0dc67bfc5a1ac99d43612ac1acdfdcb451ff7d26a20feb0eba3234f7`.
- New sample validation passed: maker acceptance and market-view passed; T009 decision join coverage is `1.0`; future/gap/missing joins are `0/0/0`; `top5_join_age_ms_p99=27.6887635`; top5 tick/qty match is `0.9618792312/0.9463946567`.
- New sample derived labels are usable: Stage 5 has `5742` submits, `106` fills, and `42` fill-after-cancel orders; Stage 5C has `0` post-only risk rows after recheck; Stage 6 is `methodology_valid_single_sample` with `5741` matched submits and live/replay fills `106/108`.
- Stage 9K aggregate after adding the sample has clean-only rows/fills `41008/1102`, up from `35266/994`, but `ready_for_policy_design` remains `0`; Shape A and Shape B candidate rows remain `0`.
- The top Stage 9L clean-fill gap only improved from `2605` rows / `34` fills / `7` samples / `6` fill samples to `2901` rows / `36` fills / `8` samples / `7` fill samples. It still needs `4` more fills for the Stage 9L minimum and did not meet the interpretive `+20` top-gap target.
- Stage 9L final classification remains `needs_targeted_clean_fills`; coarsened ready bucket count remains `0`; coarsened needs-more-clean-fills buckets are `335`; coarsened reject-quality-negative buckets are `677`; shape candidate count remains `0`.
- QA reproduced the Stage 9M chain and did not find metric/artifact failures. Original fixed logs/reports did not record the required pre-start approval for the remote/live `120min` collection, but total controller later ratified / accepted the already collected artifact; QA acceptance is based on that current controller decision.
- Policy design remains blocked. This task does not authorize strategy behavior changes, candidate enablement, guard relaxation, parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, exact queue claims, hidden queue assumptions, connector/core API changes, or Hyperliquid work.

## 0531T001 Task Boundary

- `0531T001` has been created as a read-only Hyperliquid public market-data research consumer implementation task.
- It is unblocked by `0530T001` QA, which passed on 2026-05-31.
- The implementation must read only accepted local `0529T004` public artifacts and write deterministic research artifacts under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Required outputs are `run_manifest.json`, `market_view_timeseries.csv`, `pricing_features.csv`, `feature_quality_summary.json`, `sample_session_quality_summary.json`, and `research_recommendation.md`.
- The task must not collect a new sample, connect to Hyperliquid public or private endpoints, implement private connector or order lifecycle, run strategy live, run parameter search, default-on behavior, tiny-live, promotion, connector/core API changes, standard npz schema changes, or canonical audit schema changes.

## 0531T001 Implementation Notes

- The consumer has been implemented at `examples/hyperliquid/hyperliquid_market_data_research.py` with focused tests in `examples/hyperliquid/test_hyperliquid_market_data_research.py`.
- It reads only the accepted `0529T004` public sample artifacts, validates raw sha256 consistency across the raw file and manifests, and writes deterministic research outputs under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Generated outputs include `run_manifest.json`, `market_view_timeseries.csv`, `pricing_features.csv`, `feature_quality_summary.json`, `sample_session_quality_summary.json`, and `research_recommendation.md`.
- Final classification on the accepted sample is `passes_pricing_research_market_view`.
- Trade pressure is intentionally left disabled with explicit `unverified_side_semantics` status so ambiguous public trade side semantics do not become candidate-ready decision features.
- No fresh collection, private connector, order lifecycle, strategy live, parameter search, default-on behavior, tiny-live, promotion, or canonical schema change was introduced.

## 0531T002 Findings

- `0531T002` finalized as `short_collection_to_cross_minimum_only`.
- Top Stage 9L gap advanced from `34 -> 36` fills and remains `4` fills short of the clean `40` minimum.
- The new `120min` control sample added `+108` aggregate Stage 9K clean-only fills, but only `+2` fills reached the top gap.
- Top-gap viability is still better than the main alternatives on markout and spread capture; the alternatives are higher-fill-rate but materially worse quality.
- Estimated follow-up window is `4h-6h` only to cross the minimum; the `+20` interpretive target is not worth pursuing on this line.
- No new data collection, strategy change, candidate enablement, guard relaxation, parameter search, default-on, tiny-live, promotion, replay semantic change, connector/core API/schema change, or Hyperliquid work was introduced.

## 0530T002 Task Boundary

- `0530T002` has been created as the Binance Stage 9M targeted clean-fill evidence collection / read-only rerun task following `0529T005`.
- The task targets the top Stage 9L gap: `churn_warning_coarsened / large_skew_or_low_score / add_side / step_back_gt1 / edge_non_adverse / market_view_usable / post_only_clean / warning_churn_context`, which had `2605` rows, `34` fills, `7` samples, and needed `6` more clean fills for the Stage 9L minimum threshold.
- Target evidence is current-format no-rule/default-off control only. The task should first scan existing accepted current-format samples; if none can add relevant clean-fill evidence, it may prepare one `120min` control collection, but remote/live startup requires separate explicit approval.
- The target is preferably at least `+20` clean fills in the top-gap regime, or at least `+60` clean fills across Stage 9L gap regimes, followed by maker acceptance, T009 sidecar/join, Stage 5, Stage 5C, Stage 6, Stage 9K, and Stage 9L rerun. This target is interpretive, not a hard QA pass/fail gate.
- The task must report whether any coarsened bucket reaches `ready_for_policy_design_after_coarsening`; absent that, policy design remains blocked.
- It does not authorize strategy behavior changes, candidate enablement, guard relaxation, parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, exact queue claims, hidden queue assumptions, connector/core API changes, or Hyperliquid work.

## 0530T001 Findings

- `0530T001` passed QA as a design-only/read-only Hyperliquid public market-data research consumer contract task.
- Direct input is the accepted `0529T004` fresh public-only BTC sample and its QA-approved artifacts under `local_live_analysis/hyperliquid_public_sample_0529T004/`.
- The design document is `docs/hyperliquid_public_market_data_research_consumer_design.md`.
- Official Hyperliquid public docs were reachable and rechecked successfully during execution; QA also rechecked the public docs URLs and received HTTP 200.
- The accepted consumer contract defines required inputs, output artifacts, allowed public decision-time-visible pricing / market-view features, diagnostic-only labels, quality gates, classification taxonomy, and a later read-only implementation boundary.
- The immediate next Hyperliquid task should be read-only consumer implementation over accepted local `0529T004` artifacts only.
- The task is independent from Binance `0529T005` and must not modify Binance Stage 9L work.
- It does not authorize consumer implementation, private connector, account endpoints, order submit/cancel/fill lifecycle, strategy live logic, parameter search, default-on behavior, tiny-live, promotion, connector/core API changes, standard npz schema changes, or canonical audit schema changes.

## 0529T005 Findings

- `0529T005` implements the read-only Stage 9L fill-quality rejection decomposition runner at `examples/binance_tick_mm/fill_quality_rejection_decomposition.py` with focused tests at `examples/binance_tick_mm/test_fill_quality_rejection_decomposition.py`.
- The runner consumes `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/run_manifest.json`, reconstructs row-level observed submit rows from existing Stage 5 labels, Stage 5 fill markouts, Stage 5C safety diagnostics, and live audit fields, and recomputes coarsened `sample_count` / `fill_sample_count` from row-level `sample_id`.
- Stage 9L artifacts are under `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/`: `run_manifest.json`, `rejection_reason_decomposition.csv`, `churn_gate_sensitivity.csv`, `coarsened_trigger_bucket_metrics.csv`, `coarsened_shape_candidates.csv`, `sample_gap_by_regime.csv`, and `stage9l_recommendation.md`.
- Final classification is `needs_targeted_clean_fills`; Shape A / Shape B candidate rows remain `0`, coarsened ready buckets remain `0`, coarsened needs-more-clean-fills buckets are `310`, and coarsened reject-quality-negative buckets are `662`.
- Churn hard-gate sensitivity shows demoting churn diagnostics to warning does not by itself create ready candidates:
  - original hard gate: `0` ready, `68` needs-more, `246` reject
  - recent reject/throttle as warning: `0` ready, `197` needs-more, `117` reject
  - fast-cancel / cancel-readd as warning: `0` ready, `154` needs-more, `160` reject
  - all non-true-reject churn diagnostics as warning: `0` ready, `283` needs-more, `31` reject
- Top targeted clean-fill gap is `churn_warning_coarsened / large_skew_or_low_score / add_side / step_back_gt1 / edge_non_adverse / market_view_usable / post_only_clean / warning_churn_context`, with `2605` rows, `34` fills, `7` samples, and `6` more clean fills needed to meet the Stage 9L minimum fill threshold.
- `0529T005` QA passed on 2026-05-30. QA reran help/py_compile/focused tests, manifest JSON parse, Stage 9L reproduction to `/tmp/qa_0529T005_stage9l`, and key artifact count checks; all passed.
- This remains read-only/default-off evidence. It does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, hidden queue assumptions, or replay semantic changes.

## 0529T004 Findings

- `0529T004` implements a narrow public-only Hyperliquid collector at `examples/hyperliquid/hyperliquid_public_sample.py` and focused tests at `examples/hyperliquid/test_hyperliquid_public_sample.py`.
- The collector writes line-oriented `raw.gz`, `raw.sha256`, `collection_manifest.json`, and `recovery_snapshots.jsonl` under `local_live_analysis/hyperliquid_public_sample_0529T004/`.
- The local `websockets` package was unavailable, so no dependency installation was performed; the collector used the already installed `websocket-client` fallback and records that fact in the manifest.
- Fresh 120s public-only BTC collection succeeded on mainnet after the first sandboxed DNS failure required network escalation: `l2Book=222`, `trades=111`, subscription responses `2`, connection attempts `1`, reconnect count `0`, startup Info `l2Book` recovery snapshot count `1`, raw sha256 `137018ef937b3692a5de0c12ee009c4a93a0e6d62ff15321061c377fc514389c`.
- The T003 alignment runner now consumes collection manifest and recovery snapshot evidence. Fresh T004 alignment produced `data.npz` with `4279` rows, raw parse errors `0`, trade events `418`, top-N coverage `1.0`, synthetic join coverage `1.0`, future joins `0`, missing joins `0`, and event-order validation `passed`.
- Final classification is `passes_pricing_research_market_view`, upgrading beyond T003's old local-sample `limited_pricing_research` because subscription/session/recovery evidence is now present.
- `0529T004` QA passed on 2026-05-30. QA reran help/py_compile/focused tests, raw sha256 check, metrics assertions, and alignment regeneration to `/tmp/qa_0529T004_alignment`; all passed.
- This remains market-data-only evidence. It does not authorize a Hyperliquid private connector, account endpoints, order submit/cancel, fill lifecycle, strategy live logic, parameter search, default-on behavior, guard relaxation, tiny-live, promotion, or Binance strategy changes.

## 0529T002 Findings

- `0529T002` implements the read-only Stage 9K fill-quality bucket synthesis runner and produces artifacts under `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/`.
- The runner uses existing artifacts only: Stage 5 execution labels, Stage 5 fill markouts, Stage 5C quote-anchor safety diagnostics, live audit quote-update fields, and Stage 6 manifest presence. It does not run live, replay, parameter search, or strategy code.
- Clean-only evidence across nine current-format samples has `35,266` observed submit rows and `994` fills.
- Decision-visible trigger bucket results: `314` clean-only buckets, `0` `ready_for_policy_design`, `68` `needs_more_clean_fills`, and `246` `reject_quality_negative`.
- Shape A passive quality gate with inventory sizing has `0` candidate rows; Shape B reduce-side participation with spread-capture floor also has `0` candidate rows.
- Current recommendation is `needs_more_clean_fills` / `collect_or_refine_read_only_evidence_before_policy_design`, not policy implementation. This does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, or replay semantic changes.
- `0529T002` QA passed. The accepted next read-only task is `0529T005`, which should test rejection decomposition and decision-visible bucket coarsening before any policy design or implementation.

## 0529T001 Findings

- `0529T001` shifts the Binance maker next-policy direction to fill-quality-first design. The immediate next task should be a read-only fill-quality bucket synthesis runner, not strategy implementation.
- `0528T001` rejected the fixed inventory-aware quote placement skeleton because request buckets increased fills but worsened quality: request 5s markout `-85.21` ticks versus no-change `-70.20` ticks, and request spread capture `6.85` ticks versus no-change `16.78` ticks.
- `0526T004` remains a negative constraint: the current `min_move_quote_age_churn_guard` projected-suppression grid produced `0` promising parameter sets, so it should not be the near-term main route.
- Stage 9I decomposition shows a tradeoff rather than a ready policy: `request_side_priority / allow_touch` has high fill rate (`0.0768`) but weak spread capture (`1.02` ticks), while `request_quote_adjustment / prefer_one_tick_tight` has better spread capture (`30.77` ticks) but lower fill rate (`0.0124`) and still adverse markout.
- Next policy candidates are design-only: passive quality gate with inventory sizing, and reduce-side participation gate with spread-capture floor. Both require bucket-level quality validation before any strategy implementation.
- `0529T001` does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, or more compact-audit work unless a regression appears.

## 0528T002 Findings

- `0528T002` implements the narrow `0527T001` recommendation: formal compact replay lifecycle audit export for Stage 6 input.
- Contract path: audit replay should write `out/backtest_audit_replay/audit_bt_audit_replay.compact_lifecycle.csv`; Stage 6 now prefers this compact artifact and falls back to the legacy CSV only when compact is absent.
- Compact semantics are intentionally limited: preserve decision rows and non-terminal lifecycle rows, while de-duplicating terminal lifecycle rows by first meaningful `event_type + order_id` fact.
- Bounded verification on the preserved `0526T008` full replay audit prefix scanned `250,000` rows, observed `227,080` `cancel_ack` rows, wrote `23,345` compact rows, and skipped `226,655` duplicate terminal rows.
- Stage 6 was validated in a task-scoped run dir against `audit_bt_audit_replay.compact_lifecycle.csv` and completed with `decision_state=methodology_valid_single_sample`; row counts matched the prior accepted lifecycle-min run shape (`live_submit_orders=11089`, `replay_submit_orders=11084`, `matched_submit_orders=11084`).
- This is an output/input scaling fix only. It does not improve or change live/replay fill/cancel semantics, queue/touch behavior, strategy behavior, live behavior, parameters, guards, default-on state, or promotion readiness.

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
- `0515T003` passed QA. It completed the narrow replay lifecycle repair and reduced the core same-sample mismatch materially, but residual queue/touch cases still need conservative handling.
- `0515T004` passed QA. It is now the accepted fact source for residual-case follow-up.
- `0516T001` passed QA. It classifies `4948` as `queue_ahead_depth_can_absorb_observed_trades`, but not as enough evidence for repair implementation.
- `0516T002` passed QA. It shows queue-ahead proxy no-fill pattern repeats, while replay-fill false-positive repeatability remains single-case (`4948`).
- `0518T001` passed QA as repair-design-only. It designs a future conservative queue proxy gate but explicitly does not authorize implementation.
- `0518T002` passed QA. It recommends fast BBO/bookTicker as the primary hard quote anchor, depth BBO as guarded fallback / consistency check, and top5 as pricing/risk/diagnostic context rather than the final hard post-only anchor. `0518T003` passed QA as a read-only diagnostic.
- `0518T004` has been created as a narrow Step 5C default-off / diagnostic-first quote-anchor safety task. It is not a source-level drift repair, not top5 hard-anchor promotion, not generic quote-control redesign, and not live promotion.
- `0518T004` passed QA. It keeps default behavior disabled, adds a reusable safety helper, and generates Stage 5C diagnostic counters with post-clamp risk `0` on `5-13-day-control-30min`.
- `0519T001` and `0519T002` have been created to close Step 6. `0519T001` is the read-only final lifecycle calibration rerun after accepted repairs; `0519T002` is the planning-only closure decision after `0519T001` QA.
- `0519T001` passed QA. Aggregate replay/live lifecycle is no longer `diagnostic_only_gap_too_large`; the rerun decision state is `requires_more_current_format_samples`, with one remaining `4948` residual and no repair authorization.
- `0519T002` passed QA. It closes Step 6 for roadmap progression, but not for promotion, live readiness, exact queue proof, or generalized queue/touch repair.
- `0519T003` passed QA. It closes Step 7 as a design-only inventory / execution model contract and does not implement strategy behavior, run experiments, start live, or authorize promotion.
- `0519T004` passed QA. It constrains quote-update mechanics, API/churn hygiene, stale/bad-price handling and post-only protection before any Step 7 controls are implemented.
- `0519T005` passed QA. Conclusion: `default_off_helper_candidate`; direct Step 9 remains blocked until a helper / instrumentation boundary is accepted.
- `0519T006` passed QA as Step 8C default-off quote-update helper / instrumentation implementation. It preserves default behavior.
- `0519T007` passed QA as Step 9A default-off quote-adjustment replay experiment design-only. It did not implement runner, run replay, start live, default-enable behavior, or make promotion claims.
- `0519T008` passed QA as Step 9B default-off quote-adjustment offline replay runner implementation. It validates runner / artifact mechanics on `5-13-day-control-30min` and correctly classifies that old sample as `needs_more_instrumentation`; no live, default-on, sample expansion, production behavior change, or promotion is authorized.
- `0519T009` passed QA. It collected `5-19-day-control-30min` as a current-format no-rule / default-off 30min control sample, verified the 15 T006 quote-update audit fields, and reran T008. It does not authorize candidate promotion or live readiness claims.
- `0519T010` passed QA as Step 9C planning-only. It defines multi-sample scenario coverage, replay validation method, cross-regime stability criteria, and promotion/live preconditions before any sample expansion or replay sweep.
- `0519T011` completed business-thread execution and passed QA. It collected 3 separated 30min current-format no-rule/default-off samples whose run ids begin `5-19-night-active`; it does not perform final multi-sample validation or authorize promotion.
- `0519T011` repaired and verified the `5-19-night-active-30min-b` second raw gzip after collection overran because the control session died. The original incomplete gzip is preserved as remote `.gz.corrupt`; local accepted artifacts use a regenerated 30min raw slice, and local/remote/archive gzip checks passed.
- `0519T011` reached Step 9C numeric research-comparison mass when combined with `5-19-day-control-30min`: about `123m01s`, `10005` submits, and `250` fills. The caveat below remains relevant for future read-only multi-sample validation.
- `0519T011` sample-quality caveat: `5-19-night-active-30min-a` has `first_valid_update_aligned=false`, `gap_crossed_join_count=28062`, and Step 5C missing anchor rows `28062`. If strict market-view quality is required on every sample, collect a replacement sample before read-only multi-sample validation.
- `0520T001` has been created for Step 9C read-only multi-sample validation. Total controller accepted `5-19-night-active-30min-a` only as a caveated research-comparison input, so T001 must also report clean-only sensitivity excluding that sample and must not claim strict market-view quality from it.
- `0520T001` passed QA. Accepted-set reaches Step 9C research-comparison mass (`123.02` min, `10005` submits, `253` fills), while clean-only sensitivity is under threshold (`92.92` min, `8721` submits, `232` fills).
- `0520T001` found no `ready_for_tiny_live_design` candidate. `fair_reservation_shift_edge_25` and `stale_latency_no_fresh_add` are rejected; `spread_widening_stale_latency` needs runner/artifact work; `inventory_reservation_shift_band`, `size_reduction_or_add_side_suppression_pressure`, `min_move_quote_age_churn_guard`, and `post_only_safety_interaction` remain `keep_for_research`.
- `0520T002` has been created as the next narrow runner/artifact hardening task. It should make the current 8 Step 9 families decisionable across existing current-format samples before any Step 10 tiny-live design discussion.
- `0520T002` QA passed. The current 8 Step 9 families now have explicit bucket verdicts, including a guard-suppressed verdict for `spread_widening_stale_latency`.
- `0521T001` collected `5-21-day-control-60min` successfully, then completed normal replay, audit replay, and archive. The sample kept no-rule / default-off behavior, no live promotion, no default-on, no strategy change, and no sample expansion.
- `0521T002` completed the read-only Step 9C candidate x scenario bucket multi-sample determination over five current-format samples and found no `ready_for_tiny_live_design` candidate.
- `0521T002` QA passed. The accepted next direction is to refine `keep_for_research` candidates by finer scenario buckets before any parameter-search or Step 10 tiny-live-design task.
- `0525T001` passed QA for Step 9D fine-bucket refinement. It determines whether `min_move_quote_age_churn_guard`, `inventory_reservation_shift_band`, and `size_reduction_or_add_side_suppression_pressure` contain stable promising buckets, reject buckets, or sample/fill gaps before any parameter-search task.
- `0525T001` completed business execution. `min_move_quote_age_churn_guard` is the strongest near-term parameter-sweep seed (`20` stable promising buckets, `8` parameter-sweep seed buckets, `0` reject buckets). `inventory_reservation_shift_band` has no parameter-sweep seed and `17` reject buckets, so it should not be a near-term search主线. `size_reduction_or_add_side_suppression_pressure` has only two narrow seed buckets (`one_tick_tight`, `volatility_high`) and `12` reject buckets, so it should not be global. Global tiny-live fill gap remains `160` fills (`340 / 500`).
- `0526T001` has been created as a targeted active current-format no-rule/default-off collection task. It should prioritize natural fills and `min_move_quote_age_churn_guard` seed regimes: young quote churn, API/churn normal plus churn pressure, stale latency medium/high, one-tick tight spread, medium volatility, and large inventory skew / low inventory score. It explicitly must not relax guards, target `inventory_reservation_shift_band`, enable candidates, or make promotion claims.
- `0526T001` QA passed after completing only the first sample because the user paused the second and third collection windows. `5-26-active-minmove-control-30min-a` is clean current-format no-rule/default-off: maker acceptance passed, sidecar/join has future/gap/missing `0/0/0`, top5 tick/qty match `0.9847/0.9711`, Stage 5 has `2526` submits and `46` fills, and Stage 6 live/replay fill/cancel counts match. The 5+1 aggregate keeps `min_move_quote_age_churn_guard` as the main seed line, but fill gap only shrinks from `160` to about `114`, so it still does not support `ready_for_tiny_live_design`.
- `0526T002` has been created for one 1H targeted active current-format no-rule/default-off control sample. It should sync the task commit to `awsserver1`, run from a task-scoped clean worktree, change only run id/output paths and stopper duration to `3600s`, and leave quote/risk/guard strategy parameters unchanged. The purpose is to reduce the remaining `~114` natural-fill gap while preserving the current safety boundary.
- `0526T002` completed business execution and is waiting for QA. `5-26-active-minmove-control-60min-a` added `5095` submits and `164` natural fills, with audit replay action/reject/throttle alignment `1.0`, Stage 6 live/replay fills `164/163`, and post-only risk after Step 5C recheck `0`. The sample is structurally clean for sidecar joins (`future/gap/missing=0/0/0`, top5 tick/qty `0.8930/0.8486`), but market-view strict acceptance is caveated because `top5_join_age_ms_p99=69.9977ms > 50ms`. Counting this caveated sample gives about `550` total fills, but strict clean-only fill mass remains about `386/500`; therefore it supports parameter-sweep design evidence but not `ready_for_tiny_live_design`.
- The 5+1+1 Step 9D aggregate after `0526T002` keeps `min_move_quote_age_churn_guard` as the only credible near-term parameter-sweep main line, but narrows decision-time-visible seed buckets to `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`. `inventory_reservation_shift_band` and `size_reduction_or_add_side_suppression_pressure` still have no parameter-sweep seed and should not be near-term main lines.
- `0526T002` QA passed. QA accepts `5-26-active-minmove-control-60min-a` only as caveated evidence, not as strict-clean market-view evidence or tiny-live readiness proof. The next reasonable task is a narrow `min_move_quote_age_churn_guard` parameter-sweep design around `large_skew_or_low_score` and `stale_latency_medium`, unless total controller chooses to prioritize another strict-clean active sample to satisfy the clean-only `500` fill gate first.
- `0526T003` passed QA as a design-only task for the narrow `min_move_quote_age_churn_guard` parameter-sweep contract. It is limited to the seed buckets `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`, designs the later sweep to use the current local `amdserver` resources (`32` CPU cores, about `60G` memory) through deterministic shard/reducer parallelism, and does not authorize runner implementation, sweep execution, live, tiny-live, default-on, guard relaxation, or promotion.
- `0526T004` passed QA. It implemented the read-only narrow `min_move_quote_age_churn_guard` sweep over seven current-format samples and found `0` `sweep_seed_promising`, `80` `reject`, and `676` `not_decisionable` parameter evaluations. This exact projected-suppression grid does not support live, tiny-live, default-on, guard relaxation, production behavior changes, or `ready_for_tiny_live_design`.
- Controller interpretation after `0526T004`: do not prioritize more filled-order collection specifically for the current `min_move_quote_age_churn_guard` projected-suppression line. The result did not produce `stable_but_low_fill` or other strong "collect more fills" evidence; it produced `0` promising, `80` reject, and `676` not-decisionable evaluations. The `inventory_only` seed produced the hard negative signal, while `stale_latency_only` and intersection were mostly under-filled. Therefore this guard should be downgraded to execution hygiene / instrumentation unless a new guard shape is proposed. Future sample collection should serve broader maker-edge evaluation, not continue this exact min-move/churn grid by default. The next higher-value research direction is fair-price / reservation / inventory / quote-distance / size-side logic that can directly improve spread capture, adverse-selection control, and inventory recovery.
- `0526T005` passed QA. The low-cost read-only triage found all five maker-edge families have enough clean evidence for follow-up, ranked by score as inventory, quote-distance, size-side, fair-price, then reservation. Inventory / quote-distance / size-side have the strongest immediate separation; fair-price and reservation remain promising but are highly similar under the current Stage 5 label view. This supports moving away from the min-move/churn guard grid and toward a focused maker-edge design task that combines inventory state, quote distance, side/size, and fair/reservation signal design. It still does not authorize live, default-on, or production behavior changes.
- `0526T006` has been formally dispatched as the current focused maker-edge design task. It should produce one inventory-aware quote-placement design, not five separate tracks: fair/reservation are pricing context, quote-distance and size/side are execution-control dimensions, and inventory state is the main organizing axis. After `0526T004` QA, the task contract was tightened to require a concrete candidate policy skeleton and a later read-only runner input/output contract, while still forbidding implementation, replay sweep, live, default-on, and promotion.
- `0526T006` passed QA. The accepted focused design is `inventory_aware_quote_placement_request`: inventory bucket controls side preference and add/reduce-side size pressure, fair/reservation edge controls whether a side is favorable enough to keep/place, quote-distance sets the touch/one-tick/step-back participation frontier, and post-only/stale/latency/anchor fields remain safety context. It supports a later read-only/default-off runner implementation task only; it does not authorize strategy implementation, parameter search, live/default-on, guard relaxation, or promotion.
- `0528T001` has been created to implement that accepted design as a read-only/default-off offline runner. The required output directory is `local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001/`; required artifacts include candidate decision rows, bucket metrics, clean-only stability, caveated sensitivity, participation/fill-loss, inventory recovery quality, quote mechanics safety, and recommendation markdown. This task still does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0528T001` passed QA. The runner output is `local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001/`; all nine current-format samples were usable with audit and Stage 5C join coverage `1.0`. Clean-only evidence has `35,266` rows, `994` fills, `14,363` candidate request rows, and `528` request fills, so fill mass is sufficient; however request buckets are worse than no-change on 5s markout (`-85.21` vs `-70.20` ticks) and spread capture (`6.85` vs `16.78` ticks). Clean-only and caveated sensitivity both recommend `reject`. Theory summary: the skeleton found more fills, but they were worse fills because it turned inventory state into quote-placement requests before proving those request buckets had positive fill quality. The result rejects this fixed policy skeleton as-is and does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0526T007` has been created and directly dispatched per controller request as a 180min current-format no-rule/default-off live control collection task. It is intended to add longer-window natural fills and maker-edge diagnostic regimes, not to relax guards, enable candidates, run parameter sweep, or authorize live/default-on/promotion.
- `0526T007` started remote collection on `awsserver1` using commit `43fb586`, worktree `/home/admin/hft_live/worktrees/0526T007-makeredge-180min`, and run id `5-26-active-makeredge-control-180min-a`. Remote preflight passed with `git.dirty=false`, `compatibility.passed=true`, and `audit_field_count=159`; early audit header check found no missing T006 fields. The run has now stopped, the data has been pulled locally, audit replay and base maker acceptance passed with `action/planned/reject/throttle = 1.0`, and the archive `local_live_analysis/archive/5-26-active-makeredge-control-180min-a.tar.gz` was written with sha256 `f4a2091de3f48e798736e2d675b40475cbbc0eb9db8778a378d31929d3152935`.
- `0526T007` post-processing is now complete on `5-26-active-makeredge-control-180min-a`: `t009_fixed_sidecar` is present with `data.npz`, `raw_provenance.csv`, `raw_to_npz_mapping.csv`, `top5_sidecar.csv`, `joined_decisions.csv`, and join metrics. Stage 5 labels were generated with `7016` submit orders and `148` fills, Stage 5C safety diagnostics reported `17004` bid-clamped rows, `6434` ask-clamped rows, and `0` post-only risk after recheck, Stage 6 calibration is `methodology_valid_single_sample` with matched submit coverage `7016/7016` and live/replay filled orders `148/156`, Step 9B classification is `promising_but_single_sample`, and Step 9D fine-bucket refinement found `0` stable promising buckets and `0` parameter-sweep seed buckets on this single sample. This sample is structurally usable for maker-edge diagnostics, but it still does not authorize live/default-on/promotion or a parameter-search claim.
- `0526T007` QA passed. Accepted scope is data collection plus derived diagnostic artifacts only; it remains no-rule / default-off control evidence and does not change the current conclusion that a multi-sample parameter-search-ready maker strategy direction has not yet been established.
- `0526T008` has been created and directly dispatched per user override as an immediate 30min current-format no-rule/default-off live control collection task based on `0526T002`, with run id `5-26-active-minmove-control-30min-b`. It intentionally does not wait for unfinished `0526T006` / `0526T007`, but it does not authorize candidate enablement, guard relaxation, strategy changes, parameter sweep, tiny live, default-on, or promotion.
- `0526T008` QA passed. `5-26-active-minmove-control-30min-b` completed collection, raw recovery, archive, maker acceptance, T009 sidecar/join, Stage 5, Step 5C, Stage 6, Step 9B, and Step 9D; maker and market-view gates passed, T009 future/gap/missing is `0/0/0`, Stage 5 has `11089` submit orders and `471` fills, Stage 6 is `methodology_valid_single_sample`, Step 9B is `promising_but_single_sample`, and Step 9D has `0` stable promising buckets. It remains no-rule/default-off control data only. The full replay audit repeated `cancel_ack` lifecycle rows and inflated to about 21GB, so Stage 6 used a documented lifecycle-minimized input workaround; future repair is tracked by `0527T001`.
- `0527T001` has been refined into a planning/diagnosis-only task for the `0526T008` replay audit bloat. It should not repair code yet; it should determine whether repeated `cancel_ack` rows originate from audit export repetition, replay lifecycle state repetition, or Stage 6 input assumptions, identify the duplicate key, decide whether repeated rows are redundant for Stage 6 labels, and recommend the smallest later implementation boundary.
- `0527T001` passed QA. The `0526T008` full replay audit has `32,893,719` `cancel_ack` rows but only `10,380` unique cancel-ack `order_id`s; top repeated orders each appear over `325k` times. The source is primarily audit replay lifecycle export / order-state tracking repetition: `OrderLifecycleTracker.observe()` retains current snapshots from `hbt.orders(0)`, and audit replay overlay writes both regular cancel_ack rows and forced live terminal constraint rows as decisions advance. Repeated rows are semantically redundant for Stage 6 once the first terminal fact per order is kept. QA recommends two next steps: create a narrow compact lifecycle replay audit export / writer-side terminal-order de-dup implementation task, and keep full forensic audit optional/off the default Stage 6 path while avoiding strategy/live/promotion changes.
- `0528T002` has been created and narrowed to implement the first `0527T001` QA recommendation through one route: produce a formal compact replay lifecycle audit artifact for Stage 6 input, with terminal lifecycle rows de-duplicated inside the compact export by Stage 6 semantics. Full forensic audit remains optional evidence only; strategy/live/fill-cancel semantic/promotion changes remain out of scope.
- Step 9 narrow `min_move_quote_age_churn_guard` parameter-sweep design can use these facts without re-discovery:
  - Primary decision sources are `local_live_analysis/stage9d_candidate_bucket_refinement_0526T002_aggregate/fine_bucket_stability_summary.csv`, `fine_bucket_metrics.csv`, and `candidate_bucket_recommendations.md`.
  - Current current-format sample set is `5-19-day-control-30min`, `5-19-night-active-30min-a`, `5-19-night-active-30min-b`, `5-19-night-active-30min-c`, `5-21-day-control-60min`, `5-26-active-minmove-control-30min-a`, `5-26-active-minmove-control-60min-a`, and `5-26-active-makeredge-control-180min-a`.
  - Caveated samples are `5-19-night-active-30min-a` and `5-26-active-minmove-control-60min-a`; use them for sensitivity / broader evidence only, not strict-clean promotion proof.
  - `5-26-active-makeredge-control-180min-a` is a newly added current-format no-rule/default-off control sample from `0526T007`: effective live audit duration is about `65.67` minutes, audit replay and base maker acceptance passed with `action/planned/reject/throttle = 1.0`, archive sha256 is `f4a2091de3f48e798736e2d675b40475cbbc0eb9db8778a378d31929d3152935`, and Stage 5 / Step 5C / Stage 6 / Step 9 derived artifacts still need to be generated before using it in aggregate candidate decisions.
  - Narrow sweep scope should be `min_move_quote_age_churn_guard` only, seeded only by decision-time-visible buckets `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`.
  - Step 9D aggregate is the primary decision evidence. Per-sample Step 9B outputs, Stage 5 execution labels, Stage 6 calibration, and Step 5C anchor/post-only safety outputs are supporting mechanism / risk diagnostics.
  - Current evidence does not support Step 10 tiny live, default-on, guard relaxation, or `ready_for_tiny_live_design`. Strict clean-only fill mass remains about `386/500`.
- `0521T002` accepted-set mass is `183.03` min, `15919` submits, `340` fills; clean-only mass is `152.93` min, `14635` submits, `319` fills.
- `0521T002` verdicts are: `baseline_control` keep_for_research, `fair_reservation_shift_edge_25` reject, `inventory_reservation_shift_band` keep_for_research, `min_move_quote_age_churn_guard` keep_for_research, `post_only_safety_interaction` reject, `size_reduction_or_add_side_suppression_pressure` keep_for_research, `spread_widening_stale_latency` keep_for_research but guard-suppressed, and `stale_latency_no_fresh_add` reject.
- `0521T002` became decisionable after restoring the missing derived chain for `5-21-day-control-60min`: `t009_fixed_sidecar/joined_decisions.csv`, Stage 5 labels, Stage 5C safety diagnostics, Stage 6 calibration, and the Stage 8B planning-decision placeholder.

## 0519T006 Task Boundary

- T006 exists to implement the next boundary recommended by T005, not to start Step 9.
- Scope is default-off helper / instrumentation only:
  - quote-update intent/action/reason
  - min-move, quote-age, join/anchor-age, latency-bucket fields
  - throttle/token/cancel-readd state fields
  - reject/throttle/drop cause
  - post-only pre/post-check fields
  - inventory request id placeholder
- Required invariant: existing action path, throttle/API/latency suppression, quote placement, cancel/submit behavior, Step 5C default-off status, and Step 7 design-only status remain unchanged by default.
- T006 does not authorize replay sweep, live, default-on behavior, Step 5C promotion, inventory-control implementation, or Step 9 promotion.

## 0519T006 Findings

- Step 8C now has a shared quote-update audit helper in `strategy_core.py`, wired from both `backtest_tick_mm.py` and `live_tick_mm.py`.
- `0519T006` QA passed on 2026-05-19 15:37 CST.
- New audit fields are present in `AUDIT_FIELDS`:
  - `quote_update_intent`
  - `quote_update_action`
  - `quote_update_reason`
  - `min_move_passed`
  - `quote_age_ms`
  - `join_age_ms`
  - `anchor_age_ms`
  - `latency_bucket`
  - `throttle_state`
  - `token_bucket_state`
  - `cancel_readd_bucket`
  - `reject_throttle_drop_cause`
  - `post_only_pre_check`
  - `post_only_post_check`
  - `inventory_request_id`
- Default behavior remains unchanged: the helper records snapshots and audit fields after existing latency/throttle/API/post-only decisions are formed; it does not choose, suppress, submit, cancel, reprice, or promote quotes.
- Placeholder / diagnostic-only boundary:
  - `inventory_request_id` is a passive placeholder.
  - `queue`, cancel-readd, quote/join/anchor age, latency bucket, token/throttle state, and post-only pre/post checks are diagnostic/proxy fields only.
  - No Step 9 replay sweep, live run, default-on behavior, Step 5C promotion, or Step 7 inventory-control implementation was done.

## 0519T007 Task Boundary

- T007 is Step 9A design-only.
- It should define candidate matrix, decision-time-visible inputs, metrics, artifacts, Step 9B implementation boundary, and non-goals for default-off quote-adjustment offline replay.
- It should explicitly carry forward:
  - Step 5C post-only safety remains default-off / diagnostic-first unless separately enabled
  - Step 6 lifecycle closure is enough for roadmap progression but not exact queue/live promotion proof
  - Step 7 inventory controls are design-only and can only express future requests through shared quote-update fields
  - Step 8 / T006 helper fields are instrumentation and explanation fields, not strategy control flow
- T007 does not authorize runner implementation, replay sweep, live, default-on behavior, sample expansion, Step 5C promotion, inventory-control implementation, or promotion claims.

## 0519T007 Findings

- Step 9A is ready for QA as a design contract, not as an implementation.
- `0519T007` QA passed on 2026-05-19 15:57 CST.
- Candidate matrix is grouped into eight families:
  - baseline/control no-change replay validation
  - fair / reservation shift
  - inventory reservation shift / recovery-side preference request
  - spread widening
  - size reduction / add-side suppression
  - stale or latency no-fresh-add regime
  - min-move / quote-age / API-churn guard
  - Step 5C post-only safety interaction
- Each candidate family is constrained to decision-time-visible inputs. Future markout, fill outcome, audit replay overlays, exact queue claims, and `4948`-specific logic remain disallowed as decision inputs.
- T006 fields become the Step 9 explanation layer:
  - `quote_update_intent`, `quote_update_action`, `quote_update_reason`
  - `min_move_passed`, `quote_age_ms`, `join_age_ms`, `anchor_age_ms`, `latency_bucket`
  - `throttle_state`, `token_bucket_state`, `cancel_readd_bucket`, `reject_throttle_drop_cause`
  - `post_only_pre_check`, `post_only_post_check`, `inventory_request_id`
- Step 9B should be a minimal default-off offline runner task if QA accepts T007:
  - validate runner mechanics on `5-13-day-control-30min`
  - emit candidate matrix, per-candidate metrics, action-path/audit coverage, fill-quality, inventory-cycle, API/churn and post-only safety artifacts
  - classify results as `no_effect`, `worse_due_to_churn_or_fill_quality`, `promising_but_single_sample`, or `blocked_by_replay_or_market_view`
  - keep all candidates default-off and offline-only
- Sample expansion should come after the runner and candidate methodology are accepted, unless QA finds a design blocker that requires data first.

## 0519T008 Task Boundary

- T008 is Step 9B default-off offline runner implementation.
- It should implement a local runner and focused tests, then validate runner / metrics / artifacts on `5-13-day-control-30min`.
- Required output directory:
  - `local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008/`
- Required candidate families:
  - `baseline_control`
  - `fair_reservation_shift`
  - `inventory_reservation_shift`
  - `spread_widening`
  - `size_reduction_or_add_side_suppression`
  - `stale_latency_no_fresh_add`
  - `min_move_quote_age_churn_guard`
  - `post_only_safety_interaction`
- Required classification:
  - `no_effect`
  - `worse_due_to_churn_or_fill_quality`
  - `promising_but_single_sample`
  - `blocked_by_replay_or_market_view`
  - `needs_more_instrumentation`
- T008 does not authorize live, default-on behavior, production strategy behavior changes, sample expansion, Step 5C promotion, inventory-control implementation, queue/touch repair, or promotion claims.

## 0519T008 Findings

- Step 9B runner exists at `examples/binance_tick_mm/quote_adjustment_replay.py` with focused tests in `examples/binance_tick_mm/test_quote_adjustment_replay.py`.
- Output directory:
  - `local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008/`
- All required artifacts were generated:
  - `run_manifest.json`
  - `candidate_matrix.csv`
  - `candidate_matrix.json`
  - `candidate_summary.json`
  - `candidate_metrics.csv`
  - `fill_quality_by_candidate.csv`
  - `inventory_cycle_metrics.csv`
  - `api_churn_metrics.csv`
  - `post_only_safety_metrics.csv`
  - `action_path_coverage.csv`
  - `audit_field_coverage.csv`
  - `candidate_decision_samples.csv`
  - `acceptance_decision.md`
- Classification is `needs_more_instrumentation`.
- Reason:
  - the existing `5-13-day-control-30min` audit was collected before T006 and is missing all 15 T006 quote-update audit fields
  - the runner therefore used proxy fields to validate mechanics and metrics rather than treating results as candidate performance proof
- Key diagnostic counts:
  - decision rows `47499`
  - submit orders `2516`
  - candidate families `8`
  - missing T006 fields `15`
  - baseline fill rate about `0.021065`
  - baseline fill-after-cancel rate about `0.006359`
- Interpretation:
  - T008 validates the runner / metrics / artifact path.
  - It does not authorize live, default-on, sample expansion, production behavior change, or promotion.
  - Before promotion-style claims or true candidate evaluation, use a sample/replay that contains T006 quote-update fields, or explicitly accept a proxy-only diagnostic boundary in a later task.

## 0519T009 Task Boundary

- T009 is a data-collection / audit-rerun task, not a strategy task.
- Dataset name is fixed as `5-19-day-control-30min`.
- The collection must remain no-rule / default-off control.
- The early audit header check must confirm all 15 T006 fields:
  - `quote_update_intent`
  - `quote_update_action`
  - `quote_update_reason`
  - `min_move_passed`
  - `quote_age_ms`
  - `join_age_ms`
  - `anchor_age_ms`
  - `latency_bucket`
  - `throttle_state`
  - `token_bucket_state`
  - `cancel_readd_bucket`
  - `reject_throttle_drop_cause`
  - `post_only_pre_check`
  - `post_only_post_check`
  - `inventory_request_id`
- If any of these fields are missing, stop the run and mark the task blocked rather than producing another proxy-only 30min sample.
- After collection, rerun `align_live_run.py`, `maker_acceptance.py`, and `quote_adjustment_replay.py` on the new dataset. A single 30min sample can remove the instrumentation blocker, but it still cannot prove promotion or live readiness.

## 0519T009 Findings

- Dataset: `5-19-day-control-30min`.
- Deployed commit: `2d0cae2`.
- Preflight passed:
  - `git.dirty=false`
  - `compatibility.passed=true`
  - audit field count `159`
- Run markers:
  - start marker UTC `2026-05-19T09:13:58Z`
  - stop marker UTC `2026-05-19T09:46:46Z`
  - stop marker exit code `0`
- Live audit:
  - rows `122124`
  - fields `159`
  - all 15 T006 quote-update fields are present
- Raw collection note:
  - collector gzip lacked a footer after tmux session shutdown
  - original remote file was preserved as `btcusdt_20260519.gz.corrupt`
  - complete raw lines were recovered and recompressed to a valid `btcusdt_20260519.gz` for local replay
- Maker acceptance:
  - passed `true`
  - hard failures `[]`
  - common rows `96340`
  - all 21 checks passed
- T009 sidecar / join:
  - first valid update aligned `true`
  - depth `pu` mismatch `0`
  - decision join coverage `1.0`
  - future join `0`
  - gap-crossed join `0`
- Stage 5 labels on the new dataset:
  - submit orders `4098`
  - filled orders `110`
  - fill-after-cancel orders `53`
  - fast-cancel churn rate about `0.86164`
- Step 5C diagnostics on the new dataset:
  - decision rows `96341`
  - bookTicker anchor rows `79449`
  - guarded depth fallback rows `16892`
  - post-only risk after re-check rows `0`
- T008 rerun on the new dataset:
  - `missing_t006_field_count=0`
  - classification `promising_but_single_sample`
  - reason: active candidates have nonzero coverage, but only one current-format sample is available
- Interpretation:
  - T009 removes the old `needs_more_instrumentation` blocker for one current-format sample.
  - The result is still single-sample default-off offline diagnostic evidence, not live readiness, not promotion, and not generalized profitability proof.

## 0519T010 Findings

- Step 9C exists to answer the question: can default-off quote-adjustment candidates improve maker execution quality across regimes, not just look good in one sample.
- Accepted input roles:
  - `0519T008` proves `quote_adjustment_replay.py` runner / metrics / artifact mechanics.
  - `0519T009` proves one current-format T006 control sample is usable and removes the one-sample instrumentation blocker.
- Primary blocker is now data scenario coverage:
  - single current-format sample evidence cannot establish stable maker strategy behavior
  - sample/event mass must cover multiple volatility, spread, trade intensity, latency/stale, API/churn, inventory, post-only safety, cancel-fill, and market-view quality regimes
- Minimum research comparison target:
  - at least `4` current-format samples including `5-19-day-control-30min`
  - at least `120` minutes aggregate duration
  - at least `10000` submit orders aggregate
  - at least `250` filled orders aggregate
- Minimum before a later `ready-for-tiny-live-design` classification:
  - at least `5` current-format samples
  - at least `180` minutes aggregate duration
  - at least `15000` submit orders aggregate
  - at least `500` filled orders aggregate
  - at least `2` distinct non-calm regimes
- Hard gates for every accepted validation sample:
  - T006 missing fields `0`
  - maker acceptance passed
  - action/planned/reject/throttle gates passed
  - working semantic/blocking mismatch `0`
  - strict replay lag passed
  - sidecar/join quality acceptable, with future/gap-crossed join `0`
  - post-only crossed-risk after re-check `0`
  - archive/raw integrity documented
- Candidate interpretation rules:
  - `reject` if a candidate fails hard gates, lacks coverage, worsens fill quality/adverse markout/cancel-fill/API-churn in multiple samples, or uses forbidden inputs
  - `keep_for_research` if it has coverage and favorable regimes but insufficient sample/event mass or proxy-only evidence
  - `ready_for_tiny_live_design` only after multi-sample hard gates, stable execution-quality behavior, no catastrophic worst-sample result, and QA; it still authorizes only a separate live-design planning task
- Recommended next step after T010 QA:
  - collect more current-format no-rule/default-off samples first, rather than modifying the runner
  - then run a read-only multi-sample validation using the existing runner
  - modify the runner only if the accepted plan cannot be executed with current artifacts
- T010 remains planning-only:
  - no code change
  - no replay sweep
  - no live
  - no default-on
  - no promotion
  - no single-sample PnL acceptance

## 0519T005 Findings

- Step 8B conclusion is `default_off_helper_candidate`, not `full_default_off_replay_candidate`.
- Current sample shows material quote-update pressure and suppression:
  - decision rows `47499`
  - planned submit decision rows `8971`
  - actual submit decision rows `2256`
  - planned/action mismatch rows `7186`
  - latency guard rows `16528`
  - quote throttle rows `5996`
  - api interval guard rows `1190`
- Submit-level churn is already high:
  - submit orders with labels `2516`
  - fast-cancel churn rows `1955`
  - fast-cancel churn rate about `0.777027`
- Step 5C safety diagnostics are useful but still default-off / diagnostic-first:
  - bid clamped rows `1394`
  - ask clamped rows `2281`
  - stale anchor rows `65`
  - post-only risk after re-check rows `0`
- Observable now:
  - `action` / `planned_action` as actual vs suppressed quote-activity proxies
  - `reject_reason` / `throttle_reason` for latency, quote-throttle, and API interval gates
  - Stage 5C fast-anchor / guarded-fallback / anchor-age / clamp / suppress / recheck diagnostics
  - Stage 5 submit-level placement, latency, inventory, recent reject/throttle, fast-cancel churn, fill horizon, and fill-after-cancel labels
- Missing or proxy-only before implementation:
  - missing `quote_update_intent`
  - missing unified `quote_update_reason`
  - missing `token_bucket_state`
  - missing `inventory_request_id`
  - proxy-only `min_move_passed`, `quote_age_ms`, `cancel_readd_bucket`, `latency_bucket`
  - diagnostic-only `anchor_age_ms` and post-clamp `post_only_post_check`
- Recommended next boundary:
  - create a separate default-off helper / instrumentation task before Step 9
  - centralize quote-update intent/action/reason
  - record throttle/token/cancel-readd/post-only/inventory-request fields
  - preserve existing throttle/API/latency suppression semantics
  - keep default behavior unchanged
- T005 does not authorize strategy implementation, replay sweep, live, default-on, Step 5C promotion, or Step 9 promotion.

## 0519T004 Findings

- Step 8 should be completed as a design contract before implementation.
- Quote-update mechanics should be driven by observable safety and usefulness triggers:
  - bad-price ticks
  - minimum quote move
  - quote age
  - stale or missing anchor
  - join-age / latency regime
  - post-clamp post-only risk
  - inventory regime request from Step 7
- Preferred future action ordering:
  - hold quote when price is still useful and API/churn budget should be preserved
  - modify/replace in place if supported and safer than cancel+new
  - cancel+new only when quote is materially unsafe, stale, crossed-risky, inventory-worsening, or past bounded age
- GTX/post-only reject is an exchange backstop and diagnostic bucket, not normal control flow.
- Step 5C quote-anchor safety remains default-off / diagnostic-first unless a later implementation task explicitly changes that boundary.
- Step 7 inventory controls must express quote-change requests through shared update-intent fields and cannot bypass anti-churn, throttling, stale-anchor suppression, or post-only re-check.
- Anti-churn controls should include:
  - per-side min tick move
  - min quote age
  - max cancel/re-add rate
  - in-flight order guard
  - cancel-pending guard
  - recent reject/throttle cooldown
  - emergency stale/bad-price override
- API hygiene must explicitly model token bucket, request spacing, per-action budgets, cancellation-limit risk, reject/throttle/drop buckets, and degraded modes.
- Required future audit fields include quote update intent/action/reason, min-move pass flag, quote/anchor/join age, latency bucket, throttle/token state, cancel-readd bucket, reject/throttle/drop cause, post-only pre/post checks, and Step 7 inventory request id.
- Recommended next task after QA is not Step 9 yet. Open a narrow Step 8B read-only diagnostic / implementation-planning task over existing artifacts to quantify current churn/API/stale/bad-price regimes and decide whether implementation should be no-change, default-off helper, or full default-off replay candidate.

## 0519T003 Findings

- Step 7 should be completed first as a design contract, not as code implementation.
- Inventory objective:
  - keep normal exposure close to flat / one-order-quantity bands
  - make larger inventory a distinct recovery regime
  - reduce time spent in directional exposure instead of relying on symmetric quote churn
- Initial candidate controls are allowed only as future default-off designs:
  - reservation / fair shift by inventory band
  - spread widening on inventory-worsening side
  - add-side size reduction or add-side suppression when beyond one order quantity
  - recovery-side size preference when inventory is above target
  - volatility / fill-intensity driven AS-style spread and order amount
  - TTL / triple-barrier style exit handling only after pricing and lifecycle evidence exists
- Zero-crossing should be treated as an inventory-cycle boundary for diagnostics:
  - report cycle duration, max inventory excursion, recovery fills, markout while reducing inventory, and whether inventory crossed through zero cleanly
- Step 7 must use decision-time-visible inputs only:
  - current position / notional
  - target and working quote ticks
  - fair/reservation signals available at decision time
  - volatility / spread / top-of-book or top5 size-age proxies
  - latency / stale / join-age flags
  - live-safe lifecycle state such as in-flight, cancel-requested, and recent fill/cancel events
- Step 7 must not use future markout, audit replay overlays, exact queue claims, or `4948`-specific repair logic as live decision inputs.
- Required future audit fields before implementation:
  - inventory band
  - inventory cycle id
  - skew regime
  - quote-side suppression reason
  - size multiplier
  - spread multiplier
  - reservation shift
  - TTL / barrier state
  - recovery-mode marker
- Evidence gates before any default-off implementation or experiment:
  - replay acceptance and market-view gate remain clean
  - Step 6 lifecycle diagnostics remain within the accepted event-classification boundary
  - inventory-cycle metrics improve without hiding fill-quality or markout degradation
  - API/churn and post-only safety stay inside Step 5C/Step 8 boundaries
  - single-sample PnL is not enough for promotion or live
- Recommended next step after QA:
  - proceed to Step 8 design-only quote-update / API-limit hygiene before implementing Step 7 controls
  - use Step 8 to constrain whether Step 7 candidates can be expressed safely without blind cancel/re-add churn

## 0519T002 Findings

- Step 6 final state is `closed_for_roadmap_progression_requires_more_samples_for_promotion`.
- The old Stage 6 blocker is resolved enough to move forward:
  - `0519T001` QA confirmed the state improved from `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`
  - matched submit coverage is complete at `2516/2516`
  - price tick and qty equality are `2516/2516`
  - live/replay filled orders are close at `53/54`
  - live/replay fill-after-cancel orders are close at `16/15`
- Step 6 remains bounded:
  - it is an event-classification and lifecycle-proxy closure, not exact queue proof
  - timing magnitude gaps remain in time-to-fill and cancel-to-fill delay
  - one residual remains: `28940|sell` / order `4948`
  - queue/priority, opportunity cost, and realized PnL decomposition remain observed-only proxies
- `4948` / queue-ahead proxy residual stays parked:
  - no generalized queue/touch repair is authorized
  - future repair would require more current-format samples or repeated replay false-positive evidence
- More current-format samples are still needed before promotion-style conclusions, live micro tests, or generalized queue/touch repair.
- More samples do not need to block Step 7 / Step 8 design work.
- Recommended sequence after QA:
  - start Step 7 as a design-only inventory / execution model task
  - then start Step 8 as a design-only quote-update / API-limit hygiene task
  - only after those design boundaries are accepted, open Step 9 as a default-off offline replay experiment
  - do not treat Step 9 as live promotion, and do not rely on single-sample PnL

## 0519T001 Findings

- `0519T001` reran the existing read-only Stage 6 calibration on `5-13-day-control-30min` after the accepted replay lifecycle repairs.
- Decision state improved from the original Stage 6B `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`.
- Matched submit coverage remains complete:
  - live submit orders `2516`
  - replay submit orders `2516`
  - matched submit orders `2516`
  - matched price tick equality `2516/2516`
  - matched qty equality `2516/2516`
- Aggregate lifecycle is now close:
  - live filled orders `53`
  - replay filled orders `54`
  - live fill-after-cancel orders `16`
  - replay fill-after-cancel orders `15`
  - final-state filled gap `0.000397`
  - final-state canceled gap `0.000397`
  - fill-after-cancel-request rate gap `0.000397`
  - fast-cancel-churn gap `0.0`
- Fill horizon gaps are aligned enough on the matched universe:
  - `100ms` gap `0.001192`
  - `500ms` gap `0.000397`
  - `1000ms` gap `0.000397`
  - `5000ms` gap `0.000397`
- Remaining non-perfect timing differences are concentrated in timing magnitude, not event classification:
  - matched-any-filled time-to-fill mean gap about `400.07ms`
  - matched-both-filled time-to-fill mean gap about `364.54ms`
  - cancel-to-fill delay mean gap about `96.88ms` on both-observed rows
- Markout observability is close but still not identical at all horizons:
  - fill markout coverage gap at `500ms` is about `0.00159`
  - fill markout coverage gap at `5000ms` is about `0.00119`
  - mean markout ticks are identical at `500ms/1000ms/5000ms`, but `100ms` has about `3.55` ticks mean gap on a small observed subset.
- Residual diagnosis still reports one case:
  - `28940|sell` / order `4948`
  - class `residual_replay_fill_trigger_uncertain`
  - no nearby supportive trade evidence before live or replay anchor
  - this remains a queue-exposure / replay trigger uncertainty, not a repair authorization.
- Queue/priority, opportunity cost, and realized PnL decomposition remain observed-only proxies, not exact queue proof.
- T001 did not modify replay behavior, strategy behavior, live scripts, schema, or default-on behavior.
- `0519T001` has passed QA and unblocks `0519T002` as a planning-only Step 6 closure decision. It does not authorize Step 9 promotion or live readiness by itself.

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

## 0518T001 Findings

- `0518T001` stayed repair-design-only and did not modify replay, strategy, live collection, sample policy, or generated sample artifacts.
- The accepted `4948` evidence package for future design is:
  - submit_key `28940|sell`
  - order_id `4948`
  - live canceled / replay filled
  - same-price trade qty before replay fill `8.884`
  - submit visible qty `21.143`
  - replay-fill visible qty `15.633`
  - same-price qty / visible qty ratios `0.4202` and `0.5683`
  - order-at-touch share `1.0`
- The case is best interpreted as a queue-ahead proxy problem, not an unknown trigger:
  - market trades did hit the touch price
  - visible queue proxy was still large enough to absorb observed same-price trade qty
  - replay likely overstates fillability because it lacks exact queue-ahead / priority / exposure state
- The proposed future gate should be conservative and diagnostic-first:
  - apply only to replay-fill candidates on accepted market-view rows
  - use same-price trade qty / visible qty, touch share, visible qty decay, unexplained depth shrink, quote age, join age, and stale/gap guards
  - treat `same_price_trade_qty / visible_qty < 1.0` as suspicious, with `< 0.75` as stronger evidence, but not as exact no-fill proof
  - block decisions on stale/future/missing/gap-crossed joins
- Current implementation is intentionally deferred:
  - replay-fill false-positive repeatability remains single-case (`4948`)
  - exact queue position and order-id-level depletion are still missing
  - a future implementation must first be diagnostic-only / default-off and validated on more current-format samples or more replay false-positive cases

## 0518T002 Findings

- `0518T002` stayed design-only and did not modify quote placement, fair/reservation, strategy behavior, risk guards, live scripts, replay generation, or generated sample artifacts.
- The recommended Step 5A quote-anchor design is layered:
  - primary hard anchor: fast BBO/bookTicker-equivalent source
  - secondary check/fallback: depth BBO under strict freshness and accepted market-view quality
  - research/context source: top5 reconstructed BBO, top5 imbalance, top5 microprice, and top5 liquidity proxies
- Top5 role is explicitly limited:
  - pricing input for fair-price / reservation research
  - risk/context input for liquidity, imbalance, age, and queue-ahead proxy strata
  - not the current final hard post-only quote anchor because Stage 3 top5 tick/qty evidence is research-grade, not exact L2/queue proof
- Hard protection contract:
  - bid candidates should round down/floor to tick and then clamp to `<= anchor_best_bid_tick`
  - ask candidates should round up/ceil to tick and then clamp to `>= anchor_best_ask_tick`
  - post-clamp validity must be rechecked against the anchor BBO
  - stale/missing/gap-crossed anchors should suppress fresh add-side submits or re-add churn instead of relying on exchange rejects
- GTX/post-only remains the exchange backstop, but post-only reject, API reject, throttle, drop, and fast churn should be treated as evidence buckets for stale anchor, latency, rounding, source drift, or lifecycle uncertainty.
- Step 5B must quantify BBO source drift, quote-distance buckets, crossed/post-only-risk candidates, reject/throttle/churn, stale/join-age/latency regimes, fill/markout/spread-capture tradeoffs, current enforcement gaps, and a read-only rounding/clamp counterfactual before any default-off implementation task.
- Follow-up clarification before dispatching `0518T003`: the five Step 5A constraints are not all currently enforced by code. T003 must explicitly report which constraints are already backed by code/parameters/audit fields and which remain design gaps.

## 0518T003 Findings

- `0518T003` implemented a read-only quote-anchor / post-only diagnostic runner and generated all required artifacts under `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/`.
- The run stayed read-only: no quote placement, fair/reservation, risk guard, live script, replay lifecycle, or standard schema behavior changed.
- Dataset shape:
  - decision rows `47499`
  - submit label rows `2516`
  - bookTicker anchor available rows `47499`
  - top5 anchor available rows `47499`
  - join stale decision rows `432`
  - join missing / gap-crossed rows `0 / 0`
- BBO source drift:
  - audit_depth vs bookTicker mismatch rate is large: bid `0.3524495252531632`, ask `0.3531653297964168`
  - audit_depth vs bookTicker p99 abs drift is bid `142` ticks and ask `150` ticks
  - bookTicker vs top5_depth mismatch is much smaller: bid `0.002652687424998421`, ask `0.006589612412892903`
  - Interpretation: sidecar bookTicker/top5 depth BBO are close to each other, but live audit depth view and sidecar/as-of anchor view are not row-exact enough to claim existing fast-bookTicker hard-anchor implementation.
- Rounding/clamp counterfactual:
  - current path vs audit_depth post-round risk rows `0 / 47499`
  - current path vs bookTicker post-round risk rows `3675 / 47499`, rate `0.07737004989578727`
  - current path vs top5_depth post-round risk rows `3615 / 47499`, rate `0.07610686540769279`
  - T002 design path reduces all three anchor-source post-round risk counts to `0` in the read-only counterfactual
  - Interpretation: current path is clean against its current audit_depth anchor, but switching hard anchor to bookTicker/top5 requires explicit side-conservative rounding, anchor clamp, and post-clamp re-check.
- Enforcement gap matrix:
  - currently satisfied: `top5_not_final_hard_anchor`
  - design gaps: `fast_bbo_bookticker_hard_anchor`, `depth_bbo_guarded_fallback_only`, `side_conservative_rounding_and_post_round_recheck`
  - partial coverage: `stale_latency_join_age_submit_suppression`, `reject_throttle_drop_cooldown_path`
- Reject / throttle / churn:
  - decision reject reasons are `latency_guard=16528`, `quote_throttle=5996`, `api_interval_guard=1190`, `none=23785`
  - stage5 submit post_only_risk is `0 / 2516`
  - fast_cancel_churn remains high at `1955 / 2516`
- Current decision: `0518T003` is diagnostic-only and not ready for direct implementation. After QA, any follow-up implementation should be narrow, default-off or diagnostic-first, and limited to anchor arbitration plus side-conservative rounding/clamp/re-check.
- T003 also makes the future boundary explicit: top5 should stay pricing/risk/diagnostic context, not the final hard post-only anchor, unless a later task proves the anchor arbitration layer can be implemented safely behind a narrow default-off gate.
- `0518T003` has passed QA. The accepted conclusion remains diagnostic-only: do not repair source-level row-exact drift and do not implement production/default-on quote control from this task.
- The retained Step 5C path is a protective execution-safety layer, not a source-alignment repair:
  - keep: anchor arbitration, side-conservative rounding, clamp, post-clamp re-check, guarded fallback, stale/join-age suppression, diagnostic counters
  - exclude: audit_depth/bookTicker/top5 row-exact drift repair, top5 hard-anchor promotion, fair/reservation changes, quote-placement redesign, replay lifecycle changes, live collection, live promotion, and default-on behavior
- `0518T004` is the formal task file for this retained Step 5C path and should be executed only within that boundary.

## 0518T004 Findings

- `0518T004` implemented a default-off / diagnostic-first quote-anchor safety helper in `quote_anchor_safety.py`.
- Default behavior is unchanged unless `quote_anchor_safety.enabled=true`; existing backtest tests passed after integration.
- The helper enforces the narrow Step 5C contract:
  - bookTicker-equivalent fast anchor is preferred when fresh
  - guarded depth fallback is used only when fast anchor is missing or stale
  - top5 is not used as the final hard post-only anchor
  - bid side uses floor-style conservative ticks, ask side uses ceil-style conservative ticks
  - target ticks are clamped to the selected anchor and re-checked for post-only/crossed risk
  - missing/stale anchors suppress fresh add-side submits through the default-off safety result
- Stage 5C diagnostic on `5-13-day-control-30min` produced:
  - decision rows `47499`
  - bookTicker anchor rows `39261`
  - guarded depth fallback rows `8173`
  - stale anchor rows `65`
  - missing anchor rows `0`
  - bid clamped rows `1394`
  - ask clamped rows `2281`
  - suppress buy/sell rows `65 / 65`
  - post-only risk after re-check rows `0`
- This task did not repair audit_depth/bookTicker/top5 row-exact drift, did not promote top5 to hard anchor, did not change fair/reservation, did not change replay lifecycle, did not start live, and did not enable any default-on behavior.

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
- Pre-repair decision state was `diagnostic_only_gap_too_large`: the Stage 6B methodology worked on a single accepted sample, but replay lifecycle still deviated too much from live to treat replay fill-side behavior as close enough for promotion-style quote-adjustment experiments. This historical finding was later superseded by the accepted `0515T003` / `0515T005` repairs and the `0519T001` final rerun.

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


## 0709T003 Findings

- T011 robustness synthesis over `0708T001` and the three `0709T001` windows produced one explicit route: `route_to_quote_fill_probability_evidence`.
- The accepted evidence set has multiple submitted/resting no-fill lifecycles that replay faithfully, plus one submitted/rejected lifecycle; all four rows pass replay overall acceptance and safety invariants.
- No fill, maker fill, ledger fill row, realized PnL, rebate, inventory-PnL attribution, or maker viability evidence exists in this set. Treat economics as `no_fill_fail_closed` only.
- The next task, if QA accepts T003, should investigate quote/fill probability evidence under a separate formal task. Do not route directly to fee/PnL calibration, T012, promotion, or live expansion without fill-supported evidence and new QA acceptance.
- T003 remained offline-only and did not call network, remote/AWS, credentials, private/account/order/cancel endpoints, live submit, or market-data collection. It did not change thresholds, quote envelope, order size, max submissions, or strategy behavior.


## 0709T003 QA Accepted Findings

- QA accepted `0709T003`; the durable T011 conclusion is `route_to_quote_fill_probability_evidence`.
- The evidence supports live artifact/lifecycle/safety and non-optimistic consistency across the accepted no-fill/reject windows, not full replay-engine regression or profitability.
- Follow-up work should measure quote/fill probability under a new formal task before any fee/PnL calibration or promotion path is considered.


## 0710T001 Findings

- Quote/fill probability evidence analysis should not treat the T011 no-fill windows as measured low fill probability. The resting holds are short (`3.008385s` and `1.542778s`) and censored.
- The two `0709T001_window_01` order attempts are post-only rejects consistent with exchange post-only protection, not fill-probability samples.
- Existing artifacts include useful decision-time public-flow/depletion proxies, but they do not reconstruct public trade-through/depletion during the actual resting interval.
- Prior `0708T001` remains a QA-accepted no-fill lifecycle reference, but its local quote/fill public-flow artifact is not present in this checkout.
- Durable next route from business execution is `route_to_public_flow_artifact_repair`; do not move to fee/PnL calibration, T012, promotion, or maker viability without fill-supported evidence and new QA.


## 0710T001 QA Accepted Findings

- QA accepted `0710T001`; durable route is `route_to_public_flow_artifact_repair`.
- The no-fill evidence remains censored and artifact-limited: current public-flow data is useful as decision-time proxy only, not as actual resting-interval fill probability.
- Follow-up should repair/design resting-interval public flow/depletion artifacts before any quote/fill probability, fee/PnL, or maker viability interpretation.
## 0718T018 Findings

- Task 7 watcher/manager/status implementation is accepted, but the live evidence is a zero-submit fail-closed result rather than a completed order lifecycle.
- A waiting-phase heartbeat gap was detected during the first diagnostic run while account position and open orders were both zero; it was fixed in commit `031a198` and verified in subsequent real public/live runs.
- The 30s public-only shadow observed `57` L2 and `18` trade messages with no reconnect and no credential/private/order/cancel access, but no fresh-touch candidate reached the fair-mid source.
- Live-02 found a fresh-touch candidate and passed the immediate guard, but the fixed 7-tick edge gate blocked both evaluations.
- Live-03 found a fresh-touch candidate, but the current queue-band reprice guard failed closed before edge/order submission.
- Across the formal live windows, real order endpoint calls, cancel endpoint calls and fills were all zero. Independent final account state was BTC position `0.0` and open orders `0`.
- Terminal artifacts are integrity-checked on remote and local: shadow `15/15`, live-02 `79/79`, live-03 `79/79`, with no missing or mismatched files.
- Conservative same-window replay passes market-view and anti-optimism checks but correctly blocks decision/lifecycle/economics acceptance because submit/resting/fill facts do not exist.
- Task 8 may proceed only as estimator/dynamic-spread observe-only. Dynamic-spread activation, fill feedback, multi-level, maker viability, PnL and promotion remain unsupported.

## 0718T019 Findings

- Fixed 1s event-time buckets remove websocket message-rate weighting while preserving accepted event counts and explicit state dedupe counters.
- The estimator records normalized event rows, bucket metrics, quarantine rows, quote exposure intervals, side-specific A/k diagnostics and a core snapshot suitable for exact replay.
- Intensity observations are defined by directional at-or-through public trades during an exposure interval and the visible pre-trade side depth; `abs(trade_px-mid)` is not used as arrival intensity.
- Dynamic candidate calculation consumes A, k, volatility, risk aversion, inventory ratio, liquidity and toxicity, then applies fixed fallback, hard bounds and rate limiting.
- The shared-kernel pricing overlay keeps Task 7 fixed half-spread authoritative and rejects dynamic activation in observe-only mode.
- The real public-only window observed `113` L2 events and `161` trades across `68` event-time buckets with quarantine `0`, but produced no eligible quote exposure lifecycle.
- Because exposure observations were zero, both side-specific A/k fits stayed unavailable and the dynamic candidate correctly fell back to fixed `0.5 tick`.
- Remote/local terminal checksums passed `22/22`; replaying `274` normalized event rows reproduced the exact core snapshot hash.
- T019 supports mechanism and replay acceptance only. It does not support dynamic activation, fill-feedback activation, maker viability, profitability, multi-level or promotion.

## 0718T020 Findings

- Lifecycle normalization now joins existing attempt, resting interval, fill ledger and public coverage artifacts through `attempt_key`; fill identities use the existing stable `fill_id`.
- Rejected/never-resting attempts are excluded rather than counted as no-fill. Short holds, run-end/forced cancellation, missing terminal public coverage and identity conflicts remain censored or fail-closed.
- Partial fills retain `filled_qty/original_qty`; pooled feedback uses original quantity times resting exposure seconds as the weight. Target fill ratio is not hardcoded and an unconfigured target produces a neutral candidate.
- Controller state is versioned and checksummed. Schema, controller version, config hash or bounds failure resets state to neutral; candidate output remains observe-only and cannot change fixed quotes or order paths.
- The first T020 remote attempt exposed an environment-only websocket dependency failure under `/usr/bin/python3`; no private/order endpoint was called. The second attempt used the existing SDK venv and collected public data successfully.
- T020 public-only evidence still has zero real resting lifecycle and zero fill feedback observations. This is an evidence limitation, not a basis for activation or economic inference.
- Remote/local artifact integrity and both estimator/feedback replay snapshots are exact. Task 10 multi-level must keep a hard prerequisite gate for an accepted single-level lifecycle; T021 is limited to default-off ladder contract and gate mechanics.

## 0718T021 Findings

- T021 is QA accepted with implementation commit `53d7db2`.
- `QuoteLadderConfigV1` and `multi_level_prerequisite_gate()` define a versioned deterministic contract for levels, gap, size decay, price/lot/min/max and aggregate exposure.
- Level 0 reuses the authoritative reservation/fixed half-spread path; deeper rows are legalized, post-only and deterministic, but remain hypothetical.
- Duplicate rounded prices coalesce or fail closed; invalid sizes, post-only violations and aggregate exposure cap breaches fail closed.
- The manager/status artifact records `activation_enabled=false` and `actual_quote_behavior_changed=false`; current single-level live behavior is unchanged.
- T021 does not complete Task 10 activation. The absence of an accepted real single-level resting/fill lifecycle remains the blocker.
- The only next task is `0718T022 / P3-REAL-TIME-STATUS-FILE`, limited to status schema and writer failure/degraded observability.

## 0718T022 Findings

- T022 is QA accepted with implementation commit `b7bca85`.
- `cross_exchange_live_status_v2` preserves legacy flat fields while adding complete identity, market, pricing, quote, signal, exposure, order, fill, toxicity, risk, kill-switch, activity and process sections.
- Open orders are grouped by side/level/state; working and submit-inflight exposure are separated while the authoritative projected exposure remains visible.
- The status writer remains stateful, atomic and monotonic-throttled. Health counters are embedded in each successful status file.
- A replace or clock failure writes `live_status_writer_audit.jsonl` and raises `LiveStatusWriteError`; status loss is not silently downgraded.
- No quote/order/risk/controller/activation behavior changed.
- The only next task is T023 cumulative tiny-live/same-window acceptance. It must remain single-level fixed-spread until real lifecycle evidence satisfies the T021 gate.

## 0718T024 QA Findings

- Exact task-scoped caps and artifact identity are necessary but not sufficient for accepted live provenance.
- A preflight source marker plus equality of remote directory paths cannot prove the source bytes used when the watcher starts. Runtime-critical source commit/hash evidence must be written before child launch and sealed with the run.
- T024 did obtain a real post-only submit/resting/cancel lifecycle with final/independent open orders zero, position zero and no observed fills.
- The producer's `fill_reconciliation_required_no_fill_unproven` blocker remains authoritative. A downstream acceptance tool must not silently ignore or downgrade an unclassified producer blocker.
- The generic exchange text "already canceled, or filled" from a redundant cancel is not by itself proof of a fill. Reconciliation should distinguish an earlier successful cancel from true unknown terminal state, while still failing closed when authoritative fill/account evidence conflicts.
- T024 is diagnostic evidence only. Principal Task 12 remains open.
- Multi-level remains default-off: T024 is a one-sided fresh-touch lifecycle, not the accepted single-level two-sided manager lifecycle required by the original Task 10 prerequisite.

## 0719T004 Findings

- Raw oid/cloid cannot be used as a persisted reconciliation key when the generic artifact writer redacts those fields.
- A type-bound SHA-256 opaque token lets producer-written raw proof survive redaction without exposing oid/cloid and keeps reference keys deterministic.
- Producer summaries should contain only opaque identity tokens; retaining raw identifiers in summaries creates representation drift under recursive redaction.
- Attempt identity is an exact protocol field, not a numeric quantity. Parsing through `int()` is unsafe because floats and fractional values can alias the same attempt.
- The accepted parser domain is positive integer values and canonical positive-integer strings only; all other forms fail closed.
- Independent acceptance reconstruction remains separate from producer helpers and exact-compares its result with both stored producer summaries.
- Actual standalone and two-sided manager writer integration is necessary; in-memory reconciliation equality alone does not prove persisted evidence integrity.
- T004 changes evidence identity only. It does not change quoting, strategy thresholds, risk caps, live envelope, controller activation or multi-level readiness.

## 0719T004 QA Findings

- A cancel response dictionary is not authoritative merely because it contains a `success` key. The success value and the entire status list cardinality/shape must match an explicit accepted contract.
- Producer and independent acceptance reproduced the same malformed-success weakness, so exact summary equality alone cannot compensate for a shared semantic bug.
- `{"success": false}`, null, numeric zero, empty string, object or list must all fail closed.
- Attempt parsing needs a bounded canonical domain before integer conversion. Regex acceptance alone is insufficient because oversized digit strings can raise at conversion time.
- The redaction-safe token and fractional-alias repairs remain valid accepted sub-results, but no live task may proceed until the new response/attempt boundary is QA accepted.

## 0719T005 Findings

- Authoritative cancel proof is a protocol validation problem, not a truthiness check.
- One target-bound cancel row permits exactly one status. Multiple statuses are ambiguous for a single reference and fail closed.
- A dictionary status is accepted only when its sole key is `success` and the value is a nonempty canonical string or positive integer reference.
- Producer and acceptance implement this independently; synchronized summaries cannot turn malformed raw evidence into a pass.
- Bounded identity parsing checks width before integer conversion and enforces `1..2147483647`.
- Malformed nested response containers must raise a controlled validation failure, not an incidental attribute error.
- T005 preserves the T004 persisted identity contract and does not alter strategy or live behavior.

## 0719T005 QA Accepted Findings

- QA accepted the strict cancel-success protocol and bounded attempt parser with no new finding.
- Complete synchronized-summary acceptance attacks cannot convert malformed success payloads into mechanism pass.
- Historical local cancel evidence is compatible with the stricter protocol.
- The offline evidence-integrity gate is now clear for the next formal two-sided manager evidence-contract task.
- This acceptance does not itself close Principal Task 12 or unlock multi-level.

## 0719T006 Findings

- An exact live envelope is not exact if its behavior profile can be inferred from a default; exact runs now require an explicit legacy or two-sided profile.
- Requote attempts and real order submission caps are separate protocol fields and must remain independently propagated and accepted.
- A two-sided manager lifecycle needs one canonical attempt, actual order response, status row, tracked reference and terminal cancel proof per side; an aggregate `buy+sell` row is not primary evidence.
- Submission accounting must count actual order endpoint calls, including rejected or ambiguous responses, rather than only orders that reach resting.
- Reconstructing an exchange-shaped resting response from manager state is weaker than persisting the real redacted `client.order()` response.
- Acceptance joins buy/sell evidence by side and exact attempt identity, then independently rebuilds raw cancel reconciliation; synchronized producer summaries alone are insufficient.
- Actual writer integration is required in addition to hand-built fixtures so task/window identity and redaction behavior are tested after persistence.
- T006 changes orchestration and evidence contracts only. It does not change pricing formulas, edge thresholds, risk caps, activation flags or multi-level readiness.

## 0719T006 QA Findings

- Counting two order-result objects is not proof that two raw exchange responses reached resting or belong to the corresponding side, attempt and terminal reference.
- Terminal proof is required per submitted attempt regardless of whether fills are present; a fill does not make failed or unrelated cancel evidence acceptable.
- An exact two-sided profile must validate `{buy,sell}` before the first order endpoint call. Post-run rejection is too late.
- Command acceptance must model argparse's last-value behavior or, preferably, reject duplicate flags and compare one canonical argv.
- A writer-only fixture is not an end-to-end producer integration. The manager watcher must generate the artifact tree that acceptance consumes.
- Green regressions cannot override deterministic fail-open counterexamples.

## 0719T007 Findings

- Persisted exchange responses need an attempt-bound row in addition to a result list; cardinality alone cannot establish lifecycle identity.
- Response reference tokens must be derived from raw oid/cloid before generic redaction. Producer-supplied conflicting tokens must fail closed.
- Terminality is per submitted attempt. A complete reference-bound maker fill may replace cancel success, but partial fill and unrelated cancel evidence cannot.
- Acceptance must account for every fill row; selecting only a self-consistent subset leaves an evidence-injection gap.
- The exact two-sided invariant belongs before `reconcile_desired()`, because post-run acceptance cannot undo a one-sided live order.
- A sealed command remains ambiguous if duplicate flags are allowed. Exact runs require one canonical argv, one mode and explicit task/window run identity.
- Canonical artifact identity includes the normalized `window_01` directory. Reading legacy `window_1` allowed stale fixtures to mask the real producer output.
- T007 changes execution/evidence safety only. Pricing formulas, risk caps, controller activation and multi-level behavior remain unchanged.

## 0719T007 QA Findings

- A fill CSV derived from producer state is not independent terminal evidence. Acceptance must rebuild fills from persisted raw pullback payloads and exact-compare the derived ledger contract.
- Fill reference identity uses all supplied identifiers. If both oid and cloid exist, both must resolve to the same canonical attempt; one matching token cannot excuse one conflicting token.
- `argparse` long-option abbreviation is part of runtime semantics unless explicitly disabled. Canonical acceptance must either disable abbreviation or model every accepted abbreviation.
- Exact command identity includes the Python executable, watcher script and output directory, not only the option values.
- A canonical artifact directory can still contain stale evidence unless the executed output path and runtime source provenance are bound to that same directory.

## 0719T008 Findings

- A fill terminal claim is independent only when acceptance reconstructs it from persisted raw pullbacks; producer-generated ledger and attribution CSV files are corroborating outputs, not the root fact source.
- Redaction-safe fill evidence needs the same tokenized identity used by stable fill IDs and payload fingerprints. Removing raw oid/cloid after deriving type-bound tokens preserves privacy and deterministic replay.
- Fill reference attribution is an all-token intersection. Every supplied oid/cloid must resolve to one canonical attempt; OR semantics creates a deterministic cross-attempt injection path.
- Pullback evidence must carry the exact mark and fee-rate context used by producer ingestion, otherwise fee and mark fields cannot be independently reproduced.
- Exact comparison should cover ledger, attribution, liquidity role and producer summary. Checking only row identity leaves quantity, role or fee drift unbound.
- `allow_abbrev=False` is part of the execution safety contract for argparse-based live tools. Acceptance must still enforce a complete canonical flag sequence because parser behavior alone does not bind artifact provenance.
- Runtime source provenance must seal the exact child command before start, while acceptance independently binds the physical run root, approved Python environment, watcher entrypoint and canonical output directory.
- T008 changes evidence and execution-command safety only. It does not change strategy formulas, thresholds, risk caps, controller activation or multi-level readiness.
