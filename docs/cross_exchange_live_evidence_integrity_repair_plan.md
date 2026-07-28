# Cross-Exchange Live Evidence Integrity Repair Plan

## Recovery Notice

This file was reconstructed on 2026-07-28 because the original 2026-07-17
controller document was not present in the canonical checkout, reachable Git
history, or searched recovery locations.

This is not a verbatim recovery. It is a conservative reconstruction from:

- the durable `task_plan.md`, `progress.md`, and `findings.md` summaries;
- the surviving `0717T007` through `0717T011` task and QA chain;
- the `0717T005` findings retained in the controller summaries.

This document records a historical offline repair sequence. It grants no
credential, private endpoint, remote execution, live order, cancel, service,
strategy, promotion, or risk-envelope authorization.

## Purpose

The plan separates four evidence and control defects found after the
`0717T005` remote-update review:

1. multi-window and attempt identity;
2. idempotent, attempt-bounded fill attribution;
3. watcher child termination and timeout handling;
4. terminal artifact sealing and checksum verification.

The controller explicitly removed `runtime_risk_envelope_not_enforced` from
this immediate repair route. The then-current max-loss and max-position
defaults were accepted for that bounded optimization stage. This did not
authorize larger exposure, longer duration, additional concurrency, taker
execution, or promotion.

## Global Constraints

- Execute one formal task at a time.
- Keep every phase offline until its own QA acceptance.
- Do not change strategy calculations, quote placement, thresholds, order
  size, submission caps, max loss, or position limits.
- Do not access credentials, private/account/order/cancel endpoints, remote
  hosts, or live services.
- Preserve fail-closed behavior for ambiguous identity, fill attribution,
  process lifecycle, and artifact integrity.

## Phase 1: Window And Attempt Identity

- The orchestrator window index is the authority for the artifact window.
- Stable attempt keys use
  `<task_id>:window_<zero-padded-window-id>:attempt_<attempt-id>`.
- Inline manifests, fill rows, attempt rows, and copied artifact paths carry
  the same window and attempt identity.
- Existing single-window callers remain compatible with `window_01`.
- Historical implementation task:
  `0717T007 / WINDOW-ATTEMPT-IDENTITY-REPAIR`.

## Phase 2: Idempotent Fill Attribution

- One window owns one fill ledger across all pullbacks.
- Stable fill identity cannot depend on list position or mark price.
- Oid/cloid matches take precedence over time-bounded fallback.
- Fallback attribution must be unique, attempt-bounded, and quantity-bounded.
- Repeated pullbacks cannot duplicate quantity or fee.
- Conflicting, ambiguous, foreign-reference, pre-attempt, post-terminal, and
  over-quantity fills remain visible but fail closed for attribution.
- Historical implementation task:
  `0717T008 / IDEMPOTENT-FILL-ATTRIBUTION-REPAIR`.

## Phase 3: Watcher Termination And Timeout

- The orchestrator owns the child process group.
- Signal and timeout handling occurs in normal polling control flow.
- Termination uses `SIGTERM`, a bounded grace period, optional `SIGKILL`, and
  an explicit reap.
- No later window starts after abort or timeout.
- Independent open-orders proof runs only after child exit and reap.
- Historical implementation task:
  `0717T009 / WATCHER-TERMINATION-TIMEOUT-REPAIR`.

## Phase 4: Terminal Artifact Seal

- Success and failure use the same terminal ordering.
- Child and window evidence become final before root terminal status.
- Heartbeat is stopped and joined before sealing.
- The manifest uses run-root-relative paths and is written once.
- Manifest entries are verified immediately.
- Only the excluded verification summary may be written after the seal.
- Historical implementation task:
  `0717T010 / TERMINAL-ARTIFACT-SEAL-REPAIR`.

## Phase 5: Integrated Offline Acceptance

- Exercise Phases 1-4 in one offline multi-window fixture.
- Cover repeated fill pullback, same-price attempts, ambiguous attribution,
  timeout, child reap, no-next-window, proof ordering, and checksum validation.
- Do not treat component test success as integrated acceptance.
- Historical implementation task:
  `0717T011 / LIVE-EVIDENCE-INTEGRATED-OFFLINE-ACCEPTANCE`.

## Completion Rule

The repair route is complete only when all five tasks have independent QA
acceptance. Completion means the historical evidence pipeline contracts were
accepted. It does not prove profitability, maker viability, live readiness,
fee/PnL calibration, or production promotion.
