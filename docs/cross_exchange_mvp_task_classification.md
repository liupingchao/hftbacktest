# Cross-Exchange MVP Task Classification

This document classifies historical cross-exchange work under the four required MVP milestones in `docs/cross_exchange_maker_mvp_plan.md`.

## Branch And Fact-Source Rules

- `cross-exchange` is the canonical branch for all formal Binance-lead / Hyperliquid-lag MVP work.
- Other branches are temporary or recovery branches. They may preserve useful commits, but they are not workflow facts until the relevant task files, reports, artifacts, and code are restored onto `cross-exchange`.
- The MVP milestone order is the highest planning constraint for this branch:
  - M-A Signal Contract
  - M-B Production-Equivalent Shadow
  - M-C Minimal Hyperliquid Alignment
  - M-D Integrated MVP
- A later milestone may reuse earlier work, but it must not claim milestone completion until the explicit milestone tasks in the MVP plan pass QA.

## M-A Signal Contract

Purpose:
- Prove Binance top5 lead information has repeatable, decision-time-clean direction and magnitude for Hyperliquid future price.
- Freeze MVP v1 signal schema, horizon, freshness, side mapping, and edge formula.

Formal MVP tasks:
- `0625T001` Public Alpha / Edge Decomposition: `已通过`.
- `0625T002` Synchronized Public Sample Expansion: superseded as a T003 input by the repaired effective-horizon gate because ordinary HL `l2Book` produced nominal `1000ms` labels with effective horizon around `5000ms`.
- `0627T001` Hyperliquid fast `l2Book` synchronized sample rerun: `已通过`; formal M-A supplement that repairs the near-target `1000ms` sample input by using HL `l2Book fast=true` and off-AWS alignment.
- `0625T003` Out-of-Sample Signal Acceptance: business execution complete, pending QA. It used the accepted `0627T001` package and recommended `signal_contract_accepted_for_shadow`.

Reusable M-A baseline:
- `0601T001`-`0601T006`: Hyperliquid public sample, Binance-to-HL as-of join, lead-lag analysis, data contract, pricing-signal runner, and public multi-sample robustness.
- `0604T001`-`0604T009`: event-mode / canonical pricing-signal evidence, source-lock, signal quality, horizon/regime diagnostics, and regime synthesis.
- `0608T002`-`0608T006`, `0609T001`-`0609T002`: maker executability, directional/basis-positive robustness, and targeted public collection diagnostics.

M-A is not complete until `0625T003` QA passes with `signal_contract_accepted_for_shadow`.

## M-B Production-Equivalent Shadow

Purpose:
- Use one shared signal / fair-mid / quote-intent kernel for public shadow and later replay/live paths.
- Produce enough no-submit would-submit decisions and counterfactual markout evidence.

Formal MVP tasks:
- `0625T004` Shared Signal and Quote-Intent Kernel: not created.
- `0625T005` Multi-Window Production Shadow Acceptance: not created.

Reusable M-B baseline:
- `0623T001`-`0623T004`: public-state freshness, fresh-touch evidence, flow taxonomy, and fair-value edge gate.
- `0623T006`-`0623T010`: fair-mid source, public-source shadow, AWS public no-submit soak, and candidate funnel diagnosis.
- `0624T001`-`0624T003`: BBO evidence-chain diagnosis / repair and AWS repaired public-shadow funnel validation.

These are M-B inputs only. They do not complete M-B because the frozen M-A signal contract does not yet exist.

## M-C Minimal Hyperliquid Alignment

Purpose:
- Build only the Hyperliquid alignment layer required for the MVP, not a full duplicate of the Binance single-exchange framework.
- Align public market view, decision cadence, signal, side, quote intent, block reason, and real lifecycle/cost evidence.

Formal MVP tasks:
- `0625T006` Hyperliquid MVP Audit and Replay Contract: not created.
- `0625T007` Hyperliquid Public Market-View Replay Alignment: not created.
- `0625T008` Edge-Qualified Tiny-Live Calibration: not created.
- `0625T009` Execution Outcome Calibration: not created.

Reusable M-C baseline:
- `0529T003`: Hyperliquid raw-to-npz and top-N provenance foundation.
- `0610T002`-`0611T004`: execution-evidence, private-order-response, replay-lifecycle, and account-inventory source-line / artifact skeleton work.
- `0618T001`-`0618T004`: SDK readiness, credential-location scan, real-order executor, and real-order canary interface validation.
- `0618T008`: fee / inventory / realized PnL fail-closed ledger.

These are reusable inputs only. They do not complete M-C until `0625T006`-`0625T009` pass in order.

## M-D Integrated MVP

Purpose:
- Calibrate the minimal execution model from tiny-live data and complete same-window replay/live acceptance plus multi-window final validation.

Formal MVP tasks:
- `0625T010` Same-Window Replay Acceptance: not created.
- `0625T011` Multi-Sample Robustness: not created.
- `0625T012` Final Controlled MVP Validation: not created.

Reusable M-D baseline:
- `0618T005`-`0618T007`: M0/M1 baseline and repeated canary windows.
- `0618T009`-`0618T010`, `0619T001`, `0622T001`-`0622T006`: tiny-live fill / submit / reprice repair attempts. These are mostly blocked or task-level accepted without stable fill/PnL proof.
- `0618T011`-`0618T012`, `0619T002`: no-fill diagnosis, public flow diagnosis, and maker quote placement / size / time-of-day redesign.

These inform D-stage design but do not complete D-stage acceptance.

## Current Controller State

- The immediate canonical branch state is `0625T003` business complete and awaiting QA, fed by the accepted `0627T001` M-A supplement.
- `0625T003` may unlock M-B only if QA accepts `signal_contract_accepted_for_shadow`.
- No M-B, M-C, or M-D formal task is currently dispatched.
