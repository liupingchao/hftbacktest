# Cross-Exchange MVP Task Classification

Updated: 2026-07-28

This document classifies the current Binance-lead / Hyperliquid-lag work under
the four milestones in `docs/cross_exchange_maker_mvp_plan.md`.

## Fact-Source Rules

- `cross-exchange` is the canonical branch.
- Formal task status comes from `.workflow/tasks/`.
- Final acceptance status comes from the latest corresponding QA report.
- `task_plan.md` describes the current controller node.
- Action-path, replay-model, and live-derived evidence must remain separate.
- A task-level pass does not imply stable PnL, maker viability, promotion, or
  final MVP completion.

## M-A Signal Contract

Status: `已完成`.

Accepted tasks:

- `0625T001`: public alpha / edge decomposition.
- `0627T001`: repaired fast-`l2Book` synchronized sample.
- `0625T003`: out-of-sample signal acceptance, QA `已通过`.

Accepted result:

- signal: `binance_lead_composite`;
- horizon: nominal `1000ms` with row-level effective-horizon gate;
- decision inputs: Binance top5 imbalance, microprice-minus-mid, and short
  Binance mid move;
- side mapping and normalization are frozen for the accepted contract;
- warning/source-age buckets remain visible.

Boundary:

- M-A proves a decision-time-clean public signal contract.
- It does not prove live fills, fees, PnL, or promotion.

## M-B Production-Equivalent Shadow

Status: `已完成`.

Accepted tasks:

- `0625T004`: shared signal and quote-intent kernel, QA `已通过`.
- `0625T005`: multi-window production-equivalent public shadow, QA `已通过`.
- `0722T061`: basis-regression candidate accepted for public shadow.
- `0722T063`: strict basis contract and production-shadow repair accepted.
- `0722T066`: deterministic public multi-distance dynamic seed accepted.
- `0722T067`: exact seed wired into the strict production quote path.

Accepted result:

- one shared decision kernel covers signal, forecast/fair value, reservation,
  quote intent, and block/fallback reasons;
- public shadow is deterministic and no-submit;
- basis regression is accepted only for public-shadow scope;
- seeded dynamic quotes fail closed unless exact seed, current candidate,
  no-fallback, and final quote-change checks pass.

Boundary:

- Same-package or public-shadow evidence is not role-known fill or economics
  evidence.

## M-C Minimal Hyperliquid Alignment

Status: `机制完成，经济性未完成`.

Accepted tasks and task chains:

- `0625T006`: Hyperliquid MVP audit and replay contract.
- `0625T007`: public market-view replay alignment.
- `0717T006`-`0717T011`: live evidence integrity foundation.
- `0718T012`-`0718T022`: Principal Alignment price, risk, exposure, pricing,
  manager, watcher, estimator, feedback, ladder-gate, and status contracts.
- `0721T044` plus `0721T046`: accepted fixed-quote single-level same-window
  mechanism/evidence baseline.
- `0721T047`: bounded dynamic-spread activation mechanism.
- `0722T055`: multi-level offline code/action-path readiness.

Accepted result:

- public replay reproduces `10704/10704` reference decisions with zero future
  joins and zero unexplained action mismatch;
- runtime source, task/window/attempt identity, account continuity, lifecycle,
  cancel proof, terminal state, and artifact sealing have fail-closed contracts;
- dynamic/fill-feedback/multi-level components are isolated behind explicit
  config and acceptance gates.

Boundary:

- M-C does not yet contain a role-known maker fill suitable for fee/rebate,
  fill-rate, markout, inventory, or realized-PnL calibration.

## M-D Integrated MVP

Status: `部分完成 / 阻塞`.

Reusable accepted work:

- supported-fact same-window replay and multi-window no-fill robustness;
- strict submit/resting/reject/cancel reconciliation;
- role-aware fill and economics schemas that fail closed when evidence is
  absent;
- exact seeded-dynamic live mechanism in `0726T068`.

Latest formal evidence:

- `0726T068` QA status: `阻塞`;
- three independent windows;
- two submitted BTC `Alo` intents in the third window;
- one post-only reject;
- one resting order with authoritative cancel;
- final open orders `0`, BTC position `0.0`;
- total fills `0`, liquidity-role rows `0`.

Remaining M-D gates:

- at least one independently accepted role-known fill;
- exchange-native fee/rebate and inventory transition;
- replay/live economics reconciliation;
- fill/markout error and anti-optimism acceptance across controlled windows;
- explicit final MVP QA.

M-D must not be described as complete while these gates remain open.

## Current Controller State

- Current documentation repair task: `0728T069`.
- Latest strategy/live QA source of truth: `0726T068`, status `阻塞`.
- The next strategy task, if created, should preserve the accepted mechanism
  and target only role-known fill/economics evidence under a new exact
  authorization boundary.
