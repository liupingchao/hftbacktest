# Cross-Exchange First Three Execution Sequence

Date: 2026-07-16

## Objective

Complete the first three shortfall-plan objectives in strict workflow order:

1.补 fill source / maker-taker role 证据。
2.跑 T004 kernel 的 production-equivalent public shadow。
3.固化 cross-exchange price taxonomy。

This sequence is deliberately gated. Only the first task is formalized immediately. Later tasks should be created only after the previous task has QA status `已通过`.

## Sequencing Rules

- Execute one formal task at a time.
- Every formal task must have a `.workflow/tasks/<TASK_ID>.md` file.
- Business/test reports end in `待验收`.
- QA reports end only in `已通过`, `未通过`, or `阻塞`.
- No live retry, threshold change, quote-envelope change, order-size/max-submission change, fee/PnL calibration, maker viability claim, T012, promotion, or final MVP pass is authorized by this sequence document.
- A controlled live evidence task requires a separate exact UTC schedule, live envelope, host/account scope, and controller authorization.

## Task Chain

| Order | Task ID | Formal now | Title | Purpose | QA gate to unlock next |
| --- | --- | --- | --- | --- | --- |
| 1 | `0716T005` | Yes | T011-FILL-SOURCE-LIQUIDITY-ROLE-CONTROLLED-EVIDENCE-PREFLIGHT | Convert 0716T003/0716T004 into an executable evidence-preflight contract for future maker/taker role and fill source capture. | QA confirms future controlled evidence can prove `confirmed_maker` / `confirmed_taker` / `unknown_liquidity_role`, exact fill timestamp/source, and fee/PnL blocking rules. |
| 2 | `TBD-after-0716T005-QA` | No | CONTROLLED-EVIDENCE-ACQUISITION-WITH-LIQUIDITY-ROLE-CONTRACT | Collect or replay only the evidence authorized by 0716T005. If live is required, this task must be separately authorized. | QA confirms actual artifacts contain role/source-path evidence or records blocker. |
| 3 | `TBD-after-role-evidence-QA` | No | T004-KERNEL-PRODUCTION-EQUIVALENT-PUBLIC-SHADOW | Run public-only/no-submit production-equivalent shadow using the accepted 0625T004 shared kernel. | QA confirms would-submit, fair/forecast-mid, edge, block reason, source-age, basis bucket, and counterfactual markout artifacts are complete. |
| 4 | `TBD-after-shadow-QA` | No | CROSS-EXCHANGE-PRICE-TAXONOMY-CONTRACT | Freeze mid/micro/fair/forecast/reservation/quote field definitions and decision-input status. | QA confirms future tasks can distinguish forecast-mid, fair-mid, reservation price, quote price, basis context, and future labels. |

## Why Four Formal Steps For Three Goals

The first goal has two layers:

- `0716T005` is an offline/preflight task. It defines exactly what evidence must be captured and how it will be accepted.
- The follow-up evidence-acquisition task is separate because it may involve controlled live collection. It must not be silently bundled into an offline design task.

This keeps the workflow aligned with the current repository state: 0716T003 repairs future liquidity-role artifacts, but current accepted evidence has not yet proven a new controlled package using that contract.

## 0716T005 Acceptance Intent

`0716T005` should produce:

- Role evidence field contract:
  - `confirmed_maker`
  - `confirmed_taker`
  - `unknown_liquidity_role`
- Fill source-path contract:
  - exchange-native fill timestamp
  - oid/client-order-id linkage
  - attempt key
  - symbol/side/price/size fallback status
  - user fills pullback audit
- Fee/PnL gate:
  - role-known fills may proceed to later economics calibration
  - unknown-role fills remain blocked
- Future controlled evidence task template:
  - exact required artifacts
  - validation commands
  - fail-closed states
  - unsupported claims

`0716T005` must not:

- run live
- change quote policy
- change thresholds or quote envelope
- change order size or max submissions
- calibrate fee/PnL
- claim maker fill count or maker viability

## Follow-Up Creation Rules

After `0716T005` QA:

- If `已通过`, create the controlled evidence acquisition task.
- If `未通过`, repair 0716T005 outputs before continuing.
- If `阻塞`, do not create shadow or price-taxonomy tasks until the blocker is resolved or the controller explicitly changes the route.

After controlled evidence QA:

- If role/source-path evidence is complete enough, create the T004-kernel public-shadow task.
- If evidence is incomplete but artifact paths work, decide whether to rerun evidence or downgrade scope.
- If role remains unknown, do not start fee/PnL calibration.

After public-shadow QA:

- Create the price taxonomy contract task.
- The taxonomy task may reference shadow fields as concrete examples, but must remain design/schema level unless separately authorized.

