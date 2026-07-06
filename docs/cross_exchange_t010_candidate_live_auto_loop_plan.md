# Cross-Exchange Full T010 Candidate/Live Evidence Auto-Loop Plan

## Purpose

This plan defines the next three-task auto-loop after `0706T007 / 0625T010-LIVE-EVIDENCE-ACQUISITION`.

The immediate problem is not total absence of data. `0706T007` collected public-flow and gate evidence, but the authorized live evidence attempt stopped before submit because the same-window gate produced:

- fresh-touch candidates: `10`
- fresh-touch allowed candidates: `0`
- submitted orders: `0`
- `real_order_endpoint_called=false`
- `fill_count=0`

Therefore the next route must first estimate whether longer public observation produces eligible candidates, before deciding between repair and another controlled live evidence attempt.

## Authorization Record

The user explicitly authorized future low-risk live testing in the current session:

```text
我这边授权后续的live test权限，因为live仓位很低，账户金额也很小，所以不用担心太大风险，授权live进行测试
```

This authorization is recorded as conditional authorization for Task 3 only, and only if Task 1 QA shows that the required no-submit gates are satisfied. It does not authorize default-on behavior, continuous bot operation, quote-envelope expansion, size increase above the plan cap, taker/crossing orders, inside-spread placement, one-tick-back placement, promotion, stable PnL claims, maker viability claims, full T010 acceptance, T011, T012, or final MVP pass.

## Auto-Loop Overview

```text
0706T008 long-window no-submit diagnosis
  |
  |-- if allowed/would-submit evidence remains absent or malformed
  |      -> 0706T009 repair task
  |
  |-- if allowed/would-submit evidence is present and boundaries pass
         -> 0706T010 controlled live evidence task
```

Only `0706T008` is dispatched immediately. `0706T009` and `0706T010` are planned branch targets and must be created as formal task files only after `0706T008` QA.

## Task 1: `0706T008 / 0625T010-LONG-WINDOW-NOSUBMIT-DIAGNOSIS`

Goal:

- Run a longer AWS public-only/no-submit window to measure whether the current accepted gate can produce eligible candidates often enough to justify a controlled live evidence task.

Default execution:

- Host: `awsserver1`
- Remote repo: `/home/admin/hftbacktest-cross-exchange`
- Required interpreter: `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`
- Runner: `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- Mode: `--event-driven-public-shadow-source-live`
- Duration: `1800s`
- If public streams are healthy but candidate evidence is too sparse, one bounded extension to `3600s` total is allowed inside this task.
- Boundary: public-only, no-submit, no credentials, no private/account/order/cancel endpoints, no live client, no final gate rerun.

Required outputs:

- Public stream health: l2Book/trades/subscription/reconnect/timeout counts.
- Candidate counts and stage-level funnel:
  - current candidates
  - fresh-touch evidence pass
  - fresh-touch allowed
  - anti-drift pass/block
  - fair-mid source pass/block
  - edge gate pass/block
  - shadow would-submit
- Dominant blocker distribution.
- No-submit/no-private/no-order boundary manifest.
- Recommendation:
  - `route_to_repair_task`
  - `route_to_controlled_live_evidence_task`
  - `blocked_public_stream_or_artifact_failure`

Branch decision after QA:

- Route to Task 2 if any of these hold:
  - public stream unusable
  - candidate rows absent despite healthy streams
  - `fresh_touch_allowed_count=0`
  - downstream fair-mid/edge path is not reached
  - artifact schema/boundary evidence is incomplete
  - would-submit rows remain absent for explainable gate/model reasons that require implementation repair
- Route to Task 3 if all of these hold:
  - public stream healthy
  - no-submit/no-private/no-order boundaries pass
  - eligible candidates appear in the long window
  - cross-exchange decision path is observable enough to bind signal/fair-mid/quote intent to a future live attempt
  - QA accepts that a controlled live evidence task is justified

## Task 2: `0706T009 / 0625T010-CANDIDATE-GATE-REPAIR`

Goal:

- Repair the first proven blocker found by `0706T008`, without placing live orders.

Allowed scope:

- Code and tests around the public shadow / candidate funnel / signal-fair-mid decision path.
- Offline or public-only no-submit validation.
- Artifact schema fixes if the blocker is missing or ambiguous evidence rather than strategy behavior.

Not allowed:

- Live-submit.
- Private/account/order/cancel endpoint use.
- Quote-envelope loosening unless explicitly scoped and QA-approved as a separate design decision.
- Size increase.
- Taker/crossing/inside-spread/one-tick-back behavior.

Exit:

- If repair makes long-window no-submit evidence pass, route back to a fresh Task 1-style no-submit validation or directly to Task 3 only if the repair task itself produced equivalent QA-accepted no-submit evidence.
- If repair shows the current signal/gate cannot produce candidates without a major strategy decision, stop for human/controller decision.

## Task 3: `0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE`

Goal:

- Execute a small, bounded live evidence task only after Task 1 or Task 2 proves no-submit eligibility.

Conditional live envelope:

- Host: `awsserver1`
- Remote repo: `/home/admin/hftbacktest-cross-exchange`
- Env file: `/home/admin/XEMM_rust_latest/.env`
- Symbol: Hyperliquid `BTC`
- Order type: limit
- TIF: post-only `Alo`
- Side policy: `fresh_touch`
- Quote offset: `0` tick touch-only
- Max windows: `1`
- Max submissions: `2`
- Max order size: `0.005 BTC`
- Quote hold: `3s`
- Wait seconds: `10`
- Fresh-touch precheck: must be tied to the accepted no-submit eligibility evidence.
- Tracked cancel required.
- Independent final open-orders proof required.

Hard stops:

- No taker/crossing orders.
- No inside-spread or one-tick-back placement.
- No quote-envelope change.
- No size increase above `0.005 BTC`.
- No default-on or continuous bot.
- No promotion or final MVP pass.

Required evidence:

- Same-window public market view.
- Decision path linking cross-exchange signal, fair-mid, side, quote intent, and pre-submit gates.
- Submit/resting/reject/cancel lifecycle if an order is submitted.
- Fill/no-fill lifecycle.
- Fee/rebate, inventory, and realized PnL evidence if fill occurs; explicit fail-closed unsupported status if no fill occurs.
- Shutdown and independent final open-orders proof.
- Replay optimism boundary.

Exit:

- Passing Task 3 can unlock a full `0625T010` same-window replay acceptance task only if QA accepts complete lifecycle/economics/open-orders evidence.
- If Task 3 submits no order or produces no lifecycle/economics evidence, it remains a fail-closed evidence attempt and does not unlock full T010/T011/T012.

## Current Dispatch

Dispatch now:

- `0706T008 / 0625T010-LONG-WINDOW-NOSUBMIT-DIAGNOSIS`

Do not create or execute `0706T009` or `0706T010` until `0706T008` has a QA result.
