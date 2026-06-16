# Corrected Next Task Boundary

Task: `0616T001`

## Decision

Do not dispatch the planned `0615T009` Binance `BTCUSDT` small-cap live test on the `cross-exchange` branch.

The branch objective is Binance-led Hyperliquid maker research:

- Binance provides lead-side pricing / volatility inputs.
- Hyperliquid is the lag maker execution venue and venue-state context.

## Correct Next Task Shape

The next formal task should be a Hyperliquid maker execution-readiness boundary, not a live order task.

Recommended title:

- `Hyperliquid maker private/order execution-readiness boundary for Binance-led cross-exchange strategy`

Allowed scope:

- Define Hyperliquid private/order response artifact contract.
- Define endpoint/permission boundary without using credentials.
- Define post-only order intent semantics, cancel-all/shutdown proof requirements, account/inventory/economics evidence requirements, redaction/storage policy, and fail-closed QA gates.
- Map how future Hyperliquid private/order/account/economics artifacts would feed a proof-limited execution evidence runner.
- Preserve Binance lead inputs from `0601T004` / `0601T005` / `0609T002` as read-only pricing context only.

Forbidden scope:

- No Hyperliquid live order placement.
- No private endpoint calls.
- No credentials, signing, nonce, or user-stream implementation.
- No strategy implementation, quote generation, shadow decision, parameter search, default-on behavior, tiny-live, deployment, promotion, or PnL proof.
- No reuse of Binance `BTCUSDT` small-cap protocol as authorization to trade.

## Required QA Gates Before Any Future Hyperliquid Live Task

- Hyperliquid private/order response source-line contract passes QA.
- Hyperliquid account/inventory artifact contract passes QA.
- Hyperliquid economics/fee/funding/rebate artifact contract passes QA.
- Hyperliquid cancel-all/shutdown proof dry-run passes QA.
- Source-chain runner-consumption gate is rebuilt over Hyperliquid-specific artifacts.
- Proof-limited runner validates missing/stale/inconsistent source lines fail closed.
- A later live-risk protocol explicitly names Hyperliquid symbol, notional caps, order size caps, position caps, loss cap, duration, post-only behavior, kill-switches, cancel-all evidence, and operator approval.

Until those gates pass, only public-only synchronized collection on `awsserver1` and local read-only analysis may continue.
