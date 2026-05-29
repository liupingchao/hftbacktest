# Hyperliquid Raw Alignment Acceptance Report

Task: `0529T003`

## Scope

- Read-only public market-data alignment only.
- No private keys, account endpoints, order submit/cancel, strategy live process, remote deploy, parameter search, tiny-live, or promotion.
- Existing local sample was used unless the run manifest says otherwise.

## Input

- Raw input: `/home/molly/project/hftbacktest/examples/hyperliquid/btcusd_20250126.gz`
- Coins: `BTC`
- Channels: `{"l2Book": 46, "trades": 9}`

## Results

- Raw parse errors: `0`
- `l2Book` messages: `46`
- `trades` messages: `9`
- trade events: `64`
- npz rows: `413`
- event order validation: `passed`
- top-N coverage: `1.000000`
- synthetic decision join coverage: `1.000000`
- future joins: `0`
- missing joins: `0`
- join age p99 ms: `623.489500`
- l2Book cadence p99 ms: `832.514760`

## Classification

- `limited_pricing_research`
- reason: `market_view_good_but_session_or_recovery_evidence_missing`

## Boundary Notes

- The top-N sidecar is built from Hyperliquid `l2Book` snapshots.
- The runner does not use Binance `U/u/pu`, `lastUpdateId`, or `bookTicker` semantics.
- Synthetic joins are market-data-only timing probes because no Hyperliquid strategy audit exists yet.
- Exact queue position, private fill lifecycle proof, strategy PnL, and live trading readiness remain out of scope.
