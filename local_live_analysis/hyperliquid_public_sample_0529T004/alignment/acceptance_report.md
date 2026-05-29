# Hyperliquid Raw Alignment Acceptance Report

Task: `0529T004`

## Scope

- Read-only public market-data alignment only.
- No private keys, account endpoints, order submit/cancel, strategy live process, remote deploy, parameter search, tiny-live, or promotion.
- Existing local sample was used unless the run manifest says otherwise.

## Input

- Raw input: `/home/molly/project/hftbacktest/local_live_analysis/hyperliquid_public_sample_0529T004/raw.gz`
- Coins: `BTC`
- Channels: `{"l2Book": 222, "pong": 3, "subscriptionResponse": 2, "trades": 111}`

## Results

- Raw parse errors: `0`
- `l2Book` messages: `222`
- `trades` messages: `111`
- trade events: `418`
- subscription responses / acks: `2`
- recovery snapshots: `1`
- connection attempts: `1`
- reconnect count: `0`
- npz rows: `4279`
- event order validation: `passed`
- top-N coverage: `1.000000`
- synthetic decision join coverage: `1.000000`
- future joins: `0`
- missing joins: `0`
- join age p99 ms: `1114.135914`
- l2Book cadence p99 ms: `1317.185171`

## Classification

- `passes_pricing_research_market_view`
- reason: `market_view_and_recovery_evidence_present`

## Boundary Notes

- The top-N sidecar is built from Hyperliquid `l2Book` snapshots.
- The runner does not use Binance `U/u/pu`, `lastUpdateId`, or `bookTicker` semantics.
- Synthetic joins are market-data-only timing probes because no Hyperliquid strategy audit exists yet.
- Exact queue position, private fill lifecycle proof, strategy PnL, and live trading readiness remain out of scope.
