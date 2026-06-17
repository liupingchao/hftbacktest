# Hyperliquid Tiny-Live Signal / Quote Policy Protocol

Task: `0617T004`

## Scope

This is a read-only protocol for the Binance-lead / Hyperliquid-lag maker path.
It defines the minimum policy needed before any later `0616T008` live window.
It does not authorize live execution.

## Evidence Basis

Accepted inputs used as policy evidence:

- `0601T004` data input schema
- `0601T005` pricing-signal recommendation
- `0604T003` canonical multi-sample pricing-signal robustness
- `0617T003` approved tiny-live caps

The local evidence supports directional lead/lag structure, but it does not
defensibly calibrate a final absolute live trigger threshold.

## Signal

Primary signal:

- `basis_mid = binance_mid - hyperliquid_mid`
- `basis_mid_ticks = basis_mid / hyperliquid_tick_size`

Decision inputs:

- Binance public mid
- Hyperliquid public mid
- local/as-of timestamps
- join age
- spread/tick size
- public top-of-book state when available

## Threshold Policy

- Signal persistence is required over multiple observations before any quote intent.
- Live threshold status: `blocked_for_live_execution`
- Reason: accepted artifacts show stable directional structure, but they do not
  provide a defensible absolute live cut point for `basis_mid_ticks`.
- Result: `0617T004` recommends a read-only threshold calibration task before
  any `0616T008` live execution.

## Side Mapping

- Positive eligible signal -> Hyperliquid maker buy intent only
- Negative eligible signal -> Hyperliquid maker sell intent only
- No discretionary/manual direction override
- No taker fallback

## Quote Policy

- Maker-only / post-only only
- No crossing Hyperliquid BBO
- No taker fallback if post-only would cross
- Quote distance must stay bounded relative to Hyperliquid BBO/mid
- Quote intent is invalid if book state is stale, missing, or outside join-age
  tolerance

## Size and Cap Policy

Approved `0617T003` caps apply only to the later separately dispatched
`0616T008`:

- max order size: `0.01 BTC`
- max position: `0.04 BTC`
- max loss: `30 USDC`
- max notional: `3000 USDC`
- duration: `10 minutes`
- host: `awsserver1`
- maker-only / post-only: `true`

Reduce-side-only behavior is required when near or at the position limit.

## Cancel / Stop Policy

Cancel or stop on:

- signal decay
- stale join
- public data gap
- post-only reject
- cap breach
- max loss proximity
- duration expiry
- operator stop

## Required Runtime Audit Fields

- signal inputs
- timestamps
- basis ticks
- threshold state
- side intent
- quote price
- quote distance
- post-only flag
- size
- position before / after
- cap checks
- cancel reason
- reject reason
- shutdown evidence

## Final Recommendation

- `hyperliquid_tiny_live_signal_quote_policy_needs_threshold_calibration`

This means the protocol is defined, but live execution must wait for a
separately scoped read-only replay / threshold calibration task.
