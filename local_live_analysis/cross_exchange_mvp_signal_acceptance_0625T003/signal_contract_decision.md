# T003 Signal Acceptance Decision

Task: `0625T003`

## Recommendation

- `signal_contract_needs_repair`
- T004 creation unlocked: `false`

## Fixed Boundary

- Train samples: `xemm_0625_t002_utc15_a`
- Evaluation samples: `xemm_0625_t002_utc15_b,xemm_0625_t002_utc16_c`
- Normalization and feature weights are fitted on train rows only.
- Future Hyperliquid values are labels only and are not used in score generation.

## Contract Status

- Feature allowlist: candidate only, not frozen for shadow.
- Horizon: nominal `1000ms` is not frozen because accepted labels are effectively around `5000ms` at median.
- Side mapping: candidate `score > 0 -> buy`, `score < 0 -> sell`, not frozen.
- Freshness limit: candidate public source-age diagnostics only, not frozen.
- Edge formula: not frozen.

## Reasons

- out-of-sample direction and signed markout are positive
- nominal 1000ms labels have effective median age around 5000ms, so the MVP shadow horizon cannot be frozen yet

## Boundary

- Offline public-only no-submit.
- No live orders, credentials, private/account/order/cancel endpoints, live client initialization, remote refresh, final gate, or T008 ledger claim.
- No strategy/live behavior change, quote/cap relaxation, canary, default-on, or promotion.
