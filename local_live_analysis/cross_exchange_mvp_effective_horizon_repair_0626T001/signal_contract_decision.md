# Signal Acceptance Decision

Task: `0626T001`

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
- Horizon: target `1000ms` with `+/-250ms` tolerance; not frozen because strict near-target label coverage is insufficient.
- Side mapping: candidate `score > 0 -> buy`, `score < 0 -> sell`, not frozen.
- Freshness limit: candidate public source-age diagnostics only, not frozen.
- Edge formula: not frozen.

## Reasons

- strict 1000ms effective-age label coverage is insufficient
- xemm_0625_t002_utc15_a has 2 near-target rows out of 668 complete nominal rows
- xemm_0625_t002_utc15_b has 0 near-target rows out of 666 complete nominal rows
- xemm_0625_t002_utc16_c has 1 near-target rows out of 665 complete nominal rows

## Boundary

- Offline public-only no-submit.
- No live orders, credentials, private/account/order/cancel endpoints, live client initialization, remote refresh, final gate, or T008 ledger claim.
- No strategy/live behavior change, quote/cap relaxation, canary, default-on, or promotion.
