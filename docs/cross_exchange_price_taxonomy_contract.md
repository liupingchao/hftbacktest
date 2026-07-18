# Cross-Exchange Price Taxonomy Contract

Status: `0718T015` implementation contract

This contract defines the shared Binance-lead / Hyperliquid-lag pricing vocabulary. It applies to offline kernel fixtures, public shadow rows, and replay rows. It does not authorize live order submission.

## Versioned Config

The kernel consumes a validated `PricingConfigV1`; it does not accept an untyped pricing-parameter dictionary.

```json
{
  "schema_version": "pricing_config_v1",
  "normalization_stats_hash": "<sha256>",
  "expected_move_ticks_per_signal_z": 4.0,
  "base_half_spread_ticks": 0.5,
  "inventory_skew_ticks_at_max": 0.0,
  "max_position_btc": 0.01,
  "enable_microprice": false,
  "enable_inventory_skew": false,
  "enable_dynamic_spread": false,
  "enable_fill_feedback": false,
  "levels": 1
}
```

The config is serialized into fixture/shadow/replay manifests and its `pricing_config_hash` plus `normalization_stats_hash` are present on every decision row. A config hash is computed from canonical sorted JSON. The normalization hash is computed from the exact stats artifact, so replay and shadow cannot silently use different statistics.

`default_normalization_stats()` is an identity-stats helper for offline fixtures only. Production, shadow, and replay paths must supply the accepted normalization artifact and matching hash. Rolling normalization is outside this task.

## Price Fields

| Field | Meaning |
| --- | --- |
| `hl_mid_px` | `(best_bid + best_ask) / 2` from the decision-time Hyperliquid BBO. |
| `hl_micro_px` | BBO/top-N microprice only when bid/ask quantities are positive, marked as coming from the same coherent fresh snapshot, and not older than the microprice age bound. |
| `signal_score` | Normalized Binance-lead signal score. |
| `alpha_adjustment_ticks` | `signal_score * expected_move_ticks_per_signal_z`. |
| `forecast_mid_px` | `hl_micro_px` when valid, otherwise `hl_mid_px`, plus alpha adjustment in ticks. |
| `reservation_px` | Forecast midpoint after the configured inventory penalty. Inventory skew is disabled by default in Task 4. |
| `quote_bid_px` | `reservation_px - base_half_spread_ticks`, passed through the authoritative post-only helper. |
| `quote_ask_px` | `reservation_px + base_half_spread_ticks`, passed through the authoritative post-only helper. |

The final quotes are always normalized and strictly non-crossing. A crossing or invalid BBO is a market-data block, not an alpha classification.

## Signal and Eligibility

Normalization failure, missing features, stale signal state, invalid BBO/tick, incoherent snapshot, and post-only failure are fail-closed quote-eligibility gates.

The absolute signal threshold remains an audit classification:

- `confidence_bucket=above_threshold`
- `confidence_bucket=below_threshold`

`below_threshold` does not by itself set `action=block`. A valid market/risk/order-state decision still emits both bid and ask quote intents. The signal score can be zero; in that case the legacy directional `side` is `both`, while `quote_intents` still contains one buy and one sell intent.

The legacy `side`, `quote_px`, and `quote_intent` fields are retained for existing read-only markout consumers. New consumers must use `quote_bid_px`, `quote_ask_px`, and `quote_intents`.

`required_edge_ticks` and `edge_ticks` remain audit fields in Task 4. The former one-sided edge gate is not a quote-eligibility gate; risk, toxicity, order-state, maintenance, and post-only controls remain authoritative in later tasks.

## Microprice Rules

- BBO quantities must be positive and tied to one snapshot.
- Missing or non-positive quantities, or an expired snapshot, use `hl_mid_px` with `fair_base=mid_fallback`.
- An explicitly incoherent snapshot blocks the decision.
- The selected microprice source and fallback reason are recorded in `microprice_reason`.

## Evidence Boundary

Task 4 is offline-only. It changes the shared decision contract and deterministic artifacts, not live authorization, credentials, private endpoints, order/cancel behavior, or strategy promotion.
