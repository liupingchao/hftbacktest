# 0627T001 Cross-Exchange Dataset Loading

This note explains how to load the QA-accepted `0627T001` Binance lead / Hyperliquid lag public dataset package.

The package is:

```text
local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/
```

Use paths relative to the current repository checkout. Some JSON manifests preserve absolute paths from the machine that generated the artifacts; those absolute paths are provenance, not the safest way to load the package in this workspace.

## Acceptance Status

`0627T001` is the current reusable three-window synchronized public dataset package for the cross-exchange MVP signal path.

Accepted facts:

- `sample_count=3`
- `all_samples_valid=true`
- `recommendation=sample_contract_ready_for_signal_acceptance`
- `t003_creation_unlocked=true`
- `complete_symmetric_context_row_count=10745`
- `valid_for_1000ms_signal_acceptance_row_count=10704`

The three sample IDs are:

- `xemm_0627_t001_hlfast_utc16_a`
- `xemm_0627_t001_hlfast_utc17_b`
- `xemm_0627_t001_hlfast_utc17_c`

Do not substitute `0702T001`: that later three-window package is explicitly invalid for signal acceptance.

## Package Files

Primary files:

- `sample_expansion_manifest.json`: package-level status, sample IDs, row counts, recommendation.
- `boundary_manifest.json`: public-only / offline / no-order boundary flags and label policy.
- `sample_quality_matrix.csv`: one row per sample with collection counts, join quality, and valid-row counts.
- `symmetric_edge_context_coverage.csv`: row-level dataset for research and signal acceptance.
- `effective_horizon_validity_matrix.csv`: per-sample 1000ms effective-horizon gate.
- `field_coverage_matrix.csv`: per-field coverage by sample.
- `regime_summary.csv`: public regime assignment by sample.
- `recommendation.md`: short human-readable package recommendation.

`symmetric_edge_context_coverage.csv` is the main data table.

## Quick Load

```python
from pathlib import Path
import json
import pandas as pd

repo = Path("/home/molly/project/hftbacktest")
pkg = repo / "local_live_analysis" / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"

manifest = json.loads((pkg / "sample_expansion_manifest.json").read_text())
boundary = json.loads((pkg / "boundary_manifest.json").read_text())
quality = pd.read_csv(pkg / "sample_quality_matrix.csv")
rows = pd.read_csv(pkg / "symmetric_edge_context_coverage.csv")

assert manifest["task_id"] == "0627T001"
assert manifest["all_samples_valid"] is True
assert manifest["recommendation"] == "sample_contract_ready_for_signal_acceptance"
assert boundary["boundary_flags"]["future_labels_not_decision_inputs"] is True

valid_1000ms = rows[
    (rows["valid_for_1000ms_signal_acceptance"] == True)
    & (rows["nominal_horizon_ms"] == 1000)
    & (rows["effective_future_age_ms"].between(1000, 1250))
].copy()

print(quality[["sample_id", "observed_regime", "valid_for_1000ms_signal_acceptance_rows"]])
print(len(valid_1000ms))
```

Expected valid row count:

```text
10704
```

## Recommended Loader Helper

Use this when you want the package checks and filtered row table together:

```python
from pathlib import Path
import json
import pandas as pd

def load_0627t001_package(repo_root: str | Path):
    repo_root = Path(repo_root)
    pkg = repo_root / "local_live_analysis" / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"

    manifest = json.loads((pkg / "sample_expansion_manifest.json").read_text())
    boundary = json.loads((pkg / "boundary_manifest.json").read_text())
    quality = pd.read_csv(pkg / "sample_quality_matrix.csv")
    rows = pd.read_csv(pkg / "symmetric_edge_context_coverage.csv")

    if manifest.get("task_id") != "0627T001":
        raise ValueError(f"unexpected task_id: {manifest.get('task_id')}")
    if not manifest.get("all_samples_valid"):
        raise ValueError("0627T001 package is not marked all_samples_valid")
    if manifest.get("recommendation") != "sample_contract_ready_for_signal_acceptance":
        raise ValueError(f"unexpected recommendation: {manifest.get('recommendation')}")
    if not boundary["boundary_flags"].get("public_market_data_only"):
        raise ValueError("package boundary is not public_market_data_only")
    if not boundary["boundary_flags"].get("future_labels_not_decision_inputs"):
        raise ValueError("future label boundary is not enforced")

    valid = rows[
        (rows["valid_for_1000ms_signal_acceptance"] == True)
        & (rows["complete_context"] == True)
        & (rows["context_fields_complete"] == True)
        & (rows["has_future_label"] == True)
        & (rows["near_target_1000ms"] == True)
        & (rows["effective_horizon_valid"] == True)
        & (rows["nominal_horizon_ms"] == 1000)
        & (rows["effective_future_age_ms"].between(1000, 1250))
    ].copy()

    expected = manifest["valid_for_1000ms_signal_acceptance_row_count"]
    if len(valid) != expected:
        raise ValueError(f"valid row count mismatch: {len(valid)} != {expected}")

    return {
        "package_dir": pkg,
        "manifest": manifest,
        "boundary": boundary,
        "quality": quality,
        "rows": rows,
        "valid_1000ms_rows": valid,
    }
```

## Field Groups

Decision-time identifiers and timing:

- `sample_id`
- `observed_regime`
- `source_row_index`
- `future_row_index`
- `hyperliquid_decision_ts`
- `hyperliquid_l2book_local_ts`
- `hyperliquid_l2book_event_ts`
- `binance_local_ts`
- `binance_exch_ts`
- `binance_source_age_ms`
- `hyperliquid_join_age_ms`
- `nominal_horizon_ms`
- `effective_future_age_ms`
- `future_hyperliquid_decision_ts`

Decision-time Binance lead inputs:

- `input_binance_top5_imbalance`
- `input_binance_microprice_minus_mid_ticks`
- `input_binance_mid_move_ticks_from_prev`
- `input_binance_top5_bid_qty`
- `binance_mid_px`
- `binance_top5_microprice_px`
- `binance_bid_top5_px`
- `binance_ask_top5_px`
- `binance_bid_top5_qtys`
- `binance_ask_top5_qtys`

Decision-time Hyperliquid context:

- `hyperliquid_current_bid_px`
- `hyperliquid_current_ask_px`
- `hyperliquid_buy_touch_quote_px`
- `hyperliquid_sell_touch_quote_px`
- `hyperliquid_mid_px`
- `hyperliquid_top5_microprice_px`
- `hyperliquid_spread_ticks`
- `hyperliquid_bid_top5_px`
- `hyperliquid_ask_top5_px`
- `hyperliquid_bid_top5_qtys`
- `hyperliquid_ask_top5_qtys`
- `hyperliquid_top5_imbalance`
- `hyperliquid_microprice_minus_mid_ticks`
- `hyperliquid_context_quality`
- `basis_mid_ticks`

Future labels, not decision inputs:

- `future_hyperliquid_mid_px`
- `future_hyperliquid_top5_microprice_px`
- `hyperliquid_future_mid_move_ticks`
- `hyperliquid_future_microprice_minus_mid_change_ticks`
- `label_row_quality`

Gate columns:

- `has_future_label`
- `context_fields_complete`
- `near_target_1000ms`
- `effective_horizon_valid`
- `valid_for_1000ms_signal_acceptance`
- `effective_horizon_bucket`
- `complete_context`

## Per-Sample Quality

The accepted package has these per-window row counts:

| sample_id | regime | complete_context_rows_1000ms | valid_for_1000ms_signal_acceptance_rows |
| --- | --- | ---: | ---: |
| `xemm_0627_t001_hlfast_utc16_a` | `high_activity_liquidity` | 3591 | 3587 |
| `xemm_0627_t001_hlfast_utc17_b` | `normal_activity_liquidity` | 3581 | 3567 |
| `xemm_0627_t001_hlfast_utc17_c` | `low_activity_liquidity` | 3573 | 3550 |

All three samples have:

- Binance raw checksum match: true
- Hyperliquid raw checksum match: true
- Binance reconnect count: `0`
- Hyperliquid reconnect count: `0`
- Future Binance join count: `0`
- Missing Binance join count: `0`
- Stale Binance join count: `0`

## Reproducing The Existing Signal Acceptance Input Gate

The `0625T003` signal acceptance runner consumes this package directly:

```bash
python examples/hyperliquid/cross_exchange_signal_acceptance.py \
  --input-dir local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001 \
  --output-dir /tmp/cross_exchange_signal_acceptance_0627T001_check
```

This should generate a `signal_contract_accepted_for_shadow` recommendation if the local artifacts are intact.

## Common Mistakes

- Do not load `0702T001` as a replacement. Its recommendation is `sample_collection_invalid`.
- Do not use `future_hyperliquid_*` or `hyperliquid_future_*` columns as model inputs. They are labels only.
- Do not read the absolute paths inside `sample_expansion_manifest.json` as local truth. Use the relative files in the package directory.
- Do not assume a side mapping from `0627T001`. This package preserves both Hyperliquid touch alternatives and does not select a quote side.
- Do not treat this package as execution, fill, fee, PnL, queue-priority, live-order, canary, or promotion evidence. It is public market-data research input only.
