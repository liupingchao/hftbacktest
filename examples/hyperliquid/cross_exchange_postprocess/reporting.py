"""Human-readable and compact machine-readable pipeline reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import atomic_write_json, atomic_write_text


def build_quality_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = manifest.get("stages", {})
    r0 = stages.get("r0", {}).get("summary", {})
    r1 = stages.get("r1", {}).get("summary", {})
    basis = stages.get("basis_dislocation", {}).get("summary", {})
    return {
        "schema_version": "cross_exchange_postprocess_quality_summary_v1",
        "pipeline_status": manifest.get("status"),
        "passes": manifest.get("passes") is True,
        "profile": manifest.get("profile"),
        "campaign_id": manifest.get("campaign_id"),
        "symbol_profile": manifest.get("symbol_profile"),
        "source_immutable": manifest.get("source_immutable"),
        "stage_status": {
            stage_id: stage.get("status")
            for stage_id, stage in stages.items()
        },
        "aggregate_counts": r0.get("aggregate_counts", {}),
        "accepted_primary_horizons_ms": r1.get(
            "accepted_primary_horizons_ms", []
        ),
        "horizon_mask_exclusion_count": r1.get(
            "horizon_mask_exclusion_count", 0
        ),
        "cross_epoch_label_count": r1.get("cross_epoch_label_count", 0),
        "basis_aggregate_counts": basis.get("aggregate_counts", {}),
        "capability_matrix": manifest.get("capability_matrix", {}),
        "claim_boundaries": manifest.get("claim_boundaries", {}),
        "failures": manifest.get("failures", []),
    }


def render_dataset_report(manifest: dict[str, Any]) -> str:
    stages = manifest.get("stages", {})
    r0 = stages.get("r0", {}).get("summary", {})
    r1 = stages.get("r1", {}).get("summary", {})
    basis = stages.get("basis_dislocation", {}).get("summary", {})
    capabilities = manifest.get("capability_matrix", {})
    lines = [
        "# Cross-Exchange Dataset Report",
        "",
        f"- Status: `{'PASS' if manifest.get('passes') is True else 'FAIL'}`",
        f"- Pipeline profile: `{manifest.get('profile', '')}`",
        f"- Campaign: `{manifest.get('campaign_id', '')}`",
        f"- Symbol profile: `{manifest.get('symbol_profile', '')}`",
        f"- Source campaign immutable: `{manifest.get('source_immutable')}`",
        f"- Stage reuse count: `{manifest.get('reused_stage_count', 0)}`",
        "",
        "## Stage Status",
        "",
    ]
    for stage_id, stage in stages.items():
        lines.append(
            f"- `{stage_id}`: `{stage.get('status', 'unknown')}` "
            f"(reused=`{stage.get('reused', False)}`)"
        )
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- Timeline rows: `{r0.get('aggregate_counts', {}).get('timeline_rows', 0)}`",
            f"- Binance hot rows: `{r0.get('aggregate_counts', {}).get('binance_hot_rows', 0)}`",
            f"- Hyperliquid hot rows: `{r0.get('aggregate_counts', {}).get('hyperliquid_hot_rows', 0)}`",
            f"- Hyperliquid auxiliary rows: `{r0.get('aggregate_counts', {}).get('hyperliquid_auxiliary_rows', 0)}`",
            f"- Mask rows: `{r0.get('aggregate_counts', {}).get('mask_rows', 0)}`",
            f"- Accepted horizons: `{r1.get('accepted_primary_horizons_ms', [])}` ms",
            f"- Horizon-mask exclusions: `{r1.get('horizon_mask_exclusion_count', 0)}`",
            f"- Cross-epoch labels: `{r1.get('cross_epoch_label_count', 0)}`",
        ]
    )
    if basis:
        basis_counts = basis.get("aggregate_counts", {})
        lines.extend(
            [
                "",
                "## Basis State",
                "",
                f"- State rows: `{basis_counts.get('state_rows', 0)}`",
                f"- Book-eligible rows: `{basis_counts.get('book_eligible_rows', 0)}`",
                f"- Feature-eligible rows: `{basis_counts.get('feature_eligible_rows', 0)}`",
                f"- Positive Binance-bid/Hyperliquid-ask rows: `{basis_counts.get('d_bh_positive_rows', 0)}`",
                f"- Positive Hyperliquid-bid/Binance-ask rows: `{basis_counts.get('d_hb_positive_rows', 0)}`",
                f"- Join rule: `{basis.get('join_rule', '')}`",
                f"- Rolling window closure: `{basis.get('rolling_closed', '')}`",
            ]
        )
    lines.extend(["", "## Capabilities", ""])
    for capability, status in capabilities.items():
        lines.append(f"- `{capability}`: `{status}`")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Public L2 does not establish L3/L4 queue position or exact maker fills.",
            "- Receipt-time precedence does not establish causal Binance leadership.",
            "- This package does not establish executable arbitrage, account PnL, or live promotion.",
            "- Additional live collection remains explicitly authorization-gated.",
            "",
        ]
    )
    if manifest.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in manifest["failures"])
        lines.append("")
    return "\n".join(lines)


def write_reports(output_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(
        output_dir / "quality_summary.json",
        build_quality_summary(manifest),
    )
    atomic_write_text(
        output_dir / "dataset_report.md",
        render_dataset_report(manifest),
    )
