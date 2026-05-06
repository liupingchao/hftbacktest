from __future__ import annotations

from pathlib import Path

from audit_schema import AUDIT_FIELDS
from validate_audit import validate


def _write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(",".join(AUDIT_FIELDS) + "\n")
    with path.open("a") as f:
        for row in rows:
            merged = {field: "0" for field in AUDIT_FIELDS}
            merged.update(row)
            f.write(",".join(str(merged[field]) for field in AUDIT_FIELDS) + "\n")


def test_validate_audit_formula_checks_only_decision_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    _write_audit(
        audit,
        [
            {
                "event_type": "decision",
                "best_bid": "100",
                "best_ask": "101",
                "mid": "100.5",
                "spread_bps": str((1.0 / 100.5) * 1e4),
                "inventory_score": "1.0",
                "vol_bps": "0.0",
            },
            {
                "event_type": "fill",
                "best_bid": "100",
                "best_ask": "101",
                "mid": "100.5",
                "spread_bps": "9999",
                "inventory_score": "1.0",
                "vol_bps": "0.0",
            },
        ],
    )

    report = validate(audit)

    assert report["rows"] == 2
    assert report["decision_rows"] == 1
    assert report["bad_spread_formula"] == 0
