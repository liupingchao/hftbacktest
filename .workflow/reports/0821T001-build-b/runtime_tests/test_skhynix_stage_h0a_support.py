from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

try:
    import skhynix_stage_h0a_support as support
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import skhynix_stage_h0a_support as support


def _segment(*, valid: bool = True) -> support.Segment:
    return support.Segment(
        session_id="jul30",
        segment_id="segment_0001",
        connection_epoch_id="segment_0001:epoch_0",
        start_ns=1_000_000,
        end_ns=301_000_000,
        bbo=(
            support.BboObservation(
                local_ts_ns=15_000_000,
                exchange_ts_ns=14_000_000,
                valid=valid,
            ),
            support.BboObservation(
                local_ts_ns=105_000_000,
                exchange_ts_ns=104_000_000,
                valid=True,
            ),
        ),
    )


def test_absolute_grid_uses_half_open_segment() -> None:
    first, last, count = support.segment_grid_bounds(
        1_000_001, 31_000_001
    )
    assert first == 10_000_000
    assert last == 30_000_000
    assert count == 3


def test_identification_classes_are_mutually_exclusive() -> None:
    binary = support.classify_support(
        reference_available=True,
        reference_valid=True,
        target_inside_segment=True,
    )
    interval = support.classify_support(
        reference_available=True,
        reference_valid=True,
        target_inside_segment=True,
        endpoint_closed=False,
        interval_bounds_supported=True,
    )
    boundary = support.classify_support(
        reference_available=True,
        reference_valid=True,
        target_inside_segment=False,
    )
    assert binary.identification_class == "binary_identification_supported"
    assert binary.binary_endpoint_identification_supported is True
    assert interval.identification_class == (
        "interval_likelihood_only_supported"
    )
    assert interval.binary_endpoint_identification_supported is False
    assert interval.interval_likelihood_eligible is True
    assert boundary.identification_class == "right_censored_segment"
    assert boundary.interval_likelihood_eligible is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "reference_available": False,
                "reference_valid": False,
                "target_inside_segment": True,
            },
            "reference_quote_unavailable",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": False,
                "target_inside_segment": True,
            },
            "invalid_quote_state",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": False,
            },
            "epoch_censored",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "source_gap": True,
            },
            "source_gap_censored",
        ),
    ],
)
def test_classifier_negative_topologies(
    kwargs: dict[str, bool], expected: str
) -> None:
    assert support.classify_support(**kwargs).identification_class == expected


def test_valid_price_mutation_does_not_change_support_commitment() -> None:
    segment = _segment(valid=True)
    facts = support.classify_support(
        reference_available=True,
        reference_valid=True,
        target_inside_segment=True,
    )
    first = support._commitment_row_bytes(
        segment=segment,
        grid_ts_ns=10_000_000,
        horizon_ms=50,
        facts=facts,
        block_id="",
    )
    mutated_segment = support.Segment(
        session_id=segment.session_id,
        segment_id=segment.segment_id,
        connection_epoch_id=segment.connection_epoch_id,
        start_ns=segment.start_ns,
        end_ns=segment.end_ns,
        bbo=(
            support.BboObservation(15_000_000, 14_000_000, True),
            support.BboObservation(105_000_000, 104_000_000, True),
        ),
    )
    second = support._commitment_row_bytes(
        segment=mutated_segment,
        grid_ts_ns=10_000_000,
        horizon_ms=50,
        facts=facts,
        block_id="",
    )
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    assert b"bid" not in first
    assert b"ask" not in first


def test_project_segment_preserves_opening_and_trailing_censors() -> None:
    rows, commitments, counts = support.project_segment(_segment())
    assert len(rows) == len(support.HORIZONS_MS)
    assert len(commitments) == len(support.HORIZONS_MS)
    assert counts[50]["reference_quote_unavailable"] > 0
    assert counts[50]["right_censored_segment"] > 0
    assert counts[50]["binary_identification_supported"] > 0
    assert counts[50]["interval_likelihood_only_supported"] == 0


def test_ratio_text_is_decimal_and_zero_denominator_is_explicit() -> None:
    assert support.ratio_text(1, 3) == "0.333333333333"
    assert support.ratio_text(0, 0) == ""


def test_exact_csv_rejects_reordered_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("b,a\n1,2\n", encoding="utf-8")
    with pytest.raises(support.H0AError) as error:
        list(support._read_exact_csv(path, ("a", "b")))
    assert error.value.code == "H0A_SOURCE_SCHEMA_MISMATCH"
