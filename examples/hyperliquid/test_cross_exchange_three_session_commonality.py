import csv
import gzip
import json

import numpy as np
import polars as pl

import examples.hyperliquid.cross_exchange_three_session_commonality as commonality
from examples.hyperliquid.cross_exchange_three_session_commonality import (
    _design_matrix,
    _deterministic_medoids,
    _effect_retention,
    _formal_primary_gate_status,
    _ols_from_sufficient,
    _search_asof_indices,
    add_first_after_outcomes,
    benjamini_hochberg,
    canonical_encode,
    canonical_hash,
    finalize_full_multitrack_surrogates,
    unbiased_index,
)


def test_canonical_encoding_and_sampler_are_deterministic():
    fields = ["bootstrap-v1", "a" * 64, "jul30", "segment_0001", "60000", "0", "0"]
    assert canonical_encode(fields).startswith(b"HFTBT-COMMONALITY-HASH-V1\0")
    assert canonical_hash(fields) == canonical_hash(fields)
    assert unbiased_index(fields, 17) == unbiased_index(fields, 17)
    assert 0 <= unbiased_index(fields, 17) < 17


def test_asof_indices_never_select_future_rows():
    timestamps = np.asarray([10, 20, 30, 40], dtype=np.int64)
    targets = np.asarray([9, 10, 19, 20, 41], dtype=np.int64)
    indices = _search_asof_indices(timestamps, targets)
    assert indices.tolist() == [-1, 0, 0, 1, 3]
    valid = indices >= 0
    assert np.all(timestamps[indices[valid]] <= targets[valid])


def test_first_after_outcome_is_strictly_after_target():
    frame = pl.DataFrame(
        {
            "decision_ts_ns": [0, 100_000_000, 101_000_000, 300_000_000],
            "quality_eligible": [True, True, True, True],
            "d_bh_q": [1.0, 0.8, 0.5, 0.0],
            "d_hb_q": [-2.0, -1.8, -1.5, -1.0],
            "reference_mid_q": [100.0, 100.0, 100.0, 100.0],
            "binance_bid_q": [101.0, 100.8, 100.5, 100.0],
            "binance_ask_q": [102.0, 101.8, 101.5, 101.0],
            "hyperliquid_bid_q": [100.0, 100.0, 100.0, 100.0],
            "hyperliquid_ask_q": [100.0, 100.0, 100.0, 100.0],
        }
    )
    result = add_first_after_outcomes(frame)
    assert result["h100_first_after_delay_ms"][0] == 1.0
    assert result["d_bh_h100_first_after_total_closure_bps"][0] == 50.0


def test_bbo_spread_and_closure_identities():
    b_bid = 100.0
    b_ask = 100.2
    h_bid = 99.8
    h_ask = 100.1
    d_bh = b_bid - h_ask
    d_hb = h_bid - b_ask
    assert np.isclose(d_bh + d_hb, -(b_ask - b_bid) - (h_ask - h_bid))

    b_bid_future = 99.9
    h_ask_future = 100.2
    total = d_bh - (b_bid_future - h_ask_future)
    b_leg = b_bid - b_bid_future
    h_leg = h_ask_future - h_ask
    assert np.isclose(total, b_leg + h_leg)


def test_ols_sufficient_statistics_match_direct_fit():
    x = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.asarray([1.0, 3.0, 5.0, 7.0])
    beta, ok = _ols_from_sufficient(x.T @ x, x.T @ y)
    assert ok
    assert np.allclose(beta, np.linalg.lstsq(x, y, rcond=None)[0])


def test_ols_rejects_rank_deficient_design():
    xtx = np.asarray([[2.0, 2.0], [2.0, 2.0]])
    beta, ok = _ols_from_sufficient(xtx, np.asarray([1.0, 1.0]))
    assert not ok
    assert np.isnan(beta).all()


def test_adjusted_and_unadjusted_designs_share_primary_predictors():
    frame = pl.DataFrame(
        {
            "segment_id": ["segment_0001", "segment_0002"],
            "d_bh_level_z": [1.0, 2.0],
            "d_bh_change_z_100ms": [0.5, -0.5],
            "binance_bid_qty": [2.0, 3.0],
            "hyperliquid_ask_qty": [1.0, 4.0],
            "binance_spread_bps": [0.1, 0.2],
            "hyperliquid_spread_bps": [0.3, 0.4],
            "binance_age_ms": [1.0, 2.0],
            "hyperliquid_age_ms": [3.0, 4.0],
            "basis_residual_bps": [0.2, -0.2],
            "binance_volatility_60s_bps": [1.5, 2.5],
        }
    )
    adjusted = _design_matrix(
        frame, "d_bh", ["segment_0001", "segment_0002"], adjusted=True
    )
    unadjusted = _design_matrix(
        frame, "d_bh", ["segment_0001", "segment_0002"], adjusted=False
    )
    assert adjusted.shape == (2, 11)
    assert unadjusted.shape == (2, 4)
    assert np.array_equal(adjusted[:, :3], unadjusted[:, :3])
    assert np.array_equal(adjusted[:, -1], unadjusted[:, -1])


def test_effect_retention_requires_positive_sign_and_half_magnitude():
    assert _effect_retention(0.6, 1.0) == (0.6, True)
    assert _effect_retention(0.4, 1.0) == (0.4, False)
    assert _effect_retention(-0.8, 1.0) == (0.8, False)
    ratio, passes = _effect_retention(float("nan"), 1.0)
    assert np.isnan(ratio)
    assert not passes


def test_formal_primary_gate_requires_every_retention_and_first_after_check():
    rows = [
        {
            "beta": 1.0,
            "bootstrap_lower": 0.1 if index < 2 else 0.0,
            "bootstrap_upper": 1.5,
            "practical_effect_pass": True,
            "bootstrap_sign_stability": 0.9,
            "effect_retention_pass": True,
            "first_after_gate_pass": True,
        }
        for index in range(3)
    ]
    assert _formal_primary_gate_status(rows)["c2_pass"]
    rows[2]["effect_retention_pass"] = False
    assert not _formal_primary_gate_status(rows)["c2_pass"]
    rows[2]["effect_retention_pass"] = True
    rows[1]["first_after_gate_pass"] = False
    assert not _formal_primary_gate_status(rows)["c2_pass"]


def test_bh_is_monotone_in_sorted_p_values():
    rows = [{"p": value} for value in [0.04, 0.001, 0.02, 1.0]]
    benjamini_hochberg(rows, "p", "q")
    ordered = sorted((row["p"], row["q"]) for row in rows)
    assert all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:]))


def test_deterministic_medoids_are_real_unique_rows():
    matrix = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [10.0, 0.0]]
    )
    medoids = _deterministic_medoids(matrix, 3)
    assert len(set(medoids.tolist())) == 3
    assert all(0 <= index < len(matrix) for index in medoids)


def test_full_multitrack_finalizer_reconciles_every_hypothesis_slot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(commonality, "PRIMARY_SURROGATE_COUNT", 2)
    output_root = tmp_path / "output"
    mechanism = output_root / "mechanism"
    mechanism.mkdir(parents=True)

    primary_rows = []
    secondary_rows = []
    for index in range(60):
        key = f"primary-{index:02d}"
        secondary_key = f"secondary-{index:02d}"
        for session_id in ("jul30", "aug03", "aug04"):
            primary_rows.append(
                {
                    "session_id": session_id,
                    "direction": "d_bh",
                    "horizon_ms": 100,
                    "outcome": "hyperliquid_leg_bps",
                    "hypothesis_key": key,
                    "predictor": "level",
                    "beta": 1.0,
                    "bootstrap_lower": 0.1,
                    "bootstrap_upper": 1.5,
                    "practical_effect_pass": True,
                    "bootstrap_sign_stability": 0.9,
                    "effect_retention_pass": True,
                    "first_after_gate_pass": True,
                }
            )
            secondary_rows.append(
                {
                    "session_id": session_id,
                    "hypothesis_key": secondary_key,
                    "beta": 1.0,
                }
            )

    def write_gzip(path, rows):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_gzip(
        mechanism / "bbo_dislocation_response_by_session.csv.gz",
        primary_rows,
    )
    write_gzip(
        mechanism / "bbo_dislocation_secondary_liquidity.csv.gz",
        secondary_rows,
    )
    surrogate_rows = []
    quality_rows = []
    for surrogate_id in range(2):
        for index in range(60):
            for family, prefix in (
                ("primary_bbo", "primary"),
                ("secondary_fast_l2", "secondary"),
            ):
                surrogate_rows.append(
                    {
                        "surrogate_id": surrogate_id,
                        "lag_ms": 60_000 + surrogate_id,
                        "family": family,
                        "hypothesis_key": f"{prefix}-{index:02d}",
                        "beta": 0.5,
                        "quality_fail": False,
                        "fit_ok": True,
                    }
                )
        for track in commonality.FULL_HYPERLIQUID_SURROGATE_TRACKS:
            quality_rows.append(
                {
                    "surrogate_id": surrogate_id,
                    "lag_ms": 60_000 + surrogate_id,
                    "track": track,
                    "retained_count": 10,
                    "native_order_preserved": True,
                    "no_wrap": True,
                    "qualification_gate_applied": True,
                    "queried_count": 10,
                    "missing_count": (
                        1
                        if surrogate_id == 0
                        and track == "main_all_mids"
                        else 0
                    ),
                    "qualification_failure_count": (
                        1
                        if surrogate_id == 0
                        and track == "main_all_mids"
                        else 0
                    ),
                    "queried_unique_state_count": 2,
                    "queried_state_checksum_sha256": "a" * 64,
                }
            )
    surrogate_path = mechanism / "full.csv.gz"
    quality_path = mechanism / "quality.csv.gz"
    write_gzip(surrogate_path, surrogate_rows)
    write_gzip(quality_path, quality_rows)
    (mechanism / "aug04_full_multitrack_surrogate_input.npz").write_bytes(
        b"input"
    )
    runtime_sources = []
    for index, role in enumerate(
        (
            "multitrack_worker",
            "secondary_family",
            "commonality_dependency",
        )
    ):
        archive_path = f"runtime_source/source_{index}.py"
        archive = output_root / archive_path
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(f"# {role}\n", encoding="utf-8")
        runtime_sources.append(
            {
                "role": role,
                "archive_path": archive_path,
                "sha256": commonality.sha256_file(archive),
                "size_bytes": archive.stat().st_size,
            }
        )
    input_path = mechanism / "aug04_full_multitrack_surrogate_input.npz"
    input_manifest_path = input_path.with_suffix(
        input_path.suffix + ".manifest.json"
    )
    input_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "full_multitrack_surrogate_input_v2",
                "runtime_sources": runtime_sources,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (
        mechanism / "aug04_full_multitrack_run_manifest.json"
    ).write_text(
        json.dumps(
            {
                "schema_version": (
                    "directional_bbo_full_multitrack_lag_surrogate_v2"
                ),
                "input_sha256": commonality.sha256_file(input_path),
                "output_sha256": commonality.sha256_file(surrogate_path),
                "quality_output_sha256": commonality.sha256_file(quality_path),
                "runtime_source_sha256": {
                    row["role"]: row["sha256"] for row in runtime_sources
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "bbo_effect_retention_manifest.json",
        "bbo_first_after_target_manifest.json",
        "bbo_secondary_liquidity_manifest.json",
    ):
        (mechanism / name).write_text(
            json.dumps({"passes": True}) + "\n", encoding="utf-8"
        )
    (mechanism / "surrogate_manifest.json").write_text(
        json.dumps({"schema_version": "bbo-only"}) + "\n",
        encoding="utf-8",
    )
    (mechanism / "bbo_dislocation_manifest.json").write_text(
        json.dumps({}) + "\n", encoding="utf-8"
    )
    (output_root / "commonality_manifest.json").write_text(
        json.dumps({"counts": {}}) + "\n", encoding="utf-8"
    )

    manifest = finalize_full_multitrack_surrogates(
        output_root, surrogate_path, quality_path
    )

    assert manifest["passes"]
    assert manifest["formal_tier_counts"] == {"C2": 60}
    assert manifest["primary_invalid_slot_count"] == 0
    assert manifest["secondary_invalid_slot_count"] == 0
    assert manifest["same_lag_tracks"][0] == "bbo"
    assert manifest["runtime_source_closure"]
    assert manifest["track_missing_state_count"] == 1
    assert manifest["track_qualification_exclusion_count"] == 1
    with gzip.open(
        mechanism / "bbo_dislocation_response_by_session.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:
        finalized = list(csv.DictReader(fh))
    aug04 = next(row for row in finalized if row["session_id"] == "aug04")
    assert aug04["full_multitrack_surrogate_count"] == "2"
    assert aug04["full_multitrack_surrogate_invalid_count"] == "0"
    assert aug04["statistical_tier"] == "C2"
