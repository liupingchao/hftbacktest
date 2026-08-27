from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

try:
    import skhynix_phase_alignment_track_a as track_a
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import skhynix_phase_alignment_track_a as track_a


def _write_raw(path: Path, *, gap: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 1_800_000_000_000_000_000
    bids = [[f"{100 - level * 0.01:.2f}", f"{level + 1:.2f}"] for level in range(5)]
    asks = [
        [f"{100.01 + level * 0.01:.2f}", f"{level + 1.5:.2f}"]
        for level in range(5)
    ]
    messages: list[tuple[int, dict[str, object]]] = [
        (
            base,
            {
                "lastUpdateId": 100,
                "bids": bids,
                "asks": asks,
                "E": 1_800_000_000_000,
            },
        )
    ]
    previous = 100
    for index in range(1, 41):
        update_id = 100 + index
        previous_id = previous + 7 if gap and index == 12 else previous
        messages.append(
            (
                base + index * 50_000_000,
                {
                    "stream": "skhynixusdt@depth@0ms",
                    "data": {
                        "e": "depthUpdate",
                        "s": track_a.SYMBOL,
                        "E": 1_800_000_000_000 + index * 50,
                        "T": 1_800_000_000_000 + index * 50,
                        "U": update_id,
                        "u": update_id,
                        "pu": previous_id,
                        "b": [["100.00", f"{1 + (index % 5) * 0.1:.2f}"]],
                        "a": [["100.01", f"{1.5 + (index % 3) * 0.1:.2f}"]],
                    },
                },
            )
        )
        messages.append(
            (
                base + index * 50_000_000 + 10_000,
                {
                    "stream": "skhynixusdt@trade",
                    "data": {
                        "e": "trade",
                        "s": track_a.SYMBOL,
                        "q": "0.25",
                        "p": "100.00",
                        "m": bool(index % 2),
                    },
                },
            )
        )
        previous = update_id
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
            for timestamp, message in messages:
                fh.write(
                    f"{timestamp} ".encode()
                    + json.dumps(message, sort_keys=True).encode()
                    + b"\n"
                )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(path: Path, digest: str) -> track_a.Capture:
    return track_a.Capture(
        capture_id="fixture",
        research_date="2026-07-29",
        role=track_a.ROLE_CALIBRATION,
        start_utc="2026-07-29T00:00:00+00:00",
        end_utc="2026-07-29T00:00:02+00:00",
        duration_seconds=2.0,
        session_id="fixture",
        manifest_path=path,
        raw_path=path,
        raw_size_bytes=path.stat().st_size,
        raw_sha256=digest,
        bookticker_count=0,
        depth_count=40,
        trade_count=40,
        connection_epoch_count=1,
        depth_gap_count=0,
    )


def test_feature_replay_is_deterministic_and_causal(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.gz"
    digest = _write_raw(raw_path)
    capture = _capture(raw_path, digest)

    first = track_a.build_capture_features(capture, tmp_path / "first")
    second = track_a.build_capture_features(capture, tmp_path / "second")

    assert first.metrics["valid_fraction"] == 1.0
    assert first.metrics["sequence_gap_count"] == 0
    assert first.metrics["output_sha256"] == second.metrics["output_sha256"]
    with np.load(first.output_path, allow_pickle=False) as data:
        assert data["base_features"].shape[1] == len(track_a.BASE_FEATURE_NAMES)
        assert np.all(np.diff(data["ts_ns"]) == track_a.GRID_NS)
        midpoint_index = track_a.BASE_FEATURE_NAMES.index("midpoint_delta_ticks")
        assert np.isfinite(data["base_features"][:, midpoint_index]).all()


def test_depth_gap_fails_closed(tmp_path: Path) -> None:
    raw_path = tmp_path / "gap.gz"
    digest = _write_raw(raw_path, gap=True)
    with pytest.raises(track_a.TrackAError, match="depth_sequence_gap"):
        track_a.build_capture_features(_capture(raw_path, digest), tmp_path / "out")


def _manual_model() -> track_a.StateModel:
    labels = [
        np.asarray([0] * 20 + [1] * 8 + [0] * 18, dtype=np.int16),
        np.asarray([0] * 15 + [1] * 10 + [0] * 20, dtype=np.int16),
    ]
    pmf, hazard, survival = track_a._fit_duration_distribution(labels, 2, 32)
    return track_a.StateModel(
        name="fixture_model",
        emission="student_t",
        duration="negative_binomial",
        k=2,
        means=np.asarray([[-2.0, -1.0], [2.0, 1.0]]),
        scales=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        initial=np.asarray([0.9, 0.1]),
        transitions=np.asarray([[0.95, 0.05], [0.10, 0.90]]),
        duration_pmf=pmf,
        duration_hazard=hazard,
        duration_survival=survival,
    )


def test_online_filter_is_prefix_invariant() -> None:
    rng = np.random.default_rng(7)
    x = np.concatenate(
        (
            rng.normal([-2, -1], 0.2, size=(30, 2)),
            rng.normal([2, 1], 0.2, size=(20, 2)),
            rng.normal([-2, -1], 0.2, size=(30, 2)),
        )
    )
    altered = x.copy()
    altered[55:] = rng.normal([8, -8], 2, size=altered[55:].shape)
    model = _manual_model()

    posterior, ages, _ = track_a._hsmm_filter(model, x)
    altered_posterior, altered_ages, _ = track_a._hsmm_filter(model, altered)
    labels, _ = track_a._hsmm_viterbi(model, x)

    np.testing.assert_allclose(posterior[:55], altered_posterior[:55], atol=1e-7)
    np.testing.assert_allclose(ages[:55], altered_ages[:55], atol=1e-6)
    assert np.mean(labels[:30] == 0) > 0.9
    assert np.mean(labels[30:50] == 1) > 0.9


def test_deterministic_gzip_writer(tmp_path: Path) -> None:
    payload = "a,b\n1,2\n"
    paths = [tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"]
    for path in paths:
        with track_a._deterministic_gzip_writer(path) as fh:
            fh.write(payload)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    with gzip.open(paths[0], "rt") as fh:
        assert fh.read() == payload
