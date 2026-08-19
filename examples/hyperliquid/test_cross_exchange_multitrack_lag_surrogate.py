import numpy as np

from examples.hyperliquid.cross_exchange_multitrack_lag_surrogate import (
    TRACK_NAMES,
    _query_state_track,
    _raw_channel_states,
)


def test_full_multitrack_contract_contains_every_hyperliquid_track():
    assert TRACK_NAMES == (
        "bbo",
        "fast_l2",
        "trades",
        "standard_l2",
        "asset_context",
        "main_all_mids",
        "target_dex_all_mids",
    )


def test_raw_channel_states_preserve_order_and_complete_state(tmp_path):
    import gzip

    path = tmp_path / "raw.gz"
    with gzip.open(path, "wt") as fh:
        fh.write('10 {"channel":"x","data":{"a":1}}\n')
        fh.write('20 {"channel":"y","data":{"b":2}}\n')
        fh.write('30 {"channel":"x","data":{"a":3}}\n')
    states = _raw_channel_states(path, {"x"})
    assert np.array_equal(
        states["timestamps"], np.asarray([10, 30], dtype=np.int64)
    )
    assert states["state_hashes"].shape == (2,)
    assert states["state_hashes"][0] != states["state_hashes"][1]


def test_state_query_is_strict_asof_and_rejects_stale_rows():
    qualified, ages_ms, quality = _query_state_track(
        np.asarray([9, 10, 19, 31], dtype=np.int64),
        np.asarray([10, 20, 30], dtype=np.int64),
        np.asarray([101, 202, 303], dtype=np.uint64),
        max_age_ns=5,
    )
    assert qualified.tolist() == [False, True, False, True]
    assert ages_ms.tolist() == [-1e-6, 0.0, 9e-6, 1e-6]
    assert quality["queried_count"] == 3
    assert quality["missing_count"] == 1
    assert quality["qualification_failure_count"] == 2
    assert quality["queried_unique_state_count"] == 2
    assert len(quality["queried_state_checksum_sha256"]) == 64
