from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from examples.hyperliquid import cross_exchange_online_estimators as estimators
from examples.hyperliquid import (
    cross_exchange_public_multi_distance_dynamic_seed as seed,
)


BASE_MS = 1_000_000
SOURCE_COMMIT = "a" * 40


def _book(event_ms: int) -> dict[str, Any]:
    return {
        "event_kind": "book",
        "event_time_ms": event_ms,
        "local_receive_time_ms": "",
        "bid_px": 100.0,
        "ask_px": 101.0,
        "bid_depth_btc": 1.0,
        "ask_depth_btc": 1.0,
        "trade_px": "",
        "trade_size_btc": "",
        "aggressor_side": "",
        "trade_id": "",
    }


def _trade(
    event_ms: int,
    *,
    price: float,
    side: str,
    trade_id: str,
) -> dict[str, Any]:
    return {
        "event_kind": "trade",
        "event_time_ms": event_ms,
        "local_receive_time_ms": "",
        "bid_px": "",
        "ask_px": "",
        "bid_depth_btc": "",
        "ask_depth_btc": "",
        "trade_px": price,
        "trade_size_btc": 0.001,
        "aggressor_side": side,
        "trade_id": trade_id,
    }


def _events(*, with_trades: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trade_index = 0
    for interval in range(5):
        start = BASE_MS + interval * 1_000
        rows.append(_book(start))
        if not with_trades or interval == 4:
            continue
        for side, prices in (
            ("buy", [101.0] * 4 + [103.0] * 2 + [106.0]),
            ("sell", [100.0] * 4 + [98.0] * 2 + [95.0]),
        ):
            for offset, price in enumerate(prices, start=1):
                trade_index += 1
                rows.append(
                    _trade(
                        start + offset * 10,
                        price=price,
                        side=side,
                        trade_id=f"trade-{trade_index}",
                    )
                )
    return sorted(rows, key=lambda row: int(row["event_time_ms"]))


@pytest.fixture
def isolated_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(seed, "PROJECT_ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    runner.write_text("# deterministic runner fixture\n", encoding="utf-8")
    return tmp_path


def _build(
    root: Path,
    *,
    with_trades: bool = True,
) -> dict[str, Any]:
    return seed.build_artifacts(
        event_rows=_events(with_trades=with_trades),
        output_dir=root / "out",
        source_commit=SOURCE_COMMIT,
        interval_ms=1_000,
        distance_ticks=(0.5, 2.5, 5.5, 10.5),
        runner_path=root / "runner.py",
    )


def test_public_multi_distance_seed_is_eligible_and_loadable(
    isolated_project: Path,
) -> None:
    result = _build(isolated_project)

    assert result["contract"]["seed_eligible"] is True
    assert result["manifest"]["final_recommendation"] == (
        "accept_source_pinned_public_dynamic_seed_for_later_explicit_live"
    )
    for side in ("buy", "sell"):
        side_result = result["contract"]["per_side"][side]
        assert side_result["observation_count"] == 16
        assert side_result["distance_variation_count"] == 4
        assert side_result["fit_status"] == "pass"
        assert side_result["k"] > 0
    assert result["boundary"]["order_endpoint_called"] is False
    assert result["boundary"]["actual_quote_behavior_changed"] is False

    target = estimators.EventTimeOnlineEstimator()
    loaded = seed.load_seed_into_estimator(
        estimator=target,
        contract_path=isolated_project / "out" / "dynamic_spread_seed_contract.json",
        exposure_path=isolated_project / "out" / "quote_exposure_intervals.csv",
        expected_seed_contract_sha256=(
            result["contract"]["seed_contract_sha256"]
        ),
    )
    assert loaded["loaded_row_count"] == 32
    assert loaded["fits"]["buy"]["status"] == "pass"
    assert loaded["fits"]["sell"]["status"] == "pass"


def test_no_trade_source_remains_ineligible(
    isolated_project: Path,
) -> None:
    result = _build(isolated_project, with_trades=False)

    assert result["contract"]["seed_eligible"] is False
    assert result["manifest"]["final_recommendation"] == (
        "blocked_public_dynamic_seed_threshold_or_fit_not_met"
    )
    assert result["contract"]["per_side"]["buy"]["fit_status"] == "unavailable"
    assert result["contract"]["per_side"]["sell"]["fit_status"] == "unavailable"


def test_seed_contract_and_exposure_tamper_fail_closed(
    isolated_project: Path,
) -> None:
    result = _build(isolated_project)
    contract_path = (
        isolated_project / "out" / "dynamic_spread_seed_contract.json"
    )
    exposure_path = isolated_project / "out" / "quote_exposure_intervals.csv"

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["interval_ms"] = 2_000
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(seed.SeedError, match="seed_contract_self_hash_mismatch"):
        seed.load_seed_into_estimator(
            estimator=estimators.EventTimeOnlineEstimator(),
            contract_path=contract_path,
            exposure_path=exposure_path,
            expected_seed_contract_sha256=(
                result["contract"]["seed_contract_sha256"]
            ),
        )

    _build(isolated_project)
    rows = seed._read_csv(exposure_path)
    rows[0]["arrival_count"] = "999"
    with exposure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=estimators.quote_exposure_fieldnames(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(seed.SeedError, match="seed_exposure_file_hash_mismatch"):
        seed.load_seed_into_estimator(
            estimator=estimators.EventTimeOnlineEstimator(),
            contract_path=contract_path,
            exposure_path=exposure_path,
            expected_seed_contract_sha256=(
                result["contract"]["seed_contract_sha256"]
            ),
        )


def test_rebuild_is_deterministic(isolated_project: Path) -> None:
    first = _build(isolated_project)
    output_dir = isolated_project / "out"
    first_files = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    second = _build(isolated_project)
    second_files = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    assert first == second
    assert first_files == second_files


def test_main_source_commit_gate_distinguishes_collection_and_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    isolated_project: Path,
) -> None:
    event_path = isolated_project / "events.csv"
    seed._write_csv(
        event_path,
        _events(),
        estimators.estimator_event_fieldnames(),
    )
    monkeypatch.setattr(seed, "_git_head", lambda: "b" * 40)
    monkeypatch.setattr(seed, "_git_object_exists", lambda value: value == SOURCE_COMMIT)
    monkeypatch.setattr(seed, "__file__", str(isolated_project / "runner.py"))
    monkeypatch.setattr(
        seed,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output_dir": isolated_project / "rebuilt",
                "watcher_seconds": 1.0,
                "interval_ms": 1_000,
                "distance_ticks": [0.5, 2.5, 5.5, 10.5],
                "source_commit": SOURCE_COMMIT,
                "hyperliquid_l2book_fast": False,
                "rebuild_from_event_rows": event_path,
            },
        )(),
    )

    assert seed.main() == 0
