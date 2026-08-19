from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "cross_exchange_trigger_aligned_episode_contract.py"
)
SPEC = importlib.util.spec_from_file_location("episode_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def test_frozen_contract_closes_family_measurement_scoring_and_observation() -> None:
    payload = contract.build_frozen_contract()
    contract.validate_frozen_contract(payload)
    assert (
        payload["family_views"]["family_a"]["confirmed_only_filter_forbidden"] is True
    )
    assert payload["measurement_contract"]["trade_quantity_source"] == "binance_trade_q"
    assert payload["measurement_contract"]["nq_available"] is False
    assert payload["measurement_contract"]["depth_stream"] == "depth@0ms"
    assert payload["scoring_contract"]["continuous_predictive_distribution"] == "CRPS"
    assert payload["scoring_contract"]["binary_outcome_probability"] == "Brier"
    assert (
        payload["scoring_contract"]["interval_or_right_censored_event_time"]
        == "interval_log_loss"
    )
    assert (
        payload["feature_observation_contract"]["invariant"]
        == "observed_at_ns <= decision_landmark_ns"
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("trade_quantity_source", "binance_aggtrade_nq", "trade_quantity_source"),
        ("rpi_adjustment_status", "available", "rpi_adjustment_status"),
        ("depth_stream", "depth@100ms", "depth_stream"),
        ("nq_available", True, "nq_available"),
    ],
)
def test_rejects_rpi_nq_and_depth_contract_drift(
    field: str, bad_value: object, message: str
) -> None:
    payload = contract.build_frozen_contract()
    payload["measurement_contract"][field] = bad_value
    with pytest.raises(contract.AdmissionError, match=message):
        contract.validate_frozen_contract(payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("continuous_predictive_distribution", "pinball"),
        ("binary_outcome_probability", "log_score"),
        ("interval_or_right_censored_event_time", "point_time_mae"),
        ("not_more_than_five_percent_worse", "normalized_loss <= 1.10"),
    ],
)
def test_rejects_scoring_metric_drift(field: str, bad_value: str) -> None:
    payload = contract.build_frozen_contract()
    payload["scoring_contract"][field] = bad_value
    with pytest.raises(contract.AdmissionError, match="scoring_contract"):
        contract.validate_frozen_contract(payload)


def test_feature_observation_invariant_fails_on_future_value() -> None:
    contract.validate_feature_observations(
        [
            {
                "value": 1.0,
                "observed_at_ns": 100,
                "decision_landmark_ns": 100,
            }
        ]
    )
    with pytest.raises(contract.AdmissionError, match="future observation"):
        contract.validate_feature_observations(
            [
                {
                    "value": 1.0,
                    "observed_at_ns": 101,
                    "decision_landmark_ns": 100,
                }
            ]
        )


def test_unknown_krx_state_cannot_be_inferred_from_future_prices() -> None:
    rows = [
        {
            "session_id": spec.session_id,
            "calendar_authority_status": "unavailable_in_frozen_local_inputs",
            "underlying_market_state": "unknown_calendar_state",
            "future_price_inference_used": False,
        }
        for spec in contract.SESSION_SPECS
    ]
    contract.validate_underlying_coverage(rows)
    forged = copy.deepcopy(rows)
    forged[0]["underlying_market_state"] = "continuous_trading"
    forged[0]["future_price_inference_used"] = True
    with pytest.raises(contract.AdmissionError):
        contract.validate_underlying_coverage(forged)


def _complete_cadence_rows() -> list[dict[str, object]]:
    rows = []
    for spec in contract.SESSION_SPECS:
        for channel in contract.CADENCE_CHANNELS:
            for metric in contract.CADENCE_METRICS:
                not_applicable = channel == "trades" and metric == "no_new_information"
                values = {
                    field: "" if not_applicable else "1"
                    for field in contract.CADENCE_VALUE_FIELDS
                }
                rows.append(
                    {
                        "session_id": spec.session_id,
                        "channel": channel,
                        "source": "test_fixture",
                        "metric": metric,
                        "observation_unit": "test_observation",
                        "availability": (
                            "not_applicable" if not_applicable else "available"
                        ),
                        **values,
                        "missing_fields": "",
                        "semantic_note": "test fixture",
                    }
                )
    return rows


def test_qa_fabricated_cadence_missing_values_fail_closed() -> None:
    rows = _complete_cadence_rows()
    contract.validate_cadence_rows(rows)

    partial = copy.deepcopy(rows)
    partial[0]["availability"] = "partial_compact_metadata"
    partial[0]["p01_ms"] = ""
    partial[0]["p10_ms"] = ""
    partial[0]["missing_fields"] = "p01_ms|p10_ms"
    contract.validate_cadence_rows(partial)

    fabricated = copy.deepcopy(partial)
    fabricated[0]["p01_ms"] = "0.125"
    with pytest.raises(contract.AdmissionError, match="complement"):
        contract.validate_cadence_rows(fabricated)

    unavailable = copy.deepcopy(rows)
    unavailable[1]["availability"] = "unavailable"
    for field in contract.CADENCE_VALUE_FIELDS:
        unavailable[1][field] = ""
    unavailable[1]["missing_fields"] = "|".join(contract.CADENCE_VALUE_FIELDS)
    contract.validate_cadence_rows(unavailable)

    fabricated_unavailable = copy.deepcopy(unavailable)
    fabricated_unavailable[1]["p50_ms"] = "9.5"
    with pytest.raises(contract.AdmissionError, match="explicitly missing"):
        contract.validate_cadence_rows(fabricated_unavailable)


def test_cadence_missing_row_schema_and_combined_partition_drift_fail() -> None:
    rows = _complete_cadence_rows()
    with pytest.raises(contract.AdmissionError, match="key set"):
        contract.validate_cadence_rows(rows[:-1])

    missing_key = copy.deepcopy(rows)
    del missing_key[0]["semantic_note"]
    with pytest.raises(contract.AdmissionError, match="schema drift"):
        contract.validate_cadence_rows(missing_key)

    extra_key = copy.deepcopy(rows)
    extra_key[0]["unexpected"] = ""
    with pytest.raises(contract.AdmissionError, match="schema drift"):
        contract.validate_cadence_rows(extra_key)

    partial = copy.deepcopy(rows)
    partial[0]["availability"] = "partial_compact_metadata"
    partial[0]["p01_ms"] = ""
    partial[0]["p10_ms"] = ""
    partial[0]["p90_ms"] = ""
    partial[0]["missing_fields"] = "p01_ms|p10_ms"
    with pytest.raises(contract.AdmissionError, match="complement"):
        contract.validate_cadence_rows(partial)


def test_qa_aug07_compact_full_event_payload_fails_closed(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    compact_root = tmp_path / "compact"
    raw_root.mkdir()
    compact_root.mkdir()
    policy = contract.Aug07AccessPolicy(raw_root, compact_root)
    for root in (raw_root, compact_root):
        for relative in (
            "raw.gz",
            "other_raw.gz",
            "decision_labels.csv.gz",
            "basis_features.csv.gz",
        ):
            path = root / relative
            path.write_bytes(b"x")
            with pytest.raises(contract.AdmissionError, match="forbidden"):
                policy.assert_content_read_allowed(path)

        unknown = root / "unknown_metadata.json"
        unknown.write_text("{}\n", encoding="utf-8")
        with pytest.raises(contract.AdmissionError, match="not_allowlisted"):
            policy.assert_content_read_allowed(unknown)

    raw_manifest = raw_root / "campaign_manifest.json"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    compact_manifest = compact_root / "evidence/final_acceptance_summary.json"
    compact_manifest.parent.mkdir(parents=True)
    compact_manifest.write_text("{}\n", encoding="utf-8")
    assert policy.read_json(raw_manifest) == {}
    assert policy.read_json(compact_manifest) == {}

    alias = raw_root / "aliased_raw.gz"
    alias.symlink_to(raw_manifest)
    with pytest.raises(contract.AdmissionError, match="forbidden"):
        policy.assert_content_read_allowed(alias)

    ledger = policy.ledger()
    contract.validate_aug07_access_ledger(ledger)
    assert ledger["event_rows_opened"] is False
    assert ledger["event_row_read_count"] == 0
    assert ledger["content_read_paths"] == [
        "compact:evidence/final_acceptance_summary.json",
        "raw:campaign_manifest.json",
    ]


def _complete_topology_rows() -> list[dict[str, object]]:
    rows = [_topology_row(spec) for spec in contract.SESSION_SPECS]
    return rows


def _topology_row(spec: object) -> dict[str, object]:
    payload = contract._expected_collection_topology_payload(spec)
    return {
        "session_id": spec.session_id,
        "campaign_id": spec.expected_campaign_id,
        "evidence_label": spec.evidence_label,
        **contract._expected_collection_time_facts(spec),
        "segment_count": spec.expected_segment_count,
        "collection_host_alias": payload["collection_host_alias"],
        "instance_id": payload["instance_id"],
        "instance_type": payload["instance_type"],
        "private_hostname": payload["private_hostname"],
        "cloud_region": payload["cloud_region"],
        "availability_zone": payload["availability_zone"],
        "availability_zone_id": payload["availability_zone_id"],
        "receipt_clock": payload["receipt_clock"],
        "clock_sync_evidence": payload["clock_sync_evidence"],
        "collection_task_id": payload["collection_task_id"],
        "collector_sha256": payload["collector_sha256"],
        "supervisor_sha256": payload["supervisor_sha256"],
        "python_executable": payload["python_executable"],
        "websocket_library": payload["websocket_library"],
        "binance_websocket_url": payload["binance_websocket_url"],
        "hyperliquid_websocket_url": payload["hyperliquid_websocket_url"],
        "binance_streams_json": json.dumps(
            payload["binance_streams"],
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "hyperliquid_channels_json": json.dumps(
            payload["hyperliquid_channels"],
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "collection_mode": payload["collection_mode"],
        "local_receipt_clock_manifest_text": payload[
            "local_receipt_clock_manifest_text"
        ],
        "physical_topology_fingerprint": contract.canonical_json_sha256(
            contract.PHYSICAL_TOPOLOGY
        ),
        "collection_topology_fingerprint": contract.canonical_json_sha256(payload),
        "topology_provenance_path": spec.topology_provenance_path,
        "research_execution_host": "test-research-host",
        "research_execution_platform": "test-platform",
        "research_execution_machine": "test-machine",
        "research_execution_role": ("research_execution_host_not_acquisition_host"),
        "admission_status": (
            "admitted_contract_freeze_only_event_rows_locked"
            if spec.session_id == "aug07"
            else "admitted_with_explicit_measurement_limitations"
        ),
    }


def test_qa_forged_collection_topology_fingerprint_fails_closed() -> None:
    rows = _complete_topology_rows()
    contract.validate_topology_rows(rows)
    forged = copy.deepcopy(rows)
    forged[0]["collection_topology_fingerprint"] = "0" * 64
    with pytest.raises(contract.AdmissionError, match="fingerprint drift"):
        contract.validate_topology_rows(forged)


def test_topology_recomputed_mutation_session_set_and_schema_fail_closed() -> None:
    rows = _complete_topology_rows()
    mutated = copy.deepcopy(rows)
    mutated[0]["python_executable"] = "/tmp/forged-python"
    mutated[0]["collection_topology_fingerprint"] = contract.canonical_json_sha256(
        contract._topology_payload_from_row(mutated[0])
    )
    with pytest.raises(contract.AdmissionError, match="acquisition_payload"):
        contract.validate_topology_rows(mutated)

    duplicate = copy.deepcopy(rows)
    duplicate[-1]["session_id"] = duplicate[0]["session_id"]
    with pytest.raises(contract.AdmissionError, match="cardinality"):
        contract.validate_topology_rows(duplicate)

    missing_key = copy.deepcopy(rows)
    del missing_key[0]["topology_provenance_path"]
    with pytest.raises(contract.AdmissionError, match="schema drift"):
        contract.validate_topology_rows(missing_key)

    extra_key = copy.deepcopy(rows)
    extra_key[0]["unexpected"] = ""
    with pytest.raises(contract.AdmissionError, match="schema drift"):
        contract.validate_topology_rows(extra_key)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (
            ("family_views", "family_a", "population"),
            "confirmed_only_candidates",
        ),
        (
            ("family_views", "family_b", "decision_landmark"),
            "t_candidate",
        ),
        (
            ("landmarks", "rejected_t_confirm_nullable"),
            False,
        ),
        (
            ("feature_observation_contract", "invariant"),
            "observed_at_ns < decision_landmark_ns",
        ),
        (
            ("outcome_contract", "primary_outcomes"),
            list(reversed(contract.PRIMARY_OUTCOMES)),
        ),
        (
            ("interval_censoring_contract", "point_coercion_forbidden"),
            False,
        ),
        (
            ("hypothesis_contract", "family_a", "A0"),
            "direction only",
        ),
        (
            ("hypothesis_contract", "family_b", "B0"),
            "direction only",
        ),
        (
            ("scoring_contract", "normalized_loss"),
            "model_loss - frozen_family_baseline_loss",
        ),
        (
            ("aug07_first_read_contract", "freeze_before_first_event_row"),
            False,
        ),
    ],
)
def test_complete_frozen_contract_rejects_same_schema_drift(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    payload = contract.build_frozen_contract()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(contract.AdmissionError, match="exact freeze drift"):
        contract.validate_frozen_contract(payload)


def test_qa_family_landmark_interval_contract_drift_fails_closed() -> None:
    payload = contract.build_frozen_contract()
    payload["family_views"]["family_b"]["decision_landmark"] = "t_candidate"
    payload["landmarks"]["rejected_t_confirm_nullable"] = False
    payload["interval_censoring_contract"]["point_coercion_forbidden"] = False
    with pytest.raises(contract.AdmissionError):
        contract.validate_frozen_contract(payload)


def test_frozen_contract_missing_extra_and_combined_mutations_fail_closed() -> None:
    missing = contract.build_frozen_contract()
    del missing["hypothesis_contract"]["family_b"]["B0"]
    with pytest.raises(contract.AdmissionError, match="key set drift"):
        contract.validate_frozen_contract(missing)

    extra = contract.build_frozen_contract()
    extra["aug07_first_read_contract"]["event_row_override"] = False
    with pytest.raises(contract.AdmissionError, match="key set drift"):
        contract.validate_frozen_contract(extra)

    combined = contract.build_frozen_contract()
    combined["family_views"]["family_a"]["confirmed_only_filter_forbidden"] = False
    combined["feature_observation_contract"]["family_b_decision_landmark"] = (
        "t_candidate"
    )
    combined["scoring_contract"]["family_a_baseline"] = "B0"
    combined["aug07_first_read_contract"]["unknown_content_path_fails_closed"] = False
    with pytest.raises(contract.AdmissionError, match="exact freeze drift"):
        contract.validate_frozen_contract(combined)


def test_inventory_parser_rejects_missing_hash(tmp_path: Path) -> None:
    path = tmp_path / "inventory.tsv"
    path.write_text("not-a-sha\t1\tcampaign_manifest.json\n", encoding="utf-8")
    with pytest.raises(contract.AdmissionError, match="SHA256"):
        contract.parse_sha_size_path_inventory(path)


def test_required_manifest_missing_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(contract.AdmissionError, match="required file missing"):
        contract._require_file(tmp_path / "campaign_manifest.json")


def test_atomic_publication_hides_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "artifact"

    def failing(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise contract.AdmissionError("injected")

    with pytest.raises(contract.AdmissionError, match="injected"):
        contract.atomic_publish_directory(output, failing, clean_output=False)
    assert not output.exists()
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def test_atomic_publication_preserves_previous_output_on_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "accepted.txt").write_text("accepted\n", encoding="utf-8")

    def failing(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise contract.AdmissionError("injected")

    with pytest.raises(contract.AdmissionError, match="injected"):
        contract.atomic_publish_directory(output, failing, clean_output=True)
    assert (output / "accepted.txt").read_text(encoding="utf-8") == "accepted\n"
    assert not (output / "partial.txt").exists()


def test_canonical_serialization_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"z": 3, "y": 2}}
    right = {"a": {"y": 2, "z": 3}, "b": [2, 1]}
    assert contract._canonical_json_bytes(left) == contract._canonical_json_bytes(right)
    assert contract.canonical_json_sha256(left) == contract.canonical_json_sha256(right)


def test_aug07_inventory_parser_is_headerless_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "inventory.tsv"
    path.write_text(
        f"{'a' * 64}\t12\tcampaign_manifest.json\n"
        f"{'b' * 64}\t34\tsegments/segment_0001/raw.gz\n",
        encoding="utf-8",
    )
    assert contract.parse_sha_size_path_inventory(path) == [
        {
            "relative_path": "campaign_manifest.json",
            "bytes": 12,
            "sha256": "a" * 64,
        },
        {
            "relative_path": "segments/segment_0001/raw.gz",
            "bytes": 34,
            "sha256": "b" * 64,
        },
    ]


def test_contract_scope_explicitly_excludes_episode_and_live_actions() -> None:
    scope = contract.build_frozen_contract()["scope"]
    assert scope == {
        "input_freeze_and_data_admission_only": True,
        "episode_v3_built": False,
        "trigger_density_run": False,
        "outcomes_run": False,
        "models_run": False,
        "actionability_run": False,
        "collection_started": False,
        "private_or_order_cancel_access": False,
    }


def test_contract_json_round_trip(tmp_path: Path) -> None:
    payload = contract.build_frozen_contract()
    path = tmp_path / "contract.json"
    contract._write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
