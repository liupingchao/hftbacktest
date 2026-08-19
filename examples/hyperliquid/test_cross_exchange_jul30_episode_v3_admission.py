from __future__ import annotations

from copy import deepcopy
import json

import pytest

import cross_exchange_jul30_episode_v3_admission as admission


def _valid_manifest():
    semantic_scope = admission.episode_v3._source_semantic_projection_contract()
    aggregate_contract = semantic_scope["aggregate_evidence_contract"]
    aggregate = {}
    aggregate_output_counts = {}
    for index, (projection_name, projection) in enumerate(
        aggregate_contract["projections"].items(),
        start=1,
    ):
        digest = f"{index:064x}"
        aggregate[projection_name] = {
            "expected_rows": projection["expected_rows"],
            "observed_rows": projection["expected_rows"],
            "expected_sha256": digest,
            "observed_sha256": digest,
            "mismatch_rows": 0,
            "fields": list(projection["fields"]),
        }
        for binding in projection["manifest_count_bindings"]:
            if binding["section"] == "aggregate_output_counts":
                aggregate_output_counts[binding["field"]] = binding[
                    "expected_value"
                ]
    return {
        "artifacts": [],
        "core_package_sha256": "a" * 64,
        "exact_counts": admission.episode_v3._expected_manifest_exact_counts(),
        "aggregate_output_counts": aggregate_output_counts,
        "_source_semantic_verification": {
            "source_semantic_verified": True,
            "scope": semantic_scope,
            "aggregate": aggregate,
        },
    }


def _run_with_manifest(monkeypatch, capsys, manifest):
    monkeypatch.setattr(
        admission.episode_v3, "verify_package", lambda _path: manifest
    )
    monkeypatch.setattr(
        admission.episode_v3, "_directory_inventory", lambda _path: []
    )
    return_code = admission.main(
        ["--verify-only", "--package-dir", "/tmp/package"]
    )
    return return_code, json.loads(capsys.readouterr().out)


def test_admission_requires_verify_only(capsys):
    assert admission.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is False


def test_admission_help_contract():
    args = admission.parse_args(["--verify-only", "--package-dir", "/tmp/package"])
    assert args.verify_only is True
    assert args.package_dir == "/tmp/package"


def test_admission_reports_source_semantic_verification(monkeypatch, capsys):
    manifest = _valid_manifest()
    aggregate_contract = manifest["_source_semantic_verification"]["scope"][
        "aggregate_evidence_contract"
    ]
    assert aggregate_contract[
        "manifest_aggregate_output_count_key_policy"
    ] == {
        "policy": "exact",
        "derivation": (
            "unique binding.field values from "
            "projections[*].manifest_count_bindings where "
            "binding.section=aggregate_output_counts"
        ),
    }
    expected_aggregate_output_counts = (
        admission.episode_v3._manifest_count_binding_expected_values(
            aggregate_contract["projections"],
            section="aggregate_output_counts",
        )
    )
    assert set(manifest["aggregate_output_counts"]) == (
        set(expected_aggregate_output_counts)
    )
    assert manifest["aggregate_output_counts"] == expected_aggregate_output_counts
    return_code, payload = _run_with_manifest(
        monkeypatch, capsys, manifest
    )
    assert return_code == 0
    assert payload["source_semantic_verified"] is True
    assert payload["source_semantic_scope"] == manifest[
        "_source_semantic_verification"
    ]["scope"]
    assert len(payload["source_semantic_aggregate"]) == 12
    assert (
        payload["source_semantic_aggregate"]["features_family_a"][
            "mismatch_rows"
        ]
        == 0
    )


def test_admission_rejects_missing_complete_source_semantic_evidence(
    monkeypatch, capsys
):
    manifest = {
        "artifacts": [],
        "core_package_sha256": "a" * 64,
        "exact_counts": admission.episode_v3._expected_manifest_exact_counts(),
        "aggregate_output_counts": {},
    }
    monkeypatch.setattr(
        admission.episode_v3, "verify_package", lambda _path: manifest
    )
    assert (
        admission.main(["--verify-only", "--package-dir", "/tmp/package"])
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is False
    assert "complete source-semantic verification evidence missing" in payload[
        "error"
    ]


@pytest.mark.parametrize(
    "case",
    (
        "aggregate_not_dict",
        "missing_projection",
        "extra_projection",
        "entry_not_dict",
        "missing_entry_key",
        "extra_entry_key",
        "fields_drift",
        "fields_order_drift",
        "expected_rows_drift",
        "expected_rows_bool",
        "expected_rows_string",
        "observed_rows_drift",
        "observed_rows_bool",
        "observed_rows_string",
        "digest_mismatch",
        "expected_digest_malformed",
        "observed_digest_malformed",
        "expected_digest_uppercase",
        "observed_digest_uppercase",
        "mismatch_nonzero",
        "mismatch_bool",
        "mismatch_string",
        "scope_drift",
        "semantic_extra_key",
        "exact_count_missing",
        "exact_count_contradiction",
        "exact_count_bool",
        "exact_count_string",
        "aggregate_count_missing",
        "aggregate_count_extra",
        "aggregate_count_contradiction",
        "aggregate_count_bool",
        "aggregate_count_string",
    ),
)
def test_admission_rejects_invalid_aggregate_evidence(
    case, monkeypatch, capsys
):
    manifest = deepcopy(_valid_manifest())
    semantic = manifest["_source_semantic_verification"]
    aggregate = semantic["aggregate"]
    projection_name = "features_family_a"
    entry = aggregate[projection_name]
    if case == "aggregate_not_dict":
        semantic["aggregate"] = []
    elif case == "missing_projection":
        del aggregate["outcomes"]
    elif case == "extra_projection":
        aggregate["unexpected_projection"] = deepcopy(entry)
    elif case == "entry_not_dict":
        aggregate[projection_name] = []
    elif case == "missing_entry_key":
        del entry["observed_sha256"]
    elif case == "extra_entry_key":
        entry["unexpected"] = 0
    elif case == "fields_drift":
        entry["fields"].append("unexpected")
    elif case == "fields_order_drift":
        entry["fields"][0], entry["fields"][1] = (
            entry["fields"][1],
            entry["fields"][0],
        )
    elif case == "expected_rows_drift":
        entry["expected_rows"] += 1
    elif case == "expected_rows_bool":
        entry["expected_rows"] = True
    elif case == "expected_rows_string":
        entry["expected_rows"] = str(entry["expected_rows"])
    elif case == "observed_rows_drift":
        entry["observed_rows"] -= 1
    elif case == "observed_rows_bool":
        entry["observed_rows"] = False
    elif case == "observed_rows_string":
        entry["observed_rows"] = str(entry["observed_rows"])
    elif case == "digest_mismatch":
        entry["observed_sha256"] = "f" * 64
    elif case == "expected_digest_malformed":
        entry["expected_sha256"] = "g" * 64
    elif case == "observed_digest_malformed":
        entry["observed_sha256"] = "a" * 63
    elif case == "expected_digest_uppercase":
        entry["expected_sha256"] = "A" * 64
    elif case == "observed_digest_uppercase":
        entry["observed_sha256"] = "B" * 64
    elif case == "mismatch_nonzero":
        entry["mismatch_rows"] = 999
    elif case == "mismatch_bool":
        entry["mismatch_rows"] = False
    elif case == "mismatch_string":
        entry["mismatch_rows"] = "0"
    elif case == "scope_drift":
        semantic["scope"]["exact_feature_projection"]["fields"].reverse()
    elif case == "semantic_extra_key":
        semantic["unexpected"] = True
    elif case == "exact_count_missing":
        del manifest["exact_counts"]["family_a_rows"]
    elif case == "exact_count_contradiction":
        manifest["exact_counts"]["family_a_rows"] += 1
    elif case == "exact_count_bool":
        manifest["exact_counts"]["family_a_rows"] = True
    elif case == "exact_count_string":
        manifest["exact_counts"]["family_a_rows"] = "268522"
    elif case == "aggregate_count_missing":
        del manifest["aggregate_output_counts"]["feature_ledger_family_a_rows"]
    elif case == "aggregate_count_extra":
        manifest["aggregate_output_counts"]["unexpected"] = 0
    elif case == "aggregate_count_contradiction":
        manifest["aggregate_output_counts"]["feature_ledger_family_a_rows"] += 1
    elif case == "aggregate_count_bool":
        manifest["aggregate_output_counts"]["feature_ledger_family_a_rows"] = True
    elif case == "aggregate_count_string":
        manifest["aggregate_output_counts"]["feature_ledger_family_a_rows"] = (
            "23092892"
        )
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(case)

    return_code, payload = _run_with_manifest(
        monkeypatch, capsys, manifest
    )
    assert return_code == 2
    assert payload["verified"] is False


def test_admission_rejects_round3_qa_minimal_forged_aggregate(
    monkeypatch, capsys
):
    manifest = _valid_manifest()
    manifest["_source_semantic_verification"]["aggregate"] = {
        "features_family_a": {
            "expected_rows": 1,
            "observed_rows": 0,
            "expected_sha256": "b" * 64,
            "observed_sha256": "c" * 64,
            "mismatch_rows": 999,
            "fields": [],
        }
    }
    return_code, payload = _run_with_manifest(
        monkeypatch, capsys, manifest
    )
    assert return_code == 2
    assert payload["verified"] is False
    assert "projection universe drift" in payload["error"]


def test_admission_propagates_source_semantic_failure(monkeypatch, capsys):
    def fail(_path):
        raise admission.episode_v3.EpisodeV3Error(
            "anchors source-semantic drift"
        )

    monkeypatch.setattr(admission.episode_v3, "verify_package", fail)
    assert admission.main(["--verify-only", "--package-dir", "/tmp/package"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is False
    assert "source-semantic drift" in payload["error"]
