#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path


FEED_METRICS = (
    "binance_feed_p50_ns",
    "binance_feed_p99_ns",
)

LOCAL_METRICS = (
    "binance_tick_p50_ns",
    "binance_tick_p99_ns",
    "hyperliquid_tick_p50_ns",
    "hyperliquid_tick_p99_ns",
    "benchmark_p50_ns",
    "benchmark_p99_ns",
)
METRICS = FEED_METRICS + LOCAL_METRICS
CLOCK_BOUND_LIMIT_US = 750.0
BALANCED_WEIGHTS = {
    "binance_feed_p99_ns": 4,
    "binance_feed_p50_ns": 2,
    "binance_tick_p99_ns": 2,
    "hyperliquid_tick_p99_ns": 1,
    "benchmark_p99_ns": 1,
}


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_rows(rows, metric):
    ordered = sorted(
        rows,
        key=lambda row: (
            row[metric],
            row["instance_type"],
            row["instance_id"],
        ),
    )
    return [
        {
            "rank": index + 1,
            "instance_type": row["instance_type"],
            "instance_id": row["instance_id"],
            "candidate_label": row.get("candidate_label"),
            "value_ns": row[metric],
        }
        for index, row in enumerate(ordered)
    ]


def balanced_selection(rows, rankings):
    rank_by_metric = {
        metric: {
            row["instance_id"]: row["rank"]
            for row in rankings[metric]
        }
        for metric in BALANCED_WEIGHTS
    }
    scored = []
    for row in rows:
        component_ranks = {
            metric: rank_by_metric[metric][row["instance_id"]]
            for metric in BALANCED_WEIGHTS
        }
        weighted_score = sum(
            BALANCED_WEIGHTS[metric] * rank
            for metric, rank in component_ranks.items()
        )
        scored.append(
            {
                "instance_id": row["instance_id"],
                "instance_type": row["instance_type"],
                "candidate_label": row.get("candidate_label"),
                "weighted_score": weighted_score,
                "worst_component_rank": max(component_ranks.values()),
                "clock_max_bound_us": row["clock_max_bound_us"],
                "component_ranks": component_ranks,
            }
        )
    scored.sort(
        key=lambda row: (
            row["weighted_score"],
            row["worst_component_rank"],
            row["clock_max_bound_us"],
            row["instance_id"],
        )
    )
    for index, row in enumerate(scored, 1):
        row["balanced_rank"] = index
    return {
        "weights": BALANCED_WEIGHTS,
        "tie_break_order": [
            "weighted_score",
            "worst_component_rank",
            "clock_max_bound_us",
            "instance_id",
        ],
        "winner_instance_id": scored[0]["instance_id"] if scored else None,
        "ranking": scored,
    }


def validate_summary(summary):
    integrity = summary["integrity"]
    return (
        summary["trace_count"] > 0
        and integrity["complete_required"] == summary["trace_count"]
        and integrity["incomplete_required"] == 0
        and integrity["duplicate_trace_ids"] == 0
        and integrity["duplicate_stage_marks"] == 0
        and integrity["out_of_order"] == 0
        and integrity["dropped_traces"] == 0
    )


def validate_clock_pair(before, after, limit_us=CLOCK_BOUND_LIMIT_US):
    required = (
        "captured_at_utc",
        "leap_status",
        "error_bound_us",
        "error_bound_limit_us",
        "gate",
    )
    if any(field not in before or field not in after for field in required):
        return False
    return (
        before["captured_at_utc"] != after["captured_at_utc"]
        and before["leap_status"] == "Normal"
        and after["leap_status"] == "Normal"
        and before["gate"] == "pass"
        and after["gate"] == "pass"
        and before["error_bound_limit_us"] == limit_us
        and after["error_bound_limit_us"] == limit_us
        and before["error_bound_us"] <= limit_us
        and after["error_bound_us"] <= limit_us
    )


def ensure_extracted(run_dir: Path, instance_id: str) -> Path:
    result = run_dir / "extracted" / instance_id / "full"
    if result.exists():
        return result
    archive = run_dir / "full_archives" / instance_id / "full.tar.gz"
    target = result.parent
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(target, filter="data")
    return result


def analyze(run_dir: Path):
    expected_binary_sha = (
        run_dir / "build" / "artifact-sha256.txt"
    ).read_text().split()[0]
    rows = []
    for line in (run_dir / "candidate_instances.tsv").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 3:
            instance_type, instance_id, placement_group = fields
            candidate_label = None
        elif len(fields) == 4:
            instance_type, instance_id, placement_group, candidate_label = fields
        else:
            raise ValueError(f"invalid candidate row: {line}")
        result = ensure_extracted(run_dir, instance_id)
        status = load_json(result / "run-status.json")
        metadata_before = load_json(result / "metadata-before.json")
        metadata_after = load_json(result / "metadata-after.json")
        clock_before = load_json(result / "clock-before.json")
        clock_after = load_json(result / "clock-after.json")
        binance = load_json(result / "binance-summary.json")
        hyperliquid = load_json(result / "hyperliquid-summary.json")
        benchmark = load_json(result / "benchmark.json")
        binary_sha = (result / "sha256sums.txt").read_text().split()[0]
        metadata_identity_match = (
            metadata_before["instance_id"] == metadata_after["instance_id"]
            and metadata_before["instance_type"] == metadata_after["instance_type"]
        )
        checks = {
            "run_status": all(
                status[name] == 0
                for name in (
                    "benchmark_status",
                    "binance_status",
                    "hyperliquid_status",
                    "rebuild_status",
                )
            ),
            "duration": status["duration_seconds"] == 900,
            "warmup": status["warmup_messages"] == 100,
            "clock_gate": validate_clock_pair(clock_before, clock_after),
            "binary_hash": binary_sha == expected_binary_sha,
            "binance_integrity": validate_summary(binance),
            "hyperliquid_integrity": validate_summary(hyperliquid),
            "benchmark": benchmark["passed"] and benchmark["dropped_traces"] == 0,
            "metadata_identity": metadata_identity_match,
        }
        local_pipeline_eligible = all(
            value for name, value in checks.items() if name != "clock_gate"
        )
        feed_eligible = local_pipeline_eligible and checks["clock_gate"]
        row = {
            "instance_type": instance_type,
            "instance_id": instance_id,
            "candidate_label": candidate_label,
            "placement_group": placement_group,
            "eligible": feed_eligible,
            "feed_eligible": feed_eligible,
            "local_pipeline_eligible": local_pipeline_eligible,
            "checks": checks,
            "clock_before_bound_us": clock_before["error_bound_us"],
            "clock_after_bound_us": clock_after["error_bound_us"],
            "clock_max_bound_us": max(
                clock_before["error_bound_us"], clock_after["error_bound_us"]
            ),
            "clock_before_captured_at_utc": clock_before["captured_at_utc"],
            "clock_after_captured_at_utc": clock_after["captured_at_utc"],
            "cpu_model": metadata_before["cpu_model"],
            "metadata_identity_match": metadata_identity_match,
            "archive_sha256": sha256(
                run_dir / "full_archives" / instance_id / "full.tar.gz"
            ),
            "binance_trace_count": binance["trace_count"],
            "hyperliquid_trace_count": hyperliquid["trace_count"],
            "binance_feed_p50_ns": binance["intervals"]["feed_network"]["p50_ns"],
            "binance_feed_p99_ns": binance["intervals"]["feed_network"]["p99_ns"],
            "binance_tick_p50_ns": binance["intervals"]["tick_to_wire"]["p50_ns"],
            "binance_tick_p99_ns": binance["intervals"]["tick_to_wire"]["p99_ns"],
            "hyperliquid_feed_p50_ns": hyperliquid["intervals"]["feed_network"]["p50_ns"],
            "hyperliquid_feed_p99_ns": hyperliquid["intervals"]["feed_network"]["p99_ns"],
            "hyperliquid_tick_p50_ns": hyperliquid["intervals"]["tick_to_wire"]["p50_ns"],
            "hyperliquid_tick_p99_ns": hyperliquid["intervals"]["tick_to_wire"]["p99_ns"],
            "benchmark_p50_ns": benchmark["p50_ns_per_trace"],
            "benchmark_p99_ns": benchmark["p99_ns_per_trace"],
        }
        rows.append(row)

    feed_eligible = [row for row in rows if row["feed_eligible"]]
    local_eligible = [row for row in rows if row["local_pipeline_eligible"]]
    rankings = {
        **{metric: rank_rows(feed_eligible, metric) for metric in FEED_METRICS},
        **{metric: rank_rows(local_eligible, metric) for metric in LOCAL_METRICS},
    }
    selection = balanced_selection(feed_eligible, rankings)
    return {
        "schema_version": "ec2-latency-hunt-v2",
        "candidate_count": len(rows),
        "eligible_count": len(feed_eligible),
        "feed_eligible_count": len(feed_eligible),
        "local_pipeline_eligible_count": len(local_eligible),
        "binary_sha256": expected_binary_sha,
        "duration_seconds": 900,
        "warmup_messages": 100,
        "clock_error_bound_limit_us": CLOCK_BOUND_LIMIT_US,
        "hyperliquid_feed_network_ranking_eligible": False,
        "rows": rows,
        "rankings": rankings,
        "balanced_selection": selection,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and rank an EC2 latency hunt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.run_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.csv_output.open("w", newline="") as handle:
        fields = [
            "instance_type",
            "instance_id",
            "eligible",
            "feed_eligible",
            "local_pipeline_eligible",
            "clock_before_bound_us",
            "clock_after_bound_us",
            "clock_max_bound_us",
            *METRICS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: row[field] for field in fields})
    if report["eligible_count"] != report["candidate_count"]:
        raise SystemExit("one or more candidates failed validation")


if __name__ == "__main__":
    main()
