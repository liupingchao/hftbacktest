# 0529T002 Fill-Quality Bucket Recommendation

- overall_verdict: `needs_more_clean_fills`
- overall_reason: candidate axes exist but clean fill mass is below bucket thresholds
- shape_a_candidate_count: `0`
- shape_b_candidate_count: `0`
- ready_bucket_count: `0`
- needs_more_clean_fills_bucket_count: `68`
- reject_quality_negative_bucket_count: `246`
- next_recommended_task: `collect_or_refine_read_only_evidence_before_policy_design`

Boundary: read-only/default-off synthesis only. No live run, no replay run, no strategy behavior change, no parameter search, no guard relaxation, no default-on behavior, no tiny-live, and no promotion claim.

Proxy limitations: metrics are observed submitted-order labels from existing artifacts. The runner does not infer counterfactual fills, exact queue position, or hidden queue behavior.

Samples:
- 5-19-day-control-30min: status=usable, caveated=False, rows=4098, notes=stage6_manifest_missing
- 5-19-night-active-30min-a: status=usable, caveated=True, rows=1284, notes=stage6_manifest_missing
- 5-19-night-active-30min-b: status=usable, caveated=False, rows=2263, notes=stage6_manifest_missing
- 5-19-night-active-30min-c: status=usable, caveated=False, rows=2360, notes=stage6_manifest_missing
- 5-21-day-control-60min: status=usable, caveated=False, rows=5914, notes=none
- 5-26-active-minmove-control-30min-a: status=usable, caveated=False, rows=2526, notes=none
- 5-26-active-minmove-control-60min-a: status=usable, caveated=True, rows=5095, notes=none
- 5-26-active-makeredge-control-180min-a: status=usable, caveated=False, rows=7016, notes=none
- 5-26-active-minmove-control-30min-b: status=usable, caveated=False, rows=11089, notes=none
