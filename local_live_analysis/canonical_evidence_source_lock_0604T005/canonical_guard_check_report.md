# Canonical Guard Check Report

Task: `0604T005`

## Source Lock

- Formal evidence source: `0604T003` at `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Foundation loader source: `0604T004` at `/home/molly/project/hftbacktest/local_live_analysis/canonical_event_mode_evidence_0604T004`
- Guarded input: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Required decision mode: `event`
- Required canonical status: `canonical_event_mode`
- Diagnostic-only status: `diagnostic_only_synthetic_decision_grid`

## Result

- Guard status: `accepted_formal_canonical_evidence`
- Canonical sample count: `3`
- Diagnostic rejection count: `0`
- Downstream ranking and horizon/regime diagnostics must call this guard before consuming evidence.

## Negative Validation

- diagnostic_source_without_override: `rejected_formal_evidence` canonical_sample_count=`0` diagnostic_rejection_count=`3`
- diagnostic_source_with_negative_validation: `diagnostic_only_validation` canonical_sample_count=`0` diagnostic_rejection_count=`3`

## Boundary

- Read-only source-lock / guard hardening only.
- No new collection, signal ranking, regime selection, private/order endpoints, order lifecycle, strategy implementation, live/default-on/tiny-live, parameter search, or promotion is authorized.
