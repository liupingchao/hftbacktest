# Canonical Evidence Validation Report

Task: `0604T004`

## Scope

- Input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_event_mode_evidence_0604T004/synthetic_diagnostic_validation`
- Inputs are existing local `0604T003` aggregate artifacts only.

## Result

- Source sample count: `3`
- Canonical event-mode sample count: `0`
- Diagnostic rejection count: `3`
- Required files and columns passed validation.
- Canonical evidence requires `decision_mode=event` and `canonical_status=canonical_event_mode`.
- Synthetic fixed-grid samples are parseable diagnostics only and are excluded from canonical outputs.

## Boundary

- No new data collection, signal ranking, regime selection, case-library construction, or shadow decision generation was performed.
- No private/order endpoints, order lifecycle, strategy implementation, parameter search, live/default-on/tiny-live, or promotion is authorized.
