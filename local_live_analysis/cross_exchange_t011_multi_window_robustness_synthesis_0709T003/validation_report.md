# 0709T003 Multi-Window Robustness Synthesis

Final recommendation: `route_to_quote_fill_probability_evidence`

- Accepted window count: `4`
- Classification counts: `{'submitted_resting_no_fill': 3, 'submitted_rejected': 1}`
- Safety invariant counts: `{'pass': 4}`
- Replay acceptance counts: `{'pass': 4}`
- Economics support counts: `{'no_fill_fail_closed': 4}`
- Handoff/candidate-age evidence: see `multi_window_synthesis_matrix.csv` columns `handoff_phase`, `guard_candidate_age_seconds`, `max_candidate_age_seconds_at_phase_end`, and `fail_closed_reason_counts`.

No fills occurred, so fee/rebate/realized PnL remain unsupported. The accepted route is quote/fill probability evidence, not profitability or promotion.
