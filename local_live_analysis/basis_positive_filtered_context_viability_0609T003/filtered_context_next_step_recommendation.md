# Filtered Context Next Step Recommendation

Task: `0609T003`

- Final recommendation: `candidate_for_read_only_case_design`
- Reason: clean filtered basis-positive context improves wrong-way p95 loss and shows no sample/horizon reversal
- Clean context rows: `3545`, samples: `7`, mean future move: `43.39492243` ticks.
- Clean context p95 wrong-way loss improvement vs raw: `34` ticks.
- Rejected tail-risk rows: `2181`, wrong-way count: `63`.

## Boundary

- This recommendation is read-only public observation-layer research only.
- It does not authorize strategy implementation, private/order endpoint use, order lifecycle, case-library implementation, shadow decision generation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.
