# 0518T004 Step 5C Quote-Anchor Safety Diagnostic

## Boundary

- Mode: default-off / diagnostic-first candidate.
- Top5 is not used as the final hard post-only anchor.
- This does not repair audit_depth/bookTicker/top5 row-exact drift.
- This does not start live, change replay lifecycle, or prove production readiness.

## Key Counts

- decision rows: `96341`
- bookTicker anchor rows: `79449`
- guarded depth fallback rows: `16892`
- missing anchor rows: `0`
- stale anchor rows: `0`
- bid clamped rows: `4530`
- ask clamped rows: `3278`
- suppress buy rows: `0`
- suppress sell rows: `0`
- post-only risk after re-check rows: `0`

## Interpretation

The candidate safety layer uses bookTicker when fresh, guarded depth fallback when bookTicker is unavailable or stale, and otherwise suppresses fresh add-side submits. Clamp and post-clamp re-check leave zero post-only/crossed-risk rows in this diagnostic.
