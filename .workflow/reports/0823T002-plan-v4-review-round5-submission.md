# 0823T002 H0-B V4 Round 5 Independent Review

- review_date: 2026-08-24
- reviewer_role: independent_read_only
- reviewer_actor_id: codex-independent-reviewer-0823T002-v4-round5-72257a3b
- disposition: REJECTED
- formal_build: NO-GO
- severity: P0/P1/P2/P3=0/1/0/0

## Frozen Identities

- candidate_commit: 72257a3b5a4c7ca91b6616ac64bee0784aa94164
- candidate_tree: 67c402ea1f1d1a5e63abdcc24d78ba4a45800a95
- candidate_parent: 83c0ac48ea4fa874679c6f18f776b6027596c47e
- receipt_commit: 7ad2e9e8c596046d111d071af38c278ce492829b
- receipt_path: .workflow/reports/0823T002-v4-candidate-receipt-round5.json
- receipt_sha256: adc8c904ab5ae26ac43e94d487cab5e04418b975034dfd8de9689de782ca4285
- plan_sha256: 03050180ba115bb9a93ada9c090989a47afbbb98bffe46448304c3ac4aca3826
- matrix_sha256: 8255d49917eaf0abedc62f74ef9609dfc1ccac9fea4e98abbcc764445ccee25d
- runtime_source_tree_sha256: 26a6b2abd1a7f0b344e1149fb01eee6beca9924fd2f7d6317c2a0ee41158c249
- immutable_execution_authority: 71adbfa678ff3646982160d220f5c223e0f7e59f

## Git Scope And Provenance

Candidate parent/tree identities passed. The candidate changes exactly nine
H0-B paths: the matrix, business report, task, V4 plan, runtime, package
tests, findings.md, progress.md and task_plan.md.

The receipt commit is a direct child of the candidate and introduces only the
Round 5 receipt. Candidate receipt/review/submission absence checks passed.
Current controlled bytes match the candidate Git blobs, and current receipt
bytes match the receipt-commit blob and declared SHA256.

The two staged hl_staleness documents and existing untracked historical
evidence were not present in either candidate or receipt commit. The original
worktree was not modified.

## Matrix And Hostile Verification

- surfaces: 65
- mutations: 89
- artifacts: 99
- task human-readable table equals canonical matrix rendering
- artifact projection: 99 entries, 99 unique path/surface assignments
- current hostile execution: 89
- frozen hostile execution: 89
- fail_open_count: 0
- current/frozen code contracts: exact match
- current/frozen normalized location contracts: exact match

The six formal-related locations reached their individual targets. macOS
`/var` canonicalization did not short-circuit the mutations. Root-symlink and
parent-chain-symlink cases executed as separate mutations.

Generic Trust Kernel compatibility passed: the V3 surface_contract remains
the exact three-field mutation/code contract. The H0-B validator additionally
checks exact current/frozen target_contract rows.

## Finding

P1: Seven mutation implementations do not construct the semantic mutation
declared by the Surface Matrix:

1. `mutate_interval_likelihood` uses an unknown branch instead of rounding
   interval bounds.
2. `mutate_horizon_straddle` supplies invalid straddle bounds instead of
   dropping a valid straddle row.
3. `mutate_design_matrix` selects unknown model `H2` instead of reordering or
   dropping an indicator.
4. `mutate_missing_value_policy` supplies all-NaN training data instead of
   using a test-set median.
5. `mutate_walk_forward` supplies reverse-sorted blocks instead of
   constructing a random split.
6. `mutate_numeric_seed_conventions` supplies an empty nearest-rank input
   instead of changing quantile/RNG conventions.
7. `mutate_rq3_km_ties` supplies a negative duration instead of changing
   event/censor tie ordering or interpolation.

These cases produce the declared error codes and locations, but fail before
exercising the promised negative behavior. Because expected target locations
are defined by the same runtime that constructs the mutations, the
`89 + 89 / fail_open_count=0` result does not independently close this
semantic gap.

## Tests And Preserved Research Identity

- focused core tests: 126 passed
- focused package tests: 65 passed
- Ruff: passed
- compileall: passed
- candidate diff check: passed
- generic Trust Kernel validator: passed
- old package verification: passed, zero_write=true
- old package tree: 42 files, 5 directories
- R: cfefe6b1d4e95a9caa5781984e5b75c0ce0f2bd528365fcc298d071e5adae2b4
- C: f9868b4a658e3cfac64af9849d9459b104e762ce0a78e3767b05a199608ce46e
- E: ddfcec05e49598e175687f14729bf61549e699d939db07a3c3617eb92aa23ea5
- composite: a196f3e743e8281c3cc5f82c4e30c57dc10af7b0c91e88ff16194065f10f7e05

No outcome, diagnostic, h0b0 or build-formal command was run. Research outcome
inputs were not read or recomputed.

