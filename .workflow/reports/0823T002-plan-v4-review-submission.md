# 0823T002 H0-B V4 Round 6 Independent Review

- review_date: 2026-08-24
- reviewer_role: independent_read_only
- reviewer_actor_id: codex-independent-reviewer-0823T002-v4-round6-180572c6
- controller_actor_id: codex-main-controller
- disposition: ACCEPTED
- formal_build: GO
- final_severity: P0/P1/P2/P3=0/0/0/0

## Exact Identities

- candidate_commit: 180572c669b4d1259eb26b9d268fa6335c0780a1
- candidate_tree: 5aee4d64dc6ce280b94e9d74afb6d24f6f0ccbaf
- candidate_parent: 848af12895736fac84ad0a71a5167bc3c2b3e4ef
- receipt_commit: a394c7f2d84b817227966627b45b573b397d2198
- receipt_path: .workflow/reports/0823T002-v4-candidate-receipt-round6.json
- receipt_sha256: 246d4dfd3caca72b1db9cdff2070bd3fafcd41a8531cc1da4c8ce30c6a0df659
- reviewed_plan_sha256: c1d78dbdd00adf34a06973dd8e7d4b5e759c5ad1b07bd2f4a2618d0e909658a7
- reviewed_surface_matrix_sha256: 8255d49917eaf0abedc62f74ef9609dfc1ccac9fea4e98abbcc764445ccee25d
- reviewed_hostile_target_contract_sha256: 67e1977b6d29a07f86e6542eb5c2a8a8fc70fc4e42fc1f296213c123f2e6ba72
- reviewed_runtime_source_tree_sha256: 800fb6380a5117139ccde27f1b1212f0e624ff2ea7fe2565861be6cd5bc54a1d
- immutable_execution_authority: 71adbfa678ff3646982160d220f5c223e0f7e59f

## Findings

No findings.

- P0: 0
- P1: 0
- P2: 0
- P3: 0

## Git Scope And Provenance

The candidate changes exactly nine H0-B paths: the new hostile target
contract, business report, task, V4 plan, runtime, package tests, findings.md,
progress.md and task_plan.md.

The receipt commit is a direct child of the candidate and introduces exactly
one path: the Round 6 candidate receipt. Candidate tree/parent, receipt
ancestry, receipt introduction blob and current-byte bindings all passed.

The candidate lacks its receipt, review and review submission. The receipt
commit still lacks the review and submission. The receipt additionally binds
the exact hostile target contract path and SHA256 to the candidate Git blob.

The Round 5 rejection commit and its `P0/P1/P2/P3=0/1/0/0` disposition remain
unchanged in candidate ancestry.

## Surface And Target Contracts

- surfaces: 65
- negative mutations: 89
- artifacts: 99
- hostile target rows: 89
- semantic probe rows: 7
- task human-readable Surface Matrix: exact canonical match

All 89 target rows match the canonical matrix in order and bind exact
`surface_id`, target, operation, description and `expected_error_code`
fields. Eighty-two locations preserve the reviewed Round 5 locations; only
the seven remediated semantic targets changed.

The runtime no longer defines `HOSTILE_EXPECTED_ERROR_LOCATIONS`. Expected
locations and semantic probes are loaded from the reviewed, SHA-pinned hostile
target contract.

## Semantic Mutation Audit

The seven remediated mutations use valid canonical inputs before constructing
the declared mutation:

1. Interval bounds: exact likelihood `0.018965243697369294` versus
   rounded-grid likelihood `0.19000000000000006`.
2. Horizon straddle: a valid straddle row is computed, then row count changes
   from 1 to 0.
3. Design matrix: a valid 11-column H0 design is built, then a real
   missing-indicator column is removed, producing 10 columns.
4. Missing values: training median `0.0` is replaced by test-fold median
   `101.0`.
5. Walk-forward: canonical chronological 60/20 blocks are replaced by a
   PCG64-permuted random split.
6. Numeric conventions: nearest-rank `10.0` with PCG64 is replaced by linear
   quantile `15.0` with MT19937.
7. KM ties: valid tied durations produce canonical event-first median `20.0`
   versus censor-first median `10.0`.

None of the seven cases fails first on an unknown branch/model, empty input,
invalid ordering or negative duration.

## Verification Results

- core test suite: 126 passed
- package test suite at exact candidate: 73 passed
- Ruff: passed
- compileall: passed
- candidate diff check: passed
- candidate receipt validator: passed
- current hostile mutations: 89
- frozen hostile mutations: 89
- current/frozen target rows: 89/89 exact
- current/frozen semantic probes: 7/7 exact
- fail_open_count: 0
- Generic Trust Kernel validator: verified, 65/89/99
- hostile receipt `surface_contract` keys: `mutation_id`,
  `expected_error_code`, `error_code`

The v4 hostile receipt remains compatible with the generic Trust Kernel's
strict three-field code contract. The H0-B validator additionally verifies
the exact current/frozen target and semantic-probe contracts.

The complete-package portability mutation reached `H0B_BUILD_MISMATCH` at
`outcome_access_permit_build_a.json` under runtime SHA256
`800fb6380a5117139ccde27f1b1212f0e624ff2ea7fe2565861be6cd5bc54a1d`.

All six formal-related normalized locations matched. macOS `/private/var`
resolution did not short-circuit symlink checks, and root-symlink and
parent-chain-symlink mutations remained independent.

## Existing Package

- verified: true
- file_count: 42
- directory_count: 5
- zero_write: true
- R: cfefe6b1d4e95a9caa5781984e5b75c0ce0f2bd528365fcc298d071e5adae2b4
- C: f9868b4a658e3cfac64af9849d9459b104e762ce0a78e3767b05a199608ce46e
- E: ddfcec05e49598e175687f14729bf61549e699d939db07a3c3617eb92aa23ea5
- composite: a196f3e743e8281c3cc5f82c4e30c57dc10af7b0c91e88ff16194065f10f7e05

The package tree and all four identities are unchanged from immutable
execution authority commit `71adbfa678ff3646982160d220f5c223e0f7e59f`.

## Review Boundaries

No outcome, diagnostic, h0b0 or build-formal command was run. Real research
outcome inputs were neither mounted, read nor recomputed. One-byte
non-research placeholders were used only to satisfy Stage 4 path-stat metadata
inside isolated portability fixtures.

The original worktree remains unchanged. It contains two staged hl_staleness
documents and nine unrelated untracked historical evidence/document paths.
None entered the candidate or receipt commit, and they were excluded from all
frozen-scope judgments.

A supplemental package-suite run at the receipt introduction commit produced
the expected candidate-state assertion that a pending-review candidate must
not yet contain its receipt. The canonical candidate suite passed 73/73, and
the dedicated receipt validator passed independently.

## Final Disposition

ACCEPTED. `P0/P1/P2/P3=0/0/0/0`. Formal build GO.

