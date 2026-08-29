# 0829T002 Plan Review Round 1

Date:
- 2026-08-29

Reviewed commit:
- `2dc0ada6`

Plan revision:
- Revision 1

Status:
- 未通过

Severity:
- P0: 0
- P1: 4
- P2: 2
- P3: 0

P1:
- Rolling 100/500ms ratios were incorrectly allowed to refresh channel TTL
  every checkpoint even without a new underlying channel contribution.
- The null did not freeze its conditional H0 or require complete detector
  pipeline recomputation from randomized causal features.
- Legitimate zero support, NOT_ESTIMABLE semantics and A-1 gate precedence
  conflicted; raw sparsity was incomplete.
- Source/cache authority and seven inherited callable AST identities were
  deferred rather than frozen in the reviewed plan.

P2:
- The delete-only common-anchor family is monotonic, but it is not the natural
  onset family of independently run stricter M-states. Orphan strict onsets
  require explicit diagnostic treatment and claim limits.
- Hostile tests omitted the load-bearing freshness, full-null-recompute,
  orphan-onset, zero-precedence and authority mutations.

Accepted parts:
- Selection/evaluation RNG banks are disjoint.
- Historical reuse and no-prospective claim are explicit.
- Future outcome, A0 and live/private permissions remain locked.
- The Required Outputs namespace is explicit.

Decision:
- Data execution lock remains active.
- Revision 2 must close all P1/P2 findings before another review.
