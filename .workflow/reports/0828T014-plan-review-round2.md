# 0828T014 Independent Plan Review Round 2

Status:
- 未通过

Scope:
- Read-only review of Revision 2 of the A-1 audit contract.
- Zero future-price outcome access.

Severity:
- P0/P1/P2/P3 = 1/4/1/0

Findings:
- Five-minute Rademacher sign flips require unregistered sign
  exchangeability, alter signed direction marginals and introduce asymmetric
  dwell truncation at randomized boundaries.
- The candidate allowed stale, scattered conflict exposure rather than a
  directly adjacent conflict-to-coherence transition.
- The horizon-conflict family used a medium composite containing trade and
  therefore was not necessarily trade-depth conflict.
- The availability denominator was circular and did not measure the
  trade-plus-depth support required by the primary predicate.
- The task and plan named different structural nulls.
- Two near-continuity gates were largely guaranteed by the detector's
  one-second refractory.

Disposition:
- Data execution remains locked.
- Revision 3 replaces sign flips with activity-matched direction-path
  microblock permutation, applies symmetric boundary censor, requires a
  contiguous adjacent component conflict, makes horizon conflict diagnostic,
  fixes the availability denominator and evaluates density before refractory.
