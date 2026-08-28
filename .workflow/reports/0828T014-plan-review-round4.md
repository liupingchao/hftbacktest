# 0828T014 Independent Plan Review Round 4

Status:
- 未通过

Scope:
- Read-only review of Revision 4.
- Zero future-price outcome access.

Severity:
- P0/P1/P2/P3 = 1/2/1/0

Findings:
- Encoding a directionless donor position as `+1` preserved numeric masks but
  could create artificial positive direction at a nonzero target checkpoint.
- Jittered minimum-cost derangement did not define a uniform randomization law
  and had no effective assignment-diversity floor.
- Comparable coverage lacked an exact numerator/denominator, and 10s/60s
  lacked observed-anchor/date support floors.
- The p95 joint-distance threshold was weaker than the edge calipers and could
  not fail.

Disposition:
- Formal execution remains locked.
- Revision 5 replaces checkpoint path transfer with exact paired swaps of
  genuinely nonzero microblock orientation labels and freezes support and
  assignment-diversity gates.
