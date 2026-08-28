# SKHYNIX Binance LIQUIDITY_BREAK_ONSET_V1 A1 Target Support Audit Plan - 2026-08-28

Date: 2026-08-28

Status:

```text
plan frozen
execution blocked at preflight
future target access remains zero
```

Hypothesis:

```text
LIQUIDITY_BREAK_ONSET_V1
```

Upstream authority:

```text
docs/skhynix_binance_liquidity_break_onset_v1_a0_causal_anchor_contract_20260828.md
```

Upstream result:

```text
A0 classification: A0_anchor_near_continuous
A1_authorized: false
```

## 1. Decision

A1 is designed but not unlocked.

The frozen stage chain requires:

```text
A0_causal_anchor_contract_supported
  -> authorize A1 target materialization
```

The actual result is:

```text
A0_anchor_near_continuous
  -> block A1 before any future-price read
```

Therefore this document freezes the A1 protocol so the intended audit is
explicit, but no target ledger may be materialized for
`LIQUIDITY_BREAK_ONSET_V1`.

## 2. A1 Purpose

A1 is a target-support and timeliness audit. It is not a predictive-model
stage.

If unlocked, A1 would answer:

1. Are both frozen competing-risk outcomes observable often enough?
2. Is the causal onset early enough relative to the first market transition?
3. Are censoring, reset boundaries and simultaneous outcomes controlled?
4. Do anchors and matched controls share a usable target risk set?
5. Is there enough per-date support to justify A2 target-variation closure?

A1 would not answer:

```text
does H1 beat H0
is the signal profitable
is maker execution feasible
does the effect transport prospectively
```

## 3. Preflight Gate

Required inputs:

```text
classification.json:
  classification == A0_causal_anchor_contract_supported
  A1_authorized == true

contracts/outcome_access_ledger.json:
  targets_materialized == false
  H0_H1_fitted == false
  future fields == []

run_manifest.json:
  full artifact closure
```

Current preflight:

```text
classification == A0_anchor_near_continuous
A1_authorized == false
```

Canonical A1 preflight classification:

```text
A1_blocked_by_A0
```

Required action:

```text
stop before reading future midpoint or best-price transitions
```

## 4. Frozen Audit Population

If A1 were unlocked, the primary population would be:

```text
all A0 primary onset anchors
  +
their no-reuse outcome-blind matched controls
```

Each row must retain:

- capture and research date;
- role;
- exact decision `event_key`;
- direction;
- segment and reset boundary;
- `precursor_at`;
- `start_anchor`;
- current OBI, spread, depth and activity support state;
- matching stratum and relaxation level;
- selected `tau_max`.

No row may be removed using its future outcome.

## 5. Frozen Competing Risks

At `start_anchor=t0`:

```text
m0 = causal midpoint at t0
tick = frozen tick size
```

### 5.1 Adverse Continuation

```text
d = +1:
  midpoint first reaches m0 + 1 tick

d = -1:
  midpoint first reaches m0 - 1 tick
```

### 5.2 Liquidity Recovery

Let:

```text
Q_v,baseline =
  median weighted vulnerable-side depth over [t0-500ms,t0-100ms]
```

Recovery is first causally complete when:

```text
weighted vulnerable-side depth >= 0.8 * Q_v,baseline
same-direction pressure_score < 1.5
same-direction coherence_count == 0
all conditions persist for 100ms
no prior adverse continuation
```

Recovery is not backdated.

### 5.3 Censoring

Right-censor at the earliest of:

- `tau_max=10000ms`;
- capture end;
- sequence gap;
- snapshot reset;
- source-quality failure.

If adverse continuation and recovery become identifiable at the same ordered
event:

```text
target_status = interval_ambiguous
```

Do not pointify the event.

## 6. Event Ordering And Information Boundary

Target replay must use:

```text
(local_receive_ts_ns,event_seq_in_file)
```

Rules:

1. Preserve file order for equal receive timestamps.
2. Apply the current message before evaluating a target transition.
3. Do not use exchange timestamps that were not yet locally received.
4. Do not fill a target across reset or sequence-gap boundaries.
5. Do not move `start_anchor` after seeing the target.
6. Do not change target barriers, recovery fraction, dwell or `tau_max`.

## 7. Frozen Timeliness Audit

At `precursor_at`:

```text
m_precursor = causal midpoint at precursor_at
```

Define:

```text
pre_detection_adverse =
  d = +1 and midpoint reaches m_precursor + 1 tick before start_anchor
  or
  d = -1 and midpoint reaches m_precursor - 1 tick before start_anchor
```

Primary timeliness gates:

```text
pre_detection_adverse_fraction <= 0.30

median post-anchor identified transition time
  >= 3 * median precursor_to_detection_ms
```

Failure classification:

```text
causal_anchor_too_late_for_target
```

Action:

```text
stop before H0/H1 fitting
```

## 8. Frozen Target-Support Gates

These gates are frozen before target access.

### Gate A1-0: Upstream Authorization

```text
A0 classification == A0_causal_anchor_contract_supported
A1_authorized == true
```

### Gate A1-1: Exact Join Closure

Require:

- every anchor/control row joins to one source capture;
- exact decision event is found;
- direction and segment are unchanged;
- zero duplicate target identities;
- zero target traversal across reset boundaries.

### Gate A1-2: Cause Support

Require:

```text
minimum adverse events overall:       500
minimum recovery events overall:      500
minimum represented dates per cause:  8
minimum events per represented date:  25
maximum single-date cause share:      0.35
```

### Gate A1-3: Censoring And Ambiguity

Require:

```text
overall right-censor fraction <=       0.30
minimum per-date identified fraction >= 0.60
interval-ambiguous fraction <=         0.02
quality/reset censor fraction <=       0.05
```

### Gate A1-4: Timeliness

Require both frozen timeliness gates in Section 7.

### Gate A1-5: Common Risk Set

Require:

```text
minimum unique anchor-control pairs:    500
overall target-eligible pair fraction:  0.90
minimum per-date eligible pair fraction: 0.75
maximum single-date eligible-pair share: 0.35
```

Eligibility may use source and censoring geometry, but it may not select rows
by which cause occurred.

## 9. Required Outputs

If unlocked, A1 must publish:

```text
contracts/
  A1_source_manifest.json
  target_materialization_contract.json
  event_ordering_contract.json
  timeliness_gate_contract.json
  target_support_gate_contract.json
  outcome_access_ledger.json

support/
  target_state_ledger.csv
  competing_risk_target_ledger.csv
  target_support_by_date.csv
  target_support_by_role.csv
  target_time_distribution.csv
  censoring_reason_composition.csv
  interval_ambiguity_ledger.csv
  pre_detection_adverse_ledger.csv
  timeliness_support_by_date.csv
  target_eligible_matched_pairs.csv
  unmatched_or_ineligible_pairs.csv

reports/
  A1_target_support_summary.json

classification.json
run_manifest.json
```

Large event ledgers remain outside Git. Git tracks compact contracts,
summaries, support tables and exact hashes.

## 10. A1 Classifications

```text
A1_blocked_by_A0
A1_source_join_failed
A1_target_support_insufficient
A1_target_censoring_excessive
A1_target_simultaneity_ambiguous
causal_anchor_too_late_for_target
A1_common_risk_set_insufficient
A1_target_support_supported
```

Only:

```text
A1_target_support_supported
```

may authorize A2.

## 11. Explicitly Forbidden Rescue

After any target is visible, do not:

- increase the onset threshold to reduce anchor density;
- change the 50ms pressure window;
- change the two-of-three rule;
- modify active release or opposite-switch semantics;
- discard adverse rows as microstructure noise;
- redefine recovery using favorable future depth;
- shorten or lengthen `tau_max` using event rates;
- select dates, directions or component pairs by target balance;
- replace the failed A0 anchor with an offline change point;
- describe a modified detector as the same hypothesis version.

Any detector change requires:

```text
LIQUIDITY_BREAK_ONSET_V2
or another new hypothesis identifier
```

and a new zero-target A0.

## 12. Current Execution Record

As of 2026-08-28:

```text
A1 preflight executed
A1 target materialization not executed
future midpoint fields read: []
future best-price fields read: []
targets materialized: false
H0/H1 fitted: false
classification: A1_blocked_by_A0
```

The blocker is substantive:

```text
the frozen onset detector emits 340,068 anchors
at 9,468.1098 anchors/hour
and is classified A0_anchor_near_continuous
```

This means target support would measure reactions to a near-continuous
threshold-crossing process, not to a validated alignment landmark.

## 13. Next Research Boundary

Do not proceed to A1 for this version.

If alignment research continues, the next proposal must first define a new
interpretable M-state with a genuine novelty or persistence condition, then
repeat zero-target A0.

Examples of admissible new semantics include:

- entry into a pressure state that was absent for a frozen refractory period;
- a persistent break in replenishment capacity, not a momentary z crossing;
- a directionally coherent pressure state with an independently frozen
  survival or hysteresis condition;
- a structural transition defined relative to a causal background model,
  with thresholds frozen before outcome access.

These are new hypotheses, not robustness checks of
`LIQUIDITY_BREAK_ONSET_V1`.
