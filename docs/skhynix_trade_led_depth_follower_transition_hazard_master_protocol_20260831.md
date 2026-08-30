# SKHYNIX Trade-Led Depth-Follower Transition Hazard Master Protocol

Date: 2026-08-31

Protocol ID:
`TRADE_LED_DEPTH_FOLLOWER_TRANSITION_HAZARD_MASTER_V1`

Status:
`MASTER_PROTOCOL_DRAFT`

Immediate execution authorization:
`Q0_ONLY`

## 1. Purpose

This protocol defines the complete staged route for testing whether a causal
trade-flow onset contains incremental information about a subsequent
order-book transition and, only after structural acceptance, about a
subsequent price transition.

The route is:

```text
Q0  pipeline qualification
  -> A-1a causal-anchor, risk-set and overlap support audit
  -> A-1b structural competing-risk H0/H1 test
  -> A0 conditional price first-passage H0/H1 test
```

The complete methodology is registered now. Only Q0 is immediately
executable. Every later stage remains locked until its predecessor produces
the exact accepted artifact required by this protocol.

This document is a research protocol, not a formal task dispatch, claim,
runner, data authorization or trading authorization.

## 2. Inherited Evidence and Non-Result

The predecessor hypothesis was:

```text
FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1
```

Its intended primary was `TRADE_LED`. Its confirmation rule required:

```text
at least one additional same-direction leader update within 200ms
and no explicit opposite update from any channel
```

The two formal attempts did not produce an A/B/P scientific package:

```text
0830T002:
  interrupted before Build A

0830T003:
  passed baseline authority preflight
  entered Build A
  interrupted before Build A publication
```

Both therefore ended with:

```text
scientific classification = NONE
prediction evaluation = NOT_EVALUATED
```

No support count, date coverage, concentration statistic, transition effect
or price result exists. This protocol must not treat either interruption as
support for or evidence against the scientific hypothesis.

## 3. Scientific Correction

The predecessor confirmation tested leader self-persistence:

```text
trade onset
-> another same-direction trade update
```

It did not require a later state change in depletion or OFI. It therefore
could not establish that trade flow was a temporal leader of a depth response.

The new hypothesis replaces self-repeat as the endpoint:

```text
causal TRADE onset
-> first same-direction depth response
   versus first explicit contradiction
   versus right censoring
```

Additional same-direction trade updates remain descriptive post-anchor
evidence. They cannot satisfy the structural follower endpoint and cannot
enter H0 or H1 in V1.

## 4. Claim Ladder

The protocol permits only the following ordered claims.

### Q0 claim

```text
the exact research pipeline can complete its registered evidence package
on non-scientific qualification fixtures
```

### A-1a claim

```text
the frozen causal M-state, matched-control risk sets and date support
are estimable in the registered historical dataset
```

### A-1b claim

```text
the causal TRADE arrival path provides out-of-fold incremental information
about a later depth-state competing-risk transition
```

### A0 claim

```text
the causal TRADE arrival path provides out-of-fold incremental information
about a later midprice first-passage transition
```

None of these claims establishes fill quality, fee-adjusted profitability,
maker safety, deployment readiness or cross-market transportability.

## 5. Stage Locks

| Stage | Initial status | Unlock requirement |
|---|---|---|
| Q0 | `UNLOCKED_FOR_TASK_DRAFT` | Independently reviewed Q0 task |
| A-1a | `LOCKED_PENDING_Q0_PASS` | Accepted Q0 terminal package |
| A-1b | `LOCKED_PENDING_A_MINUS1A_ESTIMABLE` | Frozen A-1a anchor/control manifest |
| A0 | `LOCKED_PENDING_A_MINUS1B_PASS` | Accepted A-1b structural result |

Failure behavior is exact:

```text
Q0 FAIL:
  stop; no scientific task is authorized

A-1a NOT_ESTIMABLE:
  stop; do not relax M, matching or date-support definitions

A-1b FAIL:
  stop; do not open future price outcomes

A0 FAIL:
  record the result; do not tune the structural hypothesis
```

No failed stage may be repaired inside a consumed one-shot task. Any software
repair requires a separate non-scientific successor task and a new
qualification result before a new scientific task can be registered.

## 6. Shared Data and Information Boundary

All stages must bind an immutable raw-cache inventory before execution.

Permitted structural inputs are limited to contemporaneous or historical:

```text
timestamps and segment identity
trade-flow features
depletion features
OFI features
OBI and book-depth state
spread
activity and event intensity
short-horizon realized volatility
time-of-day
data-validity and reconnect state
```

Q0 may use only synthetic or explicitly registered non-scientific fixture
caches.

A-1a may inspect:

```text
M-state construction
anchor-time covariates
control-pool covariates
future observation availability
segment and gap censoring capacity
```

A-1a must not inspect:

```text
post-anchor follower direction
post-anchor competing-risk cause
future midprice
future return
fill, fee or PnL
```

A-1b may inspect future structural channel transitions but must not inspect
future price, return, fill, fee or PnL.

A0 may inspect only its separately frozen midprice first-passage outcomes. It
must not inspect fill, fee, queue outcome or PnL.

### 6.1 New staged feature authority

The predecessor `build_features` authority consumed flow fields only and
explicitly prohibited reading OBI, spread, depth and midpoint values. It
cannot serve as the complete feature authority for this protocol because H0
must condition on the observable current market state.

Q0 must therefore qualify a new versioned feature authority. It may reuse
accepted predecessor callables, but it must register new files and callable
AST identities rather than editing or silently extending the predecessor
authority.

The structural causal builder may consume exactly:

```text
activity
ask_depletion
ask_depth
bid_depletion
bid_depth
event_seq
midpoint
obi
ofi
ofi_abs
ready
segment_id
spread_ticks
tick_size
trade_signed
trade_total
ts_ns
valid_book
```

Midpoint access in the structural builder is restricted to trailing
covariates ending at the current checkpoint. Future midpoint first-passage
labels require a separate A0 outcome-builder process and are prohibited in
Q0, A-1a and A-1b model inputs.

All raw fields outside this exact staged allowlist remain poison authority.
Q0 must prove that changing them cannot change any causal feature, anchor,
control, structural outcome or model input.

## 7. Shared Clock and Causal Order

The canonical checkpoint is:

```text
20ms
```

At checkpoint `t`, the order is:

1. validate source, segment and book state at `t`;
2. process channel actions generated by evidence available at or before `t`;
3. update channel memories, including explicit neutral overwrite and expiry;
4. compute M-state preconditions using only checkpoints before `t`;
5. evaluate the anchor-time TRADE onset using evidence at `t`;
6. record anchor-time snapshot covariates;
7. create a provisional causal anchor or matched-control risk-set entry;
8. apply only outcome-blind availability finalization, such as complete-epoch
   and future-observation-capacity checks;
9. only later stages may inspect structural events after `t`.

No future market value, direction, structural cause or price may admit,
reject, thin, match or weight an anchor. Future availability may only
invalidate or censor a provisional anchor/control symmetrically; it cannot
turn an otherwise ineligible checkpoint into an eligible one.

## 8. Fixed-Epoch Alignment Infrastructure

The fixed causal epoch remains alignment and deterministic-suppression
infrastructure:

```text
epoch origin = Unix epoch 0
epoch width = 60s
checkpoint = 20ms
eligible anchor core = [epoch_start + 15s, epoch_start + 45s)
```

The fixed epoch determines:

```text
anchor eligibility
deterministic thinning identity
dependence clustering
slice/reset reproducibility
```

It does not impose a 200ms scientific response horizon.

Post-anchor structural observation may continue beyond the anchor core and
across the next epoch boundary if:

```text
the same source segment remains valid
there is no registered data gap
the maximum 60s observation time has not elapsed
```

The dependence cluster remains the anchor's:

```text
capture_id : anchor_epoch_id
```

## 9. Primary M-State

The unique primary M-state is `TRADE_LED`.

For direction `d` in `{-1,+1}`, checkpoint `t` is a primary causal M-state
iff all conditions hold.

### 9.1 Leader onset

```text
trade action at t = NEW_d
```

Frozen directional amplitudes are inherited:

```text
fast 100ms signed ratio:
  >= +0.50 for d=+1
  <= -0.50 for d=-1

medium 500ms signed ratio:
  >= +0.25 for d=+1
  <= -0.25 for d=-1
```

Threshold equality is admitted. The protocol does not authorize lowering
these thresholds.

### 9.2 Leader prestate

At:

```text
t-120ms, t-100ms, ..., t-20ms
```

the trade channel must be observable and exactly `BACKGROUND`.

Unknown, stale, directional, invalid or cross-segment prestate does not
qualify.

### 9.3 Follower anchor-time state

At `t`, depletion and OFI must each be observable and non-directional:

```text
depletion memory = BACKGROUND or explicit NEUTRAL
OFI memory       = BACKGROUND or explicit NEUTRAL
```

The following do not qualify:

```text
same-direction follower state
opposite-direction follower state
stale follower state
unknown/unavailable follower state
```

This anchor-time requirement is intentionally precision-first. It proves only
that no follower response was already visible at the alignment point. It does
not require a full historical three-channel path.

### 9.4 Source and epoch admission

At `t`, a provisional anchor must also satisfy:

```text
valid source and book state
one segment across leader prestate and t
t inside the registered anchor core
```

Offline A-1a finalization retains the provisional anchor only if its fixed
epoch is complete. This uses future availability, not a future market value.
The same finalization is applied to controls. An incomplete epoch may delete
a provisional entry but may never create one.

## 10. Deterministic Anchor Thinning

For every:

```text
(capture_id, anchor_epoch_id, direction)
```

retain the earliest admitted M-state by:

```text
(anchor_ts_ns, anchor_event_seq)
```

All later M-states in the same key are suppressed and reported.

Opposite directions in the same epoch remain in one dependence cluster.

No later event or outcome may replace an earlier retained anchor.

## 11. Structural Competing Risks

The A-1b observation process starts strictly after `t`.

### 11.1 Primary same-direction follower event

`DEPTH_FOLLOWER_SAME` occurs at the first checkpoint where either depletion
or OFI emits `NEW_d`.

Depletion and OFI are both depth-derived. They form one composite primary
follower event, not two independent votes.

The event ledger records:

```text
first follower channel
first follower timestamp and event sequence
whether the other depth channel joined later
```

### 11.2 Explicit contradiction event

`EXPLICIT_CONTRADICTION` occurs at the first checkpoint where any channel
emits `NEW_-d`.

### 11.3 Tie precedence

If same-direction depth evidence and any opposite evidence first appear at
the same checkpoint:

```text
EXPLICIT_CONTRADICTION wins
```

This is a precision-first convention. Ambiguous simultaneous evidence cannot
be counted as confirmation.

If depletion and OFI both emit `NEW_d` at the same checkpoint, the result is
one `DEPTH_FOLLOWER_SAME` event with a two-channel detail flag.

### 11.4 Leader self-repeat

Additional trade `NEW_d` updates after `t`:

```text
do not satisfy the follower endpoint
do not reset the structural clock
do not create a new anchor
are reported only as post-anchor descriptive evidence
```

An opposite trade update is an `EXPLICIT_CONTRADICTION`.

### 11.5 Censoring

The risk set is right-censored by the earliest of:

```text
t + 60s
segment boundary
registered source gap
book invalidation
end of available capture
```

Censoring is not a third scientific event cause. It must not be represented
as same-direction support or contradiction.

## 12. Market-Determined Time Scale

The protocol does not select one winning fixed horizon.

The initial piecewise-constant baseline-hazard bins are:

```text
[20ms, 80ms)
[80ms, 320ms)
[320ms, 1.28s)
[1.28s, 5.12s)
[5.12s, 20.48s)
[20.48s, 60s]
```

The bins span subsecond propagation through minute-scale structural
continuation with only six baseline parameters per cause.

### 12.1 Deterministic support merging

Within each outer training fold, a bin lacks support if either M or matched
control risk sets contribute fewer than:

```text
20 at-risk dependence clusters at bin start
```

Unsupported bins merge only rightward, from shortest to longer duration.
The last bin may merge leftward if it remains unsupported.

The merging algorithm, not the realized merged bins, is frozen now. Realized
bins are training-fold artifacts and must be applied unchanged to that fold's
held-out date.

No event effect, price outcome, p-value or model score may influence merging.

## 13. Matched Controls

The purpose of matched controls is to distinguish:

```text
arrival path into the current state
```

from:

```text
the current state itself
```

### 13.1 Control eligibility

A control checkpoint must:

```text
belong to the same research date and fixed epoch as its anchor
have valid source and book state
have enough future observation capacity for the same censoring contract
not be a TRADE NEW_d onset
```

Controls may be in a persistent directional trade state. This is intentional:
the snapshot may resemble the anchor while the causal arrival path differs.

### 13.2 Signed normalization

All directional covariates are multiplied by anchor direction `d`, so that:

```text
positive = aligned with the registered anchor direction
negative = opposed to the registered anchor direction
```

### 13.3 Matching variables

Coarsened exact matching uses only anchor-time state:

```text
research date
fixed epoch
spread in ticks
signed OBI decile
total visible depth decile
activity decile
short-horizon volatility decile
fast trade-ratio bin
medium trade-ratio bin
future observation-capacity bin
30-minute time-of-day block
```

All bins are learned from outer-training dates only and applied unchanged to
the held-out date.

Future observation capacity uses only source availability and is binned by
the registered initial hazard-bin edges. It may shorten or censor a risk set;
it contains no structural direction or price value.

### 13.4 Matching ratio and replacement

For each M anchor:

```text
maximum controls = 3
matching without replacement within a research date
```

Ties are resolved deterministically by:

```text
SHA256(anchor identity, control identity, protocol ID)
```

### 13.5 Frozen fallback ladder

Matching attempts occur only in this order:

```text
L0:
  exact registered bins

L1:
  allow visible-depth decile difference <= 1

L2:
  additionally allow volatility decile difference <= 1

STOP:
  no further relaxation
```

Every anchor records the first successful level or `UNMATCHED`.

### 13.6 Overlap support

A-1a is estimable only if:

```text
overall matched-anchor share >= 0.80
each represented date matched-anchor share >= 0.60
combined L1 and L2 fallback share <= 0.60 of matched anchors
```

Failure is `Aminus1a_control_overlap_not_estimable`. It does not authorize
new calipers or a new fallback level.

### 13.7 Matched-set weights

Each anchor has analysis weight:

```text
1.0
```

If it has `k` matched controls, each control has weight:

```text
1.0 / k
```

The total control weight for every matched set is therefore one. The weights
are used in H0/H1 fitting, held-out scores, cumulative-incidence summaries
and bootstrap statistics. A matched set is the smallest permutation unit;
epoch remains the outer dependence cluster.

## 14. H0 Baseline Model

H0 represents information available from the current snapshot.

The cause-specific discrete-time competing-risk model includes:

```text
piecewise baseline time-bin intercepts
signed fast trade ratio
signed medium trade ratio
signed OBI
spread in ticks
log visible depth
log activity
short-horizon realized volatility
time-of-day sine and cosine
```

All continuous covariates are robust-standardized using outer-training dates
only.

No future structural or price field may enter H0.

Robust standardization is exact:

```text
center = outer-training median
scale  = 1.4826 * outer-training MAD
```

If `MAD = 0`, use `scale = 1.0`. The feature remains present; it is not
dropped or replaced.

### 14.1 Exact H0 transforms

For anchor direction `d`:

```text
signed_fast_trade_ratio
  = d * trade_ratio_100ms(t)

signed_medium_trade_ratio
  = d * trade_ratio_500ms(t)

signed_obi
  = d * obi(t)

spread_ticks
  = canonical finite spread_ticks(t)

log_visible_depth
  = log1p(max(0, bid_depth(t)) + max(0, ask_depth(t)))

log_activity
  = log1p(sum(activity over [t-500ms, t]))

trailing_realized_volatility
  = sqrt(sum of squared log-midpoint changes over (t-1s, t])

time_of_day_sin
  = sin(2*pi*UTC_seconds_of_day(t)/86400)

time_of_day_cos
  = cos(2*pi*UTC_seconds_of_day(t)/86400)
```

Rolling values must stop at `t`, reset at segment boundaries and become
missing if their complete required history is unavailable. A missing H0
value makes the anchor/control ineligible; imputation is prohibited in V1.

## 15. H1 Arrival-Path Model

H1 is H0 plus one frozen arrival-path group:

```text
is_causal_trade_onset
joint_threshold_overshoot
signed_fast_minus_medium_acceleration
leader_background_run_length
time_since_last_opposite_trade_update
```

For matched controls:

```text
is_causal_trade_onset = 0
onset-only path terms use their registered neutral value
```

The exact anchor transforms are:

```text
is_causal_trade_onset
  = 1

joint_threshold_overshoot
  = min(
      d * trade_ratio_100ms(t) - 0.50,
      d * trade_ratio_500ms(t) - 0.25
    )

signed_fast_minus_medium_acceleration
  = d * trade_ratio_100ms(t) - d * trade_ratio_500ms(t)

leader_background_run_length
  = log1p(
      min(2000ms, consecutive observable TRADE BACKGROUND ending at t-20ms)
      - 120ms
    )

time_since_last_opposite_trade_update
  = log1p(
      min(60000ms, elapsed time from the last TRADE NEW_-d before t)
    )
```

If no opposite trade update exists inside the same segment during the prior
60s, the elapsed value is exactly `60000ms`.

For a control, all five H1 fields are zero. H0 already contains its current
trade-ratio levels, so the H1 group represents the registered causal arrival
path rather than another copy of the snapshot.

The primary hypothesis test concerns the entire H1 group. Individual
coefficients are descriptive and cannot rescue a failed group test.

## 16. Model Class and Parameter Control

Use a low-parameter discrete-time multinomial hazard:

```text
at risk
-> DEPTH_FOLLOWER_SAME
-> EXPLICIT_CONTRADICTION
-> remain at risk
```

The model uses ridge regularization.

The only permitted penalty grid is:

```text
lambda in {0.01, 0.1, 1.0, 10.0}
```

Penalty selection occurs inside each outer-training fold using inner
leave-one-date-out log loss. The selected penalty is applied unchanged to the
held-out date.

The registered optimizer contract is:

```text
deterministic L-BFGS
maximum iterations = 1000
relative objective tolerance = 1e-8
no random initialization
no class reweighting beyond registered matched-set weights
```

Non-convergence in any outer fold is
`Aminus1b_structural_outcomes_not_estimable`. It does not authorize a new
optimizer or model.

No spline, neural model, tree model, interaction search, feature selection or
post-result model change is authorized in V1.

## 17. Cross-Fit Contract

The outer split is:

```text
leave one research date out
```

For each outer fold:

1. derive covariate bins from training dates;
2. realize deterministic hazard-bin merging from training exposure only;
3. construct training matching bins and controls;
4. standardize covariates on training dates;
5. select ridge penalty using inner date folds;
6. fit H0 and H1 on the outer-training data;
7. score both models once on the held-out date;
8. retain predictions, causes and censoring without refitting.

Dependence-aware summaries cluster by:

```text
capture_id : anchor_epoch_id
```

No random row split is permitted.

## 18. Null and Incremental Test

### 18.1 Primary score

The primary model statistic is held-out log-score improvement:

```text
DeltaLL = OOF log likelihood(H1) - OOF log likelihood(H0)
```

Positive values favor arrival-path incrementality.

### 18.2 Matched-set permutation null

Within each matched set and research date:

```text
permute the complete causal arrival-path bundle among set members
preserve timestamps, snapshot covariates, causes and censoring
```

Permutation occurs at the dependence-cluster level. Use:

```text
10,000 deterministic permutations
```

The RNG seed is:

```text
SHA256(protocol ID, frozen input-manifest SHA256)
```

The primary p-value is one-sided for `DeltaLL > 0`.

### 18.3 Structural interpretability summaries

Report:

```text
cumulative incidence of DEPTH_FOLLOWER_SAME
cumulative incidence of EXPLICIT_CONTRADICTION
their M-minus-control differences
time-bin-specific cause hazards
date-specific DeltaLL
leave-one-date-out influence
```

Bootstrap confidence intervals use dependence clusters and a fixed
deterministic seed derived from the same protocol/input identity.

Use:

```text
10,000 clustered bootstrap replicates
two-sided percentile 95% intervals
```

## 19. Precision-First Acceptance

False positives are treated as direct cost. Missed M-states are treated as
opportunity loss and do not create a minimum-recall objective.

### 19.1 A-1a estimability gate

All must hold:

```text
retained M dependence clusters >= 60
represented research dates >= 4
maximum single-date M-cluster share <= 0.50
overall matched-anchor share >= 0.80
each represented date matched-anchor share >= 0.60
at least three realized hazard bins remain
```

The `60`-cluster floor supports five incremental H1 degrees of freedom
under a minimum ten-clusters-per-incremental-parameter rule.

Failure is an estimability result, not evidence that the transition
hypothesis is false.

### 19.2 A-1b primary incremental gate

Before fitting, structural outcomes are estimable only if:

```text
DEPTH_FOLLOWER_SAME occurs in >= 20 dependence clusters overall
EXPLICIT_CONTRADICTION occurs in >= 20 dependence clusters overall
each outer-training fold contains >= 10 clusters of each cause
```

Otherwise classify:

```text
Aminus1b_structural_outcomes_not_estimable
```

No binary fallback, cause merging or longer horizon is permitted.

If outcome support is estimable, all primary incremental conditions must hold:

```text
DeltaLL > 0
matched-set permutation one-sided p <= 0.01
at least 4 represented dates have date-level DeltaLL > 0
maximum absolute leave-one-date-out contribution share <= 0.50
```

### 19.3 A-1b precision gate

At the registered 60s structural endpoint:

```text
lower 95% clustered-bootstrap bound of:
  P(DEPTH_FOLLOWER_SAME first | M)
  - P(DEPTH_FOLLOWER_SAME first | control)
  > 0

upper 95% clustered-bootstrap bound of:
  P(EXPLICIT_CONTRADICTION first | M)
  - P(EXPLICIT_CONTRADICTION first | control)
  <= 0
```

The contradiction condition is intentionally strict. A path that increases
both same-direction following and explicit contradiction is not accepted as
precision-first structural evidence.

### 19.4 Sensitivities

Permitted non-rescue sensitivities are:

```text
DEPLETION_LED with trade/OFI follower detail
OFI_LED with trade/depletion follower detail
```

They must use the same thresholds, dates, cross-fit, matching, hazard bins and
statistics. They cannot rescue a failed `TRADE_LED` primary.

## 20. Q0 Pipeline Qualification

Q0 exists to prevent one-shot scientific claims from serving as software
integration tests.

### 20.1 Q0 permitted inputs

Use only:

```text
fully synthetic cache fixtures
registered static qualification fixtures with no scientific interpretation
temporary Git repositories and temporary local controller remotes
```

Q0 must not open the formal 29-cache source root.

### 20.2 Required fixture matrix

At minimum:

```text
QF01:
  one complete epoch, no anchors

QF02:
  valid TRADE_LED M -> DEPTH_FOLLOWER_SAME

QF03:
  valid M -> EXPLICIT_CONTRADICTION

QF04:
  valid M -> right censoring

QF05:
  simultaneous same/opposite evidence, contradiction precedence

QF06:
  segment boundary inside risk window

QF07:
  fixed-epoch slice start before an anchor

QF08:
  fixed-epoch slice start after a prior unrelated state

QF09:
  depletion and OFI same-direction tie

QF10:
  poison mutation in an unconsumed field

QF11:
  missing, extra and reordered package artifact

QF12:
  interrupted worker before and after slice materialization

QF13:
  mutate midpoint and every non-availability value strictly after anchor t;
  require all H0/H1 inputs and anchor/control admission at t unchanged
```

### 20.3 Exact production-path requirement

Q0 must call the same production functions intended for scientific:

```text
feature building
full-cache analysis
slice materialization
sliced-cache analysis
slice invariance
A/B/P comparison
package sealing
terminal verification
```

A test-only reimplementation is not qualification.

### 20.4 Q0 acceptance

All must hold:

```text
every positive fixture reaches terminal PASS
every negative fixture fails at its registered first boundary
full-cache/slice state contracts are exact
no missing internal feature payload such as _features
A/B deterministic equality is exact
A/P poison invariance is exact
terminal verifier accepts only the exact package
runner, verifier and tests are clean and reproducible from a fresh worktree
causal-prefix invariance holds for every anchor-time model input
```

Q0 may authorize drafting an A-1a formal task only. It cannot authorize
opening price outcomes.

## 21. A-1a Support Audit

A-1a is outcome-blind with respect to post-anchor direction and price.

It produces:

```text
causal M-anchor manifest
suppressed-anchor ledger
control-candidate pool manifest
matched-set manifest
anchor-time covariate manifest
future-observation-capacity ledger
segment/gap censor-capacity ledger
training-fold hazard-bin realization ledger
date/epoch support summary
overlap and fallback summary
```

The primary classification is one of:

```text
Aminus1a_trade_led_risk_sets_estimable
Aminus1a_mstate_support_not_estimable
Aminus1a_control_overlap_not_estimable
Aminus1a_time_support_not_estimable
Aminus1a_integrity_failed
```

A-1a may not change:

```text
M-state thresholds
prestate duration
follower anchor-time observability
epoch/core geometry
thinning
matching variables
fallback ladder
support gates
```

An accepted A-1a terminal package freezes exact anchor and matched-control
identities for A-1b.

## 22. A-1b Structural Transition Audit

A-1b consumes only:

```text
accepted A-1a causal-anchor manifest
accepted A-1a matched-control manifest
the same immutable structural source inventory
```

It adds post-anchor structural outcomes under Sections 11-19.

The primary classification is one of:

```text
Aminus1b_trade_led_depth_transition_incremental
Aminus1b_no_incremental_arrival_path_information
Aminus1b_precision_gate_failed
Aminus1b_structural_outcomes_not_estimable
Aminus1b_integrity_failed
```

Only:

```text
Aminus1b_trade_led_depth_transition_incremental
```

may unlock A0 drafting and execution binding.

## 23. Conditional A0 Price Protocol

A0 is fully specified here but remains locked.

### 23.1 A0 population

A0 must use the causal M anchors and controls frozen by accepted A-1a.

It must not restrict the population to anchors that later achieved
`DEPTH_FOLLOWER_SAME`. Conditioning on structural confirmation would use
future information and create look-ahead selection.

The accepted A-1b result authorizes outcome access; it does not redefine the
anchor population.

### 23.2 Price competing risks

Starting from the midprice observable at anchor/control time `t`:

```text
PRICE_ALIGNED:
  first midprice move of at least one tick in direction d

PRICE_OPPOSED:
  first midprice move of at least one tick in direction -d

CENSOR:
  same structural censoring contract, with no first passage
```

If aligned and opposed thresholds become observable at the same checkpoint:

```text
PRICE_OPPOSED wins
```

This is the precision-first tie rule.

### 23.3 A0 time scale

A0 uses the same training-fold hazard-bin geometry produced by the frozen
support-merging algorithm. It cannot select a new horizon from price results.

### 23.4 A0 H0/H1

```text
H0:
  anchor-time snapshot model from Section 14

H1:
  H0 plus the arrival-path group from Section 15
```

Structural follower outcomes after `t` do not enter either price model in V1.
This preserves the direct test of information available at the alignment
point.

### 23.5 A0 acceptance

Price outcomes are estimable only if:

```text
PRICE_ALIGNED occurs in >= 20 dependence clusters overall
PRICE_OPPOSED occurs in >= 20 dependence clusters overall
each outer-training fold contains >= 10 clusters of each cause
```

Failure is an A0 estimability result and does not authorize a larger price
threshold, longer horizon or binary fallback.

The primary price gate mirrors A-1b:

```text
OOF DeltaLL > 0
matched-set permutation one-sided p <= 0.01
at least 4 dates with positive date-level DeltaLL
maximum leave-one-date-out contribution share <= 0.50
```

The price precision gate is:

```text
lower 95% bound of:
  P(PRICE_ALIGNED first | M)
  - P(PRICE_ALIGNED first | control)
  > 0

upper 95% bound of:
  P(PRICE_OPPOSED first | M)
  - P(PRICE_OPPOSED first | control)
  <= 0
```

Passing A0 establishes conditional directional information only. A separate
execution and cost study is still required before any strategy claim.

## 24. Artifact and Provenance Requirements

Every formal stage must freeze:

```text
protocol/document SHA256 and Git blob
task SHA256 and Git blob
runner/verifier/tests SHA256 and Git blob
raw source inventory SHA256
exact CLI and roots
exact stage input manifests
output schemas
RNG seeds
model and penalty grids
Git implementation/consumption/terminal identities
```

Every scientific build uses:

```text
Build A:
  canonical deterministic build

Build B:
  independent canonical deterministic rebuild

Build P:
  poison build changing every unconsumed field
```

Required:

```text
A == B exactly
A == P over every consumed scientific artifact
slice/reset invariance exact
no future field crosses an earlier stage boundary
```

## 25. Required Review Sequence

Each executable stage follows:

```text
idea/protocol review
-> implementation review
-> focused and inherited tests
-> frozen task and claim
-> one formal execution
-> independent terminal verification
-> independent QA
-> controller decision
```

No chat conclusion may replace a tracked task, report or QA artifact.

## 26. Prohibited Adaptation

After a scientific claim is consumed, prohibit:

```text
threshold relaxation
M-state redefinition
follower endpoint redefinition
new matching variables or calipers
new hazard bins
new model family
new covariates
new null
alternative date split
sensitivity rescue
repair or rerun inside the same task
```

Before a claim is consumed, software defects may be addressed only through a
separate Q0 qualification revision. Scientific definitions cannot be changed
because a qualification fixture or support audit looks inconvenient.

## 27. Decision Tree

```text
Q0 terminal PASS?
  no  -> software qualification failure; stop
  yes -> draft and review A-1a formal task

A-1a estimable?
  no  -> structural support/overlap/time support not estimable; stop
  yes -> freeze exact anchors, controls and fold geometry

A-1b incremental and precision gates pass?
  no  -> leader arrival path not accepted as structural signal; stop
  yes -> bind conditional A0 to frozen causal anchor/control population

A0 incremental and precision gates pass?
  no  -> structural propagation exists without accepted price increment; stop
  yes -> candidate for a separate execution/cost research protocol
```

## 28. Immediate Next Task

The only immediate next task authorized by this master protocol is:

```text
Q0:
  TRADE_LED_DEPTH_FOLLOWER_PIPELINE_QUALIFICATION_V1
```

Its scope is limited to:

```text
production-path fixture qualification
full/slice/A/B/P package closure
terminal verification
fresh-worktree reproducibility
```

It must not:

```text
open the formal historical source-cache root
count real M anchors
construct real matched controls
inspect structural transition outcomes
inspect future prices
produce a scientific classification
```

Only an independently accepted Q0 result may authorize preparation of the
A-1a formal task.

## 29. Binance-Only Replication Boundary

The master hypothesis concerns a recurring market microstructure transition,
not one venue-specific implementation. A market-general claim therefore
requires a separately registered Binance-only replication after the primary
structural result.

The replication must preserve:

```text
M-state logic
directional thresholds
anchor-time follower-background requirement
competing-risk precedence
time-bin algorithm
matching variables and fallback ladder
H0/H1 transforms
cross-fit and acceptance gates
```

It may change only:

```text
raw source inventory
exchange-specific cache schema
tick-size normalization
the versioned raw-to-feature authority
research dates
```

The Binance dataset cannot be used to repair or tune a failed primary study.
It is either:

```text
prospective replication of an accepted primary result
```

or:

```text
a separately identified primary dataset declared before any structural
outcome is opened
```

Passing only one dataset supports a dataset-specific structural claim.
Passing both under frozen semantics is required before using terms such as
`market-general leader-trigger pattern`.
