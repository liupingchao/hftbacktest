# SKHYNIX Trade-Led Depth-Follower Q0 Pipeline Qualification Execution Plan

Date: 2026-08-31

Task ID: `0831T001`

Qualification ID:
`TRADE_LED_DEPTH_FOLLOWER_PIPELINE_QUALIFICATION_V1`

Status:
`REVISION_21_CANDIDATE_PENDING_INDEPENDENT_REVIEW`

Parent protocol:
`TRADE_LED_DEPTH_FOLLOWER_TRANSITION_HAZARD_MASTER_V1`

Frozen parent identity:

```text
commit:
  2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19

SHA256:
  4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40

Git blob:
  69c5cdf51b7fdf07d55170ed58bc791ff37bd0af
```

Any parent-protocol byte drift fails before implementation or formal
qualification.

## 1. Purpose

This task qualifies the shared causal feature, anchor, structural-outcome,
slice and package core intended for later structural studies without opening
the formal historical cache root.

It addresses the observed predecessor failure mode:

```text
formal one-shot execution was used as an integration test
-> Build A reached slice reconstruction
-> a hidden `_features` dependency was absent
-> the scientific attempt terminated before publication
```

Q0 does not qualify A-1a control matching, overlap, folds or support gates. It
does not qualify A-1b model fitting, permutation, bootstrap or statistical
classification. Each later stage requires a separate qualification of those
stage-specific entry points.

Q0 does not test market support and cannot produce a scientific
classification.

### 1.1 Revision 4 closure matrix

| Round 3 finding | Revision 4 closure |
|---|---|
| production partition | exact accepted `17` row + `10` metadata fields; scalar `tick_size` is not row-expanded |
| rolling boundary | accepted complete-window semantics; no prefix substitution |
| QF04 censor | independently derived `150.2s / 7510 / 60000ms` |
| reset non-vacuity | actual `NEW_NEG` and `[-1,0,0]` memory at `2999`, cleared at `3000` |
| child receipts | all loader/hasher/detector/FD/EOF fields persisted in `feature_calls.csv` |
| QF12 first error | one published slice with other required slices absent expects `SLICE_PUBLICATION` |
| QF13 background run | production-derived `log1p(200ms-120ms)=log(81)` with boundary probe |
| CSV bytes | exact dialect, final LF and QUOTE_ALL hostile probe |
| readiness identity | separate 37-file structural-only mode; no formal identity |
| one-shot | durable-state resolver, blocked controller divergence and stage-truthful exact reports |

### 1.2 Revision 10 closure matrix

| Round 9 finding | Revision 10 closure |
|---|---|
| legal `Popen` errors could not render a terminal receipt | producer/verifier exit fields are `NONE` for `POPEN_ERROR` and integers only for `STARTED` |
| invalid artifact order had no deterministic outcome | ordered A01-A11 artifact-state rules route the first integrity defect to `ARTIFACT_STATE_CORRUPTION`, classification/profile `NONE` |
| controller blocker was not total across pre-root, observation failure and restart | separate pre/post-attempt-root outcome tables plus ten blocker-specific restart rows freeze every allowed local transition |
| only one of 49 legal FAIL reports had an exact derivation | the complete profile/error matrix, row count, size range, unique-hash count and aggregate canonical-row SHA256 are frozen |

### 1.3 Revision 11 closure matrix

| Round 10 finding | Revision 11 closure |
|---|---|
| `ABSENT` remained legal after durable consumption proof | expected ref sets are split into six receipt-sensitive proof stages; after consumption receipt only consumption SHA is legal, after terminal receipt only terminal SHA is legal |
| wrong tag target or extra HEAD had no restart outcome | ordered G01-G07 rules bind claim, HEAD, commit parent/message/tree, index/worktree and both annotated tags; G02-G07 route to `ARTIFACT_STATE_CORRUPTION` |
| restart authority was not machine-total | a 10,368-row proof-stage/ref/claim/HEAD/tag/commit/worktree cross-product freezes 9 legal rows, 10,359 invalid rows and one canonical aggregate SHA256 |

### 1.4 Revision 12 closure matrix

| Round 11 finding | Revision 12 closure |
|---|---|
| legal commit-before-tag states matched G06/G07 | consumption-commit-before-tag and terminal-commit-before-tag are distinct action phases with `ABSENT` as the exact expected tag state |
| exact staged index states matched G05 | claim rename, consumption staged index, terminal receipt-only worktree, terminal unstaged delta, PASS common-stage split and complete terminal staged index each have a distinct exact tracked-transition state |
| proof-stage table omitted restart substates | 15 action phases generate a 116,640-row pre-blocker table; the 11 controller restart phases also generate a separate 21,384-row post-blocker table that skips G01 |

### 1.5 Revision 13 closure matrix

| Round 12 finding | Revision 13 closure |
|---|---|
| staged armed-to-claimed proof depended on rename detection | all index/worktree observations use `git diff --no-renames --raw -z --full-index --abbrev=40`; rename and name-only output are forbidden authority |
| destination-only output did not prove armed deletion | the consumption staged preimage is exactly `D armed` plus `A claimed`, both mode `100644`, with armed old blob equal to claimed new blob and independently hashed claim bytes |
| unstaged additions lacked exact observation | untracked paths use NUL-delimited `git ls-files --others --exclude-standard`, no-follow `lstat`, exact mode and `git hash-object --stdin`; cached, worktree and untracked rows jointly define each transition state |

### 1.6 Revision 14 closure matrix

| Round 13 finding | Revision 14 closure |
|---|---|
| staged mode and normalized-byte mutations could remain invisible | derive the complete expected index from `HEAD tree + cached preimage`, derive the complete expected physical tree from `index + worktree/untracked preimage`, then no-follow `lstat` and exact-byte hash every stage-0 path; Git diff cleanliness is no longer physical authority |
| raw parser contradicted actual NUL framing | freeze repeated `metadata NUL path NUL` pairs, exact metadata regex, final-NUL/even-field rules, stage-0 index grammar, ASCII path normalization and duplicate rejection |
| only two dirty states had detailed preimages | map 22 action-phase/terminal-branch variants to seven machine-expandable transition preimages; PASS expands the exact 57 baseline paths and FAIL expands only the three common terminal paths |
| mutation guarantee was not reproducible | freeze four one-field mutation recipes across all 22 variants and eight repository configurations: 704 rows, every row expects G05, canonical aggregate `8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c` |

### 1.7 Revision 15 closure matrix

| Round 14 finding | Revision 15 closure |
|---|---|
| `repository_config` nested schema was implicit | list all eight exact nested objects with snake_case keys `core_autocrlf`, `core_filemode`, `diff_renames`; values are JSON booleans and dotted-key flattening is forbidden |
| `variant_ordinal` base/type was implicit | persist zero-based integer ordinals `0..21` directly in all 22 phase/branch rows and require equality with array index |
| PATH probe formatting was ambiguous | freeze exactly two zero-padded ASCII decimal digits, yielding `00` through `21` |
| aggregate preimage was not unique | freeze row nesting, exact fields and array orders; the explicit 704 rows independently rederive the unchanged aggregate `8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c` |

### 1.8 Revision 16 closure matrix

| Round 3 implementation-readiness finding | Revision 16 closure |
|---|---|
| controller ref disappearance after durable consumption proof is defined as divergence, but `controller_divergence_value_domains.observed_sha` allowed only a 40-hex SHA | amend that single value domain to `ABSENT or one 40 lowercase hex SHA outside expected_sha_set_json`; all receipt fields, expected-set rules, blocker code, restart semantics and other surface bytes remain unchanged |

### 1.9 Revision 17 closure matrix

| Round 16 finding | Revision 17 closure |
|---|---|
| value-domain amendment still conflicted with a trigger, POST_ATTEMPT_ROOT table and plan prose that treated `ABSENT` as globally legal | define the observed token domain as `ABSENT or 40-hex`, then make legality depend only on membership in the exact receipt-sensitive expected set for the current durable proof stage; any token outside that set, including `ABSENT`, is divergence |

### 1.10 Revision 18 closure matrix

| Round 6 implementation-readiness finding | Revision 18 closure |
|---|---|
| an invalid tracked receipt could advance controller proof-stage before G02-G07 | PRE_BLOCKER first evaluates the exact local Git phase; a receipt may select a proof-stage only after its canonical schema, transition tuple, source-union bytes and tracked-copy bytes validate |
| `recovery_start.json` could be edited and re-self-hashed | the final target becomes recovery authority only after exact-byte hard-link publication, mode `0444`, macOS `UF_IMMUTABLE`, parent fsync and no-follow seal verification; no recovery mutation is legal before the seal, so an interrupted unsealed publication must be independently regenerated from the unchanged snapshot |
| durable control publication accepted symlink targets | every target and sibling temporary is inspected with no-follow `lstat`; committed targets must be regular files, and callers may not resolve the final component before publication or validation |

Revision 18 was rejected because inode sealing conflicted with temporary
cleanup, lacked a complete crash matrix and was not an external trust anchor.
It remains historical review evidence and is superseded by Revision 19.

### 1.11 Revision 19 closure matrix

| Round 18 finding | Revision 19 closure |
|---|---|
| inode seal crash states conflicted with generic publication | remove inode sealing entirely; use the existing controller bare repository as an external recovery witness before any local recovery-start path is created |
| temporary cleanup conflicted with a shared immutable inode | witness ref points to an independent Git blob; local target and temporary retain the unchanged generic hard-link/unlink protocol |
| observed state did not select one action phase | freeze a total ordered derivation over canonical receipt unions, tracked-copy equality, local Git state and the observed controller token; zero or multiple matches are G05 |
| owner could clear and restore `UF_IMMUTABLE` | create `refs/tags/skhynix-trade-led-depth-follower-q0-recovery-start-v1` in the controller bare repo with CAS-from-ABSENT, pointing directly to the exact recovery-start blob; local bytes are accepted only when equal to that externally bound blob |
| durable untracked consumption receipt had no action phase | add `NORMAL_CONSUMPTION_UNTRACKED_RECEIPT_COMMITTED`; current machine has 16 action phases, 23 Git preimage variants, 736 mutation rows and a 124,416-row pre-blocker table |

Revision 19 was rejected with four boundary-completeness findings. Its
witness architecture and phase authority were accepted and are retained.

### 1.12 Revision 20 closure matrix

| Round 19 finding | Revision 20 closure |
|---|---|
| witness abnormal states had no workflow outcome | add `A12_RECOVERY_WITNESS_MISMATCH`; every witness/local matrix violation publishes or verifies `ARTIFACT_STATE_CORRUPTION`, classification `NONE`, and uses existing blocker restart/QA semantics |
| exact temporary crash cut was missing | freeze separate absent/exact/mismatched/nonregular temporary rows; an exact regular temporary is reopened no-follow, fstat/read/fsync verified, then hard-linked and unlinked through the generic protocol |
| ordinal schema still ended at 21 | set maximum `22` and update all current prose to 23 variants |
| witness was absent from QA schema | add exact `recovery_witness_ref` and `recovery_witness_blob_oid` evidence fields; both are `NONE` only on the normal path |

Revision 20 was rejected with three witness-boundary findings. Its corrected
variant authority, aggregate tables and witness ownership split are retained.

### 1.13 Revision 21 closure matrix

| Round 20 finding | Revision 21 closure |
|---|---|
| invalid witness branch conflicted with `forbid recovery mutation` | define A12 blocker publication and deterministic local claim consumption as the only allowed blocker writes; explicitly forbid controller mutation, recovery-start/observation publication, terminalization and baseline mutation |
| A12 receipt and QA evidence were not total | embed canonical `recovery_witness_evidence_json` in `artifact_state_corruption.json`; freeze command exit/output hashes plus ref/OID/object and no-follow target/temp observations; select exactly one QA mode from `NORMAL`, `RECOVERY`, `WITNESS_BLOCKED` |
| exact temporary resume omitted final binding and race handling | retain the open temporary FD identity, require final no-follow FD bytes and matching `st_dev/st_ino` after successful link, and route `EEXIST` through the exact existing-target row before any temporary removal |

## 2. Authorization Boundary

Permitted inputs:

```text
deterministic synthetic NPZ fixtures generated by this task
temporary directories
temporary Git worktrees used only for reproducibility checks
accepted tracked fixed-epoch source files and their Git objects
```

Prohibited inputs:

```text
the formal 29-cache source root
any local_live_analysis historical cache
future price labels
returns
fills
fees
PnL
private or live exchange data
```

The runner must reject any input-root option. Fixture construction is
internal and closed.

## 3. Frozen Production Architecture

The implementation uses new files. No accepted predecessor source file may
be edited.

### 3.1 Structural core

```text
examples/hyperliquid/
  skhynix_trade_led_depth_follower_transition_hazard.py
```

This module owns the functions intended for later scientific execution:

```text
build_features
build_anchor_frame
finalize_anchor_availability
label_structural_outcomes
analyze_cache_in_stage
materialize_slice
compare_slice
build_raw_package
seal_package
verify_package
```

The stage entry points are distinct:

```text
A_MINUS1A:
  build_features
  -> build_anchor_frame
  -> finalize_anchor_availability

A_MINUS1B:
  accepted frozen anchor manifest
  + build_features
  -> label_structural_outcomes
```

`A_MINUS1A` cannot call `label_structural_outcomes`. The Q0 runner and later
tasks must call these functions directly. Test-only reimplementation is
prohibited.

### 3.2 Q0 runner

```text
examples/hyperliquid/
  skhynix_trade_led_depth_follower_q0_pipeline_qualification.py
```

The runner owns deterministic fixture generation, A/B/P orchestration,
negative-boundary probes and publication of the qualification package.

### 3.3 Independent verifier

```text
examples/hyperliquid/
  skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py
```

The verifier must reconstruct package hashes, schemas, fixture expectations
and A/B/P identities without trusting the runner's PASS field. It reads the
frozen fixture truth authority directly and independently checks physical
fixture arrays, not only runner-produced summaries.

### 3.4 Tests

```text
examples/hyperliquid/
  test_skhynix_trade_led_depth_follower_q0_pipeline_qualification.py
```

Tests may construct fixtures but must invoke the production core for every
scientific-state transition and package operation.

## 4. Bound Authorities

### 4.1 Fixture truth authority

The independent machine-readable oracle is:

```text
.workflow/contracts/0831T001-fixture-truth-v1.json

SHA256:
  c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529

Git blob:
  ea66f4ff2e7cddf9302215d62c3268299682add7
```

It freezes:

```text
clock and default arrays
exact patch coordinates and values
fixture order
expected anchor identity
expected structural cause, detail and timestamp
expected epoch dispositions
exact slice starts and common epoch IDs
non-vacuous comparison floors
reset expectations
negative-probe order and first error codes
```

The runner may not write or derive this authority. The verifier checks its
tracked Git blob and SHA before opening any fixture output. Hostile tests must
mutate the truth bytes and observed output independently; either mutation
must fail closed.

### 4.2 Surface contract authority

The executable schema, formula, package and provenance authority is:

```text
.workflow/contracts/0831T001-q0-surface-contract-v1.json

SHA256:
  2e470e647e85bc60249a6661cadf451c95735fbd655a693ddf5a3aeef84eb52e

Git blob:
  eeb3e7345e9ae42ac169bf96b5ffe9d41dc8d559
```

It freezes:

```text
production cache schema v4
rolling ratios and base eligibility
canonical hash preimages
57-call A/B/P instrumentation
all JSON field sets
allowed package directories
hostile mutation recipes
readiness comparison projection
claim/lock/receipt/controller state machine
```

Where this prose and the surface contract differ, execution fails closed
rather than selecting one interpretation.

### 4.3 Accepted fixed-epoch authorities

The new feature authority consumes exactly the 17 row-aligned fields
registered by the accepted production cache:

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
trade_signed
trade_total
ts_ns
valid_book
```

The production cache also contains the ten schema-v4 metadata fields frozen
in the surface contract, including scalar `tick_size float32[1]`. Their names,
dtypes, shapes and registered constants are validated by the independent
input verifier. Their values are not exposed to the production feature
loader. QF10 changes only the existing
`bin_boundary_violations` metadata value.

The structural core directly calls the accepted authorities for:

```text
build_features
base_masks
source_preflight
channel_actions
channel_memories
epoch_support_ledger
```

The inherited state constants remain:

```text
checkpoint = 20ms
channel memory TTL = 100ms inclusive
leader prestate = 120ms / six checkpoints
epoch width = 60s
anchor core = [15s,45s)
fast threshold = 0.50
medium threshold = 0.25
```

Q0 freezes source path, Git blob and callable AST identities for every reused
authority.

The accepted feature authority identity is:

```text
path:
  examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py

commit:
  45544ecc3901623ca7c2e34a059afca6c551d625

Git blob:
  494c203e7195f292e057f7708c99f52096259a02

SHA256:
  f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c

build_features AST:
  e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933

base_masks AST:
  bc2155a38bd1707fcdb77bdebea611da3889934a47d415bdfd0d5a95842d7114
```

The accepted fixed-epoch authority identity is:

```text
path:
  examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py

commit:
  f06eb5cb012cb62b2a778ad90d433c4083f9ba14

source Git blob:
  5672a8ca9f6d4ced2b2deaaf5689e2e7bb7935da

source SHA256:
  dfa8af1f4b8410370ec7ccd0bea30b63840ebe484d74446c8cbe2564918ac070

source_preflight AST:
  9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6

channel_actions AST:
  0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab

channel_memories AST:
  5507a492a9984d0aec1f36ef16a9977ff2321af4c86e245fa3a0ddfe8c7e4df1

epoch_support_ledger AST:
  d76a80ef31b3f229eb23099f1f3b06cf6ab2e2a3f101a07c37c2286436f5b453
```

`authority_binding.json` encodes these as the exact ordered
`accepted_authorities` array from the surface contract:

```text
FEATURE_AUTHORITY
FIXED_EPOCH_AUTHORITY
```

Each row contains `authority_id`, `path`, `commit`, `git_blob`, `sha256` and
an exact callable-name-to-AST-SHA object. Singular or flattened authority
fields are forbidden.

## 5. Feature Contract

### 5.1 Exact raw schema

The fixture cache is exact-isomorphic to production cache schema v4:

```text
17 row-aligned fields:
  ts_ns int64[n]
  event_seq int32[n]
  segment_id int32[n]
  valid_book bool[n]
  ready bool[n]
  activity int32[n]
  trade_signed/trade_total float32[n]
  ask_depletion/bid_depletion float32[n]
  ofi/ofi_abs float32[n]
  bid_depth/ask_depth float32[n]
  obi/spread_ticks/midpoint float32[n]

10 metadata fields:
  cache_schema_version int32[1] = 4
  bin_boundary_violations int32[1]
  initial_bridge_failure_count int32[1]
  non_admitted_message_contributions int32[1]
  quality_boundary_count int32[1]
  reset_count int32[1]
  sequence_gap_count int32[1]
  segment_end_ids int32[segment_count]
  segment_end_ts int64[segment_count]
  tick_size float32[1] = 1.0
```

No additional field is permitted. Exact dtype strings, shapes and value
domains are in `source_schema_v4` of the surface contract. Any mismatch fails
before feature construction.

`ts_ns` must follow the fixture authority's 20ms grid. `event_seq` is
strictly increasing. Segment IDs may change only at a row boundary and may
not return to a previously closed segment.

The independent input verifier checks all 27 physical arrays. The production
loader sees the exact field-name schema but reads values from exactly 17 row
fields:

```text
12 through accepted build_features
5 through the new staged H0 extension
0 metadata value reads
```

The new staged extension appends OBI, spread, depth and midpoint row arrays to
the accepted flow feature output. `tick_size` remains scalar metadata and is
not converted into a synthetic time series.

### 5.2 Canonical fixture serialization

Fixture caches use canonical NPZ:

```text
ZIP_STORED
entry order = ASCII field-name order
entry name = <field>.npy
ZIP timestamp = 1980-01-01 00:00:00
create_system = 0
external_attr = 0
extra = empty
comment = empty
NumPy NPY version = 1.0
allow_pickle = false
```

Canonical A and B cache bytes must match. QF10 P cache bytes must differ only
through the existing `bin_boundary_violations` NPY payload.

### 5.3 Exact FeatureBundle schema

`build_features(cache_path)` returns a frozen typed `FeatureBundle`. Its
arrays are explicit attributes; no downstream function may rely on an
undocumented dictionary key such as `_features`.

The bundle contains:

```text
raw: the 17 consumed row arrays, read-only, exact raw dtype
event_masks: bool[n,3]
ratios_100: float64[n,3]
ratios_500: float64[n,3]
base_eligible: bool[n]
actions: int8[n,3]
memories: int8[n,3]
memory_ages_ms: int32[n,3]
trailing_realized_volatility: float64[n]
source_access_ledger: typed immutable rows
```

Ratio, rolling, segment and base-eligibility formulas are exactly those in
`feature_formulas` of the surface contract. In particular:

```text
100ms = five checkpoints
500ms = 25 checkpoints
rolling sums include current but remain NaN until the complete same-segment
window is available
activity_500 threshold = 44.0
detector cooldown = 30s from segment start
```

Ratio and volatility unavailability is IEEE `NaN`. Unknown memory is `9`;
unknown age is `-1`.

Every production function receives `FeatureBundle` explicitly.
`compare_slice(full_bundle, full_analysis, slice_bundle, slice_analysis)`
receives both bundles explicitly.

### 5.4 Typed stage access boundary

`build_anchor_frame` receives a `CausalView` that exposes, for anchor
checkpoint `t`, directional and model-input arrays only at indices `<=t`.
Every read records:

```text
call_id
stage
fixture_id
anchor_ts_ns
field
minimum_index
maximum_index
authorization
read_count
```

A read above `t` fails immediately with `CAUSAL_ACCESS_BOUNDARY`.

`finalize_anchor_availability` receives a separate `AvailabilityView`. After
`t` it may read only:

```text
ts_ns
event_seq
segment_id
ready
valid_book
```

`label_structural_outcomes` receives an `OutcomeView` only in
`A_MINUS1B`. Q0 verifies this separate entry point, but A-1a cannot import or
call it through its registered stage dispatcher.

QF13 mutates every registered post-anchor non-availability field and requires
both:

```text
anchor/model-input bytes unchanged
causal access ledger maximum_index <= anchor_index
```

The negative probe directly attempts a future directional read through the
same `CausalView` and must first fail `CAUSAL_ACCESS_BOUNDARY`.

## 6. Causal State Contract

At checkpoint `t`, processing follows the parent protocol:

```text
validate current source state
-> update current channel evidence
-> update 100ms channel memories
-> evaluate six checkpoints of leader BACKGROUND before t
-> evaluate TRADE NEW_d at t
-> require depletion and OFI non-directional and observable at t
-> record snapshot and create provisional anchor
-> apply only availability finalization
-> observe structural outcomes strictly after t
```

The primary structural causes are:

```text
DEPTH_FOLLOWER_SAME:
  first depletion NEW_d or OFI NEW_d after t

EXPLICIT_CONTRADICTION:
  first trade, depletion or OFI NEW_-d after t

CENSOR:
  60s limit, segment boundary, gap, invalid book or source end
```

Contradiction wins a same-checkpoint tie. A same-direction trade self-repeat
is descriptive only and cannot satisfy the follower endpoint.

## 7. Deterministic Fixture Matrix

Every fixture is exactly three 60s epochs and 9,000 rows on the exact 20ms
grid, as frozen in the fixture truth authority. The primary non-reset fixture
anchor is at:

```text
index = 4510
ts_ns = 90,200,000,000
epoch_id = 1
event_seq = 4510
```

Production eligibility needs both trade and depth observability before a
neutral action can update memory. Every positive fixture therefore first
completes the 30s cooldown measured from the 60s slice/segment boundary, seeds
all three channels at index `4500`, refreshes trade neutral at `4504`,
refreshes depletion and OFI neutral at both `4505` and `4506`, refreshes trade
observability with neutral `trade_total=0.25` at `4509`, uses
leader-background checkpoints `4504..4509`, and places the anchor at `4510`.
Its anchor is therefore at `90.2s`. The extra refreshes keep the accepted
five-checkpoint fast denominators continuously finite; they prevent a hidden
`GLOBAL_INVALID` clear while preserving neutral memory and the frozen
direction thresholds. This makes the full/slice comparison non-vacuous while
preventing the fixture oracle from assuming memory the production base mask
would never admit.

QF14/QF15 separately establish a real pre-boundary negative trade state:
joint neutral seed at `2989`, trade refresh at `2993`, depth/OFI observability
at `2994`, follower refresh at `2995`, trade observability refresh at `2998`,
negative trade onset with `trade_signed=-2, trade_total=2` at `2999`, and
segment change at `3000`.
Production primitives must yield:

```text
actions[2999] = [NEW_NEG, NO_UPDATE, NO_UPDATE] = [3,5,5]
memories[2999] = [-1,0,0]
memory_ages_ms[2999] = [0,80,80]
memories[3000] = [9,9,9]
cross_segment_carry_count = 0
```

These rows are persisted in `support/reset_state.csv`; reset verification is
therefore non-vacuous.

| Fixture | Registered expectation |
|---|---|
| QF01 | no retained anchor; valid terminal package |
| QF02 | one `TRADE_LED` anchor; `DEPTH_FOLLOWER_SAME` first |
| QF03 | one anchor; `EXPLICIT_CONTRADICTION` first |
| QF04 | one anchor; right censoring |
| QF05 | same/opposite evidence at one checkpoint; contradiction wins |
| QF06 | segment boundary after anchor; boundary censor wins before later evidence |
| QF07 | slice begins before anchor; comparable anchor/cause identity exact |
| QF08 | slice begins after an unrelated prior state; comparable identity exact |
| QF09 | depletion and OFI same-direction tie; one follower cause with both details |
| QF10 | unconsumed-field poison changes raw bytes but no production output |
| QF11 | missing, extra and reordered artifact mutations are rejected |
| QF12 | worker interruption before and after slice publication is rejected without terminal PASS |
| QF13 | every post-anchor non-availability value mutation leaves anchor-time H0/H1 inputs unchanged |
| QF14 | segment-boundary reset clears a pre-boundary opposite trade memory |
| QF15 | full/slice reset state and cross-segment carry remain exact |

The JSON authority, not this prose table, is the fixture truth source. Its
patches are applied in listed order. Each patch is either:

```text
point patch:
  field, index, value

half-open range patch:
  field, start, stop, value
```

The verifier independently reconstructs the expected raw arrays from the
authority and compares every element and dtype with each physical A/B/P
cache. The runner may not derive expected labels from observed outputs.

For QF07, QF08 and QF15, the truth authority freezes exact full/slice:

```text
anchor rows
common epoch disposition rows
structural outcome rows
suppression/reset rows
canonical semantic preimage SHA256
```

For QF13 it freezes all 14 model-input values and the exact causal access
row ranges. Equality without matching these independent values is not a
qualification result.

The `leader_background_run_length` implementation scans backward from
`t-20ms` for at most 100 checkpoints. It reads each TRADE memory row until
the first non-`BACKGROUND` row, including that boundary probe; if all 100
rows are `BACKGROUND`, it stops at the 2000ms cap without reading an earlier
row. QF13 therefore reads `memories.trade[4499..4509]`: indices
`4500..4509` contribute 200ms and index `4499` proves the run boundary. Its
frozen value is:

```text
log1p(200ms - 120ms) = log(81) = 4.394449154672439
```

QF04 censor truth is not accepted as a hand-written tuple. A static oracle
independently computes:

```text
min(anchor_ts + 60s, first segment/gap/invalid/source-end boundary)
= 150.2s
event_seq = 7510
latency_ms = 60000
```

Any registered censor tuple inconsistent with this calculation fails before
runner implementation or formal execution.

The expected anchor ID format is exact:

```text
<fixture_id>:<epoch_id>:<direction>:<segment_id>:<ts_ns>:<event_seq>
```

Hostile tests independently mutate:

```text
one fixture-truth byte
one physical fixture value
one observed anchor identity
one observed structural cause
```

Each mutation must be rejected without changing the other surfaces.

## 8. Slice Contract

QF07, QF08, QF12 and QF15 use:

```text
nominal_start_ns = 60,000,000,000
actual_start_ns = 60,000,000,000
segment_id = fixture-truth segment at index 3000
```

The materializer:

```text
selects the first row in the registered segment at or after nominal start
copies every row-aligned field from that index
copies scalar metadata unchanged
writes through a temporary file
fsyncs the file
publishes with no-replace semantics
fsyncs the parent directory
```

For comparable complete epochs, full and sliced analysis must have exact:

```text
epoch disposition
retained anchor identity
anchor-time H0/H1 input identity
structural cause and event-time identity
censor identity
suppression counters
```

The comparison universe is frozen:

```text
common_epoch_ids = [1,2]
comparable_epoch_count >= 2
comparable_anchor_count >= 1
```

The full and slice epoch dispositions for these IDs must be `eligible`.
QF07/QF08/QF15 expected anchor identity is named in the truth authority.
An empty common set, zero comparable anchors or a different epoch universe is
`SLICE_IDENTITY_MISMATCH`, never a zero-mismatch pass.

QF06 places its segment boundary at 121s, in the epoch after the anchor
epoch. The anchor epoch therefore remains an accepted single-segment fixed
epoch while the later boundary can exercise structural censoring.

No analysis result may hide a required feature payload.

## 9. A/B/P Contract

Builds are:

```text
A:
  canonical deterministic fixtures

B:
  independently regenerated canonical deterministic fixtures

P:
  canonical fixtures with QF10 unconsumed-field poison
```

The following must hold:

```text
A raw package bytes == B raw package bytes
A sealed package bytes == B sealed package bytes
A raw package bytes == P raw package bytes
A sealed package bytes == P sealed package bytes
canonical QF10 cache SHA256 != poisoned QF10 cache SHA256
instrumented consumed-field access is exact
unconsumed-field value access count = 0
```

Build roots are exact:

```text
<attempt_root>/inputs/A
<attempt_root>/inputs/B
<attempt_root>/inputs/P
<attempt_root>/builds/A
<attempt_root>/builds/B
<attempt_root>/builds/P
```

A and B are generated independently in separate spawned processes from the
tracked truth authority. P is generated in a third process and applies only
the registered QF10 poison rule.

Every physical cache call produces typed evidence:

```text
build_label
call_index
fixture_id
unit_kind = FULL or SLICE
relative_input_path
input_file_sha256
canonical_array_sha256
consumer_input_sha256
feature_output_sha256
frame_sha256
field_access_sha256
detector_exit_sha256
detector_environment_entry_count
detector_cwd
inherited_fd_violation_count
hasher_exitcode
loader_exitcode
detector_exitcode
consumed_field_count
forbidden_value_read_count
sender_process_id
sender_closed
receiver_eof_observed
```

`detector_exit_sha256` is the canonical hash of the explicit exit record
containing exit code, environment-entry count, cwd, inherited-FD violation
count, EOF observation and sender-close state. The accepted values are
respectively `0`, `0`, `/`, `0`, `true`, and `true`; a hash without these
persisted typed fields is insufficient.

Process IDs are evidence-only and excluded from deterministic structural
package bytes.

The field-access ledger names every consumed field and has no row for
`bin_boundary_violations`. The verifier independently opens each retained
physical input, recomputes file/array hashes and checks that every production
call is bound to the corresponding A, B or P root. A missing call, reused A
path, copied A package without B/P calls, or mismatched input hash fails
`BUILD_INPUT_BINDING`.

Fixture-cache hashes and process evidence remain outside the structural
subpackages whose A/B/P bytes are compared.

Hash preimages, NaN normalization, frame encoding, loader/detector FD
boundaries, child receipts and exact call cardinality are frozen in the
surface contract:

```text
15 FULL + 4 SLICE calls per build
19 calls per build
57 calls total
```

The accepted `build_features` call reads 12 fields and the staged extension
reads five. A loader metadata-value read is a forbidden access.

## 10. Package Contract

Each `builds/<label>/structural` directory contains exactly 12 regular,
non-symlink files:

```text
contracts/authority_binding.json
contracts/feature_contract.json
contracts/state_contract.json
support/fixture_summary.csv
support/anchor_ledger.csv
support/structural_outcomes.csv
support/slice_invariance.csv
support/model_inputs.csv
support/reset_state.csv
raw_manifest.json
qualification_summary.json
sealed_manifest.json
```

Each `builds/<label>/evidence` directory contains exactly five regular,
non-symlink files:

```text
input_inventory.csv
feature_calls.csv
field_accesses.csv
slice_work.csv
evidence_manifest.json
```

The terminal root contains exactly six additional regular, non-symlink
files:

```text
contracts/formal_identity.json
contracts/fixture_truth_binding.json
abp_comparison.json
negative_boundary_results.csv
fixture_source_evidence.json
terminal_manifest.json
```

The complete terminal artifact count is therefore:

```text
3 * (12 structural + 5 evidence) + 6 terminal = 57 files
```

No extra path, directory artifact, symlink, FIFO, socket or device is
permitted below the package root.

The exact 17 allowed directory paths are the `package_directories` array in
the surface contract. Any other directory is `PACKAGE_PATH_SET_EXTRA`.

### 10.1 CSV schemas

Schemas and sort keys are exact:

```text
support/fixture_summary.csv
  fields:
    fixture_id, expected_anchor_count, observed_anchor_count,
    expected_cause, observed_cause, expected_event_ts_ns,
    observed_event_ts_ns, passed
  sort:
    fixture authority order

support/anchor_ledger.csv
  fields:
    fixture_id, anchor_id, epoch_id, segment_id, direction,
    anchor_ts_ns, anchor_event_seq, dependence_cluster_id,
    retained_rank, suppressed_count
  sort:
    fixture_id, anchor_ts_ns, anchor_event_seq, direction

support/structural_outcomes.csv
  fields:
    fixture_id, anchor_id, cause, detail, event_ts_ns,
    event_seq, latency_ms, censor_reason
  sort:
    fixture_id, anchor_id

support/slice_invariance.csv
  fields:
    fixture_id, nominal_start_ns, actual_start_ns,
    common_epoch_ids_json, comparable_epoch_count,
    comparable_anchor_count, expected_identity_sha256,
    observed_identity_sha256, mismatch_reason
  sort:
    fixture_id, nominal_start_ns

support/model_inputs.csv
  fields:
    fixture_id, anchor_id, input_name, canonical_value,
    causal_max_index, access_count
  sort:
    fixture_id, anchor_id, input_name

evidence/input_inventory.csv
  fields:
    build_label, fixture_id, relative_path, canonical_file_sha256,
    canonical_array_sha256, size_bytes, poison_status
  sort:
    build_label, fixture_id

evidence/feature_calls.csv
  fields:
    build_label, call_index, fixture_id, unit_kind, relative_input_path,
    input_file_sha256, canonical_array_sha256, consumer_input_sha256,
    feature_output_sha256, frame_sha256, field_access_sha256,
    detector_exit_sha256, detector_environment_entry_count, detector_cwd,
    inherited_fd_violation_count, hasher_exitcode, loader_exitcode,
    detector_exitcode, consumed_field_count, forbidden_value_read_count,
    sender_process_id, sender_closed, receiver_eof_observed
  sort:
    build_label, call_index

evidence/field_accesses.csv
  fields:
    build_label, call_index, stage, fixture_id, anchor_ts_ns,
    field, minimum_index, maximum_index, authorization, read_count
  sort:
    build_label, call_index, stage, anchor_ts_ns, field,
    minimum_index, maximum_index

evidence/slice_work.csv
  fields:
    build_label, fixture_id, slice_ordinal, relative_path,
    nominal_start_ns, actual_start_ns, canonical_file_sha256,
    publication_state
  sort:
    build_label, fixture_id, slice_ordinal

support/reset_state.csv
  fields:
    fixture_id, boundary_index,
    pre_action_trade, pre_action_depletion, pre_action_ofi,
    pre_memory_trade, pre_memory_depletion, pre_memory_ofi,
    pre_age_trade_ms, pre_age_depletion_ms, pre_age_ofi_ms,
    post_memory_trade, post_memory_depletion, post_memory_ofi,
    cross_segment_carry_count
  sort:
    fixture_id, boundary_index

negative_boundary_results.csv
  fields:
    probe_ordinal, probe_id, expected_first_error,
    observed_first_error, verifier_exit_code, passed
  sort:
    probe_ordinal
```

Empty optional values use the ASCII token `NONE`. Booleans use only
`true/false`. Integers have no decimal point. Finite floats use Python
`repr(float)` and non-finite values are prohibited in package CSV.

### 10.2 JSON and manifest contract

All JSON uses sorted keys, ASCII, compact separators and one trailing
newline. CSV canonicalization is fully frozen by `csv_contract` in the
surface authority:

```text
encoding = ASCII
delimiter = comma
quotechar = double quote
quoting = QUOTE_MINIMAL
doublequote = true
escapechar = NONE
line terminator = LF
header required
final record newline required
embedded CR/LF forbidden
quote a cell iff it contains comma or quote
```

Thus `[1,2]` is serialized as `"[1,2]"`; `QUOTE_ALL`, alternate escaping,
CRLF or a missing final LF is noncanonical even when a generic CSV parser
would return the same cells.

The exact field sets for:

```text
authority_binding.json
feature_contract.json
state_contract.json
qualification_summary.json
formal_identity.json
fixture_truth_binding.json
abp_comparison.json
fixture_source_evidence.json
```

are the `json_schemas` object in the surface contract. Missing or additional
keys fail `PACKAGE_SCHEMA`. Values are independently derived from frozen
authorities, physical inputs, observed rows and manifests; the verifier does
not trust producer copies.

Manifest preimages are exact:

```text
raw_manifest.json:
  hashes the nine contract/support files before itself

sealed_manifest.json:
  hashes the nine contract/support files, raw_manifest.json and
  qualification_summary.json; excludes itself

evidence_manifest.json:
  hashes the four evidence CSV files; excludes itself

terminal_manifest.json:
  hashes all 56 preceding package files; excludes itself
```

Manifest rows are sorted by relative ASCII path and contain exactly:

```text
path
size_bytes
sha256
```

No timestamp, PID, absolute path or temporary root may enter structural
bytes. PID and exact roots appear only in evidence/formal identity surfaces.

## 11. Negative Boundaries

The full formal negative set and first errors are:

```text
QF11_MISSING_ARTIFACT
  -> PACKAGE_PATH_SET_MISSING

QF11_EXTRA_ARTIFACT
  -> PACKAGE_PATH_SET_EXTRA

QF11_REORDERED_MANIFEST
  -> PACKAGE_LINEAGE_ORDER

QF12_INTERRUPT_BEFORE_SLICE_PUBLICATION
  -> SLICE_PUBLICATION_ABSENT

QF12_INTERRUPT_AFTER_SLICE_PUBLICATION
  -> SLICE_PUBLICATION

QF13_CAUSAL_PREFIX_MUTATION
  -> CAUSAL_ACCESS_BOUNDARY

ROOT_SYMLINK
  -> SOURCE_PATH_KIND_SYMLINK

ARTIFACT_FIFO
  -> PACKAGE_PATH_KIND_FIFO

NONCANONICAL_JSON
  -> PACKAGE_CANONICAL_JSON

NONCANONICAL_CSV
  -> PACKAGE_CANONICAL_CSV

NONCANONICAL_CSV_QUOTE_ALL
  -> PACKAGE_CANONICAL_CSV

SYNCHRONIZED_LINEAGE_MUTATION
  -> FIXTURE_TRUTH_OBSERVED_MISMATCH

RESET_IDENTITY_MISMATCH
  -> RESET_IDENTITY_MISMATCH

CROSS_SEGMENT_CARRY_NONZERO
  -> CROSS_SEGMENT_CARRY_NONZERO
```

The post-publication QF12 baseline has exactly one durably published QF12
slice while at least one other required QF07/QF08/QF12/QF15 publication is
absent. Complete slice publication is therefore the unique earliest boundary:
`SLICE_PUBLICATION`. No probe-specific skipping is allowed.

Verifier gate precedence is the exact `error_precedence` array in the fixture
truth authority. After the first failed gate, every later gate is
`NOT_EVALUATED`.

The verifier CLI emits one canonical JSON object to stdout:

```text
schema_version
qualification_id
package_root
result
first_error
gate_rows
verified_file_count
```

Exit codes are exact:

```text
0 = PASS
2 = registered verification failure
3 = invocation or internal verifier error
```

A negative probe passes only when exit code is `2`, `first_error` equals the
registered code, all earlier gates are `PASS` and all later gates are
`NOT_EVALUATED`. A generic later failure is insufficient.

Mutation target, operation, replacement bytes/injection checkpoint and clean
baseline precondition are exact in `hostile_mutations` of the surface
contract. No additional formal hostile probe is permitted in V1.

## 12. Formal Run

Development tests may be repeated before implementation freeze. After
implementation is committed, independent readiness runs the distinct
`READINESS_STRUCTURAL_ONLY` mode in both the primary and detached worktrees:

```text
readiness worktree:
  /Users/liu/Documents/hftbacktest-0831t001-q0-readiness

readiness output:
  /Users/liu/Documents/hftbacktest-0831t001-q0-readiness-output
```

This happens before claim consumption. It is repeatable software readiness,
not the formal Q0 attempt and not terminal-package verification. The
readiness verifier does not require or fabricate `formal_identity.json`,
claimed bytes, attempt-lock, consumption commit or controller observations.
Each worktree independently completes and checks all 57 instrumented feature
calls, then publishes only the exact 37-file projection frozen in
`readiness_comparison.projection_files`:

```text
builds/A/structural
builds/B/structural
builds/P/structural
abp_comparison.json
```

Each projection tree hash is canonical compact JSON over sorted
`{path,size_bytes,sha256}` rows. The two 37-file projections must be
byte-identical file by file and have equal tree hashes. Runtime/PID/root
evidence is independently checked in each readiness process but is ephemeral:
it is neither normalized nor compared and cannot enter structural bytes.

Formal identities are:

```text
implementation tag:
  skhynix-trade-led-depth-follower-q0-implementation-v1

implementation commit message:
  feat: implement 0831T001 Q0 pipeline qualification

arming commit message:
  audit: arm 0831T001 Q0 attempt

consumption tag:
  skhynix-trade-led-depth-follower-q0-consumed-v1

terminal tag:
  skhynix-trade-led-depth-follower-q0-terminal-v1

armed claim:
  .workflow/attempt-claims/0831T001.armed.json

claimed path:
  .workflow/attempt-claims/0831T001.claimed.json

terminal receipt:
  .workflow/attempt-receipts/0831T001.terminal.json

attempt root:
  /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1

package root:
  /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1/package

controller bare repository:
  /Users/liu/Documents/hftbacktest-0831t001-q0-controller.git

controller ref:
  refs/heads/codex/0831T001-controller-ledger

recovery witness ref:
  refs/tags/skhynix-trade-led-depth-follower-q0-recovery-start-v1
```

The pre-formal Git chronology is exact:

```text
implementation commit
-> annotated implementation tag targets that commit
-> atomically publish armed claim
-> stage only armed claim
-> arming commit, parent = implementation commit
-> formal outer driver starts with HEAD = arming commit
```

The claim binds the implementation commit, not the later arming commit.
Before each transition's first stage command, the index is empty and tracked
worktree is clean except the registered transition paths. Exact stage
commands are:

```text
arming:
  git ... add -- .workflow/attempt-claims/0831T001.armed.json

consumption:
  git ... add -A -- .workflow/attempt-claims/0831T001.armed.json
    .workflow/attempt-claims/0831T001.claimed.json

terminal common:
  git ... add -- both tracked receipts and 0831T001-business.md

terminal PASS only:
  git ... add -- baselines/skhynix_trade_led_depth_follower_q0_v1
```

The ellipsis is the exact command-scoped Git prefix below. Arming and
consumption each verify the index after their sole stage command. Terminal
PASS verifies the aggregate index only after `terminal_stage_common` and
`terminal_stage_PASS`; terminal FAIL verifies after
`terminal_stage_common`.

The observation algorithm is exact, rename-invariant and independent of
worktree normalization:

```text
cached index:
  git ... diff --cached --no-renames --raw -z --full-index --abbrev=40 HEAD

unstaged tracked:
  git ... diff --no-renames --raw -z --full-index --abbrev=40

complete HEAD tree:
  git ... ls-tree -r -z --full-tree HEAD

complete actual index:
  git ... ls-files --stage -z
  git ... ls-files -v -z

untracked:
  git ... ls-files --others --exclude-standard -z

physical inventory:
  expected index = HEAD tree + exact cached preimage
  expected physical = expected index + exact worktree/untracked preimage
  no-follow lstat every expected path
  exact 0644/0755/symlink kind and mode
  SHA256 over exact bytes
  Git blob over exact bytes with no filters
```

Raw output is parsed as repeated `metadata NUL path NUL` pairs, never as one
combined NUL field. Empty bytes mean zero records; nonempty output requires
one final NUL and an even number of nonempty fields. The exact metadata
regex, one-letter status, normalized strict-ASCII paths and unique paths are
mandatory.

The selected action phase plus immutable PASS/FAIL terminal branch resolves
one of 23 registered preimage variants. Every row is expanded to the exact
six fields, sorted by ASCII path and compared byte-for-byte. For consumption
staging, the cached rows are exactly:

```text
D .workflow/attempt-claims/0831T001.armed.json
A .workflow/attempt-claims/0831T001.claimed.json
```

Both modes and blobs are bound, and the armed old blob equals the claimed new
blob and the independent claim-byte Git blob. Every stage-0 index path is
then rebound to the physical filesystem through no-follow `lstat`, exact mode,
SHA256 and unfiltered Git-blob derivation. Thus `core.filemode=false`,
autocrlf or attributes cannot hide a mode or byte mutation. Rename similarity,
`--name-only`, porcelain directory collapse and repository configuration are
never authority. The arming commit adds one `100644` claim, the consumption
commit deletes armed and adds byte-identical claimed, and the terminal commit
has only its registered PASS/FAIL delta.

After readiness passes and before the armed claim is created, controller
infrastructure is prepared in the retryable
`PRE_CONSUMPTION_RETRYABLE_INFRASTRUCTURE` stage:

```text
if controller repository is absent:
  run the exact git init --bare command
  apply the exact two controller config commands
  fsync the bare repository and its parent

if it is already prepared:
  verify bare format, exact config, empty object database and absent controller
  ref
  do not reinitialize

if any ref exists or config/ownership is unexpected:
  fail closed before claim arming
```

After init/config, both the bare repository directory and its parent are
fsynced with `O_DIRECTORY`. Failure in this stage consumes no claim and
creates no attempt root, so repair and retry are allowed only after proving
there are no refs or objects. Controller creation/configuration is prohibited
inside the consumed formal state machine.

Before any formal fixture is generated, the state machine executes in this
single order:

```text
preflight validates all armed-claim bindings and verifies the controller ref
is absent without writing state
create the no-replace attempt root
atomically publish deterministic attempt-lock.json

the validated armed claim binds:
  implementation commit and tag
  parent/master/plan/task/truth/runner/verifier/tests hashes
  exact cwd, roots and argv
  zero historical/outcome authorization

atomically rename armed -> claimed and fsync the parent
the consumption commit and annotated tag are created
push the consumption commit with an absent-ref --force-with-lease CAS
write and fsync either the normal consumption PUSH_CALL receipt or, only
after crash recovery observes the expected ref, the distinct REF_OBSERVATION
receipt
copy the chosen consumption-transition receipt byte-for-byte to the tracked
receipt path
acquire the producer runtime flock and atomically publish its invocation claim
run the exact formal producer child argv at most once and immediately fsync
`<attempt_root>/control/formal_producer_exit.json` after waitpid
acquire the verifier runtime flock and atomically publish its invocation claim
run the exact terminal verifier child argv at most once and immediately fsync
`<attempt_root>/control/terminal_verifier_exit.json` after waitpid
```

Exact claim, attempt-lock, push-receipt and terminal-receipt fields; Git fsync
configuration; commit/tag messages; transition commands; crash-state
classification and retry policy are the `one_shot` object in the surface
contract.

Every Git command that may write objects, refs, index, config or a remote ref
uses this exact command-scoped prefix:

```text
git -c commit.gpgSign=false -c core.fsync=all -c core.fsyncMethod=fsync
```

The same prefix is used to verify the effective values before claim arming.
Repository-local stronger or weaker defaults are not authority because the
command-scoped values override them. Controller init/config, both
commit/tag pairs and both lease-CAS pushes are exact commands in
`one_shot.git_commands`. They are stored as argv token arrays and executed
without a shell; displayed command lines are explanatory only.

The only retryable state is:

```text
controller preparation failure or any other failure before no-replace
attempt-root creation
```

Any state at or after successful no-replace attempt-root creation is terminal
for this task, even if attempt-lock, claim rename, commit, tag or controller
update did not complete.

Receipt ownership is exact:

```text
<attempt_root>/control/consumption_push_receipt.json
  normal-path `PUSH_CALL`, durable immediately after the consumption CAS

<attempt_root>/control/consumption_push_observation.json
  recovery-only `REF_OBSERVATION`; allowed only when the normal receipt is
  absent and exact ls-remote observes the expected consumption SHA

.workflow/attempt-receipts/0831T001.consumption-push.json
  exact copy of whichever consumption-transition receipt exists, included in
  the terminal commit

.workflow/attempt-receipts/0831T001.terminal.json
  tracked PASS/FAIL terminal result receipt included in the terminal commit

<attempt_root>/control/terminal_push_receipt.json
  normal-path `PUSH_CALL`, durable immediately after the terminal CAS

<attempt_root>/control/terminal_push_observation.json
  recovery-only `REF_OBSERVATION`; allowed only when the normal receipt is
  absent and exact ls-remote observes the expected terminal SHA

<attempt_root>/control/recovery_observation.json
  required exactly once on every recovery path; forbidden on the normal path

<attempt_root>/control/recovery_start.json
  recovery identity published from the externally witnessed blob before any
  recovery state change; reused byte-for-byte by every recovery restart

controller recovery witness
  `refs/tags/skhynix-trade-led-depth-follower-q0-recovery-start-v1`
  points directly to the exact recovery-start Git blob

independent QA report
  `.workflow/reports/0831T001-qa.md` and
  `docs/qa-acceptance-report.md` select exactly one evidence mode:
  `NORMAL`, `RECOVERY` or `WITNESS_BLOCKED`. Normal records terminal evidence
  and no recovery fields. Recovery records terminal evidence, exact
  recovery-start/recovery-observation SHA256 and witness ref/blob OID.
  Witness-blocked records classification `NONE`, the exact artifact blocker
  SHA256 and its frozen witness diagnostic evidence; recovery hashes are
  `NONE`, while a valid pre-existing terminal receipt is recorded by exact
  SHA256 and an absent or invalid one by `NONE`.
```

The terminal push uses `--force-with-lease` expecting the exact consumption
commit. Plain observe-then-push is prohibited. Every crash boundary from
attempt-root creation through terminal-push receipt has the single
interpretation frozen in `one_shot.crash_states`.

Recovery may complete only deterministic controller/filesystem transitions:
attempt lock, claim rename, consumption commit/tag/push, receipt copy,
PASS/FAIL receipt/report, accepted baseline copy, terminal commit/tag/push
and observation receipts. Recovery never launches a producer or verifier.
An existing invocation claim consumes that child's sole launch right even
when no exit receipt exists. A missing invocation claim is not created by
recovery; it maps to the registered pre-producer or pre-verifier interruption
error.

After acquiring the orchestrator lock, recovery first observes the controller
ref under the relevant push-runtime lock. PRE_BLOCKER then evaluates the
exact local Git phase before artifact rules or proof-stage selection. A
tracked receipt cannot advance the phase unless its canonical receipt, exact
transition tuple, selected untracked source bytes and tracked-copy bytes all
validate.

On a legal ledger ref, recovery derives canonical `recovery_start.json` bytes
from the unchanged initial state. Before any local recovery-start target or
temporary exists, and before any rename, commit, tag, ledger-ref push,
receipt copy, child-state classification, baseline publication or
terminalization, recovery must create or verify:

```text
controller repo:
  /Users/liu/Documents/hftbacktest-0831t001-q0-controller.git

witness ref:
  refs/tags/skhynix-trade-led-depth-follower-q0-recovery-start-v1

witness target:
  one Git blob whose bytes are the exact canonical recovery_start.json,
  including final LF
```

The exact Git command prefix writes the blob and creates the witness ref with
`update-ref <witness-ref> <blob-oid> 000...000`; this is a CAS from ABSENT.
The controller repository and parent are fsynced, then the ref object type and
blob bytes are independently verified. Only after that external binding may
the ordinary local sibling-temporary/hard-link publication begin.

The crash interpretation is exact:

```text
witness absent + all local recovery-start paths absent:
  derive from unchanged snapshot, create witness, publish local target

witness absent + any local recovery-start path present:
  ARTIFACT_STATE_CORRUPTION / A12_RECOVERY_WITNESS_MISMATCH

witness exact + local target absent + temporary absent:
  publish witnessed bytes through the generic protocol

witness exact + local target absent + exact regular temporary:
  retain the open temporary FD and st_dev/st_ino, hard-link no-replace, reopen
  final target O_NOFOLLOW, require exact bytes and the same st_dev/st_ino,
  fsync final and parent, then unlink the still-regular temporary

hard-link returns EEXIST:
  reopen final target O_NOFOLLOW and reclassify through the exact-target row;
  accept only exact regular witnessed bytes, otherwise A12 without unlink

witness exact + local target absent + mismatched/truncated regular temporary:
  unlink and fsync, then rebuild from witnessed bytes

witness exact + local target absent + nonregular temporary:
  ARTIFACT_STATE_CORRUPTION / A12_RECOVERY_WITNESS_MISMATCH

witness exact + local target exact regular:
  accept through O_NOFOLLOW + fstat + descriptor read; remove only a regular
  temporary

witness exact + local target mismatch/nonregular:
  ARTIFACT_STATE_CORRUPTION / A12_RECOVERY_WITNESS_MISMATCH

witness invalid type/unreadable/conflicting:
  ARTIFACT_STATE_CORRUPTION / A12_RECOVERY_WITNESS_MISMATCH;
  freeze witness evidence first, then allow only blocker publication and
  deterministic local claim consumption. No controller mutation,
  recovery-start/observation publication, terminalization or baseline change.
```

The witness is part of the existing controller CAS trust boundary. Any
out-of-protocol mutation of controller refs or objects is workflow integrity
corruption and is never repaired by recovery.

Every A12 observation is frozen inside the canonical
`artifact_state_corruption.json` as `recovery_witness_evidence_json`. The
embedded object records exact ref/object/blob subprocess exit codes and
stdout/stderr SHA256 values, observed OID and object type, plus no-follow
state and SHA256 for the local target and temporary. Non-A12 blocker receipts
must set this field to `NONE`. Once the A12 blocker target commits, its
restart rows supersede the recovery matrix; later repair or drift cannot
rewrite the recorded observation.

The witnessed file's `recovery_id`, original crash boundary, initial
controller SHA and committed-control path set never change even if recovery
later advances refs or commits and then crashes. The terminal receipt
contains no recovery hash and is never rewritten. Late recovery after
terminal-receipt publication is represented only by the witnessed recovery
files and independent QA evidence fields.

Before publishing a new `recovery_start.json`, recovery observes the
controller ref under the corresponding push-runtime lock. The exact legal
set is selected from local durable state and may contain `ABSENT`, the
consumption SHA and/or the terminal SHA as registered for that proof stage by
the surface contract. Any observed `ABSENT` or 40-hex token outside that
exact set is not converted into a new Q0 FAIL. It
publishes the canonical `controller_ref_divergence.json`, completes only an
unfinished local claim consumption, and preserves the consumed claim. Before
a terminal receipt exists, classification is `NONE` and terminal artifacts
are forbidden. After a terminal receipt exists, its Q0 classification,
first error and report remain immutable; recovery may only complete a
missing local terminal commit/tag. Terminal push is forbidden in both
branches. Independent QA binds the blocker-observation SHA256 and records
workflow status `阻塞`. This is a controller-ledger integrity outcome, not a
new software qualification result.

If a push succeeded but the process crashed before persisting its normal
receipt, recovery does not reconstruct lost push stdout/stderr. It runs the
exact `ls-remote` observation and writes the separately typed
`REF_OBSERVATION` receipt. The tracked consumption receipt and QA terminal
receipt hash bind to the actually present member of the corresponding
`PUSH_CALL | REF_OBSERVATION` union.

The exact recovery-driver argv is:

```text
/Users/liu/.local/conda/bin/python examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py --recover --attempt-root /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1
```

The recovery driver itself may be restarted after a recovery-process crash.
Its writes use the atomic control-publication protocol, and every Git action
is selected from observed ref state and guarded by the registered lease. A
recovery restart may never create a producer or verifier invocation claim.

Every content-bearing control artifact is published as:

```text
open <target>.publishing O_EXCL|O_NOFOLLOW and retain its descriptor identity
-> write complete canonical bytes
-> verify full write
-> fsync temporary file
-> hard-link temporary to target no-replace
-> reopen target O_NOFOLLOW
-> require regular exact bytes and matching st_dev/st_ino
-> fsync final target
-> fsync parent
-> unlink temporary
-> fsync parent
```

If the hard-link returns `EEXIST`, the final target is reopened no-follow and
classified independently as an existing target. Exact regular bytes may win
the race; mismatch or nonregular kind fails closed and the temporary is not
deleted. No pathname `exists/is_file/read_bytes` check or final-component
`resolve()` is publication authority.

Only the final target is committed authority. A truncated temporary is never
authority. While holding the orchestrator lock, recovery removes and rebuilds
a deterministic temporary from independently derived bytes. For an
unreconstructable PUSH_CALL or child-exit temporary, it removes only the
uncommitted temporary after the matching runtime lock is acquirable, then
uses REF_OBSERVATION or interruption semantics. Attempt lock, invocation
claim, tracked receipt copy, process exit receipt, recovery observation and
terminal receipt all use this protocol.
`one_shot.control_publication_crash_states` separately freezes before-link
and after-link handling for every such artifact; a generic “file exists”
test is never enough.

The outer driver and recovery driver serialize on an exclusive flock held on
`<attempt_root>/control/orchestrator.lock`. Recovery cannot begin until it
acquires that lock. For each producer/verifier child, the parent acquires the
runtime flock before invocation publication and passes the already locked
descriptor with:

```text
runtime fd slot = 198
ACK fd slot = 199
close_fds = true
pass_fds = (198,199)
start_new_session = true
```

The parent first proves fd slots `198/199` are closed, duplicates the locked
runtime file and ACK pipe writer into those exact slots, and names them in the
child argv. Immediately after successful `Popen`, it closes its original
runtime-lock FD, original pipe writer and local fd `199`, but retains local fd
`198` until handoff resolves. On `Popen` error it closes every original and
fixed-slot descriptor before publishing the exit receipt.

The child validates fd `198` against the exact runtime-lock path and inherited
flock, validates fd `199`, writes exactly byte `0x41`, closes fd `199`, then
does any attempt work. The parent accepts only `0x41` followed by EOF within
5000ms monotonic time. EOF-before-ACK, another byte, a second byte or timeout
is a handoff error; it SIGTERMs the child process group, waits 1000ms,
SIGKILLs if required, and calls `waitpid` once. The child retains fd `198`
through its final fsynced child-owned write and closes it only immediately
before return. Recovery therefore cannot acquire the lock while any child can
still mutate attempt state.

Each Git push has a separate runtime flock. The driver passes that locked file
descriptor only to the exact `git push` child, so an orphan push retains the
lock after its parent exits. Recovery must acquire the same lock before
`ls-remote` or a side-effect-free retry; observed ref state is therefore
post-child, not a race with an in-flight push.

The two formal process receipts use the exact field sets and canonical
serializer in `one_shot.formal_process_receipts`. Before each `Popen`, the
outer driver atomically publishes an invocation claim containing the exact
child argv, kind and ordinal. The exit receipts persist child argv,
exit codes and stdout/stderr hashes before the next transition. The verifier
receipt additionally persists `first_error`, result SHA and terminal-manifest
SHA. Child kind, ordinal, launch status, handoff status and missing-value
domains are exact in the surface contract. Invocation claim without exit
receipt maps to `FORMAL_PRODUCER_INTERRUPTED` or
`TERMINAL_VERIFIER_INTERRUPTED`; it never authorizes a rerun.

All SHA fields use the exact domains in `one_shot.sha_tokens`. In particular,
an absent controller ref is the ASCII token `ABSENT`; an unavailable process
exit, result or recovery artifact is `NONE`. Empty string and all-zero SHA
sentinels are forbidden. `contracts/formal_identity.json.controller_pre_sha`
is therefore exactly `ABSENT` for Q0, and the normal consumption
`old_sha/pre_ls_remote_sha` fields are also exactly `ABSENT`.

The exact formal cwd is:

```text
/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol
```

The exact outer-driver argv is:

```text
/Users/liu/.local/conda/bin/python examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py --formal --claim .workflow/attempt-claims/0831T001.armed.json --attempt-root /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1
```

The runner consumes the armed claim before generating fixtures; the path
argument remains the pre-consumption authority named by the claim and is
renamed by the runner.

The exact producer child argv is:

```text
/Users/liu/.local/conda/bin/python examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py --formal-producer --attempt-root /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1 --package-root /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1/package --truth .workflow/contracts/0831T001-fixture-truth-v1.json --surface-contract .workflow/contracts/0831T001-q0-surface-contract-v1.json --runtime-lock-fd 198 --handoff-ack-fd 199
```

After the formal producer stops, the terminal verifier runs at most once even
if the producer exit code is nonzero. Its exact child argv is:

```text
/Users/liu/.local/conda/bin/python examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py --package-root /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1/package --result /Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol/local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1/control/terminal_verifier_result.json --runtime-lock-fd 198 --handoff-ack-fd 199
```

No post-formal regeneration is permitted. Terminalization has two exact
branches:

```text
PASS:
  classification = Q0_PIPELINE_QUALIFIED
  terminal manifest SHA256 = required hex
  copy exact 57-file package to baseline
  first_error = NONE

FAIL:
  classification = Q0_PIPELINE_NOT_QUALIFIED
  baseline copy = forbidden
  terminal manifest SHA256 = NONE if no valid manifest exists
  producer/verifier exit field = NONE when its exit receipt is absent or its
    legal tuple is POPEN_ERROR; integer only for STARTED
  result field = hash exactly when the legal verifier tuple carries a result,
    otherwise NONE
  first_error = one result from the shared durable-state resolver:
    committed terminal receipt is immutable authority
    absent producer invocation -> pre-producer interruption
    producer invocation without exit -> producer interruption
    committed producer launch/handoff/nonzero error wins over every later
      orchestration or verifier state
    only a successful producer allows pre-verifier/verifier errors to become
      first_error
    successful producer + verifier exit 2 -> registered verifier first error
    successful producer + accepted verifier exit 0 -> NONE
  completed_stages_json and missing_stages_json are required
```

The normal outer driver and recovery call the same resolver. Before the
10 chronological cases, both apply the ordered artifact-state machine:

```text
presence bits:
  producer invocation
  producer exit
  verifier invocation
  verifier exit
  verifier result
  baseline

A01-A05:
  reject impossible predecessor order

A06-A08:
  reject unregistered process tuples, result-presence mismatch and baseline
  without a successful producer plus accepted verifier

A09-A11:
  reject a terminal receipt/profile mismatch or a report without exact
  terminal-receipt authority

A12:
  reject any recovery witness ref/local-target/temporary matrix violation;
  freeze the exact command/output hashes and no-follow local path evidence
  in recovery_witness_evidence_json before any blocker restart write
```

The 64 presence combinations have a canonical derived table: 7 legal shapes,
57 invalid shapes, aggregate SHA256
`f902794088e1b7649ca78fbc7053856fa51126d45e9d0347e061a73c65dd251b`.
The first matching invalid rule publishes
`artifact_state_corruption.json`, consumes the one-shot claim locally and
sets workflow status `阻塞`, classification/profile `NONE`; it never creates
or rewrites a terminal receipt/report and never pushes the controller ref.

For a valid artifact shape, the resolver's 10 explicit cases cover absent
invocation, absent exit, producer error with or without later verifier
artifacts, successful producer with each verifier state, and accepted PASS.
Crash boundaries select recovery mode; they cannot overwrite a committed
producer error. Stage lists are selected from the frozen machine profiles
after independently observing committed invocation/exit/result artifacts;
producer-supplied stage claims are not trusted.

The terminal receipt does not use prose as stage identity. It uses only:

```text
F00_PREFLIGHT
F01_CONSUMPTION_CLOSED
F02_PRODUCER_INVOCATION
F03_PRODUCER_EXIT
F04_VERIFIER_INVOCATION
F05_VERIFIER_EXIT
F06_RESULT_CLASSIFIED
F07_BASELINE_PUBLISHED
```

`completed_stages_json` and `missing_stages_json` are JSON string fields
whose values must be one exact compact array from
`one_shot.terminal_stage_profiles`; even an empty set is `"[]"`, never
`NONE`. Every crash boundary maps through
`one_shot.crash_recovery_matrix` to one recovery mode and durable stage
profile; `RESOLVE_FROM_DURABLE_STATE` and
`FROM_COMMITTED_TERMINAL_RECEIPT` are resolved only by the shared machine
resolver. FAIL is valid with any explicitly registered early-stop FAIL
profile; it does not require all `F00..F06`. The producer/verifier launch,
handoff, exit, result, manifest and first-error tuples are an exhaustive
machine table.
The surface contract also freezes canonical PASS and recovery-FAIL receipt
preimages with their expected byte hashes as executable derivation tests.

The business execution report is exactly
`.workflow/reports/0831T001-business.md`, UTF-8 with LF and final LF. Its
complete PASS/FAIL line arrays, placeholder sources, join rule and example
hashes are `one_shot.execution_report`; FAIL has no baseline line and makes
no claim that absent producer/verifier stages executed. It prints the exact
committed and missing stage arrays from the terminal receipt.
The allowed FAIL domain contains exactly 49 profile/error renderings. Their
canonical derivation uses implementation commit `1111...1111`, produces
49 unique report SHA256 values with sizes from 1472 to 1537 bytes, and has
canonical-row aggregate SHA256
`73f050f783e7659a955c8a6650f28550665abb358b06db9c4ab9e15111bcdc48`.
The registered pre-producer example is 1537 bytes with SHA256
`17305de4942f54c75f837dd0e17abb0405dd7752b427c252ac514e545cebc2ec`.
The
terminal commit parent is the exact consumption commit. Its common tracked
delta is exactly:

```text
.workflow/attempt-receipts/0831T001.consumption-push.json
.workflow/attempt-receipts/0831T001.terminal.json
.workflow/reports/0831T001-business.md
```

PASS additionally adds exactly the 57 baseline files; FAIL adds no baseline
path. Any other tracked delta is forbidden.

Both branches create the tracked terminal result receipt and execution report,
then create the same terminal commit/tag and lease-bound controller push. A
verifier failure is therefore durably terminalized rather than left in a
state with no receipt.

If the verifier passes, the exact 57 package files are copied byte-for-byte,
without regeneration, to:

```text
baselines/skhynix_trade_led_depth_follower_q0_v1/
```

PASS publication first prepares the exact 57 files under
`<attempt_root>/control/baseline_candidate`, verifies every byte against the
accepted `terminal_manifest.json`, and fsyncs every file and directory. It
then atomically renames that directory no-replace to the baseline path and
fsyncs the baseline parent. Recovery may resume an incomplete attempt-owned
candidate; an already published baseline is accepted only if its exact path
set and bytes match. This prevents a crash during copy from exposing a
partial tracked baseline.

The terminal result receipt fields and PASS/FAIL sentinels are frozen in
`one_shot.terminal_receipt_fields` and
`one_shot.terminalization_branches`.

## 13. Acceptance

Q0 passes only if all hold:

```text
all 15 fixture expectations match
fixture truth Git/SHA binding exact
physical fixture arrays independently match truth
all registered negative probes fail at their exact first boundary
full/slice mismatch count = 0
comparable epoch and anchor floors are nonzero and satisfied
hidden feature-payload dependency count = 0
A/B raw and sealed difference counts = 0
A/P raw and sealed difference counts = 0
all A/B/P physical inputs and calls are independently bound
forbidden raw value access count = 0
causal-prefix mismatch count = 0
causal future-read count = 0
cross-segment carry count = 0
terminal verifier PASS
focused tests PASS
accepted fixed-epoch regression PASS
pre-consumption fresh detached-worktree readiness PASS
formal package file count = 57
baseline bytes equal the accepted formal package
```

The terminal classification is exactly one of:

```text
Q0_PIPELINE_QUALIFIED
Q0_PIPELINE_NOT_QUALIFIED
```

Only `Q0_PIPELINE_QUALIFIED`, followed by independent QA acceptance, may
authorize drafting A-1a.

## 14. Failure Semantics

Any non-divergent formal-run failure is terminal for `0831T001`.

The task must record:

```text
the first failing boundary
the observed error code
completed and missing package stages
Q0_PIPELINE_NOT_QUALIFIED
```

Controller observation has separate phases:

```text
before attempt-root:
  only ABSENT is legal
  unexpected SHA, command failure or malformed output writes no state
  armed claim remains retryable after controller repair

after attempt-root:
  unexpected SHA -> CONTROLLER_REF_DIVERGENCE
  nonzero command or malformed output -> CONTROLLER_OBSERVATION_FAILURE
```

After attempt-root, either controller blocker is a workflow integrity
blocker. Before a terminal receipt exists it is recorded as:

```text
workflow status = 阻塞
classification = NONE
blocker = CONTROLLER_REF_DIVERGENCE or CONTROLLER_OBSERVATION_FAILURE
claim = consumed and never reusable
terminal Q0 receipt/report/commit/tag/push = forbidden
```

If a terminal receipt already exists, its classification and first error
remain authoritative, but workflow status is still `阻塞`; a missing local
terminal commit/tag may be completed and terminal push remains forbidden.
The blocker restart rows cover observation publication, claim
rename, consumption commit/tag, report rendering from a valid receipt and,
only for controller blockers with valid immutable receipt/report, local
terminal commit/tag. Artifact corruption rows preserve any receipt, report or
terminal Git history without extending it. After a blocker observation is
committed these rows supersede the ordinary crash matrix; every row restarts
from the same observed durable state and forbids controller push.

Controller-ref authority is receipt-sensitive:

```text
attempt root / consumption commit before push:
  ABSENT

consumption push attempted but no durable receipt:
  ABSENT or consumption SHA

durable consumption receipt:
  consumption SHA only

terminal push attempted but no durable receipt:
  consumption SHA or terminal SHA

durable terminal push receipt:
  terminal SHA only
```

Thus disappearance to `ABSENT` after a durable consumption receipt is
`CONTROLLER_REF_DIVERGENCE`, not a recoverable absent-ref state.

Before any restart row is selected, the ordered local Git state machine checks:

```text
G01 expected controller ref for the exact proof stage
G02 armed/claimed state
G03 exact HEAD
G04 exact commit object, parent, message and tree delta
G05 exact raw/index/untracked/physical arrays for the selected phase/branch
G06 exact annotated consumption tag
G07 exact annotated terminal tag
```

G01 produces controller divergence. G02-G07 produce
`ARTIFACT_STATE_CORRUPTION` and prohibit further commit, tag or push. After a
controller blocker receipt is committed, its remote observation is frozen and
restart evaluation skips G01 but still applies G02-G07.

Every legal tracked-transition state is classified from the exact raw, full
index and physical inventory under the eight-way cross-product of
`diff.renames`, `core.filemode` and `core.autocrlf`. Command-scoped values
make repository configuration irrelevant, while physical hashing remains
independent of Git normalization. The 23 variants times four path/mode/blob/
staging-partition probes times eight configurations form 736 deterministic
rows; all select G05 and their canonical aggregate SHA256 is
`4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`.

Revision 19 retains seven controller-proof stages and expands the machine to
16 exact Git action phases. The new
`NORMAL_CONSUMPTION_UNTRACKED_RECEIPT_COMMITTED` phase represents a valid
durable control receipt before its tracked byte-identical copy. The tracked
transition state remains one of:

```text
CLEAN
EXACT_CONSUMPTION_RENAME_UNSTAGED
EXACT_CONSUMPTION_INDEX_STAGED
EXACT_CONSUMPTION_RECEIPT_UNSTAGED
EXACT_BOTH_RECEIPTS_REPORT_MISSING_UNSTAGED
EXACT_TERMINAL_DELTA_UNSTAGED
EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED
EXACT_TERMINAL_INDEX_STAGED
INVALID
```

The full pre-blocker cross-product has 124,416 rows: 18 legal and 124,398
invalid. Its canonical aggregate SHA256 is
`8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`.

The post-controller-blocker table skips G01 and enumerates the 11 executable
restart action phases across local claim/HEAD/tag/commit/tracked-transition
state. It has 21,384 rows: 11 legal and 21,373 invalid, with aggregate
SHA256
`d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356`.

Any invalid committed process/result/baseline/terminal/report artifact state
instead records:

```text
workflow status = 阻塞
classification = NONE
blocker = ARTIFACT_STATE_CORRUPTION
first_invalid_rule = first A01-A12 match
terminal receipt/report rewrite and controller push = forbidden
```

No repair, diagnosis, plan change or rerun is permitted inside the consumed
formal run. A software correction requires a new task and a new independent
qualification review.

No Q0 result may be described as evidence for or against the market
hypothesis.

## 15. Review And Handoff

Required order:

```text
candidate plan/task commit
-> independent plan review
-> implementation
-> independent implementation readiness
-> armed claim and implementation tag
-> one formal Q0 run
-> terminal verifier
-> independent QA
-> workflow closure
```

The plan review and readiness review report:

```text
P0/P1/P2/P3 counts
reviewed commit
reviewed plan/task hashes
outcome-boundary status
historical-cache access status
```

QA writes the latest result to:

```text
.workflow/reports/0831T001-qa.md
docs/qa-acceptance-report.md
```

## 16. Exact Gate Ownership

Formal gate order is:

```text
Q0-0:
  source roots, path kinds, tracked authority bindings and claim

Q0-1:
  raw schema, clock, domain and canonical fixture truth

Q0-2:
  FeatureBundle schema and source field access

Q0-3:
  A_MINUS1A causal access boundary and anchor contract

Q0-4:
  A_MINUS1B outcome access boundary and structural cause contract

Q0-5:
  slice publication, reset and non-vacuous identity

Q0-6:
  A/B/P physical input and consumer binding

Q0-7:
  package root/path/kind closure

Q0-8:
  package schemas, canonical bytes and lineage

Q0-9:
  fixture-truth observed semantics

Q0-10:
  A/B and A/P structural byte identity

Q0-11:
  registered hostile mutation first-error matrix

Q0-12:
  terminal closure
```

If gate `Q0-k` fails:

```text
Q0-0..Q0-(k-1) = PASS
Q0-k = FAIL
Q0-(k+1)..Q0-12 = NOT_EVALUATED
classification = Q0_PIPELINE_NOT_QUALIFIED
```

No later package or report failure may rewrite an earlier observed gate.

## 17. Prohibited Self-Proof

The following are explicitly insufficient:

```text
runner PASS consumed as verifier truth
tests importing runner expected rows
verifier importing runner fixture generator
verifier trusting runner input SHA fields without opening physical inputs
B/P copying A structural package without independent calls
zero slice mismatches with an empty comparable universe
output invariance without typed access evidence
manifest validation that trusts its own listed path set
fresh-worktree verification performed only after the one-shot attempt
```

The verifier may import canonical serialization helpers from the structural
core only if their normalized AST identity is frozen and hostile tests
independently reject non-canonical bytes. It may not import fixture
generation, expected-output or runner orchestration functions.
