# SKHYNIX Fixed Epoch Suppression Baseline and Recovery

Date: 2026-08-30

Baseline ID: `SKHYNIX_FIXED_EPOCH_SUPPRESSION_V1`

## 1. Purpose

This document freezes fixed causal epoch suppression as infrastructure for
successor research. It does not reopen the negative scientific result of
`0829T003` and does not authorize A0, future-outcome access or live trading.

The baseline separates two identities:

1. `suppression authority`: the exact code, plan, tests and QA-accepted
   workflow state at commit `f06eb5cb012cb62b2a778ad90d433c4083f9ba14`.
2. `research kit`: this recovery document, machine-readable manifest,
   verifier, tests and tracked evidence snapshot.

Future hypotheses should depend on the authority rather than edit it.

## 2. Frozen Behavior

The authority freezes:

```text
checkpoint = 20ms
epoch origin = Unix epoch 0
epoch width = 60s
eligible core = [epoch_start + 15s, epoch_start + 45s)
thinning key = (capture_id, epoch_id, direction)
retained onset = earliest by (candidate_ts_ns, candidate_event_seq)
cluster key = (capture_id, epoch_id)
complete grid = 3,000 checkpoints in one segment
```

The authority also freezes:

- ineligible epoch disposition and precedence;
- no anchor, exposure or null contribution from an ineligible epoch;
- same-epoch, same-direction deterministic thinning;
- shared epoch cluster for opposite directions;
- raw-slice feature rebuild and support-identity comparison;
- exact 25 non-cache outputs;
- A/B/P triad comparison and poison attestation;
- sequential A-1 gates and A0/future-outcome locks.

The accepted scientific classification remains:

```text
Aminus1_structural_support_not_estimable
```

QA acceptance means the execution is auditable. It does not mean the M-state
or a trading strategy has scientific support.

## 3. Git Authority Nodes

| Role | Commit |
|---|---|
| Frozen reviewed plan | `f0110d40cf6eaf21f4e8972383a01b65c3f33f70` |
| Initial implementation | `781a7cd0fd1ccceee214c3292a86611ff02546cf` |
| Hostile-test closure | `b583d02b12a2464a9d58863c3b761fca6f37e8d3` |
| Independent QA acceptance | `8463eae645a14f37532be35937c5c52ca2ce4941` |
| Accepted workflow closure | `f06eb5cb012cb62b2a778ad90d433c4083f9ba14` |

Recovery tags:

```text
skhynix-fixed-epoch-suppression-v1
  -> exact accepted workflow closure f06eb5cb

skhynix-fixed-epoch-research-kit-v1
  -> baseline documentation/verifier/evidence closure
```

The first tag is the scientific and implementation authority. The second tag
is the convenient starting point for new research.

## 4. Frozen Files

Successor tasks must not modify these files in place:

```text
docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md
examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
examples/hyperliquid/test_skhynix_fixed_causal_epoch_mstate_a_minus1.py
.workflow/tasks/0829T003.md
.workflow/reports/0829T003-execution.md
.workflow/reports/0829T003-qa.md
```

The exact file SHA256, Git blob OID, callable AST hashes and constants are in:

```text
baselines/skhynix_fixed_epoch_suppression_v1/baseline_manifest.json
```

Changing epoch width, core geometry, origin, thinning key, tie-break,
disposition semantics or cluster identity requires a new hypothesis/version.
It must not be described as the same baseline.

## 5. Evidence Preservation

`local_live_analysis*` is ignored by Git. The canonical output and temporary
Build B/P roots therefore cannot be the only recovery authority.

The research kit tracks:

```text
baselines/skhynix_fixed_epoch_suppression_v1/evidence_noncache.tar.gz
baselines/skhynix_fixed_epoch_suppression_v1/poison-attestation.json
```

The archive contains the exact 25 canonical non-cache artifacts. It excludes
the 29 source caches. `source_cache_inventory.csv` inside the archive retains
their names, row counts and hashes.

Frozen evidence identity:

```text
non-cache artifact count = 25
non-cache tree SHA256 =
  2d5505b531b2da97a928284ce9fd80696d79ee9870cf290507cf37d94e63049f
run manifest SHA256 =
  3cf8b26781909ec39f47d4efcb72e64d500d19e2f0f682ac6cabf4966dc8a748
summary SHA256 =
  67b5cc80baa086cd2c9e0f21eef47cc4f49c7949ebde00142d3c18098f8cfd85
poison attestation SHA256 =
  3eaca61d3769f2dee6c50aea45a95109fc622b2ce557277451b504c8aca8942b
```

## 6. Verification

Verify the Git authority, tracked snapshot and unchanged working-tree frozen
files:

```bash
python examples/hyperliquid/skhynix_fixed_epoch_suppression_baseline.py \
  --check-working-tree
```

When the original A/B/P roots are available, also verify all 25 outputs:

```bash
python examples/hyperliquid/skhynix_fixed_epoch_suppression_baseline.py \
  --check-working-tree \
  --canonical-output \
    local_live_analysis/skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003 \
  --build-b-output \
    /tmp/skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003_build_b_v2 \
  --poison-output \
    /tmp/skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003_poison_v2
```

After tags are created:

```bash
python examples/hyperliquid/skhynix_fixed_epoch_suppression_baseline.py \
  --check-working-tree \
  --require-tags
```

Focused regression:

```bash
python -m pytest \
  examples/hyperliquid/test_skhynix_fixed_epoch_suppression_baseline.py \
  examples/hyperliquid/test_skhynix_fixed_causal_epoch_mstate_a_minus1.py \
  examples/hyperliquid/test_skhynix_fresh_channel_consensus_mstate_a_minus1.py
```

## 7. Safe Successor Workflow

Start successor research from the research-kit tag in a new worktree:

```bash
git worktree add \
  ../hftbacktest-fixed-epoch-successor \
  -b codex/<new-hypothesis> \
  skhynix-fixed-epoch-research-kit-v1
```

The successor should:

1. create a new plan, task, runner and tests;
2. import or bind the frozen authority callables by exact Git/AST identity;
3. keep the authority files unchanged;
4. register any new epoch width or online provisional semantics explicitly;
5. begin again at outcome-blind precision-first A-1;
6. create its own A/B/P evidence namespace.

Do not run destructive reset commands in a dirty worktree. Recovery should
use a new worktree.

## 8. Recovery

Recover only the accepted suppression authority:

```bash
git worktree add \
  ../hftbacktest-fixed-epoch-authority \
  skhynix-fixed-epoch-suppression-v1
```

Recover the complete research kit:

```bash
git worktree add \
  ../hftbacktest-fixed-epoch-kit \
  skhynix-fixed-epoch-research-kit-v1
```

Restore the canonical non-cache evidence:

```bash
mkdir -p local_live_analysis/fixed_epoch_restored
tar -xzf \
  baselines/skhynix_fixed_epoch_suppression_v1/evidence_noncache.tar.gz \
  -C local_live_analysis/fixed_epoch_restored
```

The archive restores evidence, not the 29 source caches. A formal rerun still
requires the cache files listed in the restored
`support/source_cache_inventory.csv`.

## 9. Online Causality Boundary

The fixed epoch phase and earliest-onset thinning are causal. Full 60s
single-segment eligibility is finalized only at epoch close.

A live successor must distinguish:

```text
provisional_valid_at_t
final_structural_valid
```

Trading on a provisional anchor creates a real action that cannot be deleted
if the epoch is later invalidated. Any live-oriented successor must count
those invalidations rather than use final validity as an ex-post trade filter.
