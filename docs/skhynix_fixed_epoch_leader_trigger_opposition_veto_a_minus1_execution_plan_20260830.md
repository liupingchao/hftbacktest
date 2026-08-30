# SKHYNIX Fixed Epoch Leader Trigger With Opposition Veto A-1 Execution Plan

Date: 2026-08-30

Task ID: `0830T002`

Hypothesis ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1`

Audit ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1_A_MINUS1`

Revision: 13, pre-execution

## 1. Objective and Prediction

Execute one frozen, outcome-blind structural-support audit.

Unique primary:

```text
TRADE_LED
```

Non-rescue sensitivities:

```text
DEPLETION_LED
OFI_LED
```

Primary prediction:

```text
confirmed clusters >= 30
represented dates >= 4
maximum single-date share <= 0.50
```

## 2. Git and Authority

Research-kit start:

```text
tag: skhynix-fixed-epoch-research-kit-v1
commit: 45afe2446e27449bb8f3c8ffde7e95663b58e4fb
```

Suppression authority:

```text
tag: skhynix-fixed-epoch-suppression-v1
commit: f06eb5cb012cb62b2a778ad90d433c4083f9ba14
runner:
  examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
```

The successor directly calls these authority functions:

```text
source_preflight
base_eligibility
channel_actions
channel_memories
epoch_support_ledger
materialize_poisoned_cache_set
verify_poison_attestation
```

The baseline manifest supplies the epoch authority file SHA256, Git blob OID
and callable AST SHA256. Formal tests must prove direct invocation; a local
reimplementation is forbidden.

The raw-to-feature authority is independently frozen:

```text
path:
  examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py
commit:
  45544ecc3901623ca7c2e34a059afca6c551d625
blob_oid:
  494c203e7195f292e057f7708c99f52096259a02
file_sha256:
  f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c
callable:
  build_features
callable_ast_sha256:
  e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933
```

Build A, Build B, Build P and every sliced raw cache must directly invoke
this exact callable. A wrapper may count and attest calls but may not replace
the callable body. Expected direct calls equal:

```text
3 * canonical cache count
+ total artificial slice build count across A/B/P
```

Every call is bound by a formal feature-call row containing build, input path,
input SHA, output feature SHA and consumer-input SHA. The last two hashes must
be equal. Builds A, B and P are separate subprocesses with no shared Python
state.

The detector receives only the exact feature dictionary returned by the bound
builder. The successor runner itself has zero `np.load` call sites and no raw
cache value interface. The poison authority is the only component allowed to
read unconsumed raw values.

Each build subprocess installs a fail-closed `numpy.load` proxy only around
the bound feature builder. The proxy:

1. permits field-name enumeration for exact schema validation;
2. permits `__getitem__` only for the 12 consumed fields;
3. records every permitted value access;
4. raises before returning any unconsumed value;
5. restores the original loader before control returns.

The builder return is immediately hashed, passed as the sole `features`
argument to the detector, and hashed again at detector entry. A mismatch,
discarded return, alternate feature object or second consumer is a terminal
execution failure: no scientific classification is synthesized.
Poison materialization runs in the orchestrator outside this detector proxy
and is measured by the frozen poison attestation.

Each build worker has two phases:

```text
LOADER:
  exact raw path is available only to bound build_features + load proxy
DETECTOR:
  raw path references are deleted
  all feature arrays are writeable=False
  an audit hook rejects every .npz/raw open
  detector exit feature hash must equal entry hash
```

The successor's detector callables are AST-rejected if they contain or alias
`open`, `Path.open`, `os.open`, `numpy.load`, `NpzFile`, `zipfile`,
`importlib`, `__import__`, `eval` or `exec`. Proxy early restore, alternate
loader, raw open, feature mutation or boundary-ledger defect is terminal
execution failure, not A-1-1.

Enforcement uses fresh subprocesses with `close_fds=True`:

```text
HASHER:
  may byte-read exactly one registered input and returns only SHA256

SLICE_MATERIALIZER:
  may read exactly one canonical/poison input
  may write exactly one registered work slice
  returns only output path/SHA

LOADER:
  receives exactly one registered path
  installs permanent sys.addaudithook before any input open
  allowed raw read stack must terminate in the exact bound build_features
  proxy; returns only serialized feature arrays and access ledger
  process exits immediately after send

DETECTOR:
  separate process receives serialized features only
  receives no raw/work/source/poison path
  all inherited descriptors except stdio/control pipe are closed
  every .npz/raw open event is forbidden
```

Canonical audited event types are exactly `open` and `mmap.__new__`.
LOADER allows one read-open authority chain rooted at the bound
`build_features`; HASHER and SLICE_MATERIALIZER have separately frozen caller
functions. `zipfile`, `io.open`, `NpzFile` and aliases still emit one of the
audited events and are rejected outside the exact authority chain.

The static AST/import closure covers the entire successor runner, verifier,
worker entry points and orchestrator. Raw-reader call sites are allowed only
inside exact named `sha256_file`, `materialize_slice`, the imported bound
builder and imported poison authority. Dynamic import/eval/exec and audit-hook
replacement are forbidden. A successful formal package has only
`RawOpenEvent.allowed=true`, zero DETECTOR raw events, and zero raw handles,
NPZ objects, mmap objects or raw-byte buffers crossing a process boundary.

Input SHA is the SHA256 of exact input file bytes. Feature SHA is SHA256 of a
canonical JSON array sorted by feature name, with each row:

```text
name,dtype.str,shape,value_sha256
```

`value_sha256` hashes the C-contiguous array bytes. FULL inputs use:

```text
A/B:
  source_cache_root/{cache_name}
P:
  attempt_root/poison_cache/{cache_name}
```

SLICED inputs use retained paths:

```text
attempt_root/work/{A|B|P}/{cache_name}/slice_{slice_ordinal:06d}.npz
```

No random temporary path is permitted. Global `call_index` is the contiguous
zero-based range `0 .. analyzed_unit_count-1`.

Let `S` be the independently verified slice-row count for one build in
`work-manifest.json`. A, B and P each have exactly `S` slice rows:

```text
analyzed_unit_count = 3 * 29 + 3 * S
```

`source_preflight` must complete for all rows before action, memory, trigger
or output construction. Its failure is uniquely A-1-0.

## 3. Source Authority

Canonical immutable cache root:

```text
/Users/liu/Documents/
hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/
local_live_analysis/
skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache
```

The exact 29 names, row counts, schema versions, sizes and SHA256 values are
the baseline `support/source_cache_inventory.csv`.

Build A and Build B read the same immutable canonical bytes. "Fresh" means a
new empty output root and a new process invocation inside the one-shot
orchestrator; it does not mean regenerated caches.

Permitted consumed fields are exactly the accepted baseline consumed set.
Every allowed-minus-consumed field is poison authority and may not affect an
output.

## 4. Frozen Detector

The idea document's Revision 12 definitions and the task-frozen idea SHA are
normative, in this only order:

- checkpoint-exact causal order;
- raw onset;
- epoch/core omission;
- confirmation-edge omission;
- anchor-time veto;
- support counts;
- fixed-epoch thinning;
- explicit-evidence confirmation;
- cancellation booleans and primary reason.

Shared values:

```text
TTL = 100ms, age <= 100ms
prestate = [t-120ms,t-20ms], six checkpoints
confirmation = [t+20ms,t+200ms], ten checkpoints
margin = 0.00
fast threshold = +/-0.50 inclusive
medium threshold = +/-0.25 inclusive
```

Variants:

```text
TRADE_LED: leader index 0
DEPLETION_LED: leader index 1
OFI_LED: leader index 2
```

## 5. Fixed Epoch Contract

Direct-call authority `epoch_support_ledger`.

Frozen:

```text
epoch origin = 0
epoch width = 60s
checkpoint = 20ms
expected checkpoints = 3,000
core = [15s,45s)
thinning key = (capture_id,epoch_id,variant,direction)
tie-break = (candidate_ts_ns,candidate_event_seq)
cluster = capture_id:epoch_id
```

Confirmation must close inside the core. Core-close equality is allowed:

```text
candidate_ts_ns + 200ms <= core_close_ns
```

An otherwise valid raw onset with `candidate_ts_ns + 200ms > core_close_ns`
is `confirmation_edge_omitted` before veto and thinning. It does not occupy a
key. Equality is admitted; a candidate 20ms later is omitted.

## 6. Conservation Contract

For each `(date,capture,epoch,variant,direction)`:

```text
raw_onset_count
  = epoch_core_omitted_count
  + confirmation_edge_omitted_count
  + anchor_vetoed_count
  + veto_admitted_count

veto_admitted_count
  = retained_count
  + same_key_suppressed_count

retained_count
  = confirmed_count
  + cancelled_count
```

Every count is a non-negative base-10 integer. Violation is A-1-2.

## 7. Slice/Reset Invariance

Nominal artificial starts:

```text
stride = 600s per segment
guard = 122s
```

Every slice is rebuilt from sliced raw cache via the authority feature
builder frozen in Section 2. No derived feature reuse is permitted.

Comparable epochs:

```text
epoch_id >=
  (actual_start_ts_ns + 122_000_000_000 + EPOCH_NS - 1) // EPOCH_NS
eligible in both full and slice
same artificial-start segment
```

For every qualifying start, compare exact typed, numerically sorted
identities.

Epoch disposition identity:

```text
(capture_id,epoch_id,disposition,segment_id_or_empty,
 segment_ids_json,segment_set_sha256,observed_checkpoint_count,
 duplicate_timestamp_count,off_grid_timestamp_count,
 missing_expected_timestamp_count,grid_exact)
```

Epoch/variant counter identity:

```text
(capture_id,epoch_id,variant,direction,
 raw_onset_count,epoch_core_omitted_count,anchor_vetoed_count,
 confirmation_edge_omitted_count,
 veto_admitted_count,retained_count,same_key_suppressed_count,
 confirmed_count,cancelled_count,retained_candidate_id_or_empty)
```

Retained trigger identity:

```text
(capture_id,variant,epoch_id,direction,candidate_ts_ns,
 candidate_event_seq,cluster_id)
```

Retained status identity adds:

```text
secondary_same_direction_count
secondary_opposite_count
additional_same_leader_update_count
opposite_update_count
first_additional_same_update_ts_ns_or_empty
first_additional_same_update_event_seq_or_empty
confirmation_window_close_ts_ns
confirmation_window_close_event_seq
confirmation_status
cancel_reason
four cancellation booleans
```

Support identity:

```text
(capture_id,epoch_id,checkpoint_ts_ns,channel_index,
 action_int,memory_int,memory_age_ms)
```

Canonical identity hashing:

```text
JSON sort_keys=true
separators=(",",":")
ensure_ascii=true
Python typed int/bool/str
SHA256
```

Each identity stores expected/actual count and SHA256. Required:

```text
all exact flags true
cross-segment checkpoint count = 0
represented slice dates >= 4
distinct comparable epochs >= 30
positive support checkpoint count
```

## 8. Durable One-Shot Claim

The only formal command is:

```bash
python \
  examples/hyperliquid/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py \
  --formal-attempt \
  --repo-root /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate \
  --source-cache-root \
    /Users/liu/Documents/hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/local_live_analysis/skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache \
  --attempt-root \
    /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate/local_live_analysis/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_0830T002_formal_v1
```

Implementation freeze creates an annotated tag:

```text
skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1
```

Claim consumption creates:

```text
commit message:
  audit: consume 0830T002 formal attempt claim
annotated tag:
  skhynix-fixed-epoch-leader-trigger-a-minus1-consumed-v1
```

Formal completion creates:

```text
tracked receipt:
  .workflow/attempt-receipts/0830T002.terminal.json
commit message:
  audit: seal 0830T002 formal attempt result
annotated tag:
skhynix-fixed-epoch-leader-trigger-a-minus1-terminal-v1
```

The controller authority is the external GitHub remote, outside the local
worktree and object store:

```text
remote name:
  origin
remote URL:
  git@github.com:liupingchao/hftbacktest.git
ledger ref:
  refs/heads/codex/0830T002-controller-ledger
```

Before any cache read, the source repo pushes the consumption commit to the
external ledger and requires `git ls-remote` to equal the consumption head.
Formal completion pushes the terminal commit as the ledger's only
fast-forward and requires the remote ref to equal the terminal head.

Exact remote protocol:

```text
remote URL checks:
  git remote get-url --all origin
  git remote get-url --push --all origin
  each returns exactly one line:
    git@github.com:liupingchao/hftbacktest.git

pre-consumption observation:
  git ls-remote --heads origin \
    refs/heads/codex/0830T002-controller-ledger
  stdout must be empty

consumption push:
  git push --porcelain origin \
    <consumption_head>:refs/heads/codex/0830T002-controller-ledger
  expected old/new = 0000000000000000000000000000000000000000 /
                     <consumption_head>

post-consumption observation:
  same ls-remote command
  exactly "<consumption_head>\\t<ledger_ref>\\n"

terminal push:
  git push --porcelain origin \
    <terminal_head>:refs/heads/codex/0830T002-controller-ledger
  expected old/new = <consumption_head> / <terminal_head>

post-terminal and verifier observation:
  same ls-remote command
  exactly "<terminal_head>\\t<ledger_ref>\\n"
```

Pre-existing equal refs are forbidden; the first observation must be empty.
Any additional fetch/push URL, push attempt, refspec or observed old/new tuple
is terminal failure.

`git push --porcelain` stdout/stderr is diagnostic only; it is not old/new
authority. Exit code must be zero. Full old/new tuples are derived only from
the registered pre/post `ls-remote` observations:

```text
attempt-lock remote_observations:
  PRE_CONSUMPTION: exit_code=0, stdout="", stderr="",
                   observed_head=null
  POST_CONSUMPTION: stdout exact full line,
                    exit_code=0, stderr="",
                    observed_head=consumption_head

attempt-lock remote_transitions:
  CONSUMPTION:
    old_head=null
    new_head=consumption_head
    derived_from=["PRE_CONSUMPTION","POST_CONSUMPTION"]

attempt-lock successful_push_count=1

verifier remote_observations:
  exact copies of PRE_CONSUMPTION and POST_CONSUMPTION from attempt-lock
  POST_TERMINAL: exit_code=0, stderr="",
                 online ls-remote exact full line,
                 observed_head=terminal_head

verifier remote_transitions:
  CONSUMPTION as above
  TERMINAL:
    old_head=consumption_head
    new_head=terminal_head
    derived_from=["POST_CONSUMPTION","POST_TERMINAL"]

verifier successful_push_count=2
```

The verifier also requires the external ref history from consumption to
terminal to be one commit and requires terminal parent exactly consumption.
Any nonzero `ls-remote` exit code is terminal failure regardless of stdout;
empty stdout is interpreted as absence only when exit code is exactly zero.

Push-call ledger:

```text
push-ledger/000-consumption.json:
  exactly one PushCall
  ordinal=0
  phase="CONSUMPTION"
  exact registered consumption argv/refspec
  expected_old_head=null
  expected_new_head=consumption_head
  pre/post IDs PRE_CONSUMPTION/POST_CONSUMPTION
  retry_allowed=false

push-ledger/001-terminal.json:
  exactly one PushCall
  ordinal=1
  phase="TERMINAL"
  exact registered terminal argv/refspec
  expected_old_head=consumption_head
  expected_new_head=terminal_head
  pre/post IDs POST_CONSUMPTION/POST_TERMINAL
  retry_allowed=false
```

Each file is exact `PushCall` JSON, written once with the common no-replace
protocol. Attempt-lock `push_calls` contains only the byte-identical
consumption row. Verifier `push_calls` contains the two byte-identical rows in
ordinal order. Both exit codes must be zero; any nonzero call is terminal and
cannot be retried. Exact completed child closure requires precisely these two
files and rejects a third file, duplicate ordinal, up-to-date repush or any
unledgered push call. The runner/verifier AST permits remote push only through
the single `push_once` wrapper that always publishes one receipt.

Threat model:

- protects against local worktree/object-store deletion, reset, crash,
  accidental rerun and deviations by the formal runner from its exact command
  sequence;
- does not claim resistance to any actor, admin or ordinary writer, who holds
  remote write credentials and deliberately force-pushes, deletes, appends a
  third commit or otherwise changes the ledger outside the exact two-push
  protocol;
- any such remote-writer action is outside the model and invalidates the
  study; it is never a supported recovery or replacement attempt.

and commits exactly one tracked armed claim:

```text
.workflow/attempt-claims/0830T002.armed.json
```

The independent terminal verifier is:

```text
examples/hyperliquid/
skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py
```

It has no detector-building authority. It independently recomputes Git
transitions, schemas, path sets, manifests, comparisons, tree hashes, sibling
hashes and terminal receipt closure.

Its only command is:

```bash
python \
  examples/hyperliquid/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py \
  --verify-terminal \
  --repo-root /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate \
  --attempt-root /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate/local_live_analysis/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_0830T002_formal_v1 \
  --implementation-tag skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1 \
  --consumption-tag skhynix-fixed-epoch-leader-trigger-a-minus1-consumed-v1 \
  --terminal-tag skhynix-fixed-epoch-leader-trigger-a-minus1-terminal-v1 \
  --result-out /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate/.workflow/reports/0830T002-terminal-verifier.json
```

CWD is the repo root. Exit codes are exactly:

```text
0 = PASS result written
2 = verification FAIL result written
64 = CLI/usage failure, no result written
```

The verifier is read-only except for one no-replace result file. It may read
Git objects/refs, source cache bytes, poison cache bytes, retained work bytes,
the three output roots and siblings. It must not use `np.load`, inspect future
outcomes, execute A0 or modify any research artifact.

The claimed path is:

```text
.workflow/attempt-claims/0830T002.claimed.json
```

Before any `.npz` open, the orchestrator must:

1. use lexical absolute registered roots and reject symlinks in every existing
   path component with `lstat`;
2. require Git local configuration exactly `core.fsync=all`,
   `core.fsyncMethod=fsync`, `core.logAllRefUpdates=always`;
3. require a clean worktree, exact formal argv/cwd,
   `HEAD == implementation-tag`, and absence of consumption/terminal tags;
4. scan all refs, all reflogs and
   `git fsck --full --unreachable --no-reflogs`; reject any prior exact
   claim-transition blob/tree, consumption commit message, terminal receipt
   path/blob or terminal commit message;
5. verify idea, plan, task, runner, verifier, tests, armed claim, authority files,
   callable ASTs and baseline tags;
6. require armed present, claimed absent and attempt root absent;
7. create claimed as a hard link to armed, which fails if claimed exists;
8. `fsync` claimed, `fsync` the claim directory, unlink armed, then `fsync`
   the claim directory again;
9. commit the exact same-blob armed-to-claimed rename as the only tree delta,
   create the annotated consumption tag, verify its parent is the
   implementation tag, run `git fsck --full`, and reverify Git fsync settings;
10. create the attempt root and `push-ledger/` with `mkdir`, then `fsync`
    their parent directories;
11. push the consumption commit to the external controller ledger without
    force, verify the remote ref exactly, and no-replace publish
    `push-ledger/000-consumption.json`;
12. create and `fsync` `attempt-lock.json` through a same-directory temporary
   file, hard-link no-replace publish, temporary unlink and parent `fsync`;
13. only then open source caches.

The independent verifier requires the consumption tag and exact commit/tree
transition. Restoring armed and deleting claimed/root cannot erase that
authority. A missing terminal tag or result after consumption means
`INTERRUPTED_TERMINAL`; it is not recoverable or replaceable in this task.

Exact attempt children:

```text
attempt-lock.json
canonical_a/
canonical_b/
poison_cache/
poison_p/
poison-attestation.json
instrumentation-evidence.json
push-ledger/
work/
work-manifest.json
attempt-result.json
```

## 9. Build Sequence and Comparison Domains

The one-shot process performs:

1. Build A over canonical caches into empty `canonical_a`;
2. Build B in a fresh subprocess over the same canonical bytes into empty
   `canonical_b`;
3. materialize poison caches by direct authority call;
4. write and verify sibling `poison-attestation.json`;
5. Build P over poison caches into empty `poison_p`, while inventory authority
   remains canonical;
6. compare the `RAW_11` projection under the registered comparison mode;
7. derive one global gate/classification payload from A plus A/B and A/P
   comparison evidence, and write identical dynamic bytes into A/B/P;
8. compare `SEALED_15` under the registered comparison mode;
9. write identical `execution_evidence.json`;
10. write each self-excluding manifest;
11. compare `FINAL_17` externally under the registered comparison mode;
12. publish and fsync siblings `instrumentation-evidence.json` and
    `work-manifest.json`;
13. publish and fsync sibling `attempt-result.json`;
14. create the exact tracked terminal receipt, commit only that receipt,
    create the annotated terminal tag and verify its parent is the consumption
    commit; run `git fsck --full` and reverify Git fsync settings;
15. push the terminal commit without force as the only controller-ledger
    fast-forward, verify the remote ref exactly, and no-replace publish
    `push-ledger/001-terminal.json`.

Exact projections:

```text
RAW_11 =
  authority_binding.json
  detector_contract.json
  fixed_epoch_contract.json
  eight support CSV files

SEALED_15 =
  RAW_11
  + outcome_access_ledger.json
  + gate_contract.json
  + A_minus1_summary.json
  + classification.json

EVIDENCED_16 =
  SEALED_15
  + execution_evidence.json

FINAL_17 =
  EVIDENCED_16
  + run_manifest.json
```

Registered comparison modes:

```text
A/B:
  exact file-byte SHA256 for every path in RAW_11, SEALED_15 and FINAL_17

A/P:
  exact file-byte SHA256 for every path except:
    support/slice_invariance.csv
    run_manifest.json
  poison-normalized semantic SHA256 for those two paths only
```

The A/P exception is necessary because Build P deliberately changes all
registered unconsumed cache fields. The retained physical slice therefore has
a different work-file SHA even when the consumed feature projection is
identical. `slice_source_sha256` records that physical identity and must bind
the same-build WorkRow; it is not itself a consumed scientific feature.

For A/P comparison of `support/slice_invariance.csv`, the semantic hash
preimage is constructed exactly as follows:

1. decode the file as ASCII CSV;
2. require the exact registered header;
3. retain the original row order, row count and every parsed string value;
4. replace only each row's `slice_source_sha256` value with 64 ASCII zeroes;
5. encode the resulting list of row objects with
   `json.dumps(value, sort_keys=True, separators=(",", ":"),
   ensure_ascii=True).encode("ascii")`;
6. use SHA256 of those bytes as the `ComparisonRow` hash.

For A/P comparison of `run_manifest.json`, the semantic hash preimage is
constructed exactly as follows:

1. decode the file as ASCII JSON;
2. require `artifacts` to be a list with exactly one row whose `path` is
   `support/slice_invariance.csv`;
3. preserve every key, value, list order and artifact row;
4. replace only that row's `sha256` with the poison-normalized semantic SHA256
   of the same root's `support/slice_invariance.csv`;
5. encode the full JSON object with the same canonical JSON rule above;
6. use SHA256 of those bytes as the `ComparisonRow` hash.

No other path, field, value, row, count, order, size, or manifest identity is
normalized. Missing or extra files remain differences. Malformed headers,
missing or duplicated manifest slice rows, and any non-registered mutation
fail closed. The producer and terminal verifier independently recompute the
same registered projection.

The physical A/B/P slice and manifest bytes remain permanent evidence.
Same-build `slice_source_sha256`, WorkRow SHA, SLICE FeatureCall input
authority, work-manifest closure, consumer feature-output equality and
terminal manifest self-exclusion are verified before accepting the semantic
comparison. Thus normalization removes only the mechanical identity change
caused by deliberately poisoned unconsumed bytes; it does not hide a consumed
feature, detector, support, gate, or package mutation.

`execution_evidence.json` contains only `RAW_11` and `SEALED_15` comparison
rows. It never hashes itself or either manifest. `attempt-result.json` is the
pre-terminal closure over `FINAL_17` plus exactly:

```text
claimed attempt file
attempt-lock.json
push-ledger/000-consumption.json
poison-attestation.json
instrumentation-evidence.json
work-manifest.json
work tree
```

It explicitly excludes `push-ledger/001-terminal.json`, which does not exist
until after terminal push.

Every FINAL_17 file, poison attestation and `attempt-result.json` is written
once using same-directory temporary creation, file fsync, hard-link
no-replace publication, temporary unlink and parent-directory fsync. After all
children publish, each A/B/P directory and the attempt root are fsynced before
`attempt-result.json` publication.

Every retained slice is also no-replace published and fsynced.
`work-manifest.json` lists every work file and is bound by
`attempt-result.json` and the tracked terminal receipt. Work evidence is
permanent for this task and may not be cleaned after claim consumption.

Canonical A/B differences belong only to A-1-0. A/P differences under the
registered poison-normalized comparison belong only to A-1-1. An A/P
mismatch is retained as negative outcome-boundary evidence; it does not
prevent final package creation and is not reassigned to A-1-0.

Frozen poison expectations:

```text
cache_count = 29
unconsumed_field_count = 15
nonempty field instances = 435
changed field instances = 435
consumed mismatch = 0
```

## 10. Exact 17 Output Paths

Each A/B/P root contains:

```text
classification.json
contracts/authority_binding.json
contracts/detector_contract.json
contracts/execution_evidence.json
contracts/fixed_epoch_contract.json
contracts/gate_contract.json
contracts/outcome_access_ledger.json
reports/A_minus1_summary.json
run_manifest.json
support/channel_action_by_date.csv
support/epoch_support.csv
support/epoch_variant_counters.csv
support/slice_invariance.csv
support/source_cache_inventory.csv
support/support_by_date.csv
support/trigger_ledger.csv
support/variant_summary.csv
```

`run_manifest.json` excludes itself and lists exactly 16 unique entries with
path, size and SHA256. Missing, extra, duplicate or cache payload paths fail.

## 11. Common Serialization

JSON:

```text
UTF-8/ASCII-compatible
indent=2
sort_keys=true
ensure_ascii=true
one trailing newline
```

Canonical hash JSON:

```text
sort_keys=true
separators=(",",":")
ensure_ascii=true
```

CSV:

```text
UTF-8/ASCII-compatible
Unix newline
exact ordered header
True/False booleans
base-10 integers
finite decimal floats
empty string for N/A identity
```

No NaN or infinity may be emitted.

## 12. CSV Schemas

### source_cache_inventory.csv

Row grain: one canonical cache, sorted by `cache_name`.

```text
cache_name,size_bytes,row_count,cache_schema_version,cache_sha256,
source_authority_verified,cache_field_schema_verified
```

### channel_action_by_date.csv

Row grain: `(research_date,channel)`, sorted ASCII.

```text
research_date,channel,total_action_count,global_invalid_action_count,
new_invalid_action_count,new_pos_action_count,new_neg_action_count,
new_neutral_action_count,no_update_action_count,observed_new_evidence_count,
expiry_count,neutral_overwrite_count,unauthorized_ttl_refresh_count,
cross_segment_memory_carry_count,maximum_memory_age_ms,
action_partition_exact
```

`maximum_memory_age_ms` is an integer. If no memory was established for the
date/channel, its only sentinel is `-1`; otherwise it is non-negative.

### epoch_support.csv

This is the exact successor projection of the authority epoch ledger; no
other authority fields are emitted:

```text
research_date,capture_id,epoch_id,epoch_start_ns,epoch_end_ns,
core_open_ns,core_close_ns,segment_id,segment_count,segment_ids_json,
segment_set_sha256,disposition,observed_checkpoint_count,
unique_timestamp_count,duplicate_timestamp_count,
off_grid_timestamp_count,missing_expected_timestamp_count,grid_exact
```

`segment_id` is empty unless disposition is `eligible`.

### epoch_variant_counters.csv

Row grain: every enumerated
`(research_date,capture_id,epoch_id,variant,direction)`, including zeros.

```text
research_date,capture_id,epoch_id,variant,direction,
raw_onset_count,epoch_core_omitted_count,anchor_vetoed_count,
confirmation_edge_omitted_count,veto_admitted_count,retained_count,
same_key_suppressed_count,
confirmed_count,cancelled_count,retained_candidate_id
```

Sort by date ASCII, capture ASCII, epoch numeric, variant registered order,
direction numeric.

### trigger_ledger.csv

Row grain: one retained trigger.

```text
research_date,capture_id,variant,epoch_id,epoch_start_ns,core_open_ns,
core_close_ns,segment_id,direction,candidate_id,candidate_ts_ns,
candidate_event_seq,dependence_cluster_id,leader_channel,
secondary_same_direction_count,secondary_opposite_count,leader_age_ms,
secondary_age_json,additional_same_leader_update_count,
opposite_update_count,first_additional_same_update_ts_ns,
first_additional_same_update_event_seq,confirmation_window_close_ts_ns,
confirmation_window_close_event_seq,confirmation_status,cancel_reason,
insufficient_confirmation_history,confirmation_segment_boundary,
explicit_opposite_update,no_additional_same_leader_update
```

`confirmation_status` is exactly `CONFIRMED` or `CANCELLED`.

Candidate ID is canonical SHA256 of:

```text
(capture_id,variant,epoch_id,segment_id,direction,
 candidate_ts_ns,candidate_event_seq)
```

Optional first-additional fields are empty together.

### support_by_date.csv

Row grain: `(research_date,variant,direction)`.

```text
research_date,variant,direction,raw_onset_count,epoch_core_omitted_count,
confirmation_edge_omitted_count,anchor_vetoed_count,veto_admitted_count,
retained_count,
same_key_suppressed_count,confirmed_count,cancelled_count,
distinct_confirmed_cluster_count,support0_confirmed_count,
support1_confirmed_count,support2_confirmed_count
```

### variant_summary.csv

Row grain: one registered variant.

```text
variant,is_primary,raw_onset_count,veto_admitted_count,retained_count,
confirmed_count,cancelled_count,distinct_confirmed_cluster_count,
represented_date_count,maximum_single_date_cluster_share,
support_prediction_passed
```

If cluster count is zero, maximum share is empty and prediction is False.

### slice_invariance.csv

Row grain: one qualifying artificial start.

```text
research_date,capture_id,segment_id,nominal_start_ts_ns,
actual_start_ts_ns,comparison_floor_ns,first_comparable_epoch_id,
slice_source_sha256,comparable_epoch_count,
expected_epoch_disposition_count,actual_epoch_disposition_count,
expected_epoch_disposition_sha256,actual_epoch_disposition_sha256,
epoch_disposition_exact,expected_counter_count,actual_counter_count,
expected_counter_sha256,actual_counter_sha256,counter_exact,
expected_retained_count,actual_retained_count,expected_retained_sha256,
actual_retained_sha256,retained_exact,expected_status_count,
actual_status_count,expected_status_sha256,actual_status_sha256,
status_exact,expected_support_count,actual_support_count,
expected_support_sha256,actual_support_sha256,support_exact,
cross_segment_checkpoint_count,mismatch_reason
```

`mismatch_reason` is always `none` or the first failed item in this exact
precedence:

```text
epoch_disposition
counter
retained
status
support
cross_segment
```

## 13. Exact JSON and Sibling Schemas

Schema notation:

```text
str = nonempty ASCII string
text = ASCII string, empty allowed
int = base-10 integer, bool forbidden
number = finite JSON number, bool forbidden
bool = true/false
sha256 = 64 lowercase hex
sha1 = 40 lowercase hex
nullable[T] = T or null
list[T] = ordered JSON array
object{...} = exactly listed keys; additional keys forbidden
```

Named records:

```text
FileIdentity = object{
  path:str,sha256:sha256,git_blob_oid:sha1
}

CallableIdentity = object{
  path:str,commit:sha1,git_blob_oid:sha1,file_sha256:sha256,
  callable_name:str,callable_ast_sha256:sha256,direct_call_count:int
}

FeatureCall = object{
  call_index:int,build_label:str,unit_kind:str,capture_id:str,
  research_date:str,slice_ordinal:nullable[int],resolved_input_path:str,
  input_sha256:sha256,input_authority:str,feature_output_sha256:sha256,
  sender_ipc:IPCSender,receiver_ipc:IPCReceiver,
  consumer_input_sha256:sha256,
  detector_exit_sha256:sha256,field_name_schema_access_count:int,
  consumed_value_access_count:int,forbidden_value_access_count:int,
  consumer_use_count:int
}

IPCArrayRow = object{
  name:str,dtype_str:str,shape:list[int],offset_bytes:int,
  length_bytes:int,value_sha256:sha256
}

IPCSender = object{
  header_sha256:sha256,payload_sha256:sha256,payload_size_bytes:int,
  frame_sha256:sha256,frame_size_bytes:int,sent_frame_count:int,
  send_end_closed:bool
}

IPCReceiver = object{
  header_sha256:sha256,payload_sha256:sha256,payload_size_bytes:int,
  frame_sha256:sha256,frame_size_bytes:int,received_frame_count:int,
  eof_observed:bool,unused_byte_count:int
}

RemoteObservation = object{
  observation_id:str,command:list[str],exit_code:int,stdout:text,stderr:text,
  observed_head:nullable[sha1]
}

RemoteTransition = object{
  transition_id:str,old_head:nullable[sha1],new_head:sha1,
  derived_from:list[str]
}

PushCall = object{
  ordinal:int,phase:str,argv:list[str],refspec:str,
  expected_old_head:nullable[sha1],expected_new_head:sha1,
  exit_code:int,stdout:text,stderr:text,
  pre_observation_id:str,post_observation_id:str,
  started_at_utc:str,finished_at_utc:str,retry_allowed:bool
}

FieldAccess = object{
  call_index:int,build_label:str,resolved_input_path:str,
  input_sha256:sha256,field:str,value_access_count:int,
  authorization:str
}

RawOpenEvent = object{
  event_index:int,call_index:int,build_label:str,phase:str,event_type:str,
  resolved_path:str,operation:str,caller_path:str,caller_name:str,
  allowed:bool
}

ComparisonRow = object{
  path:str,a_sha256:nullable[sha256],other_sha256:nullable[sha256],
  equal:bool
}

Comparison = object{
  domain:str,expected_path_count:int,a_path_count:int,other_path_count:int,
  difference_count:int,rows:list[ComparisonRow]
}

GateCondition = object{
  condition:str,status:str,passed:nullable[bool],
  actual:nullable[int|bool|str|number],required:str
}

Gate = object{
  gate_id:str,status:str,passed:nullable[bool],
  conditions:list[GateCondition]
}

VariantRow = object{
  variant:str,is_primary:bool,raw_onset_count:int,
  veto_admitted_count:int,retained_count:int,confirmed_count:int,
  cancelled_count:int,distinct_confirmed_cluster_count:int,
  represented_date_count:int,
  maximum_single_date_cluster_share:nullable[number],
  support_prediction_passed:bool
}

Integrity = object{
  source_preflight_violation_count:int,
  action_partition_violation_count:int,
  unauthorized_ttl_refresh_count:int,
  cross_segment_memory_carry_count:int,
  conservation_violation_count:int,fixed_epoch_violation_count:int,
  slice_mismatch_count:int,cross_segment_compared_checkpoint_count:int,
  represented_slice_date_count:int,distinct_comparable_epoch_count:int,
  compared_support_checkpoint_count:int,schema_violation_count:int,
  numeric_violation_count:int
}

ManifestRow = object{
  path:str,size_bytes:int,sha256:sha256
}

RootRow = object{
  label:str,path:str,artifact_count:int,tree_sha256:sha256,
  manifest_sha256:sha256,classification:str
}

WorkRow = object{
  build_label:str,cache_name:str,slice_ordinal:int,path:str,
  size_bytes:int,sha256:sha256
}

VerifierCheck = object{
  check_id:str,status:str,actual:text,required:str
}
```

Every JSON file has `schema_version=1` and exactly these remaining keys:

```text
contracts/authority_binding.json = object{
  schema_version:int,task_id:str,hypothesis_id:str,audit_id:str,
  baseline_tag:str,baseline_commit:sha1,implementation_tag:str,
  implementation_head:sha1,consumption_tag:str,consumption_head:sha1,
  tracked_files:list[FileIdentity],
  callables:list[CallableIdentity],source_inventory_sha256:sha256,
  attempted_claim_sha256:sha256,all_verified:bool
}

contracts/detector_contract.json = object{
  schema_version:int,hypothesis_id:str,variant_order:list[str],
  channel_order:list[str],fast_threshold:number,medium_threshold:number,
  margin:number,ttl_ms:int,ttl_inclusive:bool,prestate_ms:int,
  prestate_checkpoint_count:int,confirmation_ms:int,
  confirmation_checkpoint_count:int,causal_order:list[str],
  onset_rule:str,confirmation_edge_rule:str,veto_rule:str,
  thinning_rule:str,confirmation_rule:str,cancel_reason_precedence:list[str]
}

contracts/fixed_epoch_contract.json = object{
  schema_version:int,epoch_origin_ns:int,epoch_width_ns:int,
  checkpoint_ns:int,expected_checkpoint_count:int,core_open_offset_ns:int,
  core_close_offset_ns:int,core_half_open:bool,
  confirmation_close_equality_admitted:bool,thinning_key:list[str],
  tie_break:list[str],cluster_key:list[str]
}

contracts/outcome_access_ledger.json = object{
  schema_version:int,future_target_accessed:bool,
  future_price_accessed:bool,fill_fee_pnl_accessed:bool,
  consumed_cache_fields:list[str],
  cache_count:int,unconsumed_field_count:int,
  nonempty_unconsumed_field_instance_count:int,
  changed_unconsumed_field_instance_count:int,
  consumed_field_mismatch_count:int,poison_attestation_sha256:sha256,
  raw_a_p_difference_count:int,outcome_boundary_preserved:bool
}

contracts/gate_contract.json = object{
  schema_version:int,gate_order:list[str],gates:list[Gate],
  first_failed_gate_id:nullable[str],classification:str
}

reports/A_minus1_summary.json = object{
  schema_version:int,task_id:str,hypothesis_id:str,audit_id:str,
  idea_sha256:sha256,plan_sha256:sha256,implementation_head:sha1,
  classification:str,primary_variant:str,sensitivity_variants:list[str],
  variant_rows:list[VariantRow],integrity:Integrity,gates:list[Gate],
  future_outcomes_authorized:bool,a0_authorized:bool,
  live_trading_authorized:bool
}

classification.json = object{
  schema_version:int,task_id:str,hypothesis_id:str,audit_id:str,
  classification:str,first_failed_gate_id:nullable[str],
  gate_statuses:list[str],future_outcomes_authorized:bool,
  a0_authorized:bool,live_trading_authorized:bool
}

contracts/execution_evidence.json = object{
  schema_version:int,attempt_id:str,implementation_head:sha1,
  raw_a_b:Comparison,raw_a_p:Comparison,
  sealed_a_b:Comparison,sealed_a_p:Comparison
}

run_manifest.json = object{
  schema_version:int,artifact_count:int,
  artifacts:list[ManifestRow]
}
```

The summary's `variant_rows` are exact JSON projections of
`variant_summary.csv`, using `null` for empty share. Its `integrity` object has
the exact `Integrity` schema above.

Gate enums are exact:

```text
gate_id in {"A-1-0","A-1-1","A-1-2","A-1-3"}
status in {"PASS","FAIL","NOT_EVALUATED"}
condition status in {"PASS","FAIL","NOT_EVALUATED"}
```

Exact sibling schemas:

```text
.workflow/attempt-claims/0830T002.{armed,claimed}.json = object{
  schema_version:int,task_id:str,attempt_id:str,implementation_tag:str,
  formal_argv:list[str],repo_root:str,source_cache_root:str,attempt_root:str,
  idea_sha256:sha256,plan_sha256:sha256,task_sha256:sha256,
  runner_sha256:sha256,verifier_sha256:sha256,tests_sha256:sha256,
  controller_remote:str,controller_url:str,controller_ref:str,status:str
}
```

Armed and claimed bytes are identical and `status="ARMED_FOR_SINGLE_USE"`.

```text
attempt-lock.json = object{
  schema_version:int,task_id:str,attempt_id:str,status:str,pid:int,
  started_at_utc:str,cwd:str,argv:list[str],implementation_head:sha1,
  consumption_head:sha1,claimed_sha256:sha256,repo_root:str,
  source_cache_root:str,attempt_root:str,controller_remote:str,
  controller_ref:str,controller_consumption_head:sha1,
  remote_observations:list[RemoteObservation],
  remote_transitions:list[RemoteTransition],
  push_calls:list[PushCall],successful_push_count:int
}
```

`status="CLAIMED_BEFORE_CACHE_READ"`.

The direct poison authority first writes its derived helper path, direct
verification reads that path, and the verified bytes are then renamed without
content change to sibling `poison-attestation.json`. Its exact schema is:

```text
poison-attestation.json = object{
  task_id:str,hypothesis_id:str,poison_output_root:str,
  source_inventory_sha256:sha256,cache_count:int,
  unconsumed_fields:list[str],unconsumed_field_count:int,
  nonempty_unconsumed_field_instance_count:int,
  changed_unconsumed_field_instance_count:int,
  consumed_field_mismatch_count:int,caches:list[PoisonCache]
}

PoisonCache = object{
  cache_name:str,unconsumed_fields:list[PoisonField]
}

PoisonField = object{
  field:str,dtype:str,shape:list[int],
  source_value_sha256:sha256,poison_value_sha256:sha256
}
```

The frozen authority values inside the attestation are:

```text
task_id = "0829T003"
hypothesis_id = "FIXED_CAUSAL_EPOCH_MSTATE_V2"
```

```text
attempt-result.json = object{
  schema_version:int,task_id:str,attempt_id:str,status:str,
  phase:str,exit_code:int,finished_at_utc:str,consumption_head:sha1,
  controller_ref:str,
  attempt_lock_sha256:sha256,claimed_sha256:sha256,
  poison_attestation_sha256:sha256,instrumentation_evidence_sha256:sha256,
  work_manifest_sha256:sha256,
  work_tree_sha256:sha256,
  final_a_b:Comparison,final_a_p:Comparison,
  root_rows:list[RootRow]
}
```

```text
work-manifest.json = object{
  schema_version:int,attempt_id:str,row_count:int,
  per_build_slice_count:int,rows:list[WorkRow],tree_sha256:sha256
}
```

```text
instrumentation-evidence.json = object{
  schema_version:int,attempt_id:str,status:str,
  successor_np_load_callsite_count:int,feature_calls:list[FeatureCall],
  field_accesses:list[FieldAccess],raw_open_events:list[RawOpenEvent],
  loader_boundary_violation_count:int,detector_boundary_violation_count:int,
  feature_mutation_violation_count:int,inherited_fd_violation_count:int,
  ipc_envelope_violation_count:int,raw_reference_cross_boundary_count:int,
  raw_buffer_cross_boundary_count:int,loader_process_count:int,
  detector_process_count:int
}
```

```text
.workflow/attempt-receipts/0830T002.terminal.json = object{
  schema_version:int,task_id:str,attempt_id:str,status:str,
  implementation_head:sha1,consumption_head:sha1,
  controller_remote:str,controller_ref:str,
  attempt_result_sha256:sha256,attempt_lock_sha256:sha256,
  poison_attestation_sha256:sha256,
  instrumentation_evidence_sha256:sha256,
  work_manifest_sha256:sha256,
  work_tree_sha256:sha256,root_rows:list[RootRow],
  sealed_at_utc:str
}
```

```text
.workflow/reports/0830T002-terminal-verifier.json = object{
  schema_version:int,task_id:str,attempt_id:str,status:str,
  verifier_sha256:sha256,verifier_git_blob_oid:sha1,
  implementation_head:sha1,consumption_head:sha1,terminal_head:sha1,
  controller_remote:str,controller_url:str,controller_ref:str,
  observed_remote_head:sha1,
  remote_observations:list[RemoteObservation],
  remote_transitions:list[RemoteTransition],
  push_calls:list[PushCall],successful_push_count:int,
  checked_repo_root:str,checked_attempt_root:str,
  first_failure_code:nullable[str],checks:list[VerifierCheck],
  result_created_at_utc:str
}
```

Verifier `status` is `PASS` or `FAIL`. Exact check order and first-failure
codes:

```text
V00_CLI_AND_ROOTS
V01_VERIFIER_IDENTITY
V02_GIT_TRANSITIONS_AND_FSYNC_CONFIG
V03_CLAIM_AND_LOCK
V04_EXACT_ATTEMPT_CHILDREN
V05_WORK_MANIFEST_AND_FEATURE_INPUTS
V06_POISON_ATTESTATION
V07_FINAL17_SCHEMAS_AND_PATHS
V08_MANIFESTS_AND_TREE_HASHES
V09_COMPARISON_CLOSURE
V10_ATTEMPT_RESULT
V11_TERMINAL_RECEIPT_AND_TAG
V12_POST_SEAL_DRIFT
```

Checks after the first failure have `status="NOT_EVALUATED"`,
`actual=""`, and retain their exact required string. There are always 13
unique check rows for exit 0 or 2.

For all 13 rows, `required` is exactly `"true"`. Evaluated encodings are:

```text
PASS row:
  status = "PASS"
  actual = "true"

first FAIL row:
  status = "FAIL"
  actual = "false:<check_id>"

later row:
  status = "NOT_EVALUATED"
  actual = ""
```

Global equalities:

```text
exit 0
iff status="PASS"
and first_failure_code=null
and all 13 rows are PASS

exit 2
iff status="FAIL"
and first_failure_code equals the first FAIL row's check_id
and that row actual equals "false:<check_id>"
and every later row is NOT_EVALUATED
```

`check_id` itself is the failure code; no alternate code vocabulary exists.

For a completed one-shot sequence, `status="COMPLETED"`,
`phase="FINAL_17_CLOSED"` and `exit_code=0`. If the process dies before this
receipt, the precommitted claim state itself proves
`INTERRUPTED_TERMINAL`; no synthetic result file is added later.

Missing/unknown keys, wrong types, wrong enum values, duplicate paths,
unsorted rows or malformed hashes are schema violations.

All path rows sort by path ASCII. A tree hash is SHA256 of the canonical JSON
array of `ManifestRow` records sorted by path. `Comparison.rows` is the sorted
union of both path sets; a missing side uses null SHA and `equal=false`.

The only exception is `work_tree_sha256`: its preimage is the full
`WorkRow` array, not a `ManifestRow` projection, sorted by build order A/B/P,
cache name ASCII and slice ordinal numeric, serialized with
`sort_keys=true,separators=(",",":"),ensure_ascii=true`. Duplicate path or
duplicate `(build_label,cache_name,slice_ordinal)` is rejected before hashing.

Exact list domains:

```text
tracked_files count/order:
  1 docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_research_idea_20260830.md
  2 docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_execution_plan_20260830.md
  3 .workflow/tasks/0830T002.md
  4 examples/hyperliquid/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py
  5 examples/hyperliquid/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py
  6 examples/hyperliquid/test_skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py
  7 .workflow/attempt-claims/0830T002.claimed.json

callables count/order:
  1 build_features
    path examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py
  2 source_preflight
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  3 base_eligibility
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  4 channel_actions
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  5 channel_memories
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  6 epoch_support_ledger
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  7 materialize_poisoned_cache_set
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
  8 verify_poison_attestation
    path examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py

direct call counts:
  build_features = analyzed_unit_count
  source_preflight = analyzed_unit_count
  base_eligibility = analyzed_unit_count
  channel_actions = analyzed_unit_count
  channel_memories = analyzed_unit_count
  epoch_support_ledger = analyzed_unit_count
  materialize_poisoned_cache_set = 1
  verify_poison_attestation = 1

feature_calls:
  count = analyzed_unit_count
  order = build A, build B, build P;
          then cache_name ASCII;
          FULL before SLICE;
          slice_ordinal numeric

field_accesses:
  exactly 12 rows per feature call
  field order ASCII over:
    activity,ask_depletion,bid_depletion,event_seq,ofi,ofi_abs,
    ready,segment_id,trade_signed,trade_total,ts_ns,valid_book
  value_access_count = 1
  authorization = "CONSUMED_VALUE"

outcome_access_ledger.consumed_cache_fields count/order:
  12, ASCII order:
    activity,ask_depletion,bid_depletion,event_seq,ofi,ofi_abs,
    ready,segment_id,trade_signed,trade_total,ts_ns,valid_book

FeatureCall enums:
  build_label in {"A","B","P"}
  unit_kind in {"FULL","SLICE"}
  input_authority in {"CANONICAL","POISON","SLICED_CANONICAL",
                      "SLICED_POISON"}
  field_name_schema_access_count = 1
  consumed_value_access_count = 12
  forbidden_value_access_count = 0
  consumer_use_count = 1
  feature_output_sha256 = consumer_input_sha256 = detector_exit_sha256

instrumentation evidence:
  status = "PASS"
  successor_np_load_callsite_count = 0
  all seven violation counts = 0
  loader_process_count = analyzed_unit_count
  detector_process_count = analyzed_unit_count

RawOpenEvent:
  event_index is global zero-based contiguous after canonical sorting
  call_index references one FeatureCall
  build_label in {"A","B","P"}
  phase order/enums = {"HASHER","SLICE_MATERIALIZER","LOADER"}
  event_type in {"open","mmap.__new__"}
  operation in {"READ_INPUT","WRITE_SLICE"}
  caller_path is repo-relative authority source path
  caller_name is the authority-root function, not immediate library frame
  allowed = true for every successful row
  sort = call_index, phase order, event_type ASCII, resolved_path ASCII,
         operation ASCII, caller_path ASCII, caller_name ASCII
  duplicate full rows forbidden

IPC envelope exact canonical JSON:
  header object{
    schema_version=1,
    call_index,
    arrays=list[IPCArrayRow],
    payload_size_bytes,
    payload_sha256,
    field_access_sha256,
    feature_key_count
  }
  array order = feature name ASCII
  payload = exact concatenation of each C-contiguous array bytes
  offsets start at 0 and are contiguous with no gap/overlap/trailing bytes
  payload_size_bytes = sum(length_bytes) = len(payload)
  payload_sha256 = SHA256(payload)
  header serialization:
    sort_keys=true,separators=(",",":"),ensure_ascii=true
  frame =
    uint64 big-endian header-byte length
    + header bytes
    + payload bytes
  loader sends exactly one frame with send_bytes
  detector receives exactly one frame, validates every table/hash/length,
  then requires EOF and no second frame/unused byte

field_access_sha256:
  preimage = exactly the 12 FieldAccess rows for this call_index
  order = field ASCII
  projection = full FieldAccess objects
  duplicates or foreign call_index rejected
  serialization sort_keys=true,separators=(",",":"),ensure_ascii=true

feature_key_count:
  equals len(header.arrays)
  equals loader feature dictionary key count
  equals canonical feature-hash row count
  equals detector reconstructed dictionary key count

IPCArrayRow exact arithmetic:
  dtype_str = array.dtype.str
  shape is the exact list of non-negative integer dimensions
  bool is forbidden for every integer field
  itemsize = numpy.dtype(dtype_str).itemsize
  length_bytes = product(shape) * itemsize
  zero-dimensional shape [] has product 1
  any zero dimension gives length_bytes 0
  first offset_bytes = 0
  each next offset = prior offset + prior length
  final offset + length = payload_size_bytes
  value_sha256 = SHA256(payload[offset:offset+length])
  projection (name,dtype_str,shape,value_sha256) equals the canonical
  feature-hash row exactly

Endpoint evidence:
  sender and receiver independently hash their local bytes
  sender.sent_frame_count = 1
  sender.send_end_closed = true before receiver EOF check
  receiver.received_frame_count = 1
  receiver.eof_observed = true
  receiver.unused_byte_count = 0
  sender header/payload/frame hashes and sizes equal receiver values
  sender/receiver payload SHA equals header.payload_sha256
  sender/receiver frame SHA equals SHA256(the exact frame)

RawOpenEvent phase authority matrix:
  HASHER:
    operation READ_INPUT
    resolved_path = FeatureCall.resolved_input_path
    caller_name = sha256_file
  SLICE_MATERIALIZER:
    READ_INPUT path = registered full authority input
    WRITE_SLICE path = registered WorkRow path
    caller_name = materialize_slice
  LOADER:
    operation READ_INPUT
    resolved_path = FeatureCall.resolved_input_path
    caller_name = build_features
  DETECTOR:
    no row permitted

Python native audit normalization:
  native "open" from builtins.open, io.open or os.open
    -> event_type "open"
  native "mmap.__new__"
    -> event_type "mmap.__new__"
  canonical event_type "os.open" is removed and forbidden

Comparison.domain exact values:
  "RAW_11:A_vs_B"
  "RAW_11:A_vs_P"
  "SEALED_15:A_vs_B"
  "SEALED_15:A_vs_P"
  "FINAL_17:A_vs_B"
  "FINAL_17:A_vs_P"

root_rows:
  count/order/labels = 3, ["A","B","P"]

work rows:
  count = 3 * per_build_slice_count
  order = build label A/B/P, cache_name ASCII, slice_ordinal numeric
  slice_ordinal starts at 0 and is contiguous within each build/cache
  every path is relative to attempt_root and starts with "work/"
  no duplicate path or (build_label,cache_name,slice_ordinal)

classification.gate_statuses:
  count/order = 4 in ["A-1-0","A-1-1","A-1-2","A-1-3"] order
```

All lists reject duplicates. The tracked terminal receipt closes
`attempt-result` itself but also precedes terminal push and therefore excludes
`push-ledger/001-terminal.json`. The post-terminal verifier exclusively closes
that second push receipt, online terminal head and completed exact attempt
child set. It never rewrites attempt-result or terminal receipt.

All UTC strings use exactly:

```text
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

## 14. Numeric Semantics

All counts are non-negative integers, not booleans.

All shares are finite in `[0,1]`.

```text
cluster_count = 0:
  maximum_single_date_share = null in JSON / empty in CSV
  support prediction = false

cluster_count > 0:
  share = maximum per-date distinct clusters / total distinct clusters
```

Any negative, non-finite, wrong type, inconsistent share, failed conservation
or malformed scientific value in the A-1-2 domain is A-1-2.

First-failure ownership is exact:

```text
A-1-0:
  authority_binding.json semantic/schema defects
  source_cache_inventory.csv authority/source defects
  canonical A/B RAW_11 mismatch

A-1-1:
  outcome_access_ledger.json exact future/outcome flags, poison identity
  and A/P comparison fields only
  poison attestation defects
  A/P RAW_11 mismatch

A-1-2:
  detector_contract.json
  fixed_epoch_contract.json
  channel_action_by_date.csv
  epoch_support.csv
  epoch_variant_counters.csv
  slice_invariance.csv
  support_by_date.csv
  trigger_ledger.csv
  variant_summary.csv
  in-memory scientific payload

terminal verifier only, scientific classification unchanged:
  feature-call/field-access/boundary defects
  instrumentation-evidence.json semantic/schema defects
  gate_contract.json
  A_minus1_summary.json
  classification.json
  execution_evidence.json
  run_manifest.json
  attempt-lock.json
  attempt-result.json
  terminal receipt
  post-gate sorting/hash/path/tree/commit/tag defects
```

`schema_violation_count` in A-1-2 covers only its listed A-1-2 domain.
Terminal-verifier failure is execution-package rejection; it never rewrites a
scientific classification after observation.

## 15. Sequential Gates and Classification

### A-1-0 Authority and Source

Conditions:

- baseline verifier;
- frozen idea/plan/task/successor identities;
- direct-call authority bindings;
- clean one-shot attempt receipt;
- exact 29-cache source closure;
- source preflight violations zero;
- canonical A/B `RAW_11` difference zero.

Failure:

```text
Aminus1_authority_or_source_failed
```

### A-1-1 Outcome Boundary

Conditions:

- poison 29/15/435 identities exact;
- consumed mismatch zero;
- A/P `RAW_11` difference zero.

Failure:

```text
Aminus1_outcome_boundary_violated
```

### A-1-2 Detector Integrity

Conditions:

- all channel/action/memory invariants;
- all three variants' conservation;
- all fixed-epoch and identity invariants;
- all slice exact flags;
- cross-segment checkpoint count zero;
- represented slice dates >=4;
- distinct comparable epochs >=30;
- positive compared support count;
- numeric/schema violations zero.

Any sensitivity integrity defect fails this gate.

Failure:

```text
Aminus1_detector_integrity_failed
```

### A-1-3 Primary Structural Support

Conditions in order:

```text
TRADE_LED cluster count >=30
TRADE_LED represented dates >=4
TRADE_LED maximum single-date share <=0.50
```

Within A-1-3, evaluation is sequential. If cluster count fails, date and share
rows are `NOT_EVALUATED` with `passed=null`, `actual=null`. If count passes but
date fails, only share is `NOT_EVALUATED`. A zero-cluster share remains null
and is never compared.

If count or date coverage fails:

```text
Aminus1_trade_led_structural_support_not_estimable
```

If count/date pass but concentration fails:

```text
Aminus1_trade_led_structure_date_concentrated
```

If all pass:

```text
Aminus1_trade_led_recurrent_structural_candidate
```

Sensitivities are absent from A-1-3.

Every gate after the first failed gate is retained as exact
`NOT_EVALUATED`.

Exact condition IDs, order and required strings:

```text
A-1-0:
  baseline_authority_verified             required "true"
  frozen_successor_identities_verified    required "true"
  direct_callable_bindings_verified       required "true"
  claim_and_lock_valid_before_cache       required "true"
  canonical_source_closure_exact          required "29 exact caches"
  source_preflight_violation_count        required "0"
  raw_a_b_difference_count                required "0"

A-1-1:
  poison_cache_count                      required "29"
  poison_unconsumed_field_count           required "15"
  poison_changed_field_instance_count     required "435"
  poison_consumed_field_mismatch_count    required "0"
  raw_a_p_difference_count                required "0"

A-1-2:
  action_partition_violation_count        required "0"
  unauthorized_ttl_refresh_count          required "0"
  cross_segment_memory_carry_count        required "0"
  conservation_violation_count            required "0"
  fixed_epoch_violation_count             required "0"
  slice_mismatch_count                    required "0"
  cross_segment_compared_checkpoint_count required "0"
  represented_slice_date_count            required ">=4"
  distinct_comparable_epoch_count         required ">=30"
  compared_support_checkpoint_count       required ">0"
  schema_violation_count                  required "0"
  numeric_violation_count                 required "0"

A-1-3:
  trade_led_confirmed_cluster_count        required ">=30"
  trade_led_represented_date_count         required ">=4"
  trade_led_maximum_single_date_share      required "<=0.50"
```

For evaluated rows, `actual` is the exact bool/int/finite number and status is
`PASS` or `FAIL`. For any condition after an earlier failed condition inside
the same gate, and for every later gate after the first failed gate:

```text
status = "NOT_EVALUATED"
passed = null
actual = null
required = the frozen string above
```

## 16. Frozen Hostile-Test Minimum

At minimum:

- exact +/-0.50 and +/-0.25 inclusivity;
- TTL age 100ms fresh and 120ms stale;
- six-point prestate excludes `t`;
- core open included, core close excluded for trigger;
- confirmation close equality admitted and +20ms classified exactly as
  `confirmation_edge_omitted` before veto/thinning;
- same-checkpoint neutral clears old opposite before veto;
- same-checkpoint new opposite is visible to veto;
- an earlier vetoed onset does not occupy a thinning key;
- reset clears memory and invalidates epoch before trigger;
- raw/veto/admitted/retained/suppressed/confirmed conservation;
- earliest retained failure suppresses a later confirmable trigger;
- opposite directions and variants share epoch cluster;
- no later same-key replacement;
- first additional update and window-close timestamps differ correctly;
- confirmation waits until window close;
- independent cancellation booleans and reason precedence;
- slice epoch/counter/retained/status/support hash mutations fail;
- slice cannot reuse derived full features;
- full/slice/poison direct-call `build_features` identity and call counts;
- feature-call wrong root, discarded return, canonical input for P, full
  feature reuse for slice and cross-build memoization are terminal failures;
- exact-boundary, boundary-20ms, equality and boundary+20ms integer ceiling;
- source invalid fails before action;
- attempted unconsumed/alternate-loader reads, alias import, proxy early
  restore and post-entry array mutation are terminal failures;
- every unconsumed poison value changes and consumed values do not;
- A/B missing, extra and byte mutations fail A-1-0;
- A/P missing and extra paths produce A-1-1 negative evidence;
- A/P byte mutations outside the two registered normalized fields produce
  A-1-1 negative evidence;
- A/P `slice_invariance.csv` header/row/order/non-source-field mutations fail,
  while a same-build-valid `slice_source_sha256` physical identity difference
  alone is normalized;
- A/P `run_manifest.json` structure/order/non-slice-artifact mutations fail,
  while only the slice artifact SHA derived from the registered normalized
  slice comparison is normalized;
- attestation mutation fails;
- all 17 schemas, typed sentinels, sorting and manifest self-exclusion;
- zero, negative, NaN, infinity and wrong-type gate mutations;
- later `NOT_EVALUATED` rows preserve required values;
- A-1-3 zero/count/date/concentration short-circuit bytes;
- sensitivity cannot rescue primary;
- dirty worktree, wrong tag/HEAD, wrong CLI/root, symlink, armed/claimed
  mutation, existing attempt root and successor identity mutations fail
  before cache read;
- claim hard-link/unlink/commit/tag/fsync order and interrupted
  non-replacement;
- restore-armed plus delete-claimed/root still fails because consumption tag
  and transition commit remain;
- delete tags + reset branch + delete root still fails via reflog/unreachable
  transition scan; simulated Git object/ref loss fails `git fsck`;
- local refs/reflogs/object store deletion still cannot erase the external
  origin ledger; deliberate GitHub-admin ledger tampering is outside the
  registered threat model and invalidates the study;
- missing/mutated/extra work input, work-manifest mismatch and FeatureCall
  input-SHA/path-authority mismatch fail terminal verification;
- FINAL_17/attestation/result no-replace publication, directory fsync,
  terminal receipt commit/tag and post-seal tree-drift detection;
- RAW_11/SEALED_15/FINAL_17 projection and self-reference exclusions;
- exact JSON, sibling and authority poison schemas.
- exact UTC microsecond-Z timestamps, A/B/P root-row order and `-1`
  maximum-memory-age sentinel.
- verifier wrong CLI/root/tag, verifier self-mutation, missing work evidence,
  post-seal mutation, exact exit code and 13-row result schema.
- no pre-terminal artifact contains terminal commit/ref SHA; self-reference
  graph mutation fails plan/readiness review.
- inherited FD injection, DETECTOR raw path/env/cwd reconstruction, LOADER IPC
  extra field/raw-byte smuggling, RawOpenEvent drop/reorder/caller spoof,
  HASHER/SLICE authority mutation and instrumentation sibling
  missing/extra/mutation/no-replace failures.
- sender/receiver header, payload, frame SHA/size mismatch; zero/two frames;
  missing EOF; nonzero unused bytes; open inherited send-end; second frame;
  trailing bytes and receiver-side transcript mutation.
- IPC dtype spelling, negative shape, shape/itemsize/length mismatch, offset
  gap/overlap/overflow, zero-dimensional/zero-length arithmetic, payload-slice
  SHA and canonical feature-row projection mutations.
- `ls-remote` nonzero exit with empty stdout, stderr mutation, pre-existing
  equal ref, wrong URL/refspec, wrong pre/post full SHA, transition
  derived-from mutation and abbreviated SHA substitution.
- failed push then retry, up-to-date repush, duplicate ordinal, missing push
  receipt, third push file, unledgered direct push and push wrapper bypass.

## 17. Pre-Execution Locks

Before formal execution:

```text
independent idea/plan review = 0/0/0/0
idea and plan SHA frozen in task
runner/tests implementation commit created
task updated with implementation commit and runner/tests SHA/blob, then committed
armed claim created from the committed task and exact formal identities
armed claim committed without further task/runner/tests changes
implementation tag attached to that final clean commit
independent verifier SHA/blob frozen in task and armed claim
independent readiness review confirms command/receipt/test contract
focused and inherited tests pass
```

Until then, 29-cache execution is locked.

## 18. Post-Claim Rule

Consumption of the tracked armed claim is the start of formal execution and
occurs before attempt-root creation and before any cache read.

After it exists, only the already registered one-shot sequence may continue.
No repair, diagnosis, replacement attempt, code/plan/test change or
alternative execution is permitted.

If the result contradicts the prediction, record it and stop.
