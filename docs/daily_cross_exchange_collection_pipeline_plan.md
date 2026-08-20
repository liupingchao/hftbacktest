# Daily Cross-Exchange Collection Pipeline Plan

Date: 2026-08-19

Status: revised design proposal

Scope: scheduled public-data collection for a configured Binance/Hyperliquid
symbol profile, remote artifact pullback, local preprocessing, and durable
logs. The existing `cross-exchange-postprocess` Skill and its deterministic
CLI are the standard preprocessing engine for this pipeline.

## 1. Objective

Build a deterministic daily pipeline that can run without an interactive
terminal and produce one auditable data package per run:

1. collect the configured symbol from Binance and Hyperliquid;
2. validate the remote campaign before transfer;
3. pull the completed campaign to the local repository host;
4. verify all transferred files against the remote manifests;
5. run the supervisor's local `--postprocess-only` phase to create accepted
   common L2 timelines in a working copy;
6. invoke the existing `cross-exchange-postprocess` pipeline for `dataset` or
   `basis-research` output;
7. record structured logs, status, hashes, and failure reasons.

The initial production profile is:

- Binance USD-M: `SKHYNIXUSDT`
- Hyperliquid: `xyz:SKHX`
- profile ID: `skhynix`
- default duration: `14400` seconds
- default mode: one continuous segment

The profile, duration, schedule, collector host, and preprocessing stages must
remain configuration-driven so that the same pipeline can run other profiles
registered in `cross_exchange_symbol_registry.py`.

This pipeline is public market-data collection only. It must not access private
account endpoints, place orders, run a trading strategy, or promote a strategy
to live use.

## 2. Design Principles

### 2.1 CLI is the execution contract

The core pipeline must be a normal CLI that can be invoked from a shell,
systemd, CI, or a Codex skill. The CLI must not depend on an LLM,
interactive approval, terminal state, or an SSH session remaining open.

### 2.2 systemd schedules; it does not contain business logic

The systemd timer starts one pipeline service. Collection, transfer,
preprocessing, validation, and reporting stay in versioned repository code.

### 2.3 Raw data is immutable

The original remote campaign is copied without mutation. Preprocessing writes
separate derived directories. A failed preprocessing step must never rewrite
or delete accepted raw data.

### 2.4 Every stage is independently observable

Collection, transfer, timeline prerequisite, and unified postprocess must have
separate status, exit code, start/end time, log, and artifact references. The
postprocess stage must also record the selected Skill profile and its output
manifest.

### 2.5 Publication is atomic

Partial remote transfers and incomplete preprocessing outputs must not appear
under the directory used by downstream readers. Write to a staging directory,
verify it, then atomically rename it to the final run directory.

### 2.6 The Skill is the preprocessing contract

The repository Skill at
`.agents/skills/cross-exchange-postprocess/SKILL.md` defines how an accepted
campaign becomes an auditable R0/R1 or basis-research dataset. The daily
pipeline must call its deterministic module entry point:

```text
python -m examples.hyperliquid.cross_exchange_postprocess ...
```

The Skill text is useful for interactive operation and result explanation, but
systemd must not depend on an LLM or a Skill session remaining active.

### 2.7 No duplicate research logic

The daily orchestrator owns scheduling, collection, transfer, stage state and
logs. It must not reimplement replay, alignment, masks, provenance closure,
basis/dislocation features, or research eligibility checks already provided by
the postprocess pipeline.

## 3. Existing Components To Reuse

The pipeline should wrap the existing components rather than duplicate their
collection or research logic:

- `examples/hyperliquid/synchronized_public_collection.py`
  - public Binance and Hyperliquid collection entry point;
  - profile-based symbol mapping;
  - raw manifests and runtime source metadata.
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
  - profile supervision;
  - continuous or segmented collection;
  - collection-only and postprocess-only modes;
  - heartbeat, supervisor events, and campaign manifest.
- `examples/hyperliquid/cross_exchange_symbol_registry.py`
  - canonical Binance symbol and Hyperliquid coin mapping.
- `examples/hyperliquid/cross_exchange_l2_timeline.py`
  - common local-receipt-time L2 timeline;
  - source hashes, segment boundaries, reconnect epochs, and timeline manifest.
- `examples/hyperliquid/cross_exchange_research_dataset.py`
  - fail-closed R0 event-store construction and source provenance.
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
  - called by the unified postprocess pipeline, not directly by the daily
    orchestrator.
- `.agents/skills/cross-exchange-postprocess/SKILL.md`
  - operator contract for inspect, run, resume, validate, and report;
  - explicit research boundaries and capability language.
- `examples/hyperliquid/cross_exchange_postprocess/`
  - deterministic implementation used by both the Skill and the daily CLI;
  - `dataset` profile for raw audit/R0/R1;
  - `basis-research` profile for R0/R1 plus point-in-time basis state.

Before implementation, the current supervisor and timeline changes must be
committed and their runtime hashes frozen. The runtime source archive recorded
inside every campaign must remain the source of truth for that run. The daily
CLI must record both the collection runtime source and the postprocess runtime
source/hashes.

## 4. Proposed CLI

Add a pipeline-level CLI, preferably under:

```text
examples/hyperliquid/daily_cross_exchange_pipeline.py
```

The implementation may later move to a dedicated `scripts/` or package
directory, but the command contract should remain stable.

### 4.1 Full run

```bash
python examples/hyperliquid/daily_cross_exchange_pipeline.py run \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix \
  --stages collect,pull,timeline,postprocess,report
```

The `run` command performs all requested stages in order and exits non-zero
if a required stage fails.

### 4.2 Individual stages

```bash
python examples/hyperliquid/daily_cross_exchange_pipeline.py preflight \
  --config configs/daily_cross_exchange/skhynix.json

python examples/hyperliquid/daily_cross_exchange_pipeline.py collect \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py pull \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py timeline \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py postprocess \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py postprocess-inspect \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py postprocess-validate \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py verify \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py status \
  --run-id 20260819-skhynix
```

Stage commands must be idempotent. Re-running a successful stage should
verify and reuse its existing artifact, not silently overwrite it. A `--force`
option may be added later for explicit rebuilds into a new attempt directory.

### 4.3 Recovery commands

```bash
python examples/hyperliquid/daily_cross_exchange_pipeline.py retry-pull \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py retry-timeline \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py retry-postprocess \
  --run-id 20260819-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py report \
  --run-id 20260819-skhynix
```

Recovery must not start a second remote collection when a valid completed
campaign already exists.

## 5. Configuration

Use a versioned JSON configuration to avoid adding a new runtime dependency.
Example:

```json
{
  "schema_version": "daily_cross_exchange_config_v1",
  "profile_id": "skhynix",
  "duration_seconds": 14400,
  "continuous_collection": true,
  "collector": {
    "host": "collector-host",
    "remote_root": "/var/lib/hftbacktest/daily_cross_exchange",
    "python_executable": "/opt/hftbacktest/bin/python",
    "ssh_connect_timeout_seconds": 15
  },
  "local": {
    "root": "local_live_analysis/daily_cross_exchange",
    "staging_root": "/tmp/hftbacktest-daily-cross-exchange"
  },
  "postprocess": {
    "enabled": true,
    "profile": "basis-research",
    "python_executable": "/opt/hftbacktest/bin/python",
    "resume_on_retry": true,
    "task_id_prefix": "daily"
  },
  "quality": {
    "require_network_collection_complete": true,
    "require_raw_sha256": true,
    "require_raw_row_reconciliation": true,
    "allow_recovered_binance_reconnects": false,
    "allow_recovered_core_l2_reconnects": false
  },
  "retention": {
    "keep_days": 30,
    "keep_failed_runs": true
  }
}
```

The configuration must not contain private keys, API secrets, or credentials.
SSH authentication must come from the host's restricted service account or
agent environment.

## 6. Run Identity And Directory Layout

Every run gets a stable UTC-based ID. The date in the ID is the scheduled
collection date, not the local machine's display date.

Recommended ID:

```text
YYYYMMDD-<profile_id>
```

If more than one attempt is needed, preserve the original run ID and append:

```text
YYYYMMDD-<profile_id>-attempt-02
```

### 6.1 Remote layout

```text
/var/lib/hftbacktest/daily_cross_exchange/<profile_id>/<run_id>/
  campaign/
    campaign_manifest.json
    run_status.json
    heartbeat.json
    supervisor_events.jsonl
    runtime_source.json
    runtime_source_archive/
    segments/
  logs/
    pipeline.log
    collection.log
  status/
    pipeline_status.json
```

The remote campaign must retain the existing campaign layout so the existing
supervisor and preprocessing tools can consume it.

### 6.2 Local layout

```text
local_live_analysis/daily_cross_exchange/<profile_id>/<run_id>/
  raw_campaign/                 # immutable transfer result
  campaign_working_copy/        # timeline postprocess-only input/output
  preprocess/
    postprocess_output/          # Skill/CLI output and its reports
  logs/
    pipeline.jsonl
    pull.log
    preprocess.log
  manifests/
    remote_campaign_manifest.json
    local_transfer_manifest.json
    timeline_manifest.json
    postprocess_manifest.json
  run_status.json
  run_report.md
```

The `raw_campaign/` directory must remain byte-preserving. The supervisor
`--postprocess-only` phase operates on `campaign_working_copy/`, never on the
raw source. The `cross-exchange-postprocess` output is a separate derived
directory. The `latest` pointer, if introduced, must point only to a fully
verified raw-plus-derived package and must never point to a staging directory.

## 7. Pipeline Stages

### 7.1 Preflight

Before starting collection:

1. validate the configuration schema;
2. resolve the profile through the symbol registry;
3. verify the collector host and Python executable;
4. check remote disk space and local disk space;
5. check that no run lock is active for the same profile;
6. check that the source files are clean or that their expected commit/hash is
   explicitly configured;
7. create the run status file and lock;
8. verify that the target run ID does not already contain a conflicting
   campaign.

A preflight failure must not start a collector.

### 7.2 Remote collection

The pipeline starts the existing supervisor in a detached remote service
context, not as a child of the controlling SSH session.

The first implementation may use a remote systemd transient unit:

```text
systemd-run --unit=hftbacktest-daily-<run-id> \
  --collect --property=...
```

The exact remote launch command must be written into the run manifest with
credentials and secret values redacted.

Required collection properties:

- public-only Binance and Hyperliquid endpoints;
- configured profile and symbol identity;
- configured duration;
- explicit continuous or segmented mode;
- `--collection-only` when preprocessing is intentionally local;
- heartbeat and supervisor event logging;
- runtime source archive and SHA-256 recording.

The collection stage is successful only when the remote campaign manifest
proves:

- `state=complete`;
- `network_collection_complete=true`;
- `passes=true`;
- requested and actual duration satisfy the configured ratio;
- all required channels and subscription acknowledgements are present;
- raw row counts reconcile;
- every raw file has a valid SHA-256;
- reconnect policy is satisfied;
- both venue identities match the configured profile.

### 7.3 Pullback

Pullback starts only after remote collection is complete and accepted by the
remote quality gate.

Recommended sequence:

1. create a local staging directory;
2. copy the remote campaign and runtime source archive;
3. use resumable transfer for large files;
4. copy the remote manifest and status files again after data transfer;
5. recompute local SHA-256 for every manifest-listed raw file;
6. compare file count, byte count, hashes, segment IDs, and profile identity;
7. write `local_transfer_manifest.json`;
8. atomically rename the verified staging directory to `raw_campaign/`.

A transfer with a missing file, hash mismatch, or incomplete remote status is
failed and quarantined. It must not update `latest`.

### 7.4 Timeline prerequisite through supervisor postprocess-only

The current postprocess Skill requires an accepted campaign with a common L2
timeline. Therefore the daily pipeline has one prerequisite stage before
calling the Skill:

1. copy the verified `raw_campaign/` into a new
   `campaign_working_copy/` staging directory;
2. invoke `cross_exchange_collection_supervisor.py` with
   `--postprocess-only`, the configured profile and the original campaign
   identity;
3. do not pass `--clean-output` and do not spawn collectors;
4. preserve the supervisor's `postprocess_history/`, runtime source archive,
   heartbeat, events, and terminal manifest;
5. verify every segment's `common_l2_timeline.csv.gz`, timeline manifest and
   top-level `timeline_index.csv` before publication of the working copy.

This stage is a timeline prerequisite, not a second collection and not a
replacement for the unified postprocess Skill. Direct calls to
`cross_exchange_l2_timeline.py` are reserved for focused development or
recovery; the daily path should use the supervisor's postprocess-only contract
so campaign identity, segment boundaries and reconnect evidence remain bound.

The working-copy stage passes only when:

- no collector process was started;
- the campaign identity and profile match the transferred raw campaign;
- all required child results and sample manifests are present;
- every timeline manifest has `passes=true`;
- the timeline index, segment boundaries, row counts, first/last timestamps
  and source hashes reconcile.

### 7.5 Unified Skill/CLI postprocess

After the timeline prerequisite passes, the pipeline invokes the existing
deterministic postprocess CLI. The Skill's documented command sequence is the
source of truth:

```bash
python -m examples.hyperliquid.cross_exchange_postprocess inspect \
  --campaign-dir CAMPAIGN_WORKING_COPY \
  --symbol-profile skhynix

python -m examples.hyperliquid.cross_exchange_postprocess run \
  --campaign-dir CAMPAIGN_WORKING_COPY \
  --output-dir POSTPROCESS_OUTPUT \
  --symbol-profile skhynix \
  --profile basis-research \
  --task-id TASK_ID

python -m examples.hyperliquid.cross_exchange_postprocess validate \
  --output-dir POSTPROCESS_OUTPUT

python -m examples.hyperliquid.cross_exchange_postprocess report \
  --output-dir POSTPROCESS_OUTPUT
```

Use `--profile dataset` when the daily package only needs raw audit, R0 and
R1. Use `--profile basis-research` when the package should also contain
reconnect-aware point-in-time basis/dislocation state. The default profile is
configuration-driven and must not be inferred from a natural-language request.

Use `resume` for an interrupted postprocess output. Reuse is allowed only when
the stored input fingerprint and artifact hashes validate. A profile change or
source campaign change requires a new output directory or a new attempt ID.

The daily pipeline must consume the Skill output contract rather than inspect
individual implementation details. Required output checks are:

- `pipeline_manifest.json` has `status=complete` and `passes=true`;
- every required stage has `status=complete` and `passes=true`;
- `source_immutable=true`;
- source/R0 hashes and row counts remain unchanged;
- R1 exact masks, horizon masks, future-join and timestamp-regression checks
  pass;
- at least one accepted primary horizon exists;
- for `basis-research`, `basis_dislocation` passes and binds accepted R0/R1
  manifests with strict as-of/trailing-left semantics;
- reconnect epochs show no old-state leakage before the reconnected venue's
  higher-epoch BBO recovery.

The daily pipeline must not directly call or duplicate the R0, R1 or basis
builders. The Skill CLI owns those contracts.

### 7.6 Report

The report stage generates both machine-readable JSON and a concise Markdown
summary:

- run ID and profile;
- remote and local paths;
- code/runtime source commit and hashes;
- requested/actual duration;
- Binance and Hyperliquid channel counts;
- reconnect and degraded interval summary;
- transfer file count and hash result;
- timeline, R0, R1 and optional basis row counts;
- postprocess profile and pipeline manifest status;
- input inventory fingerprint and provenance-lock result;
- stage statuses, exit codes, and elapsed time;
- final classification:
  - `complete`;
  - `collection_complete_pull_pending`;
  - `raw_complete_timeline_failed`;
  - `raw_complete_postprocess_failed`;
  - `failed`;
  - `quarantined`.

## 8. State Machine

The pipeline status file should use explicit states:

```text
created
preflight_running
preflight_failed
collecting
collection_failed
collection_complete
pulling
pull_failed
raw_verified
timeline_postprocessing
timeline_failed
timeline_verified
postprocess_inspecting
postprocessing
postprocess_failed
postprocess_validated
complete
quarantined
```

Each transition records:

- `state`;
- `updated_at` in UTC;
- stage start/end timestamps;
- process ID and command identifier;
- exit code;
- artifact path;
- error code and short error message;
- retry count.

The state file must be updated atomically and flushed before a stage is
considered complete.

## 9. Locking, Retry, And Recovery

Use one lock per `(profile_id, scheduled_date)` on the controller and one
campaign lock on the collector host.

Retry policy:

- preflight: no automatic retry for configuration or disk failures;
- remote launch/status polling: bounded retry with exponential backoff;
- collection: do not automatically start a second campaign after partial
  failure; quarantine the first run and require a new attempt ID;
- transfer: retry resumably while the remote campaign remains immutable;
- timeline prerequisite: retry from a fresh working copy of verified raw data;
- postprocess: use Skill `resume` when the output fingerprint is valid, or
  create a new attempt/output directory when it is not;
- report/notification: retry without changing data artifacts.

The system must distinguish:

- remote collection failed;
- remote collection passed but pullback failed;
- pullback passed but timeline prerequisite failed;
- timeline passed but Skill postprocess failed;
- complete data package.

These states must not be collapsed into a generic `failed` message.

## 10. Logging

Use three complementary logging layers:

1. systemd journal for service lifecycle, exit code, restart, and resource
   events;
2. stage-specific text logs for human debugging;
3. structured JSONL events for machine inspection and daily summaries.

Each structured event should contain:

```json
{
  "run_id": "20260819-skhynix",
  "profile_id": "skhynix",
  "stage": "pull",
  "event": "file_hash_verified",
  "timestamp": "2026-08-18T00:00:00Z",
  "path": "segments/segment_0001/...",
  "sha256": "...",
  "result": "pass"
}
```

Do not log private keys, access tokens, full environment variables, or
unredacted SSH commands containing secrets.

Logs must include enough information to answer:

- which code version ran;
- which host ran it;
- which profile and symbols were collected;
- when each stage started and ended;
- what was transferred;
- which check failed;
- whether data was published or quarantined.

## 11. systemd Units

The controller host should have:

```text
hftbacktest-daily-cross-exchange.service
hftbacktest-daily-cross-exchange.timer
```

The service should:

- use a dedicated service account;
- set the repository working directory explicitly;
- invoke the pipeline CLI with an explicit config;
- use a bounded timeout longer than collection plus transfer and preprocessing;
- write to the journal and pipeline log;
- use `flock` or the pipeline lock to prevent overlap;
- keep environment and credentials outside the repository.

The timer should:

- use UTC;
- enable `Persistent=true` so a missed run is caught up;
- avoid overlapping runs;
- use a small randomized delay if multiple profiles will later be scheduled;
- trigger a notification or status check after completion.

The collector host should run the actual long-lived campaign under its own
systemd unit or transient unit so network collection is not tied to the
controller's SSH connection.

## 12. Retention And Publication

Keep at least:

- all manifests;
- runtime source archive;
- raw SHA files;
- pipeline status and reports;
- failed-run metadata;
- derived dataset manifests.

Raw data retention must be configurable by profile. Failed and quarantined
runs should be retained longer than successful runs until their failure has
been reviewed.

Only a complete and verified run may update:

```text
local_live_analysis/daily_cross_exchange/<profile_id>/latest
```

The pointer update must be atomic. Existing accepted runs must remain
available after a later run fails.

## 13. Verification And Acceptance

### 13.1 CLI acceptance

- `--help` works without network access;
- `preflight` rejects an invalid profile and conflicting run ID;
- `run` can execute against a local fixture without SSH;
- stage commands are idempotent;
- all status transitions are persisted after process interruption.

### 13.2 Collection acceptance

- Binance and Hyperliquid identities match configuration;
- required tracks and subscription acknowledgements are present;
- duration and overlap pass configured thresholds;
- raw rows and SHA-256 values reconcile;
- reconnect policy is explicit and fail-closed;
- runtime source archive matches the code hash recorded in the manifest.

### 13.3 Pullback acceptance

- transfer can resume after interruption;
- a corrupted file fails verification;
- incomplete staging data never becomes the published raw directory;
- successful transfer produces a complete local manifest.

### 13.4 Timeline and Skill postprocess acceptance

- supervisor `--postprocess-only` runs without starting collectors;
- source raw bytes remain unchanged;
- timeline manifests and the campaign timeline index pass before the Skill is
  invoked;
- Skill `inspect` passes against the working copy;
- Skill `run` or `resume` produces the documented output contract;
- `validate` passes after the build;
- source immutability, provenance, masks, row counts and output hashes pass;
- one failed stage does not remove a successful earlier stage.

### 13.5 Scheduled-run acceptance

Before enabling daily unattended operation:

1. pass one dry-run with no network writes;
2. pass one short live public-data canary;
3. pass one complete daily collection;
4. interrupt transfer and prove resumable recovery;
5. inject a hash mismatch and prove quarantine;
6. interrupt preprocessing and prove local retry;
7. complete two consecutive scheduled runs without overlap;
8. verify reports and logs from the systemd journal and local files.

## 14. Implementation Sequence

### Phase 1: Freeze the current collection contract

- commit the current supervisor and timeline changes;
- define the runtime source hash contract;
- add a configuration schema and one `skhynix` config;
- document the remote collector host and local controller host.

### Phase 2: Implement the pipeline CLI and stage state

- add run identity, state machine, locks, and stage dispatch;
- wrap the existing supervisor in collection-only mode;
- add explicit remote completion polling and transfer admission;
- implement local fixture mode;
- write structured stage events and final reports.

### Phase 3: Implement verified pullback

- add remote status polling;
- add resumable staging transfer;
- verify all manifest-listed files and hashes;
- atomically publish `raw_campaign`.

### Phase 4: Integrate the existing postprocess Skill

- copy raw data to an immutable-source working copy;
- invoke supervisor `--postprocess-only` to create accepted timelines;
- call Skill CLI `inspect` before `run`;
- select `dataset` or `basis-research` from configuration;
- use Skill CLI `resume`, `validate`, and `report` for recovery and closure;
- record the postprocess runtime source, pipeline manifest and artifact hashes.

### Phase 5: Install systemd automation

- add service and timer templates;
- configure service account, environment, timeout, and retention;
- run dry-run and interruption tests;
- enable one daily profile only.

### Phase 6: Operational hardening

- add alerts and daily summary;
- add disk-space and stale-lock checks;
- add cleanup based on retention policy;
- run a seven-day reliability observation before adding more profiles.

## 15. Relationship To The Existing Skill

The existing Skill is:

```text
.agents/skills/cross-exchange-postprocess/SKILL.md
```

It should remain the operator-facing explanation and policy layer. The daily
systemd service must invoke the same deterministic Python module directly, so
the schedule remains independent of an interactive Codex session.

The Skill may provide:

- `status` interpretation;
- `report` summarization;
- `recover` guidance;
- manual `run` or `retry` commands after explicit user confirmation.

The Skill must not be the scheduler, the source of truth for state, or the
only place where validation logic exists. The pipeline status file and the
postprocess manifests remain the machine-readable sources of truth.

## 16. Recommended First Implementation Boundary

The first formal implementation task should cover one profile only:

- profile: `skhynix`;
- duration: four hours;
- one continuous segment;
- remote collection;
- verified local pullback;
- supervisor `--postprocess-only` timeline prerequisite;
- Skill `dataset` profile as the minimum postprocess path;
- optional Skill `basis-research` profile;
- structured logs and Markdown/JSON report;
- no downstream signal, lead-lag, maker-PnL research, or live trading.

Once this path passes interruption, hash-mismatch, and two-consecutive-run
acceptance, add other profiles through configuration rather than cloning the
pipeline.
