# Daily Cross-Exchange Collection Pipeline Plan

Date: 2026-08-18

Status: design proposal

Scope: scheduled public-data collection for a configured Binance/Hyperliquid
symbol profile, remote artifact pullback, local preprocessing, and durable
logs.

## 1. Objective

Build a deterministic daily pipeline that can run without an interactive
terminal and produce one auditable data package per run:

1. collect the configured symbol from Binance and Hyperliquid;
2. validate the remote campaign before transfer;
3. pull the completed campaign to the local repository host;
4. verify all transferred files against the remote manifests;
5. build the common L2 timeline and the local R0 research event store;
6. record structured logs, status, hashes, and failure reasons.

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
systemd, CI, or a future Codex skill. The CLI must not depend on an LLM,
interactive approval, terminal state, or an SSH session remaining open.

### 2.2 systemd schedules; it does not contain business logic

The systemd timer starts one pipeline service. Collection, transfer,
preprocessing, validation, and reporting stay in versioned repository code.

### 2.3 Raw data is immutable

The original remote campaign is copied without mutation. Preprocessing writes
separate derived directories. A failed preprocessing step must never rewrite
or delete accepted raw data.

### 2.4 Every stage is independently observable

Collection, transfer, timeline construction, R0 construction, and optional R1
alignment must have separate status, exit code, start/end time, log, and
artifact references.

### 2.5 Publication is atomic

Partial remote transfers and incomplete preprocessing outputs must not appear
under the directory used by downstream readers. Write to a staging directory,
verify it, then atomically rename it to the final run directory.

## 3. Existing Components To Reuse

The pipeline should wrap the existing components rather than duplicate their
collection logic:

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
  - optional R1 alignment acceptance after the daily R0 package is available.

Before implementation, the current supervisor and timeline changes must be
committed and their runtime hashes frozen. The runtime source archive recorded
inside every campaign must remain the source of truth for that run.

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
  --run-id 20260818-skhynix \
  --stages collect,pull,timeline,r0,report
```

The `run` command performs all requested stages in order and exits non-zero
if a required stage fails.

### 4.2 Individual stages

```bash
python examples/hyperliquid/daily_cross_exchange_pipeline.py preflight \
  --config configs/daily_cross_exchange/skhynix.json

python examples/hyperliquid/daily_cross_exchange_pipeline.py collect \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py pull \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py preprocess \
  --config configs/daily_cross_exchange/skhynix.json \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py verify \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py status \
  --run-id 20260818-skhynix
```

Stage commands must be idempotent. Re-running a successful stage should
verify and reuse its existing artifact, not silently overwrite it. A `--force`
option may be added later for explicit rebuilds into a new attempt directory.

### 4.3 Recovery commands

```bash
python examples/hyperliquid/daily_cross_exchange_pipeline.py retry-pull \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py retry-preprocess \
  --run-id 20260818-skhynix

python examples/hyperliquid/daily_cross_exchange_pipeline.py report \
  --run-id 20260818-skhynix
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
  "preprocess": {
    "timeline": true,
    "research_dataset": true,
    "alignment_acceptance": false
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
  raw_campaign/
  preprocess/
    timeline/
    research_dataset/
    alignment/
  logs/
    pipeline.jsonl
    pull.log
    preprocess.log
  manifests/
    remote_campaign_manifest.json
    local_transfer_manifest.json
    preprocess_manifest.json
  run_status.json
  run_report.md
```

The raw campaign directory must remain byte-preserving. The `latest` pointer,
if introduced, must point only to a fully verified directory and must never
point to a staging directory.

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

### 7.4 Timeline preprocessing

Run `cross_exchange_l2_timeline.py` independently for each segment or through
the supervisor's postprocess path.

The stage must:

- preserve segment boundaries;
- preserve connection epoch and reconnect interval evidence;
- build the common local-receipt-time timeline;
- record timeline row count, first/last timestamps, source hashes, and
  continuity status;
- publish only after the timeline manifest reports `passes=true`.

### 7.5 R0 research preprocessing

Run `cross_exchange_research_dataset.py` over the verified local campaign and
timeline.

The R0 stage must:

- read only the local verified raw campaign;
- preserve original raw files as inputs;
- produce normalized hot-event and auxiliary sidecars;
- include source path, SHA-256, row counts, profile identity, and segment
  identity;
- fail closed on missing tracks, identity mismatch, timestamp regression,
  row-count mismatch, or source hash mismatch.

The minimum daily published package is the raw campaign plus passing timeline
and R0 manifests.

R1 alignment acceptance is configurable. It should be a separate stage rather
than making the daily raw/R0 package unavailable when a longer research
acceptance run is delayed.

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
- timeline and R0 row counts;
- stage statuses, exit codes, and elapsed time;
- final classification:
  - `complete`;
  - `collection_complete_pull_pending`;
  - `raw_complete_preprocess_failed`;
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
preprocessing
preprocess_failed
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
- preprocessing: retry locally from verified raw data;
- report/notification: retry without changing data artifacts.

The system must distinguish:

- remote collection failed;
- remote collection passed but pullback failed;
- pullback passed but preprocessing failed;
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
  "run_id": "20260818-skhynix",
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

### 13.4 Preprocessing acceptance

- timeline and R0 commands run from the local package only;
- source raw bytes are unchanged;
- segment boundaries and degraded intervals are preserved;
- output manifests contain source hashes and row counts;
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

### Phase 2: Implement the pipeline CLI

- add run identity, state machine, locks, and stage dispatch;
- wrap the existing supervisor in collection-only mode;
- implement local fixture mode;
- write structured stage events and final reports.

### Phase 3: Implement verified pullback

- add remote status polling;
- add resumable staging transfer;
- verify all manifest-listed files and hashes;
- atomically publish `raw_campaign`.

### Phase 4: Implement preprocessing

- invoke timeline construction;
- invoke R0 research dataset construction;
- write preprocessing manifests;
- add optional R1 stage without blocking raw/R0 publication.

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

## 15. Relationship To A Future Skill

The future Codex skill should call this CLI rather than reimplement the
pipeline. It may provide:

- `status` interpretation;
- `report` summarization;
- `recover` guidance;
- manual `run` or `retry` commands after explicit user confirmation.

The skill must not be the scheduler, the source of truth for state, or the
only place where validation logic exists.

## 16. Recommended First Implementation Boundary

The first formal implementation task should cover one profile only:

- profile: `skhynix`;
- duration: four hours;
- one continuous segment;
- remote collection;
- verified local pullback;
- common L2 timeline;
- R0 research dataset;
- structured logs and Markdown/JSON report;
- no R1 signal research and no live trading.

Once this path passes interruption, hash-mismatch, and two-consecutive-run
acceptance, add other profiles through configuration rather than cloning the
pipeline.
