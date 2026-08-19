# Cross-Exchange Postprocess Pipeline

Task: `0805T002`

## Purpose

The pipeline turns one accepted, postprocessed collection campaign into a
versioned and auditable research dataset:

```text
campaign/raw audit
-> R0 normalized event store
-> R1 exact-mask alignment labels
-> optional point-in-time basis/dislocation state
-> dataset manifest and compact report
```

The deterministic implementation lives in Python. The repo-scoped Codex Skill
selects the workflow, invokes the CLI and explains the result; it does not
implement replay, alignment or signal algorithms.

## Commands

Inspect an existing campaign without writing a dataset:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess inspect \
  --campaign-dir /path/to/campaign \
  --symbol-profile skhynix
```

Build a new dataset:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess run \
  --campaign-dir /path/to/campaign \
  --output-dir /path/to/postprocess-output \
  --symbol-profile skhynix \
  --profile dataset \
  --task-id MMDDTxxx
```

Resume after interruption and reuse only stages whose input fingerprint and
artifact hashes still validate:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess resume \
  --campaign-dir /path/to/campaign \
  --output-dir /path/to/postprocess-output \
  --symbol-profile skhynix \
  --profile dataset \
  --task-id MMDDTxxx
```

Validate or regenerate the compact report:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess validate \
  --output-dir /path/to/postprocess-output

/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess report \
  --output-dir /path/to/postprocess-output
```

## Output Contract

```text
postprocess-output/
  pipeline_manifest.json
  quality_summary.json
  provenance_lock.json
  dataset_report.md
  state/
    raw_audit.json
    r0.json
    r1.json
  stages/
    raw_audit/
  r0/
  r1/
  basis/                         # basis-research profile only
```

Every completed stage records:

- stage and schema version;
- input fingerprint;
- runtime source SHA;
- output file path, byte count and SHA;
- duration, status and reuse state;
- a compact stage-specific summary.

The provenance lock records the complete source campaign inventory. A run
fails closed if the source tree changes while the pipeline is executing.

## Profiles

`dataset` is executable and stops after accepted R0/R1.

`basis-research` is executable and adds:

```text
Binance bookTicker + Hyperliquid BBO receipt-time union
-> reconnect-aware point-in-time state
-> basis_mid / d_bh / d_hb
-> trailing-only dislocation features and quality report
```

For a Binance or Hyperliquid fast reconnect, the basis stage requires a higher
`connection_epoch_id` BBO for the reconnected venue, suppresses any old-epoch
BBO after the reset, and fails closed when that recovery proof is absent.

The supervisor proves the two venue families differently. Binance binds
attempt/subscription-response/bridge/disconnect counts to embedded snapshot
and depth bridge rows, then requires depth, bookTicker and trade recovery.
Hyperliquid additionally binds subscription identities, transport markers and
recovery snapshots. These are separate evidence contracts, not interchangeable
proofs.

The 100ms changes, trailing median/MAD/z state, 60-second Binance volatility
and 15-minute feature warmup restart when either consumed venue changes
connection epoch. The stage also verifies every consumed R0 artifact against
the manifest-declared SHA and row count before processing.

Run it by replacing `--profile dataset` with
`--profile basis-research`.

`signal-research` remains registered but fails explicitly until the remaining
stages migrate:

```text
lead_lag
-> maker_diagnostics
```

`full-research` and the Atom/Episode/Motif/Regime hierarchy are deferred. They
are not part of the current migration path because they have not demonstrated
incremental alpha.

## Boundaries

- The input must already have an accepted campaign manifest and common L2
  timelines. Raw-only supervisor postprocess remains a separate controlled
  phase.
- The pipeline performs no live collection and accesses no private or order
  endpoints.
- Public L2 does not provide L3/L4 queue position or exact fills.
- Receipt-time precedence is not causal Binance leadership.
- Dataset validity is not executable arbitrage, maker PnL or live promotion.
- Basis/dislocation output is point-in-time feature state only. It contains no
  future horizon outcome, lead-lag conclusion or maker-fill claim.
