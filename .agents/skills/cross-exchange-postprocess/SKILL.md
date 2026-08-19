---
name: cross-exchange-postprocess
description: Run, resume, validate, or explain the Binance/Hyperliquid public-data postprocess pipeline when the user asks to turn a collected raw campaign into an auditable R0/R1 dataset, add point-in-time basis/dislocation state, compare a rebuild with a golden dataset, or inspect alignment and replay eligibility. Do not use it to start live collection without explicit authorization.
---

# Cross-Exchange Postprocess

Use the repository's deterministic Python pipeline. Do not reimplement replay,
alignment, masks, basis, or hierarchy logic inside the Skill.

## Before Running

1. Read `AGENTS.md` and the required workflow-kit documents.
2. Confirm the campaign path, symbol profile and output path.
3. Treat the source campaign as immutable.
4. Use the local `hftbacktest` Conda Python unless the task explicitly targets
   another host.

## Commands

Inspect:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess inspect \
  --campaign-dir CAMPAIGN \
  --symbol-profile PROFILE
```

Run:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess run \
  --campaign-dir CAMPAIGN \
  --output-dir OUTPUT \
  --symbol-profile PROFILE \
  --profile dataset \
  --task-id TASK_ID
```

Use `--profile basis-research` when the requested output includes midpoint
basis, `binance bid1 - hyperliquid ask1`, or
`hyperliquid bid1 - binance ask1` point-in-time features.

For a campaign with accepted Binance or Hyperliquid core reconnect masks,
confirm the reconnected venue has higher-epoch BBO recovery and zero old-state
leakage before describing the basis stage as passing.

Use `resume` instead of `run` when OUTPUT contains an interrupted or completed
pipeline. Reuse is allowed only when the stored input fingerprint and every
artifact hash validate.

Validate:

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  -m examples.hyperliquid.cross_exchange_postprocess validate \
  --output-dir OUTPUT
```

Read `references/output-contract.md` before explaining the report or making a
research-eligibility statement.

## Profile Selection

- `dataset`: executable; raw audit, R0 and R1.
- `basis-research`: executable; dataset plus reconnect-aware point-in-time
  basis/dislocation state.
- `signal-research`: registered but not executable until its stages migrate.
- `full-research`: deferred; hierarchy stages are not in the current migration
  path.

Never describe a registered but unimplemented profile as complete.

## Hard Boundaries

- Never start or schedule live collection without explicit user authorization.
- Never modify the source campaign or accepted golden artifacts.
- Never claim L3/L4 queue reconstruction, exact fill, executable arbitrage,
  account PnL, causal Binance leadership, or live readiness from this pipeline.
- Never interpret basis state as a future outcome or alpha result; lead-lag and
  maker diagnostics require their own accepted downstream stages.
