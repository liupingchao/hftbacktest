# Output Contract

Read these files in order:

1. `pipeline_manifest.json`
2. `quality_summary.json`
3. `dataset_report.md`
4. `provenance_lock.json` when source immutability matters

Required success conditions:

- pipeline `status=complete` and `passes=true`;
- every required stage has `status=complete` and `passes=true`;
- `source_immutable=true`;
- R0 source hashes unchanged;
- R1 exact masks, exact horizon masks and reconciliation pass;
- cross-epoch labels, future joins and timestamp regressions are zero;
- at least one accepted primary horizon.
- for `basis-research`, the `basis_dislocation` stage passes, binds the accepted
  R0/R1 manifests, preserves input hashes, and reports strict as-of plus
  trailing-left feature semantics.
- consumed-venue reconnect quality reports zero old-epoch state leakage and
  requires a higher-epoch Binance or Hyperliquid BBO before state resumes.
- 100ms changes, rolling state, Binance volatility and feature warmup do not
  cross Binance or Hyperliquid fast reconnect epochs.
- every consumed R0 mask/hot artifact matches its manifest-declared SHA and
  row count before the basis stage starts.

Capability language:

- `pass`: implemented and passed for this dataset;
- `not_run`: the dataset pipeline did not execute that research family;
- `not_supported`: the public-data contract cannot establish the claim;
- `pending`: an upstream stage has not completed.

The compact report is a summary. Manifests and bound artifact SHA values are
the audit source of truth.
