# Tutorial Reproduction

This directory reproduces the first three example groups with existing
Tardis data:

1. Basic usage
2. Data preparation
3. Backtesting capabilities

The original notebooks are not modified. `run.py` stages a bounded time
window from Tardis `.csv.zst` files, builds reusable hftbacktest artifacts,
and writes one JSON result per notebook plus a top-level manifest.

## Data layouts

Mac sample:

```text
~/Documents/tardis/<data_type>/BTCUSDT.csv.zst
```

amdserver:

```text
/home/molly/data/tardis/binance-futures/<data_type>/YYYY/MM/DD/BTCUSDT.csv.zst
```

## Run

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  examples/tutorial_reproduction/run.py \
  --tardis-root ~/Documents/tardis \
  --date 2025-01-01 \
  --duration-seconds 300 \
  --output-root local_live_analysis/tutorial_reproduction_0804T002/mac
```

On amdserver:

```bash
/home/molly/anaconda3/bin/python \
  examples/tutorial_reproduction/run.py \
  --tardis-root /home/molly/data/tardis/binance-futures \
  --date 2025-08-01 \
  --duration-seconds 300 \
  --output-root local_live_analysis/tutorial_reproduction_0804T002/amdserver
```

The runner does not download data and does not require a Tardis API key.
Generated `.csv.gz`, `.npz`, `.parquet`, and plot files stay under the
ignored output root.
