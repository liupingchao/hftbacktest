# Tutorial Reproduction

This directory reproduces the first three example groups with existing
Tardis data:

1. Basic usage
2. Data preparation
3. Backtesting capabilities

`run.py` stages a bounded time window from Tardis `.csv.zst` files, builds
reusable hftbacktest artifacts, and writes one JSON result per notebook plus
a top-level manifest. The nine notebooks are modified only to add the runnable
Tardis section described below; their original content remains as reference.

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

## Runnable notebooks

The notebooks covered by this reproduction now contain a short active
Tardis test section. Their original tutorial code is retained as non-executing
reference cells, so `Run All` uses only the configured existing dataset.

The notebooks auto-detect the amdserver and Mac paths shown above. Override
the defaults before starting Jupyter when needed:

```bash
export HFTBACKTEST_TARDIS_ROOT=/home/molly/data/tardis/binance-futures
export HFTBACKTEST_TARDIS_DATE=2025-08-01
export HFTBACKTEST_NOTEBOOK_SECONDS=300
```

Notebooks share prepared-data caches under their task-specific
`local_live_analysis/tutorial_reproduction_<task-id>/` directory.

To regenerate or validate their structured notebook content:

```bash
/home/molly/anaconda3/bin/python \
  examples/tutorial_reproduction/refresh_notebooks.py --check
```

## Advanced strategy data boundary

The grid, GLFT, OBI, basis, APT, and pricing notebooks use BTCUSDT futures
depth, trades, book ticker, and derivative ticker data from the same Tardis
window. The available amdserver mount for this date does not include Binance
spot/FDUSD or the original multi-asset panel. Basis, APT, and pricing examples
therefore use `derivative_ticker.index_price` as an observable external factor
and report `adapted`; they do not claim that index price is identical to the
original spot or cross-asset inputs.

## Multi-market and queue-model copies

Task `0804T005` keeps the five source notebooks unchanged and writes runnable
copies under:

```text
examples/tutorial_reproduction/notebooks/0804T005/
```

The copies cover:

- market diversification across available BTC perpetual venues
- a deterministic version of the introductory diversification simulation
- Square, Log, and Power probability queue-model comparison
- causal book-pressure, trade-impulse, and thin-queue backoff signals
- common-parameter comparison across Binance, Bybit, OKX, Bitget, and Gate

On amdserver the additional venue roots are discovered under `/mnt/4t_sda1`.
Override that location with:

```bash
export HFTBACKTEST_MULTI_TARDIS_ROOT=/mnt/4t_sda1
```

The Mac sample contains only Binance Futures, so multi-market copies run there
with one real venue and report `adapted`. The amdserver acceptance run requires
all five mounted venues. The queue-based large-tick tutorial uses BTCUSDT
because the specified mounts do not contain the original CRVUSDT input; its
manifest states that this is a signal-mechanics adaptation rather than an
equivalent large-tick reproduction.
