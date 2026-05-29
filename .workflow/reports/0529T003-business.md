```md
执行线程：
- 业务线程-python

任务ID：
- 0529T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0529T003.md`
- `.workflow/reports/0529T003-business.md`
- `examples/hyperliquid/hyperliquid_raw_alignment.py`
- `examples/hyperliquid/test_hyperliquid_raw_alignment.py`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/run_manifest.json`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/collection_manifest.json`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/converter_manifest.json`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/data.npz`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/raw_provenance.csv`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/raw_to_npz_mapping.csv`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/topn_sidecar.csv`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/synthetic_joined_views.csv`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/metrics.json`
- `local_live_analysis/hyperliquid_raw_alignment_0529T003/acceptance_report.md`

action：
- Read the workflow rules, `task_plan.md`, `progress.md`, `findings.md`, `0529T003` task scope, `0528T003` design/report/QA context, and the existing Hyperliquid converter.
- Used the official Hyperliquid docs / SDK references recorded in `run_manifest.json` for schema assumptions: API, WebSocket, subscriptions, info endpoint, tick/lot size, and official `hyperliquid-python-sdk`.
- Implemented a task-scoped read-only runner at `examples/hyperliquid/hyperliquid_raw_alignment.py`.
- The runner consumes a local line-oriented Hyperliquid gzip sample, parses `l2Book` and `trades`, invokes the existing Hyperliquid converter without changing the standard npz schema, writes raw provenance and raw-to-npz row mapping, reconstructs a Hyperliquid top-N sidecar from `l2Book` snapshots, and writes a market-data-only synthetic as-of join.
- Added focused tests for artifact generation, classification, Hyperliquid sidecar fields, and as-of join behavior.
- Ran the runner on `examples/hyperliquid/btcusd_20250126.gz` and wrote the required artifacts under `local_live_analysis/hyperliquid_raw_alignment_0529T003/`.
- Kept the scope read-only: no private connector, no account endpoint, no order submit/cancel, no fill lifecycle, no strategy behavior, no live/tiny-live, no remote deploy, no parameter search, no default-on behavior, no guard relaxation, and no promotion claim.
- Did not copy Binance `U/u/pu`, `lastUpdateId`, `bookTicker`, or Binance top5 bootstrap semantics into the Hyperliquid validation.

verify：
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help` -> passed.
- `python -m pytest examples/hyperliquid/test_hyperliquid_raw_alignment.py -q` -> passed, `3 passed`.
- `python -m py_compile py-hftbacktest/hftbacktest/data/utils/hyperliquid.py` -> passed.
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip examples/hyperliquid/btcusd_20250126.gz --output-dir local_live_analysis/hyperliquid_raw_alignment_0529T003 --source-label existing_local_sample` -> passed.
- `git diff --check --cached` before implementation/artifact commit -> passed.
- `python3 .workflow/build_dashboard.py` -> not run for this task because the current worktree contains unrelated parallel `.workflow/tasks/0529T002.md`; the dashboard script loads every task file and would mix T002 into generated dashboard/dispatch outputs, violating the T003 isolation boundary.
- No strategy live command was run.
- No remote deploy command was run.
- No private key, private account, order submit, or order cancel endpoint command was run.

done：
- Generated `run_manifest.json`, `collection_manifest.json`, `converter_manifest.json`, `data.npz`, `raw_provenance.csv`, `raw_to_npz_mapping.csv`, `topn_sidecar.csv`, `synthetic_joined_views.csv`, `metrics.json`, and `acceptance_report.md`.
- Input sample: `examples/hyperliquid/btcusd_20250126.gz`.
- Sample source: existing local sample, not a fresh network collection.
- Key metrics: raw messages `55`, `l2Book` messages `46`, `trades` messages `9`, trade events `64`, converted npz rows `413`, parse errors `0`, top-N coverage `1.0`, synthetic join coverage `1.0`, future joins `0`, missing joins `0`, join-age p99 `623.4895ms`, event-order validation `passed`.
- Final sample classification: `limited_pricing_research`.
- Classification reason: market view and conversion evidence are usable, but the existing local sample lacks subscription ack/session/reconnect/recovery snapshot evidence required for `passes_pricing_research_market_view`.
- Next recommended task: Hyperliquid read-only public collector manifest hardening / fresh short sample collection for `l2Book + trades` session, reconnect, subscription ack, and recovery snapshot evidence only; still no private connector, strategy, order lifecycle, or live trading.
- No Binance strategy behavior changed.
- No Hyperliquid private connector was implemented.
- No order submit/cancel/fill lifecycle code was implemented.
- No trading live or remote deploy was run.
- No default-on, guard relaxation, parameter search, tiny-live, or promotion claim was made.

blockers：
- 无

commit：
- 8dfff8b

提交信息：
- Add Hyperliquid raw alignment artifacts
```
