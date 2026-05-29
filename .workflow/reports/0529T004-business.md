```md
执行线程：
- 业务线程-python

任务ID：
- 0529T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0529T004.md`
- `.workflow/reports/0529T004-business.md`
- `examples/hyperliquid/hyperliquid_public_sample.py`
- `examples/hyperliquid/test_hyperliquid_public_sample.py`
- `examples/hyperliquid/hyperliquid_raw_alignment.py`
- `examples/hyperliquid/test_hyperliquid_raw_alignment.py`
- `local_live_analysis/hyperliquid_public_sample_0529T004/raw.gz`
- `local_live_analysis/hyperliquid_public_sample_0529T004/raw.sha256`
- `local_live_analysis/hyperliquid_public_sample_0529T004/collection_manifest.json`
- `local_live_analysis/hyperliquid_public_sample_0529T004/recovery_snapshots.jsonl`
- `local_live_analysis/hyperliquid_public_sample_0529T004/alignment/**`
- `progress.md`
- `findings.md`

action：
- Read the workflow rules, `docs/thread-playbook.md`, task/progress/findings context, `0529T003` business report, existing Hyperliquid converter, T003 alignment runner/tests, and Hyperliquid design notes.
- Rechecked official Hyperliquid references before relying on current schema facts:
  - `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api`
  - `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket`
  - `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions`
  - `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint`
  - `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size`
  - `https://github.com/hyperliquid-dex/hyperliquid-python-sdk`
- Added `examples/hyperliquid/hyperliquid_public_sample.py`, a public-only collector for `l2Book` and `trades` plus public Info `l2Book` startup/reconnect recovery snapshots.
- The local `websockets` package was unavailable; no dependency install was performed. The collector used the already installed `websocket-client` fallback and records this in `collection_manifest.json` as `preferred_websockets_available=false`, `selected_websocket_library=websocket-client`.
- Extended `examples/hyperliquid/hyperliquid_raw_alignment.py` to consume collector `collection_manifest.json` and `recovery_snapshots.jsonl`, carry session/connection evidence into provenance/sidecar rows, and classify fresh samples with subscription/recovery evidence as `passes_pricing_research_market_view`.
- Added focused tests for collector manifest/raw output and alignment integration with session/recovery evidence.
- Ran one fresh 120s public-only Hyperliquid mainnet BTC sample and wrote T004 artifacts under `local_live_analysis/hyperliquid_public_sample_0529T004/`.
- Ran the T003 alignment runner on the fresh T004 raw gzip into `local_live_analysis/hyperliquid_public_sample_0529T004/alignment/`.
- Kept scope public market data only: no private connector, no private keys, no account endpoints, no order submit/cancel, no fill lifecycle, no strategy behavior, no live trading strategy process, no remote deploy, no parameter search, no default-on behavior, no guard relaxation, no tiny-live, and no promotion claim.
- Did not add Binance `U/u/pu`, `lastUpdateId`, `bookTicker`, or Binance bootstrap semantics to Hyperliquid artifacts.

verify：
- `python examples/hyperliquid/hyperliquid_public_sample.py --help` -> passed.
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help` -> passed.
- `python -m py_compile py-hftbacktest/hftbacktest/data/utils/hyperliquid.py examples/hyperliquid/hyperliquid_public_sample.py examples/hyperliquid/hyperliquid_raw_alignment.py` -> passed.
- `python -m pytest examples/hyperliquid/test_hyperliquid_raw_alignment.py -q` -> passed, `4 passed`.
- `python -m pytest examples/hyperliquid/test_hyperliquid_public_sample.py -q` -> passed, `2 passed`.
- `python -m pytest examples/hyperliquid/test_hyperliquid_raw_alignment.py examples/hyperliquid/test_hyperliquid_public_sample.py -q` -> passed, `6 passed`.
- First sandboxed network collection attempt failed with DNS/name-resolution errors and wrote no market-data messages; reran the same public-only collection with approved network escalation.
- `python examples/hyperliquid/hyperliquid_public_sample.py --duration-seconds 120 --output-dir local_live_analysis/hyperliquid_public_sample_0529T004` with network escalation -> passed.
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip local_live_analysis/hyperliquid_public_sample_0529T004/raw.gz --output-dir local_live_analysis/hyperliquid_public_sample_0529T004/alignment --source-label fresh_public_sample --task-id 0529T004 --collection-manifest local_live_analysis/hyperliquid_public_sample_0529T004/collection_manifest.json --recovery-snapshots local_live_analysis/hyperliquid_public_sample_0529T004/recovery_snapshots.jsonl` -> passed.
- `git diff --check` -> passed.
- `python3 .workflow/build_dashboard.py` -> not run because `.workflow/tasks/0529T005.md` is present outside T004 scope and dashboard regeneration would mix unrelated task state into this T004 report/update.
- No strategy live command was run.
- No remote deploy command was run.
- No private key, account, order submit, or order cancel endpoint command was run.

done：
- Fresh sample path: `local_live_analysis/hyperliquid_public_sample_0529T004/raw.gz`.
- Raw sha256: `137018ef937b3692a5de0c12ee009c4a93a0e6d62ff15321061c377fc514389c`.
- Collection duration: requested `120.0s`, actual `120.100474382s`.
- Collection manifest: mainnet BTC, channels `l2Book` and `trades`, session id `hl-66959ecb09e941cab61b020ac1e06419`, connection attempts `1`, reconnect count `0`, close reason `duration_elapsed`.
- Subscription evidence: total ack `2`, `l2Book=1`, `trades=1`.
- Recovery evidence: startup public Info `l2Book` snapshot count `1`, status `ok`, best bid/ask `73681.0 / 73682.0`, bid/ask level counts `20 / 20`.
- Fresh raw message counts: `l2Book=222`, `trades=111`, `subscriptionResponse=2`; raw alignment also observed `pong=3`.
- Alignment metrics: raw parse errors `0`, trade events `418`, `data.npz` rows `4279`, event-order validation `passed`, top-N coverage `1.0`, synthetic join coverage `1.0`, future joins `0`, missing joins `0`, join-age p99 `1114.135914ms`, l2Book cadence p99 `1317.185171ms`.
- Final classification: `passes_pricing_research_market_view`.
- Required artifacts were generated: `run_manifest.json`, `converter_manifest.json`, `data.npz`, `raw_provenance.csv`, `raw_to_npz_mapping.csv`, `topn_sidecar.csv`, `synthetic_joined_views.csv`, `metrics.json`, and `acceptance_report.md`.
- Next recommended task after QA acceptance: a narrow planning/design task for the next Hyperliquid public market-data research consumer, if desired, still before any private connector, order lifecycle, strategy live logic, parameter search, default-on, tiny-live, or promotion task.
- No Binance strategy behavior changed.
- No Hyperliquid private connector was implemented.
- No order submit/cancel/fill lifecycle code was implemented.
- No trading live or remote deploy was run.
- No default-on, guard relaxation, parameter search, tiny-live, or promotion claim was made.

blockers：
- 无

commit：
- c79d7af

提交信息：
- Add Hyperliquid public sample evidence
```
