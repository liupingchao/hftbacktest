# 线程回报

执行线程：
- 业务线程-hyperliquid-public-data

任务ID：
- 0601T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0601T001.md`
- `.workflow/reports/0601T001-business.md`
- `examples/hyperliquid/hyperliquid_public_sample.py`
- `examples/hyperliquid/test_hyperliquid_public_sample.py`
- `local_live_analysis/hyperliquid_public_sample_0601T001/**`

action：
- 将 `0601T001` 从草案改成正式待执行派发口径，明确采集命令、输出目录、采样时长、失败重试规则、通过条件和阻塞条件。
- 给 Hyperliquid public collector 增加 `--task-id` 参数，使正式采样可以把 `collection_manifest.json` 和 recovery snapshots 标记为 `0601T001`，避免沿用旧 `0529T004` task id。
- 用 accepted `0529T004` public artifact 做旧样本 alignment preflight，确认正式采样窗口到来前本地 collector/alignment CLI、参数和 artifact 生成链路可用。
- 按 `0601T001` 正式命令采集 Hyperliquid public-only BTC sample。
- 首次 sandbox 内采样因 DNS/name resolution 失败，保留空 artifact 证据到 `local_live_analysis/hyperliquid_public_sample_0601T001/failed_attempt_dns_1/`。
- 经网络放行后按同一正式命令重试一次，完成 `1800s` public-only BTC 采样。
- 对正式 raw sample 运行 Hyperliquid raw alignment，生成 top-N sidecar、synthetic joins、npz、metrics 和 acceptance report。
- 初次 alignment 使用默认 `--buffer-size 100000` 时因正式 30min 样本事件数超过预分配空间失败；随后用 `--buffer-size 1000000` 重跑成功。该重跑只扩大 converter 预分配空间，不改变 schema 或数据语义。

collection command：
- `python examples/hyperliquid/hyperliquid_public_sample.py --coin BTC --duration-seconds 1800 --channels l2Book,trades --output-dir local_live_analysis/hyperliquid_public_sample_0601T001 --network mainnet --max-reconnects 3 --task-id 0601T001`

collection result：
- output dir：`local_live_analysis/hyperliquid_public_sample_0601T001/`
- start：`2026-06-02T02:05:19.355040+00:00`
- end：`2026-06-02T02:35:19.448312+00:00`
- requested duration：`1800.0s`
- actual duration：`1800.093269476s`
- task id：`0601T001`
- coin：`BTC`
- network：`mainnet`
- channels：`l2Book,trades`
- message counts：`l2Book=3328`, `trades=4939`, `subscriptionResponse=2`
- subscription ack：`2`, with `l2Book=1`, `trades=1`
- connection attempts：`1`
- reconnect count：`0`
- close reason：`duration_elapsed`
- recovery snapshot count：`1`
- raw sha256：`62ebed4f4cc7a5fc9846f9491f9bd3ae0f06ab5b5f1a766d15aa844c50c1bd4c`
- sha256 reconciliation：`raw.gz`, `raw.sha256`, and `collection_manifest.json.raw_sha256` match.

alignment command：
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip local_live_analysis/hyperliquid_public_sample_0601T001/raw.gz --output-dir local_live_analysis/hyperliquid_public_sample_0601T001/alignment --source-label hyperliquid_lag_venue_public_sample_0601T001 --task-id 0601T001 --collection-manifest local_live_analysis/hyperliquid_public_sample_0601T001/collection_manifest.json --recovery-snapshots local_live_analysis/hyperliquid_public_sample_0601T001/recovery_snapshots.jsonl --buffer-size 1000000`

alignment result：
- output dir：`local_live_analysis/hyperliquid_public_sample_0601T001/alignment/`
- classification：`passes_pricing_research_market_view`
- reason：`market_view_and_recovery_evidence_present`
- raw parse errors：`0`
- raw messages：`8328`
- `l2Book` messages：`3328`
- trade messages：`4939`
- trade events：`24784`
- npz rows：`141327`
- top-N rows：`3328`
- top-N coverage：`1.0`
- synthetic decision count：`3598`
- decision join coverage：`1.0`
- future joins：`0`
- missing joins：`0`
- join age p99 ms：`767.9827462599994`
- l2Book cadence p99 ms：`927.1796087599987`
- event order validation：`passed`
- alignment task id：`0601T001`
- source label：`hyperliquid_lag_venue_public_sample_0601T001`

required outputs：
- `local_live_analysis/hyperliquid_public_sample_0601T001/raw.gz`
- `local_live_analysis/hyperliquid_public_sample_0601T001/raw.sha256`
- `local_live_analysis/hyperliquid_public_sample_0601T001/collection_manifest.json`
- `local_live_analysis/hyperliquid_public_sample_0601T001/recovery_snapshots.jsonl`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/run_manifest.json`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/converter_manifest.json`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/data.npz`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/raw_provenance.csv`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/raw_to_npz_mapping.csv`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/topn_sidecar.csv`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/synthetic_joined_views.csv`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/metrics.json`
- `local_live_analysis/hyperliquid_public_sample_0601T001/alignment/acceptance_report.md`
- `.workflow/reports/0601T001-business.md`

verify：
- `python examples/hyperliquid/hyperliquid_public_sample.py --help`
  - 通过；CLI 暴露 `--task-id TASK_ID`。
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help`
  - 通过。
- `python -m pytest examples/hyperliquid/test_hyperliquid_public_sample.py examples/hyperliquid/test_hyperliquid_raw_alignment.py -q`
  - 通过：`6 passed`。
- `python -m py_compile examples/hyperliquid/hyperliquid_public_sample.py examples/hyperliquid/hyperliquid_raw_alignment.py examples/hyperliquid/test_hyperliquid_public_sample.py examples/hyperliquid/test_hyperliquid_raw_alignment.py`
  - 通过。
- Old-sample alignment preflight:
  - `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip local_live_analysis/hyperliquid_public_sample_0529T004/raw.gz --output-dir /tmp/0601T001_preflight_alignment --task-id 0601T001-preflight --source-label accepted_0529T004_preflight --collection-manifest local_live_analysis/hyperliquid_public_sample_0529T004/collection_manifest.json --recovery-snapshots local_live_analysis/hyperliquid_public_sample_0529T004/recovery_snapshots.jsonl`
  - 通过；classification：`passes_pricing_research_market_view`。
- Formal collection command:
  - 通过，经一次 DNS 失败后网络放行重试成功。
- Formal alignment command:
  - 通过，使用 `--buffer-size 1000000`。
- `git diff --check`
  - 通过。
- `rg -n '0531T002' .workflow/tasks/0601T001.md .workflow/reports/0601T001-business.md examples/hyperliquid/hyperliquid_public_sample.py examples/hyperliquid/test_hyperliquid_public_sample.py local_live_analysis/hyperliquid_public_sample_0601T001/collection_manifest.json local_live_analysis/hyperliquid_public_sample_0601T001/alignment/metrics.json`
  - 无匹配。

done：
- `0601T001` 正式 public-only BTC sample 采集完成。
- Raw provenance、manifest、recovery snapshot、sha256、alignment artifact chain 完整。
- Market-view classification 达到 `passes_pricing_research_market_view`。
- 样本可以作为 `0601T002` 的 Hyperliquid lag-venue state / execution-context 输入。
- Hyperliquid public data 仍只作为 lag-venue state / execution-context evidence，不作为 sole pricing source。
- 未连接 private/account/order endpoints，未实现或触碰 order lifecycle，未运行 strategy live，未做 parameter search/default-on/tiny-live/promotion。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
