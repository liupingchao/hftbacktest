# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0602T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0602T001.md`
- `.workflow/reports/0602T001-business.md`
- `examples/hyperliquid/synchronized_public_collection.py`
- `examples/hyperliquid/test_synchronized_public_collection.py`
- `local_live_analysis/cross_exchange_public_sample_0602T001/**`

action：
- 将 `0602T001` 从草案改为正式 public-data-only 同步采集任务，并在执行完成后更新为 `待验收`。
- 新增 `examples/hyperliquid/synchronized_public_collection.py`：
  - `collect`：同时启动 Binance public raw collector 和 Hyperliquid public collector，随后运行两侧 alignment/sidecar，并写同步 manifest。
  - `collect-binance-public`：只采 Binance USD-M Futures public WebSocket streams 与 public REST depth snapshot，不启动 Binance connector、`live_tick_mm.py` 或 `run_live.sh`。
  - 写入 `sample_manifest.json`、`run_manifest.json`、`synchronization_quality_summary.json`，并显式记录 no-private/no-order/no-strategy/no-parameter/no-promotion flags。
- 新增 focused tests 覆盖命令构造、Binance public stream name、trade normalization、overlap 计算、manifest 写入和 public-only boundary。
- 首次 sandbox 内正式采集因 DNS/name resolution 失败产生空样本；修正脚本使质量不达标返回非零，并经网络权限放行后重跑正式 `1800s` 同步采集。
- Binance sidecar 首次后处理因 `--buffer-size 2000000` 不足失败；将 orchestration 的 Binance sidecar buffer 调整为 `10000000`，对同一 raw sample 本地重跑 Binance sidecar 成功。该修正只影响后处理容量，不改变 raw 数据或策略/connector 语义。

collection command：
- `python examples/hyperliquid/synchronized_public_collection.py collect --duration-seconds 1800 --output-dir local_live_analysis/cross_exchange_public_sample_0602T001 --task-id 0602T001 --clean-output`

collection result：
- output dir：`local_live_analysis/cross_exchange_public_sample_0602T001/`
- synchronized overlap：`1800.105259472s`
- overlap gate：`passes_min_overlap_600s=true`, `passes_target_1800s=true`
- Binance collection window：`2026-06-02T05:31:45.349873+00:00` to `2026-06-02T06:01:45.483100+00:00`
- Hyperliquid collection window：`2026-06-02T05:31:45.348119+00:00` to `2026-06-02T06:01:45.455133+00:00`
- Binance raw sha256：`5ce54984cac6dd18775848eec74806b0a9a5b6f465b2fff4fe22fd6249056ee9`
- Hyperliquid raw sha256：`2c8ccbeb9a7935b0562af3eaef174519b3e6f570a0ad12d41c10644b3ba63459`
- sha256 reconciliation：both raw files match `raw.sha256` and `collection_manifest.json.raw_sha256`.

key metrics：
- Binance public data:
  - `depthUpdate=67210`
  - `trade=135126`
  - `bookTicker=816079`
  - public depth snapshot status：`ok`
  - `npz_row_count=5862530`
  - `top5_row_count=67211`
  - `snapshot_alignment_status=present`
  - `first_valid_update_aligned=true`
  - `depth_pu_mismatch_count=0`
  - `final_data_row_mapping_coverage=1.0`
- Hyperliquid public data:
  - `l2Book=3332`
  - `trades=3311`
  - `trade_event_count=12840`
  - `npz_row_count=105789`
  - `topn_coverage=1.0`
  - `decision_join_coverage=1.0`
  - `future_join_count=0`
  - `missing_join_count=0`
  - classification：`passes_pricing_research_market_view`

required outputs：
- `local_live_analysis/cross_exchange_public_sample_0602T001/sample_manifest.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/run_manifest.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/synchronization_quality_summary.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_public_raw/raw.gz`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_public_raw/raw.sha256`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_public_raw/collection_manifest.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_public_raw/depth_snapshot.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment/data.npz`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment/raw_provenance.csv`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment/raw_to_npz_mapping.csv`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment/top5_sidecar.csv`
- `local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment/metrics.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/hyperliquid_public_sample/raw.gz`
- `local_live_analysis/cross_exchange_public_sample_0602T001/hyperliquid_public_sample/raw.sha256`
- `local_live_analysis/cross_exchange_public_sample_0602T001/hyperliquid_public_sample/collection_manifest.json`
- `local_live_analysis/cross_exchange_public_sample_0602T001/hyperliquid_public_sample/recovery_snapshots.jsonl`
- `local_live_analysis/cross_exchange_public_sample_0602T001/hyperliquid_public_sample/alignment/metrics.json`

verify：
- `python examples/hyperliquid/synchronized_public_collection.py --help`
  - 通过。
- `python examples/hyperliquid/synchronized_public_collection.py collect --help`
  - 通过。
- `python examples/hyperliquid/synchronized_public_collection.py collect-binance-public --help`
  - 通过。
- `python -m pytest examples/hyperliquid/test_synchronized_public_collection.py -q`
  - 通过：`6 passed`。
- `python -m py_compile examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/test_synchronized_public_collection.py`
  - 通过。
- `python examples/hyperliquid/hyperliquid_public_sample.py --help`
  - 通过。
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help`
  - 通过。
- `python -m pytest examples/hyperliquid/test_hyperliquid_public_sample.py examples/hyperliquid/test_hyperliquid_raw_alignment.py -q`
  - 通过：`6 passed`。
- `python examples/binance_tick_mm/pipeline_live_raw.py --help`
  - 通过。
- `python examples/binance_tick_mm/binance_top5_provenance.py --help`
  - 通过。
- `python -m pytest examples/binance_tick_mm/test_pipeline_live_raw.py examples/binance_tick_mm/test_binance_top5_provenance.py -q`
  - 通过：`12 passed`。
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/cross_exchange_public_sample_0602T001/binance_public_raw/raw.gz --out-dir local_live_analysis/cross_exchange_public_sample_0602T001/binance_alignment --sample-id 0602T001 --symbol BTCUSDT --tick-size 0.1 --opt t --buffer-size 10000000`
  - 通过。
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_0602T001/sample_manifest.json`
  - 通过。
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_0602T001/run_manifest.json`
  - 通过。
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_0602T001/synchronization_quality_summary.json`
  - 通过。
- Raw sha256 reconciliation script
  - 通过：Binance and Hyperliquid raw hashes match raw files, sha files, and collection manifests.
- `git diff --check`
  - 通过。

done：
- `0602T001` synchronized public-data sample 已生成，可作为 `0601T002` 的 synchronized data input。
- 输出只证明同步 public market-data artifacts 可用；本任务没有计算或声称 Binance lead / Hyperliquid lag 统计效应成立。
- 未启动 `run_live.sh`，未启动 Binance connector，未运行 `live_tick_mm.py`。
- 未连接 private/account/order endpoints，未实现或触碰 order lifecycle，未运行 strategy live，未做 parameter search/default-on/tiny-live/promotion。

blockers：
- 无。

commit：
- `3cf9745`

提交信息：
- `Add synchronized public collection orchestration`
