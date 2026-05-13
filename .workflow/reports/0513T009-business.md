```md
执行线程：
- 业务线程-python

任务ID：
- 0513T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0513T009.md`
- `.workflow/reports/0513T009-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `docs/binance_top5_provenance.md`
- `examples/binance_tick_mm/binance_top5_provenance.py`
- `examples/binance_tick_mm/test_binance_top5_provenance.py`
- `local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/**`
- `local_live_analysis/5-13-day-control-30min/t009_sample_classification.md`

action：
- 修复 standalone top5 sidecar 的 Binance snapshot/bootstrap 逻辑。
- 在 snapshot 前缓存 depthUpdate metadata 和 top-level deltas。
- snapshot 到达后，从 buffered depthUpdate 中选择第一条满足 `U <= lastUpdateId + 1 <= u` 的 update，从该 update 开始应用 deltas。
- buffered replay 行的 `local_ts` 使用 snapshot 到达时间，保证 decision join 仍是 as-of only，不让 snapshot 后才可用的重放 book 在 snapshot 前被 decision 行看到。
- 后续 depthUpdate 继续按 `pu == previous_u` 校验；断链时保留显式 `sync_gap=true`。
- `top5_sidecar.csv` 保留 `raw_seq`、`depth_U`、`depth_u`、`pu`、`snapshot_lastUpdateId`、sync flags、startup flags、first-valid alignment、bookTicker/depth BBO match/age。
- 使用既有 `5-13-day-control-30min` raw gzip 全量重跑 T009 fixed sidecar 和 decision join。
- 写入 `local_live_analysis/5-13-day-control-30min/t009_sample_classification.md`。
- 刷新 workflow dashboard。

verify：
- `python -m pytest examples/binance_tick_mm/test_binance_top5_provenance.py`
  - 通过：`8 passed`
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/5-13-day-control-30min/raw_market_data/btcusdt_20260513.gz --out-dir local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar --sample-id 5-13-day-control-30min-t009 --symbol BTCUSDT --tick-size 0.1 --buffer-size 8000000`
  - 通过，完整 raw gzip，无 `--max-messages` smoke。
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --audit-csv local_live_analysis/5-13-day-control-30min/audit_live_5-13-day-control-30min.csv --top5-csv local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/top5_sidecar.csv --out-csv local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/joined_decisions.csv --max-age-ms 250`
  - 通过。
- `python3 .workflow/build_dashboard.py`
  - 通过。

T009 fixed sidecar metrics：
- raw message count：`461402`
- npz row count：`2418372`
- raw messages with npz rows：`105579`
- raw message mapping coverage：`0.22882215508385312`，主要因为 bookTicker 默认不写入 standard npz。
- final `data` row mapping coverage：`1.0`
- standard npz dtype：`ev|exch_ts|local_ts|px|qty|order_id|ival|fval`，未改变。
- depth `pu` mismatch count：`0`
- snapshot alignment status：`present`
- first valid update aligned：`true`
- top5 row count：`67322`
- bookTicker/depth BBO match/mismatch：`67307 / 11`

Bootstrap evidence：
- snapshot `raw_seq=6`，`lastUpdateId=10537138804218`。
- buffered replay first valid update：`raw_seq=5`，`depth_U=10537138802913`，`depth_u=10537138805036`，覆盖 `lastUpdateId + 1 = 10537138804219`。
- snapshot 后第一条 future depth：`raw_seq=7`，`pu=10537138805036`，接上 buffered `raw_seq=5` 的 `u`。
- full-run `sync_gap=true` rows：`0`。

T009 decision join metrics：
- decision count：`47499`
- joined decision count：`47499`
- decision join coverage：`1.0`
- future join count：`0`
- missing join count：`0`
- stale join count：`432`
- gap-crossed join count：`0`
- top5 join age p50/p90/p99 ms：`13.602237 / 24.205795400000003 / 28.13446387999999`
- max top5 join age ms：`251.033249`
- max bookTicker join age ms：`603.705003`
- stale rows are driven by `max_join_age_ms > 250`, mostly bookTicker age; top5/depth join age remains low for p99.

done：
- T007 QA 发现的 snapshot/bootstrap bug 已修复：sidecar 不再把 snapshot 后 `raw_seq=7` 错当 first valid update，而是在 snapshot 到达后重放 buffered `raw_seq=5`，再让 `raw_seq=7` 按 `pu` 接续。
- Focused tests 覆盖：
  - snapshot 前 buffered update 覆盖 `lastUpdateId+1` 后 `first_valid_update_aligned=true`。
  - first valid update 来自 snapshot 前 buffer 后，snapshot 后下一条 `pu` 可接上 buffered update 的 `u`。
  - buffer 中没有覆盖 update 时仍显式 `sync_gap=true`。
  - decision join 不使用未来值，`future_join_count=0`。
- `5-13-day-control-30min` 样本分类升级：可作为后续 top5 microprice / top5 OFI proxy / top5 imbalance pricing research 的 candidate。
- 分类边界仍保留：这不是 full L2 equivalence，不是 exact queue position，不是 queue-fill model proof，不是策略收益证明。
- 没有启动 live，没有重新采集，没有修改策略行为，没有修改 core/connector/API，没有修改 standard npz 主事件 schema。

blockers：
- 无。

commit：
- 6710b5f

提交信息：
- fix(binance): replay buffered depth bootstrap
```
