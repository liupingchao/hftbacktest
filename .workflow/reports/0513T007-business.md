```md
执行线程：
- 业务线程-python

任务ID：
- 0513T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0513T006.md`
- `.workflow/tasks/0513T007.md`
- `.workflow/reports/0513T006-qa.md`
- `.workflow/reports/0513T007-business.md`
- `docs/qa-acceptance-report.md`
- `docs/binance_top5_provenance.md`
- `examples/binance_tick_mm/binance_top5_provenance.py`
- `examples/binance_tick_mm/test_binance_top5_provenance.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/0513T007_provenance_sidecar_smoke/**`

action：
- 使用 `qa_task.py` 验收通过 `0513T006`，结论写入 `.workflow/reports/0513T006-qa.md` 和 `docs/qa-acceptance-report.md`。
- 解阻并启动 `0513T007`。
- 新增 standalone Binance top5 provenance 工具 `examples/binance_tick_mm/binance_top5_provenance.py`。
- 新增 focused tests `examples/binance_tick_mm/test_binance_top5_provenance.py`。
- 新增文档 `docs/binance_top5_provenance.md`。
- 生成 bounded smoke artifacts 到 `local_live_analysis/0513T007_provenance_sidecar_smoke/`。

verify：
- `python -m pytest examples/binance_tick_mm/test_pipeline_live_raw.py`
  - 通过：`4 passed`
- `python -m pytest examples/binance_tick_mm/test_binance_top5_provenance.py`
  - 通过：`4 passed`
- `python examples/binance_tick_mm/binance_top5_provenance.py --help`
  - 通过
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --help`
  - 通过
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --help`
  - 通过
- Smoke artifact 存在性检查：
  - `raw_provenance.csv`
  - `raw_to_npz_mapping.csv`
  - `top5_sidecar.csv`
  - `joined_decisions.csv`
  - `sidecar_manifest.json`
  - `metrics.json`
- `python3 .workflow/build_dashboard.py`
  - 通过，刷新 dashboard 与 dispatch suggestions。

done：
- T007 保留标准 hftbacktest `data` npz 主事件 schema 不变，未修改 core `Event`、py `event_dtype`、connector、本地 `hbt.depth(0)` API、strategy 行为、`align_live_run.py`、canonical `audit_schema.py` 或 live audit CSV schema。
- 新增 sidecar / mapping / join 工具：
  - `raw_provenance.csv`：message-level `raw_seq`、depth `U/u/pu`、snapshot `lastUpdateId`、bookTicker 字段、最终 npz row metadata。
  - `raw_to_npz_mapping.csv`：`raw_seq -> final standard data row indices`，mapping 指向 `correct_local_timestamp()` / `correct_event_order()` 后的最终 `data` row。
  - `top5_sidecar.csv`：reconstructed top5 px/ticks/qtys、snapshot sync 状态、`last_u`/`pu`、bookTicker/depth BBO match、source timestamps。
  - `joined_decisions.csv`：read-only as-of decision join diagnostic artifact。
  - `sidecar_manifest.json`：schema version、raw file identity、converter opt、top5 levels、tick_size、generated_at、sample id。
- Smoke metrics：
  - raw message count：`5000`
  - npz row count：`15062`
  - final `data` row mapping coverage after order correction：`1.0`
  - raw messages with npz rows：`644`
  - raw message mapping coverage：`0.1288`，主要因为 bookTicker 默认不写入 standard npz，按设计 `row_count=0`。
  - depth `pu` mismatch count：`0`
  - snapshot alignment status：`present`
  - first valid update alignment：`false`
  - top5 row count：`156`
  - bookTicker/depth BBO match/mismatch：`151 / 1`
- Decision join smoke：
  - decision count：`244`
  - joined decision count：`244`
  - decision join coverage：`1.0`
  - future join count：`0`
  - join missing count：`0`
  - stale join count：`243`
  - gap-crossed join count：`244`
  - top5 join age p50/p90/p99 ms：`1065.327 / 1090.839 / 1096.596`
- Known limitation：
  - Smoke 使用 `5-13-day-control-15min` raw gzip 的前 `5000` 条 message，是 bounded slice，不是完整 run-level book sync 验收。
  - `first_valid_update_aligned=false`、join stale/gap-crossed 高，说明该切片从 raw stream 中间开始或 snapshot/update 对齐不足；T007 工具正确暴露了不可用状态，没有静默伪造可用 top5 join。
  - 全样本重跑、正式 sample usability 重新分类、`align_live_run.py` 集成、正式 audit schema 字段升级、market-view acceptance gate 都需要后续任务。
- 后续研究边界：
  - T007 只支持 top5-only provenance 和 top5 pricing / top5 OFI proxy / top5 microprice proxy 的后续基础数据检查。
  - T007 不能证明 full L2 equivalence、exact queue position、策略 PnL 或 live promotion。

blockers：
- 无。

commit：
- cb4aabe

提交信息：
- feat(binance): add top5 provenance sidecars
```
