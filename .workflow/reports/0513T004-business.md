```md
执行线程：
- 业务线程-python

任务ID：
- 0513T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 live 启动前 preflight 是否能记录 deployment manifest，并在 `AUDIT_FIELDS` 与 `strategy_core.build_audit_row(...)` / lifecycle row 不兼容时失败；确认未启动 live、未改策略语义。

files：
- .workflow/tasks/0513T004.md
- .workflow/reports/0513T004-business.md
- examples/binance_tick_mm/deploy/preflight_live_run.py
- examples/binance_tick_mm/deploy/run_live.sh
- examples/binance_tick_mm/test_deploy_preflight.py
- .workflow/dashboard.html
- .workflow/dispatch_suggestions.md

action：
- 新增 `examples/binance_tick_mm/deploy/preflight_live_run.py`。
- preflight 会在 live 启动前读取 config / connector config，记录 git commit、branch、dirty status、config hash、connector config hash、`audit_schema.py` hash、`strategy_core.py` hash、`live_tick_mm.py` hash、`run_live.sh` hash、preflight 脚本 hash、symbol、data dir、run dir、manifest path、Python 环境。
- preflight 会导入当前 `strategy_core`，构造代表性 decision row 和 lifecycle row，并验证它们的 keys 与最终 `AUDIT_FIELDS` 完全一致；字段重复、缺失、额外字段都会 `PREFLIGHT FAILED` 并返回非零。
- 修改 `examples/binance_tick_mm/deploy/run_live.sh`，在 tmux/collector/connector/live bot 启动前执行 preflight；preflight 失败时不会启动 tmux。
- `run_live.sh` 默认写入：
  - `deployment_manifest.json`
  - `start_marker.json`
  - `stop_marker.json`
- start marker 在 preflight 通过后、tmux 启动前写入；stop marker 由 live bot pane 退出时通过 `--write-stop-marker-only` 写入。
- 增加 `examples/binance_tick_mm/test_deploy_preflight.py`，覆盖当前 schema 通过、缺字段失败、重复字段失败、manifest/start marker 输出、stop marker 输出。

verify：
- `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> 5 passed。
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --help` -> exit 0。
- `bash -n examples/binance_tick_mm/deploy/run_live.sh` -> exit 0。
- `python -m py_compile examples/binance_tick_mm/deploy/preflight_live_run.py examples/binance_tick_mm/test_deploy_preflight.py` -> exit 0。
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --project-root /home/molly/project/hftbacktest --config examples/binance_tick_mm/config.example.toml --connector-config examples/binance_tick_mm/deploy/binancefutures.toml --symbol BTCUSDT --data-dir /tmp/hft_t004_preflight/data --run-dir /tmp/hft_t004_preflight --manifest-out /tmp/hft_t004_preflight/deployment_manifest.json --start-marker-out /tmp/hft_t004_preflight/start_marker.json --stop-marker-out /tmp/hft_t004_preflight/stop_marker.json` -> exit 0，输出 `preflight passed: commit=192470f schema_fields=144 manifest=/tmp/hft_t004_preflight/deployment_manifest.json`。
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --write-stop-marker-only --manifest-in /tmp/hft_t004_preflight/deployment_manifest.json --stop-marker-out /tmp/hft_t004_preflight/stop_marker.json --exit-code 0` -> exit 0。
- `/tmp/hft_t004_preflight/deployment_manifest.json` 检查结果：
  - `decision=preflight_passed`
  - `git.short_commit=192470f`
  - `compatibility.audit_field_count=144`
  - manifest hashes 包含 `config`、`connector_config`、`audit_schema`、`strategy_core`、`live_tick_mm`、`run_live`、`preflight_live_run`
  - decision row compatibility：`missing=[]`、`extra=[]`、`duplicates=[]`、`row_key_count=144`
- `python3 .workflow/build_dashboard.py` -> exit 0。

done：
- T004 完成了十步计划 Step 1 的本地启动门禁：每次通过 `run_live.sh` 启动前，会先生成可追溯 manifest，并证明当前 code/schema/config 组合可写出兼容 audit row。
- T003 暴露的 stale `audit_schema.py` 风险现在会在启动前被发现：如果 `strategy_core.build_audit_row(...)` 生成了 `AUDIT_FIELDS` 未声明字段，preflight 会非零退出，tmux/live 不会启动。
- 本任务未启动 live，未修改 AWS，未改 strategy rule、fair/target、risk guard、quote placement、core API、converter/npz schema 或 connector book management。
- 本任务不证明策略收益，不证明 market-view/full L2 对齐，只提供部署可复现和启动兼容性门禁。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
