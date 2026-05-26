执行线程：
- 测试线程

任务ID：
- 0526T007

状态：
- 执行中

是否进行QA验收：
- 是

QA说明：
- 无；当前先记录 180min live collection 已启动，采集完成并完成后处理后再进入待验收。

files：
- `.workflow/tasks/0526T007.md`
- `.workflow/reports/0526T007-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新建并直接派发 `0526T007`。
- 已推送可部署 commit 到 origin。
- 已在 `awsserver1` 创建 task-scoped clean worktree：
  - `/home/admin/hft_live/worktrees/0526T007-makeredge-180min`
- 已准备 remote run dir：
  - `/home/admin/hft_live/runs/5-26-active-makeredge-control-180min-a`
- baseline config source：
  - `/home/admin/hft_live/runs/5-26-active-minmove-control-60min-a/config_live.toml`
  - `/home/admin/hft_live/runs/5-26-active-minmove-control-60min-a/binancefutures.toml`
- allowed config changes：
  - run id/path 从 `5-26-active-minmove-control-60min-a` 替换为 `5-26-active-makeredge-control-180min-a`
  - stopper duration 设置为 `10800s`
  - 未修改 quote/risk/guard/fair/impact/order sizing/API 参数
- 已启动 current-format no-rule / default-off 180min live collection。

collection：
- run id: `5-26-active-makeredge-control-180min-a`
- deployed commit: `43fb586`
- remote worktree: `/home/admin/hft_live/worktrees/0526T007-makeredge-180min`
- remote run dir: `/home/admin/hft_live/runs/5-26-active-makeredge-control-180min-a`
- Python: `/home/admin/hft_live/venv/bin/python`
- tmux session: `hft_live`
- stopper pid: `438109`
- start marker UTC: `2026-05-26T11:01:51Z`
- expected stop UTC: `2026-05-26T14:02:11Z`
- expected stop CST: `2026-05-26 22:02:11 CST`

preflight / early checks：
- remote preflight passed:
  - commit `43fb586`
  - git dirty `false`
  - compatibility passed `true`
  - audit field count `159`
- early audit header check:
  - audit csv exists: `true`
  - field count: `159`
  - missing T006 fields: `[]`
- live process status after start:
  - tmux `hft_live` exists
  - stopper process is running
  - audit csv already has rows
  - raw gzip is being written

verify：
- Local pre-deploy checks passed:
  - `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> `5 passed`
  - `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py` -> `8 passed`
  - `python examples/binance_tick_mm/deploy/preflight_live_run.py --help`
  - `python examples/binance_tick_mm/align_live_run.py --help`
  - `python examples/binance_tick_mm/maker_acceptance.py --help`
  - `python examples/binance_tick_mm/quote_adjustment_replay.py --help`
  - `python examples/binance_tick_mm/candidate_bucket_refinement.py --help`
  - `bash -n examples/binance_tick_mm/deploy/run_live.sh`

done：
- 0526T007 已派发并开始执行。
- 当前还没有完成 180min 采集、拉回、archive、audit replay 或 Stage 5/5C/6/9 后处理。
- 本任务仍保持 no-rule / default-off control data 边界：未放宽 guard，未启用 candidate，未做 parameter sweep，未授权 live/default-on/promotion。

blockers：
- 无当前执行 blocker。
- 初次启动时使用系统 `python3` 触发 preflight import failure，因为缺少 `numpy`；该失败发生在 tmux/live 启动前。已改用 `/home/admin/hft_live/venv/bin/python` 重新启动并通过 preflight。

commit：
- 43fb586

提交信息：
- docs(workflow): refresh dashboard for 0526T007 dispatch
