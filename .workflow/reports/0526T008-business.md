```md
执行线程：
- 测试线程

任务ID：
- 0526T008

状态：
- 执行中

是否进行QA验收：
- 是

QA说明：
- 无；当前先记录 30min live collection 已启动，采集完成并完成后处理后再进入待验收。

files：
- `.workflow/tasks/0526T008.md`
- `.workflow/reports/0526T008-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 按用户要求，以 `0526T002` 为蓝本，新建并直接派发 `0526T008`。
- 用户随后将采集时长从 `60min` 改为 `30min`，本任务已同步改为 30min。
- 已推送可部署 workflow commit 到 origin。
- 已在 `awsserver1` 创建 task-scoped clean detached worktree：
  - `/home/admin/hft_live/worktrees/0526T008-active-minmove-30min-b`
- 已准备 remote run dir：
  - `/home/admin/hft_live/runs/5-26-active-minmove-control-30min-b`
- baseline config source：
  - `/home/admin/hft_live/runs/5-26-active-minmove-control-60min-a/config_live.toml`
  - `/home/admin/hft_live/runs/5-26-active-minmove-control-60min-a/binancefutures.toml`
- allowed config changes：
  - run id/path 从 `5-26-active-minmove-control-60min-a` 替换为 `5-26-active-minmove-control-30min-b`
  - stopper duration 设置为 `1800s`
  - 未修改 quote/risk/guard/fair/impact/order sizing/API 参数
- 已启动 current-format no-rule / default-off 30min live collection。

collection：
- run id: `5-26-active-minmove-control-30min-b`
- deployed commit: `ff6f1b7`
- remote worktree: `/home/admin/hft_live/worktrees/0526T008-active-minmove-30min-b`
- remote run dir: `/home/admin/hft_live/runs/5-26-active-minmove-control-30min-b`
- Python: `/home/admin/hft_live/venv/bin/python`
- tmux session: `hft_live`
- stopper pid: `439227`
- start marker UTC: `2026-05-26T14:32:55Z`
- expected stop UTC: `2026-05-26T15:02:55Z`
- expected stop CST: `2026-05-26 23:02:55 CST`

preflight / early checks：
- local focused checks passed before deploy:
  - `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> `5 passed`
  - `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py` -> `8 passed`
  - `python examples/binance_tick_mm/deploy/preflight_live_run.py --help`
  - `python examples/binance_tick_mm/align_live_run.py --help`
  - `python examples/binance_tick_mm/maker_acceptance.py --help`
  - `python examples/binance_tick_mm/quote_adjustment_replay.py --help`
  - `python examples/binance_tick_mm/candidate_bucket_refinement.py --help`
  - `bash -n examples/binance_tick_mm/deploy/run_live.sh`
- remote process check before start:
  - no existing `tmux` live session
  - no existing live bot / connector / collector process
- remote preflight passed:
  - commit `ff6f1b7`
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
- Initial dispatch / start verification complete.
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- `0526T008` 已派发并开始执行。
- 当前还没有完成 30min 采集、拉回、archive、audit replay 或 Stage 5/5C/6/9 后处理。
- 本任务仍保持 no-rule / default-off control data 边界：未放宽 guard，未启用 candidate，未做 parameter sweep，未授权 live/default-on/promotion。

blockers：
- 无当前执行 blocker。

commit：
- 待提交

提交信息：
- 待提交
```
