```md
执行线程：
- 测试线程

任务ID：
- 0526T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 已完成 30min current-format no-rule/default-off live collection、artifact pull、raw recovery、audit replay、maker acceptance、T009 sidecar/join、Stage 5/5C/6/9B/9D 后处理和归档。

files：
- `.workflow/tasks/0526T008.md`
- `.workflow/reports/0526T008-business.md`
- `task_plan.md`
- `local_live_analysis/5-26-active-minmove-control-30min-b/**`
- `local_live_analysis/archive/5-26-active-minmove-control-30min-b.tar.gz`
- `local_live_analysis/archive/5-26-active-minmove-control-30min-b.tar.gz.sha256`

action：
- 按用户要求以 `0526T002` 为蓝本创建并直接派发 `0526T008`。
- 用户随后将采集时长从 `60min` 改为 `30min`，本任务按 30min 执行。
- 已在 `awsserver1` 使用 task-scoped clean worktree 执行 live run。
- 已拉回本地并完成既有后处理链路。

collection：
- run id: `5-26-active-minmove-control-30min-b`
- deployed commit: `ff6f1b7`
- remote worktree: `/home/admin/hft_live/worktrees/0526T008-active-minmove-30min-b`
- remote run dir: `/home/admin/hft_live/runs/5-26-active-minmove-control-30min-b`
- baseline config source: `/home/admin/hft_live/runs/5-26-active-minmove-control-60min-a/config_live.toml`
- allowed config diff: only run id/path and stopper duration changed to `1800s`
- start marker UTC: `2026-05-26T14:32:55Z`
- stop marker UTC: `2026-05-26T15:02:55Z`
- stop marker exit code: `0`
- boundary: no guard relaxation, no candidate enablement, no strategy/default change, no parameter sweep, no tiny-live/promotion authorization.

preflight / header：
- local focused checks passed before deploy:
  - `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> `5 passed`
  - `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py` -> `8 passed`
  - `python examples/binance_tick_mm/deploy/preflight_live_run.py --help`
  - `python examples/binance_tick_mm/align_live_run.py --help`
  - `python examples/binance_tick_mm/maker_acceptance.py --help`
  - `python examples/binance_tick_mm/quote_adjustment_replay.py --help`
  - `python examples/binance_tick_mm/candidate_bucket_refinement.py --help`
  - `bash -n examples/binance_tick_mm/deploy/run_live.sh`
- remote preflight passed:
  - commit `ff6f1b7`
  - git dirty `false`
  - compatibility passed `true`
  - audit field count `159`
- early audit header:
  - missing T006 fields: `[]`

raw / replay caveat：
- Remote raw gzip was truncated at tail; original local copy is preserved as `raw_market_data/btcusdt_20260526.gz.corrupt`.
- Recovered streamable gzip passes `gzip -t`: `raw_market_data/btcusdt_20260526.gz`.
- Full audit replay generated a 21GB `audit_bt_audit_replay.full.csv` because replay emitted repeated `cancel_ack` lifecycle rows.
- Full-event `compare_audit.py` over that 21GB replay CSV was killed by memory pressure.
- `alignment_report_audit_replay.json` was produced from decision-only extracted audit CSVs.
- Stage 6 calibration used a lifecycle-minimized replay audit at canonical path `out/backtest_audit_replay/audit_bt_audit_replay.csv`; the full original is preserved as `audit_bt_audit_replay.full.csv`.
- Boundary note: this is a postprocessing input-size workaround only; it did not change strategy behavior or replay decision semantics. Details are in `replay_input_note_0526T008.md`.

acceptance：
- `maker_acceptance.json`: passed `true`, hard failures `[]`.
- `maker_acceptance_with_market_view.json`: passed `true`, hard failures `[]`.
- common rows: `340657`
- action match: `1.0`
- planned action match: `1.0`
- reject reason match: `1.0`
- throttle reason match: `1.0`
- working lifecycle semantic mismatch rows: `0`
- working lifecycle blocking mismatch rows: `0`
- strict replay lag gate: passed `true`, breach/drop/fail `0`
- audit replay scheduled/consumed/unconsumed: `340731 / 340657 / 74`

T009 sidecar / market-view：
- sidecar raw messages: `2073983`
- top5 rows: `67781`
- final data row mapping coverage: `1.0`
- depth pu mismatch count: `0`
- bookTicker/depth BBO mismatch count: `6`
- decision join coverage: `1.0`
- joined decisions: `340731`
- future join count: `0`
- gap crossed join count: `0`
- join missing count: `0`
- stale join count: `76`
- top5 join age p50/p90/p99 ms: `13.577669 / 23.795169 / 26.8948756`
- market-view classification: `passes_pricing_research_market_view`
- top5 tick match: `0.9704893778786287`
- top5 qty match: `0.9474662196872514`

Stage 5 labels：
- submit orders: `11089`
- filled orders: `471`
- fill-after-cancel orders: `246`
- partial fill orders: `0`
- missing lifecycle orders: `1`
- fill by 100/500/1000/5000ms: `149 / 361 / 416 / 470`
- markout observable 100/500/1000/5000ms: `467 / 463 / 464 / 460`

Step 5C：
- decision rows: `340731`
- bookTicker anchor rows: `329579`
- depth fallback rows: `11152`
- missing anchor rows: `0`
- stale anchor rows: `0`
- post-only risk after recheck rows: `0`
- bid/ask clamped rows: `46569 / 31772`

Stage 6：
- output: `stage6_execution_calibration_0526T008`
- decision state: `methodology_valid_single_sample`
- live/replay submit orders: `11089 / 11084`
- matched submit orders: `11084`
- price tick equality on matched submits: `11084/11084`
- qty equality on matched submits: `11084/11084`
- live/replay filled orders: `471 / 480`
- live/replay fill-after-cancel orders: `246 / 249`
- aligned enough:
  - submit-key coverage `11084/11089`
  - fill@100/500/1000/5000ms gaps `0.0073 / 0.0022 / 0.0002 / 0.0003`
  - final state filled gap `0.0036`
  - final state canceled gap `0.0007`
  - fill-after-cancel rate gap `0.0003`
- not aligned: `none` in Stage 6 summary.

Step 9B / 9D：
- Step 9B output: `stage9b_quote_adjustment_replay_0526T008`
- Step 9B classification: `promising_but_single_sample`
- Step 9B candidate count: `8`
- Step 9B reason: active candidates have nonzero coverage, but only one current-format sample is available.
- Step 9D output: `stage9d_candidate_bucket_refinement_0526T008`
- Step 9D sample ids: `5-26-active-minmove-control-30min-b`
- Step 9D stable promising bucket count: `0`
- Step 9D candidate ids:
  - `min_move_quote_age_churn_guard`
  - `inventory_reservation_shift_band`
  - `size_reduction_or_add_side_suppression_pressure`

archive：
- archive: `local_live_analysis/archive/5-26-active-minmove-control-30min-b.tar.gz`
- archive size: `446M`
- sha256: `90cb6279ddce0913d04567245b6a29ab49bf2fe580479faf8615046145f9926a`
- archive excludes the preserved 21GB full replay CSV and debug-only `slim/dedup` intermediates; it includes canonical lifecycle-min replay CSV, raw/replay summaries, T009, Stage 5/5C/6/9 outputs, and `replay_input_note_0526T008.md`.

verify：
- `gzip -t local_live_analysis/5-26-active-minmove-control-30min-b/raw_market_data/btcusdt_20260526.gz`
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars ...`
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions ...`
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report ... --backtest-result ... --out .../maker_acceptance.json`
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report ... --backtest-result ... --sidecar-metrics ... --joined-decision-metrics ... --top5-sidecar-csv ... --joined-decisions-csv ... --out .../maker_acceptance_with_market_view.json`
- `python examples/binance_tick_mm/execution_outcome_labels.py --run-dir local_live_analysis/5-26-active-minmove-control-30min-b`
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir ... --output-dir .../stage5c_quote_anchor_safety_0518T004`
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir ... --output-dir .../stage5c_quote_anchor_safety_0526T008`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir ... --output-dir .../stage6_execution_calibration_0526T008`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --run-dir ... --output-dir .../stage9b_quote_adjustment_replay_0526T008`
- `python examples/binance_tick_mm/candidate_bucket_refinement.py --run-dir ... --output-dir .../stage9d_candidate_bucket_refinement_0526T008`
- `sha256sum local_live_analysis/archive/5-26-active-minmove-control-30min-b.tar.gz`

done：
- `0526T008` 采集和后处理完成，状态切为 `待验收`。
- 当前样本是 current-format no-rule/default-off control data only。
- 不授权 live/default-on/promotion。

blockers：
- 无阻塞。主要 caveat 是 full replay audit lifecycle rows 过大，需要 lifecycle-min postprocessing 输入才能完成 Stage 6。

commit：
- 待提交

提交信息：
- 待提交
```
