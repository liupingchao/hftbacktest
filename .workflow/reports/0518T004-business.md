```md
执行线程：
- 业务线程-python

任务ID：
- 0518T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0518T004.md`
- `.workflow/reports/0518T004-business.md`
- `examples/binance_tick_mm/quote_anchor_safety.py`
- `examples/binance_tick_mm/test_quote_anchor_safety.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

generated outputs：
- `local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004/quote_anchor_safety_summary.md`
- `local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004/quote_anchor_safety_counters.csv`
- `local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004/quote_anchor_safety_rows.csv`
- `local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004/quote_anchor_safety_changed_rows.csv`
- `local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004/run_manifest.json`

action：
- 新增 `quote_anchor_safety.py`，实现 default-off / diagnostic-first quote-anchor safety helper 和 Stage 5C diagnostic runner。
- 新增 focused tests `test_quote_anchor_safety.py`。
- 在 `backtest_tick_mm.py` 和 `live_tick_mm.py` 中接入默认关闭的 `quote_anchor_safety` 配置。
- 配置未显式开启时，target ticks 和 action path 原样保留。
- 启用后只允许：
  - bookTicker-equivalent fresh anchor 优先
  - guarded depth fallback
  - bid floor / ask ceil side-conservative rounding
  - anchor clamp
  - post-clamp post-only re-check
  - stale/missing anchor suppress fresh add-side submit
  - diagnostic counters
- 没有修复 audit_depth/bookTicker/top5 row-exact drift。
- 没有把 top5 提升为 final hard anchor。
- 没有修改 fair/reservation、replay lifecycle、standard schema、live scripts 或 AWS/remote state。
- 没有启动 live，没有 default-on，没有 promotion。

Stage 5C diagnostic：
- output dir：`local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004`
- decision rows：`47499`
- bookTicker anchor rows：`39261`
- guarded depth fallback rows：`8173`
- stale anchor rows：`65`
- missing anchor rows：`0`
- bid clamped rows：`1394`
- ask clamped rows：`2281`
- suppress buy / sell rows：`65 / 65`
- post-only risk after re-check rows：`0`

verify：
- `python -m pytest examples/binance_tick_mm/test_quote_anchor_safety.py`
  - `4 passed`
- `python -m pytest examples/binance_tick_mm/test_quote_anchor_diagnostic.py`
  - `1 passed`
- `python examples/binance_tick_mm/quote_anchor_safety.py --help`
  - passed
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py`
  - `136 passed`
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage5c_quote_anchor_safety_0518T004`
  - returned `status: ok`
- `git diff --check`
  - passed

done：
- 默认行为保持不变：`quote_anchor_safety.enabled` 默认为 false；现有 `test_backtest_tick_mm.py` 全量通过。
- anchor arbitration、rounding、clamp、re-check、guarded fallback、stale/join-age suppression 实现在 `examples/binance_tick_mm/quote_anchor_safety.py`。
- backtest/live 接入点只在显式开启配置后应用 safety result。
- top5 仍不是 final hard anchor。
- 没有修复 audit_depth/bookTicker/top5 row-exact drift。
- 没有改 fair/reservation、没有改 replay lifecycle、没有启动 live、没有 default-on。
- 已生成 Stage 5C diagnostic 产物。

blockers：
- 无

commit：
- b931c96

提交信息：
- feat(binance-mm): add quote anchor safety layer
```
