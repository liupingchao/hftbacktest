# 线程回报

执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0730T015

状态：
- 待验收

是否进行QA验收：
- 是

files：
- `.workflow/tasks/0730T015.md`
- `.workflow/reports/0730T015-business.md`
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_alignment_acceptance.py`
- `local_live_analysis/skhynix_cross_exchange_research_0730T013/alignment/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 以 Binance bookTicker bid/ask price change 作为 decision event。
- 所有 as-of 和 response join 使用同机 local receipt timestamp。
- Primary response label 使用目标时间之后第一条 Hyperliquid BBO，并执行
  frozen tolerance。
- 输出每段 freshness tier、三类 top-of-book reconciliation、八个
  horizon coverage 和 segment/mask 质量。
- Clean rebuild 使用 temporary + backup/rollback，失败保留旧输出。

verify：
- Unit/failure-injection tests: `8 passed`.
- `py_compile`, CLI `--help`, `git diff --check` pass.
- Real eight-segment run: `passes=true`.
- Decision events: `2,366,631`.
- Eligible after segment warmup: `2,366,629`; warmup excluded: `2`.
- Timestamp regressions/future joins/cross-segment labels: `0/0/0`.
- Masks: `8` segment epochs + `2` auxiliary degraded intervals.
- Accepted primary horizons:
  - `1000ms`: per-segment coverage `95.56%` to `97.25%`
  - `2000ms`: per-segment coverage `95.26%` to `97.03%`
- Diagnostic-only horizons:
  - `10-50ms`: minimum per-segment coverage about `40%`
  - `100-500ms`: minimum per-segment coverage about `68%`
- Aggregate primary freshness:
  - Hyperliquid BBO `96.46%` at `<=250ms`
  - fast L2 `90.01%` at `<=500ms`
  - standard L2 `55.82%` at `<=3000ms`
- Public feeds are asynchronous; exact top match is therefore reported
  together with p50/p99/max price-distance distributions, not treated as a
  tick-to-order or exact simultaneity claim.

done：
- R1 outputs are complete and `passes=true`.
- Only `1000ms` and `2000ms` are unlocked for primary R2/R3 research.
- Shorter horizons remain diagnostic and cannot support primary claims.
- T015 enters `待验收`.

boundary：
- Existing local data only.
- No network/AWS/SSH/new collection/order endpoint.
- No basis, lead-lag or maker model fitting.
- New collection remains authorization-gated.

commit：
- 无
