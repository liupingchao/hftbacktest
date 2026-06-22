```md
执行线程：
- 业务线程-research

任务ID：
- 0619T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0619T002.md`
- `.workflow/reports/0619T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Re-read the accepted `0618T011` / `0618T012` diagnostics, the blocked `0619T001` QA result, the `0619T001` local artifacts, and the current `hyperliquid_tiny_live_m2_fill_window.py` policy shape.
- Reframed the next M2 route from “generic flow-aware touch join” to a design-only contract centered on:
  - fresh-touch maker quote placement
  - dynamic size by recent visible throughput
  - buy-default side discipline
  - no fixed time-of-day claim until cross-hour evidence exists
- Updated `task_plan.md`, `progress.md`, and `findings.md` with the redesign and later implementation contract.
- Did not change strategy code, did not place orders, did not read credentials, did not refresh remote checkout, and did not rerun final gate.

result：
- Key evidence that forced the redesign:
  - `0618T012` longer public sample still supports `queue_too_deep`, `wrong_side`, and `quote_aging_or_fast_drift`.
  - `0619T001` still allowed a sell candidate at about `120.25x` same-side top depth multiple and a buy candidate at about `199.18x`; neither filled.
  - The buy candidate lost touch and drifted adversely after about `1.01s`, while the sell candidate rested a full `15s` without fill.
  - The remaining skipped states were still crowded at about `329.98x-836.14x`, so the current loop is mostly sampling bad queues.
- Redesign conclusion:
  - next policy must stop treating `120x-200x` same-side top depth multiples as acceptable fill-acquisition states;
  - next policy must stop holding passive quotes for `15s` in M2 fill acquisition;
  - next policy must stop using a fixed `0.00999 BTC` size.
- New quote-placement contract:
  - quote only at touch with `quote_offset_ticks=0`
  - no one-tick-back workaround
  - require fresh-touch entry and conservative queue bands
  - `quality_a`: `<=20x` top-depth multiple, `<=6` top orders, same-side strict-through support, hold `<=3s`
  - `quality_b`: `20x-100x` top-depth multiple, `<=12` top orders, same-side strict-through support plus enough recent touch-trade throughput, hold `<=1s`
  - otherwise skip
- New size contract:
  - hard cap `<=0.005 BTC`
  - computed size `min(bucket_cap, 0.25 * recent_same_side_at_or_through_trade_qty_btc_last_3s, 0.005 BTC)` floored to lot size
  - bucket caps:
    - `quality_a`: `0.005 BTC`
    - `quality_b`: `0.002 BTC`
  - no forced order when recent throughput is too low
- New side contract:
  - default `buy_only`
  - sell stays disabled unless both same-window precheck and a later hour-by-side scorecard materially favor sell
- New time-of-day contract:
  - no fixed allowed hour is justified yet
  - next live execution may only occur inside a current precheck-confirmed micro-window
  - any fixed UTC-hour include/exclude decision requires a later cross-hour public scorecard task
- New loop contract:
  - at most `2` live submissions in a window
  - stop the window when no `quality_a` / `quality_b` candidate appears quickly
  - keep `Alo`, tracked cancel, final-open-orders proof, and T008 ledger fail-closed unchanged

verify：
- `git diff --check` -> passed

done：
- Wrote the M2 redesign as a concrete implementation contract rather than another retry suggestion.
- Captured the required later artifacts:
  - `touch_freshness_matrix.csv`
  - `dynamic_size_decision_matrix.csv`
  - `session_side_eligibility.csv`
  - `time_gate_decision_matrix.csv`
- Kept M2 explicitly blocked: this task provides no live fill, no fee/inventory proof, and no realized PnL proof.

blockers：
- No cross-hour public scorecard exists yet, so time-of-day remains gated.
- No live maker fill exists yet, so M2 remains blocked regardless of this redesign.

commit：
- e8909f4

提交信息：
- 0619 redesign M2 maker entry policy
```
