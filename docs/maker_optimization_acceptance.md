# Maker Optimization Acceptance Contract

本 contract 定义进入 maker 参数优化前必须满足的 live/backtest 对齐门槛。目标不是让所有诊断字段完全一致，而是确保会污染优化目标的语义状态、计划动作、API/throttle、replay lag 已经对齐。

## 输入文件

每个候选 live sample 必须先完成 `align_live_run.py`，并至少生成：

- `alignment_report_audit_replay.json`
- `backtest_audit_replay_result.json`
- `live_alignment_summary.md`

验收命令：

```bash
python examples/binance_tick_mm/maker_acceptance.py \
  --alignment-report local_live_analysis/<RUN_ID>/alignment_report_audit_replay.json \
  --backtest-result local_live_analysis/<RUN_ID>/backtest_audit_replay_result.json \
  --out local_live_analysis/<RUN_ID>/maker_acceptance.json
```

命令 exit code 为 `0` 才能进入 maker 参数优化；exit code 为 `1` 时必须先修复或解释失败项。

注意：`audit_replay` 可以使用 audit-only overlays 来验证 live/backtest
action-path parity。Stage 6G 后，`working_order_overlay = "audit"` 用于验收
replay，以重建 live 决策可见 working orders 和 hidden in-flight lifecycle
ordering。这个 overlay 不能用于参数优化 sweep；优化 replay 必须关闭
`market_state_overlay`、`strategy_position_overlay` 和 `working_order_overlay`。

## Hard Gates

以下条件全部是硬门槛：

- `alignment.common_rows >= 1`
- `alignment.action_match_rate == 1.0`
- `alignment.planned_action_match_rate == 1.0`
- `alignment.reject_reason_match_rate == 1.0`
- `alignment.throttle_reason_match_rate == 1.0`
- `alignment.working_order_lifecycle.semantic_mismatch_rows == 0`
- `alignment.working_order_lifecycle.blocking_mismatch_rows == 0`
- `alignment.api_throttle.mismatch_attribution.mismatch_rows == 0`
- `alignment.api_throttle.mismatch_attribution.target_tick_mismatch_rows == 0`
- `alignment.replay_lag.missing_lag_rows == 0`
- `alignment.replay_lag.missing_exchange_lag_rows == 0`
- `alignment.replay_lag.stateful_gate.startup_excluded_gate.passed == true`
- `alignment.replay_lag.stateful_gate.startup_excluded_gate.post_startup_outside_dual_gate_rows == 0`
- `backtest_audit_replay_result.audit_replay_lag_gate.enabled == true`
- `backtest_audit_replay_result.audit_replay_lag_gate.strict == true`
- `backtest_audit_replay_result.audit_replay_lag_gate.passed == true`
- `backtest_audit_replay_result.audit_replay_lag_gate.breach_count == 0`
- `backtest_audit_replay_result.audit_replay_lag_gate.drop_count == 0`
- `backtest_audit_replay_result.audit_replay_lag_gate.fail_count == 0`
- `abs(bt_summary.drop_latency_rate - live_summary.drop_latency_rate) <= 1e-4`
- `abs(bt_summary.drop_api_rate - live_summary.drop_api_rate) <= 1e-4`

## Diagnostics Only

这些字段必须保留并查看，但默认不阻塞参数优化：

- `working_order_lifecycle.non_blocking_mismatch_rows`
- `working_order_lifecycle.identity_only_mismatch_rows`
- `working_order_lifecycle.diagnostic_mismatch_rows`
- `working_order_lifecycle.rest_local_divergence_rows`
- `top5_book_state`

允许它们非零的前提是 hard gates 全部通过，尤其是：

- semantic working-order mismatch 为 `0`
- action/planned/reject/throttle 全部为 `1.0`
- API/throttle mismatch 为 `0`
- target tick mismatch 为 `0`
- post-startup dual replay lag breach 为 `0`

如果 top5 feed-state parity 变差但 target ticks、actions 和 throttle 仍完全一致，它只说明 raw feed reconstruction 仍有诊断差异，不应单独阻塞 maker 参数优化。

## Current Stage 3 Baseline

历史已验收样本：

- run: `5-8-stage3-15m-livetest-v4`
- live audit rows: `80183`
- audit replay common rows: `62340`
- action/planned/reject/throttle match: `1.0 / 1.0 / 1.0 / 1.0`
- working semantic mismatch rows: `0`
- working blocking mismatch rows: `0`
- API/throttle mismatch rows: `0`
- strict replay lag gate breaches: `0`
- post-startup outside dual-gate rows: `0`
- BT/live latency drop: `0.1288418351 / 0.1288397684`
- BT/live API drop: `0.1766923324 / 0.1766894981`

该样本可作为进入阶段5 dry-run maker 参数优化的 alignment gate baseline，但不能替代后续 out-of-sample 验证。

## Current Stage 6G Baseline

当前最新已验收样本：

- run: `5-9-small`
- live decision rows: `71792`
- audit replay common rows: `71787`
- audit replay working-order overlay: `audit`
- live lifecycle events replayed for in-flight exposure: `11264`
- action/planned/reject/throttle match: `1.0 / 1.0 / 1.0 / 1.0`
- working semantic mismatch rows: `0`
- working blocking mismatch rows: `0`
- API/throttle mismatch rows: `0`
- strict replay lag gate breaches: `0`
- post-startup outside dual-gate rows: `0`
- BT/live API drop: `0.16748157744438408 / 0.16748157744438408`
- remaining working-order mismatch is non-blocking REST/local diagnostic evidence

该样本证明当前 live/backtest action-path gate 已恢复；下一步参数优化仍必须使用
无 audit overlays 的 optimization replay，并用多个 OOS 窗口筛选候选。
