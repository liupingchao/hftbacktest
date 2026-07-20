# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T015

状态：
- 待验收

更新时间：
- 2026-07-20 09:38 CST

是否进行QA验收：
- 是

QA说明：
- 验收 external expected duration、standing `1800s` cap、historical `900s` replay compatibility 和 no-live boundary。

files：
- `.workflow/tasks/0720T015.md`
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- 将 orchestrator exact-envelope 最大窗口从历史 `900s` 对齐到 standing authorization 的 `1800s`。
- Acceptance 新增 external `expected_window_seconds`：
  - 默认 `900s`，保留历史 artifact replay；
  - CLI 可显式指定未来任务的 exact duration；
  - expected duration 必须为正且不超过 `1800s`；
  - preflight duration 必须 exact-match external expected；
  - runtime watcher argv 必须 exact-match preflight；
  - runtime duration另受独立 `1800s` hard cap。
- Acceptance manifest 持久化 `expected_window_seconds`，artifact 自报值不能替代外部授权。
- 新增回归：
  - exact two-sided `1800s` preflight passes；
  - `1800.001s` preflight before-output fail closed；
  - synchronized valid `1800s` acceptance fixture passes；
  - artifact `900s` 与 external expected `1800s` mismatch blocked；
  - external/artifact `1800.001s` blocked。
- 未修改 signal freshness、edge threshold、quote formula、risk cap、activation、manager lifecycle 或 evidence identity。

verify：
- Exact implementation commit：
  - `35ceea2a5ca861a9329e384a2faeca11d9615cf7`
- Focused orchestrator + acceptance：
  - `171 passed in 5.32s`
- Full Hyperliquid post-commit：
  - `712 passed in 35.69s`
- Modified modules/tests `py_compile`：pass。
- Orchestrator and acceptance CLI `--help`：pass；acceptance exposes `--expected-window-seconds`。
- Exact local `1800s` preflight：
  - status `pass`；
  - source commit `35ceea2a5ca861a9329e384a2faeca11d9615cf7`；
  - preflight/runtime command duration `1800.0/1800.0`；
  - watcher/private/account/order/cancel/credential activity all `false`；
  - dynamic spread/fill feedback/inventory skew/multi-level/actual quote behavior all `false`。
- Exact local `1800.001s` preflight：
  - rejected with `exact_envelope_mismatch:window_seconds`；
  - no preflight artifact or window output created。
- T011 byte-exact replay with explicit legacy bridge and expected `900s`：
  - exit `2`；
  - provenance `112 pass / 0 fail`；
  - config `71 pass / 1 fail`；
  - decision `16 pass / 27 fail`；
  - lifecycle `34 pass / 27 fail`；
  - `validation_reasons=[]`；
  - final lifecycle blocked。
- Implementation `git show --check` and worktree `git diff --check`：pass。
- 未执行 live、private、account、order、cancel、network、remote 或 service。

done：
- Orchestrator 与 acceptance 已在一个 external exact-duration 合同下对齐到 standing `1800s` cap。
- 新任务可以显式验收 `1800s`，历史 T011 仍可显式按 `900s` deterministic replay。
- Artifact 不能通过自报 duration 扩大授权。
- Strategy、risk 和 adaptive/multi-level activation 未变化。

blockers：
- 当前唯一流程节点是独立 QA 验收 T015。
- T015 QA 通过前不得启动下一 live task。

commit：
- `35ceea2a5ca861a9329e384a2faeca11d9615cf7`

提交信息：
- `Align exact live duration contract`
