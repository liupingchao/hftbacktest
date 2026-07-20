# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T027

状态：
- 待验收

更新时间：
- 2026-07-20 20:12 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_fill_feedback.py`
- `.workflow/tasks/0720T027.md`
- `.workflow/reports/0720T027-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增 `per_attempt_reference_terminal_reconciliation_v5`：每个 submitted attempt 必须有一个 exact submit response；同步 rejected 只在 outer success、单一非空 error、attempt/side/key/intent cloid 和 manager action/state/query 全部一致时成为 authoritative terminal。
- Rejected attempt 保留在 submitted/reference ledger，但与 cancel/query/fill 路径严格互斥；duplicate submit response、reject 加 cancel/query、malformed/multi-status、identity mismatch 和 forged manager classification 均 fail closed。
- Producer 将 exact exchange `error` 持久化为 canonical `rejected`，post-only reject count 从 validated raw response reconciliation 派生。
- Acceptance 独立重建同一 v5 terminal matrix，不调用 producer helper；支持 rejected、resting/partial plus cancel/query、full fill 和 partial-fill remainder reconciliation。
- 为 immutable T026 v4 producer false-negative 增加 exact bridge，仅绑定 task `0720T026` 和 source `40dc56a3225df4afb0f2185873c91f17b578f550`；bridge 先要求旧 v4 summary 与 raw v4 精确一致，再从 raw response 独立重建 v5。
- Fill-feedback 按 attempt key 选择唯一 submitted lifecycle row；同 key 的 earlier no-submit candidate rows 仅是 decision evidence，不再 quarantine；冲突 submitted rows 继续 fail closed。
- Acceptance 不再直接序列化 unordered set，comparison CSV 在不同 `PYTHONHASHSEED` 下保持逐字节一致。

verify：
- Focused four-file regression：`451 passed in 18.98s`。
- Full `python -m pytest -q examples/hyperliquid`：`955 passed in 39.23s`。
- 新增 positive mixed `buy rejected / sell resting+cancel`、11 类 hostile submit-response、duplicate/cancel conflict、canonical fill-feedback、conflicting submitted rows、watcher persistence、full acceptance 和 hash-seed determinism tests。
- `py_compile`、`git diff --check`、cached diff check 和 implementation `git show --check`：通过。
- Exact T026 replay exit `0`：
  - provenance `112/112`；
  - config `72/72`；
  - decision `43/43`；
  - lifecycle `64/64`；
  - economics `6/6`；
  - bridge `authorized=true / applied=true`；
  - boundary `offline_only=true`，network/private/order/cancel/remote/new-live 全部 false。
- Exact T026 fill-feedback normalization：quarantine `0`；buy 为 `submitted=true / rejected=true / excluded_rejected`；sell 保留 resting/censored 事实。
- T016/T022 replay 均保持 exit `2`、decision `43/43`、lifecycle `52 pass / 12 fail`；新增通过项没有补造历史 terminal fact。
- T016 input aggregate hash 前后均为 `a5443f8f9cb777509ffde21f9caed3837a9ef58f31b25cd82acef2d6a3991398`。
- T022 input aggregate hash 前后均为 `71c9ba35f62e6657d0d48f63abb451ad538f14d85d7113a567b16dfe2b92833c`。
- T026 raw evidence root 未修改；本任务未发生 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T026 buy exact rejected terminal 与 sell resting/cancel terminal 现在被独立重建为两个完整且互斥的 submitted attempt lifecycle。
- Producer、acceptance 和 fill-feedback 对 rejected 状态、reject count 和 no-fill denominator 的语义一致。
- Immutable T026 acceptance 通过，T016/T022 historical blockers 保持。
- T026 QA 的 rejected-terminal、fill-feedback canonical-row 和 deterministic-output 三项 findings 已进入独立 QA 验收。

blockers：
- 无实现阻塞；独立 QA 是当前流程节点。QA 通过前 Task 8、dynamic spread activation、fill-feedback activation、inventory skew activation 和 multi-level 继续锁定。

commit：
- `b8e008afc90a25ffb155f09bb5d77aa90cdbb4c4`

提交信息：
- `Repair submit rejection terminal evidence`
