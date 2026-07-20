执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T023

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/hyperliquid_tiny_live_sdk_readiness.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_sdk_readiness.py`
- `.workflow/tasks/0720T023.md`
- `.workflow/reports/0720T023-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 按官方 nested `orderStatus` envelope 严格解析 exact oid/cloid terminal status；`unknownOid`、malformed、identity mismatch 和 filled-without-raw-fill 保持 fail closed。
- 在现有五秒 terminal-reconciliation budget 内执行最多五轮 direct oid/cloid query；每个 direct attempt 持久化 `direct_round`、sequence 和起止时间。
- 仅在全部 direct round 持续 unknown 后，对每个 unresolved reference 最多调用一次 `historical_orders`；只接受 exact unique identity 和合法 authoritative terminal status。
- 将完整 query-attempt audit 与每个 reference 的唯一 canonical terminal result 分离；producer 和 acceptance 独立重建相同 v4 contract。
- 历史回退后要求同一预算内完成 final open-order snapshot；不完整 snapshot、duplicate/conflicting history 和提前 history fallback 均 fail closed。
- Exact tracked OID recovery 严格校验 symbol、side、exact price、remaining size、declared size 及同层 alias 一致性，避免错误方向、价格或暴露被恢复。
- SDK readiness 要求 `hyperliquid-python-sdk==0.24.0`、`historical_orders` surface、constructor timeout 及离线 write/read/restore round-trip。
- Acceptance `validation_report.md` 标题由 external expected task id 生成；T016/T022 历史证据不被升级。
- 未改变 strategy、quote、signal、risk cap、submission cap、activation、dynamic spread、inventory skew 或 multi-level 行为。

verify：
- Focused executor/readiness/manager/watcher/fill/acceptance：`523 passed in 18.87s`。
- Full `python -m pytest -q -p no:cacheprovider examples/hyperliquid`：`861 passed in 39.06s`。
- 修改模块和测试 `py_compile`、readiness/acceptance CLI `--help`、`git diff --check`、implementation `git show --check`：通过。
- 离线 SDK readiness：官方版本 `0.24.0`，surface/timeout compatibility 全部通过，最终 recommendation 为 READY；credential、HTTP/WS、private/account/order/cancel flags 均为 false。
- T016 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`，标题为 `0720T016`，118 个输入文件 hash 保持 `a5443f8f9cb777509ffde21f9caed3837a9ef58f31b25cd82acef2d6a3991398`。
- T022 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`，标题为 `0720T022`，118 个输入文件 hash 保持 `71c9ba35f62e6657d0d48f63abb451ad538f14d85d7113a567b16dfe2b92833c`。
- 两次 replay boundary 均为 `offline_only=true`，network/private/order/cancel/remote/new-live 全部为 false。
- 独立只读复核经过多轮 hostile counterexample 检查后结论为 `No findings`。
- 开发中首次 runtime probe 错误传入 `perp_dexs`，SDK 尝试访问 `localhost:9`；沙箱在 socket 建立前拒绝，未连接任何服务、Hyperliquid 或私有端点。最终实现移除该参数，后续 readiness 与完整验证均未发起 endpoint 请求。

done：
- Future cancel-unknown lifecycle 已具备 bounded official-schema direct/history recovery path。
- 每个 accepted terminal fact 都绑定 exact reference、完整轮次审计和唯一 canonical source。
- Account-wide absence 不能替代 reference-bound terminal history；filled 不能绕过 raw fill proof。
- T016/T022 历史 blocker、输入字节和结论保持不变。
- T023 实现已完成并等待独立 QA。

blockers：
- 业务实现无已知缺陷；独立 QA 尚未完成。
- QA 通过前不得执行 private historical read、新 bounded live、Task 8 或 adaptive/multi-level activation。

commit：
- `b8474af3b94a0d5145398a9c82e577e5ab038d51`

提交信息：
- `Repair bounded Hyperliquid terminal history recovery`
