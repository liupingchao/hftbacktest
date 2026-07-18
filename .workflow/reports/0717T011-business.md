# 线程回报

执行线程：
- 业务线程-integrated-offline-acceptance

任务ID：
- 0717T011

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/test_cross_exchange_live_integrated_offline_acceptance.py`

action：
- 新增三窗口 offline fake watcher fixture。
- window 1 写入同一 fill 的两次 pullback；window 2 写入两个同价 attempt、一个唯一 time-bounded fill 和一个 ambiguous fill；window 3 挂起并由 orchestrator timeout。
- 使用真实 `LiveFillLedger` 回放 window 1/2 artifacts，验证 window/attempt identity、幂等数量、唯一时间归因和 ambiguous quarantine。
- 使用真实 remote orchestrator 验证 child timeout/reap、no window 4、abort/open-orders proof ordering、Python checksum verification 和 `sha256sum -c`。
- 未修改生产 watcher、策略、订单参数或 live execution path。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_live_integrated_offline_acceptance.py` -> `1 passed`
- Phase 1-4 integrated regression -> `110 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_live_remote_orchestrator.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/test_cross_exchange_live_integrated_offline_acceptance.py` -> pass
- `python examples/hyperliquid/cross_exchange_live_remote_orchestrator.py --help` -> pass
- `git diff --check` -> pass
- fixture 内 `sha256sum -c remote_sha256_manifest.txt` -> pass
- 未执行 live、credential、private、order、cancel、network、remote 或 service 动作。

done：
- 三窗口 integrated fixture 通过，window 4 未创建。
- window 1 的一份 fill 在重复 pullback 中只计一次；window 2 的唯一 fill 归因到 `window_02:attempt_2`，ambiguous fill 保持未归因。
- window 3 child 被 timeout、终止并 reaped，abort/root proof 和 final checksum evidence 完整。

blockers：
- 无

commit：
- `6fadc95`

提交信息：
- `Add 0717T011 integrated offline acceptance`
