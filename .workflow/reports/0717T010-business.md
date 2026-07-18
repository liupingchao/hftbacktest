# 线程回报

执行线程：
- 业务线程-terminal-seal

任务ID：
- 0717T010

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `docs/cross_exchange_live_collection_resilience.md`

action：
- 将 `remote_sha256_manifest.txt` 改为 run-root-relative paths，并排除 manifest 与 `remote_sha256_verification.json`。
- 新增逐项 checksum verification，写出 entry/verified/missing/mismatch/status summary。
- 统一 success/failure terminal ordering：window/root evidence final、最终 status/event row、heartbeat stop/join、manifest 单次生成、立即 verification。
- success 不再二次改写 `run_complete.json` 或二次生成 manifest；failure 不再在最终 `run_status.json` 写入前生成 manifest。
- 增加 post-seal mutation detection 与 ordering/relative-path 回归覆盖。
- 未修改 watcher 策略、live 参数、订单边界、private proof mode 或 Phase 5 集成路径。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py` -> `13 passed`
- T008/T009/T010 相关联合回归 -> `109 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_live_remote_orchestrator.py examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py` -> pass
- `python examples/hyperliquid/cross_exchange_live_remote_orchestrator.py --help` -> pass
- `git diff --check` -> pass
- 未执行 live、credential、private、order、cancel、network、remote 或 service 动作。

done：
- 成功和失败 fixture 均生成并立即验证 checksum manifest。
- heartbeat、最终 status 和 event-log ordering 有直接测试证据。
- 修改已 seal artifact 后 verification 返回 mismatch。
- manifest 使用相对路径且两个 checksum 文件不进入自身校验集合。

blockers：
- 无

commit：
- `b5247d9`

提交信息：
- `Repair terminal artifact sealing and verification`
