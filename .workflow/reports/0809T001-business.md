```md
执行线程：
- 业务线程-python/research

任务ID：
- 0809T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `/Users/liu/Documents/glft/src/skhynix_glft/maker_policy_replay.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/maker_diagnostics.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/level_guard_attribution.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/side_regime_oos.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/arrival_candidate.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/nautilus_scenario_matrix.py`
- `/Users/liu/Documents/glft/src/skhynix_glft/scenario_matrix.py`
- `/Users/liu/Documents/glft/tests/`
- `/Users/liu/Documents/glft/docs/skhynix-maker-diagnostics-native-matrix-0809T001.md`
- `/home/molly/projects/glft/runtime/0809T001-maker-slice/output/`

action：
- 精确重放 4H full/control 策略与三个 protection 合同。
- 建设 side/regime 决策漏斗、理论成交 proxy 和多周期 markout。
- 完成 Level guard ask/bid 成交、库存和 paired PnL 归因。
- 完成 side/regime arrival-intensity 训练与 Aug07 OOS 校准。
- 生成 global side-arrival research candidate，不修改 active/live config。
- 在 AMD 完成 hftbacktest 12 x 3 与 Nautilus 6 x 3 场景矩阵。
- 输出正式研究报告并同步到 AMD。

verify：
- 本机完整测试：`70 passed, 2 skipped, 2 subtests passed`。
- AMD `nt-backtest` 完整测试：`61 passed, 2 skipped, 2 subtests passed`。
- AMD hftbacktest 定向测试：`13 passed, 2 skipped`。
- Ruff、compileall、`git diff --check` 通过。
- 4H replay `144001` 行对齐，策略字段 mismatch 为 `0`。
- hftbacktest `36` 行和 Nautilus `18` 行均 `stream_complete=True`。
- 两框架 candidate config hash 均为
  `45ff3862dc6ef205c5a9925e4241cd796fe23af8c8ea780ffaa5dc16fcb85195`。
- local/AMD 报告 SHA256 均为
  `c55ea387ba52c1a0a24a18190ff3fb8c1b7b1751b401d0412d4198cff2f0bb89`。

done：
- 用户指定的五步执行顺序全部完成。
- 保留 4.5bp floor 与 change guard；Level guard 保持 research freeze。
- 当前证据不授权 live order。下一门禁是 Aug07 4H pure-OOS native replay。

blockers：
- full-policy 理论 markout 在两侧和全部报告 horizon 仍为负。
- arrival-intensity 存在明显跨 session shift。
- candidate 三 session PnL 部分 in-sample。
- 250ms hftbacktest 与联合压力场景 PnL 为负。
- queue/partial-fill 敏感性尚未被当前路径识别。

commit：
- 无

提交信息：
- 无
```
