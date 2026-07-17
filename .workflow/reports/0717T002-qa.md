# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0717T002

状态：
- 阻塞

更新时间：
- 2026-07-17 14:10 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-live-evidence 0717T002

验收范围：
- 验收本轮 SSM-first controlled role-evidence live rerun 是否完成采集、是否安全收口、是否产出 fill source / maker-taker role evidence。
- 本 QA 不验收 fee/PnL、maker viability、T012、promotion、final MVP pass、策略收益或 quote-policy 改进。

验收步骤：
1. 检查 task/report 边界和 live envelope。
2. 检查 remote run status、window status、run complete、final open-orders proof。
3. 检查本地拉回 artifact 的 JSON/CSV parseability。
4. 检查 `remote_sha256_manifest.txt` 对拉回文件的完整性。
5. 汇总 `order_intent_audit.csv`、`live_fill_ledger.csv`、`fill_liquidity_role_evidence.csv`。

实际结果：
- SSM-first orchestrator 完整跑完 3 个窗口。
- 三个窗口 runner return code 均为 `0`。
- 三个窗口 independent open-orders proof 均为 `0`。
- final root open-orders proof 为 `0`。
- 本地 artifact JSON/CSV 均可解析。
- sha manifest `241/241` 条匹配。
- 三窗合计 `4` 条真实 order intent。
- `live_fill_ledger.csv` 合计 `0` 行。
- `fill_liquidity_role_evidence.csv` 合计 `0` 行。

验收结论：
- 阻塞
- 结论说明：
  - 采集链路和安全收口通过，但本任务的核心 P0 目标是补 fill source / maker-taker role evidence；本轮无 fill，因此没有 role evidence，P0 仍阻塞。

通过项：
1. SSM-first live execution and recovery contract worked.
2. Three windows completed with status/heartbeat/window artifacts.
3. Final and per-window open-orders proofs are empty.
4. Artifact parse and sha validation passed.

不通过项：
1. 没有 fill rows。
2. 没有 maker/taker liquidity-role evidence rows。

缺陷清单：
1. 当前 CLI path 不暴露 `--max-loss 1` 或 `--max-position-delta 0.01` runtime 参数；本轮只能按 post-run evidence 验证边界，不能声称 runtime 已新增这两个控制项。

阻塞项：
- `no_fill_role_evidence_absent`

建议总控下一步：
1. 不要进入 fee/PnL calibration、maker viability 或 promotion。
2. 总控需选择：继续单独授权 role-evidence rerun，或明确降级 P0 gate 后再做 T004 public shadow。

提交信息：
- commit：0f894efd570b08954967219d925c2b4afc9e6f10 / Record 0717T002 SSM live rerun evidence
