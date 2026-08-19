# 业务线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0815T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_trigger_density_core.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_core.py`
- `examples/hyperliquid/cross_exchange_candidate_episode_merging.py`
- `examples/hyperliquid/test_cross_exchange_candidate_episode_merging.py`
- `examples/hyperliquid/cross_exchange_trigger_density_inputs.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_inputs.py`
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 以 QA 已验收的 Stage 1 core SHA
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  为冻结依赖，只读取 Jul30、Aug03、Aug04 trigger audit、collector
  manifest 和 common-L2 timeline。
- Family A 保留全部 `463612` 个 Candidate，包括 rejected；
  Family B 仅保留 `267554` 个 `primary_episode=true`，并使用
  `decision_ts_ns` 作为 confirmed 时间原点。
- 为每个 Candidate 发布唯一 ShockCluster、ContinuousFlowEpisode、
  `2000ms` overlap block、segment 和 connection epoch membership。
  所有 membership 在 segment/epoch 边界终止。
- 以 detector 原始 `pre_state_ts_ns` 精确重建 Binance pre-state；
  `463612/463612` 均有 exact strict-pre-shock state，零 future fallback。
- 冻结 `impact>=0.50/0.70`、同向 refractory
  `100/250/500ms` 和每个 primary flow 的第一个 Candidate 六类
  sensitivity membership；选择只依赖 trigger/pre-state 结构。
- 分开发布 raw row、cluster、flow、overlap、完整/部分 60 秒 time
  block、occupied block 和基于 1 秒 arrival-count series 的
  segment-summed Bartlett ESS；没有把行数命名为 `N_eff`。
- C2 收口时发现入口脚本实际依赖另外四个本地 Python 模块。正式包现
  归档并内容寻址全部五份 runtime source，包括 accepted recovery
  helper 源码；任一依赖漂移都会 fail closed。
- 修复包内 CLI 运行会生成 `__pycache__` 的自修改问题。入口在导入
  本地依赖前禁用 bytecode 写入；包内 `--verify-only` 后目录零新增。
- 未读取 response/outcome/markout/model/PnL，未读取 Aug07 event
  rows，未执行 detector parity、Episode v3、actionability、采集、
  private/account/order/cancel。

实际结构结果：

| session | Candidate / confirmed | rate/s A / B | 2000ms coverage A / B | cluster / flow / overlap | 60s all / complete / partial | Bartlett ESS A / B |
|---|---:|---:|---:|---:|---:|---:|
| Jul30 | 268522 / 141768 | 18.646809 / 9.844708 | 99.982323% / 99.970411% | 39928 / 10536 / 9 | 248 / 240 / 8 | 1096.308299 / 837.102917 |
| Aug03 | 127622 / 82533 | 7.089796 / 4.584963 | 99.236037% / 98.937104% | 47193 / 23397 / 232 | 310 / 300 / 10 | 2195.991765 / 2085.852134 |
| Aug04 | 67468 / 43253 | 9.370386 / 6.007252 | 99.985568% / 99.955562% | 23113 / 9451 / 6 | 121 / 120 / 1 | 275.582545 / 263.249320 |

- Candidate inter-trigger p50 为
  `29.139860ms / 59.912763ms / 58.078078ms`；
  confirmed decision-time p50 为
  `79.394451ms / 117.593711ms / 108.719261ms`。
- 三个 session 的 raw row 与 Bartlett ESS 相差两个数量级左右，
  `2000ms` 窗口又覆盖约 `99-100%` 的结构时间；本阶段因此只支持
  “近连续 order-flow process”解释，不支持 row-level IID event study。
- Aug03 继续标记为 diagnostic、`formal_eligible=false`；本阶段允许
  使用其 arrival structure，但不升级其后续 outcome 证据等级。

sensitivity selected support：

| session / family | impact50 | impact70 | refr100 | refr250 | refr500 | first-flow |
|---|---:|---:|---:|---:|---:|---:|
| Jul30 A | 254572 | 243124 | 119415 | 71481 | 43970 | 10536 |
| Jul30 B | 139864 | 138316 | 94677 | 57720 | 35613 | 9579 |
| Aug03 A | 119765 | 113472 | 75993 | 53737 | 38032 | 23397 |
| Aug03 B | 80687 | 79162 | 65285 | 46505 | 32977 | 21456 |
| Aug04 A | 63274 | 60198 | 39657 | 27041 | 18311 | 9451 |
| Aug04 B | 42347 | 41720 | 33706 | 23135 | 15639 | 8621 |

verify：
- 聚焦回归：
  `python -m pytest -q
  examples/hyperliquid/test_cross_exchange_trigger_density_core.py
  examples/hyperliquid/test_cross_exchange_candidate_episode_merging.py
  examples/hyperliquid/test_cross_exchange_trigger_density_inputs.py
  examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
  通过：`76 passed`。
- Ruff、compileall、CLI help 和 `git diff --check` 全部通过。
- 当前源码入口正式包 `--verify-only` 通过。
- 正式包内归档入口正式包 `--verify-only` 通过，执行前后没有
  `__pycache__` 或额外路径。
- 正式包、`/tmp/0815T001-density-a.PlkRiF/package` 和
  `/tmp/0815T001-density-b.Q6QxdJ/package` 均为 `16` 个文件；
  三者逐 path/bytes/SHA256 完全一致。
- 三次构建均含 `15` 个 core artifacts，core package SHA256 为
  `ed4cbaf7f6474739f6008d7717a3c7fe228c564d6974006e6478f75828e17598`。
- 正式包 full-directory inventory SHA256 为
  `457ebb7aa31aa3b514132e025ce9e16a6c1886cc37bdb6bbe06ecd03bfc0d58b`，
  共 `16` files、`19,801,138` bytes。
- 每次构建的只读 source inventory before/after 均为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- 每次构建的 accepted Stage 1 inventory before/after 均为 `14`
  files、`1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- hostile tests 覆盖 dropped/duplicated Candidate、Family B 错用 shock
  landmark、cross-session/segment/epoch、outcome-like path/field、
  input/stage1 drift、contract drift、五类 runtime source drift、
  partial publication 和 package comparison drift。

done：
- Stage 2 正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- trigger density、episode merging、named sensitivity membership 和
  effective-support admission 已完成，并具备两次独立全量重建证据。
- 当前状态为 `待验收`。独立 QA `已通过` 前，detector parity、
  Episode v3、outcome/model/actionability 和 Aug07 first-read 继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
