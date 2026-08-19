# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0729T008

状态：
- 已通过

更新时间：
- 2026-07-29 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-python/public-collector + 0729T008

验收范围：
- Hyperliquid information-max 多轨 public collector、同步采集默认开关、
  strict quality propagation、dual L2 语义、raw 对账、runtime provenance
  和最终 c6in 60 秒 live canary。

验收步骤：
1. 独立复核五轨 subscription、输出目录和 fast/standard L2 可归因性。
2. 构造缺轨和 standard 不深于 fast 的失败反例。
3. 检查 ACK method、raw/control 行数和同步 acceptance 传播。
4. 独立解析最终 canary 五个 raw，重算 SHA、count、shape 和 gap。
5. 核对本地源码、远端运行源码、artifact 源码归档和 tar。

实际结果：
- 首轮 QA 的 P1/P2/P3 已全部关闭：
  - `all_tracks_pass=false` 抛出 `ResearchBundleQualityError`，CLI 非零退出。
  - synchronized acceptance 拒绝失败的 enabled research bundle。
  - fast 必须为非空 top5；standard bid/ask 两侧必须更深。
  - raw row、pong control 和 non-object count 可精确对账并进入门禁。
  - 只有 `subscriptionResponse.data.method=subscribe` 计为 ACK。
  - runtime source SHA 和归档副本写入 bundle。
- Focused tests：
  - `30 passed`
- Final collector/runtime/archive SHA256：
  - `3cb8ea3a37ae12c15e431f7433f05995101ffb867e0da244335e1c0308c0aa2a`
- Final canary：
  - overlap `60.005571989s`
  - fast L2 `112`，全部 `5x5`，arrival gap p50 `536.202ms`
  - standard L2 `12`，全部 `20x20`，arrival gap p50 `5379.518ms`
  - BBO `471`，arrival gap p50 `107.423ms`
  - 五轨 ACK 完整、reconnect `0`、parse error `0`
  - 五轨 row reconciliation 和 raw SHA 全部通过
- 远端/本地 tar SHA256：
  - `5ac24aebb44c7467b9bdcd5e6303f038be4661fd74256e65beae6846c86da4cd`
- tar 成员与本地 final 目录逐文件一致。

验收结论：
- 已通过
- 结论说明：
  - 信息优先多轨采集、严格失败传播、可归因 dual L2、频率统计、
    runtime provenance 和 live evidence 满足本任务验收标准。

通过项：
1. Fast/standard L2 同时保留时间和空间信息，且来源可归因。
2. BBO、trades、asset context、main/target allMids 均独立保留。
3. 缺轨、错误 shape、错误 ACK 和 raw 对账失败均失败关闭。
4. synchronized collection 默认 research-max，且质量失败不会静默通过。
5. 最终 live canary 与当前源码字节绑定。

不通过项：
1. 无

缺陷清单：
1. 无 P0-P2。

阻塞项：
- 非任务代码阻塞：本机完整 Hyperliquid suite 被
  `NumPy 2.4 / Numba <=2.3` 环境冲突挡在 collection。

残余边界：
- 不支持 exact L3/L4 queue reconstruction。
- 不支持 exact fill simulation。
- 不声称 Hyperliquid public feed 达到 `1-2ms`。
- `1-2ms tick2order` 仍只表示消息抵达执行机后的本地处理目标。

建议总控下一步：
1. 进入 collector supervisor、8H 分段采集和严格长窗口质量门禁。
2. 若需要更高频深度，另立 order-book-server/node 数据接入任务。

提交信息：
- commit：无
