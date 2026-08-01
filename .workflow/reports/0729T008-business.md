# 线程回报

执行线程：
- 业务线程-python/public-collector

任务ID：
- 0729T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 完整 Hyperliquid suite 被本机 NumPy/Numba 版本冲突挡在 test collection；
  focused tests 和独立 60 秒 live canary 已通过。

files：
- `.workflow/tasks/0729T008.md`
- `.workflow/reports/0729T008-business.md`
- `examples/hyperliquid/hyperliquid_public_sample.py`
- `examples/hyperliquid/test_hyperliquid_public_sample.py`
- `examples/hyperliquid/synchronized_public_collection.py`
- `examples/hyperliquid/test_synchronized_public_collection.py`
- `local_live_analysis/hyperliquid_research_max_skhynix_0729T008_60s_final/`
- `local_live_analysis/hyperliquid_research_max_skhynix_0729T008_60s_final_v2.tar.gz`

action：
- 将 Hyperliquid public collector 升级为 `research_max` 信息优先多轨采集：
  - `fast_market`: `l2Book fast=true + trades + bbo`
  - `standard_l2`: standard `l2Book`
  - `asset_context`: `activeAssetCtx + candle 1m`
  - `main_all_mids`: main-dex `allMids`
  - `target_dex_all_mids`: `allMids dex=xyz`
- fast 与 standard L2 使用独立连接和独立 raw，避免接收 payload 不携带
  subscription variant 时失去来源归因。
- 每轨 manifest 新增：
  - 精确 subscription request/identity/ACK payload
  - 按身份 ACK 计数
  - channel arrival/exchange gap `P50/P90/P99/max`
  - L2 level shape
  - reconnect、parse error、SHA 和起止时间
- aggregate `research_bundle_manifest.json` 新增 track role、overlap、
  per-track quality、information-max 和 latency semantics。
- 严格门禁新增：
  - 任一轨缺 ACK、缺数据、parse error 或 raw 行数无法对账则失败。
  - fast L2 必须为非空 top5 浅档。
  - standard L2 必须在 bid/ask 两侧均比 fast L2 更深。
  - `all_tracks_pass=false` 通过异常和进程退出码传播。
  - synchronized quality acceptance 二次拒绝失败的 enabled bundle。
- 每轨记录 `raw_row_count`、control message count 和 reconciliation。
- ACK 只接受 `subscriptionResponse.data.method=subscribe`。
- bundle 内置运行源码 SHA256 和源码归档副本。
- synchronized collection 默认启用 Hyperliquid research-max，并提供
  `--no-hyperliquid-research-max` 显式回退。
- Binance、alignment 输入根路径、策略、私有接口和交易行为未修改。

verify：
- Focused:
  - `30 passed in 3.37s`
  - `python -m py_compile` 通过
  - 两个 CLI `--help` 暴露 research-max 开关
  - `git diff --check` 通过
- Full Hyperliquid:
  - 未执行到测试主体
  - collection blocker:
    `Numba needs NumPy 2.3 or less. Got NumPy 2.4`
- c6in winner 60 秒 public-only live canary:
  - host: `i-0a962e47210528526 / c6in.xlarge`
  - coin: `xyz:SKHX`
  - final collector SHA256:
    `3cb8ea3a37ae12c15e431f7433f05995101ffb867e0da244335e1c0308c0aa2a`
  - runtime source archive SHA matches collector SHA
  - track count: `5`
  - overlap: `60.005571989s`
  - all tracks pass: `true`
  - dual L2 information gate: `true`
  - all exact ACKs: `true`
  - all tracks reconnect: `0`
  - parse errors: `0`
  - all raw row reconciliations: `true`
  - all raw SHA checks: `true`
- Fast market:
  - L2 `112`, all `5x5`
  - L2 arrival gap p50/p90/p99:
    `536.202 / 609.153 / 739.229 ms`
  - BBO `471`, arrival gap p50/p90/p99:
    `107.423 / 247.263 / 463.624 ms`
  - trade messages `98`
- Standard L2:
  - L2 `12`, all `20x20`
  - arrival gap p50/p90/p99:
    `5379.518 / 5443.949 / 5602.683 ms`
- Context:
  - activeAssetCtx `60`
  - candle `77`
  - main allMids `12`
  - target dex allMids `12`
- Pullback:
  - remote/local tar SHA256:
    `5ac24aebb44c7467b9bdcd5e6303f038be4661fd74256e65beae6846c86da4cd`
  - local gzip/JSON/SHA/source/archive/row-count revalidation passed.

done：
- 信息最大化 public WebSocket 采集已实现并通过 live canary。
- 去掉 `fast` 后的 standard L2 实测约 `5.3s`，但提供 `20x20`；
  fast L2 实测约 `0.54s`，提供 `5x5`。两者同时采集，分别保留空间深度
  和时间分辨率。
- BBO 是当前 Hyperliquid public WebSocket 中更适合执行侧热状态的流，
  但本次实测 p50 仍约 `107ms`，不声称上游达到 `1-2ms`。
- `1-2ms tick2order` 仅指消息抵达执行机后的本地处理目标；不能抵消
  交易所发布 cadence 和网络传输。
- 首轮独立 QA 的 P1/P2/P3 已修复：
  - quality failure 现在失败关闭并传播。
  - dual L2 shape 语义进入严格门禁。
  - 最终 canary 与运行源码 SHA/归档绑定。
  - raw/control 行数可精确对账。
  - ACK method 必须为 `subscribe`。

blockers：
- 非代码阻塞：本机 NumPy/Numba 版本冲突阻止 full Hyperliquid suite
  collection。
- 能力边界：
  - 不声称 Hyperliquid public feed 为毫秒级连续深度增量。
  - 不声称 exact L3/L4 queue reconstruction。
  - 不声称仅凭当前 public data 可做 exact fill simulation。
  - order-book-server/node 级数据接入留给后续独立任务。

commit：
- 无

提交信息：
- 无
