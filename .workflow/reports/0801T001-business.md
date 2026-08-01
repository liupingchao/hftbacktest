执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_episodes.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py`
- `docs/skhynix_liquidity_response_motif.md`
- `local_live_analysis/skhynix_liquidity_response_0730T017/`
- `.workflow/tasks/0801T001.md`
- `.workflow/reports/0801T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 T017 QA 的 P1 x3 / P2 x1。
- segment manifest exact 校验 campaign/segment/profile/symbols，并要求
  R0 descriptor outputs 与 segment manifest outputs 完全一致。
- timeline 每行 exact 校验 campaign/segment/profile；Binance trade 和
  Hyperliquid BBO 分别校验 symbol/coin identity。
- 八个 segment manifest 进入 initial provenance、构建后 rehash 和最终
  manifest，输入闭环从 `26` 扩展到 `34` 个文件。
- exact 校验 R1 primary horizons、`1000/2000ms -> 250ms` tolerance、
  diagnostic subset、七个 acceptance gates，以及 timestamp/future/cross
  segment 三个零错误 gate。
- 已有 output 使用 macOS `renamex_np(RENAME_SWAP)` 原子目录交换；
  Linux 路径使用 `renameat2(RENAME_EXCHANGE)`。交换过程 output path
  始终存在，交换完成后再清理 temporary path。
- schema 升级为 `hyperliquid_liquidity_response_motif_v2`，真实 manifest
  task ID 为 `0801T001` 并记录 publication contract。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py -q`
  -> `14 passed`
- `python -m pytest examples/hyperliquid/test_cross_exchange_alignment_acceptance.py examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py -q`
  -> `22 passed`
- `python -m py_compile ...` -> pass
- `git diff --check ...` -> pass
- 真实八段 v2 builder 使用原子目录交换完成，退出码 `0`。
- 独立全量 rescan：input/output SHA、row、identity、threshold、
  confirmation、attribution、confirmation uniqueness、horizon direction、
  coverage、isolation 和 aggregate reconciliation 错误均为 `0`。

done：
- T017 的 profile/timeline identity 缺陷由 exact gates 和 failure
  injections 关闭。
- 八个 segment manifest 已完整进入 provenance 和构建后 stability gate。
- R1 tolerance/gate 漂移现在 fail closed。
- existing-output publication 无 visibility gap；原子交换与 clean rebuild
  测试通过，真实发布后无 `.tmp` 或 `.backup-*` 残留。
- 修复版真实产物保持 `268,522` candidates、`141,768` primary episodes、
  `71,507/70,261` buy/sell 和最低 `95.51959489211801%` primary coverage。
- 34 个 provenance roles：
  - R0 manifest `1`
  - R1 manifest `1`
  - segment manifest `8`
  - Binance hot events `8`
  - Hyperliquid hot events `8`
  - timeline `8`
- 本轮未访问网络/AWS/SSH，未新增采集、聚类、信号拟合或 maker/PnL
  推断。

blockers：
- 无；等待第二轮独立 QA。

commit：
- 无

提交信息：
- 无
