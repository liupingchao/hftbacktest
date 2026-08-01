执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0730T017

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
- `.workflow/tasks/0730T017.md`
- `.workflow/reports/0730T017-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 冻结并实现 Binance aggressive-trade burst -> Binance queue confirmation
  -> Hyperliquid liquidity response 的 M1 episode contract。
- 使用固定起点 `10ms` 同方向 burst、触发前严格 as-of best queue、
  `30%` trade impact threshold 和 `100ms` depth confirmation。
- attribution 只使用不晚于 `decision_ts` 的 touch trades；同 burst 的
  post-decision trades 单独计数，不能泄漏进 trade-explained ratio。
- 零价格且零数量的 `377` 条 Binance trade 记录被计数、作为 burst
  边界并排除；其他非正价格/数量继续 fail closed。
- 每个 `(side, decision_ts, pre-best price)` 只发布一个 primary
  episode；重复确认和 `50ms` 同向重复冲击保留在 trigger audit。
- 输出 Hyperliquid withdrawal / retreat / replenishment / follow、
  fast-L2 状态、direction-normalized markout，以及逐 horizon 后续
  同向/反向 shock 数和 isolated 标记。
- `100/250/500ms` 明确为 diagnostic wall-clock/as-of；
  `1000/2000ms` 使用 R1 primary-response tolerance。
- 使用临时目录、输入二次 rehash、输出 schema/row/SHA rescan 和原子
  replacement 发布真实八段数据集。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py -q`
  -> `6 passed`
- `python -m pytest examples/hyperliquid/test_cross_exchange_alignment_acceptance.py examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py -q`
  -> 聚焦组合回归通过。
- 真实八段 builder 完成且 `motif_episode_manifest.json` 为 `passes=true`。
- Builder 输入 provenance 固定为 R0 SHA
  `c46c735d7933587af6eece4a9dd1bce241b3c093866efce976b1c3f952e72ce0`
  和 R1 SHA
  `6123b408cbdfdc95758c8d27e6e0664959caaebeb0de843a5978cfee6c6c8ffd`。

done：
- 扫描 `5,117,387` 条正经济量 Binance trades 和 `377` 条零经济量
  trade records。
- 生成 `268,522` 条 threshold candidates 和 `141,768` 条 primary
  episodes：buy `71,507`，sell `70,261`。
- attribution：trade-driven `234,526`，mixed `19,032`，uncertain
  `14,964`；没有 cancel-driven candidate。
- 拒绝原因：same-direction dedup `65,004`，confirmation reuse `27,521`，
  mixed `19,032`，无 100ms depth confirmation `14,964`，segment 尾部
  `227`，无 prior Hyperliquid BBO `6`。
- 所有八段的 `1000/2000ms` primary coverage 均至少 `95%`；最低为
  `95.51959489211801%`。
- isolated episodes 随 horizon 快速下降：100ms `25,561`、250ms
  `4,949`、500ms `540`、1000ms `19`、2000ms `1`。因此 1000/2000ms
  结果主要描述连续冲击环境，不可解释为孤立单次冲击因果效应。
- 本轮只建立 `Hyperliquid liquidity-response motif family` episode
  数据集；未聚类、未拟合信号、未识别具体 maker、未推断 exact fill /
  maker PnL。

blockers：
- 无；等待独立 QA。

commit：
- 无

提交信息：
- 无
