# 0722T053 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0722T053

状态：
- 待验收

更新时间：
- 2026-07-22 10:21 Asia/Shanghai

是否进行QA验收：
- 是

前置任务：
- `0718T021` 已通过 default-off ladder contract。
- `0722T052` 已通过单层双边 lifecycle mechanism/evidence baseline。
- 本任务 offline-only；未执行 live、private/account、order/cancel、
  network、remote 或 service 操作。

dispatch：
- Base commit：
  `316c1ad9268b48f864f457cadd8b7315fd917bee`
- Standing live envelope 保持 `0.005 BTC/order`、`0.01 BTC position`、
  `1 USDC loss`、`2 submissions/window`、`1800s/window`。

action：
- `QuoteLadderConfigV1` 和 pricing config 将 levels 限制为 `1..8`；
  level 0 保持 reservation +/- half-spread，deeper levels 使用 gap 和
  size decay，并继续执行 lot、size、post-only、duplicate/coalesce 和
  aggregate-size gate。
- Activation enabled 且 T052 lifecycle prerequisite true 时，ladder 输出
  executable `quote_intents`；default-off 或 prerequisite false 仍不改变
  executable quote behavior。
- Ladder price key 改为与 manager 相同的 Hyperliquid canonical
  normalization；新增 passed ladder 到 `DesiredQuote` 的严格 post-only
  转换边界。
- Manager 支持有限 `max_levels_per_side`，按
  `(symbol, side, canonical_price_key)` 管理同侧多个 owned orders。
- Desired reconcile 改为 exact target set：确定性排序、同 key hold、
  size 变化或过期 key cancel、cancel-confirmed 后 readd。
- Structural validation、完整 additions submission budget 和 aggregate
  projected exposure 在首个 submit 前整批校验；失败不改变 generation、
  order state 或 submission counter。
- Stale cancel set 在首个 cancel 前整批执行 state/age/price/rate gate；
  任一 guard 失败时不发生 partial cancel。Cancel unknown 后停止剩余
  cancel batch。
- Startup 对 owned duplicate key 和 per-side level cap 做 precommit
  validation；snapshot 报告实际 configured multi-level gate。

verify：
- Focused ladder/manager：
  `python -m pytest
  examples/hyperliquid/test_hyperliquid_maker_order_manager.py
  examples/hyperliquid/test_cross_exchange_quote_ladder.py -q`
  -> `114 passed in 0.16s`。
- Full Hyperliquid：
  `python -m pytest examples/hyperliquid -q`
  -> `1240 passed in 51.58s`。
- `python -m py_compile` 对四个修改/测试文件通过。
- `git diff --check` 对 T053 scope 通过。

done：
- Active two-level-per-side ladder 可产生四个 deterministic executable
  intents，并可经严格转换进入 price-keyed manager。
- 首次建梯、相同目标 hold、level removal、price/size replacement、
  cancel-before-readd、empty target cancel-all 和 restart recovery 已覆盖。
- Duplicate canonical price、per-side cap、standing two-submission budget、
  aggregate exposure 和 startup over-cap 均 fail-closed。
- Default single-level manager 和 T052 lifecycle/cancel/fill 行为通过全量
  regression。

limits：
- 本任务只证明 offline code/action-path/replay readiness，不提供
  multi-level live evidence。
- 新建 two-level-per-side ladder需要四次 submissions；standing
  `2 submissions/window` envelope 会在任何 order call 前拒绝。
- Restart 可从 authoritative open orders 恢复并 hold/cancel，但历史
  rejected/filled/canceled submission count 和完整 generation provenance
  不可仅从 open orders 重建；multi-level live 需要独立持久化/恢复设计和
  新的 exact-envelope task。
- 不支持 stable PnL、fill rate、fee/rebate、queue priority、maker
  viability、promotion 或 final MVP 结论。

blockers：
- 独立 QA 验收。
- 任何 multi-level live 均需新的风险 envelope 决策；本任务未扩张 cap。

commit：
- Implementation commit：
  `68416c180ca02d53de455f3971fbeb9d5cb5ac4b`

提交信息：
- `Enable guarded multi-level ladder execution`
