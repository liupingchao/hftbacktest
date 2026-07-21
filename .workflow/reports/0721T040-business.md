# 0721T040 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程 / live-awsserver1

任务ID：
- 0721T040

状态：
- 待验收

更新时间：
- 2026-07-21 21:30:43 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/cross_exchange_delayed_history_probe_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_delayed_history_probe_acceptance.py`
- `.workflow/tasks/0721T040.md`
- `.workflow/reports/0721T040-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/principal_alignment_delayed_history_probe_0721T040/`（本地 immutable evidence，按仓库规则忽略）

action：
- 新增与真实订单 terminal 状态机完全隔离的 delayed-history observe-only producer；synthetic cloid 由 task/run/window 确定性生成并证明不属于 managed prefix。
- Producer 只调用 `open_orders`、`user_state`、`query_order_by_cloid` 和 `historical_orders`；不进入 quote、manager、order、cancel、market-close、public-feed 或 terminal participation 路径。
- 新增独立 verifier，从 exact-key schema、raw direct/history rows、monotonic timing、raw final snapshot、redacted open orders/asset positions 和 exact call counts 重建 acceptance。
- Hostile repair关闭 mixed-schema dual write、summary-only account safety、boolean-zero count、arbitrary interpreter/watcher/env/lock、raw timing disconnect、non-`unknownOid` direct result和forged synthetic ownership绕过。
- 独立 hostile 复审结论：原五个 P1 全部关闭，无剩余 P0/P1/P2。
- Exact implementation source：`76bea45046c21d99299b4801c82a262dec6b6e7d`。
- Source archive local/remote SHA-256：`909b5e61953a8d441515b975f60d0323ec1497cd9611ee03032cf30996b9b4f7`。
- Remote source marker：
  - `/home/admin/hftbacktest-cross-exchange-0721T040/source_commit.txt`
  - `76bea45046c21d99299b4801c82a262dec6b6e7d`
- Exact no-start orchestrator preflight通过：
  - one window、`30s` child cap、size/submissions `0`、`5/1/4.0/0.5/5.0` probe contract exact；
  - fixed Python、watcher、env、lock和source exact；
  - watcher/private/account/order/cancel endpoint均未调用。
- Pre-live read-only account/service proof通过：
  - account scope与既有accepted scope一致；
  - open orders `0`、BTC position `0.0`、kill-switch clear；
  - `xemm.service`、T040 service均inactive，冲突进程 `0`；
  - runtime source `63` files、executor import、SDK Python和source marker exact。
- 启动唯一一个 detached `principal-alignment-t040.service`；运行时间为 `2026-07-21T13:24:42Z` 至 `2026-07-21T13:24:48Z`，没有第二窗口。
- Live private read path exact完成：
  - 五轮 `query_order_by_cloid` 均为 raw `unknownOid` / independent `unknown`；
  - history实际在 monotonic `start+4.000137528s` 启动，早于 `start+4.5s` deadline；
  - `historical_orders`调用恰好一次，返回empty list并保持unknown；
  - final open-orders snapshot在总预算 `4.027980s` 完成，open orders `0`；
  - pre/post BTC position均 `0.0`。
- Exact call counts：`open_orders=2`、`user_state=2`、`query_order_by_cloid=5`、`historical_orders=1`、`order=0`、`cancel=0`、`market_close=0`、`public_market_data=0`。
- Child `rc=0`、reaped true、无 abort/termination/SIGKILL；runtime source start/postrun verification均pass，remote terminal checksum `14/14`。
- Post-live read-only account/service proof再次通过：open orders `0`、BTC position `0.0`、kill-switch clear、服务inactive、冲突进程 `0`。
- Complete transfer archive remote/local SHA-256：`94e60561ed0d33b34d93be36642044eea36ecb2d0a2f63e5a2708e4875fe4601`，文件数 `19`。

verify：
- Delayed-history verifier regression：`24 passed`。
- Delayed orchestrator profile regression：`12 passed`。
- Producer-to-verifier integration：`3 passed`。
- Combined changed-file regression：`177 passed in 22.38s`。
- Full Hyperliquid regression：`1151 passed in 50.35s`。
- Python compile、`git diff --check`、cached diff check均通过。
- Local independent live acceptance exit `0`：
  - `14/14` acceptance checks pass；
  - exact schema、identity、deterministic non-owned reference、five direct unknown、canonical history result、timing、history envelope、final snapshot、account safety和zero-side-effect boundary全部pass。
- T037 exact replay exit `2`：decision `39 pass / 4 fail`、lifecycle `78/78 pass`。
- T031 exact replay exit `2`：decision `43/43 pass`、lifecycle `55 pass / 23 fail`。
- T026 exact replay exit `0`：decision `43/43 pass`、lifecycle `78/78 pass`。
- T022/T016 exact replay均exit `2`：decision `43/43 pass`、lifecycle `66 pass / 12 fail`。
- 五份 historical replay均 `offline_only=true`，network/private/order/cancel/remote/new-live全部false。
- No credential value、raw account address、raw synthetic cloid、raw signature或unredacted order reference被持久化。

done：
- 一个且仅一个授权的 delayed-history observe-only live probe已完成并由独立 verifier接受。
- Real endpoint evidence覆盖 exact `5 direct unknown -> 4.0s delay -> 1 history -> final snapshot inside 5.0s` 路径。
- Order、cancel、submit、flatten、position delta、public feed、quote manager和terminal participation全部为零/false。
- Historical exact boundaries保持不变。
- 本任务只证明 delayed-history read/evidence机制；不支持 stable A/k、fill-rate、fees/rebates、PnL、maker viability、multi-level或promotion结论。

blockers：
- 独立 QA acceptance。

commit：
- `76bea45046c21d99299b4801c82a262dec6b6e7d`

提交信息：
- `Add delayed history observe-only probe`
