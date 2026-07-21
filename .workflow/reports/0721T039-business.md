# 0721T039 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T039

状态：
- 待验收

更新时间：
- 2026-07-21 19:20:44 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- T039+ anti-drift rows持久化 canonical minimum-pressure和ratio threshold；independent acceptance从 `side/limit/current BBO` 重建 cross risk，从 raw quantity projections、ratio、canonical thresholds和 adverse-BBO timestamp/fact重建 flow status与最终 gate outcome。
- `current_cross_risk`、`adverse_flow_ratio`、`adverse_flow_status` 只作为 comparison targets；非法 quantity、boolean、side/price/BBO、timestamp、threshold和同步 derived-label forgery均产生 validation reason。
- Canonical policy exact绑定 `250ms` stability、`0.01 BTC` minimum pressure和 `2.0` ratio，不能通过改写有效正数 threshold并同步改 status 绕过。
- Late persistent kill-switch在 T039+ 使用 `late_halt_stage_v1` 独立 row；canonical precedence为 `late halt > immediate > anti-drift > edge > pass`，原始 immediate/anti/edge rows不再被原地覆盖。
- Late row、trigger、submit-decision、attempt和已执行 stage按 event/attempt exact join；missing、duplicate、forged、cross-attempt、submit drift及 `anti block + edge + late` 不可达组合均 fail closed。
- Immediate independent rebuild改为与 producer 相同的 `if/elif` reason branch；非 touch 且同时 crossing时只产生 `selected_quote_not_current_touch`。
- Producer rollout按 artifact task ID隔离：T038及更早继续使用 legacy guard overwrite和旧 CSV schema，不生成 late file；T039+ 才生成 raw columns、late submit fields和独立 late artifact。
- Pre-T039 verifier显式拒绝混入的 T039 late file、late submit columns或raw threshold columns，历史 schema boundary成为 producer与acceptance双向合同。
- Hostile review提出的 producer/verifier rollout findings已采纳并修复；“late不得覆盖 simultaneous immediate/anti”意见未采纳，因为任务明确规定 late safety halt为最高优先级，同时必须保留下级原始 stage evidence。

verify：
- QA原始反例覆盖 crossing buy、adverse `0.04`、favorable `0`、ratio `inf`、adverse BBO true及同步 derived pass forgery；independent rebuild产生 cross-risk/flow-status/semantic mismatch。
- Field forgery覆盖 side、limit、BBO、quantity projection、ratio、flow status、cross-risk、boolean、minimum pressure、ratio threshold和stability threshold。
- Late-halt tests覆盖 edge pass/block、immediate fail simultaneous override、missing artifact/row、duplicate、forged reason、cross-attempt、submit drift和impossible anti+edge stage graph。
- T038 tests覆盖 legacy late guard overwrite、旧 anti/submit headers、无 late file、mixed-schema file/header/row rejection，以及 public-state/canonical/hold既有合同。
- Final task-file regression先后通过；最终 full Hyperliquid regression：`1111 passed in 61.92s`。
- Python compile、`git diff --check`、cached diff check通过。
- T037 exact replay：exit `2`，config `72/72`、decision `39 pass / 4 fail`、lifecycle `78/78 pass`。
- T031 exact replay：exit `2`，config `72/72`、decision `43/43 pass`、lifecycle `55 pass / 23 fail`。
- T026 exact replay：exit `0`，config `72/72`、decision `43/43 pass`、lifecycle `78/78 pass`。
- T016/T022 exact replay：均 exit `2`，config `72/72`、decision `43/43 pass`、lifecycle均 `66 pass / 12 fail`。
- 五份 historical replay均 `offline_only=true`，network/private/order/cancel/remote/new-live全部 false。
- 本任务未执行 live、private/account、order、cancel、network、remote或service操作，未修改任何 immutable evidence。

done：
- T038 QA 的三个 findings均在 producer、independent verifier和 hostile regression中关闭。
- T039 raw stage schema、late-halt独立证据和 immediate branch parity具备 task-aware rollout与历史边界保护。
- Source已准备进入独立QA；只有QA通过后才可派发 delayed-history observe-only probe。

blockers：
- 独立 QA acceptance。

commit：
- `4d952d2a3d7b3657cdb428822e23f67fa50bdc54`

提交信息：
- `Repair raw stage evidence validation`
