# 0829T003 Hostile Plan Review Round 1

日期：
- 2026-08-29 17:36 CST

审查对象：
- task：`.workflow/tasks/0829T003.md`
- candidate plan：
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md`
- review commit：`ae505b6699a311a2490b26b5f6771cff5e0ca63a`
- candidate plan SHA256：
  `27e3ca557dafe9a797735a545871f015cb1c0e600e4cba4518c86385d87a4b26`

审查方式：
- 独立 hostile scientific-contract review。
- 未运行 29-cache，未读取 future outcomes，未修改 plan/task。
- 重点审查 causality、epoch/reset/slice invariance、segment boundary、
  dual-direction cluster、exposure/selection/null、统计 gates、outputs 和
  execution lock。

结论：
- **FAIL / 不可冻结**
- `P0/P1/P2/P3 = 0/6/4/0`
- 29-cache execution lock 必须保持；不得实现或执行该 candidate contract。

## Authority Check

以下 authority 无 finding：

- accepted QA/controller commit `460f6490...` 存在，且包含
  implementation commit `094ad7b5...`。
- runner blob OID `4e48d126...` 和 whole-file SHA256
  `8a9ce6ed...` 精确匹配。
- plan 列出的 13 个 normalized callable AST SHA256 全部重算匹配。
- 29-cache inventory file、predecessor classification 和 predecessor
  summary SHA256 全部匹配。
- task 和 plan 均明确保持 29-cache、future outcome、A0 和 live/private/
  order execution lock。

## P1 Findings

### P1-1 Segment reset 可以在同一 epoch 内创造多个 cluster，核心几何结论不成立

Plan `:205-207` 按
`(capture, segment, direction, epoch_id)` 独立保留最早 onset，`:221`
又把 cluster 定义为 `(capture, segment, epoch_id)`。因此同一 capture 在
一个 60s epoch/core 内发生 segment reset 时，每个新 segment 都能再次产生
每方向 anchor 和新 cluster。

这直接反例化：

- `:216` 的“每 epoch 最多两个 anchors”；
- `:224-225` 的同 epoch 双方向单 cluster 语义；
- `:229-233` 的后续 cluster 至少相隔 30s。

例如 segment 0 在 core 内 19s 产生 anchor，20s reset 后 segment 1 在 21s
产生同方向 anchor；二者属于不同 cluster，间隔仅 2s。该问题不是 wording
瑕疵，它会改变 observed/null cluster count、raw burst 和 rate。

Revision 必须唯一选择并冻结一种语义：

1. 任何 influence/core 区间与 segment/reset boundary 相交的 epoch 均
   outcome-blind ineligible；或
2. suppression/cluster 跨 segment 使用 `(capture, epoch)` 固定 identity，
   并明确接受跨 reset suppression；或
3. 明确允许 per-segment epoch clusters，同时撤回 30s separation/两-anchor
   claims，并据此重做 dependence、burst 和 statistical gates。

### P1-2 Slice/reset invariance protocol 不足以形成唯一可执行测试

Plan `:269-304` 没有定义：

- “eligible artificial start”的确定性 schedule、stride 和 minimum segment
  duration；
- checkpoint/searchsorted 边界；
- 哪些 epoch 算作 slice 后的“complete epoch”；
- comparison 起点应取 `start+122s`、下一个 epoch start、core open，还是
  第一个 influence interval 完整的 epoch；
- segment end、maximum persistence 和 partial final epoch 如何裁剪；
- M-state support-count 的精确 checkpoint domain；
- `zero cross-segment comparison` 的行级定义和计数器。

固定 `122s` guard 只是长度上界，不能替代 comparison-epoch contract。
不同实现可以合法地产生不同 slice rows 和 gate 结果。Revision 应按绝对
epoch identity 定义可比较 epoch 集合，例如只比较 causal influence interval
和完整 core/persistence interval 全部落在同一 sliced segment 内的 epochs，
并冻结完整 CSV schema。

### P1-3 Structural null 的 callable authority 和 conditional claim 未冻结

Plan `:83-100` 绑定了 M-state、RNG、selection 和 estimator callables，但
`:308-337` 仅以“reuse accepted null”引用 path-swap null。它没有直接绑定
实际生成 null 的 load-bearing primitives：

- `fixed_opposite_orientation_pairs`
- `null_layout`
- `permute_trade_direction_paths`
- fixed pair identity/canonical ordering

也没有重述 accepted `H0_conditional` 的固定 conditioning variables 和 claim
limit。Whole-file predecessor SHA 不能阻止 successor 绕过这些 transitive
callables 自行重写 null。

Revision 必须冻结并 direct-call exact null primitives/AST，重述 conditional
exchangeability law，并新增 epoch-specific invariants：每 replicate 重算
thinning、dual-direction dedup、segment/epoch cluster identity，以及 observed/
null filter-duration denominator identity。

### P1-4 Gate A-1-2/A-1-3/A-1-4 没有精确条件，first-failure classification 不唯一

Plan `:369-393` 只给 gate 名称和 classification；`:395-420` 只展开
A-1-5 至 A-1-7 thresholds。新版本最关键的 A-1-2 没有冻结以下 counters：

- epoch/core arithmetic violations；
- more-than-one retained onset per exact key；
- earliest-onset/tie-break violations；
- cluster identity/dual-direction sharing violations；
- boundary-contaminated epoch disposition；
- cross-segment slice comparison；
- slice identity/support mismatches。

A-1-3 也没有冻结 199 replicates、fingerprint、stream overlap、mask mismatch、
pair count 和 joint-distance conditions；A-1-4 没有明确继承 zero/nonfinite/
NONE/numerator-denominator precedence。不同 runner 可对同一 evidence 得出不同
首失败 gate。Revision 必须逐条件冻结实际值、required 值、zero/not-estimable
semantics、`NOT_EVALUATED` precedence 和唯一 classification。

### P1-5 Required Outputs 的声明数量与枚举集合矛盾

Plan `:424-458` 声称“Exactly 24 non-cache artifacts”，但枚举为：

- 9 contracts；
- 13 support CSVs；
- 1 summary；
- 1 classification；
- 1 manifest。

总数是 **25**，不是 24。runner 无法同时满足数量断言和路径集合断言。
Revision 必须确定 exact set，并冻结 manifest 是否排除自身、missing/extra
的 fail-closed 规则。

### P1-6 新增 epoch evidence 没有 frozen schema，关键 claims 无法独立审计

Plan 新增：

- `fixed_epoch_thinning_contract.json`
- `epoch_support_by_date.csv`
- epoch-aware `candidate_ledger.csv`
- `slice_invariance.csv`

但 `:424-458` 只列文件名，没有冻结字段、row grain、类型和 identity。至少
需要独立审计：

- capture/date/segment/epoch/direction；
- epoch start/core bounds；
- raw natural-onset count、edge-guard omission、earliest retained identity、
  same-key suppressed count；
- dual-direction anchor count和 shared cluster identity；
- segment-boundary disposition；
- filter admission/confirmation；
- slice expected/actual identity sets、support counts和 mismatch reason。

没有这些 schema，24/25-file closure 只能证明文件存在，不能证明新 anchor
contract 被执行。

## P2 Findings

### P2-1 Exposure 没有冻结 segment/epoch boundary disposition

Plan `:343-351` 只要求 `t` 位于 core、M-state supported 和 comparison
supported。它没有说明含 reset 的 epoch 是否进入 exposure、partial segment
在 core 内如何计时、以及同一 epoch 多 segment clusters 对应哪个 denominator。

在 P1-1 的语义选择完成后，必须让 observed、selection-null、evaluation-null
和 raw exposure 使用同一个 epoch/segment eligibility mask，并冻结逐
filter-duration numerator/denominator identity。

### P2-2 Hostile tests 缺少最危险的 boundary/cluster 反例

Plan `:460-480` 应补充：

- segment reset 发生在 core 内，reset 前后同方向各有 onset；
- reset 前后相反方向 onset 是否共享或拆分 cluster；
- 一个 epoch 有多个 segments 时 anchor/cluster 上限；
- observed/null 两方向同 cluster 的 distinct-count dedup；
- boundary epoch 在 exposure、selection、evaluation 和 raw 中的相同处置；
- every filter-duration numerator/denominator identity；
- A-1-2/A-1-3/A-1-4 zero/nonfinite/NOT_EVALUATED precedence；
- exact output schema、row grain 和 manifest count mutation。

### P2-3 Raw 5s burst gate 在预期单-cluster-per-epoch语义下近乎结构性恒真

Plan `:416-419` 保留 `maximum 5s burst <= 2`。若 P1-1 最终保证一个连续
capture 每 epoch 只有一个 cluster，且相邻 core 相隔 30s，则 5s burst
理论上至多 1；该 gate 不再提供经验 sparsity 检验。若它超过 1，反而主要
暴露 segment fragmentation 或 overlapping capture bookkeeping。

Revision 应将其明确降为 integrity diagnostic，或注册一个对 fixed-epoch
thinning 仍有辨识力的 burst/occupancy gate，避免把设计恒等式报告成数据
支持。

### P2-4 历史复用与 claim limit 没有完整继承

Plan `:58-59` 只说通过后可起草 A0，没有明确声明九个日期均已被多轮
hypothesis revision 使用、不是 prospective holdout，也没有冻结最强正面
claim 仅为 historical structural candidate。Revision 应显式保留 historical
reuse ledger、no-prospective-claim 和 no-economic-precision wording。

## Confirmed Design Strengths

- Unix-zero 60s epoch arithmetic本身是 causal、外生且不依赖 accepted-anchor
  timestamp。
- `[15s,45s)` 左闭右开边界明确，edge omission 符合 precision-first
  `cost(FP) >> cost(FN)`。
- 每方向最早 onset 可在线因果决定，不需要未来 core 数据。
- 同一 segment/epoch 内双方向独立 thinning、共享 cluster 的目标语义合理。
- 0.50/0.25 thresholds、three-channel consensus、UNKNOWN/ABSTAIN 和
  outcome lock 均未被放宽。
- selection/evaluation bank codes 5/6 分离，199-replicate 和 inherited
  A-1-5/A-1-6 thresholds 数值上没有直接矛盾。

## Freeze Decision

- **不可冻结。**
- 当前 candidate 不满足 `P0/P1/P2/P3=0/0/0/0`，task 中的 29-cache lock
  不得释放。
- Revision 2 应先解决 P1-1 的 boundary epoch语义，再统一修改 slice、
  exposure、null、gates、evidence schemas和 hostile tests；不得通过观察
  candidate counts 调整 epoch origin、60s width 或 `[15s,45s)` core。
