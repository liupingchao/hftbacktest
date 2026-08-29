# 0829T003 Hostile Plan Review Round 2

日期：
- 2026-08-29 17:46 CST

审查对象：
- task：`.workflow/tasks/0829T003.md`
- candidate plan：
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md`
- review commit：
  `63882399eb455c24a62ed41a7d9dbd8cc8c724c2`
- candidate plan SHA256：
  `78054b202a0991196bacad8509d79b7d6b314e4c61d2f172e9547930938b957b`
- Round 1 baseline：`P0/P1/P2/P3 = 0/6/4/0`

审查约束：
- 独立 hostile scientific-contract review。
- 未运行 29-cache，未读取 future outcomes。
- 未修改 plan、task、runner、tests 或研究结果。
- 本轮只新增本 review report。

## Verdict

- **FAIL / 不可冻结**
- **P0/P1/P2/P3 = 0/6/4/0**
- 29-cache execution lock 必须继续保持。

Revision 2 已经实质修复 Round 1 的 direct null authority/H0、固定 epoch
cluster 主体语义、25-artifact 数量闭包和 historical-only claim limit，但
complete-epoch disposition、slice 重算、support identity、gate optional-value
语义和 evidence schema 仍未达到唯一可执行合同。

## P0 Findings

无。

## P1 Findings

### P1-1 六类 epoch disposition 仍没有唯一、穷尽且可直接实现的谓词

Plan `:247-275` 冻结了完整 epoch 的必要条件、六个 disposition 名称和
precedence，但没有冻结：

- 每个 capture 要枚举哪些绝对 epoch，包括 capture 内完全没有 checkpoint
  的空 epoch 是否必须产生 evidence row；
- `partial_capture_start/end` 的精确边界谓词；
- checkpoint 数不足时何时是 `missing_checkpoint`，何时是
  `irregular_checkpoint`；
- duplicate timestamp、off-grid timestamp、额外 checkpoint 和同时跨
  segment 时的判定；
- 同时满足多个异常条件时，precedence 所作用的布尔谓词。

例如 capture 覆盖一个 epoch 的两端但缺少首 checkpoint，既可解释为
`partial_capture_start`，也可解释为 `missing_checkpoint`；完整 capture
跨度中的整段空 epoch 若只按现有 row group 枚举，甚至不会进入 disposition
partition。因而 `:558` 的 “six-way disposition partition exact” 目前没有
唯一 expected value。

Revision 必须冻结 epoch universe 和六个互斥/穷尽 predicate，最好以
capture first/last timestamp、expected timestamp set equality、duplicate/
off-grid counters 和 segment cardinality 明确定义。

### P1-2 Slice 合同没有冻结“从 sliced raw source 重建 feature”的唯一路径

Plan `:347-358` 要求 “Slice arrays ... rebuild all causal M-states”，但没有
明确 sliced arrays 是六个 raw contribution fields，还是 full-run 已经计算
好的 feature/M-state arrays。

这与直接绑定的 authority 存在关键接口问题：accepted
`build_features(cache_path)` 在
`examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py:330-387`
直接从 cache path 加载 raw fields，再计算 segment-aware rolling features。
若 runner 先对 full-capture features 切片，slice 开头会保留人工 start
之前的 rolling history；若创建临时 sliced cache 再 direct-call
`build_features`，则得到另一组结果。二者都可声称满足当前文字。

Revision 必须冻结：

- artificial slice 从 source-preflight 已通过的 raw fields 开始；
- sliced raw source 的字段、起止 index、segment handling 和身份 hash；
- slice 上再次 direct-call exact `build_features`，再进入 M-state pipeline；
- 禁止切片或复用 full-run derived features/M-states。

### P1-3 M-state support identity 没有定义 identity tuple

Plan `:385-390` 要求 checkpoint-by-checkpoint support identity，`:729-735`
又要求 support count/hash，但从未定义单条 support identity 是：

- `(capture, epoch, checkpoint, filter, mstate)`；
- `(capture, epoch, checkpoint, filter, supported_bool)`；
- 仅 non-ABSTAIN checkpoint；
- 还是包含 direction、segment 或四状态枚举的其他 tuple。

不同实现会产生不同 count/hash，却都可满足 “support identity” 文字。
Revision 必须冻结 exact typed tuple、纳入的状态集合、checkpoint domain、
去重规则、排序键和 canonical JSON serialization。

### P1-4 Slice gate 可以在零 comparable support 下真空通过

Plan `:347-396` 冻结了 600s schedule 和 comparable-epoch 条件，
`slice_invariance.csv` 也记录 `comparable_epoch_count`，但 A-1-2
`:552-567` 只要求 mismatch 为零，没有最低：

- artificial-start row count；
- comparable epoch count；
- compared support checkpoint count；
- 被实际测试的 long segment/date count。

如果完整单-segment epoch 稀少，所有 artificial starts 被 skip，或每行
`comparable_epoch_count=0`，candidate/support mismatch 都可为零，A-1-2
仍会通过，但 reset/slice invariance 实际未被检验。

Revision 必须增加非零 coverage gate 和 `NOT_ESTIMABLE`/classification
语义；至少要求全局存在 comparable epochs 和 support checkpoints，并说明
是否还要求每个 qualifying segment/date 有 coverage。

### P1-5 A-1-4 的 share null precedence 与新增 occupancy share 自相矛盾

Plan `:508-523` 明确规定：

```text
eligible_epoch_count = 0 -> occupied_eligible_epoch_share = null
eligible_epoch_count > 0 and occupied_epoch_count = 0 -> share = 0
```

但 A-1-4 `:579-590` 又规定 “burden/share is null iff observed count is
exactly zero”。当 selected filters 没有 admitted cluster、但存在 eligible
epochs 时，前者要求 occupancy share 为 `0`，后者要求所有 share 为 null。
因此同一 evidence 可触发不同 numeric-integrity 结论。

Revision 必须按 metric 分开冻结 optional-value semantics：

- burden/max-single-date-share 等 observed-count conditional 指标；
- occupied-eligible-epoch share 的 denominator conditional 语义；
- `eligible_epoch_count=0` 时 A-1-4 与 A-1-7 谁先失败、后者是否
  `NOT_EVALUATED`。

### P1-6 Epoch/slice evidence schema 无法完整表达 frozen claims

Plan `:696-709` 要求 `epoch_support_by_date.csv` 每
`(date,capture,epoch)` 一行，却只有单值 `segment_id`。一个
`segment_boundary` epoch 按定义含多个 segment，因此该字段没有唯一合法
整数。相同问题还出现在 ineligible epoch 的 `dependence_cluster_id`：
`:745-747` 只规定 absent candidate IDs 为空，没有规定 absent cluster、
non-unique segment 或不适用计数的 sentinel/empty 语义。

此外 `:734-747` 的 “lexicographically sorted identity tuples” 和
“ASCII lexical over row grain followed by numeric fields” 没有冻结 typed
sort：例如整数 epoch `10` 与 `2` 的 ASCII 和 numeric 顺序不同；canonical
JSON 也没有冻结 separators、ASCII/Unicode 和 scalar normalization。

Revision 必须让 schema 能无歧义表示所有六类 disposition，例如增加
`segment_count`、canonical segment-set identity/hash，并冻结所有 N/A
字段；同时给出 candidate/support hash 的 exact typed sort tuple 和
serialization，或绑定一个 exact canonicalization callable。

## P2 Findings

### P2-1 Raw occupied-epoch denominator 会把 support loss 计为“稀疏”

Plan `:511-519` 的 denominator 是所有 non-NONE folds 中 structurally
eligible epochs，与 selected filter 是否存在 raw M-state support/exposure
无关。一个大量 ABSTAIN、几乎没有可判定 checkpoint 的 filter，可以因为
分母包含不可观测 epochs 而获得很低的 occupied share。

这不会阻止 A-1-5 的 cluster/exposure minimum，但会让 A-1-7 的
“sparsity”同时混入 feature support loss。Revision 应冻结该指标究竟是
market-time occupancy 还是 supported-epoch occupancy；若前者是刻意设计，
应另报 supported eligible epoch share，且不得把低 structural-denominator
share 单独解释为 detector sparsity。

### P2-2 `NOT_EVALUATED` precedence 只覆盖 source-preflight failure

Plan `:525-533` 只明确 source-preflight failure 时 A-1-1 至 A-1-7
`NOT_EVALUATED`。但 A-1-0 `:535-543` 还包含 plan/ancestry/callable、
cache closure、Build-root、determinism 和 output-closure failures。
当前合同没有说明这些 authority failures 是否同样禁止 detector/null 和
later gates；A-1-1 poison/outcome-boundary failure后的 downstream 状态也
未冻结。

Revision 应对每类 A-1-0 和 A-1-1 failure 给出统一 fail-closed
`NOT_EVALUATED` 矩阵，而不只处理 invalid raw source contribution。

### P2-3 5s burst integrity diagnostic 的计算域未冻结

Plan `:642-644` 已正确把 5s burst 降为 A-1-2 bookkeeping diagnostic，
但没有说明窗口是否：

- per capture 计算；
- 可跨 epoch edge；
- 可跨 overlapping captures；
- 对双方向 cluster 先按 cluster identity 去重；
- 使用闭区间还是半开区间。

在 fixed epoch geometry 下，该值理论上应至多 1；因此必须冻结 exact
cluster set/window rule，才能让超限唯一指向 bookkeeping defect。

### P2-4 Hostile tests 尚未覆盖本轮剩余歧义

Plan `:749-782` 已补充大量 Round 1 boundary tests，但还缺：

- disposition predicate overlap、整段空 epoch、duplicate/off-grid grid；
- sliced raw-feature rebuild 与 full-derived-feature leakage mutation；
- zero artificial starts/zero comparable epochs 的真空通过；
- exact support identity tuple/hash mutation；
- zero occupied、positive eligible 时 share 必须为零而非 null；
- segment-boundary epoch 的 schema sentinel/segment-set round trip；
- numeric-vs-ASCII epoch ordering和 canonical JSON mutation；
- 非 source 的 A-1-0 failure 触发 later gates `NOT_EVALUATED`。

## Round 1 Closure Matrix

| Round 1 finding | Revision 2 status | Round 2 disposition |
|---|---|---|
| P1-1 segment reset creates multiple clusters | 主体闭合 | 完整单-segment epoch 与 `(capture,epoch)` cluster 已冻结；但 disposition predicates 仍见 P1-1 |
| P1-2 slice/reset protocol underdefined | 未闭合 | schedule/comparable epoch 已补；raw feature rebuild、support identity 和 non-vacuous coverage 仍见 P1-2/P1-3/P1-4 |
| P1-3 null authority/H0 underfrozen | **闭合** | direct authority、AST、H0、full replicate recomputation 和 invariants 均已冻结 |
| P1-4 gates underdefined | 部分闭合 | gate 条件已大幅补齐；occupancy optional-value 冲突及 fail-closed precedence 见 P1-5/P2-2 |
| P1-5 24/25 output contradiction | **闭合** | `:646-684` 精确枚举 25 项，manifest 排除自身并列 24 项 |
| P1-6 new evidence schemas absent | 部分闭合 | schemas 已新增；segment-boundary/N/A/hash canonicalization 仍见 P1-6 |
| P2-1 exposure boundary disposition missing | 主体闭合 | observed/null/raw 共用 structural eligibility mask 已冻结；raw occupancy denominator 的 claim 问题见 P2-1 |
| P2-2 hostile boundary tests missing | 部分闭合 | 原列 cases 基本补齐；本轮歧义的 mutation tests 仍见 P2-4 |
| P2-3 5s burst gate structurally vacuous | 主体闭合 | 已降为 integrity diagnostic；exact calculation scope 仍见 P2-3 |
| P2-4 historical reuse/claim missing | **闭合** | `:61-70` 明确 historical development data 和最强 claim limit |

## Confirmed Closures

以下项目本轮无 finding：

- HEAD 和 review commit 精确为
  `63882399eb455c24a62ed41a7d9dbd8cc8c724c2`。
- candidate plan SHA256 精确为
  `78054b202a0991196bacad8509d79b7d6b314e4c61d2f172e9547930938b957b`。
- predecessor runner/null authority 的 commit、blob、whole-file SHA 和所列
  normalized callable AST hashes 可核验且匹配。
- direct path-swap null authority、conditional H0、每 replicate full
  pipeline recomputation、selection/evaluation bank separation和 denominator
  identity要求已闭合。
- fixed Unix-zero epoch、`[15s,45s)` core、每 capture/epoch/direction 最早
  onset、双方向共享 `(capture,epoch)` cluster 的目标语义是 causal 且不再
  依赖 accepted-anchor renewal phase。
- Required Outputs 实际枚举恰为 25 项；`run_manifest.json` 列其余 24 项。
- historical reuse、no prospective/economic/deployability claim 已明确。
- execution lock 文本仍正确要求独立 review 达到 `0/0/0/0`。

## Freeze Decision

- **不可冻结。**
- 当前 Revision 2 为 `P0/P1/P2/P3 = 0/6/4/0`，不满足唯一可执行且可独立
  审计的 scientific contract。
- `.workflow/tasks/0829T003.md` 的 29-cache execution lock 不得释放。
- 下一 revision 应先关闭六个 P1；不得通过观察 candidate counts 调整
  Unix-zero origin、60s width、`[15s,45s)` core、`0.50/0.25` thresholds
  或 three-channel consensus。
