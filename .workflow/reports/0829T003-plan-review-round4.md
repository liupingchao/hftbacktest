# 0829T003 Hostile Plan Review Round 4

日期：
- 2026-08-29 18:02 CST

审查对象：
- task：`.workflow/tasks/0829T003.md`
- candidate plan：
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md`
- review commit：
  `4104de59f1e0314b764e3aaa5a38fae253df0570`
- candidate plan SHA256：
  `e2a914843fad622c9e8b28233453bf8cb3de1b5eef3da309f6e9fe7cb7c8be16`
- Round 3 baseline：`P0/P1/P2/P3 = 0/4/4/0`

审查约束：
- 独立 hostile scientific-contract review。
- 未运行 29-cache，未读取 future outcomes。
- 未修改 plan、task、runner、tests 或研究结果。
- 本轮只新增本 review report。

## Verdict

- **FAIL / 不可冻结**
- **P0/P1/P2/P3 = 0/3/3/0**
- 29-cache execution lock 必须继续保持。

Revision 4 已实质关闭 finite `K_segment`、qualifying-only rows、
selection-independent A-1-2 burst、coverage units和 A-1-7 precedence。
剩余阻断来自三处跨合同冲突：per-epoch order仍不能保证 global
`ts_ns` order、slice-source hash重新消费 unconsumed fields、raw occupancy
subset/hash violation没有唯一 gate归属。

## P0 Findings

无。

## P1 Findings

### P1-1 Per-epoch raw-order predicate 仍会漏掉跨 epoch 的全局乱序

Plan `:286-307` 已正确要求 `O_e` 按 raw row order严格匹配 ordered
`E_e`，可以识别同一 epoch 内的 permutation。但它没有冻结整个 capture：

```text
ts_ns[0] < ts_ns[1] < ... < ts_ns[n-1]
```

反例：交换 epoch `e` 的最后一个 timestamp和 epoch `e+1` 的第一个
timestamp。按各自 epoch过滤后：

- `O_e == E_e`；
- `O_{e+1} == E_{e+1}`；
- 两个 epoch都可被标为 `eligible`；
- 但完整 raw array在 epoch boundary处 non-monotonic。

该反例直接影响：

- `:393` 的 `searchsorted`，它要求全局有序数组；
- `:438-439` 的 `ts_ns[0]` / `ts_ns[-1]` endpoint authority；
- rolling feature、segment和 slice index的时间顺序。

Hostile test `:964` 的 “permuted/non-monotonic raw row order” 不能替代
缺失的 executable gate。Revision 必须在 source preflight冻结 global
strictly-increasing `ts_ns`；任何 violation应唯一失败于 A-1-0并令后续
gates `NOT_EVALUATED`。若选择 disposition语义，则还必须说明跨 epoch乱序
如何唯一污染 disposition，但 source-level fail-closed更直接。

### P1-2 `slice_source_sha256` 违反 unconsumed-field poison boundary

Plan `:170-172` 和 A-1-1 `:673-678` 明确要求 midpoint、OBI、spread、
depth snapshots及其他 unconsumed values不影响任何输出。

Revision 4 在 `:407-422` 精确列出的 27 个 sliced/copied fields中包含 15 个
不属于 bound `CONSUMED_CACHE_FIELDS` 的字段，包括：

```text
ask_depth, bid_depth, midpoint, obi, spread_ticks,
bin_boundary_violations, cache_schema_version,
initial_bridge_failure_count, non_admitted_message_contributions,
quality_boundary_count, reset_count, segment_end_ids, segment_end_ts,
sequence_gap_count, tick_size
```

`:429-436` 随后把每个 field的 C-contiguous raw bytes纳入
`slice_source_sha256`，该 hash又作为输出写入 `slice_invariance.csv`
`:864-874`。因此只 poison `obi` 或 `midpoint`，detector本身可以完全不变，
但输出 hash必然改变，直接违反 “poison of unconsumed values changes no
output”。

Revision 必须把 value-bearing slice identity限制为 exact consumed-field
set。为证明 temporary cache schema未变，可以对 unconsumed fields只记录
field name/dtype/shape，不能把其 value bytes写入任何 output identity；
hostile test必须 poison每个 unconsumed field并比较完整 25-artifact roots。

### P1-3 Raw occupancy subset/hash violation没有绑定到唯一失败 gate

Plan `:889-924` 已新增：

- raw-supported/occupied counts和 identity hashes；
- `occupied_subset_violation_count`；
- “occupied identities must be an exact subset”。

但 A-1-4 `:710-725` 没有要求：

```text
occupied_subset_violation_count == 0
count == reconstructed identity-set cardinality
stored hash == reconstructed canonical hash
occupied count <= raw-supported count
```

A-1-7 `:771-775` 也只检查 raw exposure、rate和 share threshold。于是 runner
可以输出 nonzero subset violation或不匹配的 hash/count，同时所有 frozen
gate conditions仍为 PASS；另一个实现又可以把它归为 A-1-0 schema failure
或 A-1-4 numeric failure。首失败 classification不唯一。

Revision 必须将上述四类 identity/subset conditions逐项加入一个明确 gate。
建议归入 A-1-4 selection/numeric integrity，并冻结 exact actual/required/
status evidence；A-1-7只保留在已验收 identity sets上计算的 sparsity阈值。

## P2 Findings

### P2-1 Structural market-time diagnostic 的 optional semantics和identity不足

Plan `:904-906` 允许 `structurally_occupied_epoch_share` 为 finite float或
null，`:919-922` 给出 numerator/denominator文字，但没有冻结：

- `structurally_eligible_epoch_count=0` 时 share必须为 null；
- positive denominator且zero occupied时 share必须为 `0.0`；
- structural numerator必须是 eligible denominator的 subset；
- structurally eligible identity-set hash。

该指标虽不参与 sparsity gate，但属于 exact summary schema和 historical
diagnostic。Revision 应补齐逐指标 zero/null语义；若保留 independent-audit
claim，应增加 structural denominator identity hash或明确该诊断仅为
可重算 count而不作 identity claim。

### P2-2 `gate_contract.json` 的 `NOT_EVALUATED` 字段值仍不唯一

Plan `:923-924` 规定每个 gate condition输出 `actual`、`required`、
`passed`、`status`，但 `status=NOT_EVALUATED` 时没有规定：

- `passed` 是 `false` 还是 `null`；
- `actual` 是 `null`、未计算 sentinel还是已产生但禁止解释的值；
- `required` 是否仍保留 frozen threshold；
- condition rows是否仍完整存在并保持固定顺序。

这些差异会改变 canonical artifacts和 Build A/B evidence semantics。
Revision 应冻结：所有 condition rows始终存在、`required`始终保留、
`status=NOT_EVALUATED` 时 `passed=null`且 `actual=null`，或选择另一套
同样唯一的 exact schema。

### P2-3 Hostile tests 未覆盖本轮剩余交叉反例

Plan `:954-1010` 已覆盖 Round 3 指定 cases，但还缺：

- 只交换相邻两个 epoch边界 timestamp，两个 per-epoch `O_e`仍各自完整；
- poison `obi`/midpoint/depth/metadata不能改变 `slice_source_sha256`或
  任何 artifact；
- subset violation、count/cardinality mismatch和 identity-hash mismatch
  必须唯一失败于指定 gate；
- structural occupancy denominator-zero/positive-denominator-zero semantics；
- `NOT_EVALUATED` condition的 actual/required/passed/status exact sentinels
  和固定 row presence/order。

## Round 3 Closure Matrix

| Round 3 finding | Revision 4 status | Round 4 disposition |
|---|---|---|
| P1-1 raw timestamp order | 部分闭合 | intra-epoch raw order已闭合；cross-epoch global order反例见 P1-1 |
| P1-2 finite `K_segment` / qualifying-only rows | **闭合** | positive finite K、termination、no early stop、qualifying-only row grain和summary counters均已冻结 |
| P1-3 selection-independent A-1-2 burst | **闭合** | 改为所有 common retained clusters，位于filter/selection之前，gate dependency正确 |
| P1-4 raw occupancy summary/hash/subset schema | 部分闭合 | summary fields和hash已补；subset/hash/count violations未绑定gate，见 P1-3 |
| P2-1 field-name authority/endpoints | 主体闭合 | 27-field partition精确覆盖 authority schema且无重叠，endpoints绑定当前 `ts_ns`；unconsumed value hash冲突见 P1-2 |
| P2-2 coverage units | **闭合** | distinct key、跨rows dedup、checkpoint与27x tuple units均已冻结 |
| P2-3 A-1-7 precedence | **闭合** | 仅在执行到 A-1-7时zero raw support失败，早期失败时明确 `NOT_EVALUATED` |
| P2-4 hostile tests | 部分闭合 | Round 3 cases已加入；本轮交叉反例见 P2-3 |

## Confirmed Closures

以下项目本轮无 finding：

- HEAD 精确为
  `4104de59f1e0314b764e3aaa5a38fae253df0570`。
- candidate plan SHA256 精确为
  `e2a914843fad622c9e8b28233453bf8cb3de1b5eef3da309f6e9fe7cb7c8be16`。
- Revision 4的 row-aligned 17 fields和non-row-aligned 10 fields无重叠，
  合集精确等于 bound 27-field allowed schema。
- finite `K_segment`、iteration termination、no early stop和
  qualifying-only CSV grain唯一可执行。
- Comparable epoch identity固定为 `(capture_id,epoch_id)`并跨 starts
  deduplicate；checkpoint count与27-filter tuple count已区分。
- Common retained-cluster 5s diagnostic不再依赖null selection。
- Raw support denominator已保持 support-conditioned，metric-specific
  share null/zero主体语义正确。
- A-1-7 sequential precedence已闭合。
- 25 non-cache artifact路径仍精确，manifest排除自身并列其余24项。
- predecessor runner/null authority、conditional H0、full null
  recomputation、bank independence和 historical-only claim未被削弱。

## Freeze Decision

- **不可冻结。**
- 当前 Revision 4 为 `P0/P1/P2/P3 = 0/3/3/0`，未达到
  `0/0/0/0`。
- `.workflow/tasks/0829T003.md` 的 29-cache execution lock 不得释放。
- 下一 revision应只修复 global timestamp preflight、unconsumed-safe
  slice identity、raw subset/hash gate和相关 exact schema/tests；不得调整
  epoch geometry、thresholds、filter order、selection/null banks或
  outcome boundary。
