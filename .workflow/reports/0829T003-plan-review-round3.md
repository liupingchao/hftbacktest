# 0829T003 Hostile Plan Review Round 3

日期：
- 2026-08-29 17:56 CST

审查对象：
- task：`.workflow/tasks/0829T003.md`
- candidate plan：
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md`
- review commit：
  `e9f7f7841fd1020079a15e8ec3e33a23c3d99216`
- candidate plan SHA256：
  `e3aef483b8e4bf13de6503897d60bde20586eff482ff2ebab2cc382878cab125`
- Round 2 baseline：`P0/P1/P2/P3 = 0/6/4/0`

审查约束：
- 独立 hostile scientific-contract review。
- 未运行 29-cache，未读取 future outcomes。
- 未修改 plan、task、runner、tests 或研究结果。
- 本轮只新增本 review report。

## Verdict

- **FAIL / 不可冻结**
- **P0/P1/P2/P3 = 0/4/4/0**
- 29-cache execution lock 必须继续保持。

Revision 3 已关闭 sliced-raw direct rebuild、exact support tuple、metric-specific
null semantics、support-conditioned occupancy 主公式、sequential
`NOT_EVALUATED` 文本和 5s window geometry的大部分缺口。但 epoch partition
仍有一个反例，artificial-start schedule/evidence row grain 仍非唯一，A-1-2
依赖尚未验收的 selected filter，新增 raw occupancy 也没有完整 frozen
evidence schema。

## P0 Findings

无。

## P1 Findings

### P1-1 Epoch `if/elif` partition 漏掉 timestamp permutation

Plan `:265-274` 将 structurally eligible 定义为完整 grid，包含
“all adjacent deltas = 20ms”；但实际 ordered partition `:276-304` 的
`irregular_checkpoint` 只检查：

```text
duplicate
off E_e
set(O_e) != set(E_e)
```

若 `O_e` 含有 E_e 的全部 3,000 个唯一 timestamp，但 raw row order 中两个
timestamp 被交换，则：

- `set(O_e) == set(E_e)`；
- 无 duplicate；
- 无 off-grid timestamp；
- 单一 segment 时最终进入 `eligible`；
- 但 observed adjacent deltas 不再全为 20ms。

因此完整-grid定义与 executable disposition predicate 给出不同结论，
`:649` 的 six-way partition 没有唯一 expected value。Revision 必须明确
`O_e` 是否先按 timestamp 排序；若不得排序，则任何 non-increasing timestamp
或 observed-order adjacent delta 不等于 20ms 必须进入
`irregular_checkpoint`。Hostile tests 也必须加入 timestamp permutation/
non-monotonic mutation。

### P1-2 Artificial-start schedule 和 `slice_invariance.csv` row grain 仍非唯一

Plan `:382-392` 定义：

```text
first checkpoint + k * 600s
```

但没有冻结 `k` 的起止集合或终止条件。按纯数学文字，`k` 可以无限增长；
按实现习惯，又可分别解释为：

- nominal timestamp 不超过 segment last timestamp；
- searchsorted index 不超过 capture end；
- 遇到第一个 absent/wrong-segment 后停止；
- 为每个 segment 预先计算有限 `floor(duration/600s)` 集合。

这些路径产生不同的 nominal/skipped counts。

同时 `:391-392` 和 `:425-435` 规定 absent/wrong-segment 或
`no_comparable_epoch` 会被 skipped，`:462-464` 却要求每个 artificial start
产生一行，`:826-837` 的 schema 又没有 `start_disposition`，且
`:862-877` 不允许 absent `actual_start_ts_ns`、comparison fields 或 hashes
使用 sentinel。实现可以合法地：

- 完全不写 skipped rows，只在 summary 计数；
- 写 skipped rows并自行选择整数/空字符串 sentinel；
- 把 “artificial start” 重新定义为 qualifying start。

Revision 必须冻结 finite `k` domain，并唯一选择：

1. CSV 只写 qualifying rows，row grain 明确改为 qualifying artificial
   start，所有 skipped cases 仅进入 exact summary counters；或
2. CSV 写全部 nominal starts，新增 exact disposition 和所有 N/A sentinel。

### P1-3 A-1-2 的 selected raw burst 依赖尚未通过 A-1-3/A-1-4 的 selection

Plan `:569-579` 规定 held-out selected filter 由 selection null bank 决定；
`:608-624` 又要求 gates 严格顺序执行。可是 A-1-2 `:643-661` 包含
`maximum raw 5s cluster burst <= 1`，其 exact domain `:740-745` 明确使用：

```text
unique selected raw dependence_cluster_id values
```

因此 A-1-2 的输入依赖：

- A-1-3 才验收的 null bank、stream 和 structural-null admissibility；
- A-1-4 才验收的 fold isolation、selection access 和 selected-filter
  integrity。

若 selection/null 有缺陷并导致 burst 异常，当前 precedence 会先分类为
`Aminus1_mstate_integrity_failed`，随后把真正负责验证 selection/null 的
A-1-3/A-1-4 标成 `NOT_EVALUATED`。这违反 sequential gate 的依赖拓扑。

Revision 必须二选一：

- A-1-2 对每个 raw filter 或 common epoch-cluster set 计算结构性 burst，
  完全不依赖 selected filters；或
- 将 selected-filter burst 移到 A-1-4 之后，并冻结对应 classification。

由于 fixed epoch geometry 使该指标只承担 bookkeeping integrity，第一种
更直接。

### P1-4 新增 raw occupancy/gate evidence 没有 frozen output schema

Plan `:588-606` 新增：

- `raw_supported_epoch_count`；
- `occupied_epoch_count`；
- `occupied_supported_epoch_share`；
- structurally eligible market-time occupancy diagnostic。

这些值直接参与 A-1-4/A-1-7 `:673-688`、`:734-737`。但 Required Outputs
`:748-877` 没有冻结：

- 它们位于 `reports/A_minus1_summary.json`、某个 CSV 还是 gate artifact；
- exact field names、类型和 null sentinel；
- occupied set 与 raw-supported denominator identity/hash；
- market-time diagnostic 的 exact numerator、denominator、field name；
- gate result 中 actual/required/status 的 evidence位置。

`:839-849` 的 “summary additionally records” 只列 slice coverage counters，
不包含 raw occupancy。继承的 T002 summary raw object也只有 raw cluster
count/exposure/rate/burst，并没有这些新字段。于是 runner 可以计算 gate，
却不输出足够证据让独立 QA 重建 denominator/numerator identity。

Revision 必须冻结 `A_minus1_summary.json` 的 changed `raw`/`integrity`
schema，至少包含两个 counts、share、optional-value状态、identity hashes和
market-time diagnostic；对应 hostile tests 必须 mutation 每个字段及
numerator-subset关系。

## P2 Findings

### P2-1 Slice raw-field分类和 capture endpoint authority 应按字段名冻结

Plan `:397-404` 以 “first dimension equals checkpoint count” 判定
row-aligned field。该规则是 shape-dependent，而不是 schema-dependent：
一个 metadata array 若长度偶然等于 checkpoint count 会被切片；一个应随
slice 重算的 metadata field则可能被原样复制。

此外 epoch universe `:249-263` 使用 `capture_first_ts_ns` /
`capture_last_ts_ns`，但没有明确它们必须分别来自当前分析数组
`ts_ns[0]` / `ts_ns[-1]`，尤其 slice cache复制了非 row-aligned metadata 后，
不得继续使用 full-capture endpoint。

Revision 应列出 exact row-aligned field-name set，并冻结 full/sliced
capture endpoints均来自对应 preflighted `ts_ns` array；source identity
应记录该 field set。

### P2-2 Comparable coverage 的 distinct identity 和 count unit 尚需精确化

Plan `:467-476` 要求至少 30 个 “distinct comparable epochs globally”，
但没有明确 distinct key 是：

```text
(capture_id, epoch_id)
```

还是只有 `epoch_id`。同一个 comparable epoch也可能出现在多个 artificial
start rows中，必须明确跨 rows deduplicate。

另外 support identity是一 checkpoint/filter 一 tuple `:443-460`，
但 summary字段叫 `compared_support_checkpoint_count` `:839-849`；
它可以表示 unique checkpoint count，也可以表示 27 倍的 support tuple
count。当前 gate只检查 positive，仍应冻结 exact unit和与
`expected_support_count` 的关系，避免执行报告中的 coverage 数字不可比。

### P2-3 Zero raw support 的 A-1-7 文案应服从完整 sequential precedence

Plan `:602-604` 写：

```text
raw_supported_epoch_count=0 -> A-1-7 fails after A-1-4 verifies null
```

但 `:608-624` 又规定任何 A-1-5/A-1-6 failure 都使 A-1-7
`NOT_EVALUATED`。由于 raw exposure不应小于 externally censored primary
exposure，zero raw support通常也会使 A-1-5先失败。

Revision 应改为：

```text
A-1-4 validates denominator-consistent null;
if and only if execution reaches A-1-7, zero raw support fails A-1-7;
otherwise A-1-7 is NOT_EVALUATED.
```

这不改变 gate order，但消除报告状态冲突。

### P2-4 Hostile tests 仍缺本轮发现的交叉反例

Plan `:879-925` 已覆盖 Round 2 所列的大部分 mutation，但还缺：

- unique full timestamp set with permuted/non-monotonic raw row order；
- finite `k` termination和最后一个 nominal start边界；
- skipped absent/wrong-segment/no-comparable row-grain/sentinel mutation；
- corrupt selection/null不能提前制造 A-1-2 selected-burst failure；
- raw occupancy output field、identity hash和 numerator-subset mutation；
- row-aligned field-name set mutation和 stale full-capture endpoint；
- comparable epoch跨 artificial starts dedup；
- support checkpoint count与 support tuple count单位错换；
- A-1-5先失败时 A-1-7 必须 `NOT_EVALUATED`。

## Round 2 Closure Matrix

| Round 2 finding | Revision 3 status | Round 3 disposition |
|---|---|---|
| P1-1 epoch universe/if-elif predicates | 部分闭合 | universe、empty epoch、precedence 已冻结；timestamp permutation 反例见 P1-1 |
| P1-2 sliced raw cache direct rebuild | 主体闭合 | direct-call `build_features` 和禁止 derived reuse 已明确；field-name/endpoints 加固见 P2-1 |
| P1-3 exact support identity tuple/hash | **闭合** | four-state tuple、typed sort、canonical JSON 已冻结且状态编码与 bound runner一致 |
| P1-4 non-vacuous slice coverage | 部分闭合 | 4 dates/30 epochs/positive support thresholds 已补；finite schedule和row grain见 P1-2 |
| P1-5 metric-specific null semantics | **闭合** | denominator-zero null和positive-denominator zero均已逐指标冻结 |
| P1-6 schema sentinels/typed canonicalization | 部分闭合 | segment/candidate/cluster sentinel和typed ordering已补；new raw evidence与skipped slice schema见 P1-2/P1-4 |
| P2-1 support-conditioned occupancy | 主公式闭合 | denominator 已改为 raw-supported epochs；exact output evidence仍见 P1-4 |
| P2-2 sequential `NOT_EVALUATED` | 文本闭合、依赖未闭合 | sequential规则已补；selected burst gate inversion见 P1-3，zero wording见 P2-3 |
| P2-3 5s domain | 几何闭合 | per-capture、half-open、dedup和epoch timestamp已冻结；selected-filter依赖见 P1-3 |
| P2-4 hostile tests | 部分闭合 | Round 2 cases基本加入；本轮交叉反例见 P2-4 |

## Confirmed Closures

以下项目本轮无 finding：

- HEAD 精确为
  `e9f7f7841fd1020079a15e8ec3e33a23c3d99216`。
- candidate plan SHA256 精确为
  `e3aef483b8e4bf13de6503897d60bde20586eff482ff2ebab2cc382878cab125`。
- predecessor runner和null authority的 blob/whole-file SHA匹配。
- plan列出的 15 个 M-state/selection callable和 7 个 null callable
  normalized AST SHA256全部重算匹配。
- M-state support编码 `SIGNAL_NEG=-1`、`BACKGROUND=0`、
  `SIGNAL_POS=1`、`ABSTAIN=2` 与 exact support tuple一致。
- Sliced analysis已明确从 raw cache direct-call bound `build_features`，
  禁止复用 full-run rolling features/M-states。
- Support-conditioned occupancy不再把 unsupported structurally eligible
  epochs直接计入 sparsity denominator。
- Metric-specific zero/null semantics主体无矛盾。
- 25 non-cache artifact路径枚举仍精确为 25，manifest排除自身并列其余
  24 项。
- conditional H0、full null recomputation、selection/evaluation bank
  independence和 historical-only claim limit未被削弱。
- execution lock仍要求独立 review达到 `0/0/0/0`。

## Freeze Decision

- **不可冻结。**
- 当前 Revision 3 为 `P0/P1/P2/P3 = 0/4/4/0`，未达到
  `0/0/0/0`。
- `.workflow/tasks/0829T003.md` 的 29-cache execution lock 不得释放。
- 下一 revision 应只修复上述 predicate、slice schedule/evidence、
  gate dependency和schema闭包；不得修改 Unix-zero origin、60s width、
  `[15s,45s)` core、`0.50/0.25` thresholds、three-channel consensus或
  null bank阈值。
