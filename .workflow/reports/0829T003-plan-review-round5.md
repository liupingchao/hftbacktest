# 0829T003 Hostile Plan Review Round 5

日期：
- 2026-08-29 18:07 CST

审查对象：
- task：`.workflow/tasks/0829T003.md`
- candidate plan：
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md`
- review commit：
  `a561981054e5ad50a48a18626b50e8b84186121e`
- candidate plan SHA256：
  `c8113c0358be91354c9263fcaa352d11bedee710687d3aa1dfab75b33d26ea0e`
- Round 4 baseline：`P0/P1/P2/P3 = 0/3/3/0`

审查约束：
- 独立 hostile scientific-contract review。
- 未运行 29-cache，未读取 future outcomes。
- 未修改 plan、task、runner、tests 或研究结果。
- 本轮只新增本 review report。

## Verdict

- **FAIL / 不可冻结**
- **P0/P1/P2/P3 = 0/2/1/0**
- 29-cache execution lock 必须继续保持。

Revision 5 已关闭 global timestamp A-1-0、slice hash的直接 unconsumed-value
依赖、raw/structural count/hash/subset gate、structural optional semantics
和 `NOT_EVALUATED` exact sentinels。剩余两个 P1 都是可构造的 evidence
counterexample：正确 counts/hashes仍可搭配伪造 share通过 gate；poison test
也尚未冻结相对于 immutable source-authority SHA的注入位置。

## P0 Findings

无。

## P1 Findings

### P1-1 Occupancy share没有 A-1-4 算术一致性 gate

Plan `:641-653` 定义：

```text
occupied_supported_epoch_share =
  occupied_epoch_count / raw_supported_epoch_count
```

并在 `:917-936` 冻结 raw/structural counts和 share字段。A-1-4
`:729-753` 已检查：

- counts等于 reconstructed set cardinalities；
- hashes等于 reconstructed canonical hashes；
- subset violations为零；
- denominator-zero和zero-occupied optional semantics。

但它没有要求 positive numerator/denominator时：

```text
occupied_supported_epoch_share
  == occupied_epoch_count / raw_supported_epoch_count

structurally_occupied_epoch_share
  == structurally_occupied_epoch_count /
     structurally_eligible_epoch_count
```

直接反例：

```text
raw_supported_epoch_count = 100
occupied_epoch_count = 20
counts/hashes/subset全部正确
occupied_supported_epoch_share = 0.01
```

当前 A-1-4只要求该 share finite；A-1-7 `:799-802` 会错误地把 `0.01`
判为 `<=0.10`，而真实 share是 `0.20`。因此 positive-count路径可以绕过
raw sparsity gate。

Revision 必须在 A-1-4 增加两个 share arithmetic identity conditions，并
冻结比较方法。为避免浮点容差歧义，gate可使用整数交叉乘法：

```text
raw A-1-7 threshold:
  10 * occupied_epoch_count <= raw_supported_epoch_count
```

输出 share仍应由同一 counts唯一计算并按 frozen float serialization写出。
Hostile tests必须用正确 counts/hashes但错误 finite share做 mutation。

### P1-2 Poison test没有冻结相对于 source-authority SHA的注入层

Revision 5 `:440-454` 已正确规定 slice-derived identity只哈希 consumed
field values，unconsumed fields只贡献 name/dtype/shape；这关闭了 Round 4
的直接 hash泄漏。

但 hostile test `:1027-1028` 要求：

```text
poison every unconsumed value
-> all 25 artifacts unchanged
```

却没有冻结 poison发生在哪个层级。Bound source authority会在任何分析前：

- 对完整 cache file做 full-file SHA256校验；
- 把 `cache_sha256` 写入 `source_cache_inventory.csv`。

对应 accepted runner：

- `examples/hyperliquid/skhynix_fresh_channel_consensus_mstate_a_minus1.py:
  316-354` 验证完整 cache identity；
- 同文件 `:2468-2479` 将 full-file SHA写入 required output。

因此：

- 若 poison直接修改 cache或poisoned cache作为 authority input，
  A-1-0 必须失败，25 artifacts不可能保持相同；
- 若 poison在 canonical source authority已验证后，仅注入到内存中的
  unconsumed arrays，再进入 slice/field routing，则25 artifacts可以相同。

两种实现都符合当前 “poison unconsumed values” 文字，却得到相反 expected
result。`:454` 的 “unconsumed value bytes never enter ... any other output
identity” 也与 immutable source inventory的 full-file SHA字面冲突。

Revision 必须冻结 poison protocol：

1. 先对 canonical unmodified cache完成 A-1-0 authority和inventory evidence；
2. 在 authority成功后、任何 detector/slice field routing之前，对独立
   in-memory copy的每个 unconsumed field注入 poison；
3. poison run复用 canonical authority evidence，不重定义 source cache；
4. 对全部25个 finalized artifact roots做 exact comparison；
5. source-file mutation仍单独属于 A-1-0 hostile test，不得与 A-1-1 poison
   test混合。

## P2 Findings

### P2-1 Hostile tests缺少 arithmetic spoof和poison injection-order反例

Plan `:1005-1068` 已覆盖 Round 4 的 timestamp、hash/subset、structural
optional和 `NOT_EVALUATED` cases，但仍缺：

- counts/cardinalities/hashes全部正确，只有 raw finite share被改小；
- structural counts/hash正确，只有 structural finite share错误；
- poison发生在 source-authority之前必须唯一失败 A-1-0；
- poison发生在 authority之后的 in-memory unconsumed copy必须保持25项
  artifact roots完全不变；
- A-1-0 source-file mutation和 A-1-1 unconsumed poison不得共享 expected
  classification。

## Round 4 Closure Matrix

| Round 4 finding | Revision 5 status | Round 5 disposition |
|---|---|---|
| P1-1 global timestamp A-1-0 | **闭合** | nonempty/global strictly increasing在epoch/searchsorted前检查，violation唯一 A-1-0 |
| P1-2 unconsumed-safe slice hash/poison | 部分闭合 | derived slice hash已不消费unconsumed values；poison相对source authority的注入层仍见 P1-2 |
| P1-3 raw occupancy count/hash/subset gate | 主体闭合 | count/cardinality/hash/subset均绑定A-1-4；positive-count share arithmetic仍见 P1-1 |
| P2-1 structural optional/hash/subset | 主体闭合 | zero/null、eligible hash和subset已补；share arithmetic仍见 P1-1 |
| P2-2 `NOT_EVALUATED` sentinels | **闭合** | fixed rows/order、`passed=null`、`actual=null`、required保留均唯一 |
| P2-3 hostile tests | 部分闭合 | Round 4 cases已加入；剩余反例见 P2-1 |

## Confirmed Closures

以下项目本轮无 finding：

- HEAD 精确为
  `a561981054e5ad50a48a18626b50e8b84186121e`。
- candidate plan SHA256 精确为
  `c8113c0358be91354c9263fcaa352d11bedee710687d3aa1dfab75b33d26ea0e`。
- 22个 bound callable normalized AST SHA256全部重算匹配。
- Required Outputs仍精确为25项，manifest排除自身并列其余24项。
- Global nonempty/strictly-increasing timestamp preflight和 A-1-0
  precedence唯一。
- Slice-derived value hash仅包含 exact consumed fields；unconsumed schema
  metadata不包含value bytes。
- Raw/structural identity counts、canonical hashes和subset violations均已
  绑定 A-1-4。
- Structural denominator-zero/positive-denominator-zero optional semantics
  已冻结。
- `NOT_EVALUATED` condition rows、actual/required/passed/status sentinels已
  唯一。
- Epoch geometry、M-state thresholds、conditional H0、bank independence和
  historical-only claim未被修改。

## Freeze Decision

- **不可冻结。**
- 当前 Revision 5 为 `P0/P1/P2/P3 = 0/2/1/0`，未达到
  `0/0/0/0`。
- `.workflow/tasks/0829T003.md` 的 29-cache execution lock 不得释放。
- 下一 revision只需冻结 occupancy share arithmetic identity和 poison
  injection protocol及对应 hostile tests；不得调整 epoch geometry、
  thresholds、filter order、selection/null banks或 outcome boundary。
