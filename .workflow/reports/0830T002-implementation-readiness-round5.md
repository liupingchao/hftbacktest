# 0830T002 Independent Implementation Readiness Review Round 5

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `cbfc7daced79fa9d453600f0988af5459c99764f`
- remediation implementation commit:
  `4cb701f63d1b370eeaa629b903c168d7d3a1cd01`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- tag peel:
  `cbfc7daced79fa9d453600f0988af5459c99764f`

更新时间：
- 2026-08-30 21:25 CST

审查限制：
- 未打开 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier 或 tests。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/0/0**
- 29-cache formal execution lock **不得释放**。

Round 4 remediation 已把 12 个 A-1-2 condition 名称接入一个独立重算函数，
并且 integrated V00-V12 tests 现在确实通过真实 `verify_terminal()` 断言
exact first-failure 和后续 `NOT_EVALUATED`。但是 hostile review 发现，
重算仍信任两个由 producer 自报的结论字段，能够接受底层 typed counts/hashes
与 PASS 结论矛盾的 package。

同时，新增 nonempty synthetic package 并不等价于 frozen runner 能生成的
package。它为 A/B/P 创建不同 slice bytes，却把 A WorkRow SHA 写入三路
`slice_invariance.csv`；verifier 也始终只用 A WorkRow 校验每个 root。
真实 runner 会把 poison 后所有 cache fields 写入 P slice，并把该文件 SHA
写进 RAW_11 的 `slice_invariance.csv`，所以任一 nonempty slice 会因
unconsumed-field poison 改变 P 的 `slice_source_sha256`。当前 synthetic PASS
baseline 隐藏了这个 production A/P mismatch。

## Findings

### P1-1 A-1-2 “重算”仍信任 self-declared action/slice PASS 字段

`recompute_a_minus1_2_actuals()` 当前：

- 直接对 `action_partition_exact` 取反计数，而不从
  `total_action_count` 与六个 action counts 重算 partition
  （verifier lines 1284-1287）；
- 直接以 `mismatch_reason != "none"` 计算 slice mismatch，而不从五组
  expected/actual count、SHA、exact flags 以及 cross-segment count 重建
  frozen precedence（verifier lines 1297-1299）。

CSV semantic validation只检查 channel 排序/domain，以及
`mismatch_reason` 是否属于允许字符串集合
（verifier lines 842-847、962-987）。它没有要求：

```text
total_action_count
  = global_invalid + new_invalid + new_pos + new_neg + new_neutral + no_update

<surface>_exact
  = expected_count == actual_count
    and expected_sha256 == actual_sha256

mismatch_reason
  = first failed surface in
    epoch_disposition/counter/retained/status/support/cross_segment order
```

这些关系在 runner 中是派生值，而不是独立 authority：
`action_partition_exact` 来自 count arithmetic
（runner lines 764-807）；slice exact flags 和 `mismatch_reason` 来自
expected/actual identities
（runner lines 1338-1371）。

本轮不读正式 cache 的直接 probe 构造了：

```text
total_action_count=1
all six action counts=0
action_partition_exact=True

counter_exact=False
expected_counter_sha256 != actual_counter_sha256
mismatch_reason=none
```

当前 production recomputation 仍返回：

```text
action_partition_violation_count=0
slice_mismatch_count=0
```

因此三路同步篡改 typed rows、summary integrity、gate 和 classification
仍可在 V07 形成 scientific false-PASS。Round 4 P1-1 未闭合。

### P1-2 Synthetic PASS package 使用 A slice identity 掩盖 production poison mismatch

Formal runner 的实际路径是：

1. poison helper 改变 29 caches 中 15 个 unconsumed fields
   （runner lines 3833-3864）；
2. `materialize_slice()` 遍历并复制 `handle.files` 的所有字段，而非只复制
   consumed fields（runner lines 2492-2518）；
3. materialized file SHA 同时写入对应 build 的 WorkRow 和
   `slice_source_sha256`
   （runner lines 2745-2753、2774-2782）；
4. `slice_invariance.csv` 属于 RAW_11，A/P RAW_11 必须比较
   （runner lines 171-183、2979-2987；plan lines 1725-1732）。

本轮 synthetic probe 只改变一个合法 unconsumed field `obi`，其余 cache
内容相同。由 production `materialize_slice()` 生成的 SHA 为：

```text
canonical slice:
73d88cd6af9666c3d71a6ee7de1097b0286fc4831b8924eba0530007e2fe53b2

poison slice:
1eab512f170ab278ca15b52b9e57c4967cd9546e2465f4b3b7bc75ba72dd155c
```

即任一 qualifying slice 都会让 A/P 的 `slice_source_sha256` 不同，从而先在
A-1-1 产生 `raw_a_p_difference_count > 0`。这不是 detector 对 poison
敏感，而是把包含 unconsumed bytes 的工作文件 SHA投影进 scientific RAW_11。
在需要 nonempty slice 才能通过 A-1-2 support gates 的同时，A-1-1 与 A-1-2
实际上不可同时通过。

新增 synthetic fixture 没有复现此路径：

- 它故意生成不同的 A/B/P slice bytes
  （tests lines 1802-1828）；
- 但三路 `slice_invariance.csv` 都写入 A WorkRow SHA
  （tests lines 1990、2042-2052）；
- verifier 对 A、B、P 三个 root 均只查询 `build_label == "A"` 的 WorkRow
  （verifier lines 1217-1252）。

所以 `test_complete_synthetic_terminal_package_passes_v00_v12` 的确通过了
production check sequence，但它通过的是 frozen runner 无法生成的 cross-build
identity package。V00-V12 exact result encoding 已闭合，完整合法 production
baseline 与 build-specific slice evidence closure 仍未闭合。

## Round 4 P1 Disposition

1. **A-1-2 typed-evidence recomputation：未闭合。**
   Gate actual 已绑定到新函数，空 slice/work 配合手填 support counts 的旧缺口
   已关闭；但 action partition 与 slice mismatch 仍信任 producer 自报结论，
   不是从底层 typed counts/hashes 独立重算。

2. **Nonempty synthetic package + real V00-V12：部分闭合。**
   所有 mutation 已进入真实 `verify_terminal()`，并断言 exit 2、exact
   first-failure、13 rows 和后续 `NOT_EVALUATED`。但 PASS fixture 使用
   A-only WorkRow binding，未证明正式 runner 的 A/B/P package 可满足同一
   verifier contract。

## Passed Checks

1. Frozen idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
2. Frozen plan SHA256：
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
3. HEAD 与 annotated implementation tag 均 peel 到
   `cbfc7daced79fa9d453600f0988af5459c99764f`。
4. Runner SHA256/blob 保持：
   `5fdfe09a42c85f9d7a135d50bca3dda680f33b429a6869ac8e0586f9663967cf`
   / `165b076f53df73818de56abfabb55c4c5b472bef`。
5. Verifier SHA256/blob：
   `692a1a1778f17e4af2f73bb034bcd87f86085b17993bbefdf1ef3afa3cedcdf7`
   / `ba728168f11d523e2f4f9ad7682a07c20bca01c8`。
6. Tests SHA256/blob：
   `33dce2f5011b8722d724bbcc893a8164ae71fb0ee172d3bd85c2d80ac442fa53`
   / `82eb491712ce6a9c3e2d45466d1e37f31a2034c9`。
7. Task SHA256：
   `94ee1103069e67186efb4ba1ce4799a5266698f54d020b5815b10f92cba6bb51`；
   armed claim SHA256：
   `1f53f548be9946eced94e9e45bf02dec1b6fe7fa9d7d372d8644a3c88e01e315`。
   Claim 中 idea/plan/task/runner/verifier/tests SHA、formal argv、roots、
   implementation tag 和 controller identities 与当前冻结状态一致。
8. Git config 为 `core.fsync=all`、`core.fsyncMethod=fsync`、
   `core.logAllRefUpdates=always`；origin fetch/push URL 精确匹配。
9. Controller ref 查询 exit 0 且为空；claimed path、formal attempt root、
   terminal verifier result、consumption tag 和 terminal tag 均不存在。
10. Baseline authority verifier PASS：
    6 authority files、9 baseline callables、25-artifact snapshot 与 recovery
    tags 全部通过。Successor 的 8 个 direct callable AST bindings 也精确通过。
11. `verify_no_historical_attempt()` PASS；未发现已消费或 terminal attempt。
12. Focused suite：`123 passed in 29.17s`。
13. Inherited suite：`68 passed in 0.21s`。
14. Ruff、Ruff format check、py_compile、git diff check、runner/verifier
    `--help` 全部通过。
15. Review 开始前工作区 clean；除本报告外未产生文件修改。

## Required Remediation

1. 从 typed channel counts 重算 action partition，不接受
   `action_partition_exact` 作为独立事实。
2. 从 slice expected/actual counts、hashes、exact flags 和 cross-segment
   count 重建 exact flags、first mismatch reason 与 `slice_mismatch_count`；
   增加同步篡改 summary/gates/classification 的 production V07 negative tests。
3. 明确并实现 build-specific slice identity：每个 root 的
   `slice_source_sha256` 必须绑定同 label WorkRow/SLICE FeatureCall。
4. 消除 poison unconsumed bytes 对 RAW_11 slice identity 的机械影响，例如
   使用 consumed-only canonical slice identity，或将 raw work-file SHA 移出
   A/P scientific equality domain。修复必须继续保持 frozen outcome boundary
   语义，不能把预定 A/P mismatch 当作研究负面结果。
5. 重建 production-faithful nonempty A/B/P synthetic package；其 PASS baseline
   必须由 frozen runner 的真实 materialization/serialization规则可生成。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/2/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
