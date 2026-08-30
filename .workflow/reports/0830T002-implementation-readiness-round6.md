# 0830T002 Independent Implementation Readiness Review Round 6

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `591e3cb57149d3e46c4223a1249508f57629e393`
- Round 5 remediation implementation commit:
  `b9fcb5554cec752967a3563bb5747abb6c202276`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- annotated tag object:
  `a36d4edaa97e74c5cdbd19ae1dd060d5da547275`
- tag peel:
  `591e3cb57149d3e46c4223a1249508f57629e393`

更新时间：
- 2026-08-30 21:50 CST

审查限制：
- 未打开或读取 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier 或 tests。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/1/0/0**
- 29-cache formal execution lock **不得释放**。

Round 5 的两个实现缺陷已经在代码和 synthetic evidence 层真实闭合：

1. verifier 不再信任 producer 自报的 `action_partition_exact`、五组
   slice exact flags 或 `mismatch_reason`；
2. A/B/P root 分别绑定同 label WorkRow，合法 poison slice 可以拥有不同
   物理 SHA，同时 verifier 强制三路 consumer feature output 相同。

但是 remediation 引入了一个未写入 frozen Revision 12 plan 的 A/P
comparison 例外。Frozen plan 要求 exact `RAW_11`、`SEALED_15`、
`FINAL_17` comparison，并要求 A/P missing/extra/byte mutation 形成 A-1-1
negative evidence；当前 runner/verifier 却把
`support/slice_invariance.csv.slice_source_sha256` 以及 manifest 中对应
artifact SHA 替换成归一化值后再计算 `ComparisonRow` SHA。因此正式执行会
使用一种 frozen plan 未定义的 semantic projection，不能视为可执行同一份
冻结合同。

## Findings

### P1-1 Poison slice normalization 改变了 frozen exact-comparison contract

Frozen plan 明确规定：

- build sequence 第 6、8、11 步分别比较 exact `RAW_11`、exact
  `SEALED_15`、exact `FINAL_17`
  （plan lines 699-716）；
- `slice_source_sha256` 是 `slice_invariance.csv` 的正式字段
  （plan lines 967-984）；
- A/P differences 属于 A-1-1，并作为 negative outcome-boundary evidence
  保留（plan lines 780-782）；
- frozen hostile minimum 要求 A/P missing、extra、byte mutation 产生
  A-1-1 negative evidence（plan lines 1879-1882）。

当前实现：

- runner 对 A/P `slice_invariance.csv` 解析 CSV，并把每行
  `slice_source_sha256` 替换为 64 个零后计算 comparison SHA
  （runner lines 1845-1864）；
- runner 对 `run_manifest.json` 再把 slice artifact 的真实 SHA 替换成上述
  normalized SHA（runner lines 1865-1888）；
- RAW/SEALED/FINAL 三个 A/P domain 全部启用此 normalization
  （runner lines 3076-3083、3158-3165、3187-3194）；
- verifier 镜像相同的 normalization，而不是独立验证 exact file-byte SHA
  （verifier lines 1468-1561、3281-3325）。

本轮 focused PASS baseline 自身证明物理文件不相等但 comparison 为零：

```text
sha256(A/support/slice_invariance.csv)
  != sha256(P/support/slice_invariance.csv)

sha256(A/run_manifest.json)
  != sha256(P/run_manifest.json)

final_a_p.difference_count = 0
```

对应 tests lines 2574-2581。另一个 test 明确把
`poison_normalize_slice_source=True` 作为成功条件
（tests lines 2697-2733）。

这不是普通 implementation detail：`raw_a_p_difference_count` 是 A-1-1 gate
的正式 actual。当前 `ComparisonRow.a_sha256/other_sha256` 在这两个 path 上
不再是文件字节 SHA，而 frozen schema/contract 没有注册 normalized hash
preimage、允许忽略的字段或 comparison mode。只要差异局限在该字段，正式
A-1-1 可以 PASS，即使 exact RAW_11 文件字节已不同。

可复现方式：

1. 创建 header/schema 合法且除 `slice_source_sha256` 外完全相同的 A/P
   `slice_invariance.csv`；
2. 确认两个文件 `sha256_file()` 不同；
3. 调用 production
   `comparison("RAW_11:A_vs_P", ..., poison_normalize_slice_source=True)`；
4. observed `difference_count == 0`。

Required remediation：

1. 由总控注册新的 normative plan revision/addendum，精确定义
   poison-normalized comparison domain、hash preimage、允许归一化的唯一
   path/field、manifest 派生语义及 hostile mutations；或
2. 恢复 frozen exact-byte comparison，并用不把 unconsumed work-file bytes
   投影进 scientific RAW_11 的数据模型解决机械 mismatch。

在 normative contract 与 implementation 一致前，不得消费 armed claim。

## Round 5 P1 Disposition

1. **A-1-2 typed producer recomputation：已闭合。**
   - action partition 从 `total_action_count` 与六个 typed action counts
     重算，self-declared flag 不一致会 terminal reject
     （verifier lines 1335-1352）；
   - 五组 slice exact flags 从 expected/actual count+SHA 重算，并按
     epoch/counter/retained/status/support/cross-segment precedence 重建
     `mismatch_reason`（verifier lines 1281-1319）；
   - synchronized hostile tests 分别覆盖 action flag、slice exact flag 和
     exact 已同步但 reason 仍伪造的情况
     （tests lines 2625-2672）。

2. **Build-specific slice binding 与 consumer equality：实现层已闭合。**
   - verifier 按 `build_label` 选择 WorkRow，并绑定各 root
     `slice_source_sha256`（verifier lines 1219-1265、3235-3240）；
   - V05 要求每个 FULL/SLICE unit 具有 exact A/B/P triad，且
     `feature_output_sha256` 三路相同
     （verifier lines 3001-3025）；
   - synthetic package 使用 A/B canonical slice、P poison slice，并将每个
     root 绑定自己的 WorkRow SHA（tests lines 1802-1856、1993-2122）；
   - cross-build consumer mutation 在真实 `verify_terminal()` 中 exact
     first-fails V05（tests lines 2675-2694）。

3. **Production materializer 临时 probe：通过。**
   本轮只在系统临时目录生成 synthetic NPZ，不读取正式 cache。Production
   `materialize_slice()` observed：

```text
A slice SHA = c70cb7a635bc2ab9597b1d39e8079a2067dddfd7e2dfbddd92943068ae51805c
B slice SHA = c70cb7a635bc2ab9597b1d39e8079a2067dddfd7e2dfbddd92943068ae51805c
P slice SHA = b17f6ec54e9097b684e221ca119e5299d0c387d51472223e6e0d251a2f23aa3b
all consumed feature SHA =
b6c917ce55fd4d0709b35b05b0cd6faaa82580d1e97243cec76c0cf82880ab04
forbidden access count = 0
```

这证明 Round 5 所描述的物理现象和 consumer invariance 均成立，但不能替代
对 comparison semantic exception 的 normative 注册。

## Passed Checks

1. Review 开始与写报告前，branch/HEAD clean；implementation tag 是 annotated
   tag 并精确 peel 到 HEAD。
2. Frozen idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
3. Frozen plan SHA256：
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
4. Task SHA256：
   `c7cd7d2b6e7ef73acedd6d8bd058bbea99aaffcd21f49510fdf461d5a230d272`。
5. Runner SHA256/blob：
   `7115b3f0b0816d80aee2192f65cf53207c47e1583c2c23e9b1887a62df4956be`
   / `5608a502e7137b5daf02066e5de41e0e7313d881`。
6. Verifier SHA256/blob：
   `6fd1883637a4f357454d5835e35d2f3d3a844c7712d7dff4a922a543a419d8d2`
   / `f18575ae670b19cb9efac0c61676ffdce1fce6c4`。
7. Tests SHA256/blob：
   `2af680f36d65715b54ad4a5ac486b003d828cd74df6515910db1cea5e726ef97`
   / `646ceb018b9bf47d5f0e47cf4c0829ef69d06fcc`。
8. Armed claim SHA256/blob：
   `3e2e5775a3bf9013956c2e49ef2a3c20032344ffbbf36172fc83c5bdb64f4248`
   / `0d842444de9ae3bdec382e83ff55803c531df294`。
   Claim 的 idea/plan/task/runner/verifier/tests SHA、exact formal argv、
   repo/source/attempt roots、implementation tag 和 controller identities
   均独立复算一致；production `verify_armed_claim()` PASS。
9. Git config：
   `core.fsync=all`、`core.fsyncMethod=fsync`、
   `core.logAllRefUpdates=always`；origin fetch/push URL exact。
10. Controller ref `refs/heads/codex/0830T002-controller-ledger`
    `git ls-remote --heads` exit 0 且 stdout empty。
11. Claimed path、terminal receipt、formal attempt root、terminal verifier
    result、consumption tag 和 terminal tag 均不存在。
12. Production `verify_no_historical_attempt()` PASS；未发现 reachable、
    reflog 或 unreachable consumption/terminal attempt。
13. Baseline authority verifier PASS；successor direct callable AST binding
    PASS，count `8`。
14. Focused suite：
    `128 passed in 34.54s`。
15. Inherited suite：
    `66 passed, 1 skipped in 0.62s`。
16. Ruff、Ruff format check、py_compile、runner/verifier `--help` 和
    `git diff --check` 全部通过。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/1/0/0**
- Round 5 original P1 defects: **CLOSED**
- New normative comparison finding: **OPEN**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
