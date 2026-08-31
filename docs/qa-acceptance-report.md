# QA 验收结果

执行线程：
- 独立计划审查线程

任务ID：
- 0831T001

状态：
- 未通过

更新时间：
- 2026-08-31 13:09 CST

验收线程：
- 0831T001 Q0 Revision 14 独立 plan reviewer

验收对象：
- reviewed commit:
  `7fae56ff22e98c1f0922776ece5afdc285fa8ceb`
- commit message:
  `workflow: harden 0831T001 Q0 contract revision 14`
- frozen execution plan、task 与 surface contract
- Round 13 两项 P1 的 Revision 14 closure

验收范围：
- raw empty/nonempty `metadata NUL path NUL` parser。
- HEAD -> complete index -> complete physical inventory。
- `core.filemode=false` mode mutation 与 autocrlf exact-byte mutation。
- 8 transition states、22 phase/branch variants、PASS60/FAIL3 expansion。
- 704-row aggregate
  `8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c`。
- strict JSON、hash references 与新 P0-P3。

验收限制：
- historical-cache access: `NONE`
- future-outcome access: `NONE`
- implementation access/review: `NONE`
- formal attempt root access: `NONE`
- claim/receipt/controller artifact access: `NONE`
- Git parser/mutation 复验只使用系统临时目录的合成仓库。
- 未修改冻结 task、execution plan 或 surface contract。

## Severity

- **P0/P1/P2/P3 = 0/1/0/0**

## Passing Evidence

1. raw parser 接受 empty bytes 与合法
   `metadata NUL path NUL` pairs，并拒绝 lone NUL、missing final NUL、
   odd fields、duplicate path 与 rename status。
2. staged `D armed + A claimed` 被解析为两个独立 records。
3. complete expected index 从 HEAD tree 与 cached preimage 唯一派生。
4. complete physical inventory 对每个 expected path 绑定 no-follow
   kind/mode、SHA256 与 unfiltered exact blob。
5. `core.filemode=false` 隐藏的 `0644 -> 0755` mutation 被 physical mode
   mismatch 捕捉。
6. autocrlf 隐藏的 LF -> CRLF mutation 被 exact blob 与 SHA256 mismatch
   捕捉。
7. contract 恰有 8 个 legal transition preimages、22 个唯一
   phase/branch variants、15 个 action phases，且无 missing state ref。
8. 57 个 package paths 与 3 个 terminal-common paths 唯一且不重叠，
   因而 PASS=60、FAIL=3。
9. strict duplicate-key JSON、commit identity、plan/task/surface hashes、
   task/plan surface references、`git diff --check` 与 `git fsck` 通过。

## 不通过项

### P1-1 704-row canonical preimage encoding 未冻结

704 行计数正确，目标 aggregate 也可复现，但仅在隐含使用以下编码时：

```text
repository_config keys =
  diff_renames, core_filemode, core_autocrlf
variant_ordinal = zero-based integer
```

该编码得到：

```text
8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c
```

surface contract 明文维度名却是 dotted
`diff.renames/core.filemode/core.autocrlf`，没有冻结
`repository_config` 的 exact JSON schema，也没有冻结
`variant_ordinal` 的 zero/one-based 基数和 two-digit formatting。

合法替代解释产生不同 hashes：

```text
snake_case + one-based =
  3158320b1904b73f6b23c2a1831cbb5a7679aa8c684998401b39290765b6eed3
dotted + zero-based =
  6997e59ebeeeedca181eb89a0b4a3a031655a8ae13d699e07a436cbf93bd685c
```

因此该 aggregate 是未公开唯一 preimage 的 commitment，而不是可由冻结
surface 唯一机械派生的 authority。Round 13 的 all-state mutation
executable-total 要求仍未完全关闭。

## 验收结论

- **未通过**
- **P0/P1/P2/P3 = 0/1/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

结论说明：
- raw parser 与 physical mode/exact-byte 缺口已关闭。
- 8 states、22 variants、PASS60/FAIL3 与 704 行计数均通过。
- 704-row canonical row representation 未唯一冻结。
- 只有 `0/0/0/0` 才可 PASS。

通过项：
1. raw empty/nonempty parser。
2. complete index/physical inventory。
3. filemode/autocrlf hostile mutations fail closed。
4. 8/22/PASS60/FAIL3 expansion。
5. strict JSON 与 hash refs。

不通过项：
1. 704-row aggregate 的 exact row encoding 不可由 surface 唯一派生。

缺陷清单：
1. P1-1：`repository_config` schema 与 `variant_ordinal` encoding 未冻结。

阻塞项：
- Revision 14 不得解锁 implementation 或 formal execution。

建议总控下一步：
1. 冻结 `repository_config` exact object keys/value types。
2. 冻结 `variant_ordinal` 基数、类型和 PATH probe formatting。
3. 由明文唯一 derivation 重新发布 704-row aggregate，再发起独立 review。

详细报告：
- `.workflow/reports/0831T001-plan-review-round14.md`

提交信息：
- commit：由本轮审查提交承载，不在报告内自引用。
