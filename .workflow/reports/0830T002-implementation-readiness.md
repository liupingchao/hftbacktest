# 0830T002 Independent Implementation Readiness Review

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `b70214e6cb5f3a3f4fba5f3d1b0e69f7b181492d`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- tag peel:
  `b70214e6cb5f3a3f4fba5f3d1b0e69f7b181492d`

更新时间：
- 2026-08-30 19:52 CST

审查限制：
- 未打开 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 runner、verifier、tests、idea、plan、task 或 armed claim。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/5/0/0**
- 29-cache formal execution lock **不得释放**。

当前 Git/tag/claim 状态是干净且一致的，但实现没有完整落实 frozen
Revision 12 的 single-use authority、independent verifier 和 hostile-test
合同。以下缺口可使不符合 frozen contract 的 formal execution 或 evidence
package 被接受，因此不能把当前实现视为 ready。

## Findings

### P1-1 Historical-attempt scan 与 consumption tree transition 未达到 single-use 合同

Frozen plan 要求扫描 refs、reflogs 和
`git fsck --full --unreachable --no-reflogs`，并拒绝 prior exact
claim-transition blob/tree、terminal receipt path/blob；consumption commit
还必须是 armed 到 claimed 的 same-blob rename，且为唯一 tree delta
（plan lines 648-668）。

实现的 `verify_no_historical_attempt()` 只从 fsck 输出提取
`unreachable commit`，随后只检查这些 commits 的 message/path；它没有检查
dangling/unreachable blob 或 tree，也没有比较 exact claim/terminal-receipt
blob identity
（runner lines 3190-3221）。

claim consumption 随后直接 `git add`/`git commit`，只验证 parent 和
annotated tag，没有复算 consumption commit 的 exact diff/tree
（runner lines 3396-3423）。Terminal verifier 同样只验证 parent/tag/distance，
未验证 consumption commit 的唯一 rename delta
（verifier lines 863-920、931-1031）。

因此 standalone dangling claim/receipt blobs、tree-level prior attempt
evidence，或含额外 staged delta 的 consumption commit 不会按合同 fail
closed。

### P1-2 Independent verifier 未绑定 exact tag、implementation、claim 与 lock identities

Plan 冻结了唯一 verifier CLI，并要求 claimed bytes、formal argv/roots、
idea/plan/task/runner/verifier/tests SHA、attempt-lock 全字段和 Git transition
可独立复算（plan lines 615-631、1235-1259、1937-1941）。

`V00` 接受调用者提供的任意 tag names，只要求这些 names 能解析；没有要求
三个 tag 参数等于 frozen constants（verifier lines 740-785）。

`V01` 只把 verifier 文件绑定到 implementation commit，并对 runner 做有限
AST blacklist；未验证 runner/task/tests/idea/plan 的 implementation blob/SHA，
也未把 verifier SHA/blob 与 claimed/task 中冻结值比较
（verifier lines 788-860）。

`V03` 虽检查 claim/lock 顶层 key set，但未验证 claim 的
`schema_version`、`formal_argv`、`source_cache_root`、六个 SHA identity；
也未验证 lock 的 `pid`/timestamp、`cwd`/`argv`、repo/source/attempt roots、
controller fields、`controller_consumption_head` 和 exact
`remote_transitions`（verifier lines 931-1031）。

这使 tag alias、claim identity mutation、lock root/argv/transition mutation
存在 verifier false-PASS 空间。

### P1-3 V07/V05/V06 未实现 frozen 17-output 与 instrumentation 的 exact typed closure

Plan 要求 all 17 schemas、typed sentinels、sorting、WorkRow contiguous
ordinals、FeatureCall authority、RawOpenEvent phase/path/caller matrix 和
poison schema 全部 exact
（plan lines 1485-1625、1883-1909）。

`validate_final_root()` 对 JSON 只检查顶层 keys 和 `schema_version=1`；对
CSV 只检查 header、列宽及文本中不存在 `nan/inf`。它没有验证 row types、
enums、sorting、sentinels，也没有验证 classification、summary、gate contract
和 outcome ledger 的交叉一致性
（verifier lines 636-656）。

`V05` 未要求 WorkRow `slice_ordinal` 在每个 build/cache 内从 0 连续，且
`row["path"].startswith("work/")` 后直接拼接路径，没有拒绝 `..` traversal
或 child symlink。FeatureCall 只按总数验证，未证明 29 个 canonical cache
各出现一次。RawOpenEvent 未绑定 `build_label` 到 FeatureCall，未验证
`caller_path`，也未验证 SLICE_MATERIALIZER 的 READ_INPUT/WRITE_SLICE
路径分别等于 full authority input 与 registered WorkRow
（verifier lines 1118-1457）。

`V06` 未校验 poison output root、source inventory SHA、exact
`unconsumed_fields` domain/order、cache-name identity和每个 cache 的 field
domain/order（verifier lines 1462-1519）。

因此格式正确但语义错误、路径越界或 source/poison/instrumentation identity
被替换的 evidence package 仍可能被 terminal verifier 接受。

### P1-4 DETECTOR inherited-FD/raw-capability contract 被硬编码为零而非验证

Plan 要求 fresh subprocess `close_fds=True`，DETECTOR 不接收任何
raw/work/source/poison path，除 stdio/control pipe 外的 inherited
descriptors 全部关闭，并要求 inherited-FD、environment/path reconstruction
hostile tests
（plan lines 145-181、1912-1918）。

实现通过 `multiprocessing` spawn 创建 DETECTOR/LOADER，但没有显式 FD
enumeration/closure 或 child-side allowed-FD assertion；DETECTOR 只安装
audit hook、`chdir("/")` 后读取 IPC
（runner lines 2238-2285、2304-2347）。

最终 instrumentation payload 直接写
`inherited_fd_violation_count=0`，同一组 boundary violation counters 也由
固定常量产生，而不是来自 child-side measurement
（runner lines 3015-3031）。

因此 evidence 中的零不能证明 frozen process-capability contract，尤其不能
证明 injected inherited FD 或 environment raw-path capability 已被拒绝。

### P1-5 Frozen hostile-test minimum 未持久化，65 focused passes 不能覆盖 execution contract

Plan 注册了 pre-cache claim/tag/root mutations、hard-link/commit/fsync order、
reflog/unreachable recovery、full V00-V12 verifier mutations、all 17 typed
schemas、inherited FD/env/path reconstruction、RawOpenEvent caller/path
mutations、terminal closure 和 push-wrapper bypass 等 hostile minimum
（plan lines 1875-1926）。

当前 focused file 的 45 个 test functions/65 collected cases 主要覆盖
scientific detector、synthetic build/seal、IPC codec 和少量 helper。末段
verifier coverage 只检查 13 check IDs、comparison mutation、三个 IPC
endpoint mutations、manifest self-exclusion、parse args 和 CLI help
（tests lines 1019-1128）。

没有测试直接调用 `consume_claim()`、`verify_no_historical_attempt()`、
`verify_terminal()` 或 `check_v00` 至 `check_v12`；也没有 claim/lock mutation、
consumption tree delta、inherited FD/env、WorkRow traversal、full
RawOpenEvent authority matrix 或 exact 17-schema hostile cases。

当前测试结果是真实的，但不能作为 frozen hostile minimum 已闭合的证据。

## Passed Checks

1. Frozen idea SHA256 精确为
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
2. Frozen plan SHA256 精确为
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
3. HEAD 与 annotated implementation tag 均 peel 到
   `b70214e6cb5f3a3f4fba5f3d1b0e69f7b181492d`。
4. Task、runner、verifier、tests 的 working SHA256、Git blob 和 tag tree
   blob 全部一致；armed claim 中对应 SHA 全部匹配。
5. Armed claim 的 formal argv、repo/source/attempt roots 精确匹配注册值；
   claimed path、attempt root、consumption tag 和 terminal tag 均不存在。
6. Git local config 精确为 `core.fsync=all`、
   `core.fsyncMethod=fsync`、`core.logAllRefUpdates=always`。
7. origin fetch/push URL 精确匹配；controller ref 查询 exit 0、stdout/stderr
   为空，确认当前 remote ref absent。
8. Baseline authority verifier 返回 PASS，6 frozen files、9 callable AST、
   25-artifact snapshot 和 recovery tags 均通过。
9. Focused synthetic suite：`65 passed in 8.18s`。
10. Inherited baseline/current suite：`53 passed, 1 skipped in 0.52s`。
11. Ruff、Ruff format check、py_compile、runner/verifier `--help` 均通过。
12. 当前实现的只读 `verify_no_historical_attempt()` 在现有仓库状态返回
    PASS；这只证明其当前实现未发现历史 attempt，不关闭 P1-1 的漏检面。

## Required Remediation Before Re-Review

1. 完整实现 exact historical blob/tree scan 和 consumption commit
   same-blob rename/only-delta verification，runner 与 verifier 双侧都需
   fail closed。
2. 让 V00-V03 精确绑定 frozen tag names、implementation tree identities、
   claimed bytes、formal argv/roots/SHA 和 attempt-lock 全字段/transition。
3. 为 17 outputs、WorkRow、FeatureCall、RawOpenEvent、poison attestation、
   gate/classification 增加 exact typed/schema/path/order/cross-file checks。
4. 将 inherited FD/raw capability counters 改为真实 child-side enforcement
   与 evidence，不得硬编码成功值。
5. 按 frozen hostile minimum 增加直接调用 production verifier/runner
   control helpers 的 mutation tests，并重新冻结 implementation identities、
   task、armed claim 和 implementation tag。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/5/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
