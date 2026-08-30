# 0830T002 Hostile Plan Review Round 10

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `288f1dddfdab5ea078f494d44cbe0c0f9a8b90aa`
- idea SHA256
  `1ecc1e68e1e4b2f047e49274056d955d39d0d61b3df64432713aadcfa0246990`
- plan SHA256
  `bd96f13091e04bfdc1b407b6bda4f1f818ca2425d730f6b8e2b4fb81c430e07d`
- task SHA256
  `0440becbe12d4b31a90f47532e07cbfd335259a4892def63eaa7414a47464bad`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 10 只修改 docs/task/workflow 记录，未修改 runner/tests。
- `git diff --check a19d3a5b..288f1ddd` 通过。

## Severity Summary

- P0: 0
- P1: 0
- P2: 1
- P3: 0

## Round 9 Closure Matrix

| Round 9 finding | Round 10 status | 结论 |
|---|---|---|
| P1-1 failed `ls-remote`可被编码为absent ref | CLOSED | 三次observation均注册`exit_code=0`与empty stderr；任何nonzero exit均terminal fail，且empty stdout仅在zero exit时表示absence |
| P2-1 nonempty `str`无法表示合法empty sentinel | CLOSED | 新增empty-allowed ASCII `text`类型；`RemoteObservation.stdout/stderr`和`VerifierCheck.actual`均改用该类型 |
| P2-2 Revision 9新增IPC/typed/remote surfaces没有hostile closure | PARTIAL | IPC endpoint、frame/EOF、typed arithmetic及多数remote mutation已逐项加入minimum；push attempt ledger及missing/extra remote rows/count仍未覆盖 |

## Finding

### P2-1 Exact push-attempt count is asserted but not machine-observed

位置：
- execution plan `:488-513`
- execution plan `:517-550`
- execution plan `:1008-1016`
- execution plan `:1191-1200`
- execution plan `:1285-1297`
- execution plan `:1839-1866`

问题：
- Plan明确规定：

```text
Any additional ... push attempt ... is terminal failure.
```

- 但machine evidence只保存：

```text
attempt-lock successful_push_count = 1
verifier successful_push_count = 2
```

- 当前没有`PushObservation`或等价per-call ledger保存：
  - push call index；
  - exact argv/refspec；
  - exit code；
  - invocation phase；
  - stdout/stderr或其hash；
  - whether the call updated the ref or returned up-to-date；
  - total attempted push count。
- `successful_push_count`可以由orchestrator直接写入期望常量。Terminal
  verifier能够从pre/post heads和commit ancestry证明最终两次逻辑transition，
  但不能区分：
  - exactly two push invocations；
  - an extra failed push；
  - an extra up-to-date push；
  - retry后最终得到同一remote head。
- 因而“最终remote state正确”已经闭合，但“formal runner严格只执行exact
  two-push command sequence”仍不是machine-observed claim。这与计划声明的
  formal-runner deviation protection及additional push-attempt prohibition不符。
- Revision 10 hostile minimum覆盖wrong URL/refspec和pre-existing equal，
  但没有覆盖：
  - third push attempt；
  - failed push followed by successful retry；
  - extra up-to-date push；
  - missing/extra `RemoteObservation`；
  - missing/extra `RemoteTransition`；
  - mutated `successful_push_count`或attempt count。
- Hostile text `:1839-1841`仍只写“GitHub-admin ledger tampering”，没有与
  normative threat model的“any remote writer outside exact protocol”使用
  同一actor vocabulary。

必须修复：
- 增加唯一typed `PushObservation`，至少包含：

```text
push_index
phase
command
exit_code
stdout
stderr
```

- Attempt-lock持久化exact one-row consumption push ledger；terminal verifier
  持久化其exact copy及exact one-row terminal push ledger。
- 冻结：

```text
push_attempt_count = successful_push_count
attempt-lock counts = 1
terminal verifier counts = 2
all exit_code = 0
exact phase/order = CONSUMPTION, TERMINAL
exact argv/refspec = registered commands
```

- 如果不希望保存diagnostic push output，应明确保存其SHA或明确空投影，但
  call existence、argv、phase与exit code必须进入machine evidence。
- Hostile minimum增加third/failed/up-to-date push、missing/extra/reordered
  push observation、missing/extra remote observation/transition及count mutation。
- 将hostile actor文字改为与normative threat model完全一致。

## Closed Contract Areas

本轮确认已闭合或未发现新回归：
- Revision 10 idea/plan/task SHA identity及pre-execution scope；
- `ls-remote` exit-code authority及empty stderr requirement；
- nonzero `ls-remote`即使stdout为空也必须terminal fail；
- successful empty observation与remote absence的唯一对应；
- empty-allowed ASCII `text` schema；
- verifier `NOT_EVALUATED actual=""`的typed schema；
- sender/receiver header、payload、frame hash及size mutations；
- sent/received frame count、send-end close、second frame、EOF和unused-byte
  hostile cases；
- typed IPC dtype、shape、itemsize、length、offset、zero-size和payload hash
  hostile cases；
- remote nonzero exit、stderr、pre-existing equal、wrong URL/refspec、
  full-SHA和derived-transition hostile cases；
- scientific detector、source/poison boundary、slice identity、17-output
  closure、gate/classification precedence、primary/sensitivity non-rescue和
  post-Build-A no-repair规则未发现Revision 10回归。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/0/1/0
```

Revision 10 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述finding闭合并经下一轮独立review达到`0/0/0/0`，才可进入
implementation/data execution。
