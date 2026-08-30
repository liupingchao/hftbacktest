# 0830T002 Hostile Plan Review Round 9

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `a19d3a5bc49250f2d01934c56e1ed489ca75ccf7`
- idea SHA256
  `86a49cbddaed1719ea281ca72158990ca4671c6e8a0cd1c14b74dad63298355a`
- plan SHA256
  `8ab0f367b6063e59b81bffc52c534c9e950dfc04e274bb4f9f3c77dde9aecbd4`
- task SHA256
  `d1462c86b1da77670b4595166a924c6145c6588850409c2501c8a371f962a00a`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 9 只修改 docs/task/workflow 记录，未修改 runner/tests。
- `git diff --check dff5792d..a19d3a5b` 通过。

## Severity Summary

- P0: 0
- P1: 1
- P2: 2
- P3: 0

## Round 8 Closure Matrix

| Round 8 finding | Round 9 status | 结论 |
|---|---|---|
| P1-1 single-frame IPC缺少sender/receiver独立持久化证据 | PARTIAL | `FeatureCall`现已持久化独立`IPCSender`/`IPCReceiver` hash、size、frame count、send-close、EOF和unused-byte字段，并冻结cross-end等式；但Frozen Hostile-Test Minimum仍未显式覆盖这些新增字段和EOF/second-frame路径 |
| P2-1 remote old/new transition machine evidence不完整 | PARTIAL | old/new authority已明确来自pre/post `ls-remote`，attempt-lock/verifier均新增observations/transitions/count；但observation未保存或要求命令exit status，失败命令的空stdout仍可伪装成absent ref |
| P2-2 IPC array table缺少typed arithmetic | PARTIAL | `IPCArrayRow`类型、dtype、shape、itemsize、length、offset、payload slice hash及feature projection等式已唯一冻结；但hostile minimum没有注册这些新增算术的mutation cases |

## Findings

### P1-1 Failed `ls-remote` can be admitted as an absent pre-consumption ref

位置：
- execution plan `:483-510`
- execution plan `:512-546`
- execution plan `:1002-1010`
- execution plan `:1184-1197`

问题：
- Formal one-shot authority要求第一次remote observation证明ledger ref
  不存在；pre-existing equal ref必须拒绝。
- `RemoteObservation`只保存：

```text
observation_id
command
stdout
observed_head
```

- 它没有保存或要求：
  - process exit code；
  - command execution success；
  - signal/timeout状态。
- `git ls-remote`在网络、DNS、SSH认证或remote错误时可以非零退出，同时
  stdout仍为空。当前schema可把该失败记录为：

```text
PRE_CONSUMPTION
stdout = ""
observed_head = null
```

- 一个具体fail-open路径是：
  1. remote ledger已存在且head恰为旧attempt的`consumption_head`；
  2. PRE_CONSUMPTION `ls-remote`失败，非零退出但stdout为空；
  3. runner将空stdout误记为remote absent；
  4. network恢复后，同SHA push返回zero exit或up-to-date；
  5. POST_CONSUMPTION观察到相同`consumption_head`；
  6. 当前observation、transition和`successful_push_count=1`字段均可被填成
     表面合规值。
- 这会绕过“first observation must be empty because the ref is absent”的
  one-shot语义，并破坏对local object-store deletion/accidental rerun的保护。
- Push exit code已明确要求zero，但三个作为old/new authority的
  `ls-remote` observations没有对称要求，因此不是纯schema美化问题。

必须修复：
- `RemoteObservation`增加exact execution status，至少：

```text
exit_code:int
terminated_by_signal:bool
timed_out:bool
```

- PRE/POST/terminal observation全部要求：

```text
exit_code = 0
terminated_by_signal = false
timed_out = false
```

- `observed_head=null`只允许在成功完成且stdout exact empty时成立。
- `successful_push_count`必须来自实际zero-exit push invocation ledger，
  不能由期望transition数量直接填充。
- Hostile case必须覆盖“nonzero `ls-remote` + empty stdout + remote already
  equal”，并证明其在push和cache read前terminal fail。

### P2-1 Exact JSON schema cannot represent its required empty strings

位置：
- execution plan `:516-520`
- execution plan `:948-960`
- execution plan `:1002-1005`
- execution plan `:1079-1081`
- execution plan `:1313-1331`

问题：
- Global schema notation定义：

```text
str = nonempty ASCII string
```

- 但新增的合法PRE_CONSUMPTION row要求：

```text
RemoteObservation.stdout = ""
```

  同时该字段类型被声明为`stdout:str`。
- 现有terminal verifier schema也有同类矛盾：
  `VerifierCheck.actual:str`，但所有first-failure后的rows必须
  `actual=""`。
- 因而不存在同时满足exact type schema与required sentinel bytes的合法JSON。
  不同实现只能隐式选择：
  - 违反`str`定义；
  - 改用null；
  - 引入未注册的empty-string exception。
- 三种选择都会破坏唯一实现和exact verifier result comparison。

必须修复：
- 定义允许空值的独立类型，例如：

```text
ascii = possibly-empty ASCII string
str = nonempty ASCII string
```

- 将`RemoteObservation.stdout`和`VerifierCheck.actual`明确改为`ascii`；
  或使用注册的nullable/literal union并冻结唯一sentinel。
- Hostile tests覆盖空字符串、null、空白字符串、non-ASCII和错误sentinel。

### P2-2 Frozen Hostile-Test Minimum does not cover the Revision 9 evidence surfaces

位置：
- execution plan `:985-1010`
- execution plan `:1460-1522`
- execution plan `:1785-1851`

问题：
- Revision 9新增了load-bearing machine surfaces：
  - `IPCSender`和`IPCReceiver`；
  - `IPCArrayRow` exact arithmetic；
  - `RemoteObservation`和`RemoteTransition`；
  - attempt-lock/verifier `successful_push_count`。
- Frozen Hostile-Test Minimum没有显式注册以下mutation：
  - sender/receiver header、payload、frame SHA或size不一致；
  - sent/received frame count不是1；
  - `send_end_closed=false`、`eof_observed=false`；
  - second frame、missing EOF、unused/trailing byte；
  - dtype spelling、shape/itemsize/length、offset、payload-slice hash或
    canonical feature projection不一致；
  - PRE/POST observation command/stdout/head/order mutation；
  - derived transition old/new/`derived_from` mutation；
  - missing/extra observation、transition或push count；
  - failed `ls-remote` producing empty stdout。
- 现有generic `LOADER IPC extra field/raw-byte smuggling`不能证明双端
  transcript、EOF或typed arithmetic的全部分支被实现。
- Remote hostile text仍只点名“GitHub-admin ledger tampering”，没有同步
  Revision 8/9 threat model中“任何remote writer偏离protocol”的actor
  vocabulary，也没有覆盖wrong URL/refspec、pre-existing equal、third push
  attempt或observation-ledger mutation。
- 由于runner/tests要在plan freeze后一次性实现，未写入minimum的case可被
  合法省略，无法依靠后续QA补写而不违反post-Build-A no-repair规则。

必须修复：
- 将上述IPC、typed arithmetic和remote ledger mutation逐项加入Frozen
  Hostile-Test Minimum。
- 每个mutation冻结预期terminal failure phase；remote precondition缺陷必须
  在cache read前失败。
- 把hostile actor文字与注册threat model统一为“any remote writer outside
  exact two-push protocol”。

## Closed Contract Areas

本轮确认已闭合或未发现新回归：
- Revision 9 idea/plan/task SHA identity及pre-execution scope；
- sender与receiver transcript字段已分别进入每个`FeatureCall`；
- sender/receiver header、payload、frame hash及size exact equality；
- sent/received frame count、send-end close、EOF和unused-byte语义；
- typed `IPCArrayRow`的`dtype.str`、shape、itemsize、length和offset算术；
- zero-dimensional和zero-length array语义；
- payload-slice hash及canonical feature-row projection；
- remote old/new不再依赖`git push --porcelain`文本；
- PRE/POST observations进入attempt-lock并由terminal verifier复制；
- POST_TERMINAL由verifier在线观察；
- consumption/terminal transitions的old/new和`derived_from`已注册；
- external history要求terminal为consumption的唯一子commit；
- scientific detector、source/poison boundary、slice identity、17-output
  closure、gate precedence、primary/sensitivity non-rescue和post-Build-A
  no-repair规则未发现Revision 9回归。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/2/0
```

Revision 9 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述findings全部闭合并经下一轮独立review达到`0/0/0/0`，才可进入
implementation/data execution。
