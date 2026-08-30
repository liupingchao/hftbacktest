# 0830T002 Hostile Plan Review Round 11

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `82a40da6c2345dfbc9b0aefa79d60d8f7b40f41b`
- idea SHA256
  `1f4fb38dadf462918e07d770fc218b13bbcafa8a49e5bd0a23b6fae248308546`
- plan SHA256
  `a1b97b69e65fb79026ec105889fd390b1f5fd4aeb3432261cb3a3c72a58ddd1a`
- task SHA256
  `d7f86ba1c2384e2fe4b9913526662db369e413ed0bb73781b8743b55f39c5ef0`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 11 只修改 docs/task/workflow 记录，未修改 runner/tests。
- `git diff --check 288f1ddd..82a40da6` 通过。

## Severity Summary

- P0: 0
- P1: 0
- P2: 1
- P3: 0

## Round 10 Closure

| Round 10 finding | Round 11 status | 结论 |
|---|---|---|
| P2-1 exact push-attempt count未被machine-observed | CLOSED | 两条固定路径no-replace `PushCall` receipts、ordinal 0/1、CONSUMPTION/TERMINAL phase、exact argv/refspec、zero exit、retry=false、single `push_once` wrapper、attempt-lock/verifier byte-identical copies及third/failed/up-to-date/wrapper-bypass hostile cases均已冻结 |

## Finding

### P2-1 Terminal PushCall is created after the artifact that claims to close all siblings

位置：
- execution plan `:577-584`
- execution plan `:701-725`
- execution plan `:752-760`
- execution plan `:1280-1290`
- execution plan `:1313-1340`
- execution plan `:1618-1620`

问题：
- Frozen sequence明确规定：
  1. `attempt-result.json`在step 13发布；
  2. tracked terminal receipt在step 14提交并打tag；
  3. terminal commit在step 15推送；
  4. `push-ledger/001-terminal.json`只能在terminal push和post observation
     完成后发布。
- 因而`001-terminal.json`在`attempt-result.json`和tracked terminal receipt
  发布时尚不存在。
- 但plan同时声称：

```text
attempt-result.json is the external terminal closure over FINAL_17
plus all sibling artifacts
```

  并在后文声明tracked terminal receipt关闭`attempt-result`。
- `attempt-result.json` schema不包含：
  - terminal PushCall；
  - terminal PushCall SHA；
  - push-ledger tree SHA。
- tracked terminal receipt schema也没有上述字段。
- 因此当前没有合法执行能够同时满足：

```text
attempt-result published before terminal push
001-terminal receipt published after terminal push
attempt-result closes all sibling artifacts including 001-terminal
```

- Revision 11的terminal verifier确实可以在post-terminal阶段读取两条receipt，
  并把两个byte-identical `PushCall` rows写入verifier result。这足以成为
  `001-terminal.json`的独立post-terminal closure，但plan没有把该ownership
  明确为唯一authority；相反，`attempt-result`的“all sibling artifacts”
  claim仍与实际时序冲突。
- 不能通过把terminal PushCall SHA加入预先提交的tracked terminal receipt来
  修复，因为那会重新制造terminal-push/post-observation的时间循环。

必须修复：
- 唯一冻结closure graph，建议明确分层：

```text
attempt-result:
  closes FINAL_17 and exact pre-terminal siblings only
  closes consumption PushCall through byte-identical attempt-lock.push_calls
  explicitly excludes 001-terminal.json

tracked terminal receipt:
  closes attempt-result and its registered pre-terminal evidence
  explicitly excludes post-terminal PushCall evidence

terminal verifier result:
  sole post-terminal closure for:
    push-ledger/000-consumption.json
    push-ledger/001-terminal.json
    both exact PushCall copies
    POST_TERMINAL observation
    completed child set
```

- 修改`:752-760`和`:1618-1620`的“all sibling artifacts”文字，使其与exact
  schemas和step 13-15时序一致。
- V02/V04/V12必须明确拥有两条receipt的path、schema、byte identity、
  no-replace、missing/extra/mutation和post-verification drift检查。
- 不得把post-terminal receipt hash反向加入terminal commit或其tracked
  receipt。

## Closed Contract Areas

本轮确认已闭合或未发现新回归：
- Revision 11 idea/plan/task SHA identity及pre-execution scope；
- exact `PushCall` typed schema；
- fixed receipt paths `000-consumption.json`和`001-terminal.json`；
- ordinals 0/1与phases CONSUMPTION/TERMINAL；
- exact argv/refspec、expected old/new及pre/post observation IDs；
- `retry_allowed=false`和两次zero exit；
- common no-replace/fsync receipt publication；
- attempt-lock精确包含byte-identical consumption row；
- terminal verifier精确包含两条byte-identical rows及ordinal order；
- exact completed child set拒绝第三个receipt与duplicate ordinal；
- AST只允许`push_once`作为remote push入口；
- failed push retry、up-to-date repush、third push、unledgered direct push及
  wrapper bypass hostile cases；
- scientific detector、source/poison boundary、slice identity、17-output
  closure、gate/classification precedence、primary/sensitivity non-rescue和
  post-Build-A no-repair规则未发现Revision 11回归。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/0/1/0
```

Revision 11 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述closure ownership冲突闭合并经下一轮独立review达到
`0/0/0/0`，才可进入implementation/data execution。
