# 0830T003 Independent Replacement Plan Review

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0830T003

状态：
- PASS

更新时间：
- 2026-08-30 23:33 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t003-leader-trigger-formal-replacement`
- branch:
  `codex/fixed-epoch-leader-trigger-formal-replacement`
- reviewed commit:
  `4c92b7cbfc0e3f05309f61e812a5141ca7ce0bbd`
- replacement plan:
  `docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_formal_replacement_execution_plan_20260830.md`
- replacement plan SHA256:
  `a8b26fafe2156b2948c84875f206219184aa8fe17098d28fc322eba05df233a5`
- replacement plan Git blob:
  `fcd676ddc11f05894230737f539c76575e97788d`
- task SHA256:
  `7941a20125408ca04b43a8fd6e67293d4d4f0e545b031695693ac744c3bdc711`
- task Git blob:
  `b50734690e41ced4dd68abdc5fd5d4c392fa06f5`
- inherited Revision 18 plan SHA256:
  `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`

审查限制：
- 未打开或读取 source-cache-root 下的正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt、A0 或 live/private/order 操作。
- 未修改 runner、verifier 或 tests。
- 本轮结论只覆盖 committed replacement plan/task，不构成 implementation
  readiness 或 formal execution 授权。

## Decision

- **PASS**
- **P0/P1/P2/P3 = 0/0/0/0**
- Findings：无。
- Replacement plan 可按当前 SHA 冻结。
- 29-cache formal execution lock 继续保持关闭，直到 implementation commit、
  committed task/claim identities、annotated implementation tag、focused /
  inherited tests 和独立 readiness review 全部满足 plan Section 17。

## 1. Revision 18 Scientific Contract

结论：**保持不变**。

对旧 Revision 18 plan 与 replacement plan 做 exact textual diff，科学部分没有
发生修改。全部差异仅为：

1. 标题、Task ID 和 replacement revision header；
2. 新增 Replacement Boundary（plan lines 15-48）；
3. runner/verifier/tests、task/claim/receipt、repo/attempt root、tag 和
   controller ref 的 0830T003 identity substitutions；
4. tracked-files identity list 的相同替换。

以下科学对象与 Revision 18 byte-comparable in substance：

- Hypothesis ID 与 Audit ID（plan lines 7-11）；
- unique primary `TRADE_LED` 和 non-rescue sensitivities
  `DEPLETION_LED` / `OFI_LED`（lines 50-73）；
- source authority、consumed/unconsumed boundary 和 direct-call authorities；
- causal order、TTL/prestate/confirmation、0.50/0.25 thresholds
  （lines 275-307）；
- 60s epoch、`[15s,45s)` core、thinning/cluster identity
  （lines 309-334）；
- conservation、slice/reset invariance、RAW/SEALED/FINAL projections；
- A-1-0 through A-1-3 gate order/classification and primary prediction；
- all outcome-blind, A0 and live-trading locks。

The inherited idea SHA remains
`a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`.
0830T002 final QA established that no Build A/B/P scientific package, support
count or classification existed, so there is no observed scientific result
available for threshold, feature, gate or prediction tuning.

## 2. Replacement Isolation

结论：**隔离完整**。

The plan registers distinct:

- task: `0830T003`;
- runner/verifier/tests paths;
- armed/claimed paths:
  `.workflow/attempt-claims/0830T003.{armed,claimed}.json`;
- terminal receipt:
  `.workflow/attempt-receipts/0830T003.terminal.json`;
- attempt root:
  `...formal_replacement_0830T003_formal_v1`;
- implementation/consumption/terminal tags with the
  `replacement-*` namespace;
- controller ref:
  `refs/heads/codex/0830T003-controller-ledger`;
- verifier result:
  `.workflow/reports/0830T003-terminal-verifier.json`.

Plan authority is exact at lines 451-716. Independent state inspection found:

- the new remote controller ref is absent;
- no local T003 controller ref exists;
- no T003 replacement tag exists;
- armed, claimed, terminal receipt, verifier result and attempt root are absent;
- no reachable/reflog/unreachable T003 consumption or terminal history exists;
- the old 0830T002 claimed file, attempt lock, tags and remote controller ref
  remain present and unchanged.

The new task explicitly forbids modification or deletion of any 0830T002
claim, receipt, tag, remote ledger, report or attempt root
(`.workflow/tasks/0830T003.md` lines 92-96).

## 3. Baseline Authority Fix

结论：**足以修复已知 TypeError**。

The accepted baseline verifier defines:

```python
def verify_authority(
    repo_root,
    manifest,
    *,
    check_working_tree: bool,
)
```

`check_working_tree` is required and keyword-only
(`examples/hyperliquid/skhynix_fixed_epoch_suppression_baseline.py`
lines 110-115). The failed 0830T002 runner omitted that argument. The
replacement plan freezes the exact call with literal
`check_working_tree=True` at lines 35-43.

Read-only inspection of the untracked implementation draft corroborated that
its scientific runner delta consists only of replacement identities plus this
call correction. Its synthetic test uses AST inspection to require exactly
one baseline `verify_authority` call and the literal boolean `True`.

This review claims only that the registered change removes the known Python
argument-binding failure. It does not predict that no unrelated latent
execution failure can occur; implementation readiness must still validate the
committed files and tests before claim arming.

## 4. Data Dependence And Rerun Semantics

结论：**无 data-dependent tuning、future-outcome access 或 same-task rerun
语义**。

- The replacement inherits all thresholds, variants, gates and prediction
  unchanged (plan lines 20-33).
- Future price, target, fill, fee, PnL, A0 and live/private/order access remain
  unauthorized.
- 0830T002 remains terminal `未通过`; its consumed claim and interruption
  evidence are immutable.
- Accepted 0830T002 final QA explicitly scoped repair/rerun prohibition to
  that task and required any later formal scientific execution to use a new,
  independently authorized formal task.
- T003 is therefore a new precommitted task, not a restoration of the consumed
  0830T002 claim and not a continuation from an observed scientific result.
- Once the T003 claim is consumed, plan lines 2146-2155 prohibit repair,
  diagnosis, replacement attempt, alternative execution, parameter change or
  plan/code/test change, including after a contradictory result or another
  interruption.

## 5. Formal Command And One-Shot Closure

结论：**自洽**。

- The sole formal command fixes runner, repo root, source root and attempt root
  exactly (plan lines 451-464).
- The sole terminal-verifier command fixes all paths/tags/result output exactly
  (lines 639-675).
- Before any `.npz` open, the contract requires exact argv/cwd, clean tagged
  HEAD, no historical attempt, armed-to-claimed same-blob rename, consumption
  commit/tag, empty-to-consumption remote transition, receipt and fsynced lock
  (lines 683-711).
- A missing terminal package after consumption is
  `INTERRUPTED_TERMINAL` and cannot be recovered or replaced in T003
  (lines 713-716).
- The controller ledger permits exactly one consumption push and one terminal
  fast-forward, with no retries or third push (lines 509-619).
- Section 17 keeps execution locked until plan SHA, implementation identities,
  claim, tag, tests and independent readiness are frozen
  (lines 2127-2144).

## Worktree Note

Three replacement implementation files were already present as untracked
drafts at review time. They were not modified, staged or accepted by this
review. They must enter a separate implementation commit and independent
readiness review. Their presence does not release the formal data lock.

## Final

- Result: **PASS**
- Severity: **P0/P1/P2/P3 = 0/0/0/0**
- Plan freeze: **AUTHORIZED**
- Implementation readiness: **NOT YET AUTHORIZED**
- Formal execution: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
