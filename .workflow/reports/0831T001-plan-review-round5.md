# 0831T001 Independent Hostile Plan Review Round 5

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `c7d5dfb597c2724ca44d82d295d0a325eef74c03`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 5 plan SHA256 / Git blob:
  `bf88377251d2c03e68a6c959c2025cb0e325c911bbe72f0dbbd6642fc5066629`
  / `2f3ef9879e019ee8b1f47b6e33c0cd7ad3d78901`
- fixture-truth SHA256 / Git blob:
  `fd1ea67f97e9d282e8fd12ae9bae4198e6742201d4bb630ed54e8dbdc6b57d85`
  / `33e78f7fa060f368b9ebfb517c61cb615b0405bc`
- surface-contract SHA256 / Git blob:
  `3f23b5d5964224606d33e6d7e7cb071e698746b76fbb6f422459d824fc49007b`
  / `ed65e8181b8dbb281adffa2440b9edcb69c59f0c`
- task SHA256 / Git blob:
  `ab9c04d028f08a7bd9c2309f7503d0a0d6c76044ff137f89d7d6d7c41ca3f7f2`
  / `dfe867b404698e110213bb859c43742d314be17c`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache 或 outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live/private/order 操作。
- 仅运行 deterministic synthetic primitive、schema、package arithmetic 和
  temporary bare-repository CAS 检查。
- 未修改 master、plan、task、contracts、production source、runner、
  verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/3/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 5 关闭了 Round 4 的 reset primitive、QF12 first-error 和 receipt
header 问题，也补上了正常 verifier FAIL 分支；但 QF13 的冻结 H1 oracle
与 production memory path 冲突，且 one-shot crash recovery 和 process/push
receipt preimage 仍不能形成唯一、可执行的终态合同。只有 `0/0/0/0` 才可
PASS。

## Round 4 Closure Matrix

| Round 4 finding | Revision 5 status | 说明 |
|---|---|---|
| QF14/QF15 reset truth | `CLOSED` | accepted strict base/action/memory primitives exact produce `[3,5,5]`, `[-1,0,0]`, `[0,80,80]`, then `[9,9,9]`; duplicate-key scan为零 |
| QF12 first error | `CLOSED` | one published required slice plus another absent now correctly expects the earlier `SLICE_PUBLICATION` boundary |
| 23-field receipt header | `CLOSED` | plan CSV order、machine CSV schema 与 instrumentation field order byte-identical |
| controller init and failed-formal terminalization | `PARTIALLY_CLOSED` | pre-consumption init and verifier-observed FAIL branch exist, but registered crash windows still lack executable terminal receipts and recovery transitions; see P1-2 |

## Findings

### P1-1 QF13 H1 oracle and causal-access range contradict the frozen production memory path

Evidence:
- master protocol lines 805-809 defines
  `leader_background_run_length =
  log1p(min(2000ms, consecutive observable TRADE BACKGROUND ending at
  t-20ms) - 120ms)`.
- QF13 patches seed a valid TRADE `NEW_NEUTRAL` at index `4500`, refresh at
  `4504`, and refresh again at `4509`; the anchor is index `4510`.
- exact accepted `build_features`, strict `base_masks(...)[2]`,
  `source_preflight`, `channel_actions` and `channel_memories` produce TRADE
  memory `BACKGROUND` continuously for indices `4500..4509`, ten checkpoints
  or `200ms`.
- the formula therefore produces:

```text
log1p(200ms - 120ms) = log(81) = 4.394449154672439
```

- fixture truth lines 1330-1344 instead freezes
  `leader_background_run_length = 0`.
- fixture truth lines 1299-1303 also freezes the access range as only
  `memories.trade[4504..4509]`. That range cannot establish the registered
  full consecutive run because the run already extends through index `4500`;
  an exact implementation must inspect enough earlier state to find or cap
  the run boundary.

Impact:
- an implementation following the master formula and accepted production
  primitives fails the independent QF13 oracle.
- an implementation returning zero or reading only the six mandatory
  prestate checkpoints silently replaces "consecutive BACKGROUND run" with
  "minimum required prestate", changing the registered H1 feature.
- causal-prefix qualification cannot pass honestly.

Required closure:
- independently re-derive and freeze the QF13 background-run value from the
  accepted primitive output.
- freeze an executable access algorithm and corresponding exact ledger range
  for boundary discovery/capping, then refresh truth SHA/blob identities.

### P1-2 Crash-state labels still cannot be terminalized under the frozen PASS/FAIL receipt rules

Evidence:
- surface crash states mark every point after attempt-root creation terminal
  and non-rerunnable, including before the attempt lock, before the
  consumption commit, after a remote push but before its receipt, during the
  producer, and after the producer but before the verifier.
- the only terminal branches require a tracked terminal receipt containing
  `consumption_commit`, `consumption_push_receipt_sha256`,
  `formal_exit_code`, `terminal_verifier_exit_code` and verifier-result
  identity, followed by a terminal commit that is a descendant of the
  consumption commit.
- after `attempt_root` but before the consumption commit, those mandatory
  identities do not exist and the descendant requirement cannot be met.
- after the consumption push but before its receipt, remote state proves the
  ref moved but cannot reconstruct the lost exact push stdout/stderr hashes
  required by `push_receipt_fields`.
- after a successful producer exit but before verifier execution, no verifier
  receipt or registered verifier `first_error` exists. The FAIL precedence
  permits only `FORMAL_PRODUCER_EXIT_NONZERO` or the verifier's registered
  first error.
- after the terminal push but before its receipt, QA is required to record
  the SHA256 of a receipt that was never durably created.
- the recovery clause says to derive terminalization from durable process
  receipts and observed state only, but several registered crash windows
  occur before the required durable receipt exists.

Impact:
- the state machine has labels for these crashes but no valid artifact/commit
  sequence satisfying its own exact schemas.
- Round 4 P1-4 is not fully closed: verifier FAIL can now terminalize on the
  normal path, while interruption recovery still cannot.
- the claim may be consumed with neither a valid PASS nor FAIL terminal
  history despite the plan requiring exactly one of the two classifications.

Required closure:
- register exact interruption/recovery branches for every crash window,
  including which controller-only transitions may be completed without
  rerunning producer/verifier.
- add distinct observation-recovery receipt schemas where the original child
  or push receipt cannot exist; do not pretend lost stdout/stderr bytes can be
  reconstructed.
- define exact first-error and missing-value semantics for pre-verifier and
  verifier-internal interruption states.

### P1-3 Formal producer identity and absent-ref push receipt bytes are not uniquely frozen

Evidence:
- the only exact formal argv is the outer runner command that validates and
  consumes the armed claim.
- `formal_process_receipts.producer_fields` requires a child `argv` and actual
  waitpid exit status, but neither the plan nor surface contract registers the
  exact formal-producer child argv or invocation mode.
- a correct implementation therefore has to invent an internal producer
  subcommand/flag, while the independent verifier has no frozen expected argv
  against which to verify "exactly once".
- the consumption push intentionally requires an absent controller ref, but
  `push_receipt_fields` requires `old_sha` and `pre_ls_remote_sha` without
  freezing an absent-ref token or field type. `formal_identity.json` likewise
  requires `controller_pre_sha` without an exact absent representation.
- the consumption receipt SHA is itself embedded in formal identity and the
  terminal receipt, so choosing `NONE`, empty string or zero SHA changes
  authoritative bytes.

Impact:
- even on a no-crash normal path, more than one process/receipt byte stream
  satisfies the prose.
- the package verifier cannot prove that the frozen producer command, rather
  than an implementation-chosen command, was the single executed producer.
- consumption/formal-identity bytes are not uniquely derivable from the
  machine contract.

Required closure:
- freeze the exact producer child argv, process topology and expected argv
  comparison.
- freeze types/domains and the exact absent token for every push/formal
  identity SHA field, including absent-ref prestate.

## Hostile Cross-Checks

Passed checks:
- reviewed commit and worktree were exact and clean at review start.
- parent, truth, surface, task and accepted source SHA256/Git blob identities
  match.
- all accepted callable AST identities match.
- strict duplicate-key parsing passes both JSON authorities.
- exact production schema is `17` row fields plus `10` metadata fields; the
  `12 + 5` consumer partition covers all and only row fields.
- all 15 full fixtures produce registered anchor counts, identities and
  structural outcomes under accepted primitives except the separately tested
  QF13 H1 model-input oracle in P1-1.
- QF03 contradiction, QF04 60s censor, QF05 tie precedence, QF06 segment
  censor and QF09 dual-depth follower are realizable.
- QF14/QF15 strict reset state is exact and cross-segment carry is zero.
- QF07/QF08/QF12/QF15 slices independently produce eligible epochs `[1,2]`
  and nonempty anchor/outcome support.
- QF07/QF08/QF15 semantic preimage hashes recompute exactly.
- package arithmetic is exact: `57` unique files, `56` terminal-manifest
  preimage files, `17` directories and `37` readiness files.
- plan and surface freeze the same 23-field `feature_calls.csv` header.
- truth and surface freeze the same 14 hostile probes in the same order.
- temporary bare-repository tests confirmed both registered absent-ref and
  exact-old `--force-with-lease` command forms succeed.

Failed checks:
- QF13 background-run model value and causal access range.
- crash recovery artifact/commit closure.
- exact formal-producer child identity and absent-ref receipt preimage.

## Verification Performed

- read AGENTS, workflow-kit manual/templates, task_plan/progress/findings,
  task, frozen parent, Revision 5 plan, both machine contracts, accepted
  production sources and Round 4 report.
- verified all current/frozen file SHA256, Git blobs and callable AST hashes.
- ran strict JSON duplicate-key checks and semantic-preimage hash checks.
- reconstructed all fixture arrays from truth using synthetic data only.
- executed accepted production `build_features`, strict `base_masks`,
  `source_preflight`, `channel_actions`, `channel_memories` and
  `epoch_support_ledger`.
- independently reconstructed full/slice anchors, reset state, structural
  outcomes and QF13 model inputs.
- checked CSV/schema/package/readiness arithmetic and hostile-probe identity.
- exercised the registered controller CAS commands against a temporary bare
  repository only.
- did not execute formal code or access historical/outcome data.

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/3/0/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
