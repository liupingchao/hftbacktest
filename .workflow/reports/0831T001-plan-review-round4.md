# 0831T001 Independent Hostile Plan Review Round 4

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
  `9621848d4f8a94fdc1274a056a093ab6232cb44b`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 4 plan SHA256 / Git blob:
  `a6408ecbc34c395e1f8ac203c2f034b03ccf3495de28af1d845b00f851781d87`
  / `1b3a39978015b8fd8a05310235911a290c2b1496`
- fixture-truth SHA256 / Git blob:
  `6d96e3ce580da85a30f349c346bcee4a390031ac1280c1233ded1591d6273006`
  / `ebc9a085b64dc6036260bcaa94f44945f8af4ee3`
- surface-contract SHA256 / Git blob:
  `a78497bb5a207c5772f5c4e9218e95d01eadaa691b4990ce87956ade2228d5b2`
  / `fe878b2539b44e165655b2ed50b218eba69a9ebd`
- task SHA256 / Git blob:
  `fd55bca2626d475afd089c0adafe87655186fa9eadd76910d6e9a2e20a07522a`
  / `375673f1e1b0c4ee1802db11c78aa7712f424068`
- Round 3 report SHA256 / Git blob:
  `19500b306d3c10cfd94d66404a3aacc85bbe535b078f915f34f041e3bc389d84`
  / `36cd3971c815ee65574a44bfa3bf92dda5fd2706`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache 或 outcome artifact。
- 未运行 formal、A-1a、A-1b、A0、live/private/order 操作。
- 未修改 master、plan、task、fixture truth、surface contract、runner、
  verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/4/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 4 关闭了多数 Round 3 finding，但 reset truth、QF12 first-error、
canonical receipt CSV 和 one-shot terminalization 仍不能同时满足 frozen
production authority、prose 和 machine contract。只有 `0/0/0/0` 才可 PASS。

## Round 3 P1 Closure Matrix

| Round 3 finding | Round 4 status | 说明 |
|---|---|---|
| P1-1 production `17+10` partition | `CLOSED` | surface 与 accepted cache 均为 17 row + 10 metadata；`tick_size` 为 `float32[1]` metadata |
| P1-2 accepted full-window rolling | `CLOSED` | prefix substitution 已删除，5/25 checkpoint 不足时保持 `NaN` |
| P1-3 QF04 60s censor | `CLOSED` | `90.2s + 60s = 150.2s`，对应 index/event_seq `7510` |
| P1-4 QF14/QF15 non-vacuous reset | `NOT_CLOSED` | trade onset 非空，但 registered follower memory/age tuple 不等于 accepted production output；见 P1-1 |
| P1-5 persisted child receipts | `NOT_CLOSED` | 23 fields 已持久化，但 plan 与 machine authority 冻结了不同 CSV header order；见 P1-3 |
| P1-6 QF12 reachable first error | `NOT_CLOSED` | “one published slice” 不能通过更早的 complete slice-publication gate；见 P1-2 |
| P1-7 machine CSV dialect | `CLOSED_FOR_DIALECT` | dialect、final LF、QUOTE_MINIMAL 与 QUOTE_ALL mutation 已冻结；header authority conflict 仍由 P1-3 阻断 |
| P1-8 37-file structural-only readiness | `CLOSED_AT_PLAN_LEVEL` | exact 37-file unique subset、tree preimage、no formal identity 和 no post-formal regeneration 已冻结 |
| P1-9 lease-CAS one-shot | `NOT_CLOSED` | CAS/receipt paths 改善成立，但 controller initialization 与 failed-formal terminal branch 仍无唯一状态机；见 P1-4 |

## Findings

### P1-1 QF14/QF15 reset truth 与 accepted production primitives 不一致

证据：
- fixture truth lines 1245-1295 / 1454 onward 注册：
  - neutral seed `2989`
  - trade refresh `2993`
  - follower refresh `2995`
  - negative trade onset `2999`
  - segment boundary `3000`
- truth lines 1223-1242、1432-1450 和 plan lines 561-575 要求：
  - `actions[2999] = [3,5,5]`
  - `memories[2999] = [-1,0,0]`
  - `memory_ages_ms[2999] = [0,80,80]`
  - `memories[3000] = [9,9,9]`
- accepted full-window `rolling_sum` at
  `skhynix_flow_coherence_a_minus1_audit.py:280-290` and accepted
  `base_masks` at `:442-467` make `base_eligible=false` at both `2994` and
  `2998`: the required trade fast-window denominator temporarily rolls out.
- accepted `channel_memories` clears all channel memory on
  `GLOBAL_INVALID` (`skhynix_fixed_causal_epoch_mstate_a_minus1.py:807-818`).
- independent synthetic reconstruction using the exact accepted current
  blobs produced:

```text
2995 base=1 actions=[5,4,4] memories=[9,0,0] ages=[-1,0,0]
2998 base=0 actions=[0,0,0] memories=[9,9,9] ages=[-1,-1,-1]
2999 base=1 actions=[3,5,5] memories=[-1,9,9] ages=[0,-1,-1]
3000 base=0 actions=[0,0,0] memories=[9,9,9] ages=[-1,-1,-1]
```

- the reconstructed source files exactly match the frozen accepted SHA/blob:
  - flow source:
    `f7dc1565...400c` / `494c203e...9f520`
  - fixed-epoch source:
    `dfa8af1f...070` / `5672a8ca...d1a`
- QF14 additionally contains duplicate JSON key `memory_at_index_3000` at
  truth lines 1210 and 1238. A strict duplicate-key parser and a last-key-wins
  parser therefore do not share one canonical parsed authority.

影响：
- a correct implementation of the accepted callables fails the frozen truth.
- an implementation special-casing follower memory to `[0,0]` violates the
  accepted production authority.
- QF15's semantic SHA is internally reproducible but seals the wrong
  production tuple, so hash consistency does not restore truth independence.
- full/slice reset comparison remains nonempty but is not production-valid.

必须闭合：
- rebuild QF14/QF15 so accepted production primitives actually produce the
  registered pre-boundary action/memory/age state without an intervening
  invalid clear.
- remove duplicate keys, rerun a duplicate-key rejection check and re-freeze
  all affected semantic preimages/SHA/blob identities.

### P1-2 QF12 post-publication `BUILD_INPUT_BINDING` is still not the unique reachable first error

证据：
- surface lines 112-115 and plan lines 1083-1087 define the baseline as
  A/B/P inputs plus **one** durably published slice, with no 57-call closure
  or structural package.
- truth precedence lines 71-74 places `SLICE_PUBLICATION` and
  `SLICE_IDENTITY` before `BUILD_INPUT_BINDING`.
- surface lines 52-61 require four slice fixtures per build:
  `QF07,QF08,QF12,QF15`, in fixture authority order with FULL before each
  fixture's SLICE.
- at a QF12 post-publication interruption, QF15 is later in the frozen order;
  the current baseline does not state that all other eleven required A/B/P
  slice publications and their identity inputs already exist.

影响：
- a verifier enforcing the registered complete slice gate can fail
  `SLICE_PUBLICATION` before reaching `BUILD_INPUT_BINDING`.
- a verifier skipping absent non-QF12 slices to force the expected code would
  weaken the ordinary full/slice closure and make gate ownership probe-specific.

必须闭合：
- either freeze a baseline in which every earlier slice-publication/identity
  obligation is complete and only the consumer closure is absent, or register
  the actual earliest `SLICE_PUBLICATION` error.

### P1-3 The persisted 23-field receipt CSV has two exact header orders

证据：
- plan lines 863-965 says CSV schemas and field order are exact.
- plan lines 914-922 orders the three detector environment fields as:

```text
detector_environment_entry_count, detector_cwd,
inherited_fd_violation_count
```

- surface `csv_contract.schemas["evidence/feature_calls.csv"].fields`
  at lines 855-879 orders them as:

```text
inherited_fd_violation_count, detector_cwd,
detector_environment_entry_count
```

- surface `production_instrumentation.per_call_receipt_fields` lines 454-477
  uses the second order.
- plan lines 237-238 require fail-closed behavior where prose and surface
  differ rather than choosing one authority.

影响：
- all 23 required values are present, but there is no single canonical header
  or file byte stream satisfying both exact contracts.
- the independent verifier cannot accept one order without violating the
  other, so evidence-manifest and terminal-manifest bytes are not uniquely
  implementable.

必须闭合：
- make the exact plan header, machine CSV schema and per-call receipt order
  byte-identical, then re-freeze plan/surface identities.

### P1-4 One-shot state machine has no exact controller-init or failed-formal terminal branch

证据：
- surface lines 367-375 freezes `controller_init` and `controller_config`
  commands, but `transition_order` lines 401-418 contains no initialization or
  configuration step.
- the first transition instead says preflight verifies the controller ref
  absent **without writing state**. The exact controller repository is
  currently absent, so initialization cannot be both omitted and performed by
  the no-write preflight.
- crash state `after_terminal_verifier_fail` is labelled
  `FORMAL_FAIL_FIXED_NO_RERUN` at surface line 353, but the only terminal
  sequence is:
  verifier -> copy accepted package -> terminal receipt/report -> terminal
  commit/tag/push.
- plan lines 1324-1337 require every formal failure to record the first
  boundary, completed/missing stages and `Q0_PIPELINE_NOT_QUALIFIED`.
- `terminal_receipt_fields` lines 391-399 always requires
  `package_terminal_manifest_sha256`; no absent-package token or failed-build
  receipt schema is registered.
- receipt ownership prose distinguishes the tracked terminal result receipt
  from the untracked terminal-push receipt, while surface key
  `terminal_tracked_ownership` describes “terminal receipt bytes” under the
  attempt root. The named object is therefore not unambiguous.

影响：
- controller creation/configuration has no frozen chronology, crash boundary
  or commit ownership.
- a producer/verifier failure before a terminal manifest cannot be converted
  into the required durable terminal receipt/commit/controller state without
  inventing an unregistered sentinel or skipping required fields.
- immediately after verifier failure, `FORMAL_FAIL_FIXED_NO_RERUN` may have no
  durable artifact proving what was fixed.

必须闭合：
- place controller initialization/configuration in an explicit
  pre-consumption authority stage with exact retry/crash semantics.
- freeze separate PASS and FAIL terminalization branches, including exact
  missing-package representation, receipt/report fields, commit/tag/push
  ownership and recovery rules.
- name terminal result receipt and terminal push receipt ownership separately.

## Hostile Cross-Checks

Passed checks:
- JSON syntax parses and all frozen SHA256/Git blob identities match current
  bytes.
- production schema has exactly 17 row fields and 10 metadata fields;
  `tick_size` is scalar metadata and QF13 no longer mutates it as a time series.
- full-window rolling semantics match the accepted callable.
- QF04 independently recomputes to
  `150200000000 / 7510 / 60000ms`.
- QF07/QF08 semantic hashes recompute exactly; both freeze common epochs
  `[1,2]`, epoch floor `2` and anchor floor `1`.
- QF15 semantic hash also recomputes exactly, but its reset preimage is rejected
  by P1-1 because the frozen tuple is not produced by the accepted authority.
- negative-probe order and expected error pairs are identical across truth and
  surface contracts.
- package layout arithmetic is exact:
  - 12 structural + 5 evidence files per build
  - 6 terminal files
  - 57 unique files total
  - 56 terminal-manifest preimage files
  - 17 exact parent directories
- readiness projection has 37 unique files, all are package-layout members,
  and no formal identity is required.
- CSV dialect fixes are exact for delimiter, quotechar, QUOTE_MINIMAL,
  doublequote, no escapechar, LF, final LF and QUOTE_ALL rejection.
- consumption and terminal push commands now use absent-ref/exact-old
  `--force-with-lease` CAS forms.

Failed checks:
- duplicate-key scan finds one duplicate key in fixture truth:
  `QF14.expected.memory_at_index_3000`.
- accepted primitive reconstruction rejects the registered QF14/QF15 follower
  memory and age tuple.
- exact receipt CSV order differs between plan and machine authority.
- partial QF12 baseline cannot prove all earlier slice gates pass.
- one-shot success chronology does not cover controller creation or terminal
  failure publication.

## Verification Performed

- read AGENTS, workflow-kit manual/templates, task_plan/progress/findings,
  task, master protocol, Q0 plan, both machine contracts and Round 1-3 reports.
- verified commit scope with `git show` and confirmed review-start worktree was
  clean.
- validated JSON syntax and independently scanned duplicate object keys.
- recomputed SHA256/Git blobs and QF07/QF08/QF15 canonical semantic hashes.
- independently counted package files/directories, manifest preimages,
  readiness projection and 23-field receipt schemas.
- compared truth negative-probe order against surface hostile mutations.
- reconstructed QF14/QF15 with accepted full-window/base/action/memory
  callables using synthetic arrays only.
- did not read historical cache/outcome data and did not execute formal code.

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/4/0/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
