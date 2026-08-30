# 0831T001 Independent Hostile Plan Review Round 3

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
  `fb4fe0e4ee1d0155604d6fc47e57850938ea35dc`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 3 plan SHA256 / Git blob:
  `9a95a84e53a33d96541aaf82e4fba0a71a363a4cbf972d34eb1294adb9b1a032`
  / `032e6c9f46ffb75c5f07e7c48a91bac6b5f85fa8`
- fixture-truth SHA256 / Git blob:
  `1640b76a690e0e17a1ed2ff788b412a36f04f73e21e9f893aff6d7854cdae6b2`
  / `f4ca3b35d01bae9f10e0eb36fd75b3a87c8ce798`
- surface-contract SHA256 / Git blob:
  `b4b6cd7bc5335e05746a26a677f29e03db42513668ace4a5bca69aec99cac64a`
  / `4ef330fe363532079fc3b54a7fbd9fc25f9c1e90`
- task SHA256 / Git blob:
  `1029e4ef914f4788ec007fb122ef15eae090321cb5cbd96685af36c6c16c962d`
  / `747f1227c140907e2648e29ffbf0342f7c9f755e`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未运行 formal、A-1a、A-1b、A0、live/private/order 操作。
- 未修改 master、plan、task、fixture truth、surface contract、runner、
  verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/9/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 3 增加了真实的机器合同，也关闭了若干 Round 1/2 finding，但当前
合同仍不能唯一实现为 production-isomorphic、truth-independent、可持久化
复验的一次性 Q0。按任务规则，任一 P0-P2 均阻断 implementation。

## Prior Finding Closure Matrix

| Prior finding | Round 3 status | 说明 |
|---|---|---|
| R1 P1-1 / R2 P1-3 fixture self-proof | `NOT_CLOSED` | semantic preimage 已冻结，但 QF04 truth 与自身 horizon 冲突，见 P1-3 |
| R1 P1-2 A-1a causal no-read | `CLOSED_AT_PLAN_LEVEL` | stage entry、CausalView、AvailabilityView、OutcomeView 和 QF13 access oracle 已注册 |
| R1 P1-3 / R2 P1-1 source schema | `NOT_CLOSED` | 27-field count正确，但 `tick_size` row/metadata partition 仍不等于 accepted production，见 P1-1 |
| R2 P1-2 rolling/base formulas | `NOT_CLOSED` | base predicate基本一致；rolling-prefix 语义与 accepted callable 相反，见 P1-2 |
| R1 P1-4 slice non-vacuity | `CLOSED_FOR_QF07_QF08` | exact start、common epochs、floors、semantic hash 均非空且 internally exact |
| R1 P2-3 reset fixture | `NOT_CLOSED` | QF14/QF15 存在，但 pre-boundary signal 未达到 production base eligibility，见 P1-4 |
| R1 P1-5 / R2 P1-4 A/B/P binding | `NOT_CLOSED` | runtime receipts 已命名但未进入任何 exact artifact schema，见 P1-5 |
| R1 P1-6 / R2 P1-5 package exactness | `NOT_CLOSED` | path/file arithmetic改善；CSV canonical bytes 仍不唯一，见 P1-7 |
| R2 P1-6 first-fail precedence | `PARTIAL` | ordinary missing/extra 已移到 AB/AP 前；QF12 post-publication first-fail 仍不可达，见 P1-6 |
| R2 P1-7 readiness projection | `NOT_CLOSED` | projection 已缩窄，但 pre-consumption package identity 与 normalization 未闭合，见 P1-8 |
| R1 P1-7 / R2 P1-8 one-shot | `NOT_CLOSED` | exact names/fields/commands已有；transition order、receipt location 和 remote CAS 仍矛盾，见 P1-9 |
| R1 P2-1 claim scope | `CLOSED` | Q0 明确只资格验证 shared core，不再覆盖 matching/model/statistics |
| R1 P2-2 parent identity | `CLOSED` | current master bytes与 frozen parent SHA/blob 完全一致 |
| R2 P2-1 AGENTS route | `CLOSED` | 0831T001 supersession 与唯一 formal-task authority 已明确 |

## Findings

### P1-1 Production schema 仍与 accepted cache partition 不同构

证据：
- plan lines 335-360 和 surface contract lines 456-623 把 `tick_size` 定义为
  `float32[n]` row-aligned field，并声称这是 production schema v4 的 exact
  isomorphism。
- accepted fixed-epoch authority
  `examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py`
  lines 122-155 明确只有 17 个 row-aligned fields；`tick_size` 属于第 10 个
  metadata field。
- fixture truth lines 1112-1125、1179-1196 又把 `tick_size` 当作可在
  `4511:9000` 逐行 post-anchor mutation 的时序数组；accepted scalar metadata
  没有这个时间域。

影响：
- 27 个字段的总数相同掩盖了 17+10 与 18+9 的角色差异。
- 严格实现 Revision 3 会拒绝或重塑 accepted production cache；严格复用
  accepted partition 又会违反 surface contract。
- QF13 无法同时是 production-isomorphic 和当前 truth 所定义的 mutation。

必须闭合：
- 按 accepted authority 恢复 `tick_size` metadata shape/dtype，或冻结并让
  Q0/后续 scientific entry 共同调用一个明确的 scalar-to-row adapter。
- 相应重写 QF13 mutation domain，禁止把 scalar metadata 伪装成未来序列。

### P1-2 Rolling-window 公式与绑定的 accepted callable 相反

证据：
- surface contract line 36 和 plan lines 426-435 要求窗口在 segment 开头使用
  “available same-segment prefix rows”。
- frozen accepted `build_features` 的 `rolling_sum`
  (`skhynix_flow_coherence_a_minus1_audit.py` lines 280-290) 在不足完整
  `count` 个 checkpoint 时保持 `NaN`，只从 `count-1` 行开始写值。
- master protocol lines 767-769 也要求完整历史不可用时 missing，禁止 prefix
  substitution。

影响：
- Revision 3 一方面要求直接调用并冻结 accepted `build_features` AST，
  另一方面冻结了不同的 feature bytes、availability 和 sentinel 语义。
- full/slice/reset 的开头区域会产生不同 FeatureBundle；两个实现不可能同时
  满足 accepted callable 和 surface contract。

必须闭合：
- 将 surface/plan 的 rolling boundary 改为 accepted full-window semantics，
  并重新计算受影响的 fixture/access/hash truth。

### P1-3 QF04 的独立 truth 与 60s censor contract 数学冲突

证据：
- fixture truth lines 281-295 把 anchor 固定在 `90.2s`，却把
  `CENSOR_60S` 固定为 `135.0s`、latency `44.8s`、event sequence `6750`。
- plan lines 506-517 和 master lines 523-531 规定 censor 是 `t+60s`、segment
  boundary、gap、invalid book 或 source end 的最早值。
- default fixture lines 30、55-56 覆盖到 `179.98s`，QF04 没有 segment/gap/
  validity patch。因此应为 `150.2s`、latency `60s`、event sequence `7510`。
- 同一 truth 的 QF14 lines 1215-1220 已使用这个正确的 `150.2s` endpoint。

影响：
- 正确 production outcome builder 必然被 verifier 判错；为使 QF04 PASS 而
  特判 `135s` 又会破坏 scientific horizon。
- tracked truth 虽独立于 observed output，但其 oracle 本身不成立，fixture
  independence finding 未闭合。

必须闭合：
- 修正 QF04 的 exact censor tuple，并重新冻结 truth SHA/blob。
- 增加静态 hostile test，从 anchor、source end 和 boundary patches 独立复算
  censor timestamp，不接受手填 tuple。

### P1-4 QF14/QF15 reset fixture 仍是 vacuous

证据：
- QF14 truth lines 1225-1240 和 QF15 lines 1359-1374 只在 boundary 前一行
  `2999` 写入 opposite `trade_signed/trade_total`，随后在 index `3000`
  切 segment。
- accepted `base_masks` lines 442-467 要求 fast 与 medium 两个窗口都具有
  finite trade ratio 和至少一个 finite depth ratio。
- 这些 fixture 在 boundary 前没有 depletion/OFI denominator patch；depth
  ratios 均为 `NaN`。因此 index `2999` 的 `base_eligible=false`，
  `channel_actions` lines 718-750 产生 `GLOBAL_INVALID`，不是 `NEW_NEG`。
- plan line 560 却宣称 QF14 会清除“pre-boundary opposite trade memory”。

影响：
- index `3000` memory 为 unknown、cross-segment carry 为零，即使 reset
  implementation 完全缺失也可能成立；QF15 的 full/slice reset identity同样
  没有非空 pre-boundary state。

必须闭合：
- 在 boundary 前构造满足 fast/medium trade+depth eligibility 的真实
  `NEW_NEG` memory，并冻结 boundary 前 state/age 与 boundary 后 clear tuple。
- hostile mutation 必须从底层 action/memory rows 重算，而不是只修改汇总值。

### P1-5 Child runtime receipts 未进入 54-file evidence closure

证据：
- surface contract lines 421-442 要求每个 feature call 持久化 hasher、loader、
  detector exit、frame hash、field-access hash、sender close 和 receiver EOF。
- package contract lines 765-774 只允许四个 evidence CSV 加一个 manifest。
- `feature_calls.csv` schema lines 851-858 仅含 11 个 producer summary 字段，
  不含 `detector_exit_sha256`、三个 exit code、`frame_sha256`、
  `field_access_sha256`、`receiver_eof_observed` 或 `sender_closed`。
- 其他 JSON schema也没有 per-call receipt rows；`fixture_source_evidence.json`
  只有 aggregate fields（surface lines 210-219）。

影响：
- formal 后运行的 independent verifier 无法从 package bytes 证明 loader/
  detector/FD/EOF path 真正执行，也无法区分真实 B/P calls 与事后填写 ledger。
- Round 2 的 physical A/B/P consumer binding finding 仍成立。

必须闭合：
- 将 exact per-call receipt schema 放入一个允许的持久 artifact，或扩充
  `feature_calls.csv`，并纳入 evidence/terminal manifests。
- verifier 必须从这些 typed rows、physical inputs 和 child hashes独立复算
  57-call closure。

### P1-6 QF12 post-publication first-error 仍不可达

证据：
- surface contract lines 112-116 定义 baseline 为“slice 已发布、无 terminal
  package”，却要求 first error `TERMINAL_CLOSURE_ABSENT`。
- truth error precedence lines 69-80 在 terminal closure 前仍有
  `BUILD_INPUT_BINDING`、`PACKAGE_PATH_SET`、schema、lineage、truth 和 AB/AP。
- formal gates plan lines 1255-1277 同样在 Q0-12 terminal closure 前执行
  Q0-6 physical binding、Q0-7 package closure 和 Q0-8..Q0-11。

影响：
- 当 structural package 尚未创建时，Q0-6 或 Q0-7 必先失败；不可能让所有
  earlier gates PASS 后到达 Q0-12。
- 若 verifier 特判跳过缺失 package，则与 exact gate ownership 和 ordinary
  missing-artifact contract 冲突。

必须闭合：
- 将该 probe 的 expected first error 放到实际最早可达 boundary；或用完整
  clean package 仅删除一个 terminal-closure artifact，并精确冻结此前所有
  gates仍可计算的 mutation。

### P1-7 Canonical CSV bytes 仍没有唯一 serializer

证据：
- plan lines 800-881 冻结 headers/sort keys，lines 884-891 冻结 token、ASCII
  和 LF，但没有冻结 quote character、quoting mode、doublequote/escape、
  embedded delimiter/newline policy 或 final-record newline。
- `slice_invariance.csv` 的 `common_epoch_ids_json` 正常值如 `[1,2]` 含逗号，
  因而不同合法 CSV writers 会产生不同 bytes。
- surface hostile mutation lines 143-147 只覆盖 CRLF，不覆盖
  `QUOTE_ALL`、alternate quoting 或 escaping。

影响：
- runner 与 independent verifier 没有唯一 producer-canonical bytes；
  raw/sealed manifest、A/B/P equality 和 NONCANONICAL_CSV first-fail 可由实现
  自选 serializer。

必须闭合：
- 冻结完整 CSV dialect 和 exact trailing-newline rule。
- hostile minimum 加入 alternate quoting/escaping/QUOTE_ALL mutation。

### P1-8 Pre-consumption readiness package 与 formal identity contract 不闭合

证据：
- plan lines 1018-1046 要求 claim consumption 前在两个 worktree 生成并由同一
  verifier 接受“complete qualification” evidence packages。
- 54-file package强制包含 `contracts/formal_identity.json`
  （lines 776-785）。
- surface lines 228-241、269-276 要求该 JSON 从 claimed bytes、
  attempt-lock、consumption commit 和 controller observations 派生；这些
  authority 在 pre-consumption readiness 阶段按定义尚不存在。
- readiness projection surface lines 444-453 只列四个 directory/path 名和一句
  normalization prose，没有定义 readiness-mode identity、recursive file
  preimage、normalized evidence path set或 field-level transform。

影响：
- 同一个 verifier CLI 无法唯一判断 readiness package 的
  `formal_identity.json` 应填真实 formal authority、sentinel、还是省略；任一
  选择都会违反 package schema、pre-consumption order 或“independently
  verify”要求。

必须闭合：
- 明确独立 readiness package schema/mode，或把 readiness 比较限定为不需要
  formal identity 的 sealed structural projection。
- 冻结 exact projected file list/tree hash和 normalized evidence field
  transform，禁止 generic root-string replacement。

### P1-9 One-shot transition order与可审计状态仍不唯一

证据：
- plan lines 1091-1094 按文字顺序先 rename claim，再创建 attempt root/lock。
- surface exact `transition_order` lines 381-390 则先创建 attempt root、再
  O_EXCL lock、然后 rename claim；plan lines 223-224 规定 prose 与 surface
  不同即 fail closed。
- surface只冻结 push-receipt field set（lines 358-370），plan/task没有冻结
  consumption/terminal push receipt 的 exact paths、canonical bytes或它们
  属于哪个 commit tree。
- controller command lines 354-356 是普通 `git push`，没有绑定 expected old
  SHA/no-create lease；“observe absent -> push”不是 atomic no-replace CAS。
- crash-state table lines 335-342 仅命名到 consumption controller update，
  没有覆盖 receipt、producer、verifier、baseline copy、terminal commit/tag/
  push之间的 distinct observable states。

影响：
- 实现无法同时遵守两个消费顺序。
- race、partial push 或 post-consumption crash 后，reviewer无法仅从冻结
  artifacts唯一判断 controller transition、receipt durability 和 terminal
  authority。

必须闭合：
- 统一 prose 与 machine transition order。
- 冻结两份 push receipt 的 exact path/serializer/commit ownership，并使用
  expected-old remote CAS 或明确单-writer threat model。
- 完整枚举 attempt-root 创建后的所有 crash boundaries及唯一 terminal
  interpretation。

## Positive Checks

以下 Revision 3 修复成立：
- parent protocol、task、fixture truth、surface contract 的 current SHA/blob
  与文档绑定一致；worktree 在审查开始时干净。
- accepted fixed-epoch tag peel 到
  `f06eb5cb012cb62b2a778ad90d433c4083f9ba14`；tag 为 annotated tag。
- Q0 claim 已正确缩窄到 shared core，不覆盖 A-1a matching 或 A-1b model
  qualification。
- QF07/QF08/QF15 semantic-preimage SHA256 均可按 compact sorted JSON独立复算。
- QF07/QF08 的 slice start、common epoch `[1,2]` 和 anchor floor 非空。
- QF08 的 index `2260` / `45.2s` patch 在 accepted 100ms/500ms ratios下形成
  negative raw onset，且位于 epoch-0 core close `45s` 之后；slice从 `60s`
  开始，目标语义成立。
- QF13 已冻结 14 个 model-input values 和 causal access ranges；typed stage
  access boundary在 plan 层面可实现。
- ordinary package missing/extra path closure 已移到 AB/AP identity 前。
- 17-directory、54-file、terminal-manifest 53-file preimage 算术一致。
- AGENTS current route 已明确 supersede 2026-08-20 historical route。

这些通过项不足以覆盖上述九个 P1。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/9/0/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
