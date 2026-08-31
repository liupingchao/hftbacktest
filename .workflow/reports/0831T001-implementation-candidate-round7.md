# 0831T001 Implementation Candidate Round 7

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 业务线程

任务ID：
- 0831T001

状态：
- 待验收

日期：
- 2026-09-01

是否进行QA验收：
- 否

QA说明：
- 当前仅申请 implementation readiness 独立复核。
- readiness PASS 前不得进入 arming、controller creation、formal 或最终
  QA。

review basis：
- round 6 readiness failure commit：
  `62b0438c9b2a27bd776cb254546419df93ca53b3`
- accepted Revision 26 review commit：
  `cf7962ff`

authority：
- execution plan SHA256 / blob：
  `bdc934202cd9ee9e1743830121eec80f1cf3ab7e8bb4f3bbc1f8728c3619f7dc`
  / `87ffe9e70d050342b74b258e9ae5578f56319368`
- task SHA256 / blob：
  `19585d2501994535eca3d462b62860be76c54cc1196609ce9fd9a788255b138b`
  / `f9bcdef809a02635cacfe0da2b2d0aed142862f3`
- surface SHA256 / blob：
  `096b70b70ce723f30d2c719309d3431081041a195933c050d78157b6a4f91657`
  / `5db7f47abcb47935b2c28d93035086d21a47ea1b`
- fixture truth SHA256 / blob：
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`

implementation：
1. PRE_BLOCKER：
   - invalid/source-mismatched receipt 不再提前选择 proof stage；
   - G01-G04 保持优先，receipt preimage defect 最终为 G05；
   - report-present/receipt-absent 的 Revision 25 counterexample 唯一选择
     pre-terminal phase，随后命中 A10。
2. recovery witness：
   - 使用 Revision 26 exact `rev-parse --verify --quiet` observation；
   - direct Git blob CAS-from-ABSENT、object type、exact bytes 与 canonical
     evidence 全部执行；
   - witnessed bytes 成为原始 crash boundary/controller/path-set identity，
     合法字段重写并重算 recovery ID 仍为 A12。
3. control publication：
   - Darwin `renamex_np(RENAME_EXCL)`；
   - target、temporary、quarantine 全部 no-follow FD/inode/bytes 验证；
   - same-byte symlink、nonregular path、invalid inventory fail closed；
   - mismatched regular temporary 移入 content-addressed contiguous ordinal
     quarantine，不执行 pathname unlink；
   - EEXIST exact target/temporary 作为 race evidence 永久保留。
4. recovery reconciliation：
   - observational temporary 在 target absent 时 quarantine；
   - committed exact observational race residue 保留；
   - deterministic temporary 在 target absent 时留给其 exact publisher；
   - committed deterministic target 只接受 exact temporary residue；
   - artifact temporary 在 ordered G/A rule 确定后才提交或 quarantine。
5. runner、verifier 与 tests 全部绑定 Revision 26 plan/task/surface。

verification：
- focused pytest：
  `130 passed in 108.46s`。
- expanded accepted predecessor regression：
  `410 passed, 1 skipped in 131.34s`。
- ruff check / format check、py_compile、runner/verifier `--help`、
  `git diff --check` 与 HEAD object verification：通过。
- full `git fsck --no-dangling`：
  运行约 9 分钟无输出后人工中止；不记录为通过，也不作为本任务必需
  acceptance gate。

boundary：
- historical cache / future market outcome：未访问。
- formal attempt root、baseline、controller repo/ref、witness ref、
  armed/claimed claim、receipt 与三个 task tags：全部不存在。
- formal execution 保持锁定。

remaining gate：
- 提交后必须在 primary 与 fresh detached exact commit 上执行 readiness，
  逐文件 byte comparison，并独立按 ASCII path order 重算 tree SHA256。
- 随后必须完成独立 implementation readiness review；PASS 前不得 arming
  或 formal。

提交信息：
- `implementation: checkpoint 0831T001 Q0 candidate round 7`
