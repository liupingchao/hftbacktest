# 0831T001 Plan Review Round 20

执行线程：
- 独立 plan review 线程

任务ID：
- 0831T001

状态：
- 未通过

reviewed commit：
- `3405542a6a30c62b7ee4742620938c178a3d5ec2`

severity counts：
- P0：0
- P1：3
- P2：0
- P3：0

findings：
1. P1：invalid witness ref 分支同时要求发布/验证 A12 blocker 和
   `forbid recovery mutation`；而 blocker publication、claim rename、
   consumption commit/tag 都是 mutation，restart authority 不唯一。
2. P1：A12 receipt 与 QA schema 对 witness absent/invalid、target mismatch
   等 blocker path 不具备 totality。现有 artifact blocker receipt 没有记录
   witness state/OID、target/temp kind/hash，QA 无法验证 A12 原因。
3. P1：exact temporary resume 在 hard-link 后没有重新通过 no-follow FD
   验证 final target 的 kind/bytes/inode，也没有冻结 `EEXIST` race 分支，
   尚未完整执行 control publication protocol。

accepted checks：
- exact commit 工作树干净。
- ordinals 为唯一 `0..22`，共 23 variants。
- 736 mutation rows aggregate：
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- 124416 PRE_BLOCKER rows、18 legal、124398 invalid，aggregate：
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER 21384 / 11 / 21373，aggregate 保持不变。
- plan/task/surface/truth SHA256 与 Git blob：匹配。
- recovery-start temporary 已从 generic reconciliation 排除。
- CAS-from-ABSENT、direct-blob witness 与 Darwin no-follow/FD 操作：
  接受为可实现基础。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ref 与 tag：未创建。

结论：
- `FAIL`
- Revision 20 不得解锁 implementation。
- 下一版必须分离 witness infrastructure blocker 与正常 recovery progress，
  冻结可审计的 witness diagnostic evidence，并补全 final target FD/inode
  verification 与 `EEXIST` 分支。

提交信息：
- `review: reject 0831T001 plan amendment round 20`
