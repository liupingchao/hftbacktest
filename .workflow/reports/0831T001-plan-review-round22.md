# 0831T001 Plan Review Round 22

执行线程：
- 独立 plan review 线程

任务ID：
- 0831T001

状态：
- 未通过

reviewed commit：
- `3579dcd5759ea272e59d0c4a5003ee0cafeb691d`

severity counts：
- P0：0
- P1：4
- P2：0
- P3：0

findings：
1. P1：`recovery_evidence_state` 缺少 witness-only tuple，并且
   `WITNESS_MISMATCH` 与 START_ONLY/COMPLETE 重叠、反向依赖 workflow
   outcome，未形成 partition。
2. P1：QA 的 `INVALID + raw SHA256` 不能编码 symlink、directory、FIFO
   等 nonregular corruption。
3. P1：已有 VALID_QUARANTINE 后，deterministic rebuild 再次部分写入并
   崩溃会形成 quarantine + mismatched temporary；当前单-quarantine
   规则把正常二次 crash 升级为 A12。
4. P1：artifact post-receipt restart row 只覆盖 A09-A12；terminal receipt
   已存在时的 A01-A08 first match 仍无 restart row。

accepted checks：
- exact commit 工作树干净。
- A12 valid-terminal-receipt preservation row、无 terminal render/commit：
  接受。
- Darwin `renamex_np(RENAME_EXCL=0x00000004)`、final FD inode binding、
  禁止 pathname unlink：接受。
- `.publishing/.abandoned` ownership 与 initial committed-path exclusion：
  接受。
- ordinals 为唯一 `0..22`，共 23 variants。
- 736 mutation rows aggregate：
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- 124416 PRE_BLOCKER rows、18 legal、124398 invalid，aggregate：
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER 21384 / 11 / 21373，aggregate：
  `d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356`。
- plan/task/surface/truth SHA256 与 Git blob：匹配。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ref 与 tag：未创建。

结论：
- `FAIL`
- Revision 22 不得解锁 implementation。
- 下一版必须建立无循环的 recovery-state partition、细分 QA path kind、
  支持重复 quarantine crash，并覆盖 A01-A12 post-receipt restart。

提交信息：
- `review: reject 0831T001 plan amendment round 22`
