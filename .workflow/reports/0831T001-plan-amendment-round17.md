# 0831T001 Plan Amendment Round 17

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 16 的唯一 P1，使 controller token domain 与
  receipt-sensitive legality 具有唯一解释。

amendment：
- observed controller token 的语法域为 `ABSENT or 40-hex`。
- observed token 的合法性只由当前 durable proof stage 对应的 exact
  `expected_sha_sets_by_local_state` membership 决定。
- 任意 set 外 token 都是 `CONTROLLER_REF_DIVERGENCE`，包括 durable
  consumption/terminal receipt 后出现的 `ABSENT`。
- 同步统一：
  - `CONTROLLER_REF_DIVERGENCE.trigger`
  - `controller_precheck_phase_table.POST_ATTEMPT_ROOT`
  - `controller_divergence_value_domains.observed_sha`
  - execution plan recovery prose
- 其他 surface fields、fixture truth、restart rows、Git phases 和 formal
  one-shot 规则不变。

new authority：
- execution plan SHA256 / blob：
  `0369379087dab0b1f2cd9ab4c6be5b6a34e56c6765a0a0b67e935c41384352ab`
  / `858484e6bedcafc8a6a50aae5bc63e3c47f89e20`
- task SHA256 / blob：
  `05310858a48b6b17872d64c3f19eaed76a5905b694ae6703aa5f1c4081e6c5de`
  / `96623460205ee47ed50ed76d14a59d12f314e2fb`
- surface SHA256 / blob：
  `a77f6fd0d4a36b2be9974c8fcf2d2d920f7ab7b5a2e17b1eead81695bc98600a`
  / `74533850d2bf173c3d2acefb71f2d83bfe7a9999`
- fixture truth：保持不变。

boundary：
- current date：2026-08-31。
- historical cache / future outcome：未访问。
- implementation tag、claim、controller、formal root、receipt 和 task tag：
  均未创建。
- 未提交 round 4 implementation 不属于本 amendment commit。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: amend 0831T001 proof-stage controller legality`
