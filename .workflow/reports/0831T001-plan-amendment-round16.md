# 0831T001 Plan Amendment Round 16

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 修正独立 implementation-readiness round 3 发现的单字段 authority
  自相矛盾；implementation 和 formal 均不得在独立 plan review 通过前推进。

amendment：
- `controller_divergence_value_domains.observed_sha` 从
  `one 40 lowercase hex SHA outside expected_sha_set_json` 修正为
  `ABSENT or one 40 lowercase hex SHA outside expected_sha_set_json`。
- 该修正与既有 prose、receipt-sensitive expected sets 和 Revision 11
  语义一致：durable consumption receipt 后 controller ref 消失到
  `ABSENT` 是 `CONTROLLER_REF_DIVERGENCE`。
- blocker 字段集合、blocker code、expected-set 规则、restart rows、
  Git phases、fixture truth、negative probes 和 formal one-shot 规则均未
  改变。

new authority：
- execution plan SHA256 / blob：
  `3590b6df71caa8521d8e7003231ca21fc2a452935b4f0b49bcf1a2dfd52b5ad4`
  / `38c3f17ae2c93a649bfe2d64497dfdafc3abf95b`
- task SHA256 / blob：
  `2e24448486f59cbf225797c7341fe83eab281e15dd06721bbe7b1ea4aa98b0f1`
  / `93ebbdc5b8f61b78b884fbfb5c32dc588a3a374a`
- surface SHA256 / blob：
  `6d6ce0e7733ed17d888bfa7dd0fd1033f854f8dd4df3a7f82b503dedbe2066f9`
  / `1dfd67e18786a797796cff5e12e1d84b5c5cef81`
- fixture truth：保持不变。

boundary：
- historical cache / future outcome：未访问。
- implementation tag、claim、controller、formal root、receipt 和 task tag：
  均未创建。
- round 4 implementation working changes 不属于本 amendment commit。

结论：
- 请求独立 plan review。
- review 通过前不得继续实现 authority-dependent behavior，不得 arming 或
  formal。

提交信息：
- `plan: amend 0831T001 ABSENT controller divergence domain`
