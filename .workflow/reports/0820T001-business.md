# 业务执行回报

执行线程：
- 业务线程-python/research-infra

任务ID：
- 0820T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/runners/0820T001_archive_stage4_to_amdserver.sh`
- `.workflow/tasks/0820T001.md`
- `.workflow/reports/0820T001-*`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 第二轮独立 QA 于 `2026-08-21 12:04 CST` 返回 `未通过`，
  `P0/P1/P2/P3=0/1/1/0`。本轮严格限定为 archive chronology 和 QA
  host/runbook 两项修复。
- Runner 不再复用 parity completion 作为 archive start。所有 precondition
  通过后、第一项 archive/envelope 工作前采集 observed start；远端
  publish/swap 返回后采集 observed completion。
- Initial archive 和三条 envelope refresh 路径均在写 receipt 前结构化断言
  `parity.completed_at_utc < archive.started_at_utc <
  archive.completed_at_utc`。
- 新增 `--refresh-envelope-after-qa-round2`，保全第二轮 current
  trust envelope/evidence 后只刷新 envelope，不重传或重建 1.5GB package。
- 新增 amdserver 专用 `--verify-kernel-only <output.json>`。该模式验证
  hostile/first-full/parity/archive/cleanup receipt self-hash 和 binding、
  严格时间链、byte-exact package、R/C/E/composite、固定 portability
  contract 和 zero-write；不调用 legacy/source replay。
- QA runbook 明确拆分：
  - Mac formal package、Python 3.10+、package frozen runtime source：
    full source-semantic admission；
  - amdserver durable archive：kernel/package admission only，
    `source_semantic_replay_executed=false`。
- Research bytes、Stage 4 package、kernel source universe 和 accepted
  registry 均未修改。第二轮 QA 报告保持当前失败事实源，等待第三轮 QA
  写新结论。

verify：
- Implementation commit：
  `95a66e77ee280e55e124d833bb5c5d72e2b3e359`
  (`fix: enforce trust archive chronology`)。
- Kernel source/frozen identity 保持：
  `cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203`，
  source files=`20`。
- 旧错误 receipt 对新 kernel-only 入口按预期 fail closed：
  `hostile/full/parity/archive/cleanup order is not strict`。
- Envelope refresh 后严格顺序：
  - hostile completed：
    `2026-08-21T03:13:31.813992Z`
  - first-full started：
    `2026-08-21T03:13:39.232975Z`
  - parity completed：
    `2026-08-21T03:28:16.895742Z`
  - archive started：
    `2026-08-21T05:43:02.377380Z`
  - archive completed：
    `2026-08-21T05:43:05.937537Z`
  - cleanup final-envelope attested：
    `2026-08-21T05:43:07.984663Z`
- Final archive receipt canonical self-hash：
  `6382813911c6aaf88e99fcf1e56d63c37b29981efa21af1c1b481b22697162bf`；
  local/remote file SHA256：
  `87185af37f98b1d3c7419273bac67c6713ceffcd0ae441d07468015326a5c971`。
- Final cleanup receipt canonical self-hash：
  `50c9615cd151c0c63df03589dc28ad818cb24849e78df58bc9b3c9956742a499`；
  local/remote file SHA256：
  `a55a1c84df9574e8a311922cfa202dae36b055b0ef10a7a611efa7ecf1232e57`。
- amdserver 已保全
  `trust_envelope_superseded_pre_qa_round2_repair` 与
  `evidence/superseded_pre_qa_round2_repair`，refresh temp 不存在。
- Corrected Mac runbook full source-semantic admission：
  - first-full receipt：
    `d91ecf5bad57a2624cda8918bf494fc1da31d59f49ed44ada20a2ee0acef670c`
  - parity report：
    `4927a1d90bd5e8d0210ebee61d2f10a77b39a1c6f67211f5cfdf8dd7258475eb`
  - legacy/kernel source semantic verified=`true/true`
  - files/dirs/artifacts/bytes=`107/21/106/1,561,307,420`
  - full rebuild/package mutation=`0/0`
  - pre/post metadata exact=`true`
- amdserver kernel-only admission report：
  `300f5cbd51193279b4f2239267fcd0d8e74009cb2aa951fa8824a2445eb68649`；
  `accepted_source_semantic_evidence_consumed=true`，
  `source_semantic_replay_executed=false`，strict order 和 zero-write
  均为 `true`。
- Final R/C/E/composite：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`
  /
  `f1c84731b71400254a6742b8620db7f096349dac7e4e7d3de906d0f66bc9a579`
  /
  `7320058cf2021a132fd1869d8864727b33be8a99e054d75856b772f70d04535a`
  /
  `50680a81b02f7cedac3fef7881d560d075e7b8b110ba2cceeb0d33ab50ebb128`。
- 聚焦 Trust Kernel/workflow suite：`50 passed`。
- Current formal / amdserver archived Stage 4 focused suite：
  `97 passed / 97 passed`。
- Gate 0 validator：
  `7 surfaces / 27 artifacts / 10 declared+executed mutations / EC1-EC7`。
- Ruff：`All checks passed!`；`bash -n`、`py_compile`、
  source identity、receipt binding、`git diff --check` 通过。
- Formal package 内 `__pycache__` / `*.pyc`=`0`。
- Accepted registry 保持 `registry_revision=0`、`versions=[]`，raw SHA256：
  `d4e045a5aeca78288ce38d66497ace3baca1962058c583adfc33ac5644b0d285`。

done：
- 第二轮 P1 chronology proof 和 P2 portability/runbook finding 均有
  实现、正负验证和 durable evidence。
- 第三轮 QA candidate 可在 amdserver 运行 kernel-only admission；full
  source-semantic admission 的 Mac 命令、host、package、runtime source
  和独立 qa3 outputs 已在 task/runbook 中冻结。
- Candidate 未标记 accepted；registry promotion 仍只允许在独立 QA
  通过后由 controller closure commit 执行。Stage H0-A 继续锁定。

blockers：
- 无

commit：
- `95a66e77ee280e55e124d833bb5c5d72e2b3e359`

提交信息：
- `fix: enforce trust archive chronology`
