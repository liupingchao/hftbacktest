# QA 验收结果

执行线程：
- 独立 QA 验收线程

任务ID：
- `0820T001`

状态：
- 已通过

更新时间：
- 2026-08-21 14:39 CST

验收线程：
- 独立 QA 验收线程

验收对象：
- `0820T001 / RESEARCH-PACKAGE-TRUST-KERNEL-LAYERED-IDENTITY-AND-STAGE4-PARITY`
- implementation commit：
  `95a66e77ee280e55e124d833bb5c5d72e2b3e359`
- candidate/evidence commit：
  `42b242c0217a0db191a1325bdf5100ed7a9187cb`
- QA candidate：
  `/home/molly/project/durable_archives/research_package_trust_kernel/qa_candidates/0820T001/42b242c0217a0db191a1325bdf5100ed7a9187cb`

验收范围：
- 第三轮独立复核第二轮 QA 的两项返修：
  - archive/envelope operation 的真实 start/completion 采集及严格时序断言；
  - Mac full source-semantic admission 与 amdserver kernel-only admission
    的 host/package/runbook 分离。
- 按执行方案独立验收 Gate 0-7。
- 保持 accepted registry 为空，Stage H0-A 不启动，不修改 research bytes、
  Stage 4 package 或 kernel source universe。

P0/P1/P2/P3：
- `0 / 0 / 0 / 0`

验收步骤：
1. 验证 candidate provenance、165-file Gate inventory、bundle、同步收据、
   detached checkout、源分支和 `origin/episode-research` 一致性。
2. 执行 Gate 0、Trust Kernel/workflow focused suite、Stage 4 focused tests、
   shell syntax、Python compile、Ruff 和 registry-only 校验。
3. 在 fresh candidate checkout 执行 hostile preflight；在 Mac formal package
   和 frozen runtime source 上执行独立 full source-semantic admission。
4. 在 amdserver durable archive 上仅执行
   `--verify-kernel-only`，不执行 source-semantic replay。
5. 独立核对 archive/cleanup receipt self-hash、R/C/E/composite binding、
   exact tree、zero-write、严格时间链和 registry/H0-A 状态。

实际结果：
- Candidate detached checkout、源工作树和 `origin/episode-research` 均精确
  对齐至 `42b242c0217a0db191a1325bdf5100ed7a9187cb`；工作树干净。
- Bundle verify 通过；Gate inventory 为 165 files，path/bytes/mode/SHA
  mismatch=`0`。
- Gate 0 通过：`7 surfaces / 27 artifacts / 10 declared+executed
  mutations / EC1-EC7`。
- Trust Kernel/workflow focused suite：`50 passed`。
- Stage 4 focused suite：formal package `97 passed`，amdserver archive
  `97 passed`。
- `bash -n`、`py_compile`、`git diff --check` 通过；Mac formal package
  Ruff：`All checks passed!`。
- Fresh hostile preflight 通过：
  - receipt：
    `cf1b843335a42dc65cf6d9ad17edfece4669b6e3365ef4c25fa21a56d43d8d2d`
  - topology：`98/36/12/4/10`
  - fail-open：`0`
  - kernel source/frozen SHA：
    `cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203`
  - Surface Matrix SHA：
    `c21d27b2e55cb14cf2e3edb24f2b63de91e72e8bd4e89d83144c539e46048831`
- Fresh Mac full source-semantic admission 通过，使用 formal package 的
  frozen `runtime_source`：
  - first-full receipt：
    `08f0b704f21b69455eee30896d0e5a0459ca29167594bec43640ff2d28bd3f89`
  - parity report：
    `ee6837a68f0b460b1999c85e12c225335906bce38511c1e920d79c7329935584`
  - `source_semantic_verified=true`
  - `full_rebuild_count=0`
  - `package_mutation_count=0`
  - `pre_post_metadata_exact=true`
  - package=`107 files / 21 dirs / 1,561,307,420 bytes`
- Fresh full admission 的 R/C/E/composite：
  - R：
    `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`
  - C：
    `f1c84731b71400254a6742b8620db7f096349dac7e4e7d3de906d0f66bc9a579`
  - E：
    `7320058cf2021a132fd1869d8864727b33be8a99e054d75856b772f70d04535a`
  - composite：
    `50680a81b02f7cedac3fef7881d560d075e7b8b110ba2cceeb0d33ab50ebb128`
- amdserver kernel-only admission 通过：
  - report：
    `300f5cbd51193279b4f2239267fcd0d8e74009cb2aa951fa8824a2445eb68649`
  - `admission_mode=kernel_package_only`
  - `accepted_source_semantic_evidence_consumed=true`
  - `kernel_trust_admission_portable=true`
  - `full_source_semantic_replay_portable=false`
  - `source_semantic_replay_executed=false`
  - `strict_receipt_order_verified=true`
  - `pre_post_metadata_exact=true`
- Formal package 与 durable archive 的 identity、99 research-file anchor、
  107-file exact tree 和 1,561,307,420 bytes 全部一致；source、destination、
  post inventory 完全相同，archive package 无 forbidden entry。
- 正式 candidate 的历史证据链严格成立：
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
  - 上述时间满足严格 `<`。
- Archive receipt self-hash：
  `6382813911c6aaf88e99fcf1e56d63c37b29981efa21af1c1b481b22697162bf`。
- Cleanup receipt self-hash：
  `50c9615cd151c0c63df03589dc28ad818cb24849e78df58bc9b3c9956742a499`。
- Archive/cleanup 的 R/C/E/composite binding 与 fresh full admission、
  kernel-only admission exact；cleanup attestation 严格晚于 archive completion。
- Formal package 保留，七个 exact empty-dir cleanup target 不存在，task
  processes remaining=`0`。
- `baselines/research_package_trust_kernel/accepted_versions.json` 保持
  `registry_revision=0`、`versions=[]`，raw SHA：
  `d4e045a5aeca78288ce38d66497ace3baca1962058c583adfc33ac5644b0d285`。
- Stage H0-A 继续锁定；QA 未执行 registry promotion。

验收结论：
- 已通过
- 结论说明：
  - 第二轮 QA 的 P1 chronology proof 和 P2 portability/runbook 缺陷均已
    通过实现、负向边界和独立 durable evidence 验证关闭；Trust Kernel、
    Stage 4 parity、archive、cleanup 和 portability contract 满足任务
    acceptance criteria。

通过项：
1. Gate 0-7 全部通过。
2. Hostile topology、Surface Matrix stable-code mutation 和 fail-closed
   约束通过。
3. Mac formal package full source-semantic admission 与 amdserver
   kernel-only admission 均通过，且明确禁止不可移植的 source replay。
4. Archive/cleanup 严格时序、self-hash、exact tree 和 R/C/E/composite
   binding 全部通过。
5. Research bytes、Stage 4 package、kernel source universe 和 accepted
   registry 未发生非授权修改。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. 保持本轮 QA 报告为当前最终验收事实源。
2. 由总控按既定流程执行 closure/promotion；QA 本轮不写 accepted registry。
3. 在 controller closure 完成前继续保持 Stage H0-A 锁定。

提交信息：
- commit：待提交
