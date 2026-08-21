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
- `examples/hyperliquid/research_package_trust/**`
- `examples/hyperliquid/research_package_trust_cli.py`
- `examples/hyperliquid/research_package_trust_stage4_adapter.py`
- `examples/hyperliquid/test_research_package_trust_*.py`
- `.workflow/workflow-kit/research-package-task-template.md`
- `.workflow/workflow-kit/validate_research_package_task.py`
- `.workflow/workflow-kit/test_validate_research_package_task.py`
- `.workflow/runners/0820T001_archive_stage4_to_amdserver.sh`
- `.workflow/reports/0820T001-*`
- `.workflow/tasks/0820T001.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 第一轮独立 QA 于 `2026-08-21 10:28 CST` 返回 `未通过`，
  `P0/P1/P2/P3=0/4/1/0`。本轮严格限定为同一任务的五项 trust repair，
  未修改 research bytes、Stage 4 package、研究语义或 accepted registry。
- Gate 2 先复制 source snapshot，再动态加载 frozen package；`49` 个 unique
  aggregate mutations、六类 tree attacks x 三个 boundary、六类 production
  CLI attacks 均在 current/frozen 各执行一次。
- Gate 1 分离 empty-bootstrap raw pin 与 future accepted registry semantics；
  增加 append-only prior revision、exact acceptance package、receipt/
  inventory/source bytes 和 accepted pin 验证。
- Gate 0/EC6 实际执行 Surface Matrix 的全部 `10` 个 mutation，并 exact
  比较 declared stable error code。Archive/cleanup 使用结构化
  `ARCHIVE_TREE_MISMATCH` / `CLEANUP_PREFLIGHT_FAILED`。
- Source inventory 对 repo root 和每个 source path 同时 canonical resolve，
  覆盖 macOS `/tmp -> /private/tmp` relocation。Stage 4 adapter 支持 QA
  独立指定 first-full、layer-assignment 和 parity 输出路径。
- 第一轮业务 evidence 原样保留到
  `.workflow/reports/0820T001-superseded-pre-qa-round1-repair/`；重新生成
  hostile、business full parity、candidate source snapshot 和远端 current
  trust envelope。
- amdserver 保留
  `trust_envelope_superseded_pre_gate0_fix` 与
  `trust_envelope_superseded_pre_qa_round1_repair`。新 current envelope
  安装完成后生成 cleanup final-envelope attestation，绑定最终 archive
  receipt、R/C/E/composite、七目录持续缺失和正式 package exact identity。

verify：
- Kernel candidate source/snapshot SHA256：
  `cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203`。
- Surface matrix / API contract / negative matrix / fixture inventory
  SHA256：
  `c21d27b2e55cb14cf2e3edb24f2b63de91e72e8bd4e89d83144c539e46048831`
  /
  `2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f`
  /
  `f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97`
  /
  `4e16522ba448ec1a3e2cc7f0384a80558f130adfdaf8834501818f04661c7bf4`。
- Hostile preflight receipt：
  `cc2c335dd5ba379d721cdc40552be6a075717d8cfc93f26fe2fc9412fdc89797`；
  aggregate/direct-tree/production-shape/metamorphic/surface-contract
  =`98/36/12/4/10`，fail-open=`0`。Topology exact 为
  `49 x current/frozen`、`6 x 3 x current/frozen` 和
  `6 x current/frozen`。
- First-full-admission start receipt：
  `8aa0a333646f2e04886f240c0d3e7857c9a77a2ef0b591655d5edff5cce783e9`。
- Gate 0 canonical validator通过：`7` surfaces、`27` artifacts、`10`
  declared/executed negative mutations、`7` exit criteria。
- Final R/C/E/composite：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`
  /
  `f1c84731b71400254a6742b8620db7f096349dac7e4e7d3de906d0f66bc9a579`
  /
  `7320058cf2021a132fd1869d8864727b33be8a99e054d75856b772f70d04535a`
  /
  `50680a81b02f7cedac3fef7881d560d075e7b8b110ba2cceeb0d33ab50ebb128`。
- Stage 4 parity report：
  `11c493aef12a848cb309c918071a26d23ebb1895e629400b8723c653ae17a41a`；
  `107` files、`21` descendant dirs、`106` artifacts、
  `1,561,307,420` bytes，legacy core/full/contract/manifest exact，
  source semantic verified=`true`，rebuild/package mutation=`0/0`，
  pre/post metadata exact=`true`。
- Durable archive：
  `/home/molly/project/durable_archives/skhynix_episode_research_v1/stage04_jul30_episode_v3/669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433`；
  receipt
  `6e1d42f197d916fab5bcc8b610958b81d0c8487e810d05470fdafcb3cc5fbdb4`，
  byte exact/kernel admission portable=`true/true`，full source-semantic
  replay portable=`false`，temp/task processes remaining=`0/0`。
- Cleanup final-envelope binding receipt：
  `c4dd06d2efa502163f08010d996a2e484aae67662c067a87513924fe11dc6413`；
  保留原 exact mutations=`7`、recursive/glob delete=`false/false`，并在
  final archive 完成后重新证明七目录持续缺失、正式 package 保留且 full
  identity 仍为
  `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433`。
- Receipt 顺序：
  `2026-08-21T03:13:31.813992Z <
  2026-08-21T03:13:39.232975Z <
  2026-08-21T03:28:16.895742Z <
  2026-08-21T03:28:26.863661Z <
  2026-08-21T03:28:31.624512Z`。
- 聚焦 Trust Kernel/workflow suite：`50 passed`。
- current / amdserver archived Stage 4 focused suite：
  `97 passed / 97 passed`。
- Ruff：`All checks passed!`；runner `bash -n`、purity/boundary scan、
  `py_compile`、registry bootstrap、receipt self-hash/time/identity binding
  和 `git diff --check` 均通过。

done：
- 第一轮 QA 的五项 finding 均有实现、永久测试和重建 evidence；EC1-EC7
  business evidence 完整，进入第二轮独立 QA。
- `baselines/research_package_trust_kernel/accepted_versions.json` 保持
  `registry_revision=0`、`versions=[]`，raw SHA256
  `d4e045a5aeca78288ce38d66497ace3baca1962058c583adfc33ac5644b0d285`。
- Candidate 未标记 accepted；registry promotion 只允许在独立 QA 通过后由
  controller closure commit 执行。v2 Stage H0-A 继续锁定。

blockers：
- 无

commit：
- 268b3795874c6487d3fadbd1a5ea812b803ded46

提交信息：
- fix: close trust kernel QA gaps
