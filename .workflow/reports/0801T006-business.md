执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/episode_v2/`
- `.workflow/tasks/0801T006.md`
- `.workflow/reports/0801T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 保留历史 `episode/` package，新增独立 `episode_boundary_v2` builder 和
  `--stage episode-v2` CLI。
- phase 改为 rolling `100ms` signed shock impact；只有 rolling sign 与当前
  phase 相反，且至少两个连续 atom 同方向支持时，才建立 reversal phase。
- phase rows 新增 rolling window、phase 首尾 rolling impact、确认 atom 数和
  algorithm 字段。
- boundary sensitivity 仅使用 discovery segments `0001-0003` 的 `60,820`
  个 atom，并记录 discovery row SHA。
- 新增 existing-v2 freeze gate：schema、boundary version、参数、phase
  algorithm、discovery split 或既有 output SHA 漂移均失败。
- `episode_v2` 使用独立同文件系统原子发布；构建前后复核历史
  `episode/` 全文件 SHA 不变。
- 本轮未执行 baseline、motif、regime、held-out outcome 读取、信号拟合或
  策略回测。
- 第一轮 QA 后完成 R1：
  - freeze gate 要求六个 required outputs 的 key set 精确相等；
  - 每个 output 的 path、CSV schema、实际 row count 和 SHA 必须闭合；
  - existing-v2 重建在发布前比较 candidate/frozen 六项
    `path/row_count/sha256`；
  - 任一候选结构输出漂移均要求新版本，当前版本直接失败；
  - phase membership conservation 成为 builder acceptance gate。
- 第二轮 QA 后完成 R2：
  - existing-v2 calibration contract 的 discovery atom count 和 canonical
    row SHA 成为同版本冻结输入；
  - discovery 输入任一字段漂移，即使六个结构输出不变，也必须失败；
  - 增加真实 atom catalog unused-field 漂移注入并验证 frozen v2 不变。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py -q`
  -> `22 passed`。
- `python -m py_compile examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
  -> pass。
- `ruff check examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
  -> pass。
- CLI help 包含 `episode-v2` -> pass。
- 真实八段 `episode-v2` builder -> `passes=true`。
- 独立 full-data rescan：
  - atom membership `141,768`，无 duplicate/omission/cross-segment；
  - clusters `48,777`；
  - episodes `12,677`；
  - phase rows `26,428`；
  - 每个 episode 的 phase atom_count 总和等于 episode atom_count；
  - boundary audit `48,769`；
  - discovery sensitivity `13` 行，scope 全为 `discovery_only`；
  - rescan errors `0`。
- v1/v2 membership、cluster、episode 和 boundary-audit SHA 完全一致。
- 同 task/config 重复真实构建，`episode_v2` 全文件 SHA byte-identical。
- R1 hostile tests：
  - 缺失 required output -> fail closed；
  - 额外 output -> fail closed；
  - candidate phase SHA 漂移 -> fail closed；
  - 实际三 discovery segment builder 注入 phase-result drift -> 发布前失败，
    frozen phase SHA 不变；
  - phase 遗漏 membership atom -> fail closed。
- R2 hostile test：
  - discovery atom `basis_mid_bps` 漂移、atom catalog/manifest SHA 同步更新，
    但 episode 结构不变 -> discovery input SHA gate 失败，旧 v2 manifest
    SHA 不变。
- `git diff --check` -> pass。

done：
- `episode_v2/shock_atom_membership.csv.gz`
  SHA-256 `10a0fa622c797c09fe08771d373a0d7eedea6d08a35a35c63ae736d8654e97dc`。
- `episode_v2/shock_cluster_catalog.csv.gz`
  SHA-256 `574a4a89f297544841aafd15587cd94fa9923c194215e11d21cacfde96de8dfb`。
- `episode_v2/continuous_flow_episode_catalog.csv.gz`
  SHA-256 `bd7b3c83536ec6bad7bf9df934a5b05f2ae18203d0fe0103fa7b15d55a6e2439`。
- `episode_v2/flow_episode_phases.csv.gz`
  SHA-256 `778e6bd08a44839fc1188253c7eba9d415192de329d2b614a80716d7362482f2`。
- `episode_v2/episode_boundary_audit.csv.gz`
  SHA-256 `c1a12bef8db7fe56c8f23bc8de729e2f208716381fb5a54d8b06325f79897c4b`。
- `episode_v2/episode_boundary_sensitivity.csv`
  SHA-256 `311ac58111152c4900a0fb989a0caaaed607debae5a165a0a6040000cd97dd90`。
- `episode_v2/episode_manifest.json`
  SHA-256 `be375c83fe3568c9d595037c615b6030798ee0635b565f08ea66bce6f4f53033`。
- discovery atom rows SHA-256
  `bce395180f2773011d323a7647efa7513709cdf4190aa89c53572c2f4d220b2d`。
- 历史 `episode/` 七个文件 SHA 均保持不变。

blockers：
- 无；等待独立 QA。

commit：
- 无

提交信息：
- 无
