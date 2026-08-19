执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/`
- `.workflow/tasks/0801T004.md`
- `.workflow/reports/0801T004-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增 Goal 3 `--stage baseline` builder。
- 校验 T003 episode manifest、所有 episode outputs SHA、M1 manifest task/schema
  和 accepted primary count。
- 构建 discovery `segment_0001-0003` / held-out `segment_0004-0008` split。
- 用 discovery blocked folds 和 matched-neighbor median baseline 输出
  `baseline_predictions.csv.gz`、`episode_residual_features.csv.gz`、
  `baseline_calibration.csv`、`frozen_research_contract.json` 和
  `heldout_consumption_manifest.json`。
- 使用 discovery residual features 做 robust scaling、PCA、mutual-nearest
  neighbor graph 和 Louvain communities，输出 motif membership、prototype、
  response-curve NPZ、counterexamples、walk-forward stability 和 surrogate
  summary。
- Motif 输入排除 segment ID、absolute timestamp、PnL-like outcome 和 adverse
  markout selection field。
- 本轮发现 `14` 个 response-structure prototypes，但全部分类为
  `response_structure_only`；没有输出 `motif_candidate_supported`、交易信号、
  maker identity、exact fill 或 maker PnL 声称。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py -q`
  -> `10 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
  -> pass
- `python examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py --help`
  -> pass
- 真实 Goal 3 builder：
  `python examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py --stage baseline --m1-dir local_live_analysis/skhynix_liquidity_response_0730T017 --output-dir local_live_analysis/skhynix_liquidity_response_case_hierarchy --task-id 0801T004`
  -> pass。
- 独立 rescan：
  - prediction rows `12,677`
  - available predictions `12,215`
  - discovery available `3,298`
  - held-out available `8,917`
  - residual rows `12,677`
  - calibration rows `16`
  - motif membership rows `10,184`
  - prototype rows `14`
  - surrogate rows `14`
  - graph max degree `10`
  - independent errors `0`
- `git diff --check` 覆盖本轮代码、任务、QA 文档和总控文档 -> pass。

done：
- Baseline outputs:
  - `baseline/baseline_predictions.csv.gz` row count `12,677`, SHA-256
    `80d0a11b1dd9bcf83e0b1e608ba253c2cf52b40fd49993c480d7d7ee379d3a80`
  - `baseline/episode_residual_features.csv.gz` row count `12,677`,
    SHA-256 `1bc5463aa00fb4f5d55666cafd03028b683e25cf343fc7111f05f1f1324e7f47`
  - `baseline/baseline_calibration.csv` row count `16`, SHA-256
    `af0e82b569271b18bc494f9aee37f14abe91e4c0d9aa72cb85987a89c2cfac11`
  - `baseline/frozen_research_contract.json` SHA-256
    `9d51a0d98640b5f3a3d129360656fd67bbf8f7b2c0485f23cccd5a84c0aec668`
  - `baseline/heldout_consumption_manifest.json` SHA-256
    `106bb4519b16f10bc5caed25943a9b5893eee5107dabc48ecdd244d9ffbe9240`
- Motif outputs:
  - `motif/motif_membership.csv.gz` row count `10,184`, SHA-256
    `6122176a155635745f32a445116453c79595b98d1e457be7e44573183ed1a0b3`
  - `motif/motif_prototypes.csv` row count `14`, SHA-256
    `3f8ef696b70625c2ad4ad017aa6ed35fb26fb3e050bfcb817986e809c73d60be`
  - `motif/motif_response_curves.npz` row count `14`, SHA-256
    `db475b6b70d09f027e5208f1bdb6b2a53bc207fd30010c44a4875e599aa18812`
  - `motif/motif_counterexamples.csv.gz` row count `70`, SHA-256
    `28316d9a11254e898a68936f870074b667fac2996fce15d86a23c0e81a105ae4`
  - `motif/walk_forward_stability.csv` row count `70`, SHA-256
    `cac200e6ea7e1f58e515f6746b996fb1a1255e116c8adf1783bd091169d35cc8`
  - `motif/surrogate_test_results.csv` row count `14`, SHA-256
    `c2a7194ac2838c3e97c215efe050546f7a7a13da7e4e9157d933131975994f39`
- 本轮未访问网络、AWS 或 SSH，未新增采集。

blockers：
- 无；等待独立 QA。

commit：
- 无

提交信息：
- 无
