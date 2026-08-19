执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 首轮独立 QA 为 `未通过`；本报告已更新为 provenance/output metadata
  closure 修复后的第二轮验收输入。

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_motif_v2.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_motif_v2.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/motif_v2/`
- `.workflow/tasks/0801T008.md`
- `.workflow/reports/0801T008-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增独立 `motif_v2/` builder，不修改历史 `motif/`。
- 严格验证 T007 R2 discovery/post manifests、baseline contract、Episode v2
  manifests 和十项 source provenance 的实际 row-count/SHA。
- 使用 18 个 episode/phase structural fields、25 个 official HGB
  standardized residuals 和 25 个 observation masks，共 `68` 个 motif
  features。
- Feature allowlist 禁止 adverse/PnL/profit/fee/fill/markout、segment ID、
  timestamp 和 `ts_ns`，实际命中数为 `0`。
- Robust transform、PCA、mutual `k=10` graph、Louvain、community filter、
  medoid、distance p95 和 response curves 全部只在 discovery
  `0001-0003` 拟合。
- kNN 显式按 identity 去除 self，并保留距离顺序后取前十；graph maximum
  degree 为 `10`。
- Prototype 使用最小化 community 内总 pairwise distance 的真实 member
  medoid，不使用 centroid 伪 prototype。
- 运行 `199` 次 full-pipeline response-block permutation surrogate，并做
  BH 校正。
- `0004-0008` 仅使用冻结 transform/PCA/prototype 做 `post_selection`
  distance assignment；不生成正式 held-out 标签或支持结论。
- Surrogate 未通过时直接分类 `not_supported`；只有先过 `q<=0.05` 才可能
 进入 `needs_fresh_holdout`，且无 fresh data 时永远不能 supported。
- 输出 exact manifest/contract/input/output closure 和 deterministic NPZ。
- 首轮 QA 后，existing-package validator 进一步绑定 `_validate_source()`
  重算的 exact 10-role/path/row-count/SHA 集合，并将 baseline manifest、
  post manifest、baseline contract 三项 source SHA 与 provenance 对账。
- 每个 output entry exact key set 固定为
  `path/row_count/sha256`；CSV/NPZ 行数按内容验证，JSON row-count 必须为
  `1`，未知 nested key fail closed。

verify：
- focused tests：
  `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_motif_v2.py -q`
  -> `9 passed`。
- combined：
  `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_baseline_v2.py examples/hyperliquid/test_cross_exchange_liquidity_response_motif_v2.py -q`
  -> `41 passed`。
- `ruff`、`py_compile`、CLI help、internal validator 和
  `git diff --check` -> pass。
- hostile tests 覆盖 duplicate provenance role + coordinated contract/output
  SHA、unknown output nested key 和 JSON row-count `999`；均 fail closed。
- 真实 discovery：
  - feature rows `3,500`，eligible `3,350`；
  - PCA components `8`；
  - graph nodes `3,350`、edges `9,960`、max degree `10`；
  - qualifying communities/prototypes `8`；
  - discovery memberships `874`；
  - 每个 community size `101-130`，覆盖三个 discovery segments，
    max segment fraction `0.3524-0.4851`。
- 真实 surrogate：
  - full-pipeline runs `199`；
  - surrogate max community size p95 `169.1`；
  - 八个 empirical p-value 为 `0.73` 或 `1.0`；
  - 八个 BH q-value 全为 `1.0`；
  - classification 全为 `not_supported`。
- 真实 post-selection：
  - feature rows `9,177`，eligible `9,071`；
  - assignments `6,838`；
  - stability rows `8 motifs x 5 segments = 40`；
  - exact `held_out` token 数 `0`。
- 两次真实构建 manifest、contract 和全部八个 output SHA 完全一致。
- 历史 `motif/motif_manifest.json` SHA 保持
  `007bb7f1c00a9f15a81a86aea336f20d5f2cff963a9e4e36868fe01fed86e783`。

done：
- motif manifest SHA-256
  `0d11d98a3b8fa624e625a6a5ada75b83281f9d5c300298d687a8278729dba221`。
- frozen motif contract SHA-256
  `b61a24eab6a047d0ea467b1a92157500812392d3bf822c59194e25750f19dc82`。
- feature/membership/prototype/curve SHA：
  - `2de501bdf36d27e3068225a44a1b7bc8595b9569d82f7e9ab35fa78fca3009d7`
  - `0bbf6dd6de3502238d5844d1092cfd74f9db86c4082aff2bf366a9e4f69dcbcd`
  - `cc896c673557849007b6900e5ee990003771f249d3f522acd9b55616e609273c`
  - `e506f8907526220e3dbf941a0e01b7a5895ea11609b16ad0c2474438bc5e5437`
- counterexample/post-selection/surrogate SHA：
  - `79fd8f4e3e6d2f4f488e70c444bc129dbb4e9f88ff3a37eb252500acb5d853e6`
  - `73825130e8001e4d6b29e0cdcceb09ef85243921aad6aa53bba3e6b493c8b430`
  - `9811fbfe3c3a67d7638fd96bc97d164a044a7dc531f70e2499b5014079a8a06f`
- 正式 supported motif 数为 `0`，不声明 tradable signal、maker identity、
  exact fill 或 maker PnL。

blockers：
- 无；等待独立 QA。T009 在 QA `已通过` 前保持锁定。

commit：
- 无

提交信息：
- 无
