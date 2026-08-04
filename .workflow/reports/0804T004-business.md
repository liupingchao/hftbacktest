# 线程回报

执行线程：
- 业务线程-python/advanced-tutorial-notebooks

任务ID：
- 0804T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0804T004.md`
- `.workflow/reports/0804T004-business.md`
- `examples/GLFT Market Making Model and Grid Trading.ipynb`
- `examples/High-Frequency Grid Trading.ipynb`
- `examples/High-Frequency Grid Trading - Simplified from GLFT.ipynb`
- `examples/Market Making with Alpha - Order Book Imbalance.ipynb`
- `examples/Market Making with Alpha - Basis.ipynb`
- `examples/Market Making with Alpha - APT.ipynb`
- `examples/Pricing Framework.ipynb`
- `examples/tutorial_reproduction/advanced_experiments.py`
- `examples/tutorial_reproduction/notebook_support.py`
- `examples/tutorial_reproduction/refresh_notebooks.py`
- `examples/tutorial_reproduction/test_advanced_experiments.py`
- `examples/tutorial_reproduction/test_notebook_support.py`
- `examples/tutorial_reproduction/README.md`
- amdserver:
  `local_live_analysis/tutorial_reproduction_0804T004/executed_notebooks/`
- amdserver:
  `local_live_analysis/tutorial_reproduction_0804T004/20250801_300s/`

action：
- 新增七项 Tardis-backed advanced experiments。
- Grid notebook 比较 plain、weak skew 和 strong skew。
- Simplified GLFT 使用 microprice、滚动波动率和自适应网格间距。
- GLFT 从成交到达深度拟合 `A/k`，估计 tick volatility，计算
  `c1/c2`、half-spread 和 inventory skew 后运行五层网格。
- OBI 对 0.5% 深度不平衡做滚动标准化，用 z-score 调整 fair price。
- Basis 使用 `index_price + rolling(futures-index basis)`。
- APT 使用 index return、beta=1 和 futures past price 计算 fair price。
- Pricing Framework 组合 basis、APT return、microprice 和 OBI 四因子，
  对比 zero-alpha 与 combined-alpha 做市。
- 七本 notebook 均改成可 Run All 活动区，原教程保留为 raw reference。
- 原教程所需 Binance spot/FDUSD、多币种和 Bybit 数据在指定挂载不可用；
  Basis/APT/Pricing 明确标为 `adapted`，没有虚构这些输入。

verify：
- 本机及 amdserver focused pytest：
  `test_run.py + test_notebook_support.py + test_advanced_experiments.py`
  -> `9 passed`。
- advanced module/helper/test `py_compile` 通过。
- 七本 notebook `nbformat.validate`、活动代码 AST 和生成器 `--check`
  通过；连续生成 SHA256 一致。
- 本机 Mac Tardis sample 的 30 秒与 300 秒七项执行全部无失败。
- amdserver 使用 2025-08-01、300 秒窗口，按用户顺序执行七次
  `jupyter nbconvert --execute`，全部退出码为 `0`。
- 七个 executed notebook 均有四个活动 code cell，execution count 为
  `[1,2,3,4]`，error output 为 `0`。
- 七份 manifest 齐全，状态为 `1 passed / 6 adapted / 0 failed`。
- amdserver GLFT：
  `A=27.201247`、`k=0.011699`、volatility `49.871515 ticks/sqrt(s)`、
  half-spread `52.752001 ticks`、arrival samples `20820`。
- amdserver Simplified GLFT mean half-spread 为 `248.828408 ticks`。
- amdserver OBI z-score std 为 `1.206048`。
- amdserver basis mean 为 `-59.555688` price units。
- amdserver index-return std 为 `0.000113426`。
- amdserver combined pricing 1 秒 forward IC 为 `0.223848`；仅作为短窗
  教学诊断，不声明统计显著性或生产 alpha。
- 共享 prepared data 为 `563454` 条 fused event。
- amdserver Git notebook/helper/task 文件保持干净，HEAD 为
  `14a9a7442cc6dbd8c7887b0249c4de1c76d8b43b`。
- `git diff --check` 通过。

done：
- 用户列出的七本 notebook 已可在 amdserver 直接打开并 Run All。
- 每本有独立 manifest、策略参数、signal parquet 和回测摘要。
- 所有活动实验只读取现有 Tardis 文件，共享 prepared-data 缓存。
- 大体积执行副本、NPZ 和 Parquet 留在 ignored output root。

blockers：
- 指定日期的 Binance futures 路径只具备 BTCUSDT；原教程的 spot、
  FDUSD、多资产和 Bybit 横截面不能完全等价复现。
- 本任务已用 index price 做明确适配，因此 notebook Run All 不阻塞。

commit：
- 14a9a7442cc6dbd8c7887b0249c4de1c76d8b43b

提交信息：
- add Tardis grid and alpha notebooks
