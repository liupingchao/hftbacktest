# 线程回报

执行线程：
- 业务线程-python/tutorial-notebooks

任务ID：
- 0804T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0804T003.md`
- `.workflow/reports/0804T003-business.md`
- `examples/Getting Started.ipynb`
- `examples/Working with Market Depth and Trades.ipynb`
- `examples/Data Preparation.ipynb`
- `examples/Fusing Depth Data.ipynb`
- `examples/Order Latency Data.ipynb`
- `examples/Impact of Order Latency.ipynb`
- `examples/Accelerated Backtesting.ipynb`
- `examples/Level-3 Backtesting.ipynb`
- `examples/Integrating Custom Data.ipynb`
- `examples/tutorial_reproduction/README.md`
- `examples/tutorial_reproduction/notebook_support.py`
- `examples/tutorial_reproduction/refresh_notebooks.py`
- `examples/tutorial_reproduction/test_notebook_support.py`
- amdserver:
  `local_live_analysis/tutorial_reproduction_0804T003/executed_notebooks/`
- amdserver:
  `local_live_analysis/tutorial_reproduction_0804T003/20250801_300s/`

action：
- 为九个 notebook 增加可直接 Run All 的活动 Tardis 测试区。
- 共享 helper 自动检测 amdserver
  `/home/molly/data/tardis/binance-futures` 和 Mac
  `~/Documents/tardis`，并允许通过环境变量覆盖根目录、日期、窗口和输出。
- 每本 notebook 只执行自身对应实验，九本共享 prepared-data 缓存。
- 原教程 markdown 保留；原代码转为 raw reference cell，因此不会在
  Run All 时访问旧下载链接、缺失数据或非 Tardis 输入。
- 生成器使用 nbformat 结构化写入、固定 cell ID，并支持重复生成。
- L3 notebook 正常执行并返回 `blocked_expected`，没有把 L2 数据声明为
  真实 MBO/L3。
- commit 推送后，amdserver 使用 `git pull --ff-only` 同步。

verify：
- focused pytest：
  `python -m pytest examples/tutorial_reproduction/test_run.py
  examples/tutorial_reproduction/test_notebook_support.py -q`
  -> 本机和 amdserver 均为 `5 passed`。
- 九个 notebook `nbformat.validate` 通过。
- 每本恰好 `4` 个活动 code cell，活动代码均通过 AST 语法检查。
- 生成器连续执行两次，九个 notebook SHA256 完全一致。
- 本机 30 秒 `Getting Started` helper 冒烟执行为 `passed`。
- amdserver 使用 2025-08-01、300 秒窗口，按指定顺序运行九次
  `python -m jupyter nbconvert --execute`，全部退出码为 `0`。
- 九个 executed notebook 均通过 nbformat 校验；每本四个 code cell 的
  execution count 为 `[1, 2, 3, 4]`，error output 为 `0`。
- 九份 notebook manifest 齐全，状态为
  `5 passed / 3 adapted / 1 blocked_expected`。
- 共享 prepared data 为 `563454` 条 fused event。
- amdserver notebook/helper/task 文件 Git status 干净，HEAD 为
  `4f22b6106f9d938a1f5780e8ea12e3db1f25c730`。
- `git diff --check` 通过。

done：
- 用户可在 amdserver 的
  `/home/molly/project/hftbacktest/examples/` 直接打开九个 notebook
  并执行 Run All。
- 第一本构建共享 Tardis 缓存，后续 notebook 复用缓存并运行各自实验。
- 执行副本和 manifest 留在 ignored output root，不进入 Git。
- 原教程内容仍在 notebook 内可供对照，但不会干扰可重复执行。

blockers：
- 真实 L3/MBO 数据仍不可用，因此 Level-3 notebook 只能如实展示
  `blocked_expected` 和 L2 控制实验；这不阻塞 notebook 本身 Run All。

commit：
- 4f22b6106f9d938a1f5780e8ea12e3db1f25c730

提交信息：
- make tutorial notebooks runnable with Tardis
