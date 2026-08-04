# 线程回报

执行线程：
- 业务线程-python/tutorial-reproduction

任务ID：
- 0804T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0804T002.md`
- `.workflow/reports/0804T002-business.md`
- `examples/tutorial_reproduction/__init__.py`
- `examples/tutorial_reproduction/README.md`
- `examples/tutorial_reproduction/run.py`
- `examples/tutorial_reproduction/test_run.py`
- `local_live_analysis/tutorial_reproduction_0804T002/mac_20250101_5m/`
- amdserver:
  `local_live_analysis/tutorial_reproduction_0804T002/amdserver_20250801_5m/`

action：
- 盘点并映射“基础用法、数据准备、回测能力”三个系列共九个 notebook。
- 新增串行 runner，兼容本机扁平 Tardis sample 布局和 amdserver
  `YYYY/MM/DD` 日目录布局，不使用 Tardis Key，也不下载数据。
- 对指定时间窗口流式解压 `trades`、`incremental_book_L2`、
  `book_ticker`、`derivative_ticker`，记录源文件和 staged 文件 SHA256。
- 构建 non-fused/fused HBT 数据、末尾快照和三组 feed-derived
  order-latency 文件，再按 notebook 顺序执行核心实验。
- `Impact of Order Latency`、`Accelerated Backtesting` 和
  `Integrating Custom Data` 按现有 Tardis 字段做有说明的适配。
- `Level-3 Backtesting` 检测到输入没有 MBO add/modify/cancel 订单身份，
  明确输出 `blocked_expected`，同时运行 L2 控制实验，不把 L2 冒充 L3。
- 代码提交后推送 `cross-exchange`，amdserver 使用 `git pull --ff-only`
  同步到同一 commit。

verify：
- 本机：
  `/Users/liu/.local/conda/envs/hftbacktest/bin/python -m pytest
  examples/tutorial_reproduction/test_run.py -q` -> `2 passed`。
- 本机 `py_compile`、runner `--help` 和 `git diff --check` 通过。
- 本机使用 `/Users/liu/Documents/tardis` 的 2025-01-01 数据执行
  300 秒窗口：`completed=true`、`failed=[]`、融合后 `327451` 条事件，
  总运行时间约 `6.16s`。
- amdserver 聚焦 pytest -> `2 passed`。
- amdserver 使用
  `/home/molly/data/tardis/binance-futures` 的 2025-08-01 数据执行
  300 秒窗口：`completed=true`、`failed=[]`、融合后 `563454` 条事件，
  总运行时间约 `13.39s`。
- 本机与服务器 manifest 均记录九项顺序状态：
  `passed, passed, passed, passed, passed, adapted, adapted,
  blocked_expected, adapted`。
- amdserver HEAD 与 origin 均为
  `a0c54391daa44b48df4baf1f4d5c0064d7fb65ee`。

done：
- 九个 notebook 都有独立结果 JSON 和顶层
  `reproduction_manifest.json`，包含输入身份、顺序、状态、关键指标和失败列表。
- 基础用法、深度/成交、数据准备、深度融合和订单延迟数据实验在本机及
  amdserver 均通过。
- 三个受原 notebook 外部数据或实现边界影响的实验均以 `adapted` 标记，
  未宣称与原始外部数据完全等价。
- 唯一预期阻塞为真实 L3/MBO；服务器规范化事件统计为
  `DEPTH_EVENT=540213`，MBO add/modify/cancel 均为 `0`。
- 大体积原始数据和生成产物均留在 ignored output root，没有进入 Git。

blockers：
- 当前挂载只提供 L2 增量深度和聚合成交，不具备真实 L3/MBO 订单身份；
  `Level-3 Backtesting.ipynb` 无法在现有 Tardis 数据上完整复现。

commit：
- a0c54391daa44b48df4baf1f4d5c0064d7fb65ee

提交信息：
- add Tardis tutorial reproduction runner
