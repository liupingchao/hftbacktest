```md
执行线程：
- 总控

任务ID：
- 0514T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T004.md`
- `.workflow/reports/0514T004-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 根据 controller review，扩展 maker execution outcome research requirements。
- 保留 T004 的需求任务边界：不制定实现方案、不实现 runner、不运行实验、不改策略、不启动 live。
- 在原有 fill probability、time-to-fill、adverse selection after fill、spread capture、queue/priority proxy、cancel-to-fill race、post-only/reject/throttle/churn、inventory impact 基础上，补齐 7 类新增 label：
  - quote placement / distance
  - missed-fill / opportunity cost
  - realized PnL decomposition
  - tail risk
  - partial-fill / order lifecycle
  - inventory cycle
  - sample validity / censoring
- 明确后续分析不能用单一算法覆盖所有 label，必须按 label 类型选择统计方法：
  - continuous labels：Spearman rank correlation 为主，Pearson 为辅，并报告 quantile bucket、top-bottom spread、monotonicity、time split stability。
  - binary labels：event rate、lift vs baseline、odds ratio；logistic/AUC 只能作为辅助排序。
  - time-to-event / censored labels：显式处理 right censoring，优先 Kaplan-Meier / discrete hazard bucket 或 Cox-style hazard。
  - count / rate labels：count rate、exposure-normalized rate、rate ratio；Poisson / negative-binomial 只能作为辅助。
  - multiclass / lifecycle labels：contingency table、conditional probability、mutual information 或 one-vs-rest lift。
  - tail labels：tail quantile、CVaR-like tail mean、worst bucket concentration、exceedance rate。
  - placement / opportunity-cost labels：placement bucket x signal bucket 二维 tradeoff table。
- 同步更新 `task_plan.md`、`progress.md`、`findings.md`，让后续线程读取到同一需求口径。

verify：
- 人工检查 `.workflow/tasks/0514T004.md`、`task_plan.md`、`progress.md`、`findings.md` 的 T004 口径一致。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- T004 需求已覆盖新增 7 类 label，并把每类 label 的统计方法写清楚。
- T004 仍是 requirements-only precursor，不授权实现方案、runner、实验、策略改动、live、queue/fill calibration 或 counterfactual queue/fill proof。
- 当前结果进入 QA 验收。

blockers：
- 无

commit：
- a7e9d87

提交信息：
- docs(workflow): expand T004 maker outcome labels
```
