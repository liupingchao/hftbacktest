# 业务执行回报

执行线程：
- SKHYNIX OBI Reversal Track A3 业务线程

任务ID：
- 0828T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T006.md`
- `docs/skhynix_binance_continuous_background_interpretable_m_state_competing_risk_track_a_plan_20260828.md`
- `examples/hyperliquid/skhynix_obi_reversal_track_a3.py`
- `examples/hyperliquid/test_skhynix_obi_reversal_track_a3.py`
- `local_live_analysis/skhynix_obi_reversal_track_a3_0828T006/`
- `.workflow/reports/0828T006-business.md`

action：
- 验证 0828T005 的 15 个 artifacts、classification、state/target identity
  和 outcome-access closure。
- 冻结 train、blocked-validation、no-refit replay dates。
- 使用 H0-only development-date LODO entry NLL，从冻结 ridge grid
  `[1e-4,1e-3,1e-2,1e-1]` 选择 `0.001`。
- 实现 baseline-category discrete-time multinomial competing-risk hazard；
  no-event 为基准类，follow/fail 各有独立线性预测器。
- H0 使用 6 个 elapsed-time bins 和 19 个 side-oriented、
  train-standardized decision-time features。
- H1 使用完全相同的 risk rows、targets、scaler、time basis 和 ridge，
  只增加每个 cause 一个 `beta_k * reversal_indicator_R`。
- 在 1,296 train entries 拟合，在 3,187 blocked-validation entries 和
  2,681 replay entries 上 no-refit 评分。
- 计算 entry-level competing-risk NLL、integrated/cause-specific Brier、
  cumulative incidence、calibration、OOD、per-date/per-time-bin influence
  和 5,000 次 date-block bootstrap。
- 执行 diagnostic-only ridge sensitivity；禁止其救回 primary。
- 将 A3 失败结论、日期/时间分解和研究停止边界写回 canonical plan。

verify：
- A0/A1/A2/A3 focused pytest：`16 passed`。
- finite-difference multinomial gradient check：通过。
- synthetic reversal-direction recovery：通过。
- CIF probability conservation、risk-row identity 和 H1-only-R contract：
  通过。
- Python compile、`git diff --check` 和 Markdown fence parity：通过。
- 两个独立 A3 builds 的 20 个 artifact manifests 完全一致。
- 正式输出 20/20 artifact size/SHA closure 通过。
- H0/H1 均收敛，maximum absolute gradient 分别约 `9.12e-8` 和
  `7.19e-8`。
- Git 使用 exact-path commit，未提交工作树已有无关 staged/untracked
  files。

done：
- Selected ridge 为 `0.001`；H0/H1 parameter counts 为 `50/52`。
- Train `Delta_NLL=+0.003015`、`Delta_IBS=+0.002690`。
- Blocked validation `Delta_NLL=-0.001603`、
  `Delta_IBS=-0.000675`，H1 比 H0 更差。
- Date-block bootstrap 95% interval 为
  `[-0.003007,-0.000245]`，整个区间低于 0 和 materiality `0.002`。
- 三个 blocked-validation 日期的 Delta NLL 全部为负：
  - Aug07 `-0.000245`
  - Aug24 `-0.003007`
  - Aug25 `-0.001558`
- No-refit replay `Delta_NLL=-0.008676`、
  `Delta_IBS=-0.001749`；Aug26/Aug27 均为负。
- `beta_follow=+0.164963`、`beta_fail=-0.158056`，方向正确但未形成
  cross-date proper-score increment。
- 只有 100ms 和 200-500ms bins 为微弱正增量；600ms 后均为负。
- 排除 first-100ms 后 `Delta_NLL=-0.002711`；排除 first-500ms 后为
  `-0.008775`。
- Ridge `0.1` 仅产生 `+0.000118` 的 negligible validation increment，
  远低于 materiality，不能救回 primary。
- Scientific classification 为 `A3_no_increment_over_H0`。
- 未运行 A4 nulls、stronger-H0、maker economics 或 prospective
  confirmation。

blockers：
- 无 A3 执行阻塞。
- Primary `OBI_REVERSAL_V1` 已在 A3 失败，不能进入正向 A4/A5 路径。
- 新的 path descriptor、barrier、alignment 或 state-machine 假设必须
  另行版本化，不能作为 robustness rescue。

commit：
- 168839e0

提交信息：
- research: test OBI reversal A3 increment
