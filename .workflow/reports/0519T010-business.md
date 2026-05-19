```md
执行线程：
- 业务线程-python

任务ID：
- 0519T010

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T010.md`
- `.workflow/reports/0519T010-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 将 `0519T010` 从 `待执行` 切到 `执行中`，按 planning-only 边界制定 Step 9C multi-sample quote-adjustment validation plan。
- 在 `task_plan.md` 中新增 Step 9C 计划正文。
- 在 `findings.md` 中记录 T010 durable findings。
- 在 `progress.md` 中同步当前状态和下一步方向。
- 未修改策略代码、runner、tests、live scripts、AWS/remote state、生产配置、connector/core、converter/schema 或大数据 artifact。
- 未运行 replay sweep，未采集 live，未 default-on，未 promotion。

Step 9C 计划摘要：
- Step 9 要解决的问题不是单样本 PnL，而是 default-off quote-adjustment candidates 是否能跨场景稳定改善 maker execution quality。
- 当前 immediate blocker 是 current-format data scenario coverage。
- `0519T008` 的角色：
  - 证明 `quote_adjustment_replay.py` runner / metrics / artifact mechanics。
  - 旧样本 `5-13-day-control-30min` 因缺 T006 字段而正确分类为 `needs_more_instrumentation`。
- `0519T009` 的角色：
  - 证明一个 current-format T006 sample 可用。
  - T008 rerun 为 `promising_but_single_sample`，只支持后续 multi-sample validation planning，不支持 promotion 或 live readiness。
- 后续 Step 9 主线：
  1. 先补 current-format no-rule/default-off 样本场景覆盖。
  2. 再用已接受 runner 做 read-only multi-sample validation。
  3. 只有当现有 artifacts 无法执行计划时，才开 runner-change task。
  4. 多样本 QA 之后，才可能考虑 Step 10 tiny-live-design planning task。

场景覆盖要求：
- volatility / markout regime
- spread width / tick distance regime
- trade intensity / fill opportunity regime
- stale / latency / join-age / anchor-age regime
- API / token / throttle / churn regime
- inventory state / recovery regime
- post-only / bad-price / clamp / suppress regime
- cancel-fill / fill-after-cancel risk regime
- top5 / market-view quality regime

样本和事件量门槛：
- Research comparison minimum:
  - at least `4` current-format samples including `5-19-day-control-30min`
  - at least `120` minutes aggregate duration
  - at least `10000` submit orders aggregate
  - at least `250` filled orders aggregate
- Before any later `ready-for-tiny-live-design` classification:
  - at least `5` current-format samples
  - at least `180` minutes aggregate duration
  - at least `15000` submit orders aggregate
  - at least `500` filled orders aggregate
  - at least `2` distinct non-calm regimes

hard gates：
- T006 missing fields `0`
- maker acceptance passed
- action / planned / reject / throttle gates passed
- working semantic / blocking mismatch `0`
- strict replay lag passed
- sidecar/join quality acceptable, with future/gap-crossed join `0`
- post-only crossed-risk after re-check `0`
- archive/raw integrity documented
- no default-on, no production behavior change

candidate classification：
- `reject`:
  - fails hard gates
  - lacks intended-regime coverage
  - worsens fill quality / adverse markout / cancel-fill / API churn in multiple samples
  - uses forbidden inputs
- `keep_for_research`:
  - has coverage and favorable regimes, but sample count, event mass, dispersion, or proxy-only evidence is insufficient
- `ready_for_tiny_live_design`:
  - passes hard gates across the required sample set
  - improves or does not worsen execution quality in most eligible regimes
  - has no catastrophic worst-sample behavior
  - has QA acceptance
  - still authorizes only a separate live-design planning task, not live execution

next recommendation：
- After T010 QA, create a current-format no-rule/default-off sample collection task first.
- Then create a read-only multi-sample validation task using existing `quote_adjustment_replay.py`.
- Do not modify the runner unless the accepted validation plan cannot be executed with current artifacts.

verify：
- Confirmed documents explicitly state:
  - no code change
  - no replay sweep
  - no live
  - no default-on
  - no promotion
  - no single-sample PnL acceptance
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 9C planning-only contract written.
- Next task direction is sample collection first, then read-only multi-sample validation.
- T008/T009 roles and boundaries are recorded.

blockers：
- 无 execution blocker。
- Evidence limitation remains: only one current-format T006 sample is accepted so far.

commit：
- `2b8d88e`

提交信息：
- `docs(workflow): plan step 9 multi-sample validation`
```
