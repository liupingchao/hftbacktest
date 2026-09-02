# AGENTS.md

This repository uses `workflow-kit` for task dispatch, execution reports, QA acceptance, and progress tracking.

The current active development focus in this worktree is the SKHYNIX
continuous conditional-risk research workflow under:

- `docs/skhynix_continuous_hazard_maker_research_framework_v2.md`
- `docs/research_package_trust_kernel_execution_plan.md`
- `docs/skhynix_trigger_aligned_episode_research_implementation_plan.md`
- `docs/skhynix_stage4_episode_v3_execution_repair_postmortem.md`
- `examples/hyperliquid/cross_exchange_trigger_*`
- `examples/hyperliquid/cross_exchange_*episode*`
- `.workflow/tasks/`
- `.workflow/reports/`
- `baselines/`

Controller route as of `2026-09-02`:

- This isolated worktree was created from exact accepted commit
  `824e0431b96bda16515efb41544fd9e1feb78868` for `0902T002`.
- `0902T002 / SUCCESSOR_Q0_RECOVERY_AND_EFFECT_FREE_PREFLIGHT` is the only
  active engineering task, with status `待执行`.
- Current authority stops at revision-specific successor candidate/plan
  authoring handoff. Effect-free preflight, arming and formal Q0 execution are
  locked.
- The accepted `0902T001` worktree was created from exact commit
  `1051f2b29059e6b7465fe8051de01072f9ff7e19`.
- `0902T001 / TARGET_PROJECT_ARGV_CONTRACT_REPAIR_V1` is closed as
  `已通过` after independent QA at `P0/P1/P2/P3 = 0/0/0/0`.
- `0831T001` remains closed as `未通过`; its plan, implementation, tags,
  arming commit, claim and reports are immutable failure evidence.
- The accepted `0902T001` repair established:
  `exec_argv = [python_executable, script_path, ...args]`,
  `program_argv = [script_path, ...args]`, and Python `sys.argv` must equal
  only `program_argv`.
- A later effect-free preflight must freshly reverify runtime, script, argv,
  cwd, `shell = false`, accepted repair commit and V2.1.1 release identity.
- Business, scientific, historical-cache, future-outcome, private and live
  execution remain prohibited.
- Every candidate, attempt, claim, arming, baseline, output-root,
  controller-ref, tag and report identity must be new and `0902T002` scoped.
- No `0831T001` mutable identity may be reused. Registration creates no claim,
  attempt, receipt, baseline, ref, tag or output root.

External scope authority:

```text
repository = /Users/liu/Documents/workflow-proj
scope_rebaseline_commit = d6f5147f4fd8d98b1a15d4cd90706f015fcc033a
controller_boundary_commit = 45514a9a380552fe03c1a6c5fbdb7a7bf7585d86
controller_authorization_commit =
  288cf86ea4db8abc47477cae5f4a955020786190
controller_authorization_sha256 =
  5f3c1062b526884959240e28adb3684df08ab1f038526871f8b2e39e27d17334
controller_active_boundary_commit =
  8894646cea97b4c1e980b65ff0018d5e265e202b
```

Historical controller route as of `2026-08-20`:

- The user approved the continuous conditional-risk v2 framework as the
  active master research goal.
- Accepted v1 Ordered Research Queue Stages 1-4 remain immutable dependencies.
- Undispatched v1 Stage 5 and later items are superseded and must not be
  dispatched under the old queue.
- The Research Package Trust Kernel was the next mandatory prerequisite at
  that historical point; it is not the current dispatch authority.
- After independent QA accepts the kernel, execution proceeds through
  `Stage H0-A -> QA -> Stage H0-B -> QA`.

## Required Workflow

Before starting a non-trivial task, read:

- `.workflow/workflow-kit/workflow-manual.md`
- `.workflow/workflow-kit/task-dispatch-template.md`
- `.workflow/workflow-kit/thread-report-template.md`
- `.workflow/workflow-kit/qa-acceptance-template.md`
- `.workflow/tasks/0902T001.md`
- `.workflow/tasks/0902T002.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

## Role Model

- 总控：the main planning session. It defines scope, writes task files, reads QA results, and decides the next task.
- 业务线程：the execution session. It changes code or runs verification within the assigned scope.
- 测试线程：used for focused test discovery, regression runs, and evidence collection.
- QA验收线程：checks whether the task met its acceptance criteria and writes the latest QA result.

Default chain:

```text
业务线程 -> QA验收线程 -> 总控
```

## Task Rules

- Every formal task must have a task file under `.workflow/tasks/`.
- Task IDs use `MMDDTxxx`, for example `0510T001`.
- Use only these statuses:
  - `待执行`
  - `执行中`
  - `待验收`
  - `已通过`
  - `未通过`
  - `阻塞`
  - `作废`
- Default to one formal task at a time unless tasks are independent and explicitly marked as parallel.
- Do not silently expand task scope.
- Do not modify unrelated files.

## Report Rules

- Every execution result must be reported under `.workflow/reports/`.
- Business/test reports should normally end in `待验收`, not `已通过`.
- QA reports may end only in `已通过`, `未通过`, or `阻塞`.
- The latest QA result should also be copied to `docs/qa-acceptance-report.md`.

## Verification Rules

For hftbacktest, choose the smallest useful verification command for the task scope.

For the active Binance maker MM work, start with focused commands around `examples/binance_tick_mm`:

- `python -m pytest examples/binance_tick_mm`
- `python examples/binance_tick_mm/maker_acceptance.py --help`
- `python examples/binance_tick_mm/align_live_run.py --help`
- `python examples/binance_tick_mm/run_env_test.py --help`

For repository-level work:

- Rust workspace or core logic: `cargo test`
- Single crate: `cargo test -p <crate>`
- Python bindings: inspect `py-hftbacktest/README.md`; likely requires `maturin develop` before Python tests
- Focused Python tests: `python -m pytest py-hftbacktest/tests`

If a command cannot run because of dependencies or environment setup, report the blocker instead of inventing a passing result.
