#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_ID = "0510T002"
RUN_IDS = [
    "5-10-day-control-1h-06",
    "5-9-small",
    "5-9-noon",
    "5-8-stage3-15m-livetest-v4",
]
OUT_DIR = Path("local_live_analysis/stage6j_cross_sample_0510T002")


def run_command(repo_root: Path, cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_ready(repo_root: Path, run_id: str) -> tuple[bool, list[str]]:
    run_dir = repo_root / "local_live_analysis" / run_id
    missing: list[str] = []
    required = [
        run_dir / "config_backtest_audit_replay.toml",
        run_dir / "maker_acceptance.json",
        run_dir / "alignment_report_audit_replay.json",
        run_dir / "backtest_audit_replay_result.json",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path.relative_to(repo_root)))

    manifest_matches = sorted((run_dir / "out" / "live_raw").glob("*/manifest_*.json"))
    if not manifest_matches:
        missing.append(str((run_dir / "out" / "live_raw" / "<symbol>/manifest_*.json").relative_to(repo_root)))
    return not missing, missing


def ensure_maker_acceptance(repo_root: Path, run_id: str) -> tuple[bool, str]:
    run_dir = repo_root / "local_live_analysis" / run_id
    out_path = run_dir / "maker_acceptance.json"
    if out_path.exists():
        return True, f"{run_id}: maker_acceptance.json exists"

    alignment = run_dir / "alignment_report_audit_replay.json"
    backtest = run_dir / "backtest_audit_replay_result.json"
    if not alignment.exists() or not backtest.exists():
        return False, f"{run_id}: cannot generate maker_acceptance.json because alignment/backtest inputs are missing"

    cmd = [
        sys.executable,
        "examples/binance_tick_mm/maker_acceptance.py",
        "--alignment-report",
        str(alignment.relative_to(repo_root)),
        "--backtest-result",
        str(backtest.relative_to(repo_root)),
        "--out",
        str(out_path.relative_to(repo_root)),
    ]
    returncode, output = run_command(repo_root, cmd)
    if returncode != 0:
        return False, f"{run_id}: maker_acceptance generation failed with exit {returncode}: {output[-1000:]}"
    return True, f"{run_id}: generated maker_acceptance.json"


def summarize_candidates(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    candidates = summary.get("candidates", {})
    for name, data in sorted(candidates.items()):
        lines.append(
            "- {name}: runs={runs}, pnl_sum={pnl:.6f}, max_abs_notional={max_pos:.6f}, cancel_fill={cancel_fill}, inv_worsening_no_readd={inv_worse}, same_side_worsening={same_worse}".format(
                name=name,
                runs=data.get("runs", 0),
                pnl=float(data.get("pnl_mtm_sum", 0.0)),
                max_pos=float(data.get("max_abs_position_notional_max", 0.0)),
                cancel_fill=int(data.get("cancel_fill_count_sum", 0)),
                inv_worse=int(data.get("inventory_worsening_no_readd_count_sum", 0)),
                same_worse=int(data.get("same_side_worsening_count_sum", 0)),
            )
        )
    return "\n".join(lines) if lines else "- 无候选汇总"


def derive_next_step(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    hard_failures = int(summary.get("hard_failure_count", 0))
    sample_count = int(summary.get("sample_count", 0))
    if hard_failures:
        return "先处理 Stage 6J hard failures，不进入规则设计或 live micro test。"
    if sample_count < 2:
        return "样本数不足，继续收集 current-format 样本。"

    baseline_rows = [r for r in rows if r.get("candidate") == "baseline_inflight_only" and r.get("status") == "ok"]
    candidate_rows = [r for r in rows if r.get("candidate") != "baseline_inflight_only" and r.get("status") == "ok"]
    if not baseline_rows or not candidate_rows:
        return "缺少 baseline 或候选结果，先检查 Stage 6J 输出完整性。"

    return (
        "保持 diagnostic-only，不自动进入 live micro test。"
        "总控应人工比较跨样本 PnL、max position、drop rate、churn 和 cancel-fill source-path，"
        "再决定是否创建 adverse-selection timing rule 设计任务。"
    )


def write_report(
    repo_root: Path,
    *,
    command: list[str],
    returncode: int,
    output: str,
    missing_by_sample: dict[str, list[str]],
    summary: dict[str, Any] | None,
    rows: list[dict[str, str]],
    next_step: str,
    preflight_notes: list[str],
) -> None:
    report_path = repo_root / ".workflow" / "reports" / f"{TASK_ID}-business.md"
    status = "待验收" if returncode == 0 and summary is not None and not missing_by_sample else "阻塞"
    qa_required = "是" if status == "待验收" else "否"
    qa_note = "无" if qa_required == "是" else "当前任务结果暂不进入QA验收，待总控确认后再决定是否派发QA验收。"

    sample_lines = "\n".join(f"- {run_id}" for run_id in RUN_IDS)
    missing_lines = (
        "\n".join(f"- {run_id}: {', '.join(paths)}" for run_id, paths in missing_by_sample.items())
        if missing_by_sample
        else "- 无"
    )
    summary_lines = summarize_candidates(summary or {})
    preflight_lines = "\n".join(f"- {note}" for note in preflight_notes) if preflight_notes else "- 无"
    hard_failures = 0 if summary is None else int(summary.get("hard_failure_count", 0))
    sample_count = 0 if summary is None else int(summary.get("sample_count", 0))
    candidate_count = 0 if summary is None else int(summary.get("candidate_count", 0))
    decision = "未生成" if summary is None else str(summary.get("decision", "未生成"))

    text = f"""```md
执行线程：
- 测试线程

任务ID：
- {TASK_ID}

状态：
- {status}

是否进行QA验收：
- {qa_required}

QA说明：
- {qa_note}

files：
- .workflow/tasks/{TASK_ID}.md
- .workflow/runners/run_task.py
- .workflow/runners/run_{TASK_ID}.py
- examples/binance_tick_mm/stage6j_replay.py
- {OUT_DIR.as_posix()}/*

action：
- 自动检查跨样本输入是否存在。
- 自动补算缺失的 `maker_acceptance.json`。
- 自动执行 Stage 6J 跨样本 replay。
- 自动解析 `stage6j_replay_decision.json` 和 `stage6j_replay_summary.csv`。
- 自动生成本执行回报。
- 自动刷新 workflow dashboard。
- 未启动真实 live，未连接交易所，未改 AWS 状态。

verify：
- 命令：`{' '.join(command)}`
- exit code：`{returncode}`
- 输出目录：`{OUT_DIR.as_posix()}`
- 样本列表：
{sample_lines}
- 缺失输入：
{missing_lines}
- preflight：
{preflight_lines}

done：
- Stage 6J decision：`{decision}`
- 样本数：`{sample_count}`
- 候选数：`{candidate_count}`
- hard failures：`{hard_failures}`
- 候选汇总：
{summary_lines}
- 下一步建议：{next_step}
- runner stdout 摘要：
```text
{output.strip()[-2000:]}
```

blockers：
- {"无" if status == "待验收" else "存在缺失输入或 runner 执行失败，详见 verify/done。"}

commit：
- 无

提交信息：
- 无
```
"""
    report_path.write_text(text, encoding="utf-8")


def refresh_dashboard(repo_root: Path) -> None:
    build = repo_root / ".workflow" / "build_dashboard.py"
    subprocess.run([sys.executable, str(build)], cwd=repo_root, check=False)


def run(repo_root: Path) -> int:
    preflight_notes: list[str] = []
    for run_id in RUN_IDS:
        ok, note = ensure_maker_acceptance(repo_root, run_id)
        preflight_notes.append(note)
        if not ok:
            break

    missing_by_sample: dict[str, list[str]] = {}
    for run_id in RUN_IDS:
        ok, missing = sample_ready(repo_root, run_id)
        if not ok:
            missing_by_sample[run_id] = missing

    out_dir = repo_root / OUT_DIR
    command = [
        sys.executable,
        "examples/binance_tick_mm/stage6j_replay.py",
        "--local-root",
        "local_live_analysis",
        "--out-dir",
        OUT_DIR.as_posix(),
    ]
    for run_id in RUN_IDS:
        command.extend(["--run-id", run_id])

    if missing_by_sample:
        output = "Skipped Stage 6J replay because required inputs are missing."
        summary = None
        rows: list[dict[str, str]] = []
        returncode = 2
        next_step = "补齐缺失样本输入后重跑。"
    else:
        returncode, output = run_command(repo_root, command)
        decision_path = out_dir / "stage6j_replay_decision.json"
        summary_path = out_dir / "stage6j_replay_summary.csv"
        summary = load_json(decision_path) if decision_path.exists() else None
        rows = read_csv_rows(summary_path) if summary_path.exists() else []
        next_step = derive_next_step(summary or {}, rows)

    write_report(
        repo_root,
        command=command,
        returncode=returncode,
        output=output,
        missing_by_sample=missing_by_sample,
        summary=summary,
        rows=rows,
        next_step=next_step,
        preflight_notes=preflight_notes,
    )
    refresh_dashboard(repo_root)
    print(output.strip())
    print(f"Wrote .workflow/reports/{TASK_ID}-business.md")
    return returncode
