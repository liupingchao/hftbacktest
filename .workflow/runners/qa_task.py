#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / ".workflow" / "tasks"
REPORT_DIR = ROOT / ".workflow" / "reports"
QA_ACCEPTANCE_REPORT = ROOT / "docs" / "qa-acceptance-report.md"
BUILD_DASHBOARD = ROOT / ".workflow" / "build_dashboard.py"

FINAL_STATUSES = {"已通过", "未通过", "阻塞"}


@dataclass
class MarkdownRecord:
    path: Path
    fields: dict[str, list[str]]
    raw: str

    def first(self, key: str, default: str = "") -> str:
        values = self.fields.get(key, [])
        return values[0] if values else default

    def all(self, key: str) -> list[str]:
        return self.fields.get(key, [])


def normalize_task_id(value: str) -> str:
    raw = value.strip().upper()
    compact = raw.replace("_", "-")
    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})-?T(\d{3})", compact)
    if match:
        month, day, task_num = match.groups()
        return f"{int(month):02d}{int(day):02d}T{task_num}"
    match = re.fullmatch(r"(\d{4})-?T(\d{3})", compact)
    if match:
        mmdd, task_num = match.groups()
        return f"{mmdd}T{task_num}"
    return raw


def parse_markdown_file(path: Path) -> MarkdownRecord:
    text = path.read_text(encoding="utf-8")
    fields: dict[str, list[str]] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"```md", "```"} or line.startswith("#"):
            continue

        if line.endswith("：") or line.endswith(":"):
            current_key = line.rstrip("：:")
            fields.setdefault(current_key, [])
            continue

        if current_key is None:
            continue

        if line.startswith("- "):
            fields[current_key].append(line[2:].strip())
        elif line[:2].isdigit() and ". " in line[:4]:
            fields[current_key].append(line.split(". ", 1)[1].strip())
        elif fields[current_key]:
            fields[current_key][-1] = f"{fields[current_key][-1]} {line}".strip()
        else:
            fields[current_key].append(line)

    return MarkdownRecord(path=path, fields=fields, raw=text)


def status_from_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.pass_task:
        return "已通过", args.pass_task or "QA 验收通过。"
    if args.fail:
        return "未通过", args.fail
    if args.block:
        return "阻塞", args.block
    return "阻塞", "未指定 --pass / --fail / --block，QA 结果保持阻塞。"


def bullet_lines(items: list[str], default: str = "无") -> str:
    values = items or [default]
    return "\n".join(f"- {item}" for item in values)


def numbered_lines(items: list[str], default: str = "无") -> str:
    values = items or [default]
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(values, 1))


def report_exists_for(task_id: str, suffix: str) -> Path:
    return REPORT_DIR / f"{task_id}-{suffix}.md"


def build_qa_report(
    *,
    task_id: str,
    task: MarkdownRecord,
    business: MarkdownRecord | None,
    status: str,
    reason: str,
    timezone: str,
) -> str:
    now = datetime.now().strftime(f"%Y-%m-%d %H:%M {timezone}")
    task_thread = task.first("执行线程", "未指定执行线程")
    qa_mode = task.first("QA验收方式", "正常验收")
    title = task.first("标题", "未命名任务")
    scope = task.first("简短描述", title)
    commit = (business.first("commit", "无") if business else "无") or "无"
    business_status = business.first("状态", "缺失") if business else "缺失"
    business_done = business.all("done") if business else []
    business_verify = business.all("verify") if business else []
    business_blockers = business.all("blockers") if business else ["缺少业务回报"]

    actual_results = [
        f"任务文件存在：{task.path.relative_to(ROOT)}",
        f"业务回报状态：{business_status}",
    ]
    if business:
        actual_results.append(f"业务回报存在：{business.path.relative_to(ROOT)}")
    else:
        actual_results.append(f"业务回报缺失：.workflow/reports/{task_id}-business.md")
    if business_done:
        actual_results.extend(business_done[:6])

    pass_items: list[str] = []
    fail_items: list[str] = []
    block_items: list[str] = []

    if business and business_status == "待验收":
        pass_items.append("业务回报已进入待验收状态")
    elif business:
        fail_items.append(f"业务回报状态不是待验收：{business_status}")
    else:
        block_items.append(f"缺少业务回报：.workflow/reports/{task_id}-business.md")

    if task.first("是否进行QA验收", "否") == "是":
        pass_items.append("任务声明需要 QA 验收")
    else:
        fail_items.append("任务未声明需要 QA 验收")

    if business_verify:
        pass_items.append("业务回报包含 verify 证据")
    elif business:
        fail_items.append("业务回报缺少 verify 证据")

    if status == "已通过":
        pass_items.append(reason)
        fail_items = fail_items or ["无"]
        block_items = ["无"]
    elif status == "未通过":
        fail_items.append(reason)
        block_items = ["无"]
    else:
        block_items.append(reason)

    suggestions = {
        "已通过": [
            "总控可以将该任务视为验收通过。",
            "如存在后续任务，可按前置条件派发下一任务。",
        ],
        "未通过": [
            "总控应将任务退回执行线程补证或修正。",
            "补齐后重新生成业务回报并再次 QA。",
        ],
        "阻塞": [
            "总控应先解除阻塞项。",
            "解除后重新运行 QA 验收。",
        ],
    }[status]

    defect_items = ["无"] if status == "已通过" else [reason]
    blocker_text = block_items if status == "阻塞" else ["无"]

    return f"""# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- {task_id}

状态：
- {status}

更新时间：
- {now}

验收线程：
- QA验收线程

验收对象：
- {task_thread} + {task_id}

验收方式：
- {qa_mode}

验收范围：
- {scope}

验收步骤：
1. 读取 `.workflow/tasks/{task_id}.md`。
2. 读取 `.workflow/reports/{task_id}-business.md`。
3. 检查任务状态、业务回报、verify 证据、done 结论和阻塞项。
4. 写入 `.workflow/reports/{task_id}-qa.md` 并刷新看板。

实际结果：
{bullet_lines(actual_results)}

验收结论：
- {status}
- 结论说明：
  - {reason}

通过项：
{numbered_lines(pass_items)}

不通过项：
{numbered_lines(fail_items)}

缺陷清单：
{numbered_lines(defect_items)}

阻塞项：
{bullet_lines(blocker_text)}

业务回报阻塞项：
{bullet_lines(business_blockers)}

建议总控下一步：
{numbered_lines(suggestions)}

提交信息：
- commit：{commit}
"""


def write_outputs(task_id: str, qa_text: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = report_exists_for(task_id, "qa")
    qa_path.write_text(qa_text, encoding="utf-8")
    QA_ACCEPTANCE_REPORT.write_text(
        "# QA Acceptance Report\n\n" + qa_text.strip() + "\n",
        encoding="utf-8",
    )
    return qa_path


def refresh_dashboard() -> int:
    if not BUILD_DASHBOARD.exists():
        print(f"warning: dashboard builder not found: {BUILD_DASHBOARD}", file=sys.stderr)
        return 0
    result = subprocess.run(
        [sys.executable, str(BUILD_DASHBOARD)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a workflow QA report for a task")
    parser.add_argument("task_id", help="Task ID, for example 0510T001 or 5-10-T001")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pass", dest="pass_task", nargs="?", const="QA 验收通过。", help="Mark QA as passed")
    group.add_argument("--fail", help="Mark QA as failed with the given reason")
    group.add_argument("--block", help="Mark QA as blocked with the given reason")
    parser.add_argument("--timezone", default="Asia/Shanghai", help="Timezone label written to the report")
    args = parser.parse_args()

    task_id = normalize_task_id(args.task_id)
    task_path = TASK_DIR / f"{task_id}.md"
    if not task_path.exists():
        raise SystemExit(f"Task file not found: {task_path}")

    task = parse_markdown_file(task_path)
    business_path = report_exists_for(task_id, "business")
    business = parse_markdown_file(business_path) if business_path.exists() else None

    status, reason = status_from_args(args)
    if status not in FINAL_STATUSES:
        raise SystemExit(f"Unsupported QA status: {status}")

    if status == "已通过" and business is None:
        status = "阻塞"
        reason = f"Cannot pass QA because business report is missing: {business_path.relative_to(ROOT)}"

    qa_text = build_qa_report(
        task_id=task_id,
        task=task,
        business=business,
        status=status,
        reason=reason,
        timezone=args.timezone,
    )
    qa_path = write_outputs(task_id, qa_text)
    dashboard_code = refresh_dashboard()

    print(f"Wrote {qa_path.relative_to(ROOT)}")
    print(f"Wrote {QA_ACCEPTANCE_REPORT.relative_to(ROOT)}")
    print(f"QA status: {status}")
    return dashboard_code


if __name__ == "__main__":
    raise SystemExit(main())
