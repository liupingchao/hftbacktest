#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "tasks"
REPORT_DIR = ROOT / "reports"
DASHBOARD_PATH = ROOT / "dashboard.html"
SUGGESTIONS_PATH = ROOT / "dispatch_suggestions.md"

STATUS_ORDER = {
    "阻塞": 0,
    "未通过": 1,
    "待验收": 2,
    "执行中": 3,
    "待执行": 4,
    "已通过": 5,
    "作废": 6,
}

FINAL_QA_STATUSES = {"已通过", "未通过", "阻塞"}
UNKNOWN = "无明确记录"

DECISION_TOKENS = [
    "diagnostic_only_no_promotion",
    "collect_more_current_format_data",
    "proceed_to_stage6j_narrow_rule",
    "promote_to_live_micro_test",
    "blocked",
]

SUMMARY_KEYWORDS = [
    "Stage 6J",
    "decision",
    "样本",
    "候选",
    "hard failures",
    "cancel-requested",
    "cancel-fill",
    "adverse-selection",
    "inventory-reducing",
    "live micro test",
    "default-off",
    "不改策略代码",
]


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


@dataclass
class TaskView:
    task: MarkdownRecord
    reports: list[MarkdownRecord] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return self.task.first("任务ID", self.task.path.stem)

    @property
    def latest_report(self) -> MarkdownRecord | None:
        non_qa = [r for r in self.reports if r.first("执行线程") != "QA验收线程"]
        if not non_qa:
            return None
        return sorted(non_qa, key=lambda r: r.path.name)[-1]

    @property
    def latest_qa(self) -> MarkdownRecord | None:
        qa_reports = [r for r in self.reports if r.first("执行线程") == "QA验收线程"]
        if not qa_reports:
            return None
        return sorted(qa_reports, key=lambda r: r.path.name)[-1]

    @property
    def effective_status(self) -> str:
        qa = self.latest_qa
        if qa and qa.first("状态") in FINAL_QA_STATUSES:
            return qa.first("状态")
        report = self.latest_report
        if report and report.first("状态"):
            return report.first("状态")
        return self.task.first("状态", "待执行")

    @property
    def qa_required(self) -> bool:
        return self.task.first("是否进行QA验收") == "是"

    @property
    def qa_requested(self) -> bool:
        report = self.latest_report
        return bool(report and report.first("是否进行QA验收") == "是")

    @property
    def qa_mode(self) -> str:
        return self.task.first("QA验收方式", "无")

    @property
    def decision_summary(self) -> list[str]:
        return extract_decision_summary(self)

    @property
    def metric_chips(self) -> list[str]:
        return extract_metric_chips(self)

    @property
    def business_summary(self) -> list[str]:
        return extract_findings(self.latest_report, self.task)

    @property
    def qa_summary(self) -> list[str]:
        return extract_qa_summary(self.latest_qa)

    @property
    def next_steps(self) -> list[str]:
        return extract_next_steps(self)


def parse_markdown_file(path: Path) -> MarkdownRecord:
    text = path.read_text(encoding="utf-8")
    fields: dict[str, list[str]] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"```md", "```"} or line.startswith("#"):
            continue

        is_list_item = line.startswith("- ") or re.match(r"^\d+\.\s+", line)
        if not is_list_item and (line.endswith("：") or line.endswith(":")):
            current_key = line.rstrip("：:")
            fields.setdefault(current_key, [])
            continue

        if current_key is None:
            continue

        if line.startswith("- "):
            fields[current_key].append(line[2:].strip())
        elif re.match(r"^\d+\.\s+", line):
            fields[current_key].append(line.split(". ", 1)[1].strip())
        elif fields[current_key]:
            fields[current_key][-1] = f"{fields[current_key][-1]} {line}".strip()
        else:
            fields[current_key].append(line)

    return MarkdownRecord(path=path, fields=fields, raw=text)


def load_records(directory: Path) -> list[MarkdownRecord]:
    if not directory.exists():
        return []
    return [parse_markdown_file(path) for path in sorted(directory.glob("*.md"))]


def group_tasks(tasks: Iterable[MarkdownRecord], reports: Iterable[MarkdownRecord]) -> list[TaskView]:
    reports_by_task: dict[str, list[MarkdownRecord]] = {}
    for report in reports:
        reports_by_task.setdefault(report.first("任务ID", report.path.stem), []).append(report)

    views = []
    for task in tasks:
        task_id = task.first("任务ID", task.path.stem)
        views.append(TaskView(task=task, reports=reports_by_task.get(task_id, [])))

    return sorted(views, key=lambda view: (STATUS_ORDER.get(view.effective_status, 99), view.task_id))


def badge_class(status: str) -> str:
    return {
        "阻塞": "blocked",
        "未通过": "failed",
        "待验收": "review",
        "执行中": "running",
        "待执行": "todo",
        "已通过": "passed",
        "作废": "void",
    }.get(status, "unknown")


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def joined_record_text(*records: MarkdownRecord | None) -> str:
    return "\n".join(record.raw for record in records if record is not None)


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    if match.groups():
        return next((group for group in match.groups() if group is not None), None)
    return match.group(0)


def unique_items(items: Iterable[str], limit: int = 6) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = compact_text(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
        if len(result) >= limit:
            break
    return result


def is_experiment_task(view: TaskView) -> bool:
    task_text = "\n".join(
        [
            view.task.first("标题"),
            view.task.first("简短描述"),
            "\n".join(view.task.all("files")),
        ]
    )
    experiment_markers = [
        "examples/binance_tick_mm",
        "local_live_analysis",
        "Stage 6J",
        "stage6j",
        "adverse-selection",
        "maker",
        "live/backtest",
    ]
    workflow_only_markers = [
        ".workflow/build_dashboard.py",
        "dashboard",
        "看板",
    ]
    if any(marker in task_text for marker in experiment_markers):
        return True
    if any(marker in task_text for marker in workflow_only_markers):
        return False
    return False


def extract_metric_chips(view: TaskView) -> list[str]:
    fact_text = joined_record_text(view.latest_report, view.latest_qa)
    task_title = view.task.first("标题")
    task_summary = view.task.first("简短描述")
    task_rule_text = "\n".join(view.task.all("规则更新提醒"))
    chips: list[str] = []
    experiment_task = is_experiment_task(view)

    if experiment_task:
        run_id = first_match(r"`?(5-\d{1,2}-[A-Za-z0-9_.-]+)`?", fact_text)
        if run_id:
            chips.append(f"run: {run_id.strip('`')}")

        decision = next((token for token in DECISION_TOKENS if token in fact_text), None)
        if decision:
            chips.append(f"decision: {decision}")

        samples = (
            first_match(r"样本数[：:]\s*`?(\d+)`?", fact_text)
            or first_match(r"samples[=：: ]+`?(\d+)`?", fact_text)
        )
        if samples:
            chips.append(f"samples: {samples}")

        candidates = (
            first_match(r"候选数[：:]\s*`?(\d+)`?", fact_text)
            or first_match(r"candidates[=：: ]+`?(\d+)`?", fact_text)
        )
        if candidates:
            chips.append(f"candidates: {candidates}")

        hard_failures = (
            first_match(r"hard failures[：:]\s*`?(\d+)`?", fact_text)
            or first_match(r"hard_failures[=：: ]+`?(\d+)`?", fact_text)
        )
        if hard_failures:
            chips.append(f"hard failures: {hard_failures}")

        cancel_requested = (
            first_match(r"fill-after-cancel-request\s*`?(\d+)`?", fact_text)
            or first_match(r"cancel-requested fills?\s*[:：]?\s*`?(\d+)`?", fact_text)
        )
        fills = first_match(r"(?:^|\n|\s)fills\s*`?(\d+)`?", fact_text)
        if cancel_requested and fills:
            chips.append(f"cancel-fill: {cancel_requested} / {fills}")

        rate = first_match(r"notional rate\s*`?([0-9.]+)`?", fact_text)
        if rate:
            chips.append(f"cancel-fill rate: {rate[:8]}")

        if "passed=true" in fact_text or "acceptance passed" in fact_text:
            chips.append("acceptance: passed")

        if "不建议进入 live micro test" in fact_text or "不自动进入 live micro test" in fact_text:
            chips.append("live micro test: 否")
        elif "live micro test" in fact_text:
            chips.append("live micro test: 待判断")

    if (
        ("adverse-selection" in task_title or "adverse-selection" in task_summary)
        and ("default-off" in task_rule_text or "默认关闭" in task_rule_text or "默认关闭" in task_summary)
    ):
        chips.append("rule default: off")

    return unique_items(chips, limit=10)


def extract_decision_summary(view: TaskView) -> list[str]:
    report = view.latest_report
    qa = view.latest_qa
    fact_text = joined_record_text(report, qa)
    task_title = view.task.first("标题")
    task_summary = view.task.first("简短描述")
    lines: list[str] = []
    experiment_task = is_experiment_task(view)

    if report:
        lines.append(f"业务状态：{report.first('状态', UNKNOWN)}")
    else:
        lines.append("业务状态：无业务回报")

    if qa:
        lines.append(f"QA：{qa.first('状态', UNKNOWN)}")
    elif view.qa_required:
        lines.append("QA：待 QA 或未生成")
    else:
        lines.append("QA：不需要")

    if experiment_task:
        decision = next((token for token in DECISION_TOKENS if token in fact_text), None)
        if decision:
            lines.append(f"关键决策：{decision}")

        if "不建议进入 live micro test" in fact_text or "不自动进入 live micro test" in fact_text:
            lines.append("live micro test：不允许 / 不建议")
        elif "live micro test" in fact_text:
            lines.append("live micro test：有提及但需人工判断")

    if "不改策略代码" in task_summary or "不修改策略代码" in task_summary or "不改策略代码" in "\n".join(view.task.all("MUST NOT")):
        lines.append("代码范围：不改策略代码")

    prerequisites = [
        item for item in view.task.all("前置任务")
        if item and item not in {"无", "none", "None", "NONE"}
    ]
    if prerequisites:
        lines.append(f"执行约束：需等待 {', '.join(prerequisites)}")

    return unique_items(lines, limit=6)


def extract_findings(report: MarkdownRecord | None, task: MarkdownRecord) -> list[str]:
    if report is None:
        candidate_lines = task.all("简短描述") + task.all("done")
        return unique_items(
            ["无业务回报；当前显示任务目标和验收要求。", *candidate_lines],
            limit=5,
        ) or [UNKNOWN]

    source = report
    candidate_lines = source.all("done") + source.all("实际结果") + source.all("简短描述")

    done_items = source.all("done")
    for index, item in enumerate(done_items):
        if item.strip("`") in {"结论：", "结论:"}:
            conclusion_items = [
                value
                for value in done_items[index + 1 :]
                if value and not value.endswith(("：", ":"))
            ]
            if conclusion_items:
                return unique_items(conclusion_items, limit=5)

    selected = [
        line
        for line in candidate_lines
        if any(keyword.lower() in line.lower() for keyword in SUMMARY_KEYWORDS)
    ]
    if not selected:
        selected = candidate_lines
    return unique_items(selected, limit=5) or [UNKNOWN]


def extract_qa_summary(qa: MarkdownRecord | None) -> list[str]:
    if qa is None:
        return ["无 QA 结论"]
    items = []
    items.extend(qa.all("验收结论"))
    failures = [item for item in qa.all("不通过项") if item != "无"]
    blockers = [item for item in qa.all("阻塞项") if item != "无"]
    if failures:
        items.append("不通过项：" + "；".join(failures[:2]))
    if blockers:
        items.append("阻塞项：" + "；".join(blockers[:2]))
    return unique_items(items, limit=5) or [UNKNOWN]


def extract_next_steps(view: TaskView) -> list[str]:
    qa = view.latest_qa
    report = view.latest_report
    items: list[str] = []
    if qa:
        items.extend(qa.all("建议总控下一步"))
    if report:
        done_items = report.all("done")
        items.extend(
            item
            for item in done_items
            if "下一步" in item or "建议" in item or "live micro test" in item
        )
    if not items:
        status = view.effective_status
        if status == "待执行":
            items.append(f"可派发给 {view.task.first('执行线程', '对应线程')}")
        elif status == "待验收":
            items.append("应派发 QA 验收线程")
        elif status == "已通过":
            items.append("可由总控选择后续任务")
    return unique_items(items, limit=4) or [UNKNOWN]


def list_html(items: list[str]) -> str:
    if not items:
        return "<span class=\"muted\">无</span>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def chips_html(items: list[str]) -> str:
    if not items:
        return f"<span class=\"muted\">{UNKNOWN}</span>"
    return "<div class=\"chips\">" + "".join(f"<span>{escape(item)}</span>" for item in items) + "</div>"


def render_dashboard(task_views: list[TaskView]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts: dict[str, int] = {}
    for view in task_views:
        counts[view.effective_status] = counts.get(view.effective_status, 0) + 1

    metric_cards = "".join(
        f"<div class=\"metric\"><strong>{counts.get(status, 0)}</strong><span>{status}</span></div>"
        for status in ["阻塞", "未通过", "待验收", "执行中", "待执行", "已通过"]
    )

    cards = []
    for view in task_views:
        task = view.task
        latest_report = view.latest_report
        latest_qa = view.latest_qa
        status = view.effective_status
        qa_line = "需要 QA" if view.qa_required else "不需要 QA"
        if latest_qa:
            qa_line = f"QA：{latest_qa.first('状态')} / {view.qa_mode}"
        elif view.qa_requested:
            qa_line = f"待 QA / {view.qa_mode}"

        cards.append(
            f"""
            <article class="task-card">
              <div class="task-top">
                <div>
                  <div class="task-id">{escape(view.task_id)}</div>
                  <h2>{escape(task.first("标题", "未命名任务"))}</h2>
                </div>
                <span class="badge {badge_class(status)}">{escape(status)}</span>
              </div>
              <p class="summary">{escape(task.first("简短描述", ""))}</p>
              <div class="grid">
                <div><span class="label">执行线程</span><strong>{escape(task.first("执行线程", "未指定"))}</strong></div>
                <div><span class="label">QA</span><strong>{escape(qa_line)}</strong></div>
                <div><span class="label">提交代码</span><strong>{escape(task.first("是否需要提交代码", "未指定"))}</strong></div>
                <div><span class="label">前置任务</span><strong>{escape(task.first("前置任务", "无"))}</strong></div>
              </div>
              <section>
                <h3>决策摘要</h3>
                {list_html(view.decision_summary)}
              </section>
              <section>
                <h3>关键指标</h3>
                {chips_html(view.metric_chips)}
              </section>
              <section>
                <h3>范围</h3>
                {list_html(task.all("files"))}
              </section>
              <section>
                <h3>业务结果</h3>
                {list_html(view.business_summary)}
              </section>
              <section>
                <h3>QA 结论</h3>
                {list_html(view.qa_summary)}
              </section>
              <section>
                <h3>下一步</h3>
                {list_html(view.next_steps)}
              </section>
            </article>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workflow Decision Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #1d2430;
      --muted: #697586;
      --line: #d9dee7;
      --blue: #1f6feb;
      --green: #1f7a4d;
      --red: #b42318;
      --amber: #a15c07;
      --gray: #667085;
      --chip-bg: #eef4ff;
      --chip-text: #1849a9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 6px; font-size: 28px; }}
    .muted, .label {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 12px;
      padding: 20px 32px 0;
    }}
    .metric {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{ display: block; font-size: 24px; }}
    .metric span {{ color: var(--muted); font-size: 14px; }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      padding: 20px 32px 32px;
    }}
    .task-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .task-card section {{
      border-top: 1px solid var(--line);
      margin-top: 14px;
      padding-top: 12px;
    }}
    .task-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .task-id {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    h2 {{ font-size: 19px; line-height: 1.3; margin-bottom: 10px; }}
    h3 {{ font-size: 14px; margin: 16px 0 6px; }}
    .summary {{ color: #344054; line-height: 1.55; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .grid > div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-width: 0;
    }}
    .label {{
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    strong {{ overflow-wrap: anywhere; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 3px 0; }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chips span {{
      display: inline-flex;
      max-width: 100%;
      border: 1px solid #c7d7fe;
      border-radius: 999px;
      background: var(--chip-bg);
      color: var(--chip-text);
      padding: 4px 8px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      color: white;
      font-size: 13px;
      white-space: nowrap;
    }}
    .blocked, .failed {{ background: var(--red); }}
    .review {{ background: var(--amber); }}
    .running {{ background: var(--blue); }}
    .todo, .void, .unknown {{ background: var(--gray); }}
    .passed {{ background: var(--green); }}
    @media (max-width: 700px) {{
      header, .metrics, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Workflow Decision Dashboard</h1>
    <p class="muted">Generated at {escape(generated_at)} from Markdown workflow files. Metrics are parsed from task, business, and QA markdown; missing values are shown as unknown.</p>
  </header>
  <section class="metrics">{metric_cards}</section>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_suggestions(task_views: list[TaskView]) -> str:
    lines = [
        "# Dispatch Suggestions",
        "",
        "Generated from `.workflow/tasks` and `.workflow/reports`.",
        "",
    ]

    for view in task_views:
        status = view.effective_status
        if status == "已通过":
            suggestion = "已通过。总控可以选择派发后续任务。"
        elif status == "待验收" and view.qa_required and view.qa_requested and not view.latest_qa:
            suggestion = "应派发 QA 验收线程执行正常验收。"
        elif status == "执行中":
            suggestion = "等待执行线程继续回报；暂不派 QA。"
        elif status == "待执行":
            suggestion = f"可以派发给 {view.task.first('执行线程', '对应线程')}。"
        elif status in {"未通过", "阻塞"}:
            suggestion = "总控应优先处理失败或阻塞项。"
        else:
            suggestion = "无需动作或等待总控判断。"

        lines.extend(
            [
                f"## {view.task_id} {view.task.first('标题', '未命名任务')}",
                "",
                f"- 当前状态：{status}",
                f"- 执行线程：{view.task.first('执行线程', '未指定')}",
                f"- QA模式：{view.qa_mode}",
                f"- 建议：{suggestion}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    tasks = load_records(TASK_DIR)
    reports = load_records(REPORT_DIR)
    task_views = group_tasks(tasks, reports)

    DASHBOARD_PATH.write_text(render_dashboard(task_views), encoding="utf-8")
    SUGGESTIONS_PATH.write_text(build_suggestions(task_views), encoding="utf-8")

    print(f"Loaded {len(tasks)} tasks and {len(reports)} reports.")
    print(f"Wrote {DASHBOARD_PATH.relative_to(ROOT.parent)}")
    print(f"Wrote {SUGGESTIONS_PATH.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
