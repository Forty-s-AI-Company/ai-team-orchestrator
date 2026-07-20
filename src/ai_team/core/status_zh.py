from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable


Runner = Callable[[list[str]], str]


def render_chinese_status(
    project: Path,
    *,
    supervisor_state_path: Path,
    supervisor_service: str,
    watchdog_service: str,
    report_dir: Path | None = None,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> str:
    """Render one truthful, human-readable snapshot of the AI Team runtime."""

    execute = runner or _run
    checked_at = now or datetime.now().astimezone()
    supervisor = _service_status(supervisor_service, execute)
    watchdog = _service_status(watchdog_service, execute)
    processes = _process_table(execute)
    watchdog_pid = _positive_int(watchdog.get("MainPID"))
    descendants = _descendants(watchdog_pid, processes) if watchdog_pid else []
    state = _read_json(supervisor_state_path)
    repair = _latest_repair_activity(project, checked_at)
    repair_report = _repair_report(report_dir)

    overall = _overall_status(supervisor, watchdog, descendants, repair)
    owner, activity = _current_activity(descendants, repair_report)
    task_title, task_instruction = _task_description(state)
    latest_commit = _latest_commit(project, execute)

    lines = [
        "AI TEAM 中文即時狀態",
        "=" * 46,
        f"查詢時間：{checked_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"整體狀態：{overall}",
        f"目前負責：{owner}",
        f"正在進行：{activity}",
    ]

    if descendants:
        longest = max(item["elapsed"] for item in descendants)
        lines.append(f"本階段已執行：{_human_duration(longest)}")
    if repair:
        lines.extend(
            [
                f"最近檔案活動：{repair['when']}（{repair['ago']}）",
                f"修復中的檔案：{repair['files']}",
            ]
        )

    lines.extend(
        [
            "",
            "服務狀況",
            f"- 主流程：{_service_label(supervisor, paused_by_watchdog=bool(watchdog_pid))}",
            f"- 自動修復：{_service_label(watchdog)}",
            "",
            "目前任務",
            f"- 標題：{task_title}",
            f"- 內容：{task_instruction}",
            "",
            f"上一個 Git 成果：{latest_commit}",
        ]
    )
    lines.extend(_repair_history_lines(repair_report))

    deferred_tasks = state.get("deferredTasks")
    if isinstance(deferred_tasks, list) and deferred_tasks:
        lines.extend(["", "已暫緩任務（不阻塞其他工作）"])
        for item in deferred_tasks[-5:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('id') or '未知任務'}：{_shorten(str(item.get('reason') or '未提供原因'), 100)}"
            )

    if state.get("status") == "stopped" and watchdog_pid:
        lines.extend(
            [
                "",
                "說明：狀態檔的 stopped 只代表主流程暫停；",
                "      Watchdog 正在接管修復，不代表整個 AI Team 停止。",
            ]
        )
    return "\n".join(lines)


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=8,
        check=False,
    )
    return result.stdout


def _service_status(service: str, runner: Runner) -> dict[str, str]:
    output = runner(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
        ]
    )
    return dict(
        line.split("=", 1)
        for line in output.splitlines()
        if "=" in line
    )


def _process_table(runner: Runner) -> list[dict[str, int | str]]:
    output = runner(["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,stat=,args="])
    rows: list[dict[str, int | str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "elapsed": int(parts[2]),
                    "cpu": parts[3],
                    "stat": parts[4],
                    "command": parts[5],
                }
            )
        except ValueError:
            continue
    return rows


def _descendants(root_pid: int, rows: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    found: list[dict[str, int | str]] = []
    parents = {root_pid}
    while True:
        children = [row for row in rows if row["ppid"] in parents and row not in found]
        if not children:
            return found
        found.extend(children)
        parents = {int(row["pid"]) for row in children}


def _overall_status(
    supervisor: dict[str, str],
    watchdog: dict[str, str],
    descendants: list[dict[str, int | str]],
    repair: dict[str, str] | None,
) -> str:
    if descendants:
        return "自動修復中（AI Team 確實有在工作）"
    if supervisor.get("ActiveState") == "active":
        return "主流程自動開發中"
    if _positive_int(watchdog.get("MainPID")):
        if repair and not repair["stale"]:
            return "自動修復中（正在切換或收尾）"
        return "疑似卡住（修復程序存在，但近期沒有工作活動）"
    return "已停止（目前沒有 AI Team 工作程序）"


def _current_activity(
    descendants: list[dict[str, int | str]],
    repair_report: dict,
) -> tuple[str, str]:
    commands = "\n".join(str(row["command"]) for row in descendants).lower()
    if "playwright" in commands or "e2e:" in commands:
        activity = "Playwright 瀏覽器驗收測試"
    elif "next build" in commands or "npm run build" in commands:
        activity = "正式版本建置檢查"
    elif "vitest" in commands or "npm run test" in commands:
        activity = "單元與整合測試"
    elif "tsc --noemit" in commands or "typecheck" in commands:
        activity = "TypeScript 型別檢查"
    elif "eslint" in commands or "npm run lint" in commands:
        activity = "ESLint 程式碼規範檢查"
    elif "antigravity" in commands or "/agy" in commands or "gemini 3.1" in commands:
        activity = "AGY 獨立 QA 驗證修正結果"
    elif "gpt-5.6-sol" in commands:
        activity = "審查修正結果並決定通過或退回"
    elif "gpt-5.6-terra" in commands:
        activity = "依照審查意見修改程式"
    elif descendants:
        activity = "執行自動修復流程"
    else:
        activity = _phase_activity(repair_report)

    effort = "XHigh" if "xhigh" in commands else "High"
    if "gpt-5.6-terra" in commands:
        owner = f"Terra {effort}（工程修復）"
    elif "antigravity" in commands or "/agy" in commands or "gemini 3.1" in commands:
        owner = "AGY（獨立 QA）"
    elif "gpt-5.6-sol" in commands:
        owner = f"Sol {effort}（診斷／QA 審查）"
    elif descendants:
        owner = "Watchdog（自動修復控制器）"
    elif repair_report.get("status") == "running":
        owner = _phase_owner(repair_report)
    else:
        owner = "無"
    return owner, activity


def _latest_repair_activity(project: Path, now: datetime) -> dict[str, str] | None:
    candidates = list(project.parent.glob(f"{project.name}-watchdog-repair-*"))
    files: list[Path] = []
    for worktree in candidates:
        for directory_name in ("src", "tests", "scripts"):
            directory = worktree / directory_name
            if directory.is_dir():
                files.extend(item for item in directory.rglob("*") if item.is_file())
        files.extend(
            item for item in worktree.glob("playwright.config.*") if item.is_file()
        )
    if not files:
        return None
    latest = max(files, key=lambda item: item.stat().st_mtime)
    modified = datetime.fromtimestamp(latest.stat().st_mtime, tz=now.tzinfo)
    age = max(0, int((now - modified).total_seconds()))
    newest = sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[:4]
    return {
        "when": modified.strftime("%Y-%m-%d %H:%M:%S"),
        "ago": f"{_human_duration(age)}前",
        "files": "、".join(item.name for item in newest),
        "stale": "yes" if age > 600 else "",
    }


def _repair_report(report_dir: Path | None) -> dict:
    if report_dir is None:
        return {}
    current = report_dir / "watchdog-ai-repair-current.json"
    if current.is_file():
        return _read_json(current)
    try:
        candidates = sorted(
            report_dir.glob("watchdog-ai-repair-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return {}
    return _read_json(candidates[0]) if candidates else {}


def _repair_history_lines(report: dict) -> list[str]:
    cycles = report.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return []
    limit = report.get("cycleLimit") or 5
    lines = ["", f"自動修復歷程（最多 {limit} 輪）"]
    for item in cycles[-5:]:
        if not isinstance(item, dict):
            continue
        number = item.get("cycle") or "?"
        effort = "XHigh" if item.get("reasoningEffort") == "xhigh" else "High"
        outcome = _outcome_label(str(item.get("outcome") or "進行中"))
        detail = _cycle_detail(item)
        suffix = f"：{detail}" if detail else ""
        lines.append(f"- 第 {number} 輪｜Sol/Terra {effort}｜{outcome}{suffix}")
    status = str(report.get("status") or "running")
    status_labels = {
        "running": "仍在修復中",
        "passed": "已修好並通過所有 QA",
        "deferred": "五輪未修好，已記錄並跳過，不阻塞其他工作",
        "failed": "修復流程本身發生錯誤",
    }
    lines.append(f"- 最終結果：{status_labels.get(status, status)}")
    reason = report.get("deferReason") or report.get("error")
    if reason:
        lines.append(f"- 原因：{_shorten(str(reason), 160)}")
    return lines


def _cycle_detail(item: dict) -> str:
    agy = item.get("agyQa") if isinstance(item.get("agyQa"), dict) else {}
    sol = item.get("solReview") if isinstance(item.get("solReview"), dict) else {}
    parts: list[str] = []
    if agy:
        parts.append(f"AGY QA {_qa_label(agy)}")
    if sol:
        parts.append(f"Sol 複檢 {_qa_label(sol)}")
    failure = item.get("failureSummary")
    if failure:
        parts.append(_shorten(str(failure), 120))
    return "；".join(parts)


def _qa_label(value: dict) -> str:
    status = value.get("status")
    return "通過" if status == "passed" else "未通過"


def _outcome_label(value: str) -> str:
    return {
        "diagnosed": "Sol 已完成診斷",
        "passed": "修復通過",
        "unrepairable": "Sol 判定暫時無法自修",
        "terra-produced-no-diff": "Terra 沒有產生修正",
        "deterministic-qa-failed": "基礎測試未通過",
        "review-rejected": "QA／複檢退回",
        "cycle-error": "本輪執行失敗",
    }.get(value, value)


def _phase_activity(report: dict) -> str:
    return {
        "initializing": "準備自動修復證據",
        "sol-diagnosis": "Sol 正在判斷根因與修正方向",
        "terra-repair": "Terra 正在依診斷修改程式",
        "deterministic-qa": "執行 lint、型別、單元測試與建置",
        "agy-qa": "AGY 正在做獨立 QA",
        "sol-review": "Sol 正在複檢 AGY 與測試結果",
        "commit-and-push": "建立 Git checkpoint 並推送",
        "deferred": "已記錄失敗並跳過目前任務",
    }.get(str(report.get("activePhase") or ""), "目前沒有執行中的工作")


def _phase_owner(report: dict) -> str:
    phase = str(report.get("activePhase") or "")
    effort = "XHigh" if _positive_int(str(report.get("activeCycle") or 0)) > 1 else "High"
    if phase == "terra-repair":
        return f"Terra {effort}（工程修復）"
    if phase == "agy-qa":
        return "AGY（獨立 QA）"
    if phase in {"sol-diagnosis", "sol-review"}:
        return f"Sol {effort}（診斷／QA 審查）"
    return "Watchdog（自動修復控制器）"


def _task_description(state: dict) -> tuple[str, str]:
    task = state.get("currentTask") if isinstance(state.get("currentTask"), dict) else {}
    contract_path = Path(str(task.get("contractPath") or ""))
    contract = _read_json(contract_path) if contract_path.is_file() else {}
    title = str(contract.get("title") or task.get("id") or "等待 PM 掃描下一個任務")
    instruction = str(contract.get("instruction") or "等待任務分派")
    return title, _shorten(instruction, 150)


def _latest_commit(project: Path, runner: Runner) -> str:
    output = runner(
        [
            "git",
            "-C",
            str(project),
            "log",
            "-1",
            "--pretty=%h｜%ad｜%s",
            "--date=format-local:%Y-%m-%d %H:%M:%S",
        ]
    ).strip()
    return output or "尚無可讀取的 Git 成果"


def _service_label(service: dict[str, str], *, paused_by_watchdog: bool = False) -> str:
    active = service.get("ActiveState")
    sub = service.get("SubState")
    pid = _positive_int(service.get("MainPID"))
    if active == "active" and pid:
        return f"執行中（PID {pid}）"
    if active == "activating" and pid:
        return f"執行中（PID {pid}）"
    if paused_by_watchdog and active == "inactive":
        return "暫停，由自動修復接管"
    return f"未執行（{active or '未知'}/{sub or '未知'}）"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: str | None) -> int:
    try:
        number = int(value or "0")
    except ValueError:
        return 0
    return number if number > 0 else 0


def _human_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {remainder} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小時 {minutes} 分"


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
