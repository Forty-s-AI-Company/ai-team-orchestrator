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
    supervisor_pid = _positive_int(supervisor.get("MainPID"))
    watchdog_descendants = _descendants(watchdog_pid, processes) if watchdog_pid else []
    supervisor_descendants = _descendants(supervisor_pid, processes) if supervisor_pid else []
    descendants = watchdog_descendants or supervisor_descendants
    state = _read_json(supervisor_state_path)
    repair = _latest_repair_activity(project, checked_at)
    repair_report, historical_repair_report = _repair_reports(report_dir, state)

    overall = _overall_status(
        supervisor,
        watchdog,
        watchdog_descendants,
        repair,
    )
    controller = "watchdog" if watchdog_descendants else "supervisor"
    owner, activity = _current_activity(descendants, repair_report, controller=controller)
    if not descendants and supervisor.get("ActiveState") == "active":
        owner = "Supervisor（自動開發控制器）"
        activity = _supervisor_activity(state)
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
    lines.extend(_historical_repair_lines(historical_repair_report))

    deferred_tasks = state.get("deferredTasks")
    if isinstance(deferred_tasks, list) and deferred_tasks:
        lines.extend(["", "已暫緩任務（不阻塞其他工作）"])
        for item in deferred_tasks[-5:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('id') or '未知任務'}：{_shorten(str(item.get('reason') or '未提供原因'), 100)}"
            )

    release_reviews = state.get("releaseReviewTasks")
    if isinstance(release_reviews, list) and release_reviews:
        lines.extend(["", "待人工上線驗收（不阻塞測試站開發）"])
        for item in release_reviews[-5:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('id') or '未知任務'}：{_shorten(str(item.get('reason') or '等待人工驗收'), 100)}"
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
    watchdog_descendants: list[dict[str, int | str]],
    repair: dict[str, str] | None,
) -> str:
    if watchdog_descendants:
        return "自動修復中（AI Team 確實有在工作）"
    if supervisor.get("ActiveState") == "active":
        return "主流程自動開發中"
    if supervisor.get("ActiveState") == "activating":
        return "主流程正在自動重啟（短暫切換中，不是永久停止）"
    if _positive_int(watchdog.get("MainPID")):
        if repair and not repair["stale"]:
            return "自動修復中（正在切換或收尾）"
        return "疑似卡住（修復程序存在，但近期沒有工作活動）"
    return "已停止（目前沒有 AI Team 工作程序）"


def _current_activity(
    descendants: list[dict[str, int | str]],
    repair_report: dict,
    *,
    controller: str,
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
        activity = "執行自動修復流程" if controller == "watchdog" else "執行自動開發流程"
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
        owner = (
            "Watchdog（自動修復控制器）"
            if controller == "watchdog"
            else "Supervisor（自動開發控制器）"
        )
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


def _repair_reports(
    report_dir: Path | None,
    state: dict,
) -> tuple[dict, dict]:
    if report_dir is None:
        return {}, {}
    current = report_dir / "watchdog-ai-repair-current.json"
    paths: list[Path] = [current] if current.is_file() else []
    try:
        paths.extend(
            path
            for path in sorted(
                report_dir.glob("watchdog-ai-repair-*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if path != current
        )
    except OSError:
        pass

    current_report: dict = {}
    historical_report: dict = {}
    seen_reports: set[tuple[object, ...]] = set()
    for path in paths[:40]:
        report = _read_repair_report(path, report_dir)
        if not report:
            continue
        fingerprint = _repair_report_fingerprint(report)
        if fingerprint in seen_reports:
            continue
        seen_reports.add(fingerprint)
        if not current_report and _repair_report_matches_state(report, state):
            current_report = report
        elif not historical_report:
            historical_report = report
        if current_report and historical_report:
            break
    return current_report, historical_report


def _repair_report_fingerprint(report: dict) -> tuple[object, ...]:
    evidence = report.get("supervisorEvidence")
    task = evidence.get("currentTask") if isinstance(evidence, dict) else None
    return (
        report.get("schema"),
        report.get("startedAt"),
        report.get("completedAt"),
        report.get("status"),
        task.get("taskSha") if isinstance(task, dict) else None,
        evidence.get("revision") if isinstance(evidence, dict) else None,
    )


def _read_repair_report(path: Path, report_dir: Path) -> dict:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            return {}
        resolved = path.resolve(strict=True)
        resolved.relative_to(report_dir.resolve(strict=True))
    except (OSError, ValueError):
        return {}
    return _read_json(resolved)


def _repair_report_matches_state(report: dict, state: dict) -> bool:
    current_task = state.get("currentTask")
    evidence = report.get("supervisorEvidence")
    report_task = evidence.get("currentTask") if isinstance(evidence, dict) else None
    if not isinstance(current_task, dict) or not isinstance(report_task, dict):
        return False
    current_sha = current_task.get("taskSha")
    report_sha = report_task.get("taskSha")
    if not isinstance(current_sha, str) or not current_sha or report_sha != current_sha:
        return False

    # While Watchdog owns a repair, Supervisor is stopped and its revision is
    # the run-instance boundary.  Once Supervisor writes a newer revision, the
    # old report is history even if a task SHA is retried later.
    current_revision = state.get("revision")
    report_revision = evidence.get("revision")
    if isinstance(current_revision, int) and isinstance(report_revision, int):
        if report_revision != current_revision:
            return False

    current_external = state.get("externalQa")
    report_external = evidence.get("externalQa")
    current_external_revision = (
        current_external.get("revision") if isinstance(current_external, dict) else None
    )
    report_external_revision = (
        report_external.get("revision") if isinstance(report_external, dict) else None
    )
    if (
        isinstance(current_external_revision, str)
        and current_external_revision
        and report_external_revision != current_external_revision
    ):
        return False
    return True


def _historical_repair_lines(report: dict) -> list[str]:
    if not report:
        return []
    evidence = report.get("supervisorEvidence")
    task = evidence.get("currentTask") if isinstance(evidence, dict) else None
    task_id = task.get("id") if isinstance(task, dict) else None
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    task_label = str(task_id or "未知任務")
    if isinstance(task_sha, str) and task_sha:
        task_label = f"{task_label}（{task_sha[:12]}）"
    status = str(report.get("status") or "unknown")
    status_label = {
        "running": "舊執行批次未留下終態",
        "passed": "已修好並通過所有 QA",
        "deferred": "達修復上限，已記錄並暫緩",
        "failed": "修復流程發生錯誤",
    }.get(status, status)
    lines = [
        "",
        "歷史修復紀錄（非目前任務／非目前執行批次）",
        f"- 歷史任務：{task_label}",
        f"- 歷史結果：{status_label}",
    ]
    reason = report.get("deferReason") or report.get("error")
    if reason:
        lines.append(f"- 歷史原因：{_shorten(str(reason), 140)}")
    return lines


def _repair_history_lines(report: dict) -> list[str]:
    cycles = report.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return []
    limit = report.get("cycleLimit") or 5
    lines = ["", f"自動修復歷程（最多 {limit} 輪）"]
    repository = report.get("repository")
    if repository:
        target = "AI Team 編排器" if repository == "orchestrator" else "CelebrateDeal 專案"
        lines.append(f"- 修復對象：{target}")
    diagnosis = report.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get("summary"):
        lines.append(f"- 修復內容：{_shorten(str(diagnosis['summary']), 140)}")
    contract = report.get("acceptanceContract")
    if isinstance(contract, dict) and contract.get("sha256"):
        criteria = contract.get("acceptanceCriteria")
        count = len(criteria) if isinstance(criteria, list) else 0
        lines.append(
            f"- 收斂規則：驗收契約已凍結（{count} 項）；範圍外發現不得阻擋"
        )
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
        "deferred": "達修復輪次上限，已記錄並跳過，不阻塞其他工作",
        "failed": "修復流程本身發生錯誤",
    }
    lines.append(f"- 最終結果：{status_labels.get(status, status)}")
    reason = report.get("deferReason") or report.get("error")
    if reason:
        lines.append(f"- 原因：{_shorten(str(reason), 160)}")
    follow_ups = report.get("followUpFindings")
    if isinstance(follow_ups, list) and follow_ups:
        lines.extend(["", "範圍外發現（已另行記錄，不阻擋目前修復）"])
        for finding in follow_ups[-5:]:
            if not isinstance(finding, dict):
                continue
            detail = finding.get("evidence") or finding.get("action") or finding.get("id")
            lines.append(f"- {_shorten(str(detail), 140)}")
    return lines


def _cycle_detail(item: dict) -> str:
    agy = item.get("agyQa") if isinstance(item.get("agyQa"), dict) else {}
    sol = item.get("solReview") if isinstance(item.get("solReview"), dict) else {}
    parts: list[str] = []
    if agy:
        agy_blocking = item.get("agyBlockingFindings")
        if isinstance(agy_blocking, list) and not agy_blocking:
            parts.append("AGY 契約 QA 通過")
        else:
            parts.append(f"AGY QA {_qa_label(agy)}")
    if sol:
        blocking = item.get("blockingFindings")
        if isinstance(blocking, list) and not blocking:
            parts.append("Sol 契約複檢通過")
        else:
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


def _supervisor_activity(state: dict) -> str:
    next_action = str(state.get("nextAction") or "")
    status = str(state.get("status") or "")
    return {
        "bounded-delivery": "執行目前任務的分工、修改與測試",
        "run-autonomous-contract": "準備執行 PM 新建立的任務",
        "watch-contract-directory": "等待或掃描下一個開發任務",
        "next-contract": "切換到下一個優先任務",
        "provider-probe": "檢查 AI 模型供應商是否恢復",
        "retry-selected-cloud-model": "重試目前選定的 AI 模型",
    }.get(
        next_action,
        "PM 掃描專案並規劃下一項工作" if status in {"planning-next-task", "completed-development"}
        else "主流程執行中，等待下一個階段",
    )


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
    if active == "activating" and sub == "auto-restart":
        return "自動重啟等待中（短暫狀態）"
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
