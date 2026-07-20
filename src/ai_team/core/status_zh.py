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

    overall = _overall_status(supervisor, watchdog, descendants, repair)
    owner, activity = _current_activity(descendants)
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


def _current_activity(descendants: list[dict[str, int | str]]) -> tuple[str, str]:
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
    elif "gpt-5.6-sol" in commands:
        activity = "審查修正結果並決定通過或退回"
    elif "gpt-5.6-terra" in commands:
        activity = "依照審查意見修改程式"
    elif descendants:
        activity = "執行自動修復流程"
    else:
        activity = "目前沒有執行中的工作"

    if "gpt-5.6-terra" in commands:
        owner = "Terra High（工程修復）"
    elif "gpt-5.6-sol" in commands:
        owner = "Sol High（診斷／QA 審查）"
    elif descendants:
        owner = "Watchdog（自動修復控制器）"
    else:
        owner = "無"
    return owner, activity


def _latest_repair_activity(project: Path, now: datetime) -> dict[str, str] | None:
    candidates = list(project.parent.glob(f"{project.name}-watchdog-repair-*"))
    files: list[Path] = []
    for worktree in candidates:
        scripts = worktree / "scripts"
        if scripts.is_dir():
            files.extend(
                item
                for item in scripts.rglob("*")
                if item.is_file() and "payuni-sandbox" in item.name
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
