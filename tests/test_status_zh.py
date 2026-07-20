from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ai_team.core.status_zh import render_chinese_status


def test_status_explains_stopped_supervisor_while_terra_is_repairing(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"title": "修正付款", "instruction": "補齊付款失敗測試"}),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "status": "stopped",
                "currentTask": {"id": "payment-fix", "contractPath": str(contract)},
            }
        ),
        encoding="utf-8",
    )

    def runner(command: list[str]) -> str:
        joined = " ".join(command)
        if "celebratedeal-ai-team-supervisor.service" in joined:
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        if "celebratedeal-ai-team-watchdog.service" in joined:
            return "ActiveState=activating\nSubState=start\nMainPID=100\n"
        if command[:2] == ["ps", "-eo"]:
            return (
                "100 1 600 0.1 Ss ai-team watchdog\n"
                "200 100 120 2.0 Sl codex --model gpt-5.6-terra\n"
                "201 200 20 10.0 Sl npm run lint\n"
            )
        if command[0] == "git":
            return "abc1234｜2026-07-20 12:00:00｜完成上一項修正\n"
        return ""

    output = render_chinese_status(
        project,
        supervisor_state_path=state,
        supervisor_service="celebratedeal-ai-team-supervisor.service",
        watchdog_service="celebratedeal-ai-team-watchdog.service",
        runner=runner,
        now=datetime.fromisoformat("2026-07-20T12:23:00+08:00"),
    )

    assert "自動修復中（AI Team 確實有在工作）" in output
    assert "Terra High（工程修復）" in output
    assert "ESLint 程式碼規範檢查" in output
    assert "主流程：暫停，由自動修復接管" in output
    assert "stopped 只代表主流程暫停" in output
    assert "標題：修正付款" in output


def test_status_reports_a_real_stop_when_no_service_has_a_pid(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")

    def runner(command: list[str]) -> str:
        if command[0] == "systemctl":
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        return ""

    output = render_chinese_status(
        project,
        supervisor_state_path=state,
        supervisor_service="supervisor.service",
        watchdog_service="watchdog.service",
        runner=runner,
        now=datetime.fromisoformat("2026-07-20T12:23:00+08:00"),
    )

    assert "已停止（目前沒有 AI Team 工作程序）" in output
    assert "目前負責：無" in output
