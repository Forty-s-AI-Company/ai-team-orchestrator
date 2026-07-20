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


def test_status_labels_systemd_auto_restart_as_a_temporary_transition(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "releaseReviewTasks": [{
                "id": "payment-check",
                "reason": "等待人工外部 QA 驗收；不阻塞後續測試站開發",
            }]
        }),
        encoding="utf-8",
    )

    def runner(command: list[str]) -> str:
        joined = " ".join(command)
        if "supervisor.service" in joined:
            return "ActiveState=activating\nSubState=auto-restart\nMainPID=0\n"
        if command[0] == "systemctl":
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        return ""

    output = render_chinese_status(
        project,
        supervisor_state_path=state,
        supervisor_service="supervisor.service",
        watchdog_service="watchdog.service",
        runner=runner,
        now=datetime.fromisoformat("2026-07-20T21:30:00+08:00"),
    )

    assert "主流程正在自動重啟（短暫切換中，不是永久停止）" in output
    assert "主流程：自動重啟等待中（短暫狀態）" in output
    assert "待人工上線驗收（不阻塞測試站開發）" in output


def test_status_reports_active_supervisor_even_between_model_processes(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"status": "completed-development", "nextAction": "next-contract"}),
        encoding="utf-8",
    )

    def runner(command: list[str]) -> str:
        joined = " ".join(command)
        if "supervisor.service" in joined:
            return "ActiveState=active\nSubState=running\nMainPID=100\n"
        if command[0] == "systemctl":
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        if command[:2] == ["ps", "-eo"]:
            return "100 1 20 0.1 Ss ai-team supervise\n"
        return ""

    output = render_chinese_status(
        project,
        supervisor_state_path=state,
        supervisor_service="supervisor.service",
        watchdog_service="watchdog.service",
        runner=runner,
        now=datetime.fromisoformat("2026-07-20T21:35:00+08:00"),
    )

    assert "整體狀態：主流程自動開發中" in output
    assert "目前負責：Supervisor（自動開發控制器）" in output
    assert "正在進行：切換到下一個優先任務" in output


def test_status_shows_readable_failed_repair_history(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "watchdog-ai-repair-current.json").write_text(
        json.dumps({
            "status": "deferred",
            "cycleLimit": 5,
            "activePhase": "deferred",
            "cycles": [
                {
                    "cycle": 1,
                    "reasoningEffort": "high",
                    "outcome": "review-rejected",
                    "agyQa": {"status": "failed"},
                    "solReview": {"status": "failed"},
                    "failureSummary": "退款狀態仍不一致",
                },
                {
                    "cycle": 2,
                    "reasoningEffort": "xhigh",
                    "outcome": "deterministic-qa-failed",
                    "failureSummary": "單元測試失敗",
                },
            ],
            "deferReason": "連續 5 輪仍未通過，已暫緩",
        }),
        encoding="utf-8",
    )

    def runner(command: list[str]) -> str:
        if command[0] == "systemctl":
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        return ""

    output = render_chinese_status(
        project,
        supervisor_state_path=state,
        supervisor_service="supervisor.service",
        watchdog_service="watchdog.service",
        report_dir=reports,
        runner=runner,
        now=datetime.fromisoformat("2026-07-20T12:23:00+08:00"),
    )

    assert "自動修復歷程（最多 5 輪）" in output
    assert "第 1 輪｜Sol/Terra High｜QA／複檢退回" in output
    assert "AGY QA 未通過" in output
    assert "第 2 輪｜Sol/Terra XHigh｜基礎測試未通過" in output
    assert "已記錄並跳過" in output
