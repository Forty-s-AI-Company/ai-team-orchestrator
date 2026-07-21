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
    task_sha = "c" * 64
    state.write_text(
        json.dumps({
            "revision": 12,
            "currentTask": {"id": "payment-repair", "taskSha": task_sha},
        }),
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "watchdog-ai-repair-current.json").write_text(
        json.dumps({
            "status": "deferred",
            "startedAt": "2026-07-20T04:00:00+00:00",
            "cycleLimit": 5,
            "activePhase": "deferred",
            "acceptanceContract": {
                "sha256": "a" * 64,
                "acceptanceCriteria": [{"id": "AC-1"}],
            },
            "followUpFindings": [{
                "id": "ARCH-1",
                "evidence": "建議重寫發布協議，已拆成後續工作",
            }],
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
            "supervisorEvidence": {
                "revision": 12,
                "currentTask": {"id": "payment-repair", "taskSha": task_sha},
            },
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
    assert "驗收契約已凍結（1 項）" in output
    assert "範圍外發現（已另行記錄，不阻擋目前修復）" in output
    assert "建議重寫發布協議" in output
    assert "歷史修復紀錄" not in output


def test_status_marks_mismatched_repair_as_history_and_keeps_deferred_tasks(tmp_path: Path) -> None:
    project = tmp_path / "CelebrateDeal"
    project.mkdir()
    current_sha = "d" * 64
    old_sha = "e" * 64
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "revision": 21,
            "currentTask": {"id": "current-checkout", "taskSha": current_sha},
            "deferredTasks": [{
                "id": "old-payment-repair",
                "reason": "連續五輪仍未通過，已暫緩",
            }],
        }),
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    report = {
        "schema": "ai-team-watchdog-repair/v1",
        "startedAt": "2026-07-20T02:00:00+00:00",
        "completedAt": "2026-07-20T03:00:00+00:00",
        "status": "deferred",
        "cycleLimit": 5,
        "cycles": [{
            "cycle": 5,
            "reasoningEffort": "xhigh",
            "outcome": "review-rejected",
        }],
        "deferReason": "舊任務五輪未通過",
        "supervisorEvidence": {
            "revision": 19,
            "currentTask": {"id": "old-payment-repair", "taskSha": old_sha},
        },
    }
    (reports / "watchdog-ai-repair-current.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    # The timestamped report is the same run; it must not appear twice.
    (reports / "watchdog-ai-repair-20260720T030000Z.json").write_text(
        json.dumps(report),
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

    assert "自動修復歷程（最多 5 輪）" not in output
    assert output.count("歷史修復紀錄（非目前任務／非目前執行批次）") == 1
    assert "歷史任務：old-payment-repair（eeeeeeeeeeee）" in output
    assert "歷史結果：達修復上限，已記錄並暫緩" in output
    assert "已暫緩任務（不阻塞其他工作）" in output
    assert "old-payment-repair：連續五輪仍未通過，已暫緩" in output
