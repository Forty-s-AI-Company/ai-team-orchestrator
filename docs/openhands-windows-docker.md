# OpenHands Windows and Docker Runbook

This runbook keeps `ai-team-orchestrator`, `CelebrateDeal`, and official
`OpenHands` source code separated.

## Local Layout

```text
C:\Users\eden\Downloads\AI\
├─ CelebrateDeal
├─ OpenHands
└─ ai-team-orchestrator
```

Do not edit `OpenHands` from this orchestrator. Treat it as an external worker.

## Port

OpenHands loopback is reserved for:

```text
http://127.0.0.1:31024
```

The reservation is recorded in `C:\Users\eden\Downloads\AI\ports.json`.

## SESSION_API_KEY

Use a local-only key. Do not commit it.

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:SESSION_API_KEY = [Convert]::ToHexString($bytes).ToLowerInvariant()
```

The OpenHands provider fails closed if `SESSION_API_KEY` is missing.

## Start OpenHands on Loopback

Preferred production-like mode is Docker sandbox. Bind only loopback and mount
only the AI workspace path needed for local development.

Example shape:

```powershell
docker run --rm -it `
  -p 127.0.0.1:31024:8000 `
  -v C:\Users\eden\Downloads\AI:/projects `
  --name openhands-local `
  ghcr.io/openhands/agent-canvas:1.0.0-rc.11
```

Agent Canvas generates and persists its local API key at:

```text
%USERPROFILE%\.openhands\agent-canvas\api-key.txt
```

Load it into the current PowerShell process before running provider-native
smoke. Do not print or commit the value.

```powershell
$env:SESSION_API_KEY = (Get-Content -Raw "$env:USERPROFILE\.openhands\agent-canvas\api-key.txt").Trim()
```

If using a local source checkout, keep the working directory in
`C:\Users\eden\Downloads\AI\OpenHands` and configure its server to listen on
`127.0.0.1:31024`.

## Worktree Isolation

For write-capable workflows, create a disposable CelebrateDeal worktree and run
the orchestrator against that path:

```powershell
cd C:\Users\eden\Downloads\AI\CelebrateDeal
git worktree add --detach ..\CelebrateDeal-openhands-disposable HEAD

cd ..\ai-team-orchestrator
ai-team run ..\CelebrateDeal-openhands-disposable --workflow bug-fix-loop
```

Do not run non-dry-run write workflows on `master` or `main`.
Do not run non-dry-run write workflows on the primary worktree.

## Provider-Native Smoke

```powershell
$env:SESSION_API_KEY = (Get-Content -Raw "$env:USERPROFILE\.openhands\agent-canvas\api-key.txt").Trim()
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider openhands
```

If OpenHands is unavailable, the provider-native run must fail with a network or
timeout diagnostic. Do not treat a mock provider result as an OpenHands pass.
The smoke creates an idle OpenHands conversation with `run=false`; it does not
start the agent loop or spend model tokens.

## Receipts

Runtime receipts are written to:

```text
C:\Users\eden\Downloads\AI\ai-team-orchestrator\reports\receipts
```

They are redacted and ignored by Git. A receipt records project path, branch,
provider, workflow, stages, commit SHA, provider-native ready result, OpenHands
conversation id, task id when available, started/completed timestamps, duration,
and validation result.

## Diagnostics

```powershell
ai-team doctor
ai-team inspect ..\CelebrateDeal
ai-team validate ..\CelebrateDeal
ai-team run ..\CelebrateDeal --workflow project-analysis
```

Expected without OpenHands:

- `ai-team doctor` reports `ready: false`.
- `sessionKeyPresent: false` means OpenHands provider is blocked by design.
- Mock workflow still works for local control-plane validation.

## Stop

```powershell
docker stop openhands-local
```

## External Required

- Real OpenHands container/image availability.
- Local model or remote LLM credentials configured inside OpenHands.
- Human login or provider dashboard work, if OpenHands requires it.
