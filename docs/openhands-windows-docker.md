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
  -p 127.0.0.1:31024:3000 `
  -e SESSION_API_KEY `
  -v C:\Users\eden\Downloads\AI:/projects `
  --name openhands-local `
  openhands/openhands:latest
```

If using a local source checkout, keep the working directory in
`C:\Users\eden\Downloads\AI\OpenHands` and configure its server to listen on
`127.0.0.1:31024`.

## Worktree Isolation

For write-capable workflows, create a disposable CelebrateDeal worktree and run
the orchestrator against that path:

```powershell
cd C:\Users\eden\Downloads\AI\CelebrateDeal
git worktree add ..\CelebrateDeal-ai-task codex\openhands-ai-task

cd ..\ai-team-orchestrator
ai-team init ..\CelebrateDeal-ai-task
ai-team run ..\CelebrateDeal-ai-task --workflow bug-fix-loop --dry-run
```

Do not run non-dry-run write workflows on `master` or `main`.

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
