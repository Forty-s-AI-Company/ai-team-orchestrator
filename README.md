# AI Team Orchestrator

Reusable AI software-team orchestrator for local product repositories.

This repository is separate from both:

- `../CelebrateDeal`: the product repository.
- `../OpenHands`: the official OpenHands source repository.

The orchestrator talks to OpenHands through loopback HTTP only. It does not
import or modify OpenHands Python modules.

## Quick Start

```powershell
cd C:\Users\eden\Downloads\AI\ai-team-orchestrator
.\.venv\Scripts\Activate.ps1
pip install -e .

ai-team inspect ..\CelebrateDeal
ai-team validate ..\CelebrateDeal
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis
ai-team run ..\CelebrateDeal --workflow bug-fix-loop --dry-run
```

Default provider for `run` is `mock`, so local smoke tests do not require
OpenHands or a model key.

## OpenHands Provider

Default endpoint:

```text
http://127.0.0.1:31024
```

The provider requires `SESSION_API_KEY`. If it is missing, the provider fails
closed and refuses to create a conversation.

```powershell
$env:SESSION_API_KEY = "<local-session-key>"
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider openhands
```

If OpenHands is not running, `doctor` reports `ready: false` with a network or
timeout diagnostic. Do not treat a mock provider result as an OpenHands pass.
OpenHands Agent Canvas persists its generated API key at
`%USERPROFILE%\.openhands\agent-canvas\api-key.txt`; load it into the current
PowerShell process before provider-native smoke:

```powershell
$env:SESSION_API_KEY = (Get-Content -Raw "$env:USERPROFILE\.openhands\agent-canvas\api-key.txt").Trim()
```

## Receipts

Every `ai-team run` writes a redacted runtime receipt to:

```text
reports/receipts/
```

Receipts include project path, branch, provider, workflow, stages, commit SHA,
provider-native ready result, OpenHands conversation id, task id when available,
started/completed timestamps, duration, and validation result. Runtime receipts
are ignored by Git.

## Write Workflows

Non-dry-run write workflows require a disposable linked worktree. Running
`bug-fix-loop` against the primary CelebrateDeal worktree is denied.

```powershell
cd C:\Users\eden\Downloads\AI\CelebrateDeal
git worktree add --detach ..\CelebrateDeal-openhands-disposable HEAD

cd ..\ai-team-orchestrator
ai-team run ..\CelebrateDeal-openhands-disposable --workflow bug-fix-loop
```

## Safety Rules

- Project roots must remain inside the configured workspace allowlist.
- Protected branches such as `master` and `main` reject write workflows unless
  the workflow is explicitly dry-run.
- Non-dry-run write workflows require a linked disposable worktree by default.
- Workflows must explicitly forbid production deploy, real payment, and
  destructive migration.
- Provider logs and results redact token, secret, password, bearer, and `sk-*`
  patterns.
- The provider does not inherit or forward the full process environment.

## Tests

```powershell
python -m unittest discover -s tests
python -m compileall src tests
```
