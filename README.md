# AI Team Orchestrator

Reusable AI software-team orchestrator for local product repositories.

This repository is separate from both:

- `../CelebrateDeal`: the product repository.
- `../OpenHands`: the official OpenHands source repository.
- `../HandsFreeCode`: the local OpenHands-like runtime.

The orchestrator talks to OpenHands through loopback HTTP only. It does not
import or modify OpenHands Python modules.
It talks to HandsFreeCode through loopback HTTP only and does not import the
HandsFreeCode package directly.

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
ai-team supervise ..\CelebrateDeal --once
```

Default provider for `run` is `mock`, so local smoke tests do not require
OpenHands or a model key.

## HandsFreeCode Provider

Default endpoint:

```text
http://127.0.0.1:31025
```

Start the runtime from the sibling project:

```powershell
cd C:\Users\eden\Downloads\AI\HandsFreeCode
.\.venv\Scripts\Activate.ps1

$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:HANDSFREECODE_SESSION_API_KEY = [Convert]::ToHexString($bytes).ToLowerInvariant()

hfc serve
```

Use the same `HANDSFREECODE_SESSION_API_KEY` in the orchestrator shell:

```powershell
cd C:\Users\eden\Downloads\AI\ai-team-orchestrator
.\.venv\Scripts\Activate.ps1
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider handsfreecode --mode create-only
```

The HandsFreeCode provider calls `/ready` and `/api/tasks/run` with
`X-Session-API-Key`. If the local key is missing it fails closed before sending
a request. If the runtime is down, `doctor` marks HandsFreeCode as provider
native `externalRequired`; mock results must not be reported as HandsFreeCode,
Codex, or Antigravity passes.

HandsFreeCode may use Ollama internally for low-risk fallback work. The
orchestrator still records the outer provider as `handsfreecode` and the inner
runtime provider as `ollama`, never `codex` or `antigravity`.

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
ai-team run ..\CelebrateDeal --workflow project-analysis --provider openhands --mode create-only
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
provider-native ready result, conversation id, task id when available,
started/completed timestamps, duration, and validation result. Runtime receipts
are ignored by Git.

## Write Workflows

Non-dry-run write workflows require a disposable linked worktree. Running
`bug-fix-loop` against the primary CelebrateDeal worktree is denied.
OpenHands `run-agent` mode is also denied on the primary worktree, even for
read-only workflows.

```powershell
cd C:\Users\eden\Downloads\AI\CelebrateDeal
git worktree add --detach ..\CelebrateDeal-openhands-disposable HEAD

cd ..\ai-team-orchestrator
ai-team run ..\CelebrateDeal-openhands-disposable --workflow bug-fix-loop
```

`run-agent` starts the OpenHands agent loop via `/api/conversations/{id}/run`.
It requires a local LLM credential in `OPENHANDS_LLM_API_KEY`; without it the
provider returns `external_required` and does not create a conversation.

```powershell
$env:OPENHANDS_LLM_API_KEY = "<local-llm-key>"
ai-team run ..\CelebrateDeal-openhands-disposable --workflow project-analysis --provider openhands --mode run-agent
```

If `OPENHANDS_LLM_API_KEY` is not set, `run-agent` must not create an
OpenHands conversation. It returns `external_required` so the control plane can
resume later without pretending a provider-native agent pass happened.

## Autonomous Supervisor

The supervisor is the unified unattended loop entry point. It performs:

- discovery
- quality review
- triage
- safe auto-cycle
- QA handoff
- regression planning
- Git evidence collection

Run one cycle:

```powershell
ai-team supervise ..\CelebrateDeal --once
```

Run hourly safe patrol:

```powershell
ai-team supervise ..\CelebrateDeal --interval-minutes 60 --max-runtime-minutes 480
```

The current supervisor is intentionally conservative. It writes structured
reports to `reports/supervisor/` and does not push, merge, deploy, run real
payments, or modify production data. Git push, PR, and merge automation must be
enabled later through explicit project safety policy, authenticated GitHub CLI,
branch protection checks, and reviewed receipts.

## Safety Rules

- Project roots must remain inside the configured workspace allowlist.
- Protected branches such as `master` and `main` reject write workflows unless
  the workflow is explicitly dry-run.
- Non-dry-run write workflows require a linked disposable worktree by default.
- Workflows must explicitly forbid production deploy, real payment, and
  destructive migration.
- Provider logs and results redact token, secret, password, bearer, and `sk-*`
  patterns, including JSON keys such as `api_key`.
- The provider does not inherit or forward the full process environment.

## External Required

- `OPENHANDS_LLM_API_KEY` for true `run-agent` execution.
- `HANDSFREECODE_SESSION_API_KEY` and the HandsFreeCode loopback server for
  HandsFreeCode provider-native runs.
- Codex CLI authenticated quota for Codex-backed stages.
- Antigravity CLI authenticated quota for provider-native browser QA.
- GitHub CLI authentication and branch protection policy before automated push,
  pull request creation, or merge.
- PayUni remains sandbox-only until production payment approval.

## Tests

```powershell
python -m unittest discover -s tests
python -m compileall src tests
```
