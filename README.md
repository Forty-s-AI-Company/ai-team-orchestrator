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
ai-team git-policy ..\CelebrateDeal --action commit --file README.md
ai-team isolated-run ..\CelebrateDeal --workflow bug-fix-loop --provider mock --mode create-only
ai-team github-gate ..\CelebrateDeal --action push
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

The preferred unattended write entry point is `isolated-run`. It creates a
temporary disposable worktree from the source project, runs the write workflow
there, writes both the workflow receipt and executor receipt, then evaluates
the commit policy against the changed files.

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider mock `
  --mode create-only `
  --receipt-dir reports\isolated-smoke
```

Use `--remove-worktree` for short-lived smoke tests. Keep the worktree only
when a reviewer or later automation stage needs to inspect the generated diff.

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

Run hourly patrol through HandsFreeCode:

```powershell
ai-team supervise ..\CelebrateDeal `
  --provider handsfreecode `
  --mode create-only `
  --interval-minutes 60 `
  --max-runtime-minutes 480 `
  --state-path reports\supervisor\handsfreecode-state.json
```

The current supervisor is intentionally conservative. It writes structured
reports to `reports/supervisor/` and does not push, merge, deploy, run real
payments, or modify production data. Git push, PR, and merge automation must be
enabled later through explicit project safety policy, authenticated GitHub CLI,
branch protection checks, and reviewed receipts.

Supervisor state is written to `reports/supervisor/state.json` by default, or
to `--state-path` when supplied. Re-running the supervisor resumes from that
state by recording the prior revision in the next report; duplicate resumes are
safe and simply advance the state revision.

## Quota Fallback Policy

Codex and Antigravity quota exhaustion are treated as recoverable control-plane
states. The supervisor stores reset-time evidence when it can parse it and marks
the next action in state.

Ollama fallback is limited to documentation, triage, review, report, and
project-analysis workflows. It is blocked for write workflows, payments,
deployments, migrations, settlements, and payouts.

Fallback output is never reported as a Codex or Antigravity provider-native
pass. If HandsFreeCode uses Ollama internally, reports keep the outer provider
as `handsfreecode` and set `runtimeProvider: ollama`.

## Codex and Antigravity CLI Providers

The Codex and Antigravity providers are CLI bridges. They intentionally inherit
only a minimal process environment and redact command output before writing
receipts.

```powershell
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider codex --mode create-only
ai-team run ..\CelebrateDeal --workflow project-analysis --provider antigravity --mode create-only
```

Default diagnostics:

- Codex: `codex --version` and `codex doctor --json`
- Antigravity: `antigravity auth status` and `antigravity quota`

If a CLI reports quota exhaustion, the provider returns `rate_limit` and stores
the parsed reset time when available. The supervisor may then allow Ollama only
for documentation, triage, review, report, and project-analysis work. It must
not relabel fallback output as a provider-native Codex or Antigravity pass.

Antigravity execution is disabled by default until its real task command is
configured in `config/settings.yaml`; diagnostics still run provider-native.

## Git Automation Policy Gate

`ai-team git-policy` evaluates whether an automated Git action is allowed. It
does not perform the action.

```powershell
ai-team git-policy ..\CelebrateDeal --action commit --file README.md
ai-team git-policy ..\CelebrateDeal --action push
ai-team git-policy ..\CelebrateDeal --action pr
```

Default policy:

- `add` / `commit` require a disposable linked worktree.
- protected branches such as `master` and `main` block automated writes.
- ignored files, runtime artifacts, logs, reports, receipts, caches, venvs, and
  suspected secret files are blocked.
- `push`, `pr`, and `merge` remain `externalRequired` until GitHub auth, branch
  protection, reviewed receipts, and explicit project safety policy are wired.

`ai-team github-gate` evaluates GitHub-level automation. It is dry-run by
default and does not push, create PRs, or merge.

```powershell
ai-team github-gate ..\CelebrateDeal --action push
ai-team github-gate ..\CelebrateDeal --action pr --validation-log-hash <sha256>
ai-team github-gate ..\CelebrateDeal --action merge --validation-log-hash <sha256>
```

Push, PR, and merge execution require GitHub CLI authentication, branch
protection, validation log hashes, reviewed executor receipts, and explicit
project safety policy. Merge remains blocked until branch protection and review
status are wired into the gate.

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
- Ollama fallback is limited by workflow allowlist and never masquerades as a
  provider-native Codex or Antigravity result.

## External Required

- `OPENHANDS_LLM_API_KEY` for true `run-agent` execution.
- `HANDSFREECODE_SESSION_API_KEY` and the HandsFreeCode loopback server for
  HandsFreeCode provider-native runs.
- Codex / Antigravity quota reset time when the provider CLI is exhausted.
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
