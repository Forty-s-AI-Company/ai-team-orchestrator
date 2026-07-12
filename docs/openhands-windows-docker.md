# OpenHands Windows and Docker Runbook

This runbook keeps `ai-team-orchestrator`, `CelebrateDeal`, and official
`OpenHands` source code separated.
HandsFreeCode is a sibling runtime and remains separate as well.

## Local Layout

```text
C:\Users\eden\Downloads\AI\
├─ CelebrateDeal
├─ OpenHands
├─ ai-team-orchestrator
└─ HandsFreeCode
```

Do not edit `OpenHands` from this orchestrator. Treat it as an external worker.
Do not import HandsFreeCode directly; call it through loopback HTTP.

## HandsFreeCode Loopback

HandsFreeCode is reserved for:

```text
http://127.0.0.1:31025
```

Start it from its own repository:

```powershell
cd C:\Users\eden\Downloads\AI\HandsFreeCode
.\.venv\Scripts\Activate.ps1

$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:HANDSFREECODE_SESSION_API_KEY = [Convert]::ToHexString($bytes).ToLowerInvariant()

hfc serve
```

In the orchestrator shell, use the same local-only value:

```powershell
cd C:\Users\eden\Downloads\AI\ai-team-orchestrator
.\.venv\Scripts\Activate.ps1
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider handsfreecode --mode create-only
```

The provider calls:

```text
GET  /ready
POST /api/tasks/run
```

Protected calls use `X-Session-API-Key`. Missing
`HANDSFREECODE_SESSION_API_KEY` fails closed locally before a task request is
sent. A remote 401 is treated as auth failure; a remote 503 is treated as
`external_required`.

HandsFreeCode may call Ollama internally. That result is recorded as
`runtimeProvider: ollama` under the outer provider `handsfreecode`; it is never
treated as a Codex or Antigravity provider-native pass.

## Port

OpenHands loopback is reserved for:

```text
http://127.0.0.1:31024
```

The reservation is recorded in `C:\Users\eden\Downloads\AI\ports.json`.
HandsFreeCode is also recorded there on port `31025`.

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

For unattended runs, prefer the isolated executor:

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider mock `
  --mode create-only `
  --receipt-dir reports\isolated-smoke
```

The isolated executor creates a disposable linked worktree, runs the workflow,
writes a run receipt and executor receipt, and evaluates the Git commit policy
against changed files. Use `--remove-worktree` for smoke tests when the diff
does not need to be inspected later.

If the provider writes a safe diff, enable local auto-commit inside the
disposable worktree:

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider handsfreecode `
  --mode create-only `
  --auto-commit `
  --commit-message "chore(ai-team): apply guarded fix"
```

Auto-commit never runs on the primary worktree. It evaluates changed files with
`git-policy`, stages them, evaluates staged files again, then writes the new
commit SHA into the executor receipt. Push / PR / merge still require
`github-gate`.

## Provider-Native Smoke

```powershell
$env:SESSION_API_KEY = (Get-Content -Raw "$env:USERPROFILE\.openhands\agent-canvas\api-key.txt").Trim()
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider openhands --mode create-only
```

If OpenHands is unavailable, the provider-native run must fail with a network or
timeout diagnostic. Do not treat a mock provider result as an OpenHands pass.
The smoke creates an idle OpenHands conversation with `run=false`; it does not
start the agent loop or spend model tokens.

## Agent Loop Mode

`run-agent` mode explicitly calls:

```text
/api/conversations/{conversation_id}/run
```

It is only allowed on a disposable linked worktree and requires a local LLM
credential:

```powershell
$env:SESSION_API_KEY = (Get-Content -Raw "$env:USERPROFILE\.openhands\agent-canvas\api-key.txt").Trim()
$env:OPENHANDS_LLM_API_KEY = "<local-llm-key>"
ai-team run ..\CelebrateDeal-openhands-disposable --workflow project-analysis --provider openhands --mode run-agent
```

Without `OPENHANDS_LLM_API_KEY`, `run-agent` returns `external_required` and
does not create a conversation.

## Autonomous Supervisor

Use the supervisor as the single unattended entry point instead of calling
individual provider probes by hand:

```powershell
cd C:\Users\eden\Downloads\AI\ai-team-orchestrator
.\.venv\Scripts\Activate.ps1
ai-team supervise ..\CelebrateDeal --once
ai-team supervise ..\CelebrateDeal --interval-minutes 60 --max-runtime-minutes 480
```

Use HandsFreeCode as the provider:

```powershell
cd C:\Users\eden\Downloads\AI\HandsFreeCode
.\.venv\Scripts\Activate.ps1
$env:HANDSFREECODE_SESSION_API_KEY = "<local-random-value>"
hfc serve

cd C:\Users\eden\Downloads\AI\ai-team-orchestrator
.\.venv\Scripts\Activate.ps1
$env:HANDSFREECODE_SESSION_API_KEY = "<same-local-random-value>"
ai-team supervise ..\CelebrateDeal `
  --provider handsfreecode `
  --mode create-only `
  --interval-minutes 60 `
  --max-runtime-minutes 480 `
  --state-path reports\supervisor\handsfreecode-state.json
```

Pause by closing the shell or stopping the process with `Ctrl+C`.
Resume by running the same command with the same `--state-path`; the next report
will include the previous state revision. Stop by ending the supervisor process
and, if needed, stopping HandsFreeCode with `Stop-Process` or by closing its
terminal.

The supervisor stages are discovery, quality review, triage, safe auto-cycle,
QA handoff, regression planning, and Git evidence collection. Reports are
written to:

```text
C:\Users\eden\Downloads\AI\ai-team-orchestrator\reports\supervisor
```

This is currently a safe patrol loop. It does not push, merge, deploy, run
production payments, or run destructive migrations. Automated git push / PR /
merge requires a later policy gate with GitHub CLI authentication, branch
protection checks, reviewed receipts, and explicit project safety settings.

## Quota and Ollama Fallback

When Codex or Antigravity reports quota exhaustion, the supervisor records the
provider, parsed reset time when available, fallback policy, and next action in
state. It does not pretend that fallback work is provider-native.

Codex and Antigravity are checked through their native CLIs:

```powershell
ai-team doctor
ai-team run ..\CelebrateDeal --workflow project-analysis --provider codex --mode create-only
ai-team run ..\CelebrateDeal --workflow project-analysis --provider antigravity --mode create-only
```

The bridge does not inherit the full process environment. Keep credentials in
the provider's own authenticated CLI storage or explicitly documented local
variables. Do not place keys in `config/settings.yaml`.

Default commands:

- Codex: `codex --version`, `codex doctor --json`
- Antigravity: `antigravity auth status`, `antigravity quota`

If these commands timeout or report quota exhaustion, mark the provider as
`External required`; do not substitute mock or Ollama output as a provider pass.

Allowed Ollama fallback scope:

- documentation
- triage
- review
- report
- project analysis

Blocked Ollama fallback scope:

- implementation/write workflows
- production deployment
- payments
- migrations
- settlement / payout work

If HandsFreeCode uses Ollama internally, reports keep:

```json
{
  "provider": "handsfreecode",
  "runtimeProvider": "ollama",
  "masqueradeAsCodexOrAntigravity": false
}
```

## External Required Interpretation

- `handsfreecode_loopback`: start `hfc serve` on `127.0.0.1:31025`.
- `session_key`: set the local session key in the current shell.
- `llm_credentials`: set the provider's local LLM credential before agent mode.
- quota exhaustion: wait until reset time or allow only low-risk Ollama fallback.
- `codex_cli`: confirm Codex login, quota, and `codex doctor --json` latency.
- `antigravity_cli`: confirm Antigravity login, quota, and native browser QA.

## Git Automation Gate

Before any autonomous Git operation, evaluate the policy:

```powershell
ai-team git-policy ..\CelebrateDeal --action commit --file README.md
ai-team git-policy ..\CelebrateDeal --action push
ai-team git-policy ..\CelebrateDeal --action pr
```

Current gate behavior:

- `add` / `commit` are allowed only in a disposable linked worktree.
- `master` and `main` are protected by default.
- ignored files, runtime artifacts, logs, reports, receipts, cache folders,
  virtualenvs, and suspected secret files are rejected.
- `push`, PR creation, and merge are intentionally `External required` until
  GitHub CLI auth, branch protection, reviewed receipts, staged secret scan,
  and explicit project safety policy are connected.

GitHub-level automation is guarded separately:

```powershell
ai-team github-gate ..\CelebrateDeal --action push
ai-team github-gate ..\CelebrateDeal --action pr --validation-log-hash <sha256>
ai-team github-gate ..\CelebrateDeal --action merge --validation-log-hash <sha256>
```

`github-gate` is dry-run by default. It must not perform real push, PR, or
merge operations unless a later policy explicitly enables execution and the
required receipts, validation hash, GitHub CLI authentication, and branch
protection checks are present.

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
ai-team run ..\CelebrateDeal --workflow project-analysis --provider handsfreecode --mode create-only
```

Expected without OpenHands:

- `ai-team doctor` reports `ready: false`.
- `sessionKeyPresent: false` means OpenHands provider is blocked by design.
- HandsFreeCode reports `externalRequired` when its loopback server or local
  session key is missing.
- Mock workflow still works for local control-plane validation.

## Stop

```powershell
docker stop openhands-local
```

## External Required

- Real OpenHands container/image availability.
- Local model or remote LLM credentials configured inside OpenHands.
- `OPENHANDS_LLM_API_KEY` before `run-agent` can call `/api/conversations/{id}/run`.
- `HANDSFREECODE_SESSION_API_KEY` and the HandsFreeCode loopback server before
  provider-native HandsFreeCode runs.
- Codex CLI and Antigravity CLI login/quota for provider-native automation.
- GitHub CLI authentication and branch protection policy before automated push,
  PR, or merge.
- PayUni remains sandbox-only until production approval.
- Human login or provider dashboard work, if OpenHands requires it.
