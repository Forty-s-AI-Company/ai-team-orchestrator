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

For real LLM generation on a read-only workflow without creating a disposable
worktree, use the explicit HandsFreeCode-only mode:

```powershell
ai-team run ..\CelebrateDeal --workflow project-analysis --provider handsfreecode --mode read-only-agent
```

The outer receipt records `runMode=read-only-agent`; HandsFreeCode maps the
runtime call to `run-agent` with `writeAccess=false`. Write workflows and all
other providers fail closed in this mode. `create-only` continues to create
state and receipts without calling an LLM.

For workflows that declare a bounded evidence policy, `read-only-agent`
collects only platform-allowlisted text files. It rejects sensitive paths,
symlinks, binary or oversized candidates, applies per-file and total prompt
limits, and redacts content before sending it to HandsFreeCode. The receipt
stores file hashes and exclusion counts rather than excluded paths or secret
values. Provider execution, evidence collection, and analysis grounding are
validated separately; unsupported dependency or coverage claims fail closed.

For unattended Windows runs, both HandsFreeCode and this orchestrator can read a
local key file outside the repository:

```powershell
$keyDir = Join-Path $env:USERPROFILE ".handsfreecode"
New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToHexString($bytes).ToLowerInvariant() |
  Set-Content -LiteralPath (Join-Path $keyDir "session-api-key.txt") -Encoding UTF8 -NoNewline
```

The file is intentionally not tracked by Git.

## Auto Provider Routing

Without a role, `--provider auto` routes only through the three audited native
providers: Codex CLI, Antigravity CLI, then HandsFreeCode. OpenHands remains an
explicit compatibility provider and mock remains an explicit test provider;
neither can be selected automatically. Auth, unavailable, quota, timeout, and
network failures are recorded in `routeAttempts`. If HandsFreeCode uses Ollama
internally, the receipt stays
`provider=handsfreecode`, `runtimeProvider=ollama`; it is never relabeled as a
Codex or Antigravity pass.

Role-aware routing selects an exact provider, account-visible model name, and
reasoning level from the allowlisted profiles in `config/settings.yaml`:

```bash
ai-team run /home/eden/projects/CelebrateDeal \
  --workflow project-analysis \
  --provider auto \
  --role project-analyst \
  --mode read-only-agent
```

The current profiles are:

| Role | Primary | Fallback / second opinion |
| --- | --- | --- |
| Product Manager | Antigravity Gemini 3.5 Flash (High) | Codex gpt-5.6-terra, medium |
| Architect | Antigravity Gemini 3.1 Pro (High) | Codex gpt-5.6-sol, high fallback and second opinion |
| Engineer | Codex gpt-5.6-terra, high | none for write workflows |
| Reviewer | Codex gpt-5.5, high | Claude Sonnet 4.6 (Thinking) fallback and second opinion |
| QA Engineer | HandsFreeCode / qwen2.5-coder:7b, provider default | none in read-only-agent mode |
| Project Analyst | HandsFreeCode / qwen2.5-coder:7b, provider default | none in read-only-agent mode |

Fallback is limited to provider availability failures. Invalid or unvalidated
model output fails closed. Write workflows never cross-provider fallback, and
second opinions are always forced to read-only metadata. Receipts record the
role, selected model, reasoning effort, fallback chain, and redacted secondary
review without changing the existing schema version.

## OpenHands Provider

OpenHands is disabled by automatic routing policy because HandsFreeCode now
provides the local runtime path. `ai-team doctor` reports
`status=disabled_by_policy` and `externalRequired=false` for this policy state.
The explicit provider remains available for compatibility testing only.

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

On Windows, the orchestrator also reads the Agent Canvas key file configured in
`config/settings.yaml`:

```text
%USERPROFILE%\.openhands\agent-canvas\api-key.txt
```

The key value is never printed or committed. If both env var and file exist,
the env var wins.

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
started/completed timestamps, duration, and validation result. Read-only
analysis receipts also include a content-free evidence manifest with relative
paths, sizes, SHA-256 hashes, truncation flags, limits, redaction counts, and
grounding validation. Runtime receipts are ignored by Git.

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

For a deterministic control-plane smoke that cannot edit product code, use
`write-smoke`. It is accepted only by `isolated-run`, verifies the linked
worktree through Git, and exclusively creates
`docs/ai-team-smoke/isolated-write-smoke.md`:

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider write-smoke `
  --mode create-only `
  --auto-commit `
  --github-action pr `
  --github-branch ai-team/isolated-write-smoke `
  --validation-log-hash <sha256> `
  --test-evidence-hash <sha256>
```

Without `--github-execute`, the GitHub action remains a dry-run. Adding it may
push the disposable commit and create a real pull request; it never merges it.

The PR gate validates the generated run receipt against the exact output
commit, verifies that the source commit is a strict ancestor, scans every
changed Git blob in the attested range for secret-like content, and requires SHA-256 validation and test
evidence. A merge dry-run also calls `gh pr view`; missing approval or a
non-clean branch-protection state remains blocked and no merge command runs.

The Antigravity adapter uses a provider-specific compact prompt and passes the
project through `--add-dir` in sandboxed plan mode. A trivial native request can
prove login/model availability, while a workflow timeout remains a timeout and
must never be reported as an Antigravity pass or replaced by a Codex label.

Antigravity native success now requires a challenge-bound JSON response. The
`provider-smoke` workflow asks Antigravity to read a small tracked manifest and
return its SHA-256 without receiving the expected hash in the prompt:

```powershell
ai-team run ..\CelebrateDeal --workflow provider-smoke --provider antigravity --dry-run
```

Plain text, Markdown fences, stale challenges, incorrect paths or hashes, and
CLI help output are `INVALID_RESPONSE`, even when the CLI exits with code zero.
Successful diagnostics are cached briefly and share one monotonic deadline with
the model command.

Use `pr-monitor` to poll pull-request checks and save redacted evidence:

```powershell
ai-team pr-monitor <disposable-worktree> `
  --repo owner/repository `
  --pr 123 `
  --wait-seconds 600 `
  --report-dir reports/pr-123-ci
```

Dependency lock failures produce a policy-built restricted repair task. The
task is not automatically executable, permits only `package-lock.json`, allows
at most one attempt, and forbids push and merge. CI log text remains untrusted
evidence and never controls commands, paths, provider selection, or prompts.

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider mock `
  --mode create-only `
  --receipt-dir reports\isolated-smoke
```

Use `--remove-worktree` for short-lived smoke tests. Keep the worktree only
when a reviewer or later automation stage needs to inspect the generated diff.

When a provider actually writes files, `--auto-commit` can commit the generated
diff inside the disposable worktree only. It never commits on the primary
CelebrateDeal worktree.

```powershell
ai-team isolated-run ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider handsfreecode `
  --mode create-only `
  --auto-commit `
  --commit-message "chore(ai-team): apply guarded fix"
```

Auto-commit first evaluates `git-policy`, rejects ignored/runtime/secret-like
files, stages only the changed files, evaluates staged files again, and writes
the commit SHA into the executor receipt.

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

Run a write-capable workflow through the supervisor only when the isolated
executor is intended:

```powershell
ai-team supervise ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider handsfreecode `
  --mode create-only `
  --execute `
  --auto-commit `
  --once
```

In this mode the supervisor creates a disposable worktree and delegates the
write workflow to `isolated-run`. The primary project worktree remains read
only from the supervisor's point of view.

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

- Codex: `codex --version`
- Antigravity: `agy --version`

On Windows, CLI resolution prefers `.cmd`, `.exe`, `.bat`, and `.ps1` shims
before extensionless files. This avoids the `WinError 5` access issue caused by
executing the extensionless npm shim directly.

If a CLI reports quota exhaustion, the provider returns `rate_limit` and stores
the parsed reset time when available. The supervisor may then allow Ollama only
for documentation, triage, review, report, and project-analysis work. It must
not relabel fallback output as a provider-native Codex or Antigravity pass.

Antigravity execution is enabled through `agy --print` in plan mode. Current
local evidence shows trivial prompts succeed, workflow-sized prompts can return
`Error: timeout waiting for response`, and GPT-OSS 120B may report individual
quota reset evidence. The auto router records those as provider-native timeout
or rate-limit evidence and continues to the next provider; fallback must not be
relabeled as an Antigravity pass.

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
- `push` and `pr` require a disposable worktree plus explicit project push
  policy; PR also requires a validation log hash.
- `merge` remains `externalRequired` until branch protection and review status
  are wired.

`ai-team github-gate` evaluates GitHub-level automation. It is dry-run by
default. Add `--execute` only from a disposable worktree after validation
evidence exists.

```powershell
ai-team github-gate <disposable-worktree> --action push --receipt-path <run-receipt-json>
ai-team github-gate <disposable-worktree> --action pr --validation-log-hash <sha256> --receipt-path <run-receipt-json>
ai-team github-gate <disposable-worktree> --action merge --validation-log-hash <sha256> --receipt-path <run-receipt-json> --test-evidence-hash <sha256>
```

GitHub executor receipts include selected branch, validation hash, run receipt
hash, secret scan hash, test evidence hash, and redacted `git push`, `gh pr
create`, or `gh pr merge` command evidence. Merge requires validation, receipt,
secret scan, and test evidence hashes. Primary worktrees and protected branches
remain blocked.

Supervisor can connect an isolated write commit to GitHub automation:

```powershell
ai-team supervise ..\CelebrateDeal `
  --workflow bug-fix-loop `
  --provider auto `
  --execute `
  --auto-commit `
  --github-action pr `
  --validation-log-hash <sha256> `
  --test-evidence-hash <sha256> `
  --once
```

Without `--github-execute`, the GitHub action is evaluated and recorded as a
dry-run. Add `--github-execute` only when the isolated worktree commit has
passed the configured validation gates.

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

## Autonomous Delivery Supervisor

Use `--delivery` to enable deterministic discovery, trusted task promotion, an
isolated write worktree, validation-before-commit, and guarded PR creation:

```powershell
ai-team supervise ..\CelebrateDeal --provider auto --delivery --execute `
  --auto-commit --github-action pr --github-execute --interval-minutes 60
```

Runtime state and the normalized trusted queue are written under
`reports/supervisor/`. Without `--delivery`, `supervise` retains its read-only
project-analysis behavior. Automated merge remains gated by successful CI,
receipt and secret-scan evidence, and an approved GitHub review.
