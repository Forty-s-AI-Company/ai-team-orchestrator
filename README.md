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
| Engineer | Codex gpt-5.6-terra, high | Persistent Terra → Sol → Luna → HandsFreeCode/Ollama qwen2.5-coder fallback |
| Reviewer | Codex gpt-5.6-sol, xhigh | Antigravity Gemini 3.1 Pro (High) mandatory second opinion |
| QA Engineer | HandsFreeCode / qwen2.5-coder:7b, provider default | none in read-only-agent mode |
| Delivery QA | Antigravity Gemini 3.1 Pro (High) | Codex gpt-5.6-sol, xhigh mandatory second QA |
| Project Analyst | HandsFreeCode / qwen2.5-coder:7b, provider default | none in read-only-agent mode |

Fallback is limited to supervisor-classified provider availability failures.
Invalid or unvalidated model output fails closed. Each write attempt is bound
to one selected provider and the same disposable worktree; providers never
switch implicitly mid-attempt. Second opinions are always forced to read-only metadata. Receipts record the
role, selected model, reasoning effort, fallback chain, and redacted secondary
review without changing the existing schema version.

## Continuous bounded-delivery cloud recovery

The non-`--once` bounded supervisor persists Engineer recovery state in its
state file. It treats temporary rate limits, capacity errors, timeouts,
connection resets, and HTTP 429/502/503/504-style failures differently from
authentication, billing, code-validation, and infrastructure failures.

- Engineer route order is Codex `gpt-5.6-terra` (`high`) →
  `gpt-5.6-sol` (`medium`) → `gpt-5.6-luna` (`medium`) → local
  HandsFreeCode/Ollama `qwen2.5-coder:7b` (`default`).
- Each model has an independent retry count, exponential backoff with jitter,
  circuit state, cooldown, and probe timestamp. A write worktree never changes
  provider route. The local route is accepted only when its native receipt
  attests `runtimeProvider=ollama` and `writeAccess=true`.
- A temporary error first retries the same model within its configured budget;
  only then does the supervisor open that model circuit and select the next
  model. A successful low-cost readiness probe closes a circuit. At a safe
  stage boundary, Terra is preferred again.
- When all model circuits are open, the task becomes `cloud_waiting`, not
  permanently blocked. The supervisor stays active and waits for the next
  single-flight probe. Account/authentication failures, unsafe state, and
  unreconcilable Git state remain fail-closed human gates.

`providers.codex_engineer` in `config/settings.yaml` controls the model order,
retry budget, circuit breaker, and probe budget. The state output includes
`cloudResilience`, `nextAction`, and `continuity` so a status report can say
whether automatic recovery is still enabled.

### Continuity recorder and resume packets

`local_continuity` is deliberately **recorder-only**. It cannot write a product
repository, perform Git writes, run tests, generate code, create commits, push,
open PRs, merge, migrate, seed, deploy, or mark an Engineer stage complete.
If no explicitly configured local recorder is available, the supervisor uses a
deterministic Python fallback and atomically writes redacted JSON and Markdown
resume packets below the configured state directory:

```text
~/.local/state/ai-team/<project>/continuity/<task-id>/
```

Packets contain the task/stage, per-model circuit state, retry history, Git
summary, receipts, pending QA/review/PR/CI/merge stages, and a precise resume
action. They never contain secret values. Before a resumed cloud Engineer acts,
the existing bounded-delivery worktree validation reconciles the recorded
worktree, branch, HEAD, receipts, and changed files; it never resets or cleans
human changes.

An optional local recorder is configured explicitly; it is run only inside a
network-isolated `bwrap` sandbox where the product repository and home directory
are not mounted. Its
stdout is an informational summary only, never code or a completion signal:

```yaml
local_continuity:
  enabled: true
  provider: ollama            # label for observability; not a cloud fallback
  model: small-local-model
  command: ["local-recorder-adapter", "--stdin-json"]
  timeout_seconds: 180
  max_output_tokens: 2000
  allow_repository_writes: false
  allow_git_writes: false
  allow_test_execution: false
  allow_code_generation: false
  deterministic_fallback: true
```

If the sandbox or command is unavailable, the deterministic packet remains the
source of truth and the supervisor continues waiting for cloud recovery.

To pause automatic recovery, stop the supervisor service. To manually clear a
circuit, stop the supervisor first, inspect its redacted state and resume
packet, then restart it only after deciding that the persisted task/worktree is
safe to resume. Do not edit a state file while the supervisor lock is held.

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
Bounded-delivery prompts preserve the complete trusted task instruction within
an 8192-character cap. If the instruction and mandatory policy context cannot
fit losslessly, the provider fails closed before invoking the model instead of
silently truncating acceptance requirements.

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

### Bounded autonomous delivery

`--bounded-delivery` is a separate fail-closed delivery path. It never
discovers product requirements itself: an operator supplies versioned JSON task
contracts that cite either a GitHub issue or another trusted source. Every
contract must declare the exact approved write paths and must always run the
project contract's `lint`, `typecheck`, `test`, and `build` commands. A project
may additionally allowlist task-specific deterministic checks under
`commands.additional_validation`; a task can select from that tracked list,
while an undeclared command still fails closed.

For example, a project with an existing Playwright smoke script can opt in
without allowing arbitrary task-contract shell commands:

```yaml
commands:
  lint: npm run lint
  typecheck: npm run typecheck
  test: npm run test
  build: npm run build
  additional_validation:
    - npm run e2e:smoke
```

The role and repair cycle runs PM (Antigravity), Architect (Antigravity plus mandatory
read-only Codex second opinion), Codex Engineer in a disposable worktree,
deterministic validation plus `git diff --check`, Antigravity delivery QA, and
Codex review (with optional Antigravity second opinion). It records a redacted
receipt per stage and never itself invokes GitHub publication, migration, seed,
deploy, payment, secret, destructive-data, or force-push actions. Schema/API
code changes, migration artifacts, and deterministic fixture data remain denied
unless the trusted contract explicitly opts into the exact code-only change
type. The continuous supervisor may publish only after that cycle completes
with deterministic evidence and all GitHub gates pass.

Start with exactly one cycle. `--execute` and `--once` select the single-task
form:

```bash
cat > /tmp/trusted-task.json <<'JSON'
{
  "schemaVersion": 1,
  "id": "github-issue-123",
  "title": "Exact issue title",
  "source": {"kind": "github-issue", "reference": "owner/repo#123"},
  "instruction": "Implement only the documented issue acceptance criteria.",
  "allowedWritePaths": ["src/example.ts", "src/example.test.ts"],
  "validationCommands": ["npm run lint", "npm run typecheck", "npm run test", "npm run build"],
  "dependsOn": [],
  "changePolicy": {
    "schemaChanges": false,
    "apiContractChanges": false,
    "migrationArtifacts": false,
    "fixtureData": false
  }
}
JSON

ai-team supervise /home/eden/projects/CelebrateDeal \
  --provider auto --bounded-delivery --task-contract /tmp/trusted-task.json \
  --execute --once --max-iterations 2 --max-repair-attempts 1 \
  --max-token-usage 120000 --stage-timeout-seconds 180
```

The cycle writes `bounded-delivery-state.json` and per-stage receipts beneath
the selected report directory. Any provider timeout/quota/mock response,
unvalidated QA output, out-of-scope diff/finding, token limit, missing
validation, or forbidden action becomes `attention-required`; it does not
continue to a repair or GitHub action.

After the single-task form has been verified, omit `--once` and provide an
ordered contract directory to start the resumable continuous form:

```bash
ai-team supervise /home/eden/projects/CelebrateDeal \
  --provider auto --bounded-delivery \
  --task-contract-dir /home/eden/.local/share/ai-team/CelebrateDeal/contracts \
  --execute --github-execute --auto-merge \
  --allow-unreviewed-development-merge \
  --interval-minutes 15 --max-iterations 3 --max-repair-attempts 2 \
  --max-token-usage 180000 --stage-timeout-seconds 300
```

For the allowlisted test-site workflow, add the explicit trusted development
profile and supply only a loopback development database URL:

```bash
export AI_TEAM_TEST_DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:54329/celebratedeal_dev'

ai-team supervise /home/eden/projects/CelebrateDeal \
  --provider auto --bounded-delivery --trusted-dev-autopilot \
  --task-contract-dir /home/eden/.local/share/ai-team/CelebrateDeal/contracts \
  --execute --github-execute --auto-merge \
  --allow-unreviewed-development-merge \
  --interval-minutes 15 --max-iterations 6 --max-repair-attempts 4 \
  --max-token-usage 360000 --stage-timeout-seconds 1200
```

`--trusted-dev-autopilot` is double opt-in: the CLI flag is required and the
exact project path must appear in `trusted_dev_autopilot.enabled_projects`.
It also requires `project.stage=development`, disposable worktrees, and all
production deploy/database/seed/destructive flags to remain `false`.

In this mode, provider quota/auth/network failure skips the expensive
lint/build/E2E suite. A successful edit runs only the fixed loopback test-DB
bootstrap, then deterministic validation. Failed validation may create a
clearly labelled WIP Git checkpoint so the next repair or process restart can
continue from the same clean HEAD. Next.js dependencies are copied once per
worktree and reused while the package manifest/lock fingerprint is unchanged.
After a CI-green merge the clean disposable worktree is removed. Ctrl-C writes
a resumable `stopped` state instead of leaving a misleading `running` state.

The supervisor processes dependency-ready `*.json` contracts in filename order,
records each completed task SHA, exposes unmet `dependsOn` edges in
`blockedTasks`, and watches the directory for new work without rerunning an
unchanged contract. Unknown dependencies and dependency cycles stop fail closed.
This allows a complete Epic backlog—including blocked UI work—to remain visible
before its data and API prerequisites finish. Each successful task must retain its bounded-delivery
receipt and deterministic validation hash before the GitHub executor may push,
open a PR, wait for CI, merge, and fast-forward the clean primary worktree.
Provider timeouts and network failures are retried after the configured
interval. Repeated quota exhaustion uses a persisted exponential backoff that
starts at the configured interval and caps at six hours (or the configured
interval when it is longer). The next retry time and consecutive failure count
remain in supervisor state, so restarting the process cannot bypass the wait.
Invalid contracts, unsafe findings, other provider failures, failed CI,
publication evidence mismatches, or dirty primary state stop fail closed with
an `attention-required` state.

For an unattended desktop supervisor, run the zero-token watchdog once per
minute from a systemd user timer. It sends deduplicated Telegram and Windows notifications and
appends a local JSON-lines audit record when the same `attention-required`
signature is observed three times, three Engineer receipts contain the same
stop reason, the service restarts three times between checks, or the supervisor
state has not changed for 25 minutes. It also detects five consecutive healthy
heartbeats where the queue is empty but a previously created PM task is still
cached. That idle-loop repair invalidates only the bounded PM cache, restarts
the Supervisor, and forces a same-revision project rescan without calling an AI
repair model:

It also reports each newly deferred five-cycle repair, newly queued human
release review, and provider backoff transition. These lifecycle alerts are
informational and never start another automatic-repair loop.

```bash
ai-team watchdog \
  --supervisor-state ~/.local/state/ai-team/CelebrateDeal/continuous-bounded-state.json \
  --watchdog-state ~/.local/state/ai-team/CelebrateDeal/watchdog-state.json \
  --alert-log ~/.local/state/ai-team/CelebrateDeal/watchdog-alerts.log \
  --report-dir ~/.local/share/ai-team/CelebrateDeal/reports \
  --service celebratedeal-ai-team-supervisor.service \
  --repeat-count 3 --restart-count 3 --idle-count 5 \
  --stale-minutes 25 --cooldown-minutes 30 \
  --auto-repair \
  --project ~/projects/CelebrateDeal \
  --contract-dir ~/.local/share/ai-team/CelebrateDeal/contracts \
  --repair-backup-dir ~/.local/state/ai-team/CelebrateDeal/repair-backups \
  --max-auto-repair-attempts 1 \
  --ai-repair \
  --orchestrator-project ~/projects/ai-team-orchestrator \
  --ai-repair-report-dir ~/.local/share/ai-team/CelebrateDeal/reports \
  --codex-executable ~/.local/bin/codex \
  --agy-executable ~/.local/bin/agy \
  --agy-qa-model "Gemini 3.1 Pro (High)" \
  --max-ai-repair-cycles 5
```

With `--auto-repair`, the watchdog stops the Supervisor before every repair.
It deterministically restores project-profile validation commands for the
known malformed-contract failure, and performs one bounded clean restart for
a failed service or stale state. A deferred or completed autonomous task is
treated as terminal even when the Git revision did not change, so PM discovery
continues and rejects repeated terminal task IDs. Unknown failures remain
stopped with their evidence intact. With `--ai-repair`, the restart-loop path runs one High cycle
of Sol diagnosis, Terra repair, deterministic validation, independent
Antigravity QA, and Sol review. Rejected results are replanned with Sol XHigh
and repaired with Terra XHigh for at most five total cycles. A fifth rejection
is written to the repair report, the task SHA is added to `deferredTaskShas`,
and the Supervisor resumes with the next dependency-ready task instead of
blocking the whole queue.

Use `ai-team status-zh /path/to/project` to see the active repair phase and the
last five cycle outcomes, including AGY findings, Sol review findings, and
deferred-task reasons, in Traditional Chinese.

An enabled human-only external QA policy is recorded separately in
`releaseReviewTasks` after code, publication, and deterministic checks pass.
The task is counted as development-complete so its dependants and the next PM
task can continue, while the human release attestation remains visible and is
never falsely reported as passed by an autonomous model. Configure
`external_qa.trigger_paths` with project-relative path prefixes to apply that
review only when the trusted delivery result changed a sensitive integration.
An enabled policy with no trigger paths retains the conservative project-wide
behavior; missing or malformed changed-file evidence also triggers review.

Telegram delivery is optional and uses the official Bot API `sendMessage`
endpoint. Copy the orchestrator's tracked `.env.example` to its Git-ignored
`.env.local`, keep that local file mode `0600`, and configure only the Watchdog
service to load it:

```dotenv
AI_TEAM_TELEGRAM_BOT_TOKEN=
AI_TEAM_TELEGRAM_CHAT_ID=
# Optional forum topic ID
AI_TEAM_TELEGRAM_THREAD_ID=
```

Create the bot with `@BotFather`, send the bot `/start` once, then fill in its
token and the destination chat ID. The Bot API requires the user to contact a
bot before it can send a private message. Telegram delivery failures are
fail-safe and never stop automatic repair. Existing alert signatures and
cooldowns prevent the bot from repeating the same event every minute.

Use `--test-notification` to verify both configured notification paths without
reading the supervisor state or calling an AI provider. The JSON result reports
`windowsDelivered`, `telegramConfigured`, and `telegramDelivered` separately.

`changePolicy` is deny-by-default. `schemaChanges` is required before
`prisma/schema.prisma` may be in scope; `migrationArtifacts` additionally permits
tracked migration files. Normal bounded delivery never runs them; the explicit
trusted-development profile may apply them only through the fixed
`npm run db:migrate:deploy` command against an operator-supplied loopback
database whose name ends in `_dev` or `_test`. `apiContractChanges`
permits an Architect to attest a bounded API contract change; and `fixtureData`
records authorization for deterministic non-production test/demo data. Actual
migration or seed execution, deployment, real payment, secret access, and
destructive operations remain prohibited regardless of these flags.

Quota responses are returned after the first provider-native attempt. They are
not retried immediately inside the same stage; the continuous supervisor owns
the timed retry so a quota window cannot trigger repeated agent runs and builds
before its configured interval elapses.

Retries recover only dependency-ordered stage checkpoints whose provider,
structured validation, task SHA, secondary review, token usage, commit, and
receipt evidence still agree. A completed Engineer checkpoint additionally
requires the same clean disposable worktree at the attested commit; a partial
Engineer retry may reuse a dirty worktree only when every changed path remains
inside the trusted task scope. Missing, modified, cross-repository, or
out-of-scope recovery state stops fail closed instead of rerunning later stages
against unverified code. Token usage and bounded repair evidence remain
cumulative across process restarts.

The optional review waiver is intentionally narrow: it is accepted only when
the project contract declares `project.stage: development`. Omitting
`--allow-unreviewed-development-merge` preserves the existing approved-review
gate. Continuous bounded delivery requires both `--github-execute` and
`--auto-merge`; it refuses to run indefinitely while leaving completed changes
unpublished or unverifiable.

### Staging-only external operations

Database migration, seed, and deployment remain forbidden in bounded delivery.
They are not made safe by changing `safety.allow_*` to `true`: those legacy
flags remain production-capable and must stay `false`.  Instead, an operator
may use the separate deterministic `staging-operations` command for one
explicit staging/Preview contract. It does not invoke an LLM, shell, Git
write, or arbitrary contract command.

The only executable operations are fixed in the orchestrator source:

- `database-migration` → `npm run db:migrate:deploy`
- `database-seed` → `npm run db:seed`
- `preview-deploy` → `vercel deploy --yes` (never `--prod`)

The project must opt in separately while preserving every legacy production
flag as `false`. Database actions also require an operator-provided SHA-256
fingerprint of the staging `DATABASE_URL`; the executor compares it in memory
without printing the URL or its value. Preview deployment requires the tracked
attestation environment variable to equal `preview`.

```yaml
safety:
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false

staging_operations:
  enabled: true
  environment: staging
  database_url_env: DATABASE_URL
  database_url_sha256: "<SHA-256 of the approved staging DATABASE_URL>"
  allow_migration: true
  allow_seed: true
  allow_preview_deploy: true
  preview_environment_variable: VERCEL_ENV
```

The contract cannot supply command text and must identify only `staging` and
`preview`:

```json
{
  "schema": "ai-team-staging-operations/v1",
  "id": "celebratedeal-preview-smoke",
  "title": "Apply the approved staging schema and test fixture",
  "source": {"kind": "trusted-contract", "reference": "staging-approval-2026-07-16"},
  "target": {"environment": "staging", "deployment": "preview"},
  "operations": ["database-migration", "database-seed", "preview-deploy"]
}
```

Validate first, then execute only after the receipt confirms the target guard:

```bash
ai-team staging-operations /home/eden/projects/CelebrateDeal \
  --contract /tmp/celebratedeal-staging.json

ai-team staging-operations /home/eden/projects/CelebrateDeal \
  --contract /tmp/celebratedeal-staging.json --execute
```

Each invocation writes a redacted receipt containing the contract SHA, target,
fixed-command digests, target validation, and stable stop reason. A malformed
contract, production target, missing/mismatched database fingerprint, missing
Preview attestation, unapproved operation, command failure, or any legacy
production safety flag fails closed before a later operation can run. The
executor also requires a clean product worktree before it starts and stops if
an operation changes tracked project files; it never resets or cleans those
changes. `database-seed` always forces `SEED_MODE=demo`, so an inherited
production-bootstrap seed setting cannot be reused.
