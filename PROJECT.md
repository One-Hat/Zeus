# Project: Forge, a multi-LLM pipeline that turns one sentence into a tested, staged web app

## Goal
Input: a one-sentence app idea (e.g. "a todo app with auth").
Output: a working app (backend + frontend), passing tests, deployed to a staging URL, plus a run report (pass/fail, cost, time, iterations).

## Hard rules
- Python 3.11, LangGraph for orchestration, LiteLLM for all model calls, Pydantic for every schema, Docker for sandboxing.
- Models are NEVER hardcoded. All role-to-model mapping lives in `config/routing.yaml`.
- Every agent returns structured output (Pydantic). No free-text handoffs between agents.
- All generated code runs ONLY inside a Docker sandbox, never on the host.
- Hard caps: max 3 fix loops per run, max $X per run (configurable), per-step timeout.
- Deploy goes to staging only. Production requires a manual approval step.
- Every run is logged to `runs/<id>/` (spec, prompts, outputs, logs, cost, result).

## Architecture
1. **Planner**: idea -> `spec.json` containing: API contract (OpenAPI), DB schema, file tree, task list split into frontend / backend / ml tasks, and acceptance tests (endpoint X returns Y).
2. **Workers** (run in parallel via LangGraph, each only sees the spec + its own tasks):
   - frontend worker
   - backend worker
   - ml worker (only if the spec has ML tasks)
   Each writes files into the sandbox workspace.
3. **Integrator**: builds the app, runs linter/type check, unit tests, boots the app, and runs smoke tests against the OpenAPI contract. Produces a `report.json` with real pass/fail signals and logs.
4. **Fixer**: receives failing logs + relevant files, patches code, returns to Integrator. Max 3 loops, then stop and report failure.
5. **Reviewer**: reads the final report and run log, scores the run, and records which prompt version / model / plan decisions caused failures.
6. **Deployer**: if all tests pass, deploys to staging (Docker + a free host like Fly.io or Render), returns URL. Production needs explicit human approval.

## Learning loop (replaces "backprop")
No gradients exist between LLM calls. Instead: maintain a score table per (role, model, prompt_version) tracking pass rate, cost, and latency. Router picks the best-scoring option using an epsilon-greedy bandit. The Reviewer's critiques can propose new prompt versions (TextGrad/DSPy style), which enter the bandit as new arms.

## Repo layout
forge/
  config/routing.yaml
  agents/{planner,frontend,backend,ml,integrator,fixer,reviewer,deployer}.py
  graph.py            # LangGraph definition
  sandbox/            # Docker runner
  schemas.py          # Pydantic models (Spec, Task, Report, Score)
  router.py           # bandit + model selection
  cli.py              # `forge run "idea"`
  benchmarks/prompts.json   # 20 test app ideas
  benchmarks/run.py   # compares baseline vs routed chain
  runs/

## Benchmark
Run the same 20 prompts through (a) a single strong model alone and (b) the routed multi-model chain. Compare pass rate, cost, wall time, fix loops. Output a markdown table. If the chain doesn't win, say so honestly in the README.

## Definition of done (v1)
`forge run "todo app with auth"` produces a working CRUD app, passing tests, with a staging URL and a run report, with no manual edits.
