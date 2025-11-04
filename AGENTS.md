# Repository Guidelines

## AgentOps
- Files you create or edit should ideally have a brief header explaining what they do, when they were created, and who they were modified by.
- We use the Responses API with OpenAI, and NOT the old Chat Completions API.


## Project Structure & Module Organization
- Core code in `agents/` (e.g., `agent.py`, `swarm.py`, `structs.py`, tracing and templates).
- Entry point `main.py` runs agents against ARC-AGI-3 API.
- Tests in `tests/`; config in `pytest.ini`.
- Config and metadata in `pyproject.toml`; lockfile `uv.lock` managed by `uv`.
- Environment files: `.env-example` (sample), `.env` (local, not committed).

## Build, Test, and Development Commands
- Install deps: `uv sync` (use extras: `uv sync --agentops`).
- Run an agent: `uv run main.py --agent=random --game=ls20`.
- Run all tests: `pytest -q`.
- Lint/format: `ruff check .` and `ruff format .`.
- Type check: `mypy agents main.py`.
- Pre-commit: `pre-commit install` then `pre-commit run -a`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indent, UTF-8.
- Follow Ruff defaults; imports auto-sorted (`I` rule enabled). Run `ruff check --fix` before PRs.
- Type hints required; `mypy` is strict and disallows untyped defs outside `tests/`.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` available.
- Put tests under `tests/` and name files `test_*.py`/`*_test.py`.
- Aim for meaningful coverage of agent behaviors and API interactions; use `requests-mock` for HTTP.
- Run locally: `pytest -q`; add repro seeds or fixtures where nondeterminism exists.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (max ~72 chars), body explaining why and notable trade-offs.
  - Examples: `fix(recorder): handle empty frames`, `feat(swarm): tag sessions`.
- PRs: include description, linked issues, test plan/output, and screenshots or logs when relevant (e.g., AgentOps URL).
- CI hygiene: ensure `ruff`, `mypy`, and `pytest` pass before requesting review.

## Security & Configuration Tips
- Never commit secrets. Use `.env`; required: `ARC_API_KEY`. Optional: `AGENTOPS_API_KEY`.
- Network targets configured via `SCHEME`, `HOST`, `PORT` env vars.
- When adding agents, register them in `agents/__init__.py` so `--agent` choices update automatically.

## Architecture Notes
- `main.py` orchestrates: loads env, discovers games via `ROOT_URL`, initializes AgentOps, runs `Swarm`.
- `Swarm` coordinates selected agent over game list; `Recorder` supports playback; tracing decorates key entrypoints.
