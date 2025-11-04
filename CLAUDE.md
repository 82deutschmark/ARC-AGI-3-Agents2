# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARC-AGI-3-Agents is a framework for building and running agents that play games in the ARC-AGI-3 competition. Agents are autonomous systems that interact with the ARC-AGI-3 API to play puzzle games, submit actions, and receive game states in return. The project supports multiple agent architectures (random, LLM-based, LangGraph, smolagents, reasoning agents) and includes recording/playback capabilities for agent sessions.

## Build, Test, and Development Commands

### Installation & Setup
```bash
# Install dependencies with uv
uv sync

# Install with AgentOps support
uv sync --extra agentops

# Set up pre-commit hooks for linting/formatting
pip install pre-commit
pre-commit install
```

### Running Agents
```bash
# Run a specific agent against a game
uv run main.py --agent=random --game=ls20

# Run an agent against all available games (from API)
uv run main.py --agent=llm

# Run a recorded session (playback)
uv run main.py --agent=game_id.agent_name.guid.recording.jsonl

# Add tags to scorecard for tracking
uv run main.py --agent=llm --game=ls20 --tags="Max Power!"
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output and specific markers
pytest -v -m unit
pytest -m integration

# Run a single test file
pytest tests/unit/test_swarm.py

# Run a specific test function
pytest tests/unit/test_swarm.py::test_open_scorecard
```

### Code Quality
```bash
# Lint with ruff
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Type checking with mypy (strict mode)
mypy agents main.py

# Run all pre-commit checks
pre-commit run -a
```

## Architecture Overview

### High-Level Design

The system follows an orchestration pattern where:
1. **main.py** - Entry point that loads environment, discovers available games via API, initializes AgentOps, and creates a Swarm
2. **Swarm** - Coordinates multiple agents across multiple games using threads
3. **Agent** (abstract base) - Each agent plays a single game by repeatedly choosing actions and receiving game state updates
4. **GameAction & FrameData** - Pydantic models representing game actions and state frames

### Key Components

**agents/agent.py**
- `Agent` (abstract): Base class for all agents. Subclasses implement `choose_action()` and `is_done()` to define behavior
- `Playback`: Special agent type that replays recorded sessions from .jsonl files
- Core loop: `main()` → `choose_action()` → `take_action()` → `append_frame()` until `is_done()` returns True
- Recording support: Automatically saves action/frame data to .jsonl files

**agents/swarm.py**
- Orchestrates multiple agents playing multiple games in parallel
- Creates threads for each game, manages scorecard lifecycle (open → play → close)
- Tags support: Adds metadata ("agent", agent name, or "playback" + guid) to scorecard for tracing

**agents/structs.py** (Pydantic models)
- `GameState` enum: NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER
- `GameAction` enum: RESET, ACTION1–ACTION7 (ACTION6 is complex with x/y coordinates; others are simple)
- `FrameData`: Game state snapshots with frame grid, score, action input, and available_actions list
- `Card` & `Scorecard`: Track play counts, scores, states, and action counts per game

**agents/tracing.py**
- AgentOps integration for monitoring agent execution
- `@trace_agent_session` decorator on Agent.main() automatically traces execution
- Gracefully falls back to no-op if AgentOps library is not installed
- Session URLs are logged for debugging

**Agent Templates** (in agents/templates/)
- `random_agent.py`: Chooses actions randomly  -- DO NOT USE!
- `llm_agents.py`: LLM-based agents (LLM, FastLLM, GuidedLLM, ReasoningLLM) using OpenAI API
- `langgraph_*`: LangGraph-based agents (Random, Functional, Thinking) for structured workflows
- `smolagents.py`: SmolAgents-based agents (SmolCodingAgent, SmolVisionAgent) for code/vision reasoning
- `reasoning_agent.py`: ReasoningAgent using OpenAI Responses API with extended thinking

**agents/recorder.py**
- Records agent actions and game frames to .jsonl files
- Playback support: Deserializes recorded sessions for replay

### Important Patterns

**Enum-based Actions**: `GameAction` enum members are callable objects with action_type (SimpleAction or ComplexAction) and optional reasoning. Use `GameAction.from_id()` or `GameAction.from_name()` to create from values.

**API Interaction**: All agents make HTTP requests via `requests.Session` to the ARC-AGI-3 API with `X-API-Key` header. Error handling is logging-based; check response JSON for "error" keys.

**Threading**: Swarm spawns one daemon thread per game; Agent.main() runs synchronously within its thread. Use signal handlers to gracefully clean up on SIGINT.

**Configuration**: Environment variables control network (SCHEME, HOST, PORT), API key (ARC_API_KEY), and optional observability (AGENTOPS_API_KEY).

## Testing Strategy

- Framework: **pytest** with **pytest-asyncio** (for async tests if needed)
- Location: `tests/` directory with structure `tests/unit/` and `tests/integration/`
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Fixtures in `tests/conftest.py` (e.g., mock API responses)
- Mocking: Use `requests-mock` for HTTP mocking to avoid hitting real APIs
- Run before PRs: Ensure `pytest -v -m unit` passes locally

## Coding Standards

- **Language**: Python 3.12+
- **Style**: Follow Ruff defaults; imports auto-sorted via `I` rule
- **Type Hints**: Required; strict mypy (disallow_untyped_defs=true) outside tests
- **Naming**:
  - Modules: `snake_case.py`
  - Classes: `CamelCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Formatting**: 4-space indentation, UTF-8 encoding

## Commit & PR Guidelines

- Concise, imperative commits (max ~72 chars): `fix(agent): handle reset race condition`
- Include reasoning and trade-offs in commit body
- PRs must include description, linked issues, test plan, and (when relevant) AgentOps URLs or logs
- CI checks: ruff, mypy, and pytest must pass before review

## Configuration & Secrets

- **Never commit .env files**: Use `.env.example` as template
- **Required**: `ARC_API_KEY` from https://three.arcprize.org/
- **Optional**: `AGENTOPS_API_KEY` from https://app.agentops.ai/
- **Network config**: `SCHEME`, `HOST`, `PORT` for API endpoint (defaults: http://localhost:8001)
- **Debug mode**: Set `DEBUG=True` in `.env` for verbose logging

## Agent Registration

Add new agents to `agents/__init__.py`:
- Subclass `Agent` and implement `is_done()` and `choose_action()`
- Agent is auto-registered via `Agent.__subclasses__()` introspection (class name lowercased becomes `--agent` choice)
- For special agents like ReasoningAgent, manually add to `AVAILABLE_AGENTS` dict

## Notes on Responses API & Extended Thinking

The codebase integrates OpenAI's Responses API (available in langgraph_thinking and reasoning_agent templates) for extended reasoning capabilities. Use `@trace_agent_session` and AgentOps integration to monitor thinking steps and intermediate outputs during agent execution.
