# Migration Plan: Chat Completions ➝ Responses API

Author: Codex CLI
Date: 2025-11-04

This document outlines a precise, low-risk migration of our OpenAI usage from Chat Completions to the Responses API, leveraging the reference materials in this repo’s `docs/` folder:

- `docs/ResponsesAPI.md`
- `docs/OpenAI_Responses_API_Streaming_Implementation.md`
- `docs/ResponsesMigration.md`
- `docs/Responses_API_Chain_Storage_Analysis.md`
- `docs/RESPONSES-API-OCT2025.md`

We keep behavior parity for agents and enable future features (reasoning summaries, stateful chaining, structured outputs, and streaming event handling).

## Current Usage (to be migrated)

- OpenAI Chat Completions call sites:
  - `agents/templates/llm_agents.py:124`
  - `agents/templates/llm_agents.py:158`
  - `agents/templates/llm_agents.py:196`
  - `agents/templates/reasoning_agent.py:254`
  - `agents/templates/langgraph_functional_agent.py:93`
- LangChain wrapper (kept as-is for now):
  - `agents/templates/langgraph_thinking/llm.py:14`

Notes
- Templates use function calling and tool calling; both map to Responses API tool definitions.
- We already pin `openai==1.72.0` which supports `client.responses.create()` and streaming.

## Goals

- Replace Chat Completions with Responses for OpenAI-backed agents, preserving current game agent behavior and tool/function calling.
- Streaming-first: enable Responses streaming by default and fall back gracefully to non-streaming if unavailable.
- Add reasoning support where configured, plus response ID chaining.
- Keep compatibility for non-OpenAI providers (e.g., OpenRouter legacy Chat Completions) via an adapter layer.

## High-Level Strategy

1) Introduce a small OpenAI adapter that abstracts request/response shapes and provides both stream and non-stream modes.
2) Migrate direct OpenAI call sites to the adapter; use `client.responses.stream(...)` by default; auto‑fallback to `client.responses.create(...)` on errors/timeouts.
3) Parse Responses output consistently (prefer `output_text`, then `output_parsed`, fall back to scanning `output[]`).
4) Capture and persist `response.id` to enable chaining via `previous_response_id`.
5) Keep LangChain- and smolagents-based templates unchanged initially; convert later once upstream libraries expose Responses-native clients.

## Phased Plan

### Phase 0 — Preparations
- Add feature flags/envs (streaming ON by default):
  - `RESPONSES_STREAMING=1` (default enabled; disable with `0` if needed)
  - `RESPONSES_REASONING=auto|detailed|off` (controls reasoning summaries; default `auto` when agent sets `REASONING_EFFORT`)
  - `RESPONSES_STORE=1` (server-side storage for chaining; default enabled)
- Add a lightweight module `agents/templates/_oai_responses.py` (adapter) with:
  - `build_input(messages|text)` → Responses `input` array
  - `build_tools(functions)` → Responses `tools` format
  - `create(client, *, model, input, tools, tool_choice, reasoning, text, store, previous_response_id, max_output_tokens)`
  - `parse_output(result)` → `(output_text, tool_calls, usage, response_id, output_reasoning)`
  - `stream(client, payload)` → Python SDK event iterator; expose `get_final_response()`

### Phase 1 — Direct call sites (streaming‑first)
- Update the following to use the adapter with streaming default (`client.responses.stream(...)` with graceful fallback):
  - `agents/templates/llm_agents.py:124, 158, 196`
  - `agents/templates/reasoning_agent.py:254`
  - `agents/templates/langgraph_functional_agent.py:93`

Mapping details
- messages → input: use the existing `self.messages` structure and pass as `input`.
- functions/tools → tools: convert `build_functions()` into Responses `tools=[{type:'function', function:{...}}]` (we already have `build_tools()` with close shape; reuse it).
- reasoning:
  - If `REASONING_EFFORT` is set, set `reasoning: { summary: 'auto', effort: <effort> }` (per docs in `docs/ResponsesAPI.md`, `docs/RESPONSES-API-OCT2025.md`).
  - For GPT‑5 style models that support delta reasoning, also set `text: { verbosity: 'high' }` to emit reasoning deltas (see `docs/OpenAI_Responses_API_Streaming_Implementation.md`).
  - Otherwise omit reasoning/text for parity with older non-reasoning models.
- response parsing:
  - Prefer `result.output_text`.
  - Else `result.output_parsed` if we enabled structured output.
  - Else scan `result.output[]` for an item of type `message` and within it `content[]` entries of `type: 'output_text'`.
- tool/function-call parsing:
  - Responses returns tool calls as output items; normalize to the same structure our agents expect (first tool call only; log and ignore extras, as done today).
- token usage: read `usage.input_tokens`, `usage.output_tokens`, and `usage.output_tokens_details.reasoning_tokens` if present.
- persist `result.id` for potential chaining (store in agent instance or recorder metadata initially; full DB persistence optional here).

### Phase 2 — Streaming implementation (default path)
- Use `client.responses.stream(...)` as the primary execution path.
- Event handling per `docs/OpenAI_Responses_API_Streaming_Implementation.md`:
  - Handle: `response.reasoning_summary_text.delta`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.in_progress`, `response.completed`, `response.failed`.
  - In Python SDK, iterate events from the stream context and accumulate visible output + reasoning text.
  - After stream completion, call `get_final_response()` and parse with the same helper.
- Fallback: on errors/timeouts or when `RESPONSES_STREAMING=0`, use non‑streaming `client.responses.create(...)`.

### Phase 3 — Chaining and state
- When streaming or not, capture `response.id` and maintain on the agent instance.
- For multi-turn: pass `previous_response_id` to `client.responses.create()` if available and `RESPONSES_STORE=1`.
- Recorder/tracing: include `response_id`, reasoning summary (if present), and token split in traces.

### Phase 4 — Library wrappers and templates
- LangGraph template (`agents/templates/langgraph_functional_agent.py`):
  - Replace its internal `openai_client.chat.completions.create` with adapter call, keeping the return shape consistent for tool calls.
  - If strict typing becomes noisy (e.g., `ChatCompletionMessage`), relax types locally or introduce a minimal local type reflecting Responses’ tool-call item.
- LangChain (`agents/templates/langgraph_thinking/llm.py`):
  - Keep `ChatOpenAI` for now (library still maps to Chat Completions); later evaluate `langchain-openai` support for Responses.
- Smolagents templates: keep as-is unless upstream exposes a Responses-native backend.

### Phase 5 — Tests and validation
- Tests to update or add:
  - Unit: mock `/v1/responses` non-streaming responses, verify tool-call extraction and `output_text` parsing.
  - Unit: verify reasoning optionality and token accounting parsing paths.
  - Unit: verify `previous_response_id` is passed when a stored `response.id` exists.
  - Optional: streaming—mock event sequence and ensure aggregation + final `finalResponse()` parsing.
- Keep existing tests otherwise; prioritize minimal changes.

## Request/Response Mapping Cheatsheet

- Endpoint: `POST /v1/responses` (instead of `/v1/chat/completions`).
- Request keys:
  - `model`: unchanged.
  - `input`: replaces `messages` (array of `{role, content}` or a string).
  - `tools`: similar concept, different envelope (we already produce it via `build_tools()`).
  - `tool_choice`: remains available.
  - `reasoning`: `{ summary: 'auto'|'detailed', effort: 'minimal'|'low'|'medium'|'high' }` (model-dependent). For streamed reasoning on GPT‑5 style models, set this plus `text.verbosity='high'`.
  - `text`: supports `verbosity` and `format` (JSON schema for structured outputs; replaces `response_format`).
  - `store`: `true|false` (default true for most orgs), enables chaining.
  - `previous_response_id`: to continue chains.
  - `max_output_tokens`: ensure generous budgets when reasoning is enabled.
- Response keys:
  - `id`: capture and persist for chaining.
  - `output_text` (preferred), `output_parsed`, or `output[]` with `message.content[].type == 'output_text'`.
  - `output_reasoning.summary` (if present) for readable reasoning.
  - `usage.output_tokens_details.reasoning_tokens` for token split.

See: `docs/ResponsesAPI.md`, `docs/RESPONSES-API-OCT2025.md`, and `docs/OpenAI_Responses_API_Streaming_Implementation.md` for examples and caveats.

## File-by-File Action Items

- `agents/templates/llm_agents.py`
  - Replace `client.chat.completions.create(...)` calls with adapter to `responses.create(...)`.
  - Use `build_tools()` for Responses `tools` and preserve single-tool-call constraint.
  - Parse `output_text` and fall back to `output[]` scanning.
  - Respect `REASONING_EFFORT` by setting `reasoning.summary='auto'` and effort when configured.
  - Track `response.id` and token usage split.

- `agents/templates/reasoning_agent.py`
  - Replace `self.client.chat.completions.create(...)` with adapter to `responses.create(...)`.
  - Enable structured output via `text.format.json_schema` when using `ReasoningActionResponse` (replaces `response_format`).
  - Optional: enable reasoning summary for human-readable explanations in history.

- `agents/templates/langgraph_functional_agent.py`
  - Swap `openai_client.chat.completions.create(...)` for adapter call.
  - Adjust types minimally to represent tool-call output from Responses.

- `agents/templates/langgraph_thinking/llm.py`
  - Leave `ChatOpenAI` as-is initially; document that it still uses Chat Completions under-the-hood.

## Backward Compatibility and Providers

- For OpenRouter or providers that still require Chat Completions, keep a provider switch in the adapter:
  - If provider == OpenAI/xAI → Responses path.
  - Else → legacy Chat Completions path (unchanged behavior).
  - Default model map lives alongside adapter; ensure `--agent` choices remain valid.

## Observability/Tracing

- Recorder/tracing: add fields for `response_id`, `output_reasoning.summary` (if any), and `usage.output_tokens_details.reasoning_tokens`.
- Log raw responses for failures (as advised in `docs/ResponsesAPI.md`).

## Rollout Plan

1. Land adapter and migrate `llm_agents.py` non-streaming path.
2. Migrate `reasoning_agent.py` with optional structured output and reasoning.
3. Migrate `langgraph_functional_agent.py` minimal changes for tool calls.
4. Add streaming path behind flag; validate with local runs of representative agents.
5. Keep LangChain-based template as-is; revisit once upstream supports Responses.
6. Remove or gate legacy Chat Completions code after a trial window.

## Validation Checklist

- Agents still choose valid actions and respect single-tool-call rule.
- Tests pass with mocked `/v1/responses` payloads.
- Token accounting includes reasoning tokens when present.
- Optional streaming shows incremental output and reasoning deltas; final response parsing matches non-streaming.
- Chaining works when `store=true` and prior `response.id` is supplied.

## Risks and Mitigations

- Model differences (reasoning vs. non-reasoning):
  - Mitigate via feature flags; default to non-reasoning on unsupported models.
- Empty visible output due to reasoning token consumption:
  - Set generous `max_output_tokens`; inspect `usage.output_tokens_details.reasoning_tokens` on failures.
- Tool/function calling differences:
  - Normalize Responses tool items to current internal structure; retain single-tool-call enforcement.
- Library wrappers (LangChain, smolagents):
  - Keep legacy paths until upstream supports Responses, then migrate.

---

For detailed event handling and payload shapes, reference:
- `docs/OpenAI_Responses_API_Streaming_Implementation.md` (streaming)
- `docs/Responses_API_Chain_Storage_Analysis.md` (chaining/storage)
- `docs/ResponsesAPI.md` and `docs/RESPONSES-API-OCT2025.md` (reasoning and request/response caveats)

## Streaming-First Python Skeleton

Use this adapter pattern inside the new `agents/templates/_oai_responses.py`:

```python
from typing import Any, Iterable
from openai import OpenAI

def stream_response(client: OpenAI, **payload: Any):
    # Required: payload includes model, input, plus optional tools/tool_choice, reasoning, text, etc.
    with client.responses.stream(**payload) as stream:
        visible_text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for event in stream:
            et = getattr(event, "type", None) or event.get("type")
            if et == "response.output_text.delta":
                visible_text_parts.append(event.delta)
            elif et == "response.reasoning_summary_text.delta":
                # Some models emit reasoning summary deltas
                reasoning_parts.append(event.delta)
            # Optionally forward status events to recorder/tracer here

        final = stream.get_final_response()
        return final, "".join(visible_text_parts), "".join(reasoning_parts)

def create_response(client: OpenAI, **payload: Any):
    return client.responses.create(**payload)
```

Adapter defaults for agents (streaming enabled):
- Build `input` from existing `messages`.
- If `REASONING_EFFORT` is set, include `reasoning={"summary":"auto","effort":REASONING_EFFORT}` and `text={"verbosity":"high"}` for models that support delta reasoning.
- Set `store=True` and pass `previous_response_id` when present.
- Use generous `max_output_tokens` when reasoning is enabled.
