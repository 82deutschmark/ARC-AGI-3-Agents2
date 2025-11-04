"""
OpenAI Responses API adapter utilities for template agents.

Created: 2025-11-03
Last modified by: Cascade (AI assistant)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Mapping, Sequence

from openai import OpenAI

logger = logging.getLogger(__name__)

_RESPONSES_STREAMING_DEFAULT = os.getenv("RESPONSES_STREAMING", "1")
_RESPONSES_REASONING_DEFAULT = os.getenv("RESPONSES_REASONING", "auto").lower()
_RESPONSES_STORE_DEFAULT = os.getenv("RESPONSES_STORE", "1")

_FALSEY = {"0", "false", "no", "off", ""}


def should_stream() -> bool:
    """Return True when streaming should be enabled by default."""

    return _RESPONSES_STREAMING_DEFAULT.strip().lower() not in _FALSEY


def should_store() -> bool:
    """Return True when server-side response storage should be enabled."""

    return _RESPONSES_STORE_DEFAULT.strip().lower() not in _FALSEY


def build_input(
    messages: Sequence[Mapping[str, Any]] | None = None,
    *,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Convert chat-style messages into Responses `input` format."""

    if messages and text is not None:
        raise ValueError("Provide either `messages` or `text`, not both.")

    if text is not None:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        ]

    normalized: list[dict[str, Any]] = []
    for message in messages or []:
        role = message.get("role", "user")
        entry: dict[str, Any] = {"role": role}

        if "name" in message:
            entry["name"] = message["name"]
        if "tool_call_id" in message:
            entry["tool_call_id"] = message["tool_call_id"]

        content = message.get("content")
        entry["content"] = _normalize_content(content)

        # Preserve tool call metadata for assistant messages if present so that
        # downstream callers can reason about historical tool calls.
        if "tool_calls" in message:
            entry["tool_calls"] = message["tool_calls"]
        if "function_call" in message:
            entry["function_call"] = message["function_call"]

        normalized.append(entry)

    return normalized


def build_tools(functions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions `functions` to Responses `tools` payload."""

    tools: list[dict[str, Any]] = []
    for function in functions:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "parameters": function.get("parameters", {}),
                    "strict": function.get("strict", True),
                },
            }
        )
    return tools


def create(
    client: OpenAI,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]] | None = None,
    input: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any | None = None,
    reasoning_effort: str | None = None,
    store: bool | None = None,
    previous_response_id: str | None = None,
    max_output_tokens: int | None = None,
    text: Mapping[str, Any] | None = None,
    include: Iterable[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Invoke the non-streaming Responses API."""

    payload = _build_payload(
        model=model,
        messages=messages,
        input=input,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        store=store,
        previous_response_id=previous_response_id,
        max_output_tokens=max_output_tokens,
        text=text,
        include=include,
        **kwargs,
    )
    logger.debug(
        "Responses.create payload keys: %s",
        sorted(k for k in payload.keys() if k != "input"),
    )
    return client.responses.create(**payload)


def stream(
    client: OpenAI,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]] | None = None,
    input: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any | None = None,
    reasoning_effort: str | None = None,
    store: bool | None = None,
    previous_response_id: str | None = None,
    max_output_tokens: int | None = None,
    text: Mapping[str, Any] | None = None,
    include: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, str, str, list[Any]]:
    """Invoke the streaming Responses API and aggregate deltas."""

    payload = _build_payload(
        model=model,
        messages=messages,
        input=input,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        store=store,
        previous_response_id=previous_response_id,
        max_output_tokens=max_output_tokens,
        text=text,
        include=include,
        **kwargs,
    )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    events: list[Any] = []

    with client.responses.stream(**payload) as stream_ctx:
        for event in stream_ctx:
            events.append(event)
            event_type = _get_type(event)
            if event_type == "response.output_text.delta":
                delta = _get_attr(event, "delta")
                if delta:
                    text_parts.append(str(delta))
            elif event_type in {
                "response.reasoning.delta",
                "response.reasoning_summary.delta",
                "response.reasoning_summary_text.delta",
            }:
                delta = _get_attr(event, "delta")
                if isinstance(delta, Mapping):
                    reasoning_delta = delta.get("text") or delta.get("summary")
                    if reasoning_delta:
                        reasoning_parts.append(str(reasoning_delta))
                elif delta:
                    reasoning_parts.append(str(delta))

        final_response = stream_ctx.get_final_response()

    return final_response, "".join(text_parts), "".join(reasoning_parts), events


def parse_output(
    result: Any,
    *,
    fallback_text: str | None = None,
    fallback_reasoning: str | None = None,
) -> dict[str, Any]:
    """Parse the Responses result into text, tool calls, and metadata."""

    output_text = _coalesce(
        _get_nested(result, "output_text"),
        _extract_output_text(_get_nested(result, "output")),
        fallback_text,
    )

    tool_calls = _extract_tool_calls(_get_nested(result, "output"))

    output_reasoning = _get_nested(result, "output_reasoning") or {}
    if fallback_reasoning and not output_reasoning:
        output_reasoning = {"summary": fallback_reasoning}

    usage = _get_nested(result, "usage") or {}
    response_id = _get_nested(result, "id")
    structured = _get_nested(result, "output_parsed")

    return {
        "text": output_text,
        "tool_calls": tool_calls,
        "usage": usage,
        "response_id": response_id,
        "output_reasoning": output_reasoning,
        "raw_output": _get_nested(result, "output"),
        "parsed": structured,
    }


def _build_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]] | None = None,
    input: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any | None = None,
    reasoning_effort: str | None = None,
    store: bool | None = None,
    previous_response_id: str | None = None,
    max_output_tokens: int | None = None,
    text: Mapping[str, Any] | None = None,
    include: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model}

    if input is not None:
        payload["input"] = list(input)
    else:
        payload["input"] = build_input(messages)

    if tools:
        payload["tools"] = list(tools)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    reasoning_config, default_text_config = _build_reasoning_config(reasoning_effort)
    if reasoning_config:
        payload["reasoning"] = reasoning_config

    text_config = text or default_text_config
    if text_config:
        payload["text"] = text_config

    payload["store"] = should_store() if store is None else store

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if include is not None:
        payload["include"] = list(include)

    payload.update(kwargs)
    return payload


def _build_reasoning_config(
    effort: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mode = _RESPONSES_REASONING_DEFAULT

    if mode == "off":
        return None, None

    summary = "auto" if mode in {"auto", "detailed"} else mode

    if effort is None and mode == "auto":
        # Respect default behaviour: only enable reasoning when the agent opts in.
        return None, None

    reasoning: dict[str, Any] = {"summary": summary}
    if effort:
        reasoning["effort"] = effort

    text_config: dict[str, Any] | None = {"verbosity": "high"}

    return reasoning, text_config


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, Mapping) and "type" in item:
                normalized.append(dict(item))
            elif isinstance(item, str):
                normalized.append({"type": "text", "text": item})
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized

    if isinstance(content, Mapping) and "type" in content:
        return [dict(content)]

    return [{"type": "text", "text": str(content)}]


def _extract_output_text(output: Any) -> str | None:
    if not output:
        return None

    texts: list[str] = []
    if isinstance(output, Mapping):
        output = output.get("data") or output.get("items") or []

    for item in output or []:
        item_type = _get_nested(item, "type")
        if item_type == "output_text":
            text_val = _coalesce(
                _get_nested(item, "text"),
                _get_nested(item, "output_text"),
            )
            if text_val:
                texts.append(str(text_val))
        elif item_type == "message":
            content_items = _get_nested(item, "content") or []
            for content in content_items:
                if _get_nested(content, "type") in {"output_text", "text"}:
                    text_val = _coalesce(
                        _get_nested(content, "text"),
                        _get_nested(content, "output_text"),
                    )
                    if text_val:
                        texts.append(str(text_val))

    return "".join(texts) if texts else None


def _extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    if not output:
        return []

    tool_calls: list[dict[str, Any]] = []

    if isinstance(output, Mapping):
        output = output.get("data") or output.get("items") or []

    for item in output or []:
        if _get_nested(item, "type") != "tool_call":
            continue

        tool_call = _get_nested(item, "tool_call") or {}
        call_id = _coalesce(_get_nested(tool_call, "id"), _get_nested(item, "id"))
        function_name = _get_nested(tool_call, "name")
        arguments = _get_nested(tool_call, "arguments")

        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else str(arguments or "{}"),
                },
            }
        )

    return tool_calls


def _get_type(event: Any) -> str | None:
    return _get_attr(event, "type")


def _get_attr(obj: Any, attribute: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(attribute)
    return getattr(obj, attribute, None)


def _get_nested(obj: Any, *keys: str) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


__all__ = [
    "build_input",
    "build_tools",
    "create",
    "parse_output",
    "should_stream",
    "should_store",
    "stream",
]
