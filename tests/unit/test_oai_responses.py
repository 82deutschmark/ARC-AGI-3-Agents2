"""Tests for the OpenAI Responses adapter helpers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

import pytest

from agents.templates import _oai_responses as adapter


def test_parse_output_prefers_output_text_and_extracts_tool_calls() -> None:
    result = {
        "id": "resp-123",
        "output_text": "Hello there",
        "output": [
            {
                "type": "tool_call",
                "tool_call": {
                    "id": "call-1",
                    "name": "RESET",
                    "arguments": "{}",
                },
            }
        ],
        "usage": {"output_tokens": 42},
    }

    parsed = adapter.parse_output(result)

    assert parsed["text"] == "Hello there"
    assert parsed["response_id"] == "resp-123"
    assert parsed["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "RESET", "arguments": "{}"},
        }
    ]
    assert parsed["raw_output"] == result["output"]


def test_parse_output_falls_back_to_output_array_and_fallback_text() -> None:
    result = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Text from message content",
                    }
                ],
            }
        ],
        "output_reasoning": {},
        "usage": {},
    }

    parsed = adapter.parse_output(result)
    assert parsed["text"] == "Text from message content"

    parsed_with_fallback = adapter.parse_output({}, fallback_text="fallback")
    assert parsed_with_fallback["text"] == "fallback"


def test_parse_output_includes_structured_payload() -> None:
    result = {
        "output_parsed": [{"name": "RESET", "reason": "Observation"}],
    }

    parsed = adapter.parse_output(result)
    assert parsed["parsed"] == [{"name": "RESET", "reason": "Observation"}]


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected"),
    [
        ("RESPONSES_STREAMING", "0", False),
        ("RESPONSES_STREAMING", "1", True),
        ("RESPONSES_STORE", "0", False),
        ("RESPONSES_STORE", "1", True),
    ],
)
def test_boolean_env_flags(monkeypatch: pytest.MonkeyPatch, env_name: str, env_value: str, expected: bool) -> None:
    monkeypatch.setenv(env_name, env_value)
    module = importlib.reload(adapter)

    if env_name == "RESPONSES_STREAMING":
        assert module.should_stream() is expected
    else:
        assert module.should_store() is expected

    # Clean up by deleting the env and reloading to restore defaults for other tests
    monkeypatch.delenv(env_name, raising=False)
    importlib.reload(adapter)


def test_build_tools_preserves_function_metadata() -> None:
    functions = [
        {
            "name": "RESET",
            "description": "Reset the game",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]

    tools = adapter.build_tools(functions)
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "RESET",
                "description": "Reset the game",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        }
    ]
