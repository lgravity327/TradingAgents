from unittest import mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.codex_chatgpt_client import (
    CodexChatGPTClient,
    CodexChatModel,
)
from tradingagents.llm_clients.codex_cli_runner import CodexInvocationResult


@pytest.mark.unit
def test_plain_invoke_returns_ai_message_and_usage():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={"content": "analysis"},
        usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        elapsed_seconds=0.2,
    )
    llm = CodexChatModel(model_name="default", runner=runner)

    result = llm.invoke(
        [SystemMessage(content="system"), HumanMessage(content="question")]
    )

    assert result.content == "analysis"
    assert result.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    prompt = runner.invoke.call_args.args[0]
    assert '"type": "system"' in prompt
    assert '"type": "human"' in prompt
    assert "Do not use shell, files, web, MCP" in prompt


@pytest.mark.unit
def test_plain_invoke_rejects_stop_sequences():
    llm = CodexChatModel(model_name="default", runner=mock.Mock())
    with pytest.raises(ValueError, match="stop sequences"):
        llm.invoke("question", stop=["STOP"])


@pytest.mark.unit
def test_client_rejects_backend_url():
    client = CodexChatGPTClient("default", base_url="https://api.openai.com/v1")
    with pytest.raises(ValueError, match="does not use backend_url"):
        client.get_llm()


@pytest.mark.unit
def test_client_checks_readiness_before_returning_model():
    with mock.patch(
        "tradingagents.llm_clients.codex_chatgpt_client.CodexCliRunner"
    ) as runner_type:
        llm = CodexChatGPTClient("default", timeout=45).get_llm()

    runner_type.assert_called_once_with(timeout_seconds=45.0)
    runner_type.return_value.ensure_ready.assert_called_once_with()
    assert llm._llm_type == "codex_chatgpt"


@tool
def quote(symbol: str) -> str:
    """Return a deterministic quote fixture."""
    return f"{symbol}=100"


@pytest.mark.unit
def test_bind_tools_builds_valid_langchain_tool_call():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={
            "mode": "tool_calls",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "quote", "args": {"symbol": "AAPL"}}
            ],
        },
        usage=None,
        elapsed_seconds=0.1,
    )

    result = CodexChatModel(model_name="default", runner=runner).bind_tools(
        [quote]
    ).invoke("Get AAPL")

    assert result.tool_calls == [
        {
            "name": "quote",
            "args": {"symbol": "AAPL"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    prompt = runner.invoke.call_args.args[0]
    assert '"name": "quote"' in prompt


@pytest.mark.unit
def test_bind_tools_rejects_unknown_tool():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={
            "mode": "tool_calls",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "delete_all", "args": {}}],
        },
        usage=None,
        elapsed_seconds=0.1,
    )
    bound = CodexChatModel(model_name="default", runner=runner).bind_tools([quote])

    with pytest.raises(ValueError, match="schema validation"):
        bound.invoke("Get AAPL")


@pytest.mark.unit
def test_bind_tools_rejects_invalid_arguments():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={
            "mode": "tool_calls",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "quote", "args": {"symbol": 123}}
            ],
        },
        usage=None,
        elapsed_seconds=0.1,
    )
    bound = CodexChatModel(model_name="default", runner=runner).bind_tools([quote])

    with pytest.raises(ValueError, match="schema validation"):
        bound.invoke("Get AAPL")


@pytest.mark.unit
def test_bind_tools_rejects_final_mode_with_tool_calls():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={
            "mode": "final",
            "content": "done",
            "tool_calls": [
                {"id": "call_1", "name": "quote", "args": {"symbol": "AAPL"}}
            ],
        },
        usage=None,
        elapsed_seconds=0.1,
    )
    bound = CodexChatModel(model_name="default", runner=runner).bind_tools([quote])

    with pytest.raises(ValueError, match="final mode"):
        bound.invoke("Get AAPL")


class Decision(BaseModel):
    rating: str
    confidence: float


@pytest.mark.unit
def test_with_structured_output_returns_pydantic_model():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={"rating": "Hold", "confidence": 0.7},
        usage=None,
        elapsed_seconds=0.1,
    )
    llm = CodexChatModel(model_name="default", runner=runner)

    result = llm.with_structured_output(Decision).invoke("Decide")

    assert result == Decision(rating="Hold", confidence=0.7)
    schema = runner.invoke.call_args.args[1]
    assert schema["properties"]["rating"]["type"] == "string"


@pytest.mark.unit
def test_with_structured_output_accepts_json_schema_dict():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={"rating": "Buy"},
        usage=None,
        elapsed_seconds=0.1,
    )
    schema = {
        "type": "object",
        "properties": {"rating": {"type": "string"}},
        "required": ["rating"],
        "additionalProperties": False,
    }

    result = CodexChatModel(
        model_name="default", runner=runner
    ).with_structured_output(schema).invoke("Decide")

    assert result == {"rating": "Buy"}


@pytest.mark.unit
def test_structured_output_rejects_include_raw():
    llm = CodexChatModel(model_name="default", runner=mock.Mock())
    with pytest.raises(NotImplementedError, match="include_raw"):
        llm.with_structured_output(Decision, include_raw=True)
