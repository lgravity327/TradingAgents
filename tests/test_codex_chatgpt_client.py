from unittest import mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

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
