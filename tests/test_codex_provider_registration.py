from unittest import mock

import pytest

from cli.utils import _llm_provider_table, provider_default_url
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV, get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model


@pytest.mark.unit
def test_codex_provider_is_registered_as_keyless_without_url():
    assert "codex_chatgpt" in PROVIDER_API_KEY_ENV
    assert get_api_key_env("codex_chatgpt") is None
    assert provider_default_url("codex_chatgpt") is None
    assert any(row[1] == "codex_chatgpt" for row in _llm_provider_table())


@pytest.mark.unit
def test_codex_model_catalog_has_default_and_custom():
    assert get_model_options("codex_chatgpt", "quick") == [
        ("Codex account default", "default"),
        ("Custom model ID", "custom"),
    ]
    assert validate_model("codex_chatgpt", "future-codex-model") is True


@pytest.mark.unit
def test_factory_lazily_creates_codex_client():
    with mock.patch(
        "tradingagents.llm_clients.codex_chatgpt_client.CodexCliRunner"
    ) as runner_type:
        client = create_llm_client("codex_chatgpt", "default")
        llm = client.get_llm()

    runner_type.assert_called_once_with(timeout_seconds=300.0)
    runner_type.return_value.ensure_ready.assert_called_once_with()
    assert llm._llm_type == "codex_chatgpt"


@pytest.mark.unit
def test_graph_forwards_codex_timeout_only():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "codex_chatgpt",
        "codex_timeout_seconds": 45,
        "temperature": None,
    }
    assert graph._get_provider_kwargs() == {"timeout": 45.0}
