from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from .base_client import BaseLLMClient
from .codex_cli_runner import CodexCliRunner

PLAIN_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
    "required": ["content"],
    "additionalProperties": False,
}


def _prompt(
    messages: list[BaseMessage],
    tools: tuple[dict[str, Any], ...] = (),
) -> str:
    envelope = {"messages": messages_to_dict(messages), "tools": list(tools)}
    return (
        "Act only as a chat-completion engine. Do not use shell, files, web, MCP, "
        "or any Codex tool. Use only the supplied messages and tool schemas. "
        "Return only the JSON required by the output schema.\n\n"
        + json.dumps(envelope, ensure_ascii=False)
    )


class CodexChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "default"
    runner: Any = Field(exclude=True)
    bound_tools: tuple[dict[str, Any], ...] = Field(default_factory=tuple, exclude=True)
    structured_schema: Any | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "codex_chatgpt"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "provider": "codex_chatgpt"}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("codex_chatgpt does not support stop sequences")
        result = self.runner.invoke(
            _prompt(messages, self.bound_tools),
            PLAIN_SCHEMA,
            model=self.model_name,
        )
        message = AIMessage(content=result.value["content"], usage_metadata=result.usage)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "provider": "codex_chatgpt",
                "elapsed_seconds": result.elapsed_seconds,
            },
        )


class CodexChatGPTClient(BaseLLMClient):
    def get_llm(self) -> CodexChatModel:
        if self.base_url:
            raise ValueError("codex_chatgpt does not use backend_url; unset it")
        runner = CodexCliRunner(timeout_seconds=float(self.kwargs.get("timeout", 300)))
        runner.ensure_ready()
        return CodexChatModel(
            model_name=self.model,
            runner=runner,
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        return bool(self.model.strip())
