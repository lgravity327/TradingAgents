from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from jsonschema import ValidationError, validate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, messages_to_dict
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field

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


def _tool_response_schema(tools: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    variants = []
    for tool_spec in tools:
        function = tool_spec["function"]
        variants.append(
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "name": {"const": function["name"]},
                    "args": function.get("parameters", {"type": "object"}),
                },
                "required": ["id", "name", "args"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "properties": {
            "mode": {"enum": ["final", "tool_calls"]},
            "content": {"type": "string"},
            "tool_calls": {"type": "array", "items": {"oneOf": variants}},
        },
        "required": ["mode", "content", "tool_calls"],
        "additionalProperties": False,
    }


def _validate_output(value: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate(value, schema)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(
            f"Codex output failed schema validation at {path}: {exc.message}"
        ) from exc


def _json_schema(schema: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    if isinstance(schema, dict):
        if schema.get("type") == "function":
            return schema["function"]["parameters"]
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    raise TypeError("structured schema must be a Pydantic model or JSON schema dict")


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
        if self.structured_schema is not None:
            schema = _json_schema(self.structured_schema)
            prompt = _prompt(messages)
        elif self.bound_tools:
            schema = _tool_response_schema(self.bound_tools)
            prompt = _prompt(messages, self.bound_tools)
        else:
            schema = PLAIN_SCHEMA
            prompt = _prompt(messages)
        result = self.runner.invoke(prompt, schema, self.model_name)
        _validate_output(result.value, schema)
        if self.structured_schema is not None:
            message = AIMessage(
                content=json.dumps(result.value),
                usage_metadata=result.usage,
            )
        elif self.bound_tools:
            if result.value["mode"] == "final":
                if result.value["tool_calls"]:
                    raise ValueError("codex_chatgpt returned final mode with tool calls")
                message = AIMessage(
                    content=result.value["content"],
                    usage_metadata=result.usage,
                )
            else:
                calls = [
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": call["id"],
                        "type": "tool_call",
                    }
                    for call in result.value["tool_calls"]
                ]
                if not calls:
                    raise ValueError(
                        "codex_chatgpt returned tool_calls mode without calls"
                    )
                message = AIMessage(
                    content=result.value["content"],
                    tool_calls=calls,
                    usage_metadata=result.usage,
                )
        else:
            message = AIMessage(
                content=result.value["content"],
                usage_metadata=result.usage,
            )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "provider": "codex_chatgpt",
                "elapsed_seconds": result.elapsed_seconds,
            },
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        if tool_choice not in (None, "auto"):
            raise ValueError("codex_chatgpt supports only automatic tool selection")
        if kwargs:
            raise ValueError(f"Unsupported bind_tools arguments: {sorted(kwargs)}")
        converted = tuple(convert_to_openai_tool(tool) for tool in tools)
        return self.model_copy(update={"bound_tools": converted})

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ):
        if include_raw:
            raise NotImplementedError("codex_chatgpt does not support include_raw")
        if kwargs:
            raise ValueError(
                f"Unsupported structured-output arguments: {sorted(kwargs)}"
            )
        if self.bound_tools:
            raise ValueError("bind_tools and with_structured_output cannot be combined")
        _json_schema(schema)
        bound = self.model_copy(update={"structured_schema": schema})
        parser = (
            PydanticOutputParser(pydantic_object=schema)
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else JsonOutputParser()
        )
        return bound | parser


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
