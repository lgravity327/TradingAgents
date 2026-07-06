from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from copy import deepcopy
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
    names = [tool_spec["function"]["name"] for tool_spec in tools]
    return {
        "type": "object",
        "properties": {
            "mode": {"enum": ["final", "tool_calls"]},
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "name": {"enum": names},
                        "arguments": {"type": "string"},
                    },
                    "required": ["id", "name", "arguments"],
                    "additionalProperties": False,
                },
            },
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


def _strict_json_schema(value: Any) -> Any:
    root = deepcopy(value)

    def resolve_ref(ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ValueError(f"Only local JSON schema references are supported: {ref}")
        resolved = root
        for part in ref[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        return deepcopy(resolved)

    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized = {key: normalize(item) for key, item in node.items()}
        ref = normalized.get("$ref")
        if isinstance(ref, str) and len(normalized) > 1:
            normalized = {**resolve_ref(ref), **normalized}
            normalized.pop("$ref")
            return normalize(normalized)

        if normalized.get("type") == "object":
            normalized["additionalProperties"] = False
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties)
        if normalized.get("default") is None:
            normalized.pop("default", None)
        return normalized

    return normalize(root)


def _json_schema(schema: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    if isinstance(schema, dict):
        if schema.get("type") == "function":
            schema = schema["function"]["parameters"]
        return _strict_json_schema(schema)
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError("structured schema must be a Pydantic model or JSON schema dict")
    return _strict_json_schema(schema.model_json_schema())


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
        schema = _strict_json_schema(schema)
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
                tools_by_name = {
                    tool["function"]["name"]: tool["function"]
                    for tool in self.bound_tools
                }
                calls = []
                for call in result.value["tool_calls"]:
                    try:
                        args = json.loads(call["arguments"])
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "codex_chatgpt returned invalid tool arguments JSON"
                        ) from exc
                    function = tools_by_name[call["name"]]
                    _validate_output(args, _strict_json_schema(function["parameters"]))
                    calls.append(
                        {
                            "name": call["name"],
                            "args": args,
                            "id": call["id"],
                            "type": "tool_call",
                        }
                    )
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
