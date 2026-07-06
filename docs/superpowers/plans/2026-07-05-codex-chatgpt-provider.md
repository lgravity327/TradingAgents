# Codex ChatGPT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 TradingAgents 增加 opt-in 的 `codex_chatgpt` provider，通过官方 Codex CLI 的 ChatGPT 登录消耗 Codex 额度，同时保留普通生成、LangGraph tool calls 和 Pydantic structured output。

**Architecture:** 用独立 runner 封装 `codex exec` 的认证预检、只读隔离、subprocess、JSONL、timeout 和错误分类；在其上实现 LangChain `BaseChatModel` adapter。现有 graph/agent 不改协议，只在 factory、model catalog、CLI 和配置层注册新 provider。任何 Codex 自身工具执行事件都 fail closed，绝不读取 OAuth token、Cookie 或私有 ChatGPT 接口。

**Tech Stack:** Python 3.10+、LangChain Core、LangGraph、Pydantic v2、jsonschema、pytest、ruff、官方 Codex CLI 0.142.5+

**Design spec:** `docs/superpowers/specs/2026-07-05-codex-chatgpt-provider-design.md`

---

## 文件结构

- Create: `tradingagents/llm_clients/codex_cli_runner.py` — 官方 Codex CLI 进程边界、认证/能力预检、输出解析和错误类型。
- Create: `tradingagents/llm_clients/codex_chatgpt_client.py` — LangChain chat model、tool binding、structured output 和 `BaseLLMClient` wrapper。
- Create: `tests/test_codex_cli_runner.py` — runner 的纯单元测试，不调用真实 Codex。
- Create: `tests/test_codex_chatgpt_client.py` — message/tool/schema adapter 单元测试。
- Create: `tests/test_codex_provider_registration.py` — factory/config/CLI 注册回归。
- Create: `scripts/smoke_codex_chatgpt.py` — 显式 opt-in 的真实协议与浅层分析 smoke。
- Modify: `pyproject.toml` — 增加直接依赖 `jsonschema>=4.23.0`。
- Modify: `tradingagents/llm_clients/factory.py` — lazy register `codex_chatgpt`。
- Modify: `tradingagents/llm_clients/api_key_env.py` — 将 `codex_chatgpt` 标记为 Codex 自管认证。
- Modify: `tradingagents/llm_clients/model_catalog.py` — 增加 account default/custom model 选项。
- Modify: `tradingagents/llm_clients/validators.py` — 接受 Codex 账户可用的自定义 model ID。
- Modify: `tradingagents/default_config.py` — 增加 `codex_timeout_seconds=300` 及 env override。
- Modify: `tradingagents/graph/trading_graph.py` — 仅向 Codex provider 传递 timeout。
- Modify: `cli/utils.py` — provider 下拉菜单和 keyless 流程。
- Modify: `README.md` — 登录、配置、额度边界、smoke 与风险说明。

## 执行前置条件

- [ ] **Step 0.1: 创建隔离环境并安装开发依赖**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Expected: editable install 成功，`pytest`、`ruff`、LangChain 和 Pydantic 可导入。

- [ ] **Step 0.2: 运行基线测试**

Run:

```bash
.venv/bin/python -m pytest tests/test_api_key_env.py tests/test_provider_registry.py tests/test_structured_agents.py tests/test_cli_env_skip.py -q
```

Expected: 全部 PASS。若基线失败，先记录并修复环境，不把既有失败归因于新 provider。

### Task 1: Codex CLI readiness 与错误契约

**Files:**
- Create: `tests/test_codex_cli_runner.py`
- Create: `tradingagents/llm_clients/codex_cli_runner.py`

- [ ] **Step 1.1: 写 readiness 失败测试**

在 `tests/test_codex_cli_runner.py` 写入：

```python
from types import SimpleNamespace
from unittest import mock

import pytest

from tradingagents.llm_clients.codex_cli_runner import (
    CodexCapabilityError,
    CodexCliRunner,
    CodexInvocationError,
    CodexIsolationError,
    CodexLoginError,
    CodexTimeoutError,
)


def completed(code: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


@pytest.mark.unit
def test_missing_codex_binary_is_actionable():
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(CodexCapabilityError, match="Codex CLI"):
            CodexCliRunner(executable="codex")


@pytest.mark.unit
def test_readiness_requires_chatgpt_login():
    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.run", return_value=completed(0, "Logged in using an API key")):
        with pytest.raises(CodexLoginError, match="ChatGPT"):
            runner.ensure_ready()


@pytest.mark.unit
def test_readiness_requires_exec_flags():
    runner = CodexCliRunner(executable="/opt/codex")
    responses = [
        completed(0, "Logged in using ChatGPT"),
        completed(0, "Usage: codex exec --json --ephemeral"),
    ]
    with mock.patch("subprocess.run", side_effect=responses):
        with pytest.raises(CodexCapabilityError, match="--output-schema"):
            runner.ensure_ready()
```

- [ ] **Step 1.2: 验证测试失败**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_cli_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError: tradingagents.llm_clients.codex_cli_runner`。

- [ ] **Step 1.3: 实现错误类型、环境白名单和 readiness**

创建 `tradingagents/llm_clients/codex_cli_runner.py`，先实现：

```python
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


class CodexCliError(RuntimeError):
    """Codex CLI provider 的可诊断基础错误。"""


class CodexLoginError(CodexCliError):
    """Codex 未使用 ChatGPT 身份登录。"""


class CodexCapabilityError(CodexCliError):
    """本机 Codex CLI 缺少 adapter 依赖的命令能力。"""


class CodexInvocationError(CodexCliError):
    """Codex invocation 退出或返回无效结果。"""


class CodexTimeoutError(CodexInvocationError):
    """Codex invocation 超时。"""


class CodexIsolationError(CodexInvocationError):
    """Codex 尝试执行 adapter 禁止的自身工具。"""


@dataclass(frozen=True)
class CodexInvocationResult:
    value: dict[str, Any]
    usage: dict[str, int] | None
    elapsed_seconds: float


_ENV_ALLOWLIST = {
    "PATH", "HOME", "CODEX_HOME", "TMPDIR", "LANG", "LC_ALL",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
}

_REQUIRED_EXEC_FLAGS = {
    "--ephemeral", "--ignore-user-config", "--ignore-rules", "--sandbox",
    "--skip-git-repo-check", "--output-schema", "--output-last-message", "--json",
}


def _codex_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}


class CodexCliRunner:
    def __init__(self, executable: str = "codex", timeout_seconds: float = 300.0):
        resolved = executable if os.path.isabs(executable) else shutil.which(executable)
        if not resolved:
            raise CodexCapabilityError("未找到官方 Codex CLI；请先安装并运行 codex login")
        self.executable = resolved
        self.timeout_seconds = float(timeout_seconds)

    def ensure_ready(self) -> None:
        login = subprocess.run(
            [self.executable, "login", "status"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_codex_env(),
        )
        if login.returncode != 0 or "Logged in using ChatGPT" not in login.stdout:
            raise CodexLoginError(
                "Codex CLI 必须使用 ChatGPT 登录；请运行 codex login 并选择 ChatGPT"
            )

        help_result = subprocess.run(
            [self.executable, "exec", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_codex_env(),
        )
        missing = sorted(flag for flag in _REQUIRED_EXEC_FLAGS if flag not in help_result.stdout)
        if help_result.returncode != 0 or missing:
            detail = ", ".join(missing) or "codex exec --help failed"
            raise CodexCapabilityError(f"Codex CLI 缺少所需能力: {detail}")
```

- [ ] **Step 1.4: 运行 readiness 测试**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_cli_runner.py -v
```

Expected: 3 PASS。

- [ ] **Step 1.5: 提交 readiness 契约**

```bash
git add tests/test_codex_cli_runner.py tradingagents/llm_clients/codex_cli_runner.py
git commit -m "feat: define Codex CLI readiness contract"
```

### Task 2: 隔离的 `codex exec` invocation

**Files:**
- Modify: `tests/test_codex_cli_runner.py`
- Modify: `tradingagents/llm_clients/codex_cli_runner.py`

- [ ] **Step 2.1: 写 invocation、隔离与 timeout 测试**

在 `tests/test_codex_cli_runner.py` 追加一个可写最终输出的 fake process，并断言命令参数、stdin 和 forbidden item：

```python
import json
import subprocess


class FakeProcess:
    def __init__(self, argv, stdout_text, result, returncode=0, stderr_text=""):
        self.argv = argv
        self.stdout_text = stdout_text
        self.result = result
        self.pid = 4321
        self.returncode = returncode
        self.stderr_text = stderr_text
        self.stdin_seen = None

    def communicate(self, input=None, timeout=None):
        self.stdin_seen = input
        output_path = self.argv[self.argv.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.result, handle)
        return self.stdout_text, self.stderr_text


@pytest.mark.unit
def test_invoke_is_ephemeral_read_only_and_omits_default_model():
    holder = {}

    def factory(argv, **kwargs):
        proc = FakeProcess(
            argv,
            '\n'.join([
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}),
            ]),
            {"content": "ok"},
        )
        holder["proc"] = proc
        return proc

    runner = CodexCliRunner(executable="/opt/codex", timeout_seconds=10)
    with mock.patch("subprocess.Popen", side_effect=factory):
        result = runner.invoke("PROMPT", {"type": "object"}, model="default")

    argv = holder["proc"].argv
    assert "--ephemeral" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--model" not in argv
    assert holder["proc"].stdin_seen == "PROMPT"
    assert result.value == {"content": "ok"}
    assert result.usage == {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


@pytest.mark.unit
def test_invoke_rejects_codex_tool_execution():
    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            json.dumps({"type": "item.started", "item": {"type": "command_execution"}}),
            {"content": "unsafe"},
        )
    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(CodexIsolationError, match="command_execution"):
            runner.invoke("PROMPT", {"type": "object"})


@pytest.mark.unit
def test_invoke_kills_process_group_on_timeout():
    process = mock.Mock(pid=4321)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="codex", timeout=1),
        ("", "stopped"),
    ]
    runner = CodexCliRunner(executable="/opt/codex", timeout_seconds=1)
    with mock.patch("subprocess.Popen", return_value=process), mock.patch("os.killpg") as killpg:
        with pytest.raises(CodexTimeoutError, match="1"):
            runner.invoke("PROMPT", {"type": "object"})
    killpg.assert_called_once()


@pytest.mark.unit
def test_invoke_classifies_limit_and_redacts_bearer_token():
    def factory(argv, **kwargs):
        return FakeProcess(
            argv, "", {"content": ""}, returncode=1,
            stderr_text="usage limit reached Bearer secret-token-value",
        )
    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(CodexInvocationError, match="usage limit") as caught:
            runner.invoke("PROMPT", {"type": "object"})
    assert "secret-token-value" not in str(caught.value)


@pytest.mark.unit
def test_invoke_rejects_non_object_output():
    def factory(argv, **kwargs):
        return FakeProcess(argv, json.dumps({"type": "turn.completed"}), None)
    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(CodexInvocationError, match="JSON object"):
            runner.invoke("PROMPT", {"type": "object"})
```

- [ ] **Step 2.2: 验证新增测试失败**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_cli_runner.py -v
```

Expected: FAIL because `CodexCliRunner.invoke` 尚不存在。

- [ ] **Step 2.3: 实现进程、JSONL 和错误分类**

在 runner 中增加 `json`、`re`、`signal`、`tempfile`、`time`、`Path` imports，并实现：

```python
_ALLOWED_ITEM_TYPES = {"agent_message", "reasoning", "plan", "todo_list"}


def _safe_stderr(stderr: str, limit: int = 500) -> str:
    flattened = " ".join(stderr.split())
    redacted = re.sub(
        r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|access_token[=:]\S+)",
        "[REDACTED]",
        flattened,
        flags=re.IGNORECASE,
    )
    return redacted[:limit]


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, int] | None:
    for event in reversed(events):
        usage = event.get("usage")
        if isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
    return None


def _assert_no_codex_tools(events: list[dict[str, Any]]) -> None:
    for event in events:
        if event.get("type") not in {"item.started", "item.completed"}:
            continue
        item_type = (event.get("item") or {}).get("type")
        if item_type and item_type not in _ALLOWED_ITEM_TYPES:
            raise CodexIsolationError(f"Codex isolation violation: {item_type}")


def invoke(self, prompt: str, output_schema: dict[str, Any], model: str = "default"):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="tradingagents-codex-") as tmp:
        root = Path(tmp)
        schema_path = root / "schema.json"
        output_path = root / "output.json"
        schema_path.write_text(json.dumps(output_schema), encoding="utf-8")
        argv = [
            self.executable, "exec", "--ephemeral", "--ignore-user-config",
            "--ignore-rules", "--sandbox", "read-only", "--skip-git-repo-check",
            "--cd", str(root), "--json", "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
        ]
        if model != "default":
            argv.extend(["--model", model])
        argv.append("-")
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=_codex_env(),
        )
        try:
            stdout, stderr = process.communicate(input=prompt, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate()
            raise CodexTimeoutError(
                f"Codex invocation exceeded {self.timeout_seconds:g} seconds"
            ) from exc
        if process.returncode != 0:
            detail = _safe_stderr(stderr)
            lowered = detail.lower()
            if "rate limit" in lowered or "usage limit" in lowered or "credit" in lowered:
                detail = f"Codex usage limit reached: {detail}"
            elif "model" in lowered:
                detail = f"Codex rejected model {model!r}: {detail}"
            raise CodexInvocationError(
                detail or f"Codex invocation failed for model {model}: exit {process.returncode}"
            )
        try:
            events = _parse_events(stdout)
            _assert_no_codex_tools(events)
            value = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CodexInvocationError("Codex returned invalid structured output") from exc
        if not isinstance(value, dict):
            raise CodexInvocationError("Codex structured output must be a JSON object")
        return CodexInvocationResult(
            value=value,
            usage=_usage_from_events(events),
            elapsed_seconds=time.monotonic() - started,
        )
```

将该函数缩进为 `CodexCliRunner.invoke`；pure helpers 保持 module-level，便于测试。

- [ ] **Step 2.4: 运行 runner 测试和 lint**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_cli_runner.py -v
.venv/bin/ruff check tradingagents/llm_clients/codex_cli_runner.py tests/test_codex_cli_runner.py
```

Expected: 全部 PASS，ruff 无输出。

- [ ] **Step 2.5: 提交 runner**

```bash
git add tradingagents/llm_clients/codex_cli_runner.py tests/test_codex_cli_runner.py
git commit -m "feat: add isolated Codex CLI runner"
```

### Task 3: 普通 LangChain chat model

**Files:**
- Create: `tests/test_codex_chatgpt_client.py`
- Create: `tradingagents/llm_clients/codex_chatgpt_client.py`

- [ ] **Step 3.1: 写 plain invoke 与 usage 测试**

```python
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.llm_clients.codex_chatgpt_client import CodexChatGPTClient, CodexChatModel
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
    result = llm.invoke([SystemMessage(content="system"), HumanMessage(content="question")])
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
def test_client_rejects_backend_url():
    client = CodexChatGPTClient("default", base_url="https://api.openai.com/v1")
    with pytest.raises(ValueError, match="does not use backend_url"):
        client.get_llm()
```

- [ ] **Step 3.2: 验证测试失败**

Run: `.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py -v`

Expected: FAIL because client module 不存在。

- [ ] **Step 3.3: 实现 plain adapter 和 client wrapper**

创建 `tradingagents/llm_clients/codex_chatgpt_client.py`：

```python
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


def _prompt(messages: list[BaseMessage], tools: tuple[dict[str, Any], ...] = ()) -> str:
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
    runner: CodexCliRunner = Field(exclude=True)
    bound_tools: tuple[dict[str, Any], ...] = Field(default_factory=tuple, exclude=True)
    structured_schema: Any | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "codex_chatgpt"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "provider": "codex_chatgpt"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError("codex_chatgpt does not support stop sequences")
        result = self.runner.invoke(
            _prompt(messages, self.bound_tools), PLAIN_SCHEMA, model=self.model_name
        )
        message = AIMessage(content=result.value["content"], usage_metadata=result.usage)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"provider": "codex_chatgpt", "elapsed_seconds": result.elapsed_seconds},
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
```

- [ ] **Step 3.4: 运行 plain adapter 测试和 lint**

```bash
.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py -v
.venv/bin/ruff check tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
```

Expected: PASS。

- [ ] **Step 3.5: 提交 plain adapter**

```bash
git add tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
git commit -m "feat: add plain Codex ChatGPT chat model"
```

### Task 4: LangGraph tool-call 协议

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_codex_chatgpt_client.py`
- Modify: `tradingagents/llm_clients/codex_chatgpt_client.py`

- [ ] **Step 4.1: 写 tool binding 测试**

追加测试：

```python
from langchain_core.tools import tool


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
            "tool_calls": [{"id": "call_1", "name": "quote", "args": {"symbol": "AAPL"}}],
        },
        usage=None,
        elapsed_seconds=0.1,
    )
    result = CodexChatModel(model_name="default", runner=runner).bind_tools([quote]).invoke(
        "Get AAPL"
    )
    assert result.tool_calls == [{
        "name": "quote", "args": {"symbol": "AAPL"}, "id": "call_1", "type": "tool_call"
    }]


@pytest.mark.unit
def test_bind_tools_rejects_unknown_tool():
    runner = mock.Mock()
    runner.invoke.return_value = CodexInvocationResult(
        value={
            "mode": "tool_calls", "content": "",
            "tool_calls": [{"id": "call_1", "name": "delete_all", "args": {}}],
        }, usage=None, elapsed_seconds=0.1,
    )
    bound = CodexChatModel(model_name="default", runner=runner).bind_tools([quote])
    with pytest.raises(ValueError, match="delete_all"):
        bound.invoke("Get AAPL")
```

- [ ] **Step 4.2: 验证失败**

Run: `.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py -v`

Expected: FAIL because `bind_tools` 未实现。

- [ ] **Step 4.3: 增加 jsonschema 依赖并实现 tool schema**

在 `pyproject.toml` dependencies 增加：

```toml
"jsonschema>=4.23.0",
```

在 client module 增加 imports 与 helpers：

```python
from collections.abc import Callable, Sequence

from jsonschema import ValidationError, validate
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


def _tool_response_schema(tools: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    variants = []
    for tool_spec in tools:
        function = tool_spec["function"]
        variants.append({
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "name": {"const": function["name"]},
                "args": function.get("parameters", {"type": "object"}),
            },
            "required": ["id", "name", "args"],
            "additionalProperties": False,
        })
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
```

在 `CodexChatModel` 增加：

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
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
```

把 `_generate` 的 plain branch 改为：

```python
schema = _tool_response_schema(self.bound_tools) if self.bound_tools else PLAIN_SCHEMA
result = self.runner.invoke(_prompt(messages, self.bound_tools), schema, model=self.model_name)
if self.bound_tools:
    _validate_output(result.value, schema)
    if result.value["mode"] == "final":
        if result.value["tool_calls"]:
            raise ValueError("codex_chatgpt returned final mode with tool calls")
        message = AIMessage(content=result.value["content"], usage_metadata=result.usage)
    else:
        calls = [
            {"name": call["name"], "args": call["args"], "id": call["id"], "type": "tool_call"}
            for call in result.value["tool_calls"]
        ]
        if not calls:
            raise ValueError("codex_chatgpt returned tool_calls mode without calls")
        message = AIMessage(content=result.value["content"], tool_calls=calls, usage_metadata=result.usage)
else:
    message = AIMessage(content=result.value["content"], usage_metadata=result.usage)
```

- [ ] **Step 4.4: 安装更新后的 editable package并运行测试**

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py -v
.venv/bin/ruff check tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
```

Expected: PASS。

- [ ] **Step 4.5: 提交 tool-call 支持**

```bash
git add pyproject.toml tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
git commit -m "feat: support TradingAgents tools through Codex"
```

### Task 5: Pydantic structured output

**Files:**
- Modify: `tests/test_codex_chatgpt_client.py`
- Modify: `tradingagents/llm_clients/codex_chatgpt_client.py`

- [ ] **Step 5.1: 写 structured output 测试**

```python
from pydantic import BaseModel


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
def test_structured_output_rejects_include_raw():
    llm = CodexChatModel(model_name="default", runner=mock.Mock())
    with pytest.raises(NotImplementedError, match="include_raw"):
        llm.with_structured_output(Decision, include_raw=True)
```

- [ ] **Step 5.2: 验证失败**

Run: `.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py -v`

Expected: structured test FAIL，因为 BaseChatModel 默认实现依赖未实现的 tool binding 语义。

- [ ] **Step 5.3: 实现 schema 归一化与 parser pipeline**

增加 imports：

```python
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel
```

增加 helper：

```python
def _json_schema(schema: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    if isinstance(schema, dict):
        if schema.get("type") == "function":
            return schema["function"]["parameters"]
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    raise TypeError("structured schema must be a Pydantic model or JSON schema dict")
```

在 model 增加：

```python
def with_structured_output(self, schema, *, include_raw=False, **kwargs):
    if include_raw:
        raise NotImplementedError("codex_chatgpt does not support include_raw")
    if kwargs:
        raise ValueError(f"Unsupported structured-output arguments: {sorted(kwargs)}")
    if self.bound_tools:
        raise ValueError("bind_tools and with_structured_output cannot be combined")
    bound = self.model_copy(update={"structured_schema": schema})
    parser = (
        PydanticOutputParser(pydantic_object=schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel)
        else JsonOutputParser()
    )
    return bound | parser
```

在 `_generate` 最前面的 schema/result 分支加入：

```python
if self.structured_schema is not None:
    schema = _json_schema(self.structured_schema)
    result = self.runner.invoke(_prompt(messages), schema, model=self.model_name)
    _validate_output(result.value, schema)
    message = AIMessage(content=json.dumps(result.value), usage_metadata=result.usage)
elif self.bound_tools:
    # Task 4 的 tool branch
else:
    # Task 3 的 plain branch
```

- [ ] **Step 5.4: 运行 client 与现有 structured agent 测试**

```bash
.venv/bin/python -m pytest tests/test_codex_chatgpt_client.py tests/test_structured_agents.py -v
.venv/bin/ruff check tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
```

Expected: PASS。

- [ ] **Step 5.5: 提交 structured output**

```bash
git add tradingagents/llm_clients/codex_chatgpt_client.py tests/test_codex_chatgpt_client.py
git commit -m "feat: add Codex structured output"
```

### Task 6: Provider factory、配置和 CLI 注册

**Files:**
- Create: `tests/test_codex_provider_registration.py`
- Modify: `tradingagents/llm_clients/factory.py`
- Modify: `tradingagents/llm_clients/api_key_env.py`
- Modify: `tradingagents/llm_clients/model_catalog.py`
- Modify: `tradingagents/llm_clients/validators.py`
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `cli/utils.py`
- Modify: `tests/test_api_key_env.py`
- Modify: `tests/test_cli_env_skip.py`
- Modify: `tests/test_env_overrides.py`

- [ ] **Step 6.1: 写 provider 注册测试**

```python
from unittest import mock

from cli.utils import _llm_provider_table, provider_default_url
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model


def test_codex_provider_is_keyless_and_has_no_url():
    assert get_api_key_env("codex_chatgpt") is None
    assert provider_default_url("codex_chatgpt") is None
    assert any(row[1] == "codex_chatgpt" for row in _llm_provider_table())


def test_codex_model_catalog_has_default_and_custom():
    assert get_model_options("codex_chatgpt", "quick") == [
        ("Codex account default", "default"),
        ("Custom model ID", "custom"),
    ]
    assert validate_model("codex_chatgpt", "future-codex-model") is True


def test_factory_lazily_creates_codex_client():
    with mock.patch(
        "tradingagents.llm_clients.codex_chatgpt_client.CodexCliRunner"
    ) as runner_type:
        runner_type.return_value.ensure_ready.return_value = None
        client = create_llm_client("codex_chatgpt", "default")
        assert client.__class__.__name__ == "CodexChatGPTClient"
        llm = client.get_llm()
        runner_type.assert_called_once_with(timeout_seconds=300.0)
        runner_type.return_value.ensure_ready.assert_called_once_with()
        assert llm._llm_type == "codex_chatgpt"
```

- [ ] **Step 6.2: 验证测试失败**

Run: `.venv/bin/python -m pytest tests/test_codex_provider_registration.py -v`

Expected: factory/catalog/table assertions FAIL。

- [ ] **Step 6.3: 注册 provider 和模型**

实施以下精确改动：

`factory.py` 在其他 native clients 后、OpenAI-compatible import 前增加：

```python
if provider_lower == "codex_chatgpt":
    from .codex_chatgpt_client import CodexChatGPTClient
    return CodexChatGPTClient(model, base_url, **kwargs)
```

`api_key_env.py` 增加：

```python
"codex_chatgpt": None,
```

`model_catalog.py` 增加：

```python
"codex_chatgpt": {
    "quick": [("Codex account default", "default"), ("Custom model ID", "custom")],
    "deep": [("Codex account default", "default"), ("Custom model ID", "custom")],
},
```

`validators.py` 的 `_ANY_MODEL_PROVIDERS` 增加 `"codex_chatgpt"`。

`cli/utils.py` 的 provider table 增加：

```python
("Codex via ChatGPT login", "codex_chatgpt", None),
```

- [ ] **Step 6.4: 增加 timeout config 并只向 Codex 转发**

`default_config.py`：

```python
"TRADINGAGENTS_CODEX_TIMEOUT": "codex_timeout_seconds",
```

以及：

```python
"codex_timeout_seconds": 300,
```

`TradingAgentsGraph._get_provider_kwargs` 增加：

```python
elif provider == "codex_chatgpt":
    kwargs["timeout"] = float(self.config.get("codex_timeout_seconds", 300))
```

在 `tests/test_env_overrides.py` 增加：

```python
def test_codex_timeout_env_override(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CODEX_TIMEOUT="45")
    assert dc.DEFAULT_CONFIG["codex_timeout_seconds"] == 45
```

在 `tests/test_api_key_env.py` 将 `codex_chatgpt` 加入 `expected` set，并增加：

```python
def test_ensure_api_key_no_op_for_codex_chatgpt(monkeypatch, cli_utils):
    with patch.object(cli_utils, "questionary") as mock_q:
        result = cli_utils.ensure_api_key("codex_chatgpt")
    assert result is None
    mock_q.password.assert_not_called()
```

在 `tests/test_cli_env_skip.py` 的 `TestProviderDefaultUrl` 增加：

```python
def test_codex_chatgpt_has_no_backend_url(self):
    from cli.utils import provider_default_url
    self.assertIsNone(provider_default_url("codex_chatgpt"))
```

在 `tests/test_codex_provider_registration.py` 增加 timeout forwarding 单元测试：

```python
def test_graph_forwards_codex_timeout_only():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"llm_provider": "codex_chatgpt", "codex_timeout_seconds": 45}
    assert graph._get_provider_kwargs() == {"timeout": 45.0}
```

- [ ] **Step 6.5: 运行注册和配置回归**

```bash
.venv/bin/python -m pytest tests/test_codex_provider_registration.py tests/test_api_key_env.py tests/test_cli_env_skip.py tests/test_env_overrides.py tests/test_model_validation.py -v
.venv/bin/ruff check tradingagents/llm_clients cli tests/test_codex_provider_registration.py
```

Expected: PASS；现有 provider 行为不变。

- [ ] **Step 6.6: 提交 provider 注册**

```bash
git add cli/utils.py tradingagents/default_config.py tradingagents/graph/trading_graph.py tradingagents/llm_clients/factory.py tradingagents/llm_clients/api_key_env.py tradingagents/llm_clients/model_catalog.py tradingagents/llm_clients/validators.py tests/test_codex_provider_registration.py tests/test_api_key_env.py tests/test_cli_env_skip.py tests/test_env_overrides.py
git commit -m "feat: register Codex ChatGPT provider"
```

### Task 7: Opt-in Go/No-Go smoke

**Files:**
- Create: `scripts/smoke_codex_chatgpt.py`

- [ ] **Step 7.1: 创建 protocol 与 shallow smoke 脚本**

脚本必须要求 `RUN_CODEX_CHATGPT_SMOKE=1`，默认不消耗额度。核心实现：

```python
import argparse
import os
import time

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client


class SmokeDecision(BaseModel):
    rating: str
    confidence: float


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right


def build_llm():
    return create_llm_client("codex_chatgpt", "default", timeout=300).get_llm()


def protocol_smoke():
    llm = build_llm()
    started = time.monotonic()
    plain = llm.invoke("Reply with the single word READY")
    structured = llm.with_structured_output(SmokeDecision).invoke(
        "Return rating Hold and confidence 0.7"
    )
    bound = llm.bind_tools([multiply])
    first = bound.invoke("Use multiply for 6 times 7")
    assert first.tool_calls, "No tool call returned"
    call = first.tool_calls[0]
    value = multiply.invoke(call["args"])
    final = bound.invoke([
        HumanMessage(content="Use multiply for 6 times 7"),
        first,
        ToolMessage(content=str(value), tool_call_id=call["id"]),
    ])
    print({
        "plain": plain.content,
        "structured": structured.model_dump(),
        "tool_final": final.content,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    })


def shallow_smoke(ticker: str, date: str):
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "codex_chatgpt",
        "quick_think_llm": "default",
        "deep_think_llm": "default",
        "backend_url": None,
        "codex_timeout_seconds": 300,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "output_language": "Chinese",
    })
    _, decision = TradingAgentsGraph(
        selected_analysts=["market"], config=config
    ).propagate(ticker, date)
    assert decision and decision.strip(), "Shallow analysis produced no decision"
    print(decision)


def main():
    if os.environ.get("RUN_CODEX_CHATGPT_SMOKE") != "1":
        raise SystemExit("Set RUN_CODEX_CHATGPT_SMOKE=1 to consume Codex quota")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("protocol", "shallow"), default="protocol")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--date", required=False)
    args = parser.parse_args()
    if args.mode == "protocol":
        protocol_smoke()
    elif not args.date:
        raise SystemExit("--date is required for shallow mode")
    else:
        shallow_smoke(args.ticker, args.date)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: 验证默认拒绝消耗额度**

Run:

```bash
.venv/bin/python scripts/smoke_codex_chatgpt.py --mode protocol
```

Expected: exit non-zero with `Set RUN_CODEX_CHATGPT_SMOKE=1`，且没有 Codex invocation。

- [ ] **Step 7.3: 运行真实 protocol Go/No-Go**

Run:

```bash
RUN_CODEX_CHATGPT_SMOKE=1 .venv/bin/python scripts/smoke_codex_chatgpt.py --mode protocol
```

Expected:

- plain 非空；
- structured 为 `{"rating": "Hold", "confidence": 0.7}`；
- 有合法 `multiply` tool call，最终回答包含 42；
- 没有 `CodexIsolationError`；
- 每个 invocation 小于 300 秒。

任一失败即 No-Go：停止 Task 7.4 及后续真实 smoke，保留失败证据，不改用 Cookie/private API。

- [ ] **Step 7.4: 运行一次浅层 TradingAgents smoke**

Run（日期替换为执行当天或最近交易日）：

```bash
RUN_CODEX_CHATGPT_SMOKE=1 .venv/bin/python scripts/smoke_codex_chatgpt.py --mode shallow --ticker AAPL --date 2026-07-02
```

Expected: market tool loop完成，Research Manager/Trader/Portfolio Manager structured output成功，最后打印非空 decision/rating；无 LLM API key prompt。

- [ ] **Step 7.5: 提交 smoke 工具**

```bash
git add scripts/smoke_codex_chatgpt.py
git commit -m "test: add opt-in Codex ChatGPT smoke"
```

### Task 8: 文档、全量回归和完成验证

**Files:**
- Modify: `README.md`

- [ ] **Step 8.1: 更新 README provider 与配置说明**

在 Required APIs 后增加：

````markdown
### Codex via ChatGPT login (no LLM API key)

`codex_chatgpt` uses the official Codex CLI authentication already present on
the machine. Run `codex login status` and confirm it says `Logged in using
ChatGPT`; otherwise run `codex login` and choose ChatGPT.

```python
config["llm_provider"] = "codex_chatgpt"
config["quick_think_llm"] = "default"
config["deep_think_llm"] = "default"
config["backend_url"] = None
config["codex_timeout_seconds"] = 300
```

This consumes the ChatGPT plan's Codex allowance, not OpenAI API credits or
ordinary ChatGPT message allowance. It never reads browser cookies or Codex
OAuth files. Each TradingAgents model call starts an isolated `codex exec`
process, so it is slower and may use substantial Codex allowance during a full
multi-agent run. There is no automatic fallback to a paid API provider.
````

将 “Implementation Details” 的 provider 句子替换为：

```markdown
The framework supports OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen, GLM,
MiniMax, OpenRouter, Azure OpenAI, Amazon Bedrock, Ollama, generic
OpenAI-compatible endpoints, and Codex via an existing ChatGPT login.
```

将 Python config 示例的 provider 注释替换为：

```python
config["llm_provider"] = "openai"  # also: google, anthropic, deepseek, codex_chatgpt, ollama, openai_compatible, ...
```

- [ ] **Step 8.2: 运行针对性回归**

```bash
.venv/bin/python -m pytest tests/test_codex_cli_runner.py tests/test_codex_chatgpt_client.py tests/test_codex_provider_registration.py tests/test_api_key_env.py tests/test_cli_env_skip.py tests/test_env_overrides.py tests/test_model_validation.py tests/test_structured_agents.py -q
```

Expected: 全部 PASS。

- [ ] **Step 8.3: 运行全量测试和 lint**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Expected: 全部 PASS；ruff 无错误。若既有基线失败仍存在，必须与 Step 0.2 证据逐项对照，不可笼统宣称完成。

- [ ] **Step 8.4: 安全扫描与凭据检查**

Run:

```bash
rg -n "auth\.json|backend-api|session_token|OPENAI_API_KEY.*codex_chatgpt|Cookie:" tradingagents cli scripts tests README.md
git diff --check
git status --short
```

Expected:

- 代码不读取 `auth.json`、Cookie 或 private endpoint；README 中如出现这些词只能是否定说明。
- `git diff --check` 无输出。
- 仅本功能文件和用户原有 `.DS_Store` 状态可见；不得暂存 `.DS_Store`。

- [ ] **Step 8.5: 提交文档**

```bash
git add README.md
git commit -m "docs: document Codex ChatGPT provider"
```

- [ ] **Step 8.6: 最终提交范围检查**

```bash
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git status --short --branch
```

Expected: 设计、计划和上述 8 个实现提交均可解释；工作树除用户原有未跟踪 `.DS_Store` 外干净。不要 push 或创建 PR，除非用户另行授权。

## 完成判定

只有同时满足以下条件才可宣称完成：

- `codex_chatgpt` 在 CLI 和 programmatic config 可选。
- `codex login status` 必须是 ChatGPT 登录；API-key 登录会被拒绝。
- plain、tool calling、structured output 的真实 protocol smoke 均通过。
- 一次 AAPL 单 analyst 浅层分析输出非空最终 decision/rating。
- 现有 provider 回归和全量测试通过。
- 无 Cookie、OAuth token、`auth.json` 或私有 ChatGPT endpoint 读取逻辑。
- 登录、额度、model、timeout、invalid output、isolation violation 都 fail closed，且不静默回退到付费 API。
