import json
import subprocess
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
    with mock.patch("shutil.which", return_value=None), pytest.raises(
        CodexCapabilityError, match="Codex CLI"
    ):
        CodexCliRunner(executable="codex")


@pytest.mark.unit
def test_readiness_requires_chatgpt_login():
    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch(
        "subprocess.run", return_value=completed(0, "Logged in using an API key")
    ), pytest.raises(CodexLoginError, match="ChatGPT"):
        runner.ensure_ready()


@pytest.mark.unit
def test_readiness_requires_exec_flags():
    runner = CodexCliRunner(executable="/opt/codex")
    responses = [
        completed(0, "Logged in using ChatGPT"),
        completed(0, "Usage: codex exec --json --ephemeral"),
    ]
    with mock.patch("subprocess.run", side_effect=responses), pytest.raises(
        CodexCapabilityError, match="--output-schema"
    ):
        runner.ensure_ready()


@pytest.mark.unit
def test_readiness_accepts_chatgpt_status_written_to_stderr():
    runner = CodexCliRunner(executable="/opt/codex")
    help_text = " ".join(
        [
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "--skip-git-repo-check",
            "--output-schema",
            "--output-last-message",
            "--json",
        ]
    )
    responses = [
        completed(0, "", "warning\nLogged in using ChatGPT\n"),
        completed(0, help_text),
    ]
    with mock.patch("subprocess.run", side_effect=responses):
        runner.ensure_ready()


def test_error_hierarchy_is_specific():
    assert issubclass(CodexLoginError, CodexCapabilityError)
    assert issubclass(CodexTimeoutError, CodexInvocationError)
    assert issubclass(CodexIsolationError, CodexInvocationError)


class FakeProcess:
    def __init__(
        self,
        argv,
        stdout_text,
        result,
        returncode=0,
        stderr_text="",
    ):
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
            "\n".join(
                [
                    json.dumps({"type": "thread.started"}),
                    json.dumps(
                        {"type": "item.completed", "item": {"type": "agent_message"}}
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 4, "output_tokens": 2},
                        }
                    ),
                ]
            ),
            {"content": "ok"},
        )
        holder["proc"] = proc
        holder["kwargs"] = kwargs
        return proc

    runner = CodexCliRunner(executable="/opt/codex", timeout_seconds=10)
    with mock.patch("subprocess.Popen", side_effect=factory):
        result = runner.invoke("PROMPT", {"type": "object"}, model="default")

    argv = holder["proc"].argv
    assert "--ephemeral" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--model" not in argv
    assert holder["proc"].stdin_seen == "PROMPT"
    assert holder["kwargs"]["shell"] is False
    assert "OPENAI_API_KEY" not in holder["kwargs"]["env"]
    assert result.value == {"content": "ok"}
    assert result.usage == {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


@pytest.mark.unit
def test_invoke_passes_explicit_model():
    holder = {}

    def factory(argv, **kwargs):
        holder["argv"] = argv
        return FakeProcess(argv, json.dumps({"type": "turn.completed"}), {"content": "ok"})

    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory):
        runner.invoke("PROMPT", {"type": "object"}, model="gpt-codex-custom")
    assert holder["argv"][holder["argv"].index("--model") + 1] == "gpt-codex-custom"


@pytest.mark.unit
def test_invoke_rejects_codex_tool_execution():
    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            json.dumps({"type": "item.started", "item": {"type": "command_execution"}}),
            {"content": "unsafe"},
        )

    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory), pytest.raises(
        CodexIsolationError, match="command_execution"
    ):
        runner.invoke("PROMPT", {"type": "object"})


@pytest.mark.unit
def test_invoke_kills_process_group_on_timeout():
    process = mock.Mock(pid=4321)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="codex", timeout=1),
        ("", "stopped"),
    ]
    runner = CodexCliRunner(executable="/opt/codex", timeout_seconds=1)
    with (
        mock.patch("subprocess.Popen", return_value=process),
        mock.patch("os.killpg") as killpg,
        pytest.raises(CodexTimeoutError, match="1"),
    ):
        runner.invoke("PROMPT", {"type": "object"})
    killpg.assert_called_once_with(4321, mock.ANY)


@pytest.mark.unit
def test_invoke_classifies_limit_and_redacts_bearer_token():
    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            "",
            {"content": ""},
            returncode=1,
            stderr_text="usage limit reached Bearer secret-token-value",
        )

    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory), pytest.raises(
        CodexInvocationError, match="usage limit"
    ) as caught:
        runner.invoke("PROMPT", {"type": "object"})
    assert "secret-token-value" not in str(caught.value)


@pytest.mark.unit
def test_invoke_rejects_non_object_output():
    def factory(argv, **kwargs):
        return FakeProcess(argv, json.dumps({"type": "turn.completed"}), None)

    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory), pytest.raises(
        CodexInvocationError, match="JSON object"
    ):
        runner.invoke("PROMPT", {"type": "object"})


@pytest.mark.unit
def test_invoke_surfaces_jsonl_error_when_stderr_is_empty():
    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            json.dumps(
                {
                    "type": "error",
                    "message": "invalid_json_schema: additionalProperties is required",
                }
            ),
            {"content": ""},
            returncode=1,
        )

    runner = CodexCliRunner(executable="/opt/codex")
    with mock.patch("subprocess.Popen", side_effect=factory), pytest.raises(
        CodexInvocationError, match="additionalProperties"
    ):
        runner.invoke("PROMPT", {"type": "object"})
