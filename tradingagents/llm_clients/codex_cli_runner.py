from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexCliError(RuntimeError):
    """Codex CLI provider 的基础错误。"""


class CodexCapabilityError(CodexCliError):
    """本机 Codex CLI 缺少必需能力。"""


class CodexLoginError(CodexCapabilityError):
    """Codex CLI 未使用 ChatGPT 身份登录。"""


class CodexInvocationError(CodexCliError):
    """Codex invocation 执行或输出失败。"""


class CodexTimeoutError(CodexInvocationError):
    """Codex invocation 超时。"""


class CodexIsolationError(CodexInvocationError):
    """Codex invocation 违反隔离边界。"""


@dataclass(frozen=True)
class CodexInvocationResult:
    value: dict[str, Any]
    usage: dict[str, int] | None
    elapsed_seconds: float


_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "CODEX_HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}

_REQUIRED_EXEC_FLAGS = {
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--sandbox",
    "--skip-git-repo-check",
    "--output-schema",
    "--output-last-message",
    "--json",
}

_ALLOWED_ITEM_TYPES = {"agent_message", "reasoning", "plan", "todo_list"}


def _codex_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}


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
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def _error_from_events(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "error":
            return str(event.get("message") or "")
        if event.get("type") == "turn.failed":
            error = event.get("error") or {}
            return str(error.get("message") or error)
    return ""


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


class CodexCliRunner:
    def __init__(self, executable: str = "codex", timeout_seconds: float = 300.0):
        resolved = executable if os.path.isabs(executable) else shutil.which(executable)
        if not resolved:
            raise CodexCapabilityError(
                "未找到官方 Codex CLI；请先安装 Codex 并运行 codex login"
            )
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
        login_status = f"{login.stdout}\n{login.stderr}"
        if login.returncode != 0 or "Logged in using ChatGPT" not in login_status:
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

    def invoke(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        model: str = "default",
    ) -> CodexInvocationResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="tradingagents-codex-") as tmp:
            root = Path(tmp)
            schema_path = root / "schema.json"
            output_path = root / "output.json"
            schema_path.write_text(json.dumps(output_schema), encoding="utf-8")
            argv = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--cd",
                str(root),
                "--json",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
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
                shell=False,
                start_new_session=True,
                env=_codex_env(),
            )
            try:
                stdout, stderr = process.communicate(
                    input=prompt,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                process.communicate()
                raise CodexTimeoutError(
                    f"Codex invocation exceeded {self.timeout_seconds:g} seconds"
                ) from exc

            if process.returncode != 0:
                detail = _safe_stderr(stderr) or _safe_stderr(_error_from_events(stdout))
                lowered = detail.lower()
                if any(term in lowered for term in ("rate limit", "usage limit", "credit")):
                    detail = f"Codex usage limit reached: {detail}"
                elif "model" in lowered:
                    detail = f"Codex rejected model {model!r}: {detail}"
                raise CodexInvocationError(
                    detail
                    or f"Codex invocation failed for model {model!r}: exit {process.returncode}"
                )

            try:
                events = _parse_events(stdout)
                _assert_no_codex_tools(events)
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except CodexIsolationError:
                raise
            except (json.JSONDecodeError, OSError) as exc:
                raise CodexInvocationError("Codex returned invalid structured output") from exc

            if not isinstance(value, dict):
                raise CodexInvocationError("Codex structured output must be a JSON object")

            return CodexInvocationResult(
                value=value,
                usage=_usage_from_events(events),
                elapsed_seconds=time.monotonic() - started,
            )
