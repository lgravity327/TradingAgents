"""Opt-in smoke tests for the ChatGPT-authenticated Codex provider."""

from __future__ import annotations

import argparse
import json
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


def protocol_smoke() -> None:
    llm = build_llm()
    started = time.monotonic()

    plain = llm.invoke("Reply with the single word READY")
    assert plain.content.strip(), "Plain invocation returned empty content"

    structured = llm.with_structured_output(SmokeDecision).invoke(
        "Return rating Hold and confidence 0.7"
    )
    assert structured.rating == "Hold"
    assert structured.confidence == 0.7

    bound = llm.bind_tools([multiply])
    first = bound.invoke("Use multiply for 6 times 7")
    assert first.tool_calls, "No tool call returned"
    call = first.tool_calls[0]
    value = multiply.invoke(call["args"])
    final = bound.invoke(
        [
            HumanMessage(content="Use multiply for 6 times 7"),
            first,
            ToolMessage(content=str(value), tool_call_id=call["id"]),
        ]
    )
    assert "42" in final.content

    print(
        json.dumps(
            {
                "plain": plain.content,
                "structured": structured.model_dump(),
                "tool_final": final.content,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def shallow_smoke(ticker: str, date: str) -> None:
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": "codex_chatgpt",
            "quick_think_llm": "default",
            "deep_think_llm": "default",
            "backend_url": None,
            "codex_timeout_seconds": 300,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "output_language": "Chinese",
        }
    )
    _, decision = TradingAgentsGraph(
        selected_analysts=["market"],
        config=config,
    ).propagate(ticker, date)
    assert decision and decision.strip(), "Shallow analysis produced no decision"
    print(decision)


def main() -> None:
    if os.environ.get("RUN_CODEX_CHATGPT_SMOKE") != "1":
        raise SystemExit("Set RUN_CODEX_CHATGPT_SMOKE=1 to consume Codex quota")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("protocol", "shallow"), default="protocol")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--date")
    args = parser.parse_args()

    if args.mode == "protocol":
        protocol_smoke()
    elif not args.date:
        raise SystemExit("--date is required for shallow mode")
    else:
        shallow_smoke(args.ticker, args.date)


if __name__ == "__main__":
    main()
