"""Run deterministic reliability scenarios offline; no provider or MCP network calls."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from octop_harness.middleware.task_reliability import TaskReliabilityMiddleware


class ScenarioModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScenarioModel:
        return self


def call(name: str, call_id: str, **arguments: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": arguments}])


def checkpoint(call_id: str, **arguments: Any) -> AIMessage:
    return call("task_checkpoint", call_id, goal="Inspect saved result", **arguments)


def build(responses: list[AIMessage], tools: list[Any], saver: Any) -> Any:
    return create_agent(
        ScenarioModel(responses=[*responses, AIMessage(content="scenario ended")]),
        tools=tools,
        middleware=[TaskReliabilityMiddleware()],
        checkpointer=saver,
    )


def run_scenario(name: str) -> dict[str, Any]:
    started = time.perf_counter()
    executions: list[str] = []
    interventions = 0
    authorized = False
    config = {"configurable": {"thread_id": "evaluation"}}

    def write_file() -> str:
        executions.append("write_file")
        return "saved"

    def read_file() -> str:
        executions.append("read_file")
        return "saved result"

    def remote() -> str:
        executions.append("remote")
        if name == "uncertain_write":
            raise httpx.ReadTimeout("simulated lost response")
        if name == "authorization_resume":
            return "checked" if authorized else json.dumps({"status": "authorization_required"})
        return json.dumps({"status": "error"})

    tools = [
        StructuredTool.from_function(write_file, description="save"),
        StructuredTool.from_function(read_file, description="inspect"),
        StructuredTool.from_function(
            remote, description="remote", metadata={"read_only": name == "authorization_resume"}
        ),
    ]
    scenarios = {
        "write_readback": [
            call("write_file", "write"),
            call("read_file", "read"),
            checkpoint("done", status="completed", evidence=["read"]),
        ],
        "fabricated_evidence": [checkpoint("done", status="completed", evidence=["invented"])],
        "repeated_failure": [call("remote", f"attempt-{index}") for index in range(4)],
        "uncertain_write": [call("remote", f"attempt-{index}") for index in range(3)],
        "authorization_resume": [
            call("remote", "authorize"),
            checkpoint("wait", status="blocked", blocker="OAuth", next_step="Inspect after authorization"),
        ],
        "restart_after_compaction": [checkpoint("wait", next_step="Inspect saved result")],
    }
    restored = False
    if name == "restart_after_compaction":
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(path) as saver:
                agent = build(scenarios[name], tools, saver)
                agent.invoke({"messages": [HumanMessage(content="inspect result")]}, config)
                agent.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
            with SqliteSaver.from_conn_string(path) as saver:
                agent = build(
                    [call("read_file", "read"), checkpoint("done", status="completed", evidence=["read"])], tools, saver
                )
                restored = (
                    agent.get_state(config).values["task_progress"]["wait"]["next_step"] == "Inspect saved result"
                )
                interventions += 1
                state = agent.invoke({"messages": [HumanMessage(content="continue")]}, config)
    else:
        agent = build(scenarios[name], tools, InMemorySaver())
        state = agent.invoke({"messages": [HumanMessage(content="inspect result")]}, config)
        if name == "authorization_resume":
            authorized = True
            interventions += 1
            resumed = build(
                [call("remote", "verified"), checkpoint("done", status="completed", evidence=["verified"])],
                tools,
                agent.checkpointer,
            )
            state = resumed.invoke({"messages": [HumanMessage(content="authorized; continue")]}, config)
    results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    latest_progress = list(state.get("task_progress", {}).values())[-1:]
    completed = bool(latest_progress and latest_progress[-1]["status"] == "completed")
    passed = {
        "write_readback": completed and executions == ["write_file", "read_file"],
        "fabricated_evidence": not completed and results[-1].status == "error",
        "repeated_failure": executions == ["remote", "remote"],
        "uncertain_write": executions == ["remote"],
        "authorization_resume": completed and executions == ["remote", "remote"],
        "restart_after_compaction": completed and restored,
    }[name]
    counts = Counter(executions)
    return {
        "scenario": name,
        "passed": passed,
        "retained_tool_results": len(results),
        "external_tool_executions": len(executions),
        "repeated_tool_executions": sum(max(0, count - 1) for count in counts.values()),
        "blocked_replays": sum("repeated_call_blocked" in message.text for message in results),
        "simulated_user_interventions": interventions,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "tokens": None,
        "cost_usd": None,
    }


def main() -> int:
    scenarios = [
        "write_readback",
        "fabricated_evidence",
        "repeated_failure",
        "uncertain_write",
        "authorization_resume",
        "restart_after_compaction",
    ]
    results = [run_scenario(name) for name in scenarios]
    passed = sum(result["passed"] for result in results)
    print(
        json.dumps(
            {
                "evaluation": "scripted_offline_runtime",
                "passed": passed,
                "total": len(results),
                "success_rate": passed / len(results),
                "results": results,
            },
            indent=2,
        )
    )
    return int(passed != len(results))


if __name__ == "__main__":
    sys.exit(main())
