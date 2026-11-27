"""Offline acceptance scenarios for checkpointed, evidence-based tool execution."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from octop_harness.middleware.task_reliability import TaskReliabilityMiddleware


class ScenarioModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScenarioModel:
        return self


def call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


def graph(responses: list[AIMessage], *, tools: list[Any] | None = None, saver: Any = None) -> Any:
    return create_agent(
        ScenarioModel(responses=responses),
        tools=tools or [],
        middleware=[TaskReliabilityMiddleware()],
        checkpointer=saver or InMemorySaver(),
    )


def invoke(agent: Any, *, thread: str = "one", text: str = "inspect") -> dict[str, Any]:
    return agent.invoke({"messages": [HumanMessage(content=text)]}, {"configurable": {"thread_id": thread}})


def checkpoint(call_id: str, **kwargs: Any) -> AIMessage:
    return call("task_checkpoint", call_id, goal="Inspect result", **kwargs)


def test_completion_requires_observed_readback_and_rejects_fabricated_evidence() -> None:
    write = StructuredTool.from_function(lambda: "saved", name="write_file", description="save")
    read = StructuredTool.from_function(lambda: "saved content", name="read_file", description="read")
    agent = graph(
        [
            checkpoint("false", status="completed", evidence=["invented"]),
            call("write_file", "write"),
            checkpoint("write-only", status="completed", evidence=["write"]),
            call("read_file", "read"),
            checkpoint("verified", status="completed", evidence=["read"]),
            AIMessage(content="checked"),
        ],
        tools=[write, read],
    )
    state = invoke(agent)
    results = {message.tool_call_id: message for message in state["messages"] if isinstance(message, ToolMessage)}
    assert results["false"].status == "error"
    assert results["write-only"].status == "error"
    assert results["verified"].status == "success"
    assert list(state["task_progress"].values())[-1]["status"] == "completed"


def test_readback_before_mutation_does_not_verify_new_result() -> None:
    tools = [
        StructuredTool.from_function(lambda: "ok", name=name, description=name) for name in ["read_file", "write_file"]
    ]
    agent = graph(
        [
            call("read_file", "old"),
            call("write_file", "write"),
            checkpoint("stale", status="completed", evidence=["old"]),
            AIMessage(content="unverified"),
        ],
        tools=tools,
    )
    state = invoke(agent)
    assert state["messages"][-2].status == "error"


@pytest.mark.parametrize("error", ["Error: HTTP 404 Not Found", "Error reading file: not found"])
def test_plain_text_tool_errors_cannot_verify_completion(error: str) -> None:
    state = invoke(
        graph(
            [
                call("web_fetch", "failed-read"),
                checkpoint("done", status="completed", evidence=["failed-read"]),
                AIMessage(content="unverified"),
            ],
            tools=[StructuredTool.from_function(lambda: error, name="web_fetch", description="fetch")],
        )
    )
    assert state["tool_observations"]["failed-read"]["outcome"] == "error"
    assert state["messages"][-2].status == "error"


@pytest.mark.parametrize("status,limit", [("authorization_required", 1), ("error", 2)])
def test_repeated_identical_failures_stop_before_replaying_tool(status: str, limit: int) -> None:
    attempts: list[int] = []

    def remote() -> str:
        attempts.append(1)
        return json.dumps({"status": status})

    saver = InMemorySaver()
    agent = graph(
        [call("remote", str(index)) for index in range(4)] + [AIMessage(content="blocked")],
        tools=[StructuredTool.from_function(remote, description="remote")],
        saver=saver,
    )
    state = invoke(agent)
    assert len(attempts) == limit
    assert "repeated_call_blocked" in state["messages"][-2].text
    resumed = graph(
        [call("remote", f"new-{index}") for index in range(4)] + [AIMessage(content="blocked")],
        tools=[StructuredTool.from_function(remote, description="remote")],
        saver=saver,
    )
    invoke(resumed, text="I have authorized; continue")
    assert len(attempts) == limit * 2


def test_timeout_does_not_replay_uncertain_write_or_count_as_evidence() -> None:
    attempts: list[int] = []

    def remote() -> str:
        attempts.append(1)
        raise httpx.ReadTimeout("response lost")

    agent = graph(
        [
            call("remote", "first"),
            call("remote", "again"),
            checkpoint("done", status="completed", evidence=["first"]),
            AIMessage(content="inspect remotely"),
        ],
        tools=[StructuredTool.from_function(remote, description="remote")],
    )
    state = invoke(agent)
    assert len(attempts) == 1
    assert state["tool_observations"]["first"]["outcome"] == "outcome_unknown"
    assert state["messages"][-2].status == "error"


def test_uncertain_write_stays_blocked_across_turns_until_inspection() -> None:
    attempts: list[int] = []

    def remote() -> str:
        attempts.append(1)
        raise httpx.ReadTimeout("response lost")

    tools = [
        StructuredTool.from_function(remote, description="remote"),
        StructuredTool.from_function(lambda: "remote state checked", name="read_file", description="inspect"),
    ]
    saver = InMemorySaver()
    invoke(graph([call("remote", "first"), AIMessage(content="unknown")], tools=tools, saver=saver))
    state = invoke(
        graph([call("remote", "retry"), AIMessage(content="blocked")], tools=tools, saver=saver), text="continue"
    )
    assert len(attempts) == 1
    assert "repeated_call_blocked" in state["messages"][-2].text
    invoke(
        graph(
            [call("read_file", "inspect"), call("remote", "after-inspect"), AIMessage(content="unknown")],
            tools=tools,
            saver=saver,
        ),
        text="check remote state first",
    )
    assert len(attempts) == 2


def test_unrelated_read_does_not_unblock_uncertain_connector_write() -> None:
    attempts: list[int] = []

    def connector_call(server_name: str) -> str:
        attempts.append(1)
        raise httpx.ReadTimeout("response lost")

    state = invoke(
        graph(
            [
                call("connector_call", "first", server_name="remote"),
                call("read_file", "local-read"),
                call("connector_call", "retry", server_name="remote"),
                AIMessage(content="inspect remote"),
            ],
            tools=[
                StructuredTool.from_function(connector_call, description="write remote"),
                StructuredTool.from_function(lambda: "local", name="read_file", description="local read"),
            ],
        )
    )
    assert len(attempts) == 1
    assert "repeated_call_blocked" in state["messages"][-2].text


def test_capability_note_uses_filtered_tools() -> None:
    from unittest.mock import Mock

    from langchain.agents.middleware import ModelRequest, ModelResponse

    from octop_harness.config import HarnessAgentConfig
    from octop_harness.middleware.tools_filter import ToolsFilterMiddleware

    tools = [
        StructuredTool.from_function(lambda: "ok", name=name, description=name) for name in ["visible", "disabled"]
    ]
    request = ModelRequest(
        model=Mock(), messages=[HumanMessage(content="inspect")], tools=tools, state={}, runtime=Mock()
    )
    captured: list[ModelRequest] = []

    def handler(filtered: ModelRequest) -> ModelResponse:
        captured.append(filtered)
        return ModelResponse(result=[])

    reliability = TaskReliabilityMiddleware()
    filtering = ToolsFilterMiddleware(config=HarnessAgentConfig(tools_disabled=frozenset({"disabled"})))
    filtering.wrap_model_call(request, lambda filtered: reliability.wrap_model_call(filtered, handler))
    notes = "\n".join(message.text for message in captured[0].messages)
    assert "available tools = visible." in notes
    assert "available tools = visible, disabled" not in notes


@pytest.mark.parametrize(
    "status,outcome", [(401, "authorization_required"), (403, "permission_required"), (429, "rate_limited")]
)
def test_http_failures_are_classified_and_not_retried(status: int, outcome: str) -> None:
    attempts: list[int] = []

    def remote() -> str:
        attempts.append(1)
        request = httpx.Request("POST", "https://example.com/mcp")
        raise httpx.HTTPStatusError("denied", request=request, response=httpx.Response(status, request=request))

    state = invoke(
        graph(
            [call("remote", "first"), call("remote", "retry"), AIMessage(content="blocked")],
            tools=[StructuredTool.from_function(remote, description="remote")],
        )
    )
    assert len(attempts) == 1
    assert state["tool_observations"]["first"]["outcome"] == outcome


@pytest.mark.parametrize("name", ["tool_search", "current_time", "connector_list_tools"])
def test_inventory_and_time_alone_do_not_verify_task_completion(name: str) -> None:
    state = invoke(
        graph(
            [
                call(name, "inventory"),
                checkpoint("done", status="completed", evidence=["inventory"]),
                AIMessage(content="no proof"),
            ],
            tools=[StructuredTool.from_function(lambda: "ok", name=name, description=name)],
        )
    )
    assert state["messages"][-2].status == "error"


def test_completion_cannot_run_in_parallel_with_mutation() -> None:
    state = invoke(
        graph(
            [
                call("read_file", "read"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_file", "id": "write", "args": {}},
                        {
                            "name": "task_checkpoint",
                            "id": "done",
                            "args": {"goal": "check", "status": "completed", "evidence": ["read"]},
                        },
                    ],
                ),
                AIMessage(content="needs readback"),
            ],
            tools=[
                StructuredTool.from_function(lambda: "ok", name=name, description=name)
                for name in ["read_file", "write_file"]
            ],
        )
    )
    results = {message.tool_call_id: message for message in state["messages"] if isinstance(message, ToolMessage)}
    assert results["done"].status == "error"


def test_softened_mcp_output_failure_is_observed_without_replaying_write() -> None:
    from octop_harness.middleware.mcp_tool_error import McpToolErrorMiddleware

    attempts: list[int] = []

    def remote() -> str:
        attempts.append(1)
        raise RuntimeError("Tool remote has an output schema but did not return structured content")

    agent = create_agent(
        ScenarioModel(responses=[call("remote", "first"), call("remote", "retry"), AIMessage(content="inspect")]),
        tools=[StructuredTool.from_function(remote, description="remote")],
        middleware=[TaskReliabilityMiddleware(), McpToolErrorMiddleware()],
        checkpointer=InMemorySaver(),
    )
    state = invoke(agent)
    assert len(attempts) == 1
    assert state["tool_observations"]["first"]["outcome"] == "outcome_unknown"


def test_control_flow_is_not_swallowed() -> None:
    def remote() -> str:
        raise GraphInterrupt(())

    agent = graph([call("remote", "interrupt")], tools=[StructuredTool.from_function(remote, description="remote")])
    state = invoke(agent)
    assert "tool_observations" not in state or not state["tool_observations"]


def test_existing_command_updates_and_readonly_artifact_are_preserved() -> None:
    def remote() -> Command[Any]:
        return Command(
            update={"messages": [ToolMessage("checked", tool_call_id="remote", artifact={"read_only": True})]}
        )

    agent = graph(
        [
            call("remote", "remote"),
            checkpoint("done", status="completed", evidence=["remote"]),
            AIMessage(content="checked"),
        ],
        tools=[StructuredTool.from_function(remote, description="remote")],
    )
    state = invoke(agent)
    assert state["messages"][-2].status == "success"
    assert state["tool_observations"]["remote"]["verification"] is True


def test_progress_survives_rebuild_and_message_compaction_without_cross_thread_leak(tmp_path: Any) -> None:
    config = {"configurable": {"thread_id": "one"}}
    path = str(tmp_path / "sessions.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        agent = graph(
            [
                checkpoint("wait", status="blocked", blocker="OAuth", next_step="inspect after authorization"),
                AIMessage(content="wait"),
            ],
            saver=saver,
        )
        invoke(agent)
        agent.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
    with SqliteSaver.from_conn_string(path) as saver:
        agent = graph([AIMessage(content="resume")], saver=saver)
        state = agent.get_state(config).values
        assert state["task_progress"]["wait"]["blocker"] == "OAuth"
        assert not state["messages"]
        assert not agent.get_state({"configurable": {"thread_id": "other"}}).values


@pytest.mark.asyncio
async def test_async_graph_records_parallel_results() -> None:
    read = StructuredTool.from_function(lambda: "ok", name="read_file", description="read")
    agent = graph(
        [
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": name} for name in ["a", "b"]]),
            checkpoint("done", status="completed", evidence=["a", "b"]),
            AIMessage(content="checked"),
        ],
        tools=[read],
    )
    state = await agent.ainvoke({"messages": [HumanMessage(content="read")]}, {"configurable": {"thread_id": "one"}})
    assert set(state["tool_observations"]) == {"a", "b"}
    assert state["messages"][-2].status == "success"
