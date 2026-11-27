"""Tests for phase-2 harness intelligence upgrades.

Covers:
- RubricMiddleware integration, role_bindings["verifier"] resolution, and ChatRequest.rubric.
- Subagent fork mode, structured response_format, factory model resolution, and read-only MCP passthrough.
- Ask/Plan read-only MCP tool allowance.
- Context-aware memory recall query expansion and cross-turn recall deduplication.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import var_child_runnable_config
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.agent import HarnessAgent
from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.config import HarnessAgentConfig, ModelConfig, ProviderConfig
from octop_harness.mcp import _postprocess_mcp_tools, is_readonly_tool, readonly_mcp_tool_names
from octop_harness.middleware.conversation_mode import (
    ConversationModeMiddleware,
    apply_conversation_mode_to_request,
)
from octop_harness.middleware.mcp_tools import MCPToolMiddleware
from octop_harness.middleware.memory_recall import (
    build_recall_query,
    deduplicate_recalled_text,
    stamp_recall_snapshot,
)
from octop_harness.protocols.langgraph import _graph_input, _is_internal_model_stream
from octop_harness.request import ChatRequest
from octop_harness.subagents.loader import parse_agent_markdown


@contextmanager
def _configurable(**kwargs: object):
    token = var_child_runnable_config.set({"configurable": kwargs})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _provider() -> ProviderConfig:
    return ProviderConfig(
        id="p",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        models=[ModelConfig(id="main"), ModelConfig(id="fast"), ModelConfig(id="verifier")],
    )


def _mcp_tool(name: str, *, read_only: bool = False, annotation_hint: bool = False) -> StructuredTool:
    metadata: dict[str, Any] = {}
    if read_only:
        metadata["read_only"] = True
    if annotation_hint:
        metadata["annotations"] = {"readOnlyHint": True}
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=name,
        metadata=metadata or None,
    )


# ---------------------------------------------------------------------------
# 1. Rubric integration
# ---------------------------------------------------------------------------


def test_chat_request_rubric_flows_into_graph_input_and_filters_internal_stream() -> None:
    cfg = ChatRequest(messages="solve", rubric="Must verify output with tests.").to_runnable_config()
    assert cfg["configurable"]["rubric"] == "Must verify output with tests."
    payload = _graph_input([HumanMessage(content="solve")], cfg)
    assert payload["rubric"] == "Must verify output with tests."

    plain = _graph_input([HumanMessage(content="hi")], ChatRequest(messages="hi").to_runnable_config())
    assert plain["rubric"] == ""
    assert _is_internal_model_stream({"lc_source": "rubric_grader"}) is True


def test_rubric_middleware_mounts_by_default_and_respects_config_toggle(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        role_bindings={"verifier": "p/verifier"},
        rubric_max_iterations=3,
        memory_enabled=False,
        checkpointer=False,
    )
    loaded = HarnessAgentConfig.from_dict(cfg.to_dict())
    assert loaded.rubric_enabled is True
    assert loaded.rubric_max_iterations == 3
    assert loaded.subagent_inherit_readonly_mcp is True

    resolved_refs: list[str] = []

    def fake_get_chat_model(self: Any, ref: str, **_kwargs: Any) -> MagicMock:
        resolved_refs.append(ref)
        return MagicMock()

    with (
        patch("octop_harness.llm.factory.ChatModelFactory.get_chat_model", fake_get_chat_model),
        patch("deepagents.create_deep_agent", side_effect=fake_create),
    ):
        HarnessAgent(cfg)

    mw_names = [type(m).__name__ for m in captured["middleware"]]
    assert "RubricMiddleware" in mw_names
    assert "p/verifier" in resolved_refs

    captured.clear()
    disabled_cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        rubric_enabled=False,
        memory_enabled=False,
        checkpointer=False,
    )
    with (
        patch("octop_harness.llm.factory.ChatModelFactory.get_chat_model", fake_get_chat_model),
        patch("deepagents.create_deep_agent", side_effect=fake_create),
    ):
        HarnessAgent(disabled_cfg)
    assert "RubricMiddleware" not in [type(m).__name__ for m in captured["middleware"]]


# ---------------------------------------------------------------------------
# 2. Read-only MCP detection & Subagent upgrades
# ---------------------------------------------------------------------------


def test_readonly_mcp_detection_and_metadata_normalization() -> None:
    ro_meta = _mcp_tool("ding_list", read_only=True)
    ro_hint = _mcp_tool("ding_get", annotation_hint=True)
    rw_tool = _mcp_tool("ding_send")

    assert is_readonly_tool(ro_meta) is True
    assert is_readonly_tool(ro_hint) is True
    assert is_readonly_tool(rw_tool) is False
    assert readonly_mcp_tool_names([ro_meta, ro_hint, rw_tool]) == frozenset({"ding_list", "ding_get"})

    _postprocess_mcp_tools([ro_hint, rw_tool], server_name="ding", tool_arg_aliases={})
    assert isinstance(ro_hint.metadata, dict)
    assert ro_hint.metadata["read_only"] is True
    assert not (isinstance(rw_tool.metadata, dict) and rw_tool.metadata.get("read_only"))


def test_parse_agent_markdown_supports_fork_and_response_format(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    ws = BackendWorkspace(backend, tmp_path)
    md = """---
name: Reviewer
id: reviewer
description: Reviews changes
mode: fork
model: fast
skills:
  - ignored-on-fork
response_format:
  type: object
  properties:
    verdict:
      type: string
---

Check carefully.
"""
    spec = parse_agent_markdown(md, path_fragment="agents/reviewer.md", workspace=ws)
    assert spec is not None
    assert spec.get("mode") == "fork"
    assert "skills" not in spec
    assert spec.get("model") == "fast"
    assert spec.get("response_format") == {"type": "object", "properties": {"verdict": {"type": "string"}}}


def test_subagent_inherits_only_readonly_mcp_with_guard_middleware_and_resolves_role_model(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "explorer.md").write_text(
        "---\nname: Explorer\ndescription: Reads context\nmodel: explorer\n---\nExplore.\n",
        encoding="utf-8",
    )
    ro_mcp = _mcp_tool("ding_search", annotation_hint=True)
    rw_mcp = _mcp_tool("ding_post")
    regular = StructuredTool.from_function(lambda: "ok", name="current_time", description="time")

    cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        role_bindings={"explorer": "p/fast"},
        mcp_server_configs={"ding": {"transport": "stdio", "command": "echo"}},
        memory_enabled=False,
        checkpointer=False,
    )
    captured: dict[str, Any] = {}
    models_by_ref: dict[str, MagicMock] = {}

    def fake_get_chat_model(self: Any, ref: str, **_kwargs: Any) -> MagicMock:
        models_by_ref.setdefault(ref, MagicMock(name=f"model:{ref}"))
        return models_by_ref[ref]

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("octop_harness.llm.factory.ChatModelFactory.get_chat_model", fake_get_chat_model),
        patch("octop_harness.agent.load_mcp_tools", return_value=[ro_mcp, rw_mcp]),
        patch.object(HarnessAgent, "_build_tools", return_value=[regular, ro_mcp, rw_mcp]),
        patch("deepagents.create_deep_agent", side_effect=fake_create),
    ):
        HarnessAgent(cfg)

    sub = captured["subagents"][0]
    tool_names = [getattr(t, "name", "") for t in sub["tools"]]
    assert "current_time" in tool_names
    assert "ding_search" in tool_names
    assert "ding_post" not in tool_names
    assert sub["model"] is models_by_ref["p/fast"]
    assert any(isinstance(mw, MCPToolMiddleware) for mw in sub.get("middleware", []))


# ---------------------------------------------------------------------------
# 3. Ask/Plan mode read-only MCP allowance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_mode_allows_readonly_mcp_only_when_opted_in() -> None:
    ro = _mcp_tool("ding_search", annotation_hint=True)
    rw = _mcp_tool("ding_post")
    request = ModelRequest(model=MagicMock(), messages=[], tools=[ro, rw])
    with _configurable(conversation_mode="plan"):
        default_out = apply_conversation_mode_to_request(request)
    assert list(default_out.tools or []) == []

    with _configurable(conversation_mode="plan", allow_readonly_mcp_in_restricted_modes=True):
        out = apply_conversation_mode_to_request(request)
    assert [getattr(t, "name", None) for t in (out.tools or [])] == ["ding_search"]

    mw = ConversationModeMiddleware()
    ok = ToolMessage(content="found", tool_call_id="c1")
    ro_call = ToolCallRequest(
        tool_call={"name": "ding_search", "args": {}, "id": "c1", "type": "tool_call"},
        tool=ro,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    rw_call = ToolCallRequest(
        tool_call={"name": "ding_post", "args": {}, "id": "c2", "type": "tool_call"},
        tool=rw,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    with _configurable(conversation_mode="plan"):
        default_blocked = await mw.awrap_tool_call(ro_call, AsyncMock(return_value=ok))
    assert isinstance(default_blocked, ToolMessage)
    assert default_blocked.status == "error"

    with _configurable(conversation_mode="plan", allow_readonly_mcp_in_restricted_modes=True):
        assert await mw.awrap_tool_call(ro_call, AsyncMock(return_value=ok)) is ok
        blocked = await mw.awrap_tool_call(rw_call, AsyncMock(return_value=ok))
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"

    cfg = ChatRequest(
        messages="plan",
        conversation_mode="plan",
        mcp_servers=["ding"],
        allow_readonly_mcp_in_restricted_modes=True,
    ).to_runnable_config()
    assert cfg["configurable"]["mcp_servers"] == ["ding"]
    assert cfg["configurable"]["allow_readonly_mcp_in_restricted_modes"] is True
    assert cfg["configurable"]["skills"] == []


# ---------------------------------------------------------------------------
# 4. Smarter memory recall (anaphora expansion & cross-turn deduplication)
# ---------------------------------------------------------------------------


def test_build_recall_query_expands_anaphoric_or_short_followups() -> None:
    messages = [
        HumanMessage(content="帮我整理项目助手本周的发版清单"),
        AIMessage(content="好的, 已经列出 1.0.4b5 的变更项。"),
        HumanMessage(content="按上次说的继续改一下"),
    ]
    expanded = build_recall_query(messages)
    assert "项目助手" in expanded
    assert "按上次说的继续改一下" in expanded

    standalone = HumanMessage(content="请详细分析 PostgreSQL 连接池在高并发下的超时参数配置以及最佳连接数计算公式。")
    assert build_recall_query([*messages, standalone]) == standalone.content


def test_deduplicate_recalled_text_drops_already_injected_memories() -> None:
    first = stamp_recall_snapshot(
        HumanMessage(content="hi"),
        "## Memories\n- User prefers concise Chinese replies.\n- Deploy target is staging-app.",
    )
    # Exact duplicate -> suppressed completely.
    assert (
        deduplicate_recalled_text(
            "## Memories\n- User prefers concise Chinese replies.\n- Deploy target is staging-app.",
            [first],
        )
        == ""
    )
    # Mixed -> keeps only genuinely new memory line.
    partial = deduplicate_recalled_text(
        "## Memories\n- User prefers concise Chinese replies.\n- Primary branch is dev.",
        [first],
    )
    assert "Primary branch is dev." in partial
    assert "concise Chinese replies" not in partial


def test_rubric_grader_synthetic_message_ignored_by_turn_and_memory() -> None:
    from octop_harness.middleware.memory import MemoryMiddleware
    from octop_harness.middleware.task_reliability import _turn

    user_msg = HumanMessage(content="build feature X", id="u1")
    grader_msg = HumanMessage(
        content="Revise criterion 2",
        id="g1",
        name="rubric_grader",
        additional_kwargs={"lc_source": "rubric_grader"},
    )
    assert _turn({"messages": [user_msg, grader_msg]}) == "u1"

    service = MagicMock()
    mm = MemoryMiddleware(service=service, recall_inject_enabled=True, jsonl_enabled=False)
    assert mm.before_model({"messages": [user_msg, grader_msg]}, MagicMock()) is None
    service.recall.assert_not_called()


# ---------------------------------------------------------------------------
# 5. End-to-end P0 regressions (Rubric stickiness, Ask+16 MCP, Fork isolation)
# ---------------------------------------------------------------------------


def _make_scripted_model(responder: Any, *, model_name: str = "scripted") -> Any:
    """Build a minimal BaseChatModel subclass driven by ``responder(messages, bound_tool_names)``."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class _ScriptedChatModel(BaseChatModel):
        _responder: Any = PrivateAttr()
        _model_name: str = PrivateAttr()
        _bound_tools: list[str] = PrivateAttr(default_factory=list)
        _calls: list[list[Any]] = PrivateAttr(default_factory=list)

        def __init__(self, responder_fn: Any, name: str) -> None:
            super().__init__()
            self._responder = responder_fn
            self._model_name = name
            self._bound_tools = []
            self._calls = []

        @property
        def _llm_type(self) -> str:
            return f"scripted-{self._model_name}"

        @property
        def call_count(self) -> int:
            return len(self._calls)

        def bind_tools(self, tools: Any, **_kwargs: Any) -> _ScriptedChatModel:
            names: list[str] = []
            for tool in tools or []:
                if isinstance(tool, dict):
                    fn = tool.get("function")
                    name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
                    if name:
                        names.append(str(name))
                elif hasattr(tool, "name") and tool.name:
                    names.append(str(tool.name))
                elif hasattr(tool, "__name__"):
                    names.append(str(tool.__name__))
            self._bound_tools = names
            return self

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            self._calls.append(list(messages))
            ai_msg = self._responder(list(messages), list(self._bound_tools), len(self._calls))
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        async def _agenerate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return _ScriptedChatModel(responder, model_name)


@pytest.mark.asyncio
async def test_rubric_does_not_retrigger_on_next_turn_without_rubric(tmp_path: Path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    main_replies = [
        AIMessage(content="Draft answer without table"),
        AIMessage(content="Revised answer with table\n| a | b |\n| - | - |\n| 1 | 2 |"),
        AIMessage(content="You are welcome!"),
    ]
    main_model = _make_scripted_model(
        lambda _msgs, _tools, idx: main_replies[idx - 1],
        model_name="main",
    )

    grader_replies = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "grade-1",
                    "name": "GraderResponse",
                    "args": {
                        "result": "needs_revision",
                        "explanation": "Add a markdown table.",
                        "criteria": [
                            {
                                "name": "Response includes a markdown table",
                                "passed": False,
                                "gap": "Missing table",
                            }
                        ],
                    },
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "grade-2",
                    "name": "GraderResponse",
                    "args": {
                        "result": "satisfied",
                        "explanation": "Table included.",
                        "criteria": [
                            {
                                "name": "Response includes a markdown table",
                                "passed": True,
                            }
                        ],
                    },
                }
            ],
        ),
    ]
    grader_model = _make_scripted_model(
        lambda _msgs, _tools, idx: grader_replies[idx - 1],
        model_name="verifier",
    )

    models = {"p/main": main_model, "p/verifier": grader_model}
    cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        role_bindings={"verifier": "p/verifier"},
        rubric_enabled=True,
        rubric_max_iterations=2,
        memory_enabled=False,
        web_search_tools=False,
        checkpointer=InMemorySaver(),
    )

    with patch(
        "octop_harness.llm.factory.ChatModelFactory.get_chat_model",
        lambda _self, ref, **_kw: models[ref],
    ):
        agent = HarnessAgent(cfg)
        turn1 = await agent.call(
            ChatRequest(messages="Compare A and B", rubric="Must include a table", thread_id="rubric-t1")
        )
        assert grader_model.call_count == 2
        assert "Revised answer with table" in str(turn1["messages"][-1].content)

        turn2 = await agent.call(ChatRequest(messages="Thanks, bye!", thread_id="rubric-t1"))
        assert grader_model.call_count == 2
        assert turn2["messages"][-1].content == "You are welcome!"


@pytest.mark.asyncio
async def test_ask_mode_over_15_readonly_mcp_allows_tool_search_and_blocks_write_mcp(tmp_path: Path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    write_invocations: list[str] = []

    def _make_ro(name: str) -> StructuredTool:
        return StructuredTool.from_function(
            func=lambda _n=name: f"ok:{_n}",
            name=name,
            description=f"Readonly tool {name}",
            metadata={"read_only": True},
        )

    def _make_rw(name: str) -> StructuredTool:
        def _run(_n: str = name) -> str:
            write_invocations.append(_n)
            return f"wrote:{_n}"

        return StructuredTool.from_function(
            func=_run,
            name=name,
            description=f"Write tool {name}",
        )

    ro_tools = [_make_ro(f"ding_read_{i:02d}") for i in range(16)]
    rw_tools = [_make_rw("ding_write_0"), _make_rw("ding_write_1")]

    def _respond(messages: list[Any], bound_tools: list[str], step: int) -> AIMessage:
        if step == 1:
            assert "tool_search" in bound_tools
            assert "ding_write_0" not in bound_tools
            assert "ding_write_1" not in bound_tools
            return AIMessage(
                content="",
                tool_calls=[{"id": "tc-search", "name": "tool_search", "args": {"query": "ding read 00"}}],
            )
        if step == 2:
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            assert tool_msgs
            assert tool_msgs[-1].status != "error"
            assert "ding_read_00" in str(tool_msgs[-1].content)
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc-ro", "name": "ding_read_00", "args": {}},
                    {"id": "tc-rw", "name": "ding_write_0", "args": {}},
                ],
            )
        tool_by_id = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        assert tool_by_id["tc-ro"].status != "error"
        assert "ok:ding_read_00" in str(tool_by_id["tc-ro"].content)
        assert tool_by_id["tc-rw"].status == "error"
        return AIMessage(content="Ask MCP check passed")

    main_model = _make_scripted_model(_respond, model_name="main")
    cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        mcp_server_configs={"ding": {"transport": "stdio", "command": "echo"}},
        defer_mcp_tools_threshold=15,
        tool_search_mode="client",
        rubric_enabled=False,
        memory_enabled=False,
        web_search_tools=False,
        checkpointer=InMemorySaver(),
    )

    with (
        patch("octop_harness.llm.factory.ChatModelFactory.get_chat_model", lambda _self, _ref, **_kw: main_model),
        patch("octop_harness.agent.load_mcp_tools", return_value=[*ro_tools, *rw_tools]),
    ):
        agent = HarnessAgent(cfg)
        result = await agent.call(
            ChatRequest(
                messages="Inspect readonly MCP tools",
                conversation_mode="ask",
                mcp_servers=["ding"],
                allow_readonly_mcp_in_restricted_modes=True,
                thread_id="ask-mcp-16",
            )
        )

    assert result["messages"][-1].content == "Ask MCP check passed"
    assert write_invocations == []


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:The feature `forked subagents` is in beta")
async def test_fork_subagent_uses_explicit_model_and_leaves_parent_memory_clean(tmp_path: Path) -> None:
    import json

    from langgraph.checkpoint.memory import InMemorySaver

    from octop_harness.middleware.memory import MemoryMiddleware

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Forked code reviewer\nmode: fork\nmodel: p/fast\n---\nReview carefully.\n",
        encoding="utf-8",
    )

    def _parent_respond(messages: list[Any], _bound: list[str], step: int) -> AIMessage:
        if step == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-fork",
                        "name": "task",
                        "args": {"subagent_type": "reviewer", "description": "INTERNAL_FORK_TASK"},
                    }
                ],
            )
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert any("INTERNAL_FORK_REPLY" in str(m.content) for m in tool_msgs)
        return AIMessage(content="FINAL_USER_REPLY")

    parent_model = _make_scripted_model(_parent_respond, model_name="parent")
    fork_model = _make_scripted_model(
        lambda _msgs, _bound, _step: AIMessage(content="INTERNAL_FORK_REPLY"),
        model_name="fork",
    )
    models = {"p/main": parent_model, "p/fast": fork_model}

    spy_service = MagicMock()
    spy_service.recall.return_value = MagicMock(rendered="")
    jsonl_dir = tmp_path / "sessions"
    memory_mw = MemoryMiddleware(
        service=spy_service,
        recall_inject_enabled=True,
        capture_enabled=True,
        jsonl_enabled=True,
        jsonl_dir=jsonl_dir,
    )

    cfg = HarnessAgentConfig(
        workspace_dir=tmp_path,
        providers=[_provider()],
        default_model="p/main",
        rubric_enabled=False,
        memory_enabled=False,
        memory_jsonl_enabled=False,
        session_log_enabled=False,
        web_search_tools=False,
        checkpointer=InMemorySaver(),
        middleware=[memory_mw],
    )

    with patch(
        "octop_harness.llm.factory.ChatModelFactory.get_chat_model",
        lambda _self, ref, **_kw: models[ref],
    ):
        agent = HarnessAgent(cfg)
        result = await agent.call(ChatRequest(messages="REAL_USER_PROMPT", thread_id="fork-t1"))

    with memory_mw.pause_background(timeout=5.0):
        pass
    assert result["messages"][-1].content == "FINAL_USER_REPLY"
    assert fork_model.call_count == 1
    assert parent_model.call_count == 2

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    records = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    recorded_contents = [str(rec.get("content") or "") for rec in records]
    assert any("REAL_USER_PROMPT" in text for text in recorded_contents)
    assert any("FINAL_USER_REPLY" in text for text in recorded_contents)
    assert not any("INTERNAL_FORK_REPLY" in text for text in recorded_contents)
    assert not any("The messages above are a prior conversation" in text for text in recorded_contents)

    assert spy_service.recall.call_count == 1
    assert spy_service.capture_turn.call_count == 1
    capture_kwargs = spy_service.capture_turn.call_args.kwargs
    assert capture_kwargs["user"] == "REAL_USER_PROMPT"
    assert capture_kwargs["assistant"] == "FINAL_USER_REPLY"
