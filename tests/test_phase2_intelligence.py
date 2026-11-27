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
    assert "rubric" not in plain
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
async def test_plan_mode_allows_readonly_mcp_and_blocks_mutating_mcp() -> None:
    ro = _mcp_tool("ding_search", annotation_hint=True)
    rw = _mcp_tool("ding_post")
    request = ModelRequest(model=MagicMock(), messages=[], tools=[ro, rw])
    with _configurable(conversation_mode="plan"):
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
    assert cfg["configurable"]["skills"] == []


# ---------------------------------------------------------------------------
# 4. Smarter memory recall (anaphora expansion & cross-turn deduplication)
# ---------------------------------------------------------------------------


def test_build_recall_query_expands_anaphoric_or_short_followups() -> None:
    messages = [
        HumanMessage(content="帮我整理港能助手本周的发版清单"),
        AIMessage(content="好的, 已经列出 1.0.4b5 的变更项。"),
        HumanMessage(content="按上次说的继续改一下"),
    ]
    expanded = build_recall_query(messages)
    assert "港能助手" in expanded
    assert "按上次说的继续改一下" in expanded

    standalone = HumanMessage(content="请详细分析 PostgreSQL 连接池在高并发下的超时参数配置以及最佳连接数计算公式。")
    assert build_recall_query([*messages, standalone]) == standalone.content


def test_deduplicate_recalled_text_drops_already_injected_memories() -> None:
    first = stamp_recall_snapshot(
        HumanMessage(content="hi"),
        "## Memories\n- User prefers concise Chinese replies.\n- Deploy target is gangneng-smoke.",
    )
    # Exact duplicate -> suppressed completely.
    assert (
        deduplicate_recalled_text(
            "## Memories\n- User prefers concise Chinese replies.\n- Deploy target is gangneng-smoke.",
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
