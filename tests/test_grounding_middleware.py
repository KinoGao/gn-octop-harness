"""Tests for the grounding middleware (stale reads + empty evidence + thread isolation)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.middleware.grounding import (
    NO_EVIDENCE_NOTICE,
    STALE_READ_NOTICE,
    GroundingMiddleware,
)


def _request(
    tool_name: str,
    content: str = "",
    *,
    status: str = "success",
    thread_id: str | None = None,
) -> ToolCallRequest:
    req = MagicMock(spec=ToolCallRequest)
    req.tool_call = {
        "name": tool_name,
        "args": {"file_path": "report.md", "content": "x"},
        "id": "call-1",
        "type": "tool_call",
    }
    req.runtime = MagicMock(config={"configurable": {"thread_id": thread_id}} if thread_id else {})
    req._result = ToolMessage(content=content, tool_call_id="call-1", status=status)  # type: ignore[attr-defined]
    return req


async def _call(middleware: GroundingMiddleware, request: ToolCallRequest) -> ToolMessage:
    result = request._result  # type: ignore[attr-defined]
    handler = AsyncMock(return_value=result)
    produced = await middleware.awrap_tool_call(request, handler)
    assert isinstance(produced, ToolMessage)
    return produced


@pytest.mark.asyncio
async def test_write_after_read_invalidates_the_earlier_read() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "old numbers"))

    written = await _call(middleware, _request("write_file", "ok"))

    assert STALE_READ_NOTICE.format(path="report.md").strip() in str(written.content)


def test_sync_wrap_tool_call_invalidates_stale_read() -> None:
    middleware = GroundingMiddleware()
    read_req = _request("read_file", "old numbers", thread_id="t-sync")
    write_req = _request("write_file", "ok", thread_id="t-sync")

    middleware.wrap_tool_call(read_req, lambda _r: read_req._result)  # type: ignore[attr-defined]
    written = middleware.wrap_tool_call(write_req, lambda _r: write_req._result)  # type: ignore[attr-defined]

    assert isinstance(written, ToolMessage)
    assert STALE_READ_NOTICE.format(path="report.md").strip() in str(written.content)


@pytest.mark.asyncio
async def test_read_ledgers_are_isolated_per_thread() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "old", thread_id="thread-a"))

    other_thread_write = await _call(middleware, _request("write_file", "ok", thread_id="thread-b"))
    assert other_thread_write.content == "ok"

    same_thread_write = await _call(middleware, _request("write_file", "ok", thread_id="thread-a"))
    assert "stale" in str(same_thread_write.content)


@pytest.mark.asyncio
async def test_write_without_reading_produces_no_notice() -> None:
    middleware = GroundingMiddleware()

    written = await _call(middleware, _request("write_file", "ok"))

    assert written.content == "ok"


@pytest.mark.asyncio
async def test_second_write_after_invalidation_stays_quiet() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "old numbers"))
    await _call(middleware, _request("edit_file", "ok"))

    second = await _call(middleware, _request("edit_file", "ok"))

    assert second.content == "ok"


@pytest.mark.asyncio
async def test_reread_rearms_the_invalidation() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "v1"))
    await _call(middleware, _request("write_file", "ok"))
    await _call(middleware, _request("read_file", "v2"))

    written = await _call(middleware, _request("write_file", "ok"))

    assert "stale" in str(written.content)


@pytest.mark.asyncio
async def test_path_spellings_share_one_ledger_entry() -> None:
    middleware = GroundingMiddleware()
    read = _request("read_file", "old")
    read.tool_call["args"] = {"file_path": "/report.md"}
    await _call(middleware, read)

    written = await _call(middleware, _request("write_file", "ok"))

    assert "stale" in str(written.content)


@pytest.mark.asyncio
async def test_empty_result_is_labelled_as_no_evidence() -> None:
    middleware = GroundingMiddleware()

    produced = await _call(middleware, _request("grep", ""))

    assert NO_EVIDENCE_NOTICE.strip() in str(produced.content)


@pytest.mark.asyncio
async def test_error_result_is_labelled_as_no_evidence() -> None:
    middleware = GroundingMiddleware()

    produced = await _call(middleware, _request("read_file", "boom", status="error"))

    assert NO_EVIDENCE_NOTICE.strip() in str(produced.content)


@pytest.mark.asyncio
async def test_failed_write_does_not_invalidate_reads() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "old"))

    failed = await _call(middleware, _request("write_file", "", status="error"))
    assert "stale" not in str(failed.content)

    written = await _call(middleware, _request("write_file", "ok"))
    assert "stale" in str(written.content)


@pytest.mark.asyncio
async def test_content_blocks_are_flattened() -> None:
    middleware = GroundingMiddleware()
    request = _request("grep", "")
    request._result = ToolMessage(  # type: ignore[attr-defined]
        content=[{"type": "text", "text": "found 2 matches"}],
        tool_call_id="call-1",
    )

    produced = await _call(middleware, request)

    assert produced.content == [{"type": "text", "text": "found 2 matches"}]


@pytest.mark.asyncio
async def test_subagent_namespaced_tool_names_are_understood() -> None:
    middleware = GroundingMiddleware()
    await _call(middleware, _request("fs/read_file", "old"))

    written = await _call(middleware, _request("fs/write_file", "ok"))

    assert "stale" in str(written.content)


@pytest.mark.asyncio
async def test_non_tool_message_is_passed_through() -> None:
    middleware = GroundingMiddleware()
    request = _request("read_file", "old")
    sentinel: Any = object()
    handler = AsyncMock(return_value=sentinel)

    assert await middleware.awrap_tool_call(request, handler) is sentinel


@pytest.mark.asyncio
async def test_image_result_is_evidence_not_an_empty_result() -> None:
    middleware = GroundingMiddleware()
    request = _request("screenshot", "")
    request._result = ToolMessage(  # type: ignore[attr-defined]
        content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}],
        tool_call_id="call-1",
    )

    produced = await _call(middleware, request)

    assert "no usable evidence" not in str(produced.content)


@pytest.mark.asyncio
async def test_json_error_payload_is_not_corrupted_and_preserves_reliability_guard() -> None:
    import json

    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from octop_harness.middleware.task_reliability import TaskReliabilityMiddleware

    grounding = GroundingMiddleware()
    reliability = TaskReliabilityMiddleware()
    payload = json.dumps({"status": "authorization_required", "code": "authorization_required", "message": "auth"})
    err = ToolMessage(content=payload, tool_call_id="call-1", status="error")
    req = ToolCallRequest(
        tool_call={"name": "connector_call", "args": {"server_name": "fc"}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={"messages": [HumanMessage(content="run", id="u1")], "tool_observations": {}},
        runtime=None,  # type: ignore[arg-type]
    )

    # Mirrors agent.py ordering: GroundingMiddleware outer -> TaskReliabilityMiddleware inner.
    first = await grounding.awrap_tool_call(
        req,
        lambda r: reliability.awrap_tool_call(r, AsyncMock(return_value=err)),
    )
    assert isinstance(first, Command)
    assert first.update is not None
    obs = first.update["tool_observations"]["call-1"]
    assert obs["outcome"] == "authorization_required"
    msg = first.update["messages"][0]
    assert json.loads(msg.content)["status"] == "authorization_required"

    # Second identical call must be blocked immediately by TaskReliabilityMiddleware.
    req2 = ToolCallRequest(
        tool_call={"name": "connector_call", "args": {"server_name": "fc"}, "id": "call-2", "type": "tool_call"},
        tool=None,
        state={"messages": [HumanMessage(content="run", id="u1")], "tool_observations": {"call-1": obs}},
        runtime=None,  # type: ignore[arg-type]
    )
    second = await grounding.awrap_tool_call(
        req2,
        lambda r: reliability.awrap_tool_call(r, AsyncMock(return_value=err)),
    )
    assert isinstance(second, ToolMessage)
    assert "repeated_call_blocked" in str(second.content)


@pytest.mark.asyncio
async def test_multimodal_list_error_preserves_media_blocks() -> None:
    middleware = GroundingMiddleware()
    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    request = _request("browser_use", "")
    request._result = ToolMessage(  # type: ignore[attr-defined]
        content=[image_block, {"type": "text", "text": "click failed"}],
        tool_call_id="call-1",
        status="error",
    )

    produced = await _call(middleware, request)

    assert isinstance(produced.content, list)
    assert produced.content[0] == image_block
    assert any(
        isinstance(b, dict) and b.get("type") == "text" and NO_EVIDENCE_NOTICE.strip() in str(b.get("text"))
        for b in produced.content
    )


@pytest.mark.asyncio
async def test_command_wrapped_write_receives_stale_read_notice() -> None:
    from langgraph.types import Command

    middleware = GroundingMiddleware()
    await _call(middleware, _request("read_file", "old"))

    write_req = _request("write_file", "ok")
    cmd = Command(
        update={
            "messages": [ToolMessage(content="ok", tool_call_id="call-1")],
            "tool_observations": {"call-1": {"outcome": "success"}},
        }
    )
    out = await middleware.awrap_tool_call(write_req, AsyncMock(return_value=cmd))
    assert isinstance(out, Command)
    assert out.update is not None
    assert "stale" in str(out.update["messages"][0].content)
