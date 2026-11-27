"""Tests for ``ToolResultBudgetMiddleware``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.context_usage import estimate_tokens
from octop_harness.middleware.tool_result_budget import (
    DEFAULT_MAX_TOKENS,
    ToolResultBudgetMiddleware,
    _safe_stem,
    _truncate_head_tail,
)


@pytest.fixture
def workspace(tmp_path: Path) -> BackendWorkspace:
    from deepagents.backends import FilesystemBackend

    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    return BackendWorkspace(backend, tmp_path)


def _request(*, tool: str = "web_fetch", call_id: str = "call-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,  # type: ignore[arg-type]
        state={},  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
    )


async def _run(
    mw: ToolResultBudgetMiddleware,
    *,
    content: str,
    tool: str = "web_fetch",
    call_id: str = "call-1",
) -> ToolMessage:
    handler = AsyncMock(return_value=ToolMessage(content=content, tool_call_id=call_id))
    result = await mw.awrap_tool_call(_request(tool=tool, call_id=call_id), handler)
    assert isinstance(result, ToolMessage)
    return result


@pytest.mark.asyncio
async def test_small_result_is_untouched() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=DEFAULT_MAX_TOKENS)
    body = "short output"
    result = await _run(mw, content=body)
    assert result.content == body


@pytest.mark.asyncio
async def test_disabled_budget_keeps_everything() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=0)
    body = "x" * 200_000
    result = await _run(mw, content=body)
    assert result.content == body


@pytest.mark.asyncio
async def test_block_content_is_left_alone() -> None:
    """Binary/structured blocks belong to MediaOffloadMiddleware, not here."""
    mw = ToolResultBudgetMiddleware(max_tokens=10)
    blocks: list[dict[str, Any]] = [{"type": "image", "base64": "A" * 50_000}]
    handler = AsyncMock(return_value=ToolMessage(content=blocks, tool_call_id="call-1"))
    result = await mw.awrap_tool_call(_request(), handler)
    assert isinstance(result, ToolMessage)
    assert result.content == blocks


@pytest.mark.asyncio
async def test_oversized_result_is_truncated_to_budget() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=500)
    body = "".join(f"line {i:05d} payload\n" for i in range(20_000))
    result = await _run(mw, content=body)

    assert isinstance(result.content, str)
    assert estimate_tokens(result.content) <= 500
    assert "output truncated by the harness" in result.content


def test_sync_wrap_tool_call_truncates_oversized_result() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=300)
    body = "line payload\n" * 10_000
    result = mw.wrap_tool_call(_request(), lambda _r: ToolMessage(content=body, tool_call_id="call-1"))

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, str)
    assert estimate_tokens(result.content) <= 300
    assert "output truncated by the harness" in result.content


@pytest.mark.asyncio
async def test_head_and_tail_survive_the_cut() -> None:
    """The failure message at the end must survive; a prefix cut would hide it."""
    mw = ToolResultBudgetMiddleware(max_tokens=500)
    head = "HEAD-MARKER\n"
    tail = "\nTAIL-MARKER: Traceback (most recent call last)"
    middle = "filler " * 40_000
    result = await _run(mw, content=f"{head}{middle}{tail}")

    assert isinstance(result.content, str)
    assert result.content.startswith("HEAD-MARKER")
    assert result.content.endswith(tail.strip())
    assert "omitted" in result.content


@pytest.mark.asyncio
async def test_marker_reports_omitted_size() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=200)
    result = await _run(mw, content="z" * 100_000)

    assert isinstance(result.content, str)
    assert "characters" in result.content
    assert "tokens omitted" in result.content


@pytest.mark.asyncio
async def test_metadata_of_the_tool_message_is_preserved() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=100)
    original = ToolMessage(
        content="y" * 50_000,
        tool_call_id="call-9",
        name="web_fetch",
        status="error",
    )
    handler = AsyncMock(return_value=original)
    result = await mw.awrap_tool_call(_request(call_id="call-9"), handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-9"
    assert result.name == "web_fetch"
    assert result.status == "error"


@pytest.mark.asyncio
async def test_spill_writes_full_text_and_points_at_it(workspace: BackendWorkspace, tmp_path: Path) -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=200, spill_dir=".tool-outputs", workspace=workspace)
    body = "".join(f"row {i}\n" for i in range(5_000))
    result = await _run(mw, content=body, tool="web_fetch", call_id="call-1")

    assert isinstance(result.content, str)
    spilled = tmp_path / ".tool-outputs" / "web_fetch-call-1.txt"
    assert spilled.is_file()
    assert spilled.read_text(encoding="utf-8") == body
    assert ".tool-outputs/web_fetch-call-1.txt" in result.content


@pytest.mark.asyncio
async def test_spill_failure_does_not_break_the_tool_call(tmp_path: Path) -> None:
    """A read-only or unwritable workspace must not turn into a failed turn."""

    class _Broken:
        def write_text(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("read-only")

    mw = ToolResultBudgetMiddleware(max_tokens=100, spill_dir=".tool-outputs", workspace=_Broken())
    result = await _run(mw, content="q" * 50_000)

    assert isinstance(result.content, str)
    assert "output truncated by the harness" in result.content


@pytest.mark.asyncio
async def test_empty_spill_dir_drops_the_middle_silently() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=100, spill_dir="", workspace=None)
    result = await _run(mw, content="w" * 50_000)

    assert isinstance(result.content, str)
    assert "full output:" not in result.content


def test_truncate_head_tail_handles_utf8_and_short_budget() -> None:
    text = "项目助手" * 20_000
    out = _truncate_head_tail(text, max_tokens=20, original_tokens=20_000, spill_path=None)
    assert out
    assert "omitted" in out


def test_budget_truncates_tool_message_inside_command() -> None:
    from langgraph.types import Command

    mw = ToolResultBudgetMiddleware(max_tokens=100)
    huge = "A" * 5000 + "B" * 5000
    cmd = Command(
        update={
            "messages": [ToolMessage(content=huge, tool_call_id="c-cmd")],
            "tool_observations": {"c-cmd": {"outcome": "success"}},
        }
    )
    req = _request(tool="web_fetch", call_id="c-cmd")
    out = mw.wrap_tool_call(req, lambda _r: cmd)
    assert isinstance(out, Command)
    assert out.update is not None
    assert out.update["tool_observations"] == {"c-cmd": {"outcome": "success"}}
    msg = out.update["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "omitted" in msg.content
    assert len(msg.content) < len(huge)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("web_fetch-call-1", "web_fetch-call-1"),
        ("mcp/foo bar:1", "mcp-foo-bar-1"),
        ("///", "tool-output"),
    ],
)
def test_safe_stem(raw: str, expected: str) -> None:
    assert _safe_stem(raw) == expected


@pytest.mark.asyncio
async def test_list_text_blocks_are_truncated_while_media_blocks_survive() -> None:
    mw = ToolResultBudgetMiddleware(max_tokens=100)
    image_block: dict[str, Any] = {"type": "image", "base64": "A" * 50_000}
    audio_block: dict[str, Any] = {"type": "audio", "data": "B" * 20_000}
    huge_text = "HEAD-START\n" + ("payload " * 20_000) + "\nTAIL-END"
    blocks: list[dict[str, Any]] = [
        image_block,
        {"type": "text", "text": huge_text, "annotations": [{"kind": "note"}]},
        audio_block,
    ]
    handler = AsyncMock(return_value=ToolMessage(content=blocks, tool_call_id="call-list"))
    result = await mw.awrap_tool_call(_request(tool="mcp_fetch", call_id="call-list"), handler)

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, list)
    assert len(result.content) == 3
    assert result.content[0] == image_block
    assert result.content[2] == audio_block
    text_block = result.content[1]
    assert isinstance(text_block, dict)
    assert text_block["type"] == "text"
    assert text_block["annotations"] == [{"kind": "note"}]
    assert estimate_tokens(text_block["text"]) <= 100
    assert text_block["text"].startswith("HEAD-START")
    assert text_block["text"].endswith("TAIL-END")
    assert "omitted" in text_block["text"]


@pytest.mark.asyncio
async def test_awrap_tool_call_spills_via_awrite_text_without_blocking_event_loop() -> None:
    written: dict[str, str] = {}

    class _AsyncOnlyWorkspace:
        def write_text(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("desktop relay backend cannot block the event loop; use the async API")

        async def awrite_text(self, path: str, content: str, *, force: bool = True) -> None:
            assert force is True
            written[path] = content

    mw = ToolResultBudgetMiddleware(
        max_tokens=100,
        spill_dir=".tool-outputs",
        workspace=_AsyncOnlyWorkspace(),
    )
    body = "line\n" * 10_000
    result = await _run(mw, content=body, tool="localfs_read", call_id="call-async")

    assert isinstance(result.content, str)
    assert ".tool-outputs/localfs_read-call-async.txt" in written
    assert written[".tool-outputs/localfs_read-call-async.txt"] == body
    assert ".tool-outputs/localfs_read-call-async.txt" in result.content


@pytest.mark.parametrize(
    "bad_dir",
    [
        "~/evil-spill",
        "~",
        "/tmp/evil-spill",
        "../outside",
        "sub/../outside",
        ".",
        "C:\\evil",
    ],
)
def test_spill_dir_validation_rejects_unsafe_paths(tmp_path: Path, bad_dir: str) -> None:
    from octop_harness.config import HarnessAgentConfig

    with pytest.raises(ValueError, match="tool_result_spill_dir"):
        ToolResultBudgetMiddleware(max_tokens=100, spill_dir=bad_dir)

    with pytest.raises(ValueError, match="tool_result_spill_dir"):
        HarnessAgentConfig(workspace_dir=tmp_path, tool_result_spill_dir=bad_dir)


def test_spill_dir_validation_accepts_empty_and_workspace_relative_subpaths(tmp_path: Path) -> None:
    from octop_harness.config import HarnessAgentConfig

    assert ToolResultBudgetMiddleware(max_tokens=100, spill_dir="")._spill_dir == ""
    assert ToolResultBudgetMiddleware(max_tokens=100, spill_dir=".tool-outputs/sub/")._spill_dir == ".tool-outputs/sub"
    cfg = HarnessAgentConfig(workspace_dir=tmp_path, tool_result_spill_dir="cache/tool-outputs/")
    assert cfg.tool_result_spill_dir == "cache/tool-outputs"
