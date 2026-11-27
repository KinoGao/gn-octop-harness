"""Grounding guards: keep the model from citing stale or absent evidence.

Two failure modes account for most office-work hallucination, and neither is a
prompt problem — both are context problems the harness can see:

**Stale reads.** ``read_file`` puts a snapshot in history. After ``write_file``
/ ``edit_file`` / ``delete`` that snapshot is a lie: the model keeps quoting the
pre-edit numbers, clauses and table cells as if they were current. The write
result carries an explicit invalidation notice so the correction lands exactly
where the model is looking, without rewriting history.

**Empty results.** A tool that returns nothing is the single strongest prompt to
fabricate: the model fills the vacuum with plausible content. A harness-generated
notice turns "no output" back into an explicit *no evidence* — which is what the
call actually established.

Notices are appended to the ``ToolMessage`` the model already receives, so they
cost one short paragraph of context and no extra round trip. Read tracking is
partitioned by ``thread_id`` so concurrent sessions on the same agent cannot
cross-contaminate each other.
"""

from __future__ import annotations

import json
import logging
import posixpath
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from threading import RLock
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.middleware.filesystem_guard import (
    _FS_TOOL_OPS,
    _path_from_tool_args,
    _tool_base_name,
)

logger = logging.getLogger(__name__)

STALE_READ_NOTICE = (
    "\n\n[harness] `{path}` was just modified. Anything you read from it earlier in this "
    "conversation is stale: re-read the file before quoting, citing, or reporting its "
    "contents. Do not answer from the earlier snapshot."
)

NO_EVIDENCE_NOTICE = (
    "\n\n[harness] This call produced no usable evidence. Do not assert that the content "
    "exists, and do not assert that it is absent beyond the scope you actually checked — "
    "say what you could not verify."
)

_WRITE_TOOLS = frozenset({"write_file", "edit_file", "append_file", "delete", "delete_file", "move_file"})
_READ_TOOLS = frozenset(name for name, op in _FS_TOOL_OPS.items() if op == "read")
_MAX_TRACKED_PATHS = 200
_MAX_TRACKED_THREADS = 2048
_DEFAULT_THREAD_KEY = "_default"


def _normalize_path(path: str) -> str:
    """Collapse the path spellings one file can arrive under."""
    return posixpath.normpath(path.strip().replace("\\", "/")).lstrip("/")


def _content_text(content: Any) -> str:
    """Flatten a ``ToolMessage.content`` (str or content-block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _has_media_block(content: Any) -> bool:
    """True when the result carries a non-text block (image, audio, …)."""
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") != "text" for block in content)


def _is_json_object(text: str) -> bool:
    """True when *text* is a valid JSON object payload (must not be corrupted with prose)."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except (ValueError, TypeError):
        return False


def _append_notice(content: Any, text: str, appended: str) -> Any:
    """Append *appended* without dropping non-text content blocks."""
    if isinstance(content, list):
        return [*content, {"type": "text", "text": appended}]
    return f"{text}{appended}"


def _thread_key(request: ToolCallRequest) -> str:
    runtime = getattr(request, "runtime", None)
    cfg = getattr(runtime, "config", None)
    if not isinstance(cfg, Mapping):
        try:
            cfg = get_config()
        except RuntimeError:
            cfg = None
    if isinstance(cfg, Mapping):
        configurable = cfg.get("configurable")
        if isinstance(configurable, Mapping):
            tid = configurable.get("thread_id")
            if tid is not None and str(tid).strip():
                return str(tid)
    return _DEFAULT_THREAD_KEY


class GroundingMiddleware(AgentMiddleware[Any, Any]):
    """Annotate tool results that invalidate earlier context or carry no evidence."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._read_paths_by_thread: OrderedDict[str, list[str]] = OrderedDict()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._annotate(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._annotate(request, await handler(request))

    def _annotate(
        self,
        request: ToolCallRequest,
        result: Any,
    ) -> ToolMessage | Command[Any]:
        if isinstance(result, Command):
            return self._annotate_command(request, result)
        if not isinstance(result, ToolMessage):
            return result  # type: ignore[no-any-return]
        return self._annotate_message(request, result)

    def _annotate_command(self, request: ToolCallRequest, result: Command[Any]) -> Command[Any]:
        if result.graph is not None or not isinstance(result.update, dict):
            return result
        raw_messages = result.update.get("messages")
        call_id = str(request.tool_call.get("id") or "")
        if isinstance(raw_messages, ToolMessage):
            updated = self._annotate_message(request, raw_messages)
            if updated is raw_messages:
                return result
            return Command(
                update={**result.update, "messages": updated},
                goto=result.goto,
                resume=result.resume,
                graph=result.graph,
            )
        if not isinstance(raw_messages, list):
            return result
        new_messages = list(raw_messages)
        changed = False
        for idx in range(len(new_messages) - 1, -1, -1):
            msg = new_messages[idx]
            if isinstance(msg, ToolMessage) and (not call_id or msg.tool_call_id == call_id):
                updated = self._annotate_message(request, msg)
                if updated is not msg:
                    new_messages[idx] = updated
                    changed = True
                break
        if not changed:
            return result
        return Command(
            update={**result.update, "messages": new_messages},
            goto=result.goto,
            resume=result.resume,
            graph=result.graph,
        )

    def _annotate_message(self, request: ToolCallRequest, result: ToolMessage) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "")
        raw_params = request.tool_call.get("args") or {}
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}

        notices = self._track(_thread_key(request), tool_name, params, result)
        text = _content_text(result.content)
        empty = not text.strip() and not _has_media_block(result.content)
        if empty or (result.status == "error" and not _is_json_object(text)):
            notices.append(NO_EVIDENCE_NOTICE)

        if not notices:
            return result

        appended = "".join(notice for notice in notices if notice.strip() not in text)
        if not appended:
            return result
        logger.debug("GroundingMiddleware annotated %s result", tool_name)
        return result.model_copy(update={"content": _append_notice(result.content, text, appended)})

    def _track(self, thread_key: str, tool_name: str, params: dict[str, Any], result: ToolMessage) -> list[str]:
        """Maintain the per-thread read ledger; return a stale notice when one applies."""
        base = _tool_base_name(tool_name)
        path = _path_from_tool_args(tool_name, params)
        if path is None or result.status == "error":
            return []
        key = _normalize_path(path)

        with self._lock:
            ledger = self._read_paths_by_thread.get(thread_key)
            if ledger is None:
                ledger = []
                self._read_paths_by_thread[thread_key] = ledger
            self._read_paths_by_thread.move_to_end(thread_key)
            while len(self._read_paths_by_thread) > _MAX_TRACKED_THREADS:
                self._read_paths_by_thread.popitem(last=False)

            if base in _READ_TOOLS:
                if base == "read_file" and key not in ledger:
                    ledger.append(key)
                    if len(ledger) > _MAX_TRACKED_PATHS:
                        del ledger[0]
                return []

            if base in _WRITE_TOOLS and key in ledger:
                ledger.remove(key)
                return [STALE_READ_NOTICE.format(path=path)]
            return []


__all__ = ["NO_EVIDENCE_NOTICE", "STALE_READ_NOTICE", "GroundingMiddleware"]
