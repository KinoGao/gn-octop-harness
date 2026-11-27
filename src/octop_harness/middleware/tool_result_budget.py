"""``ToolResultBudgetMiddleware`` — keep one tool result from owning the context.

Why
---
A single ``web_fetch``, MCP call, ``browser_use`` snapshot or ``execute`` run
can return hundreds of KiB. From that turn onward **every** request resends
those bytes: prompt cache is prefix-keyed, so the cost is paid on each turn
until summarization evicts the message, and summarization itself fires sooner.

``execute`` caps its own output (``max_output_bytes``), and
``MediaOffloadMiddleware`` handles inline base64, but plain-text results from
network and MCP tools went into the conversation whole.

What this does
--------------
``wrap_tool_call`` / ``awrap_tool_call``, after the tool returns: if the textual
result exceeds ``max_tokens``, keep the **head** (command, headers, first rows)
and the **tail** (errors, totals, last lines) of the allowed character budget
and replace the middle with an explicit omission marker naming the dropped
characters and tokens. The head/tail split mirrors what a coding agent reads
first and last; a plain prefix cut hides the failure message that usually sits
at the end.

With ``spill_dir`` set, the untouched text is written to
``{spill_dir}/{tool}-{call}.txt`` through the workspace backend and the marker
names that path, so the model can ``read_file`` the part it actually needs
instead of guessing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path, PureWindowsPath
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.context_usage import estimate_tokens

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 25_000
# Keep 60% of the budget from the front, 40% from the back.
HEAD_SHARE = 0.6
# A char budget derived from a token estimate lands slightly over budget after
# re-estimating, so shave a few percent before slicing.
_BUDGET_SAFETY = 0.97

_OMISSION_TEMPLATE = (
    "\n\n... [{omitted_chars} characters, ~{omitted_tokens} tokens omitted{spill}]: "
    "output truncated by the harness; re-run with a narrower command, or "
    "{reread} ...\n\n"
)


def validate_spill_dir(value: str) -> str:
    """Validate and normalize a workspace-relative spill directory, or ``""``."""
    if not isinstance(value, str):
        raise ValueError(f"tool_result_spill_dir must be a string, got {type(value).__name__}")
    raw = value.strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    if normalized.startswith(("/", "~")) or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"tool_result_spill_dir must be a workspace-relative subpath, got {value!r}")
    if Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ValueError(f"tool_result_spill_dir must be a workspace-relative subpath, got {value!r}")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"tool_result_spill_dir must not contain '.' or '..' traversal, got {value!r}")
    return "/".join(parts)


class ToolResultBudgetMiddleware(AgentMiddleware[Any, Any]):
    """Truncate oversized textual tool results to a token budget.

    Args:
        max_tokens: Per-result budget. ``<= 0`` disables truncation.
        spill_dir: Workspace-relative directory for the full text, or ``""``
            to drop the omitted middle without spilling it.
        workspace: Backend used to write spilled output. ``None`` disables
            spilling even when *spill_dir* is set.
    """

    def __init__(
        self,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        spill_dir: str = "",
        workspace: Any | None = None,
    ) -> None:
        super().__init__()
        self._max_tokens = max_tokens
        self._spill_dir = validate_spill_dir(spill_dir)
        self._workspace = workspace

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._apply(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await self._aapply(request, await handler(request))

    def _apply(
        self,
        request: ToolCallRequest,
        result: Any,
    ) -> ToolMessage | Command[Any]:
        if self._max_tokens <= 0:
            return result  # type: ignore[no-any-return]
        if isinstance(result, Command):
            return self._apply_command(request, result)
        if not isinstance(result, ToolMessage):
            return result  # type: ignore[no-any-return]
        return self._apply_message(request, result)

    async def _aapply(
        self,
        request: ToolCallRequest,
        result: Any,
    ) -> ToolMessage | Command[Any]:
        if self._max_tokens <= 0:
            return result  # type: ignore[no-any-return]
        if isinstance(result, Command):
            return await self._aapply_command(request, result)
        if not isinstance(result, ToolMessage):
            return result  # type: ignore[no-any-return]
        return await self._aapply_message(request, result)

    def _apply_command(self, request: ToolCallRequest, result: Command[Any]) -> Command[Any]:
        if result.graph is not None or not isinstance(result.update, dict):
            return result
        raw_messages = result.update.get("messages")
        call_id = str(request.tool_call.get("id") or "")
        if isinstance(raw_messages, ToolMessage):
            updated = self._apply_message(request, raw_messages)
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
                updated = self._apply_message(request, msg)
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

    async def _aapply_command(self, request: ToolCallRequest, result: Command[Any]) -> Command[Any]:
        if result.graph is not None or not isinstance(result.update, dict):
            return result
        raw_messages = result.update.get("messages")
        call_id = str(request.tool_call.get("id") or "")
        if isinstance(raw_messages, ToolMessage):
            updated = await self._aapply_message(request, raw_messages)
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
                updated = await self._aapply_message(request, msg)
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

    def _apply_message(self, request: ToolCallRequest, result: ToolMessage) -> ToolMessage:
        content: Any = result.content
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "tool")
        call_id = str(tool_call.get("id") or "")
        if isinstance(content, str):
            original_tokens = estimate_tokens(content)
            if original_tokens <= self._max_tokens:
                return result
            spill_path = self._spill(content, tool_name=tool_name, call_id=call_id)
            truncated = self._truncate_and_log(
                content,
                tool_name=tool_name,
                original_tokens=original_tokens,
                max_tokens=self._max_tokens,
                spill_path=spill_path,
            )
            return result.model_copy(update={"content": truncated})
        if isinstance(content, list):
            plan = _plan_list_truncation(content, self._max_tokens)
            if not plan:
                return result
            new_content = list(content)
            for part_idx, (idx, text, original_tokens, block_budget) in enumerate(plan):
                suffix = f"-part{part_idx}" if len(plan) > 1 else ""
                spill_path = self._spill(text, tool_name=tool_name, call_id=call_id, suffix=suffix)
                truncated = self._truncate_and_log(
                    text,
                    tool_name=tool_name,
                    original_tokens=original_tokens,
                    max_tokens=block_budget,
                    spill_path=spill_path,
                )
                block = new_content[idx]
                new_content[idx] = {**block, "text": truncated} if isinstance(block, dict) else truncated
            return result.model_copy(update={"content": new_content})
        return result

    async def _aapply_message(self, request: ToolCallRequest, result: ToolMessage) -> ToolMessage:
        content: Any = result.content
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "tool")
        call_id = str(tool_call.get("id") or "")
        if isinstance(content, str):
            original_tokens = estimate_tokens(content)
            if original_tokens <= self._max_tokens:
                return result
            spill_path = await self._aspill(content, tool_name=tool_name, call_id=call_id)
            truncated = self._truncate_and_log(
                content,
                tool_name=tool_name,
                original_tokens=original_tokens,
                max_tokens=self._max_tokens,
                spill_path=spill_path,
            )
            return result.model_copy(update={"content": truncated})
        if isinstance(content, list):
            plan = _plan_list_truncation(content, self._max_tokens)
            if not plan:
                return result
            new_content = list(content)
            for part_idx, (idx, text, original_tokens, block_budget) in enumerate(plan):
                suffix = f"-part{part_idx}" if len(plan) > 1 else ""
                spill_path = await self._aspill(text, tool_name=tool_name, call_id=call_id, suffix=suffix)
                truncated = self._truncate_and_log(
                    text,
                    tool_name=tool_name,
                    original_tokens=original_tokens,
                    max_tokens=block_budget,
                    spill_path=spill_path,
                )
                block = new_content[idx]
                new_content[idx] = {**block, "text": truncated} if isinstance(block, dict) else truncated
            return result.model_copy(update={"content": new_content})
        return result

    @staticmethod
    def _truncate_and_log(
        text: str,
        *,
        tool_name: str,
        original_tokens: int,
        max_tokens: int,
        spill_path: str | None,
    ) -> str:
        budget_chars = int(len(text) * max_tokens * _BUDGET_SAFETY / original_tokens)
        truncated = _truncate_head_tail(
            text,
            max_tokens=max_tokens,
            original_tokens=original_tokens,
            spill_path=spill_path,
            budget_chars=budget_chars,
        )
        logger.info(
            "Truncated %s result: %d -> ~%d tokens (spill=%s)",
            tool_name,
            original_tokens,
            estimate_tokens(truncated),
            spill_path or "-",
        )
        return truncated

    def _spill_path(self, *, tool_name: str, call_id: str, suffix: str = "") -> str:
        raw = f"{tool_name}-{call_id}" if call_id else tool_name
        stem = _safe_stem(f"{raw}{suffix}" if suffix else raw)
        return f"{self._spill_dir}/{stem}.txt"

    def _spill(self, text: str, *, tool_name: str, call_id: str, suffix: str = "") -> str | None:
        """Persist the full result and return its workspace path, or ``None``."""
        if not self._spill_dir or self._workspace is None:
            return None
        path = self._spill_path(tool_name=tool_name, call_id=call_id, suffix=suffix)
        try:
            self._workspace.write_text(path, text, force=True)
        except Exception:  # spilling must never fail a tool call
            logger.warning("Could not spill tool output to %s", path, exc_info=True)
            return None
        return path

    async def _aspill(self, text: str, *, tool_name: str, call_id: str, suffix: str = "") -> str | None:
        """Persist the full result asynchronously and return its workspace path, or ``None``."""
        if not self._spill_dir or self._workspace is None:
            return None
        path = self._spill_path(tool_name=tool_name, call_id=call_id, suffix=suffix)
        try:
            awrite = getattr(self._workspace, "awrite_text", None)
            if callable(awrite):
                await awrite(path, text, force=True)
            else:
                await asyncio.to_thread(self._workspace.write_text, path, text, force=True)
        except Exception:  # spilling must never fail a tool call
            logger.warning("Could not spill tool output to %s", path, exc_info=True)
            return None
        return path


def _plan_list_truncation(content: list[Any], max_tokens: int) -> list[tuple[int, str, int, int]]:
    """Return ``(index, text, original_tokens, block_budget)`` for oversized text blocks."""
    entries: list[tuple[int, str, int]] = []
    for idx, block in enumerate(content):
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"]
            entries.append((idx, text, estimate_tokens(text)))
        elif isinstance(block, str):
            entries.append((idx, block, estimate_tokens(block)))
    if not entries:
        return []
    total_tokens = sum(tok for _, _, tok in entries)
    if total_tokens <= max_tokens:
        return []
    fair_share = max(1, max_tokens // len(entries))
    kept_tokens = sum(tok for _, _, tok in entries if tok <= fair_share)
    oversized = [(idx, text, tok) for idx, text, tok in entries if tok > fair_share]
    remaining_budget = max(1, max_tokens - kept_tokens)
    oversized_total = sum(tok for _, _, tok in oversized)
    if len(oversized) == 1:
        idx, text, tok = oversized[0]
        return [(idx, text, tok, remaining_budget)]
    return [(idx, text, tok, max(1, int(remaining_budget * tok / oversized_total))) for idx, text, tok in oversized]


def _truncate_head_tail(
    text: str,
    *,
    max_tokens: int,
    original_tokens: int,
    spill_path: str | None,
    budget_chars: int | None = None,
) -> str:
    """Keep the head and tail of *text* under *max_tokens*, marker in the middle.

    The first slice is a proportional guess; the omission marker itself costs
    tokens, so shrink and retry until the estimate fits. The loop is bounded
    because each pass removes 15% of the character budget.
    """
    if budget_chars is None:
        budget_chars = int(len(text) * max_tokens * _BUDGET_SAFETY / original_tokens)
    budget_chars = max(budget_chars, 200)
    candidate = _assemble(text, budget_chars, original_tokens, spill_path)
    for _ in range(4):
        if estimate_tokens(candidate) <= max_tokens:
            return candidate
        budget_chars = int(budget_chars * 0.85)
        candidate = _assemble(text, budget_chars, original_tokens, spill_path)
    return candidate


def _assemble(text: str, budget_chars: int, original_tokens: int, spill_path: str | None) -> str:
    """Slice *text* into head + marker + tail for the given character budget."""
    head_chars = int(budget_chars * HEAD_SHARE)
    tail_chars = budget_chars - head_chars
    head = text[:head_chars]
    tail = text[len(text) - tail_chars :] if tail_chars else ""
    omitted = text[len(head) : len(text) - len(tail)]
    spill = f"; full output: {spill_path} (use read_file)" if spill_path else ""
    reread = f"read_file {spill_path}" if spill_path else "ask again with a smaller range"
    marker = _OMISSION_TEMPLATE.format(
        omitted_chars=len(omitted),
        omitted_tokens=max(1, original_tokens - estimate_tokens(head) - estimate_tokens(tail)),
        spill=spill,
        reread=reread,
    )
    return f"{head}{marker}{tail}"


def _safe_stem(raw: str) -> str:
    """Filesystem-safe stem for a spilled result."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in raw)
    return cleaned.strip("-.")[:80] or "tool-output"


__all__ = ["DEFAULT_MAX_TOKENS", "HEAD_SHARE", "ToolResultBudgetMiddleware", "validate_spill_dir"]
