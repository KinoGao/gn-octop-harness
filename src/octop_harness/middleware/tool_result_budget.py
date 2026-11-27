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

import logging
from collections.abc import Awaitable, Callable
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
        self._spill_dir = spill_dir.strip("/")
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
        return self._apply(request, await handler(request))

    def _apply(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if self._max_tokens <= 0 or not isinstance(result, ToolMessage):
            return result
        content = result.content
        if not isinstance(content, str):
            return result
        original_tokens = estimate_tokens(content)
        if original_tokens <= self._max_tokens:
            return result

        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "tool")
        call_id = str(tool_call.get("id") or "")
        spill_path = self._spill(content, tool_name=tool_name, call_id=call_id)
        budget_chars = int(len(content) * self._max_tokens * _BUDGET_SAFETY / original_tokens)
        truncated = _truncate_head_tail(
            content,
            max_tokens=self._max_tokens,
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
        return result.model_copy(update={"content": truncated})

    def _spill(self, text: str, *, tool_name: str, call_id: str) -> str | None:
        """Persist the full result and return its workspace path, or ``None``."""
        if not self._spill_dir or self._workspace is None:
            return None
        stem = _safe_stem(f"{tool_name}-{call_id}" if call_id else tool_name)
        path = f"{self._spill_dir}/{stem}.txt"
        try:
            self._workspace.write_text(path, text, force=True)
        except Exception:  # spilling must never fail a tool call
            logger.warning("Could not spill tool output to %s", path, exc_info=True)
            return None
        return path


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


__all__ = ["DEFAULT_MAX_TOKENS", "HEAD_SHARE", "ToolResultBudgetMiddleware"]
