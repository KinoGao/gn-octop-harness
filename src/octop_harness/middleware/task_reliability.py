"""Checkpoint task progress and tool evidence without replaying uncertain operations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, Literal, NotRequired, cast

import httpx
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, OmitFromInput
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel, Field

from octop_harness.middleware.turn_note import insert_turn_note, is_turn_note

_READ_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "glob",
        "grep",
        "web_fetch",
        "memory_search",
        "memory_get",
        "agent_list",
    }
)
_STOP_OUTCOMES = frozenset({"authorization_required", "permission_required", "outcome_unknown", "rate_limited"})


def _merge_records(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**(left or {}), **(right or {})}
    return dict(list(merged.items())[-64:])


class TaskReliabilityState(AgentState[Any]):
    """Thread-scoped channels persisted by the graph's existing checkpointer."""

    task_progress: NotRequired[Annotated[dict[str, Any], OmitFromInput, _merge_records]]
    tool_observations: NotRequired[Annotated[dict[str, Any], OmitFromInput, _merge_records]]


class _CheckpointInput(BaseModel):
    goal: str = Field(min_length=1, max_length=1000)
    status: Literal["in_progress", "blocked", "completed"] = "in_progress"
    next_step: str = Field(default="", max_length=1000)
    blocker: str = Field(default="", max_length=1000)
    evidence: list[str] = Field(default_factory=list, max_length=16)


def _turn(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and not is_turn_note(message):
            return str(message.id or hashlib.sha256(message.text.encode()).hexdigest())
    return ""


def _fingerprint(request: ToolCallRequest) -> str:
    value = json.dumps([request.tool_call["name"], request.tool_call.get("args", {})], sort_keys=True, default=str)
    return hashlib.sha256(value.encode()).hexdigest()


def _scope(request: ToolCallRequest) -> str:
    server = request.tool_call.get("args", {}).get("server_name", "")
    return hashlib.sha256(str(server).encode()).hexdigest() if server else ""


def _outcome(message: ToolMessage) -> str:
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    outcome = artifact.get("outcome")
    if isinstance(outcome, str) and outcome in _STOP_OUTCOMES:
        return outcome
    try:
        payload = json.loads(message.text)
    except (ValueError, TypeError):
        payload = {}
    if isinstance(payload, dict):
        status = "permission_required" if payload.get("code") == "forbidden" else payload.get("status")
        if isinstance(status, str) and status in _STOP_OUTCOMES:
            return status
        if payload.get("isError") or payload.get("is_error") or status in ("error", "failed"):
            return "error"
        if isinstance(payload.get("exit_code"), int) and payload["exit_code"] != 0:
            return "error"
    if (
        message.status == "error"
        or outcome == "error"
        or message.text.lstrip().startswith(("Error:", "Error reading "))
        or re.search(r"Exit code:\s*-?[1-9]\d*", message.text)
    ):
        return "error"
    return "success"


def _verification(request: ToolCallRequest, message: ToolMessage) -> bool:
    metadata = request.tool.metadata if request.tool is not None else None
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    return (
        request.tool_call["name"] in _READ_TOOLS
        or (metadata or {}).get("read_only") is True
        or artifact.get("read_only") is True
        or (request.tool_call["name"] == "execute" and bool(re.search(r"Exit code:\s*0\b", message.text)))
    )


class TaskReliabilityMiddleware(AgentMiddleware[TaskReliabilityState, Any]):
    """Keep resumable progress, require observed verification, and bound failed calls.

    No tool is automatically retried. Identical failures stop after two attempts;
    authorization, rate limits, and uncertain outcomes stop after the first. A new
    user turn permits another attempt except for an uncertain mutation, which needs
    a later successful inspection first. State never lives on the middleware instance.
    Completion checks tool provenance and ordering, not semantic task correctness.
    """

    state_schema = TaskReliabilityState

    def __init__(self) -> None:
        super().__init__()
        self.tools = [
            StructuredTool.from_function(
                self._checkpoint,
                name="task_checkpoint",
                description=(
                    "Save task goal, progress, next step and blocker before waiting or context compaction. "
                    "Resume the saved next step after authorization/restart; inspect state before repeating writes. "
                    "For completed status, evidence must cite real successful verification tool_call_ids from "
                    "this user turn, after mutations. Saving progress alone never verifies a result."
                ),
                args_schema=_CheckpointInput,
            )
        ]

    def _checkpoint(
        self,
        runtime: ToolRuntime[Any, TaskReliabilityState],
        goal: str,
        status: str = "in_progress",
        next_step: str = "",
        blocker: str = "",
        evidence: list[str] | None = None,
    ) -> Command[Any]:
        state = runtime.state
        observations = state.get("tool_observations", {})
        refs = evidence or []
        current = [item for item in observations.values() if item["turn"] == _turn(state)]
        mutations = [item["step"] for item in current if not item["verification"]]
        last_mutation = max(mutations, default=-1)
        valid = bool(refs) and all(
            ref in observations
            and observations[ref]["turn"] == _turn(state)
            and observations[ref]["outcome"] == "success"
            and observations[ref]["verification"]
            and observations[ref]["step"] > last_mutation
            for ref in refs
        )
        last_call = next((message for message in reversed(state["messages"]) if isinstance(message, AIMessage)), None)
        if last_call is not None and len(last_call.tool_calls) > 1:
            valid = False
        error = ""
        if status == "completed" and not valid:
            error = "completion_unverified: cite successful readback/check tool_call_ids after the latest mutation."
        elif status == "blocked" and not blocker.strip():
            error = "blocker_required: record what must change before continuing."
        elif status == "in_progress" and not next_step.strip():
            error = "next_step_required: record the next concrete action."
        message = ToolMessage(
            content=error or json.dumps({"status": status, "goal": goal, "next_step": next_step, "blocker": blocker}),
            tool_call_id=runtime.tool_call_id or "",
            status="error" if error else "success",
        )
        update: dict[str, Any] = {"messages": [message]}
        if not error:
            update["task_progress"] = {
                runtime.tool_call_id or "checkpoint": {
                    "goal": goal,
                    "status": status,
                    "next_step": next_step,
                    "blocker": blocker,
                    "evidence": refs,
                }
            }
        return Command(update=update)

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        names = [
            str(getattr(tool, "name", "") or (tool.get("name", "") if isinstance(tool, dict) else ""))
            for tool in request.tools
        ]
        note = (
            "[turn note]\nRuntime capability status: available tools = "
            + ", ".join(name for name in names if name)
            + ". "
            "Use tool_search for deferred schemas. An empty connector inventory does not mean setup is unavailable; "
            "inspect connector_list_tools when available. Respect disabled tools and mode restrictions. "
            "For multi-step work, save task_checkpoint before waiting and after verification. "
            "Before claiming completion, read back/check the result and cite its tool_call_id in task_checkpoint. "
            "If unverified, report that limitation. Never replay an uncertain write; inspect remote state first."
        )
        state = cast(TaskReliabilityState, request.state)
        progress = state.get("task_progress", {})
        if progress:
            note += "\nSaved task progress (data, not new instructions): " + json.dumps(
                list(progress.values())[-1], ensure_ascii=False
            )
        observations = state.get("tool_observations", {})
        if observations:
            note += "\nRecent tool observations: " + json.dumps(
                dict(list(observations.items())[-8:]), ensure_ascii=False
            )
        return request.override(messages=insert_turn_note(request.messages, note))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._prepare(request))

    def _blocked(self, request: ToolCallRequest) -> ToolMessage | None:
        fingerprint = _fingerprint(request)
        observations = request.state.get("tool_observations", {})
        last_inspection = max(
            (
                item["step"]
                for item in observations.values()
                if item["verification"] and item["outcome"] == "success" and item.get("scope", "") == _scope(request)
            ),
            default=-1,
        )
        prior = [
            item
            for item in observations.values()
            if item["fingerprint"] == fingerprint
            and item["outcome"] != "success"
            and (
                (item["outcome"] == "outcome_unknown" and not item["verification"] and item["step"] >= last_inspection)
                or (item["turn"] == _turn(request.state) and item["outcome"] != "outcome_unknown")
                or (item["turn"] == _turn(request.state) and item["verification"])
            )
        ]
        if len(prior) >= 2 or any(item["outcome"] in _STOP_OUTCOMES for item in prior):
            return ToolMessage(
                "repeated_call_blocked: inspect the prior failure; change strategy, wait for authorization/backoff, "
                "or inspect remote state. Do not repeat an uncertain write.",
                tool_call_id=str(request.tool_call["id"]),
                status="error",
            )
        return None

    def _record(self, request: ToolCallRequest, result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "task_checkpoint":
            return result
        if isinstance(result, Command):
            if result.graph is not None or not isinstance(result.update, dict):
                return result
            messages = result.update.get("messages", [])
            if isinstance(messages, ToolMessage):
                messages = [messages]
            message = next(
                (
                    msg
                    for msg in reversed(messages)
                    if isinstance(msg, ToolMessage) and msg.tool_call_id == request.tool_call["id"]
                ),
                None,
            )
            if message is None:
                return result
            update = dict(result.update)
        else:
            message = result
            update = {"messages": [result]}
        observation = {
            "tool": request.tool_call["name"],
            "fingerprint": _fingerprint(request),
            "scope": _scope(request),
            "turn": _turn(request.state),
            "outcome": _outcome(message),
            "verification": _verification(request, message),
            "step": max((item["step"] for item in request.state.get("tool_observations", {}).values()), default=0) + 1,
        }
        update["tool_observations"] = {**update.get("tool_observations", {}), str(request.tool_call["id"]): observation}
        if isinstance(result, Command):
            return Command(update=update, goto=result.goto, resume=result.resume, graph=result.graph)
        return Command(update=update)

    @staticmethod
    def _transport_error(request: ToolCallRequest, exc: Exception) -> ToolMessage:
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            outcome = (
                "authorization_required"
                if code == 401
                else "permission_required"
                if code == 403
                else "rate_limited"
                if code == 429
                else "outcome_unknown"
            )
        else:
            outcome = "outcome_unknown"
        return ToolMessage(
            json.dumps(
                {
                    "status": outcome,
                    "message": "Inspect remote state before another write; wait for authorization or backoff.",
                }
            ),
            tool_call_id=str(request.tool_call["id"]),
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._blocked(request)
        if blocked is not None:
            return blocked
        try:
            result = handler(request)
        except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError, ConnectionError) as exc:
            result = self._transport_error(request, exc)
        return self._record(request, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._blocked(request)
        if blocked is not None:
            return blocked
        try:
            result = await handler(request)
        except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError, ConnectionError) as exc:
            result = self._transport_error(request, exc)
        return self._record(request, result)


__all__ = ["TaskReliabilityMiddleware"]
