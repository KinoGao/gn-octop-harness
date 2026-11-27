from __future__ import annotations

import sqlite3

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from octop_harness import SnapshotSqliteSaver


class ApprovalState(TypedDict):
    question: str
    answer: str


def approval(state: ApprovalState) -> dict[str, str]:
    return {"answer": interrupt(state["question"])}


class Images:
    def __init__(self) -> None:
        self.image: bytes | None = None
        self.fail = False
        self.lost_ack = False
        self.online = True

    def load(self) -> bytes | None:
        self.validate()
        return self.image

    def save(self, image: bytes) -> None:
        self.validate()
        if self.fail:
            raise OSError("interrupted write")
        self.image = image
        if self.lost_ack:
            raise OSError("lost acknowledgement")

    def validate(self) -> None:
        if not self.online:
            raise OSError("offline")

    def saver(self) -> SnapshotSqliteSaver:
        return SnapshotSqliteSaver(load=self.load, save=self.save, validate=self.validate)


@pytest.mark.asyncio
@pytest.mark.parametrize("durability", ["exit", "async", "sync"])
async def test_interrupt_resumes_after_reloading_only_the_external_image(tmp_path, monkeypatch, durability):
    monkeypatch.chdir(tmp_path)
    images = Images()
    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    config = {"configurable": {"thread_id": "private"}}
    first = images.saver()
    paused = await builder.compile(checkpointer=first).ainvoke({"question": "Approve?"}, config, durability=durability)
    assert paused["__interrupt__"]
    assert first.conn.execute("PRAGMA database_list").fetchone()[2] == ""
    first.close()

    restored = images.saver()
    graph = builder.compile(checkpointer=restored)
    assert (await graph.aget_state(config)).next == ("approval",)
    result = await graph.ainvoke(Command(resume="approved"), config, durability=durability)
    assert result["answer"] == "approved"
    checkpoints = [item async for item in restored.alist(config)]
    assert len(checkpoints) >= 2
    assert not (await graph.aget_state(config)).next
    await restored.adelete_thread("private")
    assert await restored.aget_tuple(config) is None
    restored.close()
    with sqlite3.connect(":memory:") as conn:
        conn.deserialize(images.image)
        assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_ack", [False, True])
async def test_failed_write_requires_new_saver_and_reloads_authoritative_image(lost_ack):
    from langgraph.checkpoint.base import empty_checkpoint

    images = Images()
    saver = images.saver()
    config = {"configurable": {"thread_id": "private", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"answer": "sensitive"}
    images.fail, images.lost_ack = not lost_ack, lost_ack
    with pytest.raises(OSError):
        await saver.aput(config, checkpoint, {"source": "input", "step": 0}, {})
    assert saver.failed
    with pytest.raises(RuntimeError, match="interrupted"):
        await saver.aget_tuple(config)
    images.fail = images.lost_ack = False
    replacement = images.saver()
    recovered = await replacement.aget_tuple(config)
    assert (recovered is not None) == lost_ack
    if recovered:
        assert recovered.checkpoint["channel_values"] == {"answer": "sensitive"}
    saver.close()
    replacement.close()


@pytest.mark.asyncio
async def test_offline_reads_do_not_reuse_the_in_memory_copy():
    images = Images()
    saver = images.saver()
    images.online = False
    with pytest.raises(OSError, match="offline"):
        await saver.aget_tuple({"configurable": {"thread_id": "private"}})
    saver.close()
