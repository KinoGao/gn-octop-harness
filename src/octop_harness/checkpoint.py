"""LangGraph checkpoints in memory, durably persisted by the embedding host."""

from __future__ import annotations

import asyncio
import sqlite3
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
)
from langgraph.checkpoint.sqlite import SqliteSaver


class SnapshotSqliteSaver(SqliteSaver):
    """Keep a SQLite working image in RAM and await host persistence per write.

    ``load`` returns an existing SQLite image, or ``None`` for a new store.
    ``save`` must durably replace that image before returning; hosts can send
    only changed pages. Optional ``validate`` checks storage availability before
    reads as well as writes. Callbacks are synchronous and async saver methods
    dispatch them to an executor. Construct this saver off the event loop too.

    A failed transaction invalidates the saver, including when a successful
    save loses its acknowledgement. Recreate it from ``load`` before retrying;
    the host's image is authoritative. The caller owns the saver lifetime and
    may use ``close`` to release its memory early.
    """

    def __init__(
        self,
        *,
        load: Callable[[], bytes | None],
        save: Callable[[bytes], None],
        validate: Callable[[], None] | None = None,
    ) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        super().__init__(conn)
        self._save = save
        self._validate = validate
        self.failed = False
        try:
            image = load()
            if image is not None:
                conn.deserialize(image)
            conn.execute("PRAGMA temp_store = MEMORY")
            self.setup()
            conn.commit()
            self._image = conn.serialize()
            if self._image != image:
                save(self._image)
        except BaseException:
            conn.close()
            raise
        self._finalizer = weakref.finalize(self, conn.close)

    def ensure_available(self) -> None:
        """Reject an interrupted saver or unavailable host storage."""
        if self.failed:
            raise RuntimeError("Reload checkpoints after an interrupted write")
        if self._validate is not None:
            self._validate()

    def close(self) -> None:
        """Release the in-memory SQLite connection (idempotent)."""
        with self.lock:
            self.failed = True
            self._finalizer()

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with self.lock:
            self.ensure_available()
            cur = self.conn.cursor()
            try:
                yield cur
                if transaction:
                    self.conn.commit()
                    image = self.conn.serialize()
                    if image != self._image:
                        self._save(image)
                        self._image = image
            except BaseException:
                if transaction:
                    self.failed = True
                    self.conn.rollback()
                raise
            finally:
                cur.close()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.get_running_loop().run_in_executor(None, self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.delete_thread, thread_id)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        read = partial(self.list, config, filter=filter, before=before, limit=limit)
        rows = await asyncio.get_running_loop().run_in_executor(None, lambda: list(read()))
        for row in rows:
            yield row

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.get_delta_channel_history, config=config, channels=channels)
        )
