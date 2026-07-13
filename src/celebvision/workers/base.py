from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from celebvision.bus import Message, MessageBus
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.config import Settings
from celebvision.errors import StageError


@dataclass
class WorkerContext:
    db: Database
    storage: BlobStore
    inference: object
    llm: object
    settings: Settings


Handler = Callable[[Message, WorkerContext], Awaitable[None]]


async def run_worker(stage: str, bus: MessageBus, ctx: WorkerContext,
                     handler: Handler) -> None:
    async for msg in bus.stream([f"{stage}.requested"], group=stage):
        try:
            await handler(msg, ctx)
        except StageError as e:
            await ctx.db.set_job_error(msg.job_id, str(e))
            continue
        await bus.produce("stage.events", msg.job_id,
                          Message(job_id=msg.job_id, stage=stage, scene_id=msg.scene_id))
