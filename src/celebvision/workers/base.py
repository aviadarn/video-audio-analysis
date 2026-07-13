import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from celebvision.bus import Message, MessageBus
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.config import Settings
from celebvision.errors import StageError
from celebvision import metrics
from celebvision.logging import get_logger, log_stage


@dataclass
class WorkerContext:
    db: Database
    storage: BlobStore
    inference: object
    llm: object
    settings: Settings


Handler = Callable[[Message, WorkerContext], Awaitable[None]]

_LOG = get_logger()


async def run_worker(stage: str, bus: MessageBus, ctx: WorkerContext,
                     handler: Handler) -> None:
    async for msg in bus.stream([f"{stage}.requested"], group=stage):
        start = time.monotonic()
        try:
            await handler(msg, ctx)
        except StageError as e:
            metrics.record_failed(stage)
            metrics.observe_duration(stage, time.monotonic() - start)
            attempts = msg.attempts + 1
            if attempts < ctx.settings.max_attempts:
                metrics.record_retried(stage)
                log_stage(_LOG, "retry", msg.job_id, stage, "retry", attempts)
                await bus.produce(f"{stage}.requested", msg.job_id, Message(
                    job_id=msg.job_id, stage=stage, scene_id=msg.scene_id,
                    payload=msg.payload, attempts=attempts))
            else:
                metrics.record_dlq(stage)
                log_stage(_LOG, "dlq", msg.job_id, stage, "dlq", attempts)
                await bus.produce(f"{stage}.dlq", msg.job_id, Message(
                    job_id=msg.job_id, stage=stage, scene_id=msg.scene_id,
                    payload=msg.payload, attempts=attempts))
                await ctx.db.set_job_error(
                    msg.job_id, f"{stage}: {e.reason} (dlq after {attempts} attempts)")
            continue
        metrics.observe_duration(stage, time.monotonic() - start)
        metrics.record_processed(stage)
        log_stage(_LOG, "processed", msg.job_id, stage, "ok", msg.attempts)
        await bus.produce("stage.events", msg.job_id, Message(
            job_id=msg.job_id, stage=stage, scene_id=msg.scene_id))
