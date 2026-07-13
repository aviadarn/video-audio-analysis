# src/celebvision/coordinator.py
from dataclasses import dataclass
from celebvision.bus import Message, MessageBus
from celebvision.db import Database


@dataclass
class JobState:
    status: str
    expected_scene_count: int | None
    ingest_done: bool
    scenes_done: bool
    transcribe_done: bool
    faces_done: int
    mentions_done: int


def decide_next(event: Message, state: JobState) -> list[Message]:
    if state.status == "failed":
        return []
    jid = event.job_id
    stage = event.stage
    if stage == "ingest":
        return [Message(job_id=jid, stage="scenes"),
                Message(job_id=jid, stage="transcribe")]
    if stage in ("scenes", "transcribe"):
        if state.scenes_done and state.transcribe_done \
                and state.expected_scene_count is not None:
            out: list[Message] = []
            for sid in range(state.expected_scene_count):
                out.append(Message(job_id=jid, stage="faces", scene_id=sid))
                out.append(Message(job_id=jid, stage="mentions", scene_id=sid))
            return out
        return []
    if stage in ("faces", "mentions"):
        n = state.expected_scene_count
        if n is not None and state.faces_done == n and state.mentions_done == n:
            return [Message(job_id=jid, stage="aggregate")]
        return []
    return []


async def load_job_state(db: Database, job_id: str) -> JobState:
    job = await db.get_job(job_id)
    return JobState(
        status=job["status"] if job else "unknown",
        expected_scene_count=job["expected_scene_count"] if job else None,
        ingest_done=await db.is_stage_done(job_id, "ingest"),
        scenes_done=await db.is_stage_done(job_id, "scenes"),
        transcribe_done=await db.is_stage_done(job_id, "transcribe"),
        faces_done=await db.count_scenes_stage_done(job_id, "faces"),
        mentions_done=await db.count_scenes_stage_done(job_id, "mentions"),
    )


async def run_coordinator(bus: MessageBus, db: Database) -> None:
    async for event in bus.stream(["stage.events"], group="coordinator"):
        if event.stage == "aggregate":
            await db.set_job_completed(event.job_id)
        state = await load_job_state(db, event.job_id)
        for out in decide_next(event, state):
            await bus.produce(f"{out.stage}.requested", out.job_id, out)
