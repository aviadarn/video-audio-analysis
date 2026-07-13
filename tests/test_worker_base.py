import pytest
from celebvision.bus import Message
from celebvision.errors import StageError
from celebvision.workers.base import run_worker, WorkerContext
from celebvision.config import Settings


class FakeBus:
    def __init__(self, messages):
        self._messages = messages
        self.produced = []
    async def produce(self, topic, key, message):
        self.produced.append((topic, message))
    async def stream(self, topics, group):
        for m in self._messages:
            yield m


class FakeDB:
    def __init__(self):
        self.errors = []
    async def set_job_error(self, job_id, error):
        self.errors.append((job_id, error))


def _ctx(db, max_attempts=3):
    return WorkerContext(db=db, storage=None, inference=None, llm=None,
                         settings=Settings.from_env({"MAX_ATTEMPTS": str(max_attempts)}))


async def test_success_emits_completion_event():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1)])
    async def handler(msg, ctx):
        return None
    await run_worker("faces", bus, _ctx(FakeDB()), handler)
    assert bus.produced == [("stage.events",
                             Message(job_id="j", stage="faces", scene_id=1))]


async def test_stage_error_retries_to_requested_with_incremented_attempts():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1, attempts=0)])
    async def handler(msg, ctx):
        raise StageError("faces", "boom")
    await run_worker("faces", bus, _ctx(FakeDB(), max_attempts=3), handler)
    assert len(bus.produced) == 1
    topic, msg = bus.produced[0]
    assert topic == "faces.requested"
    assert msg.attempts == 1


async def test_stage_error_routes_to_dlq_when_attempts_exhausted():
    bus = FakeBus([Message(job_id="j", stage="ingest", attempts=2)])
    db = FakeDB()
    async def handler(msg, ctx):
        raise StageError("ingest", "boom")
    await run_worker("ingest", bus, _ctx(db, max_attempts=3), handler)
    topic, msg = bus.produced[0]
    assert topic == "ingest.dlq"
    assert msg.attempts == 3
    assert db.errors and "dlq after 3 attempts" in db.errors[0][1]
