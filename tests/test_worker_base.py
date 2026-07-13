import pytest
from celebvision.bus import Message
from celebvision.errors import StageError
from celebvision.workers.base import run_worker, WorkerContext


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


def _ctx(db):
    return WorkerContext(db=db, storage=None, inference=None, llm=None, settings=None)


async def test_success_emits_completion_event():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1)])
    calls = []
    async def handler(msg, ctx):
        calls.append(msg.job_id)
    await run_worker("faces", bus, _ctx(FakeDB()), handler)
    assert calls == ["j"]
    assert bus.produced == [("stage.events",
                             Message(job_id="j", stage="faces", scene_id=1))]

async def test_stage_error_marks_job_failed_and_does_not_emit():
    bus = FakeBus([Message(job_id="j", stage="ingest")])
    db = FakeDB()
    async def handler(msg, ctx):
        raise StageError("ingest", "boom")
    await run_worker("ingest", bus, _ctx(db), handler)
    assert db.errors == [("j", "ingest: boom")]
    assert bus.produced == []
