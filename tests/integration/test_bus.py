import pytest
from celebvision.bus import MessageBus, Message
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_produce_and_consume_round_trip():
    bus = MessageBus(Settings.from_env().kafka_bootstrap)
    await bus.start()
    try:
        await bus.produce("test.bus", "k1",
                          Message(job_id="jb", stage="ingest", payload={"n": 1}))
        got = None
        async for msg in bus.stream(["test.bus"], group="test-bus-grp"):
            got = msg
            break
        assert got is not None
        assert got.job_id == "jb"
        assert got.payload == {"n": 1}
    finally:
        await bus.stop()
