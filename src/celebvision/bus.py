from collections.abc import AsyncIterator
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer


class Message(BaseModel):
    job_id: str
    stage: str
    scene_id: int | None = None
    payload: dict = {}
    attempts: int = 0


def encode(msg: Message) -> bytes:
    return msg.model_dump_json().encode("utf-8")


def decode(raw: bytes) -> Message:
    return Message.model_validate_json(raw)


class MessageBus:
    def __init__(self, bootstrap: str) -> None:
        self._bootstrap = bootstrap
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap)
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def produce(self, topic: str, key: str, message: Message) -> None:
        assert self._producer is not None, "call start() first"
        await self._producer.send_and_wait(topic, key=key.encode("utf-8"),
                                            value=encode(message))

    async def stream(self, topics: list[str], group: str) -> AsyncIterator[Message]:
        consumer = AIOKafkaConsumer(
            *topics, bootstrap_servers=self._bootstrap, group_id=group,
            enable_auto_commit=False, auto_offset_reset="earliest",
        )
        await consumer.start()
        try:
            async for record in consumer:
                yield decode(record.value)
                await consumer.commit()
        finally:
            await consumer.stop()
