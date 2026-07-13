import asyncio
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.storage import BlobStore

TOPICS = ["ingest.requested", "scenes.requested", "transcribe.requested",
          "faces.requested", "mentions.requested", "aggregate.requested",
          "stage.events",
          "ingest.dlq", "scenes.dlq", "transcribe.dlq",
          "faces.dlq", "mentions.dlq", "aggregate.dlq"]
BUCKETS = ["media", "keyframes", "reports"]


async def _retry(coro_factory, attempts=30, delay=2.0):
    last = None
    for _ in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 - startup readiness retry
            last = e
            await asyncio.sleep(delay)
    raise last


async def _create_topics(bootstrap: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        try:
            await admin.create_topics(
                [NewTopic(t, num_partitions=3, replication_factor=1) for t in TOPICS])
        except Exception as e:  # topic-exists is fine
            print(f"topics: {e}")
    finally:
        await admin.close()


async def _apply_schema(dsn: str) -> None:
    db = Database(dsn)
    await db.connect()
    await db.apply_schema()
    await db.close()


async def _make_buckets(s: Settings) -> None:
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    for b in BUCKETS:
        await store.ensure_bucket(b)


async def main() -> None:
    s = Settings.from_env()
    await _retry(lambda: _create_topics(s.kafka_bootstrap))
    await _retry(lambda: _apply_schema(s.postgres_dsn))
    await _retry(lambda: _make_buckets(s))
    print("init complete")


if __name__ == "__main__":
    asyncio.run(main())
