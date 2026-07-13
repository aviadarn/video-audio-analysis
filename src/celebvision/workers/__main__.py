import asyncio
import sys
from prometheus_client import start_http_server
from celebvision.config import Settings
from celebvision.bus import MessageBus
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.workers.base import WorkerContext, run_worker
from celebvision.factories import build_inference_client, build_llm_client
from celebvision.metrics import REGISTRY as METRICS_REGISTRY
from celebvision.workers.ingest import handle_ingest
from celebvision.workers.scene import handle_scene
from celebvision.workers.transcribe import handle_transcribe
from celebvision.workers.face import handle_face
from celebvision.workers.mention import handle_mention
from celebvision.workers.aggregate import handle_aggregate

HANDLERS = {
    "ingest": handle_ingest, "scene": handle_scene, "transcribe": handle_transcribe,
    "face": handle_face, "mention": handle_mention, "aggregate": handle_aggregate,
}
# topic stage name -> handler key (topics use the request stage names)
TOPIC_STAGE = {
    "ingest": "ingest", "scenes": "scene", "transcribe": "transcribe",
    "faces": "face", "mentions": "mention", "aggregate": "aggregate",
}


async def main(stage_topic: str) -> None:
    settings = Settings.from_env()
    db = Database(settings.postgres_dsn)
    await db.connect()
    bus = MessageBus(settings.kafka_bootstrap)
    await bus.start()
    store = BlobStore(settings.minio_endpoint, settings.minio_access_key,
                      settings.minio_secret_key)
    ctx = WorkerContext(db=db, storage=store,
                        inference=build_inference_client(settings),
                        llm=build_llm_client(settings), settings=settings)
    handler = HANDLERS[TOPIC_STAGE[stage_topic]]
    start_http_server(settings.metrics_port, registry=METRICS_REGISTRY)
    try:
        await run_worker(stage_topic, bus, ctx, handler)
    finally:
        await bus.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
