# src/celebvision/api/app.py
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.bus import MessageBus, Message
from celebvision.storage import BlobStore
from celebvision.media import classify_source
from celebvision import metrics


class JobRequest(BaseModel):
    source: str
    watchlist_id: str
    keywords: list[str] = []


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.postgres_dsn)
        await db.connect()
        await db.apply_schema()
        bus = MessageBus(settings.kafka_bootstrap)
        await bus.start()
        store = BlobStore(settings.minio_endpoint, settings.minio_access_key,
                          settings.minio_secret_key)
        app.state.db = db
        app.state.bus = bus
        app.state.store = store
        app.state.settings = settings
        try:
            yield
        finally:
            await bus.stop()
            await db.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/jobs", status_code=201)
    async def create_job(req: JobRequest):
        job_id = str(uuid.uuid4())
        src = classify_source(req.source)
        await app.state.db.create_job(job_id, src.kind, src.locator,
                                      req.watchlist_id, req.keywords)
        await app.state.bus.produce("ingest.requested", job_id,
                                    Message(job_id=job_id, stage="ingest"))
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await app.state.db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        total = job["expected_scene_count"]
        return {
            "status": job["status"],
            "error": job["error"],
            "expected_scene_count": total,
            "scenes_done": {
                "faces": await app.state.db.count_scenes_stage_done(job_id, "faces"),
                "mentions": await app.state.db.count_scenes_stage_done(job_id, "mentions"),
                "total": total,
            },
        }

    @app.get("/reports/{job_id}")
    async def get_report(job_id: str):
        job = await app.state.db.get_job(job_id)
        if job is None or job["status"] != "done":
            raise HTTPException(status_code=404, detail="report not ready")
        data = await app.state.store.get_bytes("reports", f"{job_id}.json")
        return JSONResponse(content=json.loads(data))

    @app.get("/metrics")
    async def get_metrics():
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    return app


app = create_app()
