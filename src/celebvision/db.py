import json
from pathlib import Path
import asyncpg
from pgvector.asyncpg import register_vector

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        assert self._pool is not None, "call connect() first"
        return self._pool

    async def connect(self) -> None:
        # Ensure the vector extension exists BEFORE the pool is created, so every
        # pooled connection's init hook can register the vector codec successfully.
        boot = await asyncpg.connect(self._dsn)
        try:
            await boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await boot.close()
        self._pool = await asyncpg.create_pool(self._dsn, init=self._init_conn,
                                               min_size=1, max_size=10)

    async def _init_conn(self, conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def apply_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)

    async def create_job(self, job_id, source_kind, source_locator, watchlist_id,
                         keywords) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO jobs (id, source_kind, source_locator, watchlist_id, keywords) "
                "VALUES ($1,$2,$3,$4,$5)",
                job_id, source_kind, source_locator, watchlist_id, json.dumps(keywords),
            )

    async def get_job(self, job_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id=$1", job_id)
        return dict(row) if row else None

    async def set_job_status(self, job_id, status) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status=$2 WHERE id=$1", job_id, status)

    async def set_job_error(self, job_id, error) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status='failed', error=$2 WHERE id=$1",
                               job_id, error)

    async def set_job_completed(self, job_id) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status='done', completed_at=now() WHERE id=$1", job_id)

    async def set_expected_scene_count(self, job_id, n) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET expected_scene_count=$2 WHERE id=$1",
                               job_id, n)

    async def upsert_job_assets(self, job_id, *, video_key=None, audio_key=None,
                               transcript_key=None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO job_assets (job_id, video_key, audio_key, transcript_key) "
                "VALUES ($1,$2,$3,$4) ON CONFLICT (job_id) DO UPDATE SET "
                "video_key=COALESCE(EXCLUDED.video_key, job_assets.video_key), "
                "audio_key=COALESCE(EXCLUDED.audio_key, job_assets.audio_key), "
                "transcript_key=COALESCE(EXCLUDED.transcript_key, job_assets.transcript_key)",
                job_id, video_key, audio_key, transcript_key,
            )

    async def get_job_assets(self, job_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM job_assets WHERE job_id=$1", job_id)
        return dict(row) if row else None

    async def insert_scene(self, job_id, scene_id, start_s, end_s, keyframes) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO scenes (job_id, scene_id, start_s, end_s, keyframes) "
                "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (job_id, scene_id) DO UPDATE SET "
                "start_s=EXCLUDED.start_s, end_s=EXCLUDED.end_s, keyframes=EXCLUDED.keyframes",
                job_id, scene_id, start_s, end_s, json.dumps(keyframes),
            )

    async def get_scene(self, job_id, scene_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM scenes WHERE job_id=$1 AND scene_id=$2", job_id, scene_id)
        return dict(row) if row else None

    async def get_scenes(self, job_id) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM scenes WHERE job_id=$1 ORDER BY scene_id", job_id)
        return [dict(r) for r in rows]

    async def set_scene_faces(self, job_id, scene_id, faces) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE scenes SET faces=$3, faces_status='done' "
                "WHERE job_id=$1 AND scene_id=$2", job_id, scene_id, json.dumps(faces))

    async def set_scene_mentions(self, job_id, scene_id, transcript, mentions,
                                 keyword_hits) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE scenes SET transcript=$3, mentions=$4, keyword_hits=$5, "
                "mentions_status='done' WHERE job_id=$1 AND scene_id=$2",
                job_id, scene_id, transcript, json.dumps(mentions),
                json.dumps(keyword_hits))

    async def count_scenes_stage_done(self, job_id, stage) -> int:
        col = {"faces": "faces_status", "mentions": "mentions_status"}[stage]
        async with self.pool.acquire() as conn:
            n = await conn.fetchval(
                f"SELECT count(*) FROM scenes WHERE job_id=$1 AND {col}='done'", job_id)
        return int(n)

    async def mark_stage_done(self, job_id, stage) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO job_stages (job_id, stage) VALUES ($1,$2) "
                "ON CONFLICT DO NOTHING", job_id, stage)

    async def is_stage_done(self, job_id, stage) -> bool:
        async with self.pool.acquire() as conn:
            v = await conn.fetchval(
                "SELECT 1 FROM job_stages WHERE job_id=$1 AND stage=$2", job_id, stage)
        return v is not None
