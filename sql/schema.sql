CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id uuid PRIMARY KEY,
    source_kind text NOT NULL,
    source_locator text NOT NULL,
    watchlist_id text NOT NULL,
    keywords jsonb NOT NULL DEFAULT '[]',
    status text NOT NULL DEFAULT 'queued',
    error text,
    expected_scene_count int,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS job_assets (
    job_id uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    video_key text,
    audio_key text,
    transcript_key text
);

CREATE TABLE IF NOT EXISTS scenes (
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    scene_id int NOT NULL,
    start_s double precision NOT NULL,
    end_s double precision NOT NULL,
    keyframes jsonb NOT NULL DEFAULT '[]',
    transcript text NOT NULL DEFAULT '',
    faces jsonb NOT NULL DEFAULT '[]',
    mentions jsonb NOT NULL DEFAULT '[]',
    keyword_hits jsonb NOT NULL DEFAULT '[]',
    faces_status text NOT NULL DEFAULT 'pending',
    mentions_status text NOT NULL DEFAULT 'pending',
    PRIMARY KEY (job_id, scene_id)
);

CREATE TABLE IF NOT EXISTS job_stages (
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage text NOT NULL,
    PRIMARY KEY (job_id, stage)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id bigserial PRIMARY KEY,
    canonical_id text NOT NULL,
    watchlist_id text NOT NULL,
    name text NOT NULL,
    aliases jsonb NOT NULL DEFAULT '[]',
    embedding vector(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS watchlist_wid_idx ON watchlist (watchlist_id);
-- Note: `vector` has no btree opclass, so it cannot be part of a PRIMARY KEY.
-- Multiple embedding rows per (canonical_id, watchlist_id) are allowed.
