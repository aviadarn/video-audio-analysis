import json
from celebvision.db import Database


class PgVectorWatchlistIndex:
    def __init__(self, db: Database, watchlist_id: str) -> None:
        self.db = db
        self.watchlist_id = watchlist_id

    async def add(self, canonical_id, name, aliases, embeddings) -> None:
        async with self.db.pool.acquire() as conn:
            for emb in embeddings:
                await conn.execute(
                    "INSERT INTO watchlist (canonical_id, watchlist_id, name, aliases, embedding) "
                    "VALUES ($1,$2,$3,$4,$5)",
                    canonical_id, self.watchlist_id, name, json.dumps(list(aliases)),
                    emb,
                )

    async def search(self, embedding, threshold):
        async with self.db.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT canonical_id, name, 1 - (embedding <=> $1) AS score "
                "FROM watchlist WHERE watchlist_id=$2 "
                "ORDER BY embedding <=> $1 LIMIT 1",
                embedding, self.watchlist_id,
            )
        if row is None or float(row["score"]) < threshold:
            return None
        return (row["canonical_id"], row["name"], float(row["score"]))

    async def names(self):
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT ON (canonical_id) canonical_id, name, aliases "
                "FROM watchlist WHERE watchlist_id=$1 ORDER BY canonical_id",
                self.watchlist_id,
            )
        out = []
        for r in rows:
            aliases = r["aliases"]
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            out.append((r["canonical_id"], r["name"], aliases))
        return out
