import uuid
import pytest
from celebvision.db import Database
from celebvision.config import Settings
from celebvision.watchlist.pg_index import PgVectorWatchlistIndex

pytestmark = pytest.mark.requires_stack


def _unit(seed: int) -> list[float]:
    v = [0.0] * 512
    v[seed] = 1.0
    return v


async def test_enroll_and_search():
    db = Database(Settings.from_env().postgres_dsn)
    await db.connect()
    await db.apply_schema()
    try:
        wid = "wl-" + uuid.uuid4().hex[:8]
        idx = PgVectorWatchlistIndex(db, wid)
        await idx.add("messi", "Lionel Messi", ["messi"], [_unit(0)])
        await idx.add("ronaldo", "Cristiano Ronaldo", ["cr7"], [_unit(1)])

        hit = await idx.search(_unit(0), threshold=0.5)
        assert hit is not None and hit[0] == "messi" and hit[2] > 0.9

        assert await idx.search(_unit(2), threshold=0.5) is None
        names = await idx.names()
        assert ("messi", "Lionel Messi", ["messi"]) in names
    finally:
        await db.close()
