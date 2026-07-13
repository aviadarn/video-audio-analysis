import math
from celebvision.watchlist.local_index import LocalWatchlistIndex


def _unit(vec):
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec]


def test_search_returns_best_match_above_threshold(tmp_path):
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi"], [_unit([1.0, 0.0, 0.0])])
    idx.add("ronaldo", "Cristiano Ronaldo", ["cr7"], [_unit([0.0, 1.0, 0.0])])

    hit = idx.search(_unit([0.9, 0.1, 0.0]), threshold=0.5)
    assert hit is not None
    canonical_id, name, score = hit
    assert canonical_id == "messi"
    assert score > 0.5

def test_search_returns_none_below_threshold():
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi"], [_unit([1.0, 0.0, 0.0])])
    assert idx.search(_unit([0.0, 0.0, 1.0]), threshold=0.5) is None

def test_save_load_round_trip(tmp_path):
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi", "leo"], [_unit([1.0, 0.0, 0.0])])
    path = str(tmp_path / "wl.npz")
    idx.save(path)
    loaded = LocalWatchlistIndex.load(path)
    assert loaded.watchlist_id == "w1"
    assert loaded.names() == [("messi", "Lionel Messi", ["messi", "leo"])]
    assert loaded.search(_unit([1.0, 0.0, 0.0]), threshold=0.5)[0] == "messi"
