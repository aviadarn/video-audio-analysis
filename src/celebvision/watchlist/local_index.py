import json
import numpy as np


class LocalWatchlistIndex:
    def __init__(self, watchlist_id: str) -> None:
        self.watchlist_id = watchlist_id
        self._vectors: list[list[float]] = []   # one row per embedding
        self._row_id: list[int] = []            # identity index per row
        self._identities: list[dict] = []       # {canonical_id, name, aliases}

    def add(self, canonical_id: str, name: str, aliases: list[str],
            embeddings: list[list[float]]) -> None:
        ident_idx = len(self._identities)
        self._identities.append(
            {"canonical_id": canonical_id, "name": name, "aliases": list(aliases)}
        )
        for emb in embeddings:
            self._vectors.append(list(emb))
            self._row_id.append(ident_idx)

    def search(self, embedding, threshold: float):
        if not self._vectors:
            return None
        mat = np.asarray(self._vectors, dtype=np.float32)   # (N, D), L2-normed rows
        q = np.asarray(embedding, dtype=np.float32)
        scores = mat @ q                                    # cosine (rows unit-norm)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < threshold:
            return None
        ident = self._identities[self._row_id[best]]
        return (ident["canonical_id"], ident["name"], score)

    def names(self) -> list[tuple[str, str, list[str]]]:
        return [(i["canonical_id"], i["name"], list(i["aliases"]))
                for i in self._identities]

    def save(self, path: str) -> None:
        meta = json.dumps({
            "watchlist_id": self.watchlist_id,
            "row_id": self._row_id,
            "identities": self._identities,
        })
        vectors = np.asarray(self._vectors, dtype=np.float32) if self._vectors \
            else np.zeros((0, 0), dtype=np.float32)
        np.savez(path, vectors=vectors, meta=np.array(meta))

    @classmethod
    def load(cls, path: str) -> "LocalWatchlistIndex":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        idx = cls(meta["watchlist_id"])
        idx._identities = meta["identities"]
        idx._row_id = meta["row_id"]
        idx._vectors = data["vectors"].tolist()
        return idx
