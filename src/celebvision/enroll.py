from celebvision.interfaces import InferenceClient
from celebvision.watchlist.local_index import LocalWatchlistIndex


def enroll_identity(index: LocalWatchlistIndex, inference: InferenceClient,
                    canonical_id: str, name: str, aliases: list[str],
                    image_paths: list[str]) -> None:
    embeddings: list[list[float]] = []
    for path in image_paths:
        dets = inference.analyze_faces(path)
        if dets:  # take the first (largest) detected face
            embeddings.append(dets[0].embedding)
    if not embeddings:
        raise ValueError(f"no faces detected in reference images for {name}")
    index.add(canonical_id, name, aliases, embeddings)
