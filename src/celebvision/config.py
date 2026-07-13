from collections.abc import Mapping
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = "localhost:19092"
    postgres_dsn: str = "postgresql://celeb:celeb@localhost:5432/celeb"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    inference_backend: str = "stub"
    llm_backend: str = "stub"
    watchlist_backend: str = "pg"
    whisper_model: str = "small"
    face_model: str = "buffalo_l"
    watchlist_path: str = "./data/watchlist.npz"
    triton_url: str = "localhost:8000"
    asr_backend: str = "stub"
    triton_model: str = "arcface"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        d = cls()
        return cls(
            kafka_bootstrap=e.get("KAFKA_BOOTSTRAP", d.kafka_bootstrap),
            postgres_dsn=e.get("POSTGRES_DSN", d.postgres_dsn),
            minio_endpoint=e.get("MINIO_ENDPOINT", d.minio_endpoint),
            minio_access_key=e.get("MINIO_ACCESS_KEY", d.minio_access_key),
            minio_secret_key=e.get("MINIO_SECRET_KEY", d.minio_secret_key),
            inference_backend=e.get("INFERENCE_BACKEND", d.inference_backend),
            llm_backend=e.get("LLM_BACKEND", d.llm_backend),
            watchlist_backend=e.get("WATCHLIST_BACKEND", d.watchlist_backend),
            whisper_model=e.get("WHISPER_MODEL", d.whisper_model),
            face_model=e.get("FACE_MODEL", d.face_model),
            watchlist_path=e.get("WATCHLIST_PATH", d.watchlist_path),
            triton_url=e.get("TRITON_URL", d.triton_url),
            asr_backend=e.get("ASR_BACKEND", d.asr_backend),
            triton_model=e.get("TRITON_MODEL", d.triton_model),
        )
