from celebvision.interfaces import InferenceClient, LLMClient, WatchlistIndex
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.inference.triton_client import TritonFaceClient


def _build_asr(settings):
    if settings.asr_backend == "stub":
        from celebvision.inference.stub_client import StubInferenceClient
        return StubInferenceClient()
    if settings.asr_backend == "local":
        from celebvision.inference.local_client import LocalInferenceClient
        return LocalInferenceClient(whisper_model=settings.whisper_model,
                                    face_model=settings.face_model)
    raise ValueError(f"unknown ASR_BACKEND: {settings.asr_backend}")


def build_inference_client(settings: Settings) -> InferenceClient:
    if settings.inference_backend == "stub":
        from celebvision.inference.stub_client import StubInferenceClient
        return StubInferenceClient()
    if settings.inference_backend == "local":
        from celebvision.inference.local_client import LocalInferenceClient
        return LocalInferenceClient(whisper_model=settings.whisper_model,
                                    face_model=settings.face_model)
    if settings.inference_backend == "triton":
        from celebvision.inference.composite import CompositeInferenceClient
        return CompositeInferenceClient(transcriber=_build_asr(settings),
                                        face_analyzer=TritonFaceClient(settings))
    raise ValueError(f"unknown INFERENCE_BACKEND: {settings.inference_backend}")


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_backend == "stub":
        from celebvision.llm.stub_client import StubLLMClient
        return StubLLMClient()
    if settings.llm_backend == "anthropic":
        from celebvision.llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    raise ValueError(f"unknown LLM_BACKEND: {settings.llm_backend}")


async def build_watchlist(settings: Settings, db: Database,
                          watchlist_id: str) -> WatchlistIndex:
    if settings.watchlist_backend == "pg":
        from celebvision.watchlist.pg_index import PgVectorWatchlistIndex
        return PgVectorWatchlistIndex(db, watchlist_id)
    if settings.watchlist_backend == "local":
        from celebvision.watchlist.local_index import LocalWatchlistIndex
        return LocalWatchlistIndex.load(settings.watchlist_path)
    raise ValueError(f"unknown WATCHLIST_BACKEND: {settings.watchlist_backend}")
