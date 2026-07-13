from celebvision.config import Settings
from celebvision.errors import StageError


def test_settings_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s.kafka_bootstrap == "localhost:19092"
    assert s.postgres_dsn.startswith("postgresql://")
    assert s.inference_backend == "stub"
    assert s.llm_backend == "stub"
    assert s.watchlist_backend == "pg"

def test_settings_reads_env_overrides():
    s = Settings.from_env({"KAFKA_BOOTSTRAP": "broker:9092",
                           "INFERENCE_BACKEND": "local"})
    assert s.kafka_bootstrap == "broker:9092"
    assert s.inference_backend == "local"

def test_stage_error_str():
    e = StageError("ingest", "download failed")
    assert str(e) == "ingest: download failed"
    assert e.stage == "ingest"
    assert e.reason == "download failed"
