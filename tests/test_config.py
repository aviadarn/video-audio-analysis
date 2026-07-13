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

def test_settings_triton_defaults():
    from celebvision.config import Settings
    s = Settings.from_env({})
    assert s.triton_url == "localhost:8000"
    assert s.asr_backend == "stub"
    assert s.triton_model == "arcface"

def test_settings_triton_env_overrides():
    from celebvision.config import Settings
    s = Settings.from_env({"TRITON_URL": "triton:8000", "ASR_BACKEND": "local"})
    assert s.triton_url == "triton:8000"
    assert s.asr_backend == "local"

def test_settings_reliability_defaults():
    from celebvision.config import Settings
    s = Settings.from_env({})
    assert s.max_attempts == 3
    assert s.metrics_port == 9100
    s2 = Settings.from_env({"MAX_ATTEMPTS": "5", "METRICS_PORT": "9200"})
    assert s2.max_attempts == 5
    assert s2.metrics_port == 9200
