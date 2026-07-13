from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()
CONTENT_TYPE = CONTENT_TYPE_LATEST

_processed = Counter("celebvision_stage_processed_total",
                     "Stage messages processed", ["stage"], registry=REGISTRY)
_failed = Counter("celebvision_stage_failed_total",
                  "Stage handler failures", ["stage"], registry=REGISTRY)
_retried = Counter("celebvision_stage_retried_total",
                   "Stage messages retried", ["stage"], registry=REGISTRY)
_dlq = Counter("celebvision_stage_dlq_total",
               "Stage messages sent to DLQ", ["stage"], registry=REGISTRY)
_duration = Histogram("celebvision_stage_duration_seconds",
                      "Stage handler duration", ["stage"], registry=REGISTRY)


def record_processed(stage: str) -> None:
    _processed.labels(stage=stage).inc()


def record_failed(stage: str) -> None:
    _failed.labels(stage=stage).inc()


def record_retried(stage: str) -> None:
    _retried.labels(stage=stage).inc()


def record_dlq(stage: str) -> None:
    _dlq.labels(stage=stage).inc()


def observe_duration(stage: str, seconds: float) -> None:
    _duration.labels(stage=stage).observe(seconds)


def render() -> bytes:
    return generate_latest(REGISTRY)
