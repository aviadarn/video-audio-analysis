from celebvision import metrics


def test_counters_and_histogram_record():
    g = metrics.REGISTRY.get_sample_value
    before_processed = g("celebvision_stage_processed_total", {"stage": "faces"}) or 0.0
    before_failed = g("celebvision_stage_failed_total", {"stage": "faces"}) or 0.0
    before_retried = g("celebvision_stage_retried_total", {"stage": "faces"}) or 0.0
    before_dlq = g("celebvision_stage_dlq_total", {"stage": "faces"}) or 0.0
    before_dur = g("celebvision_stage_duration_seconds_count", {"stage": "faces"}) or 0.0

    metrics.record_processed("faces")
    metrics.record_processed("faces")
    metrics.record_failed("faces")
    metrics.record_retried("faces")
    metrics.record_dlq("faces")
    metrics.observe_duration("faces", 0.5)

    assert g("celebvision_stage_processed_total", {"stage": "faces"}) - before_processed == 2.0
    assert g("celebvision_stage_failed_total", {"stage": "faces"}) - before_failed == 1.0
    assert g("celebvision_stage_retried_total", {"stage": "faces"}) - before_retried == 1.0
    assert g("celebvision_stage_dlq_total", {"stage": "faces"}) - before_dlq == 1.0
    assert g("celebvision_stage_duration_seconds_count", {"stage": "faces"}) - before_dur == 1.0


def test_render_returns_prometheus_text():
    out = metrics.render()
    assert isinstance(out, bytes)
    assert b"celebvision_stage_processed_total" in out
    assert metrics.CONTENT_TYPE.startswith("text/plain")
