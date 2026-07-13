from celebvision import metrics


def test_counters_and_histogram_record():
    metrics.record_processed("faces")
    metrics.record_processed("faces")
    metrics.record_failed("faces")
    metrics.record_retried("faces")
    metrics.record_dlq("faces")
    metrics.observe_duration("faces", 0.5)
    g = metrics.REGISTRY.get_sample_value
    assert g("celebvision_stage_processed_total", {"stage": "faces"}) == 2.0
    assert g("celebvision_stage_failed_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_retried_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_dlq_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_duration_seconds_count", {"stage": "faces"}) == 1.0

def test_render_returns_prometheus_text():
    out = metrics.render()
    assert isinstance(out, bytes)
    assert b"celebvision_stage_processed_total" in out
    assert metrics.CONTENT_TYPE.startswith("text/plain")
