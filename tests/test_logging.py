import json
import logging
from io import StringIO
from celebvision.logging import stage_record, get_logger, log_stage


def test_stage_record_shape():
    r = stage_record("processed", "j1", "faces", "ok", 0)
    assert r == {"event": "processed", "job_id": "j1", "stage": "faces",
                 "status": "ok", "attempts": 0}

def test_log_stage_emits_json_line():
    logger = logging.getLogger("celebvision.test")
    logger.handlers.clear()
    buf = StringIO()
    h = logging.StreamHandler(buf)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    log_stage(logger, "dlq", "j2", "ingest", "dlq", 3)
    parsed = json.loads(buf.getvalue().strip())
    assert parsed["event"] == "dlq" and parsed["attempts"] == 3
