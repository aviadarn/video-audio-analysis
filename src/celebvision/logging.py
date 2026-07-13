import json
import logging
import sys


def stage_record(event: str, job_id: str, stage: str, status: str,
                 attempts: int) -> dict:
    return {"event": event, "job_id": job_id, "stage": stage,
            "status": status, "attempts": attempts}


def get_logger(name: str = "celebvision") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_stage(logger: logging.Logger, event: str, job_id: str, stage: str,
              status: str, attempts: int) -> None:
    logger.info(json.dumps(stage_record(event, job_id, stage, status, attempts)))
