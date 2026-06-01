from enum import StrEnum


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    expired = "expired"


class ErrorCode(StrEnum):
    invalid_file_type = "INVALID_FILE_TYPE"
    file_too_large = "FILE_TOO_LARGE"
    invalid_frame_metadata = "INVALID_FRAME_METADATA"
    model_not_ready = "MODEL_NOT_READY"
    model_inference_failed = "MODEL_INFERENCE_FAILED"
    job_not_found = "JOB_NOT_FOUND"
    job_not_completed = "JOB_NOT_COMPLETED"
    rate_limit_exceeded = "RATE_LIMIT_EXCEEDED"
