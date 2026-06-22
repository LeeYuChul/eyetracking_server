import json
import logging
import secrets
from collections import deque
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request
from PIL import Image

from app.core.config import Settings, get_settings
from app.models.frame import FrameAnalysisBundle, FrameChatRequest, FrameInput, FrameModelInfo
from app.models.job import ErrorCode
from app.services.model import ModelRegistry, SaliencyModel
from app.services.artifacts import image_to_artifact
from app.services.image_processing import make_overlay, normalize_upload, saliency_to_heatmap
from app.services.pipeline import analyze_image
from app.services.scanpath import build_scanpath_metrics, make_scanpath_overlay
from app.services.vlm import evaluate_frame_chat, stream_frame_chat_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
RATE_LIMIT_EXEMPT_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}


class MinuteRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> bool:
        if self.limit <= 0:
            return True
        now = monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.limit:
                return False
            self._timestamps.append(now)
            return True

settings = get_settings()
model_registry = ModelRegistry(
    saliency_model_path=settings.saliency_model_path,
    runtime=settings.model_runtime,
    device_preference=settings.device,
    cuda_device_index=settings.cuda_device_index,
    allowed_model_names=settings.allowed_model_names,
    default_model_name=settings.default_model_name,
    idle_unload_seconds=settings.model_idle_unload_seconds,
)
rate_limiter = MinuteRateLimiter(settings.rate_limit_per_minute)

app = FastAPI(title=settings.service_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_requests(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in RATE_LIMIT_EXEMPT_PATHS and not rate_limiter.allow():
        logger.info("request endpoint=%s frame_count=0 elapsed_ms=0.00 error_code=%s", path, ErrorCode.rate_limit_exceeded.value)
        return JSONResponse(
            status_code=429,
            content={"error_code": ErrorCode.rate_limit_exceeded, "message": "Too many requests. Please try again later."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


def get_model_registry() -> ModelRegistry:
    return model_registry


def make_job_id() -> str:
    return f"ana_{datetime.now(UTC).strftime('%Y%m%d')}_{secrets.token_hex(3)}"


def make_bundle_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": code, "message": message})


def parse_positive_int(value: float, field_name: str) -> int:
    if value <= 0 or int(value) != value:
        raise HTTPException(
            status_code=400,
            detail={"error_code": ErrorCode.invalid_frame_metadata, "message": f"{field_name} must be a positive integer"},
        )
    return int(value)


def log_request(endpoint: str, started_at: float, *, frame_count: int = 0, error_code: ErrorCode | None = None) -> None:
    logger.info(
        "request endpoint=%s frame_count=%s elapsed_ms=%.2f error_code=%s",
        endpoint,
        frame_count,
        (perf_counter() - started_at) * 1000,
        error_code.value if error_code else None,
    )


def safe_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("options must be a JSON object")
    return parsed


def parse_frames_meta(value: str) -> list[FrameInput]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("frames_meta must be a JSON array")
    return [FrameInput(**item) for item in parsed]


async def read_valid_image(upload: UploadFile, settings: Settings) -> tuple[bytes, Image.Image]:
    payload = await upload.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"error_code": ErrorCode.file_too_large, "message": "File exceeds 10MB limit"})
    try:
        with Image.open(BytesIO(payload)) as uploaded:
            image_format = uploaded.format
            if image_format not in {"PNG", "JPEG"}:
                raise ValueError("invalid format")
            return payload, uploaded.copy()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": ErrorCode.invalid_file_type, "message": "Only PNG and JPEG files are allowed"},
        ) from exc


def upload_key(upload: UploadFile) -> str:
    filename = upload.filename or ""
    stem = Path(filename).stem
    return stem or filename


def has_selected_images(request: FrameChatRequest) -> bool:
    return any(image.base64 for image in request.selected_images)


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/v1/health")
def health(settings: Settings = Depends(get_settings), registry: ModelRegistry = Depends(get_model_registry)):
    default_model = registry.get(settings.default_model_name)
    vlm_provider = settings.vlm_provider.strip().lower()
    vlm_available = bool(settings.ollama_model) if vlm_provider == "ollama" else bool(settings.openai_api_key)
    return {
        "status": "ok",
        "service": settings.service_name,
        "framework": "FastAPI",
        "port": settings.port,
        "base_url": settings.base_url,
        "storage_policy": settings.storage_policy,
        "default_model": settings.default_model_name,
        "model_loaded": bool(default_model and default_model.loaded),
        "model_status": {
            "heatmap_model_loaded": bool(default_model and default_model.loaded),
            "scanpath_model_loaded": True,
            "vlm_available": vlm_available,
            "vlm_provider": vlm_provider,
            "vlm_model": settings.ollama_model if vlm_provider == "ollama" else settings.openai_model,
        },
        "device": default_model.device if default_model else "unavailable",
        "cuda_available": bool(default_model and default_model.cuda_available),
        "cuda_device_name": default_model.cuda_device_name if default_model else None,
        "models": registry.snapshot(),
    }


@app.post("/api/v1/frames/analyze", response_model=FrameAnalysisBundle)
async def analyze_frames(
    files: list[UploadFile] = File(...),
    frames_meta: str = Form(...),
    options: str | None = Form(None),
    model_name: str | None = Form(None),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    started_at = perf_counter()
    try:
        frame_inputs = parse_frames_meta(frames_meta)
        parsed_options = safe_json_object(options)
    except Exception as exc:
        log_request("/api/v1/frames/analyze", started_at, error_code=ErrorCode.invalid_frame_metadata)
        return error_response(400, ErrorCode.invalid_frame_metadata, str(exc))

    frame_count = len(frame_inputs)
    if frame_count < 1 or frame_count > settings.max_frames:
        log_request("/api/v1/frames/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.invalid_frame_count)
        return error_response(400, ErrorCode.invalid_frame_count, f"Frame count must be between 1 and {settings.max_frames}")

    saliency_model = registry.begin_request_model(model_name)
    if saliency_model is None:
        return error_response(400, ErrorCode.invalid_frame_metadata, f"Unsupported model_name: {model_name}")
    if not saliency_model.loaded:
        registry.end_request()
        return error_response(503, ErrorCode.model_not_ready, "Model is not ready")

    uploads_by_key = {upload_key(upload): upload for upload in files}
    uploads_by_filename = {upload.filename or "": upload for upload in files}
    total_bytes = 0
    prepared_frames = []
    results = []

    try:
        for frame in frame_inputs:
            if not frame.client_frame_id.strip() or not frame.frame_name.strip():
                return error_response(400, ErrorCode.invalid_frame_metadata, "client_frame_id and frame_name are required")
            if frame.width is None or frame.height is None:
                return error_response(400, ErrorCode.invalid_frame_metadata, "width and height are required")
            width = parse_positive_int(float(frame.width), "width")
            height = parse_positive_int(float(frame.height), "height")
            file_key = frame.file_key or frame.client_frame_id
            upload = uploads_by_key.get(file_key) or uploads_by_filename.get(file_key)
            if upload is None:
                return error_response(400, ErrorCode.invalid_frame_metadata, f"No uploaded file matched file_key: {file_key}")

            try:
                payload, image = await read_valid_image(upload, settings)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                return error_response(exc.status_code, detail.get("error_code", ErrorCode.invalid_file_type), detail.get("message", "Invalid image"))

            total_bytes += len(payload)
            if total_bytes > settings.max_total_upload_bytes:
                return error_response(413, ErrorCode.file_too_large, "Total payload exceeds 100MB limit")

            try:
                original = normalize_upload(image, width, height)
            except Exception as exc:
                logger.exception("heatmap inference failed")
                log_request("/api/v1/frames/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.heatmap_inference_failed)
                return error_response(500, ErrorCode.heatmap_inference_failed, str(exc))

            prepared_frames.append(
                {
                    "frame": frame,
                    "width": width,
                    "height": height,
                    "original": original,
                }
            )

        try:
            saliencies = saliency_model.predict_many([item["original"] for item in prepared_frames])
        except Exception as exc:
            logger.exception("heatmap inference failed")
            log_request("/api/v1/frames/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.heatmap_inference_failed)
            return error_response(500, ErrorCode.heatmap_inference_failed, str(exc))

        for item, saliency in zip(prepared_frames, saliencies, strict=True):
            frame = item["frame"]
            original = item["original"]
            heatmap = saliency_to_heatmap(saliency)
            heatmap_overlay = make_overlay(original, heatmap, float(parsed_options.get("heatmap_alpha", settings.overlay_alpha)))

            try:
                metrics = build_scanpath_metrics(saliency)
                scanpath_overlay = make_scanpath_overlay(original, metrics)
            except Exception as exc:
                logger.exception("scanpath inference failed")
                log_request("/api/v1/frames/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.scanpath_inference_failed)
                return error_response(500, ErrorCode.scanpath_inference_failed, str(exc))

            results.append(
                {
                    "client_frame_id": frame.client_frame_id,
                    "figma_node_id": frame.figma_node_id,
                    "frame_name": frame.frame_name,
                    "width": item["width"],
                    "height": item["height"],
                    "order_index": frame.order_index,
                    "metrics": metrics,
                    "artifacts": {
                        "original": image_to_artifact(original, "original"),
                        "heatmap": image_to_artifact(heatmap, "heatmap"),
                        "heatmap_overlay": image_to_artifact(heatmap_overlay, "heatmap_overlay"),
                        "scanpath_overlay": image_to_artifact(scanpath_overlay, "scanpath_overlay"),
                    },
                }
            )

        bundle = FrameAnalysisBundle(
            analysis_bundle_id=make_bundle_id("frames"),
            created_at=now_iso(),
            frames=results,
            model_info=FrameModelInfo(
                heatmap_model=saliency_model.name,
                heatmap_version=saliency_model.version,
                heatmap_backend=saliency_model.backend,
            ),
        )
        log_request("/api/v1/frames/analyze", started_at, frame_count=frame_count)
        return bundle
    finally:
        registry.end_request()


@app.post("/api/v1/frames/chat/stream")
async def frame_chat_stream(
    request: FrameChatRequest,
    settings: Settings = Depends(get_settings),
):
    started_at = perf_counter()

    async def event_stream():
        if not request.question.strip():
            yield sse_message("error", {"error_code": ErrorCode.invalid_frame_metadata, "message": "question is required"})
            log_request("/api/v1/frames/chat/stream", started_at, error_code=ErrorCode.invalid_frame_metadata)
            return
        if not has_selected_images(request):
            yield sse_message("error", {"error_code": ErrorCode.invalid_frame_metadata, "message": "selected_images are required for frame chat"})
            log_request("/api/v1/frames/chat/stream", started_at, error_code=ErrorCode.invalid_frame_metadata)
            return
        if settings.vlm_provider.strip().lower() != "ollama":
            try:
                yield sse_message("progress", {"stage": "evaluating", "message": "프레임 이미지를 VLM으로 평가 중입니다.", "progress": 0.1})
                response = await evaluate_frame_chat(request, settings)
                yield sse_message(
                    "final",
                    {
                        "answer": response.answer.model_dump(),
                        "provider": response.provider,
                        "model": response.model,
                        "progress": 1,
                    },
                )
                log_request("/api/v1/frames/chat/stream", started_at)
            except Exception as exc:
                logger.warning("frame chat stream vlm evaluation failed: %s", exc)
                yield sse_message("error", {"error_code": ErrorCode.vlm_evaluation_failed, "message": "Frame chat evaluation failed"})
                log_request("/api/v1/frames/chat/stream", started_at, error_code=ErrorCode.vlm_evaluation_failed)
            return

        try:
            async for item in stream_frame_chat_events(request, settings):
                yield sse_message(item["event"], item["data"])
            log_request("/api/v1/frames/chat/stream", started_at)
        except Exception as exc:
            logger.warning("frame chat stream vlm evaluation failed: %s", exc)
            yield sse_message("error", {"error_code": ErrorCode.vlm_evaluation_failed, "message": "Frame chat evaluation failed"})
            log_request("/api/v1/frames/chat/stream", started_at, error_code=ErrorCode.vlm_evaluation_failed)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/analyses")
async def create_analysis(
    file: UploadFile = File(...),
    frame_id: str = Form(...),
    frame_name: str = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    export_scale: float | None = Form(None),
    plugin_version: str | None = Form(None),
    model_name: str | None = Form(None),
    options: str | None = Form(None),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    saliency_model = registry.begin_request_model(model_name)
    if saliency_model is None:
        return error_response(400, ErrorCode.invalid_frame_metadata, f"Unsupported model_name: {model_name}")
    if not saliency_model.loaded:
        registry.end_request()
        return error_response(503, ErrorCode.model_not_ready, "Model is not ready")
    if not frame_id.strip() or not frame_name.strip():
        return error_response(400, ErrorCode.invalid_frame_metadata, "frame_id and frame_name are required")

    parsed_width = parse_positive_int(width, "width")
    parsed_height = parse_positive_int(height, "height")
    parsed_options = {}
    if options:
        try:
            parsed_options = json.loads(options)
            if not isinstance(parsed_options, dict):
                raise ValueError("options must be a JSON object")
        except ValueError as exc:
            return error_response(400, ErrorCode.invalid_frame_metadata, str(exc))

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        return error_response(413, ErrorCode.file_too_large, "File exceeds 10MB limit")

    try:
        with Image.open(BytesIO(payload)) as uploaded:
            image_format = uploaded.format
            if image_format not in {"PNG", "JPEG"}:
                return error_response(400, ErrorCode.invalid_file_type, "Only PNG and JPEG files are allowed")
            image = uploaded.copy()
    except Exception:
        return error_response(400, ErrorCode.invalid_file_type, "Only PNG and JPEG files are allowed")

    try:
        try:
            result = analyze_image(
                job_id=make_job_id(),
                image=image,
                width=parsed_width,
                height=parsed_height,
                model=saliency_model,
                overlay_alpha=settings.overlay_alpha,
            )
        except Exception as exc:
            logging.exception("analysis failed")
            return error_response(500, ErrorCode.model_inference_failed, str(exc))

        result["request"] = {
            "frame_id": frame_id,
            "frame_name": frame_name,
            "width": parsed_width,
            "height": parsed_height,
            "export_scale": export_scale,
            "plugin_version": plugin_version,
            "model_name": model_name or settings.default_model_name,
            "options": parsed_options,
        }
        return result
    finally:
        registry.end_request()
