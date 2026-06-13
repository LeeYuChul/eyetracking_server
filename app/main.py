import json
import logging
import secrets
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import Settings, get_settings
from app.models.flow import (
    AnalysisBundle,
    ClientFrameInput,
    FlowParseRequest,
    FlowParseResponse,
    ModelInfo,
    PrepareTargetRequest,
    TargetResult,
    UxEvaluateRequest,
)
from app.models.job import ErrorCode
from app.services.model import ModelRegistry, SaliencyModel
from app.services.artifacts import image_from_artifact, image_to_artifact
from app.services.flow import build_flow_parse_response, resolve_target_path
from app.services.image_processing import make_overlay, normalize_upload, saliency_to_heatmap
from app.services.memory import build_target_frame_result
from app.services.pipeline import analyze_image
from app.services.scanpath import build_scanpath_metrics, make_scanpath_overlay
from app.services.vlm import evaluate_ux, evaluate_with_heuristic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

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

app = FastAPI(title=settings.service_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


def parse_frames_meta(value: str) -> list[ClientFrameInput]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("frames_meta must be a JSON array")
    return [ClientFrameInput(**item) for item in parsed]


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


def has_selected_images(request: UxEvaluateRequest) -> bool:
    images = request.evidence.get("selected_images")
    return isinstance(images, list) and any(isinstance(item, dict) and item.get("base64") for item in images)


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


@app.post("/api/v1/flow/parse", response_model=FlowParseResponse)
def parse_flow(request: FlowParseRequest):
    started_at = perf_counter()
    response = build_flow_parse_response(request.frames)
    log_request("/api/v1/flow/parse", started_at, frame_count=len(request.frames))
    return response


@app.post("/api/v1/flow/analyze", response_model=AnalysisBundle)
async def analyze_flow(
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
        log_request("/api/v1/flow/analyze", started_at, error_code=ErrorCode.invalid_frame_metadata)
        return error_response(400, ErrorCode.invalid_frame_metadata, str(exc))

    frame_count = len(frame_inputs)
    if frame_count < 1 or frame_count > settings.max_frames:
        log_request("/api/v1/flow/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.invalid_frame_count)
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
    parse_response = build_flow_parse_response(frame_inputs)
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
                saliency = saliency_model.predict(original)
                heatmap = saliency_to_heatmap(saliency)
                heatmap_overlay = make_overlay(original, heatmap, float(parsed_options.get("heatmap_alpha", settings.overlay_alpha)))
            except Exception as exc:
                logger.exception("heatmap inference failed")
                log_request("/api/v1/flow/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.heatmap_inference_failed)
                return error_response(500, ErrorCode.heatmap_inference_failed, str(exc))

            try:
                metrics = build_scanpath_metrics(saliency)
                scanpath_overlay = make_scanpath_overlay(original, metrics)
            except Exception as exc:
                logger.exception("scanpath inference failed")
                log_request("/api/v1/flow/analyze", started_at, frame_count=frame_count, error_code=ErrorCode.scanpath_inference_failed)
                return error_response(500, ErrorCode.scanpath_inference_failed, str(exc))

            parsed = next(item for item in parse_response.parsed_frames if item.client_frame_id == frame.client_frame_id)
            results.append(
                {
                    "client_frame_id": frame.client_frame_id,
                    "figma_node_id": frame.figma_node_id,
                    "frame_name": frame.frame_name,
                    "parsed": parsed,
                    "width": width,
                    "height": height,
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

        bundle = AnalysisBundle(
            analysis_bundle_id=make_bundle_id("uxflow"),
            created_at=now_iso(),
            flow_tree=parse_response.flow_tree,
            frames=results,
            warnings=parse_response.warnings,
            model_info=ModelInfo(
                heatmap_model=saliency_model.name,
                heatmap_version=saliency_model.version,
                heatmap_backend=saliency_model.backend,
            ),
        )
        log_request("/api/v1/flow/analyze", started_at, frame_count=frame_count)
        return bundle
    finally:
        registry.end_request()


@app.post("/api/v1/flow/prepare-target", response_model=TargetResult)
def prepare_target(request: PrepareTargetRequest):
    started_at = perf_counter()
    path_frame_ids = resolve_target_path(request.flow_tree, request.target_frame_id)
    if not path_frame_ids:
        log_request("/api/v1/flow/prepare-target", started_at, frame_count=len(request.frames), error_code=ErrorCode.target_path_not_found)
        return error_response(400, ErrorCode.target_path_not_found, "Target frame is not present in the flow tree")

    frames_by_id = {frame.client_frame_id: frame for frame in request.frames}
    results = []
    previous_ids = path_frame_ids[:-1]
    for index, frame_id in enumerate(previous_ids):
        frame = frames_by_id.get(frame_id)
        if frame is None or frame.original_image is None:
            return error_response(400, ErrorCode.invalid_frame_metadata, f"Missing original image for frame: {frame_id}")
        original = image_from_artifact(frame.original_image).convert("RGB")
        heatmap = image_from_artifact(frame.heatmap) if frame.heatmap is not None else None
        metrics = frame.scanpath_metrics or frame.metrics
        scanpath_length = float(metrics.get("scanpath_length", 0.0)) if isinstance(metrics, dict) else 0.0
        temporal_distance = len(path_frame_ids) - index - 1
        results.append(
            build_target_frame_result(
                client_frame_id=frame_id,
                original=original,
                heatmap=heatmap,
                temporal_distance=temporal_distance,
                scanpath_length=scanpath_length,
                options=request.options,
            )
        )

    target_result = TargetResult(
        target_result_id=make_bundle_id("target"),
        target_frame_id=request.target_frame_id,
        path_frame_ids=path_frame_ids,
        frames=results,
        memory_model_options=request.options,
        created_at=now_iso(),
    )
    log_request("/api/v1/flow/prepare-target", started_at, frame_count=len(request.frames))
    return target_result


@app.post("/api/v1/ux/evaluate")
async def ux_evaluate(
    request: UxEvaluateRequest,
    settings: Settings = Depends(get_settings),
):
    started_at = perf_counter()
    if not request.question.strip():
        return error_response(400, ErrorCode.invalid_frame_metadata, "question is required")
    try:
        response = await evaluate_ux(request, settings)
    except Exception as exc:
        logger.warning("vlm evaluation failed: %s", exc)
        log_request("/api/v1/ux/evaluate", started_at, error_code=ErrorCode.vlm_evaluation_failed)
        return error_response(500, ErrorCode.vlm_evaluation_failed, "VLM evaluation failed")
    log_request("/api/v1/ux/evaluate", started_at)
    return response


@app.post("/api/v1/ux/chat")
async def ux_chat(
    request: UxEvaluateRequest,
    settings: Settings = Depends(get_settings),
):
    started_at = perf_counter()
    if not request.question.strip():
        return error_response(400, ErrorCode.invalid_frame_metadata, "question is required")
    if not has_selected_images(request):
        return error_response(400, ErrorCode.invalid_frame_metadata, "selected_images are required for VLM chat")
    try:
        response = await evaluate_ux(request, settings)
    except Exception as exc:
        logger.warning("ux chat vlm evaluation failed: %s", exc)
        log_request("/api/v1/ux/chat", started_at, error_code=ErrorCode.vlm_evaluation_failed)
        return error_response(500, ErrorCode.vlm_evaluation_failed, "VLM chat evaluation failed")
    log_request("/api/v1/ux/chat", started_at)
    return response


@app.post("/api/v1/ux/chat/heuristic")
def ux_heuristic_chat(request: UxEvaluateRequest):
    started_at = perf_counter()
    if not request.question.strip():
        return error_response(400, ErrorCode.invalid_frame_metadata, "question is required")
    response = evaluate_with_heuristic(request)
    log_request("/api/v1/ux/chat/heuristic", started_at)
    return response


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
