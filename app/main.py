import json
import logging
import secrets
from datetime import UTC, datetime
from io import BytesIO

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import Settings, get_settings
from app.models.job import ErrorCode
from app.services.model import ModelRegistry, SaliencyModel
from app.services.pipeline import analyze_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

settings = get_settings()
model_registry = ModelRegistry(
    saliency_model_path=settings.saliency_model_path,
    runtime=settings.model_runtime,
    device_preference=settings.device,
    cuda_device_index=settings.cuda_device_index,
    allowed_model_names=settings.allowed_model_names,
    default_model_name=settings.default_model_name,
)
model_registry.load()

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


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": code, "message": message})


def parse_positive_int(value: float, field_name: str) -> int:
    if value <= 0 or int(value) != value:
        raise HTTPException(
            status_code=400,
            detail={"error_code": ErrorCode.invalid_frame_metadata, "message": f"{field_name} must be a positive integer"},
        )
    return int(value)


@app.get("/api/v1/health")
def health(settings: Settings = Depends(get_settings), registry: ModelRegistry = Depends(get_model_registry)):
    default_model = registry.get(settings.default_model_name)
    return {
        "status": "ok",
        "service": settings.service_name,
        "port": settings.port,
        "default_model": settings.default_model_name,
        "model_loaded": bool(default_model and default_model.loaded),
        "device": default_model.device if default_model else "unavailable",
        "cuda_available": bool(default_model and default_model.cuda_available),
        "cuda_device_name": default_model.cuda_device_name if default_model else None,
        "models": registry.snapshot(),
    }


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
    saliency_model = registry.get(model_name)
    if saliency_model is None:
        return error_response(400, ErrorCode.invalid_frame_metadata, f"Unsupported model_name: {model_name}")
    if not saliency_model.loaded:
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

