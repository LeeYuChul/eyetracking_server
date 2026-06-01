import base64
from io import BytesIO
from typing import Any

from PIL import Image

from app.services.image_processing import make_overlay, normalize_upload, saliency_to_heatmap
from app.services.model import SaliencyModel
from app.services.report import build_report


def png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_image(
    *,
    job_id: str,
    image: Image.Image,
    width: int,
    height: int,
    model: SaliencyModel,
    overlay_alpha: float,
) -> dict[str, Any]:
    original = normalize_upload(image, width, height)
    saliency = model.predict(original)
    heatmap = saliency_to_heatmap(saliency)
    overlay = make_overlay(original, heatmap, overlay_alpha)
    report = build_report(job_id, saliency, model.name, model.version, model.backend)

    return {
        "job_id": job_id,
        "status": "completed",
        "report": report,
        "assets": {
            "image_mime_type": "image/png",
            "heatmap_png_base64": png_base64(heatmap.convert("RGB")),
            "overlay_png_base64": png_base64(overlay),
        },
    }
