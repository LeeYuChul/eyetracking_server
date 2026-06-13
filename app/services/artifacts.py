from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.models.flow import ImageArtifactInput, VisualArtifact


def image_to_base64(image: Image.Image, *, mime_type: str = "image/png") -> str:
    buffer = BytesIO()
    fmt = "PNG" if mime_type == "image/png" else "JPEG"
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_artifact(image: Image.Image, artifact_type: str) -> VisualArtifact:
    normalized = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
    return VisualArtifact(
        artifact_type=artifact_type,
        mime_type="image/png",
        base64=image_to_base64(normalized, mime_type="image/png"),
        width=normalized.width,
        height=normalized.height,
    )


def artifact_to_data_url(artifact: ImageArtifactInput | VisualArtifact) -> str:
    if artifact.base64.startswith("data:"):
        return artifact.base64
    mime_type = artifact.mime_type or "image/png"
    return f"data:{mime_type};base64,{artifact.base64}"


def image_from_artifact(artifact: ImageArtifactInput | VisualArtifact) -> Image.Image:
    value = artifact.base64
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    payload = base64.b64decode(value)
    with Image.open(BytesIO(payload)) as image:
        return image.copy()
