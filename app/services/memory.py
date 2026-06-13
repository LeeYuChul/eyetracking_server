from __future__ import annotations

from PIL import Image, ImageFilter

from app.models.flow import MemoryMetrics, TargetFrameResult, VisualArtifact
from app.services.artifacts import image_to_artifact
from app.services.image_processing import make_overlay


def make_memory_blur(
    *,
    original: Image.Image,
    heatmap: Image.Image | None,
    temporal_distance: int,
    scanpath_length: float,
    options: dict,
) -> tuple[Image.Image, MemoryMetrics]:
    base_blur = float(options.get("blur_strength_base", 8))
    temporal_weight = float(options.get("temporal_decay_weight", 1.0))
    scanpath_weight = float(options.get("scanpath_weight", 0.6))
    load = min(1.0, scanpath_length / max(float(original.width + original.height), 1.0))
    blur_radius = max(1.0, base_blur + temporal_distance * temporal_weight * 3.0 + load * scanpath_weight * 8.0)

    blurred = original.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if heatmap is None:
        clarity_mask = Image.new("L", original.size, 80)
        attention_mean = 0.0
    else:
        heat_alpha = heatmap.convert("RGBA").getchannel("A").resize(original.size, Image.Resampling.BILINEAR)
        retention_scale = max(0.12, 1.0 / (1.0 + temporal_distance * 0.65 + load * 0.5))
        clarity_mask = heat_alpha.point(lambda value: int(min(255, value * retention_scale)))
        attention_mean = sum(heat_alpha.histogram()[idx] * idx for idx in range(256)) / max(original.width * original.height * 255, 1)

    memory = Image.composite(original.convert("RGB"), blurred, clarity_mask)
    estimated_retention = max(0.05, min(1.0, (attention_mean + 0.25) / (1.0 + temporal_distance * 0.55 + load * 0.4)))
    metrics = MemoryMetrics(
        estimated_retention=round(estimated_retention, 4),
        blur_strength_avg=round(blur_radius, 2),
        temporal_distance=temporal_distance,
    )
    return memory, metrics


def build_target_frame_result(
    *,
    client_frame_id: str,
    original: Image.Image,
    heatmap: Image.Image | None,
    temporal_distance: int,
    scanpath_length: float,
    options: dict,
) -> TargetFrameResult:
    memory_blur, metrics = make_memory_blur(
        original=original,
        heatmap=heatmap,
        temporal_distance=temporal_distance,
        scanpath_length=scanpath_length,
        options=options,
    )
    full_overlay = make_overlay(memory_blur, heatmap, 0.4) if heatmap is not None else memory_blur
    artifacts: dict[str, VisualArtifact] = {
        "memory_blur": image_to_artifact(memory_blur, "memory_blur"),
        "full_overlay": image_to_artifact(full_overlay, "full_overlay"),
    }
    return TargetFrameResult(
        client_frame_id=client_frame_id,
        temporal_distance=temporal_distance,
        memory_metrics=metrics,
        artifacts=artifacts,
    )
