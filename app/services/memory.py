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
    depth_blur_strength: float | None = None,
    cumulative_blur_strength: float | None = None,
) -> tuple[Image.Image, MemoryMetrics]:
    base_blur = float(options.get("blur_strength_base", 10))
    scanpath_weight = float(options.get("scanpath_weight", 0.0))
    load = min(1.0, scanpath_length / max(float(original.width + original.height), 1.0))
    if depth_blur_strength is None:
        depth_blur_strength = calculate_depth_blur_strength(temporal_distance=temporal_distance, options=options)
    applied_blur_strength = float(cumulative_blur_strength if cumulative_blur_strength is not None else depth_blur_strength)
    blur_radius = max(1.0, applied_blur_strength + load * scanpath_weight * 8.0)

    blurred = original.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if heatmap is None:
        clarity_mask = Image.new("L", original.size, 80)
        attention_mean = 0.0
    else:
        heat_alpha = heatmap.convert("RGBA").getchannel("A").resize(original.size, Image.Resampling.BILINEAR)
        retention_scale = max(0.08, 1.0 / (1.0 + applied_blur_strength / max(base_blur * 2.5, 1.0) + load * 0.5))
        clarity_mask = heat_alpha.point(lambda value: int(min(255, value * retention_scale)))
        attention_mean = sum(heat_alpha.histogram()[idx] * idx for idx in range(256)) / max(original.width * original.height * 255, 1)

    memory = Image.composite(original.convert("RGB"), blurred, clarity_mask)
    estimated_retention = max(0.05, min(1.0, (attention_mean + 0.25) / (1.0 + applied_blur_strength / max(base_blur * 2.0, 1.0) + load * 0.4)))
    metrics = MemoryMetrics(
        estimated_retention=round(estimated_retention, 4),
        blur_strength_avg=round(blur_radius, 2),
        temporal_distance=temporal_distance,
        depth_blur_strength=round(float(depth_blur_strength), 2),
        cumulative_blur_strength=round(float(cumulative_blur_strength), 2) if cumulative_blur_strength is not None else None,
    )
    return memory, metrics


def calculate_depth_blur_strength(*, temporal_distance: int, options: dict) -> float:
    base = float(options.get("depth_blur_base", options.get("blur_strength_base", 10)))
    step = float(options.get("depth_blur_step", 10))
    return max(0.0, base + max(0, temporal_distance) * step)


def build_target_frame_result(
    *,
    client_frame_id: str,
    original: Image.Image,
    heatmap: Image.Image | None,
    temporal_distance: int,
    scanpath_length: float,
    options: dict,
    depth_blur_strength: float | None = None,
    cumulative_blur_strength: float | None = None,
) -> TargetFrameResult:
    memory_blur, metrics = make_memory_blur(
        original=original,
        heatmap=heatmap,
        temporal_distance=temporal_distance,
        scanpath_length=scanpath_length,
        options=options,
        depth_blur_strength=depth_blur_strength,
        cumulative_blur_strength=cumulative_blur_strength,
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
