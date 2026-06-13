from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from app.models.flow import FixationPoint, FrameMetrics


def build_scanpath_metrics(saliency: np.ndarray, *, max_fixations: int = 8) -> FrameMetrics:
    values = np.clip(saliency.astype(np.float32), 0.0, 1.0)
    height, width = values.shape
    fixations = select_fixations(values, max_fixations=max_fixations)
    scanpath_length = 0.0
    for prev, current in zip(fixations, fixations[1:]):
        scanpath_length += math.dist((prev.x, prev.y), (current.x, current.y))

    hist, _ = np.histogram(values, bins=16, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float32)
    probs = probs / max(float(probs.sum()), 1.0)
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum() / 4.0)
    gy, gx = np.gradient(values)
    complexity = float(np.clip(np.mean(np.sqrt(gx * gx + gy * gy)) * 8.0, 0.0, 1.0))

    return FrameMetrics(
        scanpath_length=round(scanpath_length, 2),
        fixation_count=len(fixations),
        attention_entropy=round(float(np.clip(entropy, 0.0, 1.0)), 4),
        visual_complexity=round(complexity, 4),
        fixations=fixations,
    )


def select_fixations(values: np.ndarray, *, max_fixations: int) -> list[FixationPoint]:
    height, width = values.shape
    candidates: list[tuple[float, int, int]] = []
    grid_rows = max(2, min(6, height // 80 or 2))
    grid_cols = max(2, min(4, width // 80 or 2))

    for row in range(grid_rows):
        y0 = row * height // grid_rows
        y1 = (row + 1) * height // grid_rows
        for col in range(grid_cols):
            x0 = col * width // grid_cols
            x1 = (col + 1) * width // grid_cols
            patch = values[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            flat_index = int(np.argmax(patch))
            py, px = np.unravel_index(flat_index, patch.shape)
            candidates.append((float(patch[py, px]), x0 + int(px), y0 + int(py)))

    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    chosen: list[tuple[float, int, int]] = []
    min_distance = max(24, min(width, height) // 8)
    for score, x, y in candidates:
        if all(math.dist((x, y), (cx, cy)) >= min_distance for _, cx, cy in chosen):
            chosen.append((score, x, y))
        if len(chosen) >= max_fixations:
            break

    chosen.sort(key=lambda item: (item[2], item[1]))
    return [
        FixationPoint(index=index + 1, x=x, y=y, score=round(score, 4))
        for index, (score, x, y) in enumerate(chosen)
    ]


def make_scanpath_overlay(original: Image.Image, metrics: FrameMetrics) -> Image.Image:
    overlay = original.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    points = [(fixation.x, fixation.y) for fixation in metrics.fixations]
    if len(points) > 1:
        draw.line(points, fill=(31, 115, 255, 220), width=max(2, original.width // 140))
    radius = max(8, min(original.width, original.height) // 34)
    for fixation in metrics.fixations:
        x, y = fixation.x, fixation.y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 210), outline=(31, 115, 255, 255), width=2)
        draw.text((x - 4, y - 6), str(fixation.index), fill=(31, 31, 31, 255))
    return overlay.convert("RGB")
