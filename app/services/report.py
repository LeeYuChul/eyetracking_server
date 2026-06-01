from typing import Any

import numpy as np


def build_report(
    job_id: str,
    saliency: np.ndarray,
    model_name: str,
    model_version: str,
    model_backend: str,
) -> dict[str, Any]:
    threshold = float(np.quantile(saliency, 0.90))
    hot_y, hot_x = np.where(saliency >= threshold)
    height, width = saliency.shape

    hotspots: list[dict[str, Any]] = []
    if hot_x.size:
        hotspots.append(
            {
                "label": "primary_attention_cluster",
                "bbox": {
                    "x": int(hot_x.min()),
                    "y": int(hot_y.min()),
                    "width": int(hot_x.max() - hot_x.min() + 1),
                    "height": int(hot_y.max() - hot_y.min() + 1),
                },
                "mean_score": round(float(saliency[hot_y, hot_x].mean()), 4),
            }
        )

    top_band = float(saliency[: max(1, height // 3), :].mean())
    middle_band = float(saliency[height // 3 : max(height // 3 + 1, (height * 2) // 3), :].mean())
    bottom_band = float(saliency[(height * 2) // 3 :, :].mean())
    lowest = min(("top", top_band), ("middle", middle_band), ("bottom", bottom_band), key=lambda item: item[1])

    return {
        "job_id": job_id,
        "summary": "Eye-tracking saliency analysis completed for the submitted mobile UI frame.",
        "attention_hotspots": hotspots,
        "low_attention_areas": [
            {
                "region": lowest[0],
                "mean_score": round(lowest[1], 4),
            }
        ],
        "cta_visibility": {
            "estimate": "unknown",
            "note": "CTA detection is reserved for a later semantic analysis step.",
        },
        "visual_hierarchy": {
            "top_band_score": round(top_band, 4),
            "middle_band_score": round(middle_band, 4),
            "bottom_band_score": round(bottom_band, 4),
        },
        "content_density": {
            "estimate": "medium",
            "edge_variance": round(float(np.var(saliency)), 4),
        },
        "recommendations": [
            "Review whether the primary action overlaps with the highest-attention cluster.",
            "Use this MVP output as predictive guidance, not as a replacement for user testing.",
        ],
        "confidence": {
            "level": "reference_only",
            "message": "This report is predictive guidance and should not be treated as a replacement for user testing.",
        },
        "model": {
            "name": model_name,
            "version": model_version,
            "backend": model_backend,
        },
    }
