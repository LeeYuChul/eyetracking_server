import numpy as np
from PIL import Image


def normalize_upload(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")
    else:
        image = image.convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def saliency_to_heatmap(saliency: np.ndarray) -> Image.Image:
    values = np.clip(saliency, 0.0, 1.0)
    red = np.clip(1.5 * values, 0.0, 1.0)
    green = np.clip(1.5 * (1.0 - np.abs(values - 0.55) * 2.0), 0.0, 1.0)
    blue = np.clip(1.2 * (1.0 - values), 0.0, 1.0)
    alpha = np.clip(values * 1.35, 0.0, 1.0)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def make_overlay(original: Image.Image, heatmap: Image.Image, alpha: float) -> Image.Image:
    alpha = max(0.0, min(1.0, alpha))
    base = original.convert("RGBA")
    colored = heatmap.convert("RGBA")
    channel = colored.getchannel("A").point(lambda value: int(value * alpha))
    colored.putalpha(channel)
    return Image.alpha_composite(base, colored).convert("RGB")
