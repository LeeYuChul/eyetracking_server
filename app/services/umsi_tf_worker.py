from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.umsi_tf import MEAN_BGR, SHAPE_C, SHAPE_R, build_umsi_model, find_cuda_data_dir, pad_to_shape, postprocess_prediction


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: umsi_tf_worker <weights_path> <input_dir> <output_path>", file=sys.stderr)
        return 2

    weights_path = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    cuda_data_dir = find_cuda_data_dir()
    if cuda_data_dir is not None:
        os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_data_dir}")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = build_umsi_model()
    model.load_weights(str(weights_path))

    predictions = []
    for image_path in sorted(input_dir.glob("*.png")):
        with Image.open(image_path) as image:
            predictions.append(predict_one(model, image.copy()))

    np.savez_compressed(output_path, **{f"prediction_{index}": prediction for index, prediction in enumerate(predictions)})
    tf.keras.backend.clear_session()
    return 0


def predict_one(model, image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = rgb[..., ::-1]
    padded = pad_to_shape(bgr, SHAPE_R, SHAPE_C).astype(np.float32)
    padded -= MEAN_BGR
    batch = padded[None, ...]
    preds = model.predict(batch, verbose=0)
    pred_map = preds[0] if isinstance(preds, (list, tuple)) else preds
    pred_map = np.squeeze(pred_map[0]).astype(np.float32)
    return postprocess_prediction(pred_map, image.height, image.width)


if __name__ == "__main__":
    raise SystemExit(main())
