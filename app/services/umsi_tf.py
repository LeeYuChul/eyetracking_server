from __future__ import annotations

import logging
import os
import gc
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SHAPE_R = 256
SHAPE_C = 256
SHAPE_R_OUT = 512
SHAPE_C_OUT = 512
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def pad_to_shape(img: np.ndarray, rows: int, cols: int, channels: int = 3) -> np.ndarray:
    padded = np.zeros((rows, cols, channels), dtype=np.uint8)
    original_rows, original_cols = img.shape[:2]
    rows_rate = original_rows / rows
    cols_rate = original_cols / cols

    if rows_rate > cols_rate:
        new_cols = (original_cols * rows) // original_rows
        resized = np.asarray(Image.fromarray(img).resize((new_cols, rows), Image.Resampling.BILINEAR))
        new_cols = min(new_cols, cols)
        left = (cols - new_cols) // 2
        padded[:, left : left + new_cols] = resized[:, :new_cols]
    else:
        new_rows = (original_rows * cols) // original_cols
        resized = np.asarray(Image.fromarray(img).resize((cols, new_rows), Image.Resampling.BILINEAR))
        new_rows = min(new_rows, rows)
        top = (rows - new_rows) // 2
        padded[top : top + new_rows, :] = resized[:new_rows, :]

    return padded


def postprocess_prediction(pred: np.ndarray, rows: int, cols: int) -> np.ndarray:
    pred_rows, pred_cols = pred.shape[:2]
    rows_rate = rows / pred_rows
    cols_rate = cols / pred_cols

    if rows_rate > cols_rate:
        new_cols = (pred_cols * rows) // pred_rows
        resized = np.asarray(Image.fromarray(pred).resize((new_cols, rows), Image.Resampling.BILINEAR))
        left = max(0, (resized.shape[1] - cols) // 2)
        img = resized[:, left : left + cols]
    else:
        new_rows = (pred_rows * cols) // pred_cols
        resized = np.asarray(Image.fromarray(pred).resize((cols, new_rows), Image.Resampling.BILINEAR))
        top = max(0, (resized.shape[0] - rows) // 2)
        img = resized[top : top + rows, :]

    img = np.array(img, dtype=np.float32, copy=True)
    img -= img.min()
    peak = float(img.max())
    if peak > 0:
        img /= peak
    return img


class UMSITensorFlowRunner:
    def __init__(self, weights_path: Path, cuda_device_index: int = 0) -> None:
        self.weights_path = weights_path
        self.cuda_device_index = cuda_device_index
        self.model = None
        self.backend = "tensorflow-umsi++"
        self.device = "cpu"

    def load(self) -> None:
        self.backend = "tensorflow-umsi++-subprocess"
        self.device = f"cuda:{self.cuda_device_index}"
        logger.info("Prepared original UMSI++ TensorFlow subprocess runner on %s", self.device)

    def load_in_process(self) -> None:
        cuda_data_dir = find_cuda_data_dir()
        if cuda_data_dir is not None:
            os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_data_dir}")

        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.device = f"cuda:{self.cuda_device_index}"

        model = build_umsi_model()
        model.load_weights(str(self.weights_path))
        self.model = model
        logger.info("Loaded original UMSI++ TensorFlow model on %s", self.device)

    def predict(self, image: Image.Image) -> np.ndarray:
        return self.predict_many([image])[0]

    def predict_many(self, images: list[Image.Image]) -> list[np.ndarray]:
        if not images:
            return []
        with tempfile.TemporaryDirectory(prefix="eyetrack_umsi_") as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            input_dir.mkdir()
            output_path = temp_path / "predictions.npz"
            for index, image in enumerate(images):
                image.convert("RGB").save(input_dir / f"{index:04d}.png", format="PNG")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device_index)
            command = [
                sys.executable,
                "-m",
                "app.services.umsi_tf_worker",
                str(self.weights_path),
                str(input_dir),
                str(output_path),
            ]
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    "UMSI++ TensorFlow worker failed "
                    f"code={completed.returncode} stdout={completed.stdout[-1000:]} stderr={completed.stderr[-1000:]}"
                )
            with np.load(output_path) as data:
                return [data[f"prediction_{index}"].astype(np.float32) for index in range(len(images))]

    def predict_in_process(self, image: Image.Image) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("UMSI++ TensorFlow model is not loaded")

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = rgb[..., ::-1]
        padded = pad_to_shape(bgr, SHAPE_R, SHAPE_C).astype(np.float32)
        padded -= MEAN_BGR
        batch = padded[None, ...]
        preds = self.model.predict(batch, verbose=0)
        pred_map = preds[0] if isinstance(preds, (list, tuple)) else preds
        pred_map = np.squeeze(pred_map[0]).astype(np.float32)
        return postprocess_prediction(pred_map, image.height, image.width)

    def unload(self) -> None:
        if self.model is None:
            self.device = "cpu"
            logger.info("Released original UMSI++ TensorFlow subprocess runner")
            return
        model = self.model
        self.model = None
        del model
        self.device = "cpu"
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                logger.debug("TensorFlow default graph reset skipped", exc_info=True)
        except Exception:
            logger.debug("TensorFlow session cleanup skipped", exc_info=True)
        gc.collect()
        logger.info("Unloaded original UMSI++ TensorFlow model")


def build_umsi_model():
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    inp = layers.Input(shape=(SHAPE_R, SHAPE_C, 3), name="input_1")
    xception = build_xception_backbone(inp)
    xout = xception.output

    c0 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp_csep0")(xout)
    c6 = layers.DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False, name="aspp_csepd6_depthwise")(xout)
    c12 = layers.DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False, name="aspp_csepd12_depthwise")(xout)
    c18 = layers.DepthwiseConv2D((3, 3), dilation_rate=(18, 18), padding="same", use_bias=False, name="aspp_csepd18_depthwise")(xout)

    c6 = layers.BatchNormalization(name="aspp_csepd6_depthwise_BN")(c6)
    c12 = layers.BatchNormalization(name="aspp_csepd12_depthwise_BN")(c12)
    c18 = layers.BatchNormalization(name="aspp_csepd18_depthwise_BN")(c18)
    c6 = layers.Activation("relu", name="activation_2")(c6)
    c12 = layers.Activation("relu", name="activation_4")(c12)
    c18 = layers.Activation("relu", name="activation_6")(c18)
    c6 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp_csepd6_pointwise")(c6)
    c12 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp_csepd12_pointwise")(c12)
    c18 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp_csepd18_pointwise")(c18)

    c0 = layers.BatchNormalization(name="aspp0_BN")(c0)
    c6 = layers.BatchNormalization(name="aspp_csepd6_pointwise_BN")(c6)
    c12 = layers.BatchNormalization(name="aspp_csepd12_pointwise_BN")(c12)
    c18 = layers.BatchNormalization(name="aspp_csepd18_pointwise_BN")(c18)
    c0 = layers.Activation("relu", name="aspp0_activation")(c0)
    c6 = layers.Activation("relu", name="activation_3")(c6)
    c12 = layers.Activation("relu", name="activation_5")(c12)
    c18 = layers.Activation("relu", name="activation_7")(c18)
    concat1 = layers.Concatenate(name="concatenate_1")([c0, c6, c12, c18])

    g = layers.Conv2D(256, (3, 3), strides=(3, 3), padding="same", use_bias=False, name="global_conv")(xout)
    g = layers.BatchNormalization(name="global_BN")(g)
    g = layers.Activation("relu", name="activation_1")(g)
    g = layers.Dropout(0.3, name="dropout_1")(g)
    g = layers.GlobalAveragePooling2D(name="global_average_pooling2d_1")(g)
    g = layers.Dense(256, name="global_dense")(g)
    classif = layers.Dropout(0.3, name="dropout_2")(g)
    out_classif = layers.Dense(6, activation="softmax", name="out_classif")(classif)

    fusion = layers.Dense(256, name="dense_fusion")(classif)
    fusion = layers.Lambda(lambda t: tf.tile(tf.reshape(t, (tf.shape(t)[0], 1, 1, 256)), [1, 32, 32, 1]), name="lambda_1")(fusion)
    concat2 = layers.Concatenate(name="concatenate_2")([concat1, fusion])

    x = layers.Conv2D(256, (1, 1), padding="same", use_bias=False, name="concat_projection")(concat2)
    x = layers.BatchNormalization(name="concat_projection_BN")(x)
    x = layers.Activation("relu", name="activation_8")(x)
    x = layers.Dropout(0.3, name="dropout_3")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="dec_c1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="dec_c2")(x)
    x = layers.Dropout(0.3, name="dec_dp1")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_ups1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="dec_c3")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="dec_c4")(x)
    x = layers.Dropout(0.3, name="dec_dp2")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_ups2")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="dec_c5")(x)
    x = layers.Dropout(0.3, name="dec_dp3")(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear", name="dec_ups3")(x)
    out_heatmap = layers.Conv2D(1, (1, 1), padding="same", use_bias=False, name="dec_c_cout")(x)

    return Model(inp, [out_heatmap, out_classif], name="umsi")


def build_xception_backbone(img_input):
    from tensorflow.keras import Model, layers

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name="block1_conv1")(img_input)
    x = layers.BatchNormalization(name="block1_conv1_bn")(x)
    x = layers.Activation("relu", name="block1_conv1_act")(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(name="block1_conv2_bn")(x)
    x = layers.Activation("relu", name="block1_conv2_act")(x)

    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding="same", use_bias=False, name="conv2d_1")(x)
    residual = layers.BatchNormalization(name="batch_normalization_1")(residual)
    x = layers.SeparableConv2D(128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1")(x)
    x = layers.BatchNormalization(name="block2_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block2_sepconv2_act")(x)
    x = layers.SeparableConv2D(128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2")(x)
    x = layers.BatchNormalization(name="block2_sepconv2_bn")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="block2_pool")(x)
    x = layers.Add(name="add_1")([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False, name="conv2d_2")(x)
    residual = layers.BatchNormalization(name="batch_normalization_2")(residual)
    x = layers.Activation("relu", name="block3_sepconv1_act")(x)
    x = layers.SeparableConv2D(256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1")(x)
    x = layers.BatchNormalization(name="block3_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block3_sepconv2_act")(x)
    x = layers.SeparableConv2D(256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2")(x)
    x = layers.BatchNormalization(name="block3_sepconv2_bn")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="block3_pool")(x)
    x = layers.Add(name="add_2")([x, residual])

    residual = layers.Conv2D(728, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="conv2d_3")(x)
    residual = layers.BatchNormalization(name="batch_normalization_3")(residual)
    x = layers.Activation("relu", name="block4_sepconv1_act")(x)
    x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1")(x)
    x = layers.BatchNormalization(name="block4_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block4_sepconv2_act")(x)
    x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2")(x)
    x = layers.BatchNormalization(name="block4_sepconv2_bn")(x)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="block4_pool")(x)
    x = layers.Add(name="add_3")([x, residual])

    for idx in range(5, 13):
        residual = x
        prefix = f"block{idx}"
        x = layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
        x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv1")(x)
        x = layers.BatchNormalization(name=prefix + "_sepconv1_bn")(x)
        x = layers.Activation("relu", name=prefix + "_sepconv2_act")(x)
        x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv2")(x)
        x = layers.BatchNormalization(name=prefix + "_sepconv2_bn")(x)
        x = layers.Activation("relu", name=prefix + "_sepconv3_act")(x)
        x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv3")(x)
        x = layers.BatchNormalization(name=prefix + "_sepconv3_bn")(x)
        x = layers.Add(name=f"add_{idx - 1}")([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="conv2d_4")(x)
    residual = layers.BatchNormalization(name="batch_normalization_4")(residual)
    x = layers.Activation("relu", name="block13_sepconv1_act")(x)
    x = layers.SeparableConv2D(728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1")(x)
    x = layers.BatchNormalization(name="block13_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block13_sepconv2_act")(x)
    x = layers.SeparableConv2D(1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2")(x)
    x = layers.BatchNormalization(name="block13_sepconv2_bn")(x)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="block13_pool")(x)
    x = layers.Add(name="add_12")([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1")(x)
    x = layers.BatchNormalization(name="block14_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block14_sepconv1_act")(x)
    x = layers.SeparableConv2D(2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv2")(x)
    x = layers.BatchNormalization(name="block14_sepconv2_bn")(x)
    x = layers.Activation("relu", name="block14_sepconv2_act")(x)

    return Model(img_input, x, name="xception")


def find_cuda_data_dir() -> str | None:
    import site

    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "nvidia" / "cuda_nvcc"
        if (candidate / "nvvm" / "libdevice" / "libdevice.10.bc").exists():
            return str(candidate)
    return None
