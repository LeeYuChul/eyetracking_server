import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class SaliencyModel:
    version = "weights-bound-mvp-0.2.0"

    def __init__(
        self,
        *,
        name: str,
        weights_path: Path | None,
        runtime: str = "auto",
        device_preference: str = "cuda",
        cuda_device_index: int = 0,
    ) -> None:
        self.name = name
        self.weights_path = weights_path
        self.runtime = runtime
        self.loaded = False
        self.device = "cpu"
        self.device_preference = device_preference
        self.cuda_device_index = cuda_device_index
        self.cuda_available = False
        self.cuda_usable = False
        self.cuda_device_name: str | None = None
        self.backend = "heuristic"
        self.keras_version: str | None = None
        self.layer_count: int | None = None
        self.load_error: str | None = None
        self._keras_model = None
        self._umsi_runner = None
        self._torch: Any | None = None

    def load(self) -> None:
        try:
            self._configure_device()
            if self.weights_path is not None:
                if not self.weights_path.exists():
                    raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
                self._load_hdf5_metadata()
                if self.normalize_name(self.name) == "umsi++":
                    self._try_load_umsi_tensorflow_model()
                else:
                    self._try_load_tensorflow_model()
            elif self.device.startswith("cuda"):
                self.backend = "cuda-heuristic"
            self.loaded = True
            logger.info("Model %s loaded with backend=%s device=%s path=%s", self.name, self.backend, self.device, self.weights_path)
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            logger.exception("Model failed to load")

    def _configure_device(self) -> None:
        try:
            import torch
        except ImportError:
            self._torch = None
            self.device = "cpu"
            return

        self._torch = torch
        self.cuda_available = bool(torch.cuda.is_available())
        if self.cuda_available:
            self.cuda_device_name = torch.cuda.get_device_name(self.cuda_device_index)
        if self.device_preference != "cuda" or not self.cuda_available:
            self.device = "cpu"
            return

        candidate = f"cuda:{self.cuda_device_index}"
        try:
            probe = torch.ones((1,), device=candidate)
            _ = float((probe + 1).sum().detach().cpu())
            self.device = candidate
            self.cuda_usable = True
        except Exception as exc:
            self.device = "cpu"
            self.cuda_usable = False
            self.load_error = f"CUDA detected but unusable, falling back to CPU: {exc}"
            logger.warning(self.load_error)

    def _load_hdf5_metadata(self) -> None:
        import h5py

        if self.weights_path is None:
            return
        with h5py.File(self.weights_path, "r") as weights:
            layer_names = weights.attrs.get("layer_names", [])
            keras_version = weights.attrs.get("keras_version")
            backend = weights.attrs.get("backend")
            self.layer_count = len(layer_names)
            self.keras_version = keras_version.decode() if isinstance(keras_version, bytes) else str(keras_version)
            if backend is not None:
                self.backend = backend.decode() if isinstance(backend, bytes) else str(backend)

    def _try_load_tensorflow_model(self) -> None:
        if self.runtime == "metadata_only":
            self.backend = "hdf5-metadata"
            return
        try:
            from tensorflow import keras
        except ImportError:
            self.backend = "hdf5-weights-cuda-heuristic" if self.device.startswith("cuda") else "hdf5-weights-heuristic"
            return

        try:
            self._keras_model = keras.models.load_model(self.weights_path, compile=False)
            self.backend = "tensorflow"
        except Exception as exc:
            self.backend = "hdf5-weights-cuda-heuristic" if self.device.startswith("cuda") else "hdf5-weights-heuristic"
            self.load_error = f"TensorFlow model load skipped: {exc}"
            logger.warning("TensorFlow could not load weights as a complete model: %s", exc)

    def _try_load_umsi_tensorflow_model(self) -> None:
        if self.runtime == "metadata_only":
            self.backend = "hdf5-metadata"
            return
        try:
            from app.services.umsi_tf import UMSITensorFlowRunner

            runner = UMSITensorFlowRunner(self.weights_path, self.cuda_device_index)
            runner.load()
            self._umsi_runner = runner
            self.backend = runner.backend
            self.device = runner.device
            self.load_error = None
        except Exception as exc:
            self.backend = "hdf5-weights-cuda-heuristic" if self.device.startswith("cuda") else "hdf5-weights-heuristic"
            self.load_error = f"UMSI++ TensorFlow model load failed, using fallback: {exc}"
            logger.exception("UMSI++ TensorFlow model could not be loaded")

    def predict(self, image: Image.Image) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        if self._umsi_runner is not None:
            return self._umsi_runner.predict(image)
        if self._keras_model is not None:
            prediction = self._predict_tensorflow(image)
            if prediction is not None:
                return prediction

        if self.device.startswith("cuda") and self._torch is not None:
            return self._predict_torch_heuristic(image)

        rgb = image.convert("RGB")
        gray = rgb.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
        arr = np.asarray(gray, dtype=np.float32) / 255.0

        y = np.linspace(-1.0, 1.0, arr.shape[0], dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, arr.shape[1], dtype=np.float32)[None, :]
        center_bias = np.exp(-1.8 * (x * x + y * y))
        contrast = np.abs(arr - arr.mean())
        saliency = (0.55 * contrast) + (0.35 * center_bias) + (0.10 * arr)
        saliency -= saliency.min()
        peak = float(saliency.max())
        if peak > 0:
            saliency /= peak
        return saliency

    def _predict_tensorflow(self, image: Image.Image) -> np.ndarray | None:
        if self._keras_model is None:
            return None

    def _predict_torch_heuristic(self, image: Image.Image) -> np.ndarray:
        if self._torch is None:
            raise RuntimeError("PyTorch is not available")

        torch = self._torch
        gray = image.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        height, width = tensor.shape
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=self.device).view(height, 1)
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=self.device).view(1, width)
        center_bias = torch.exp(-1.8 * (x * x + y * y))
        contrast = torch.abs(tensor - torch.mean(tensor))
        saliency = (0.55 * contrast) + (0.35 * center_bias) + (0.10 * tensor)
        saliency = saliency - torch.min(saliency)
        peak = torch.max(saliency)
        if float(peak.detach().cpu()) > 0:
            saliency = saliency / peak
        return saliency.detach().cpu().numpy()

    @staticmethod
    def normalize_name(name: str) -> str:
        return name.strip().lower()


class ModelRegistry:
    def __init__(
        self,
        *,
        saliency_model_path: Path,
        runtime: str,
        device_preference: str,
        cuda_device_index: int,
        allowed_model_names: list[str],
        default_model_name: str,
    ) -> None:
        self.default_model_name = self.normalize_name(default_model_name)
        self.allowed_model_names = {self.normalize_name(name) for name in allowed_model_names}
        self._models = {
            "umsi++": SaliencyModel(
                name="UMSI++",
                weights_path=saliency_model_path,
                runtime=runtime,
                device_preference=device_preference,
                cuda_device_index=cuda_device_index,
            ),
            "heuristic": SaliencyModel(
                name="Heuristic",
                weights_path=None,
                runtime="metadata_only",
                device_preference=device_preference,
                cuda_device_index=cuda_device_index,
            ),
        }

    @staticmethod
    def normalize_name(name: str) -> str:
        return SaliencyModel.normalize_name(name)

    def load(self) -> None:
        for name, model in self._models.items():
            if name in self.allowed_model_names:
                model.load()

    def get(self, name: str | None) -> SaliencyModel | None:
        normalized = self.normalize_name(name or self.default_model_name)
        if normalized not in self.allowed_model_names:
            return None
        return self._models.get(normalized)

    def snapshot(self) -> dict[str, Any]:
        return {
            name: {
                "display_name": model.name,
                "loaded": model.loaded,
                "device": model.device,
                "cuda_available": model.cuda_available,
                "cuda_usable": model.cuda_usable,
                "cuda_device_name": model.cuda_device_name,
                "backend": model.backend,
                "weights_path": str(model.weights_path) if model.weights_path else None,
                "keras_version": model.keras_version,
                "layer_count": model.layer_count,
                "error": model.load_error,
            }
            for name, model in self._models.items()
            if name in self.allowed_model_names
        }
        try:
            size = self._keras_model.input_shape[1:3]
            if not size or None in size:
                size = image.size[::-1]
            resized = image.resize((int(size[1]), int(size[0])), Image.Resampling.LANCZOS)
            arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
            output = self._keras_model.predict(arr[None, ...], verbose=0)
            saliency = np.squeeze(output).astype(np.float32)
            if saliency.ndim == 3:
                saliency = saliency[..., 0]
            saliency -= saliency.min()
            peak = float(saliency.max())
            if peak > 0:
                saliency /= peak
            return np.asarray(Image.fromarray((saliency * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        except Exception as exc:
            logger.exception("TensorFlow prediction failed, falling back to heuristic: %s", exc)
            return None
