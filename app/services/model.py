import logging
import gc
import threading
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
            if self.weights_path is None or self.runtime == "metadata_only":
                self._configure_cpu_only()
            elif self.normalize_name(self.name) == "umsi++":
                self._configure_subprocess_device()
            else:
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

    def _configure_cpu_only(self) -> None:
        self._torch = None
        self.device = "cpu"
        self.cuda_available = False
        self.cuda_usable = False
        self.cuda_device_name = None

    def _configure_subprocess_device(self) -> None:
        self._torch = None
        if self.device_preference == "cuda":
            self.device = f"cuda:{self.cuda_device_index}"
            self.cuda_available = True
            self.cuda_usable = True
            self.cuda_device_name = None
            return
        self.device = "cpu"
        self.cuda_available = False
        self.cuda_usable = False
        self.cuda_device_name = None

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

    def predict_many(self, images: list[Image.Image]) -> list[np.ndarray]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        if self._umsi_runner is not None:
            return self._umsi_runner.predict_many(images)
        return [self.predict(image) for image in images]

    def _predict_tensorflow(self, image: Image.Image) -> np.ndarray | None:
        if self._keras_model is None:
            return None
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
            return (
                np.asarray(
                    Image.fromarray((saliency * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
        except Exception as exc:
            logger.exception("TensorFlow prediction failed, falling back to heuristic: %s", exc)
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

    def unload(self) -> None:
        used_tensorflow = self._keras_model is not None or bool(self._umsi_runner is not None and self._umsi_runner.model is not None)
        if self._umsi_runner is not None:
            self._umsi_runner.unload()
            self._umsi_runner = None
        self._keras_model = None
        if used_tensorflow:
            try:
                from tensorflow.keras import backend as keras_backend

                keras_backend.clear_session()
            except Exception:
                logger.debug("TensorFlow/Keras cleanup skipped", exc_info=True)
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.ipc_collect()
            except Exception:
                logger.debug("PyTorch CUDA cleanup skipped", exc_info=True)
        self._torch = None
        self.loaded = False
        self.device = "cpu"
        self.cuda_usable = False
        gc.collect()

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
        idle_unload_seconds: float = 60.0,
    ) -> None:
        self.default_model_name = self.normalize_name(default_model_name)
        self.allowed_model_names = {self.normalize_name(name) for name in allowed_model_names}
        self._usage_lock = threading.Lock()
        self._active_requests = 0
        self._idle_unload_seconds = max(0.0, idle_unload_seconds)
        self._unload_timer: threading.Timer | None = None
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

    def begin_request_model(self, name: str | None) -> SaliencyModel | None:
        with self._usage_lock:
            self._cancel_idle_unload_locked()
            model = self.get(name)
            if model is None:
                return None
            if not model.loaded:
                model.load()
            self._active_requests += 1
            return model

    def end_request(self) -> None:
        models_to_unload: list[SaliencyModel] = []
        with self._usage_lock:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                if self._idle_unload_seconds <= 0:
                    self._cancel_idle_unload_locked()
                    models_to_unload = [model for model in self._models.values() if model.loaded]
                else:
                    self._schedule_idle_unload_locked()
        for model in models_to_unload:
            model.unload()

    def unload_all(self) -> None:
        self._cancel_idle_unload()
        for model in self._models.values():
            if model.loaded:
                model.unload()

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if self._idle_unload_seconds <= 0:
            return
        self._unload_timer = threading.Timer(self._idle_unload_seconds, self._unload_if_idle)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _cancel_idle_unload(self) -> None:
        with self._usage_lock:
            self._cancel_idle_unload_locked()

    def _cancel_idle_unload_locked(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _unload_if_idle(self) -> None:
        with self._usage_lock:
            self._unload_timer = None
            if self._active_requests != 0:
                return
            models = [model for model in self._models.values() if model.loaded]
            for model in models:
                model.unload()

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
