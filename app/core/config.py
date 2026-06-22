from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "eyetrack-analysis-server"
    host: str = "0.0.0.0"
    port: int = 3781
    base_url: str = "https://eyetrack.newlearn.ai.kr"
    storage_policy: str = "stateless_response_only"
    model_weights_dir: Path = Path("model/model_weights")
    saliency_model_path: Path = Path("model/model_weights/saliency_models/UMSI++/umsi++.hdf5")
    default_model_name: str = "umsi++"
    allowed_model_names: list[str] = ["umsi++", "heuristic"]
    model_runtime: str = "auto"
    device: str = "cuda"
    cuda_device_index: int = 0
    external_api_base_url: str | None = None
    external_api_key: str | None = None
    max_frames: int = 15
    max_upload_bytes: int = 10 * 1024 * 1024
    max_total_upload_bytes: int = 100 * 1024 * 1024
    rate_limit_per_minute: int = 60
    model_idle_unload_seconds: float = 0.0
    overlay_alpha: float = Field(default=0.45, ge=0.0, le=1.0)
    vlm_provider: str = "ollama"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "gemma4:26b"
    ollama_keep_alive: str = "0"
    ollama_num_ctx: int = 32768
    ollama_num_predict: int = 768
    vlm_request_timeout_seconds: float = 600.0
    vlm_image_chunk_size: int = 5
    vlm_max_image_side: int = 768
    vlm_image_jpeg_quality: int = 82
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    cors_origins: list[str] = [
        "https://www.figma.com",
        "https://figma.com",
        "null",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EYETRACK_",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
