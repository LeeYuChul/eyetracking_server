from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "eyetrack-analysis-server"
    host: str = "0.0.0.0"
    port: int = 3781
    base_url: str = "https://eyetrack.newlearn.ai.kr"
    model_weights_dir: Path = Path("model/model_weights")
    saliency_model_path: Path = Path("model/model_weights/saliency_models/UMSI++/umsi++.hdf5")
    default_model_name: str = "umsi++"
    allowed_model_names: list[str] = ["umsi++", "heuristic"]
    model_runtime: str = "auto"
    device: str = "cuda"
    cuda_device_index: int = 0
    external_api_base_url: str | None = None
    external_api_key: str | None = None
    max_upload_bytes: int = 10 * 1024 * 1024
    overlay_alpha: float = Field(default=0.45, ge=0.0, le=1.0)
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
