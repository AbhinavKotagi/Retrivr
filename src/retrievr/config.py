"""Application configuration for Retrievr."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables with safe defaults."""

    app_name: str = "Retrievr"
    database_path: Path = _path_from_env("RETRIVR_DB_PATH", "retrievr.db")
    storage_dir: Path = _path_from_env("RETRIVR_STORAGE_DIR", "storage/images")
    vectors_dir: Path = _path_from_env("RETRIVR_VECTORS_DIR", "vectors")
    faiss_index_name: str = os.getenv("RETRIVR_FAISS_INDEX_NAME", "faiss.index")
    id_map_name: str = os.getenv("RETRIVR_ID_MAP_NAME", "id_map.json")
    embedding_dim: int = int(os.getenv("RETRIVR_EMBEDDING_DIM", "512"))
    clip_model_name: str = os.getenv("RETRIVR_CLIP_MODEL", "openai/clip-vit-base-patch32")
    blip_model_name: str = os.getenv("RETRIVR_BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    log_level: str = os.getenv("RETRIVR_LOG_LEVEL", "INFO")

    @property
    def faiss_index_path(self) -> Path:
        return self.vectors_dir / self.faiss_index_name

    @property
    def id_map_path(self) -> Path:
        return self.vectors_dir / self.id_map_name

    def ensure_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
