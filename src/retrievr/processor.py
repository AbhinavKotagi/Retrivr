"""Backward-compatible image processing API backed by modular services."""

from __future__ import annotations

from pathlib import Path

from retrievr.config import settings
from retrievr.services.captioning import captioning_service
from retrievr.services.image_processing import image_processing_service

STORAGE_DIR = settings.storage_dir
VECTORS_DIR = settings.vectors_dir
FAISS_INDEX_PATH = settings.faiss_index_path
ID_MAP_PATH = settings.id_map_path
EMBEDDING_DIM = settings.embedding_dim
BLIP_MODEL_NAME = settings.blip_model_name


def generate_caption(image_path: Path) -> str:
    return captioning_service.generate_caption(image_path)


def is_already_processed(image_id: str) -> bool:
    return image_processing_service.is_already_processed(image_id)


def save_uploaded_file(uploaded_file) -> tuple[str, Path]:
    return image_processing_service.save_uploaded_file(uploaded_file)


def process_image(image_id: str, file_path: Path) -> str:
    return image_processing_service.process_image(image_id, file_path)
