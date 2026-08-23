"""Image ingestion orchestration service."""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from services.captioning import captioning_service
from services.embedding_service import embedding_service
from services.metadata import metadata_store
from services.vector_index import vector_index_service
from utils import generate_image_id

logger = logging.getLogger(__name__)


class ImageProcessingService:
    def save_uploaded_file(self, uploaded_file) -> tuple[str, Path]:
        suffix = Path(uploaded_file.name).suffix.lower()
        image_id = generate_image_id()
        settings.ensure_directories()
        dest = settings.storage_dir / f"{image_id}{suffix}"
        with dest.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info("Saved upload %s as %s", uploaded_file.name, dest)
        return image_id, dest

    def process_image(self, image_id: str, file_path: Path) -> str:
        if self.is_already_processed(image_id):
            row = metadata_store.get_image_by_id(image_id)
            return (row["caption"] if row else None) or "An image"
        caption = captioning_service.generate_caption(file_path)
        embedding = embedding_service.embed_text(caption)
        metadata_store.upsert_image(image_id, str(file_path), caption, indexed=False)
        vector_index_service.add_embedding(image_id, embedding)
        metadata_store.mark_indexed(image_id)
        return caption

    def is_already_processed(self, image_id: str) -> bool:
        row = metadata_store.get_image_by_id(image_id)
        return row is not None and bool(row["indexed"])


image_processing_service = ImageProcessingService()
