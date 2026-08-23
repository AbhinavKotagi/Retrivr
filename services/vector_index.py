"""Incremental FAISS index service."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class VectorIndexService:
    """Owns FAISS persistence and image-id mapping."""

    def __init__(self, index_path: Path | None = None, id_map_path: Path | None = None) -> None:
        self.index_path = index_path or settings.faiss_index_path
        self.id_map_path = id_map_path or settings.id_map_path
        self.embedding_dim = settings.embedding_dim

    def exists(self) -> bool:
        return self.index_path.exists() and self.id_map_path.exists()

    def load_or_create_index(self) -> faiss.IndexFlatIP:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.embedding_dim)

    def load_id_map(self) -> list[str]:
        if not self.id_map_path.exists():
            return []
        with self.id_map_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, index: faiss.IndexFlatIP, id_map: list[str]) -> None:
        settings.ensure_directories()
        faiss.write_index(index, str(self.index_path))
        with self.id_map_path.open("w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2)

    def add_embedding(self, image_id: str, embedding: np.ndarray) -> bool:
        """Append a vector only if image_id is not already indexed."""
        index = self.load_or_create_index()
        id_map = self.load_id_map()
        if image_id in id_map:
            logger.info("Skipping already-indexed image_id=%s", image_id)
            return False
        index.add(embedding.astype(np.float32).reshape(1, -1))
        id_map.append(image_id)
        self.save(index, id_map)
        logger.info("Indexed image_id=%s at position=%s", image_id, len(id_map) - 1)
        return True

    def search(self, embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        index = self.load_or_create_index() if self.index_path.exists() else None
        id_map = self.load_id_map()
        if index is None or index.ntotal == 0 or not id_map:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        return index.search(embedding.astype(np.float32).reshape(1, -1), min(k, index.ntotal))

    def reset(self) -> None:
        empty_index = faiss.IndexFlatIP(self.embedding_dim)
        self.save(empty_index, [])


vector_index_service = VectorIndexService()
