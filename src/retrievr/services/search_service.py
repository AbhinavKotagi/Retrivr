"""Semantic search service over FAISS and SQLite metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from retrievr.services.embedding_service import embedding_service
from retrievr.services.metadata import metadata_store
from retrievr.services.vector_index import vector_index_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    image_id: str
    file_path: str
    caption: str
    score: float


class SearchService:
    def index_exists(self) -> bool:
        return vector_index_service.exists()

    def search_top_caption(self, query: str) -> SearchResult | None:
        hits = self.search_images(query, k=1)
        return hits[0] if hits else None

    def search_images(self, query: str, k: int = 5) -> list[SearchResult]:
        id_map = vector_index_service.load_id_map()
        query_embedding = embedding_service.embed_text(query)
        distances, indices = vector_index_service.search(query_embedding, k)
        if distances.size == 0 or indices.size == 0:
            return []
        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist <= 0 or idx >= len(id_map):
                continue
            image_id = id_map[int(idx)]
            row = metadata_store.get_image_by_id(image_id)
            if row is None:
                logger.warning("FAISS result missing metadata for image_id=%s", image_id)
                continue
            results.append(SearchResult(image_id, row["file_path"], row["caption"] or "", float(dist)))
        return sorted(results, key=lambda r: r.score, reverse=True)


search_service = SearchService()
