"""Backward-compatible semantic search API backed by SearchService."""

from __future__ import annotations

from config import settings
from services.search_service import SearchResult, search_service

FAISS_INDEX_PATH = settings.faiss_index_path
ID_MAP_PATH = settings.id_map_path


def index_exists() -> bool:
    return search_service.index_exists()


def search_top_caption(query: str) -> SearchResult | None:
    return search_service.search_top_caption(query)


def search_images(query: str, k: int = 5) -> list[SearchResult]:
    return search_service.search_images(query, k)
