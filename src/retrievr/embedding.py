"""Backward-compatible CLIP embedding helper."""

from __future__ import annotations

from retrievr.services.embedding_service import embedding_service
from retrievr.config import settings

MODEL_NAME = settings.clip_model_name


def get_text_embedding(text: str):
    return embedding_service.embed_text(text)
