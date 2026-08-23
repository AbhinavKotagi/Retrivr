"""CLIP text embedding service."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.clip_model_name
        self._tokenizer: CLIPTokenizer | None = None
        self._model: CLIPTextModel | None = None

    def _load_model(self) -> tuple[CLIPTokenizer, CLIPTextModel]:
        if self._tokenizer is None or self._model is None:
            logger.info("Loading CLIP text model: %s", self.model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self._model = CLIPTextModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def embed_text(self, text: str) -> np.ndarray:
        tokenizer, model = self._load_model()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


embedding_service = EmbeddingService()
