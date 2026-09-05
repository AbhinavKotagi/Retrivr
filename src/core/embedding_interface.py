"""
embedding_interface.py — Abstract embedding model interface for Retrievr evaluation.

All embedding adapters (CLIP, OpenCLIP, etc.) must implement this interface so that
the evaluation pipeline is not hardcoded to a single model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class EmbeddingModel(ABC):
    """
    Abstract base class for image-text embedding models.

    All concrete adapters must implement:
      - encode_image()   — single image → (1, D) L2-normalised embedding
      - encode_images()  — batch images → (N, D) L2-normalised embeddings
      - encode_text()    — batch texts  → (N, D) L2-normalised embeddings

    The ``model_name`` and ``embedding_dim`` properties must also be available.
    """

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. 'openai/clip-vit-base-patch32')."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        ...

    # ── Encoding ──────────────────────────────────────────────────────────────

    @abstractmethod
    def encode_image(
        self,
        image: Union[Image.Image, Path, str],
    ) -> np.ndarray:
        """
        Encode a single image.

        Parameters
        ----------
        image:
            A PIL Image, file path, or string path.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim), float32, L2-normalised.
        """
        ...

    @abstractmethod
    def encode_images(
        self,
        images: list[Union[Image.Image, Path, str]],
    ) -> np.ndarray:
        """
        Encode a batch of images.

        Parameters
        ----------
        images:
            List of PIL Images or file paths.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), float32, L2-normalised.
        """
        ...

    @abstractmethod
    def encode_text(
        self,
        texts: Union[str, list[str]],
    ) -> np.ndarray:
        """
        Encode one or more text strings.

        Parameters
        ----------
        texts:
            A single string or a list of strings.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), float32, L2-normalised.
        """
        ...

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def l2_normalise(arr: np.ndarray) -> np.ndarray:
        """
        Row-wise L2 normalisation.

        Ensures ||embedding|| = 1.0 for every row, which is required
        for FAISS IndexFlatIP to behave as cosine similarity.
        """
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard zero-vectors
        return (arr / norms).astype(np.float32)

    def validate_normalisation(
        self,
        embeddings: np.ndarray,
        tolerance: float = 1e-4,
    ) -> dict:
        """
        Verify that all rows in *embeddings* are approximately unit-norm.

        Returns a dict with:
          - all_normalised: bool
          - min_norm: float
          - max_norm: float
          - mean_norm: float
        """
        norms = np.linalg.norm(embeddings, axis=1)
        all_ok = bool(np.all(np.abs(norms - 1.0) < tolerance))
        return {
            "all_normalised": all_ok,
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
            "mean_norm": float(norms.mean()),
            "tolerance": tolerance,
        }
