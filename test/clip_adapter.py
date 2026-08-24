"""
clip_adapter.py — Independent CLIP model adapter for the Retrievr evaluation module.

Provides:
  • encode_images(images)  → np.ndarray  shape (N, D)  L2-normalised
  • encode_text(texts)     → np.ndarray  shape (N, D)  L2-normalised

Uses the full CLIPModel (vision + text) from HuggingFace transformers.
Automatically selects CUDA → MPS → CPU.

The model name defaults to the same model used by the production Retrievr app
("openai/clip-vit-base-patch32") but can be overridden via EVAL_CLIP_MODEL
or the --model CLI flag.

NOTE: The production app only uses CLIPTextModel.  This adapter additionally
loads the vision tower so that images can be embedded during evaluation.
No production files are modified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


class CLIPAdapter:
    """
    Thin wrapper around HuggingFace CLIPModel for the evaluation pipeline.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. "openai/clip-vit-base-patch32".
    device:
        torch.device to use.  Auto-detected if None.
    batch_size:
        Number of images/texts to encode per forward pass.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: torch.device | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device if device is not None else _select_device()
        self.batch_size = batch_size

        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    # ── Lazy model loading ────────────────────────────────────────────────────────
    def _load(self) -> tuple[CLIPModel, CLIPProcessor]:
        if self._model is None or self._processor is None:
            logger.info("Loading CLIP model: %s", self.model_name)
            try:
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._model.eval()
                self._model.to(self.device)
                logger.info("CLIP model loaded on %s", self.device)
            except Exception as exc:
                import sys
                print(
                    f"\n[ERROR] Could not load CLIP model '{self.model_name}': {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        return self._model, self._processor

    # ── Normalisation utility ─────────────────────────────────────────────────────
    @staticmethod
    def _l2_normalise(arr: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation."""
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return (arr / norms).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────────
    def encode_images(
        self,
        images: list[Union[Image.Image, Path, str]],
    ) -> np.ndarray:
        """
        Encode a list of PIL Images (or paths) into L2-normalised CLIP image embeddings.

        Parameters
        ----------
        images:
            List of PIL.Image.Image objects or file paths.

        Returns
        -------
        np.ndarray of shape (len(images), embedding_dim), float32, L2-normalised.
        """
        model, processor = self._load()
        all_embeddings: list[np.ndarray] = []

        for batch_start in range(0, len(images), self.batch_size):
            batch = images[batch_start : batch_start + self.batch_size]

            # Convert paths to PIL images
            pil_batch: list[Image.Image] = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    try:
                        pil_batch.append(Image.open(img).convert("RGB"))
                    except Exception as exc:
                        logger.warning("Could not open image %s: %s", img, exc)
                        continue
                else:
                    pil_batch.append(img.convert("RGB") if img.mode != "RGB" else img)

            if not pil_batch:
                continue

            inputs = processor(images=pil_batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_image_features(**inputs)

            # transformers 5.x may return a model-output object instead of a raw tensor
            if hasattr(features, 'pooler_output') and features.pooler_output is not None:
                feat_tensor = features.pooler_output
            elif hasattr(features, 'last_hidden_state'):
                # Mean pool if only hidden states are available
                feat_tensor = features.last_hidden_state.mean(dim=1)
            elif isinstance(features, torch.Tensor):
                feat_tensor = features
            else:
                # Fallback: try to get the first tensor attribute
                feat_tensor = next(v for v in vars(features).values() if isinstance(v, torch.Tensor))

            batch_embeddings = feat_tensor.cpu().numpy()
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)

        combined = np.vstack(all_embeddings)
        return self._l2_normalise(combined)

    def encode_text(self, texts: list[str] | str) -> np.ndarray:
        """
        Encode one or more text strings into L2-normalised CLIP text embeddings.

        Parameters
        ----------
        texts:
            A single string or a list of strings.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), float32, L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        model, processor = self._load()
        all_embeddings: list[np.ndarray] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_text_features(**inputs)

            # transformers 5.x may return a model-output object instead of a raw tensor
            if hasattr(features, 'pooler_output') and features.pooler_output is not None:
                feat_tensor = features.pooler_output
            elif hasattr(features, 'last_hidden_state'):
                feat_tensor = features.last_hidden_state.mean(dim=1)
            elif isinstance(features, torch.Tensor):
                feat_tensor = features
            else:
                feat_tensor = next(v for v in vars(features).values() if isinstance(v, torch.Tensor))

            batch_embeddings = feat_tensor.cpu().numpy()
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)

        combined = np.vstack(all_embeddings)
        return self._l2_normalise(combined)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        model, _ = self._load()
        return model.config.projection_dim
