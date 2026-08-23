"""BLIP image captioning service."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from config import settings

logger = logging.getLogger(__name__)


class CaptioningService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.blip_model_name
        self._processor: BlipProcessor | None = None
        self._model: BlipForConditionalGeneration | None = None

    def _load_model(self) -> tuple[BlipProcessor, BlipForConditionalGeneration]:
        if self._processor is None or self._model is None:
            logger.info("Loading BLIP captioning model: %s", self.model_name)
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self._model.eval()
        return self._processor, self._model

    def generate_caption(self, image_path: Path) -> str:
        try:
            processor, model = self._load_model()
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=50)
            return processor.decode(output[0], skip_special_tokens=True).strip() or "An image"
        except Exception:
            logger.exception("Caption generation failed for %s", image_path)
            return "An image"


captioning_service = CaptioningService()
