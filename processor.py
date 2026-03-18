"""
processor.py
------------
Orchestrates the full pipeline for a single image:
  1. Save the raw file to storage/images/
  2. Generate a caption via BLIP (with "An image" fallback)
  3. Embed the caption with CLIP via embedding.py
  4. Persist metadata to SQLite via database.py
  5. Add the embedding to the FAISS index and save it to disk

FAISS index
-----------
  Type  : IndexFlatIP  (inner product — equivalent to cosine similarity
                        because embeddings are L2-normalized)
  Dim   : 512          (CLIP clip-vit-base-patch32 text encoder output)
  Saved : vectors/faiss.index
  Map   : vectors/id_map.json  (FAISS int position -> image_id string)
"""

import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from database import insert_image, get_image_by_id
from embedding import get_text_embedding
from utils import generate_image_id

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STORAGE_DIR = Path("storage/images")
VECTORS_DIR = Path("vectors")
FAISS_INDEX_PATH = VECTORS_DIR / "faiss.index"
ID_MAP_PATH = VECTORS_DIR / "id_map.json"

# CLIP embedding dimension (clip-vit-base-patch32 text encoder)
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# BLIP model — lazy-loaded and cached at module level
# ---------------------------------------------------------------------------

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

_blip_processor: BlipProcessor | None = None
_blip_model: BlipForConditionalGeneration | None = None


def _load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Lazily load and cache the BLIP captioning model."""
    global _blip_processor, _blip_model

    if _blip_processor is None or _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
        _blip_model.eval()

    return _blip_processor, _blip_model


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def _load_or_create_index() -> faiss.IndexFlatIP:
    """
    Load the FAISS index from disk if it exists, otherwise create a new one.

    Returns:
        A faiss.IndexFlatIP instance ready for add() / search().
    """
    if FAISS_INDEX_PATH.exists():
        return faiss.read_index(str(FAISS_INDEX_PATH))

    # First-time initialisation
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _load_id_map() -> list[str]:
    """
    Load the list that maps FAISS integer positions to image_id strings.

    Returns:
        A list of image_id strings (index == FAISS row position).
    """
    if ID_MAP_PATH.exists():
        with open(ID_MAP_PATH, "r") as f:
            return json.load(f)
    return []


def _save_index(index: faiss.IndexFlatIP, id_map: list[str]) -> None:
    """Persist both the FAISS index and the id→position map to disk."""
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def generate_caption(image_path: Path) -> str:
    """
    Generate a natural-language caption for an image using BLIP.

    Falls back to "An image" if the model fails for any reason.

    Args:
        image_path: Path to the saved image file.

    Returns:
        A caption string.
    """
    try:
        blip_processor, blip_model = _load_blip()
        image = Image.open(image_path).convert("RGB")

        inputs = blip_processor(images=image, return_tensors="pt")
        output = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip() or "An image"

    except Exception:
        # Graceful degradation — never crash the upload flow
        return "An image"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_already_processed(image_id: str) -> bool:
    """Return True if this image_id already exists in the database."""
    return get_image_by_id(image_id) is not None


def save_uploaded_file(uploaded_file) -> tuple[str, Path]:
    """
    Write a Streamlit UploadedFile to storage/images/ with a UUID filename.

    Args:
        uploaded_file: The object returned by st.file_uploader.

    Returns:
        (image_id, file_path) — the UUID and the resolved Path on disk.
    """
    suffix = Path(uploaded_file.name).suffix.lower()  # e.g. ".jpg"
    image_id = generate_image_id()
    dest = STORAGE_DIR / f"{image_id}{suffix}"
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return image_id, dest


def process_image(image_id: str, file_path: Path) -> str:
    """
    Run the full processing pipeline for one image:
      1. Generate caption (BLIP).
      2. Embed caption (CLIP).
      3. Store metadata in SQLite.
      4. Add embedding to FAISS index and persist to disk.

    Args:
        image_id:  UUID string for this image.
        file_path: Path to the image file on disk.

    Returns:
        The generated caption string.
    """
    # --- Step 1: Caption ---
    caption = generate_caption(file_path)

    # --- Step 2: Embedding ---
    embedding = get_text_embedding(caption)          # shape: (512,)
    embedding_2d = embedding.reshape(1, -1)          # FAISS expects (n, dim)

    # --- Step 3: Database ---
    insert_image(image_id, str(file_path), caption)

    # --- Step 4: FAISS ---
    index = _load_or_create_index()
    id_map = _load_id_map()

    index.add(embedding_2d)   # appended at position len(id_map)
    id_map.append(image_id)

    _save_index(index, id_map)

    return caption