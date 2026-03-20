"""
search.py
---------
Encapsulates all FAISS search logic for Retrievr.

Keeping search logic here (rather than in app.py) means:
  - app.py stays purely presentational
  - search can be unit-tested independently
  - Part 4 additions (e.g. re-ranking, filters) have a clear home

Public API
----------
  search_images(query, k) -> list[SearchResult]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from database import get_image_by_id
from embedding import get_text_embedding

# ---------------------------------------------------------------------------
# Paths (must match processor.py)
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = Path("vectors/faiss.index")
ID_MAP_PATH = Path("vectors/id_map.json")

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Holds everything the UI needs to render one search hit."""
    image_id:   str
    file_path:  str
    caption:    str
    score:      float   # cosine similarity in [0, 1]; higher = more similar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_index() -> faiss.IndexFlatIP | None:
    """
    Load the FAISS index from disk.

    Returns:
        The loaded index, or None if the file does not exist yet.
    """
    if not FAISS_INDEX_PATH.exists():
        return None
    return faiss.read_index(str(FAISS_INDEX_PATH))


def _load_id_map() -> list[str]:
    """
    Load the FAISS-position → image_id mapping.

    Returns:
        A list of image_id strings, or [] if the file does not exist.
    """
    if not ID_MAP_PATH.exists():
        return []
    with open(ID_MAP_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_exists() -> bool:
    """Return True if a FAISS index has been built and saved to disk."""
    return FAISS_INDEX_PATH.exists() and ID_MAP_PATH.exists()


def search_images(query: str, k: int = 5) -> list[SearchResult]:
    """
    Semantic image search using a natural language query.

    Pipeline:
      1. Embed the query with CLIP (already L2-normalized by get_text_embedding).
      2. Run IndexFlatIP.search — inner product on unit vectors == cosine sim.
      3. Map FAISS integer positions back to image_ids via id_map.json.
      4. Fetch file_path + caption from SQLite for each hit.
      5. Skip any hits whose score <= 0 (FAISS returns -1 for empty slots).

    Args:
        query:  The user's natural language search string.
        k:      Maximum number of results to return (default 5).

    Returns:
        A list of SearchResult objects, sorted by descending similarity score.
        May contain fewer than k items if the index has fewer vectors or
        if some hits have a zero/negative score.
    """
    index = _load_index()
    id_map = _load_id_map()

    if index is None or index.ntotal == 0 or not id_map:
        return []

    # Clamp k to the actual number of indexed vectors to avoid FAISS errors
    k_actual = min(k, index.ntotal)

    # Embed + reshape to (1, 512) — FAISS expects a 2-D float32 array
    query_embedding = get_text_embedding(query)          # (512,) — already normalized
    query_2d = query_embedding.reshape(1, -1)            # (1, 512)

    # Similarity search — distances are cosine similarities in [-1, 1]
    distances, indices = index.search(query_2d, k_actual)

    results: list[SearchResult] = []

    for dist, idx in zip(distances[0], indices[0]):
        # FAISS fills unused slots with index=-1 and distance=-1
        if idx == -1 or dist <= 0:
            continue

        # Guard against id_map being shorter than the index (shouldn't happen,
        # but defensive programming costs nothing)
        if idx >= len(id_map):
            continue

        image_id = id_map[idx]
        row = get_image_by_id(image_id)

        if row is None:
            # DB record missing — index and DB are out of sync; skip silently
            continue

        results.append(
            SearchResult(
                image_id=image_id,
                file_path=row["file_path"],
                caption=row["caption"] or "",
                score=float(dist),
            )
        )

    # Already sorted by FAISS (highest score first), but sort again defensively
    results.sort(key=lambda r: r.score, reverse=True)
    return results
