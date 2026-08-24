"""
config.py — Evaluation module configuration for Retrievr.

All paths are relative to the project root, resolved at runtime.
Override via CLI flags or environment variables where noted.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root: two levels up from this file (test/config.py → project root) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Dataset ─────────────────────────────────────────────────────────────────────
# Check known locations: storage/Flickr 8k or test/dataset/flickr8k
_STORAGE_FLICKR = PROJECT_ROOT / "storage" / "Flickr 8k"
_TEST_FLICKR = PROJECT_ROOT / "test" / "dataset" / "flickr8k"

if _STORAGE_FLICKR.exists():
    DEFAULT_DATASET_DIR: Path = _STORAGE_FLICKR
else:
    DEFAULT_DATASET_DIR: Path = _TEST_FLICKR

IMAGES_SUBDIR: str = "images"
METADATA_FILENAME: str = "metadata.csv"

# ── CLIP model ───────────────────────────────────────────────────────────────────
# Mirrors the production setting in src/retrievr/config.py
CLIP_MODEL_NAME: str = os.getenv("EVAL_CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM: int = 512

# ── Evaluation defaults ──────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 20          # retrieve up to top-20 for computing all R@K metrics
DEFAULT_LIMIT: int | None = None  # None → full dataset
DEFAULT_SEED: int = 42

# Recall / Precision K-values to report
RECALL_K_VALUES: list[int] = [1, 5, 10, 20]
PRECISION_K_VALUES: list[int] = [1, 5, 10, 20]
MAP_K: int = 10
NDCG_K: int = 10

# Relevance score for the ground-truth image (supports graded relevance in future)
RELEVANCE_HIGHLY: int = 3
RELEVANCE_NONE: int = 0

# ── Results output ───────────────────────────────────────────────────────────────
RESULTS_DIR: Path = PROJECT_ROOT / "test" / "results"
RESULTS_JSON: str = "evaluation_results.json"
