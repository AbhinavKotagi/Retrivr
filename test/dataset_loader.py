"""
dataset_loader.py — Flickr8k dataset loading for the Retrievr evaluation module.

Loads:
  • Images from  <dataset_dir>/images/
  • Captions from <dataset_dir>/metadata.csv

Expected metadata.csv columns (space- or comma-separated):
  image_name  [tab or comma]  caption_number  caption

The Flickr8k "Flickr8k.token.txt" format uses:
    <image_filename>#<caption_index>\t<caption text>

This loader handles both that original format AND a plain CSV with columns:
    image_name, caption_number, caption
    OR
    image, caption
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """One evaluation query: a single human caption + its ground-truth image."""
    query: str
    image_filename: str              # e.g. "123456.jpg"
    relevant_images: set[str] = field(default_factory=set)  # extensible for multi-relevance

    def __post_init__(self) -> None:
        # Always include the primary image in the relevance set
        self.relevant_images.add(self.image_filename)


@dataclass
class DatasetInfo:
    """Container returned by load_dataset()."""
    image_dir: Path
    image_filenames: list[str]       # all unique image filenames found
    queries: list[QueryRecord]


def _abort(message: str) -> None:
    """Print a clear error and exit without a traceback."""
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def _check_dependencies() -> None:
    """Verify all required packages are importable."""
    missing = []
    for pkg in ("faiss", "torch", "transformers", "PIL", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _abort(
            f"Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install faiss-cpu torch transformers pillow numpy"
        )


def load_dataset(dataset_dir: Path) -> DatasetInfo:
    """
    Load the Flickr8k test-split dataset from *dataset_dir*.

    Returns a DatasetInfo with:
      • image_dir        — Path to the images folder
      • image_filenames  — list of unique image filenames discovered
      • queries          — list of QueryRecord (one per caption)
    """
    _check_dependencies()

    dataset_dir = dataset_dir.resolve()

    # ── Validate top-level directory ─────────────────────────────────────────────
    if not dataset_dir.exists():
        _abort(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please download Flickr8k and place it there, or use --dataset to specify the path.\n"
            f"Expected structure:\n"
            f"  {dataset_dir}/\n"
            f"    images/          ← JPEG images\n"
            f"    metadata.csv     ← captions file"
        )

    # ── Validate images sub-directory ────────────────────────────────────────────
    image_dir = dataset_dir / "images"
    if not image_dir.exists():
        if (dataset_dir / "Images").exists():
            image_dir = dataset_dir / "Images"
        else:
            _abort(
                f"Images directory not found: {image_dir} (or {dataset_dir / 'Images'})\n"
                f"Place Flickr8k JPEG images inside an 'images' or 'Images' folder."
            )

    image_files = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_files:
        _abort(f"No images found in: {image_dir}")

    logger.info("Found %d images in %s", len(image_files), image_dir)

    # ── Validate captions file ────────────────────────────────────────────────────
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        candidates = [
            dataset_dir / "captions.txt",
            dataset_dir / "captions.csv",
            dataset_dir / "Flickr8k.token.txt",
            dataset_dir / "metadata.txt",
        ]
        for cand in candidates:
            if cand.exists():
                metadata_path = cand
                break

    if not metadata_path.exists():
        _abort(
            f"Captions file not found in {dataset_dir} (looked for metadata.csv, captions.txt, etc.)\n"
            f"The file should contain image filenames and their human-written captions."
        )

    queries = _parse_metadata(metadata_path, set(image_files))
    if not queries:
        _abort(
            f"No valid captions parsed from: {metadata_path}\n"
            f"Check the file format."
        )

    logger.info("Loaded %d caption queries for %d images", len(queries), len(image_files))
    return DatasetInfo(
        image_dir=image_dir,
        image_filenames=image_files,
        queries=queries,
    )


def _parse_metadata(path: Path, valid_images: set[str]) -> list[QueryRecord]:
    """
    Parse the captions metadata file.

    Handles three common formats:
      1. Flickr8k token format  — "<filename>#<n>\\t<caption>"
      2. CSV: image_name, caption_number, caption
      3. CSV: image, caption
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Detect Flickr8k token format ("filename.jpg#0\tcaption text")
    first_line = text.splitlines()[0] if text.strip() else ""
    if "#" in first_line and "\t" in first_line:
        logger.debug("Detected Flickr8k token format")
        return _parse_token_format(text, valid_images)

    # Fall through to CSV parsing
    logger.debug("Attempting CSV parsing")
    return _parse_csv_format(text, valid_images)


def _parse_token_format(text: str, valid_images: set[str]) -> list[QueryRecord]:
    """Parse Flickr8k original token file: '<image>#<n>\\t<caption>'."""
    queries: list[QueryRecord] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            image_part, caption = line.split("\t", 1)
            image_filename = image_part.split("#")[0]
            caption = caption.strip()
        except ValueError:
            skipped += 1
            continue
        if not caption:
            skipped += 1
            continue
        if image_filename not in valid_images:
            # Caption refers to an image not present on disk — skip silently
            continue
        queries.append(QueryRecord(query=caption, image_filename=image_filename))
    if skipped:
        logger.debug("Skipped %d malformed lines in token file", skipped)
    return queries


def _parse_csv_format(text: str, valid_images: set[str]) -> list[QueryRecord]:
    """Parse CSV captions file with headers."""
    queries: list[QueryRecord] = []
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return []

    # Normalize header names (strip whitespace, lower)
    headers = [h.strip().lower() for h in reader.fieldnames]

    # Determine which columns map to image and caption
    image_col: str | None = None
    caption_col: str | None = None

    for h in headers:
        if h in ("image_name", "image", "filename", "file"):
            image_col = reader.fieldnames[headers.index(h)]
        if h in ("caption", "comment", "description", "text"):
            caption_col = reader.fieldnames[headers.index(h)]

    if image_col is None or caption_col is None:
        logger.warning(
            "Could not identify image/caption columns. Available: %s", reader.fieldnames
        )
        return []

    for row in reader:
        image_filename = (row.get(image_col) or "").strip()
        caption = (row.get(caption_col) or "").strip()
        if not image_filename or not caption:
            continue
        if image_filename not in valid_images:
            continue
        queries.append(QueryRecord(query=caption, image_filename=image_filename))

    return queries
