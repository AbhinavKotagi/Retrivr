"""
evaluation.py — Main entry point for the Retrievr evaluation module.

Usage:
    python test/evaluation.py                          # full benchmark
    python test/evaluation.py --dataset test/dataset/flickr8k
    python test/evaluation.py --limit 100             # quick dev run
    python test/evaluation.py --limit 100 --top-k 20
    python test/evaluation.py --model openai/clip-vit-base-patch32

This script:
  1. Loads the Flickr8k test-split images and captions.
  2. Generates CLIP image embeddings for all test images.
  3. Builds a TEMPORARY in-memory FAISS index (does NOT touch the production index).
  4. For each human-written caption, generates a CLIP text embedding and runs FAISS search.
  5. Calculates all retrieval metrics from actual results.
  6. Prints a formatted benchmark report.
  7. Saves results to test/results/evaluation_results.json.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
import time

# Suppress duplicate OpenMP runtime warning common in Anaconda environments
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

# ── Make sure imports work when running from project root ─────────────────────────
# e.g.  python test/evaluation.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# Local evaluation modules
from test.config import (
    CLIP_MODEL_NAME,
    DEFAULT_DATASET_DIR,
    DEFAULT_LIMIT,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    MAP_K,
    NDCG_K,
    PRECISION_K_VALUES,
    RECALL_K_VALUES,
    RELEVANCE_HIGHLY,
    RESULTS_DIR,
    RESULTS_JSON,
)
from test.clip_adapter import CLIPAdapter
from test.dataset_loader import DatasetInfo, QueryRecord, load_dataset
from test.metrics import (
    QueryResult,
    latency_stats,
    mean_average_precision,
    mean_ndcg,
    mean_precision_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
)

# ── Logging ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrievr.eval")


# ─────────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrievr — Semantic Image Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test/evaluation.py\n"
            "  python test/evaluation.py --limit 100\n"
            "  python test/evaluation.py --dataset test/dataset/flickr8k --top-k 20\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            f"Path to the Flickr8k dataset directory. "
            f"Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help="Evaluate only the first N images (and their captions). Default: all.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Number of images to retrieve per query. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CLIP_MODEL_NAME,
        help=f"CLIP model name. Default: {CLIP_MODEL_NAME}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Batch size for CLIP encoding. Default: 64.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save results to test/results/evaluation_results.json (default: True).",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Disable saving results JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────────
# Seed setting
# ─────────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────────
# FAISS index builder (temporary, in-memory)
# ─────────────────────────────────────────────────────────────────────────────────

def _build_faiss_index(
    embeddings: np.ndarray,
    dim: int,
) -> "faiss.IndexFlatIP":  # type: ignore[name-defined]
    """
    Build an in-memory FAISS IndexFlatIP from pre-computed, L2-normalised embeddings.
    Cosine similarity is equivalent to inner product for unit vectors.

    This index is TEMPORARY and completely independent of the production FAISS index.
    """
    try:
        import faiss
    except ImportError:
        print(
            "\n[ERROR] faiss is not installed.\n"
            "Install with: pip install faiss-cpu",
            file=sys.stderr,
        )
        sys.exit(1)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


# ─────────────────────────────────────────────────────────────────────────────────
# Query loop
# ─────────────────────────────────────────────────────────────────────────────────

def _run_queries(
    queries: list[QueryRecord],
    index: "faiss.IndexFlatIP",  # type: ignore[name-defined]
    id_map: list[str],
    clip: CLIPAdapter,
    top_k: int,
) -> list[QueryResult]:
    """
    For each query:
      1. Encode the caption text with CLIP  ← latency starts here
      2. Search FAISS for top_k results     ← latency ends here
      3. Wrap result in QueryResult

    Latency does NOT include model loading or image indexing.
    """
    results: list[QueryResult] = []
    n = len(queries)

    print(f"\nRunning {n} queries against {len(id_map)} indexed images ...")

    for i, qr in enumerate(queries):
        if (i + 1) % 500 == 0 or i == 0:
            logger.info("  Query %d / %d", i + 1, n)

        # ── Measure only encode + search latency ──────────────────────────────────
        t0 = time.perf_counter()
        text_emb = clip.encode_text([qr.query])           # shape (1, D)
        distances, indices = index.search(                 # type: ignore[attr-defined]
            text_emb.astype(np.float32), top_k
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Map FAISS indices → image filenames
        retrieved: list[str] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(id_map):
                continue
            retrieved.append(id_map[int(idx)])

        # Build relevance score map for nDCG
        # ground-truth image → RELEVANCE_HIGHLY (3), everything else → 0
        relevance_scores: dict[str, int] = {
            img: RELEVANCE_HIGHLY for img in qr.relevant_images
        }

        results.append(
            QueryResult(
                query=qr.query,
                relevant=qr.relevant_images,
                retrieved=retrieved,
                relevance_scores=relevance_scores,
                latency_ms=latency_ms,
            )
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────────
# Results output
# ─────────────────────────────────────────────────────────────────────────────────

def _compute_all_metrics(
    results: list[QueryResult],
    top_k: int,
) -> dict:
    """Compute and return a structured dict of all evaluation metrics."""
    recall = {k: mean_recall_at_k(results, k) for k in RECALL_K_VALUES}
    precision = {k: mean_precision_at_k(results, k) for k in PRECISION_K_VALUES}
    map_score = mean_average_precision(results, MAP_K)
    mrr = mean_reciprocal_rank(results)
    ndcg = mean_ndcg(results, NDCG_K)
    lat = latency_stats(results)

    return {
        "recall": recall,
        "precision": precision,
        "map": map_score,
        "mrr": mrr,
        "ndcg": ndcg,
        "latency": lat,
    }


def _print_report(
    metrics: dict,
    dataset_info: DatasetInfo,
    n_images: int,
    n_queries: int,
    model_name: str,
    device_str: str,
    seed: int,
    indexing_time_s: float,
    top_k: int,
) -> None:
    """Print the formatted benchmark report to stdout."""

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    def fmt4(v: float) -> str:
        return f"{v:.4f}"

    def fmt2(v: float) -> str:
        return f"{v:.2f}"

    LINE = "=" * 55
    THIN = "-" * 55

    print()
    print(LINE)
    print("RETRIEVR EVALUATION")
    print("=" * 19)
    print()
    print("## Dataset")
    print()
    print(f"  {'Dataset':<28}: Flickr8k")
    print(f"  {'Images Evaluated':<28}: {n_images}")
    print(f"  {'Queries Evaluated':<28}: {n_queries}")
    print(f"  {'Model':<28}: {model_name}")
    print(f"  {'Device':<28}: {device_str.upper()}")
    print(f"  {'Random Seed':<28}: {seed}")
    print()
    print(THIN)
    print()
    print("## Retrieval Metrics")
    print()

    r = metrics["recall"]
    for k in RECALL_K_VALUES:
        if k <= top_k:
            print(f"  {'Recall@' + str(k):<28}: {pct(r[k])}")

    print()
    p = metrics["precision"]
    for k in PRECISION_K_VALUES:
        if k <= top_k:
            print(f"  {'Precision@' + str(k):<28}: {pct(p[k])}")

    print()
    print(f"  {'mAP@' + str(MAP_K):<28}: {fmt4(metrics['map'])}")
    print(f"  {'MRR':<28}: {fmt4(metrics['mrr'])}")
    print(f"  {'nDCG@' + str(NDCG_K):<28}: {fmt4(metrics['ndcg'])}")

    print()
    print(THIN)
    print()
    print("## Performance")
    print()
    lat = metrics["latency"]
    print(f"  {'Average Query Latency':<28}: {fmt2(lat['average_ms'])} ms")
    print(f"  {'Median Query Latency':<28}: {fmt2(lat['median_ms'])} ms")
    print(f"  {'P95 Query Latency':<28}: {fmt2(lat['p95_ms'])} ms")
    print(f"  {'Indexing Time':<28}: {fmt2(indexing_time_s)} sec")
    print()
    print(LINE)
    print("Evaluation Complete")
    print("=" * 19)
    print()


def _save_results(
    metrics: dict,
    n_images: int,
    n_queries: int,
    indexing_time_s: float,
    top_k: int,
) -> None:
    """Save evaluation results to test/results/evaluation_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / RESULTS_JSON

    r = metrics["recall"]
    p = metrics["precision"]
    lat = metrics["latency"]

    payload = {
        "dataset": "Flickr8k",
        "images": n_images,
        "queries": n_queries,
        "metrics": {
            "recall_at_1": round(r.get(1, 0.0), 6),
            "recall_at_5": round(r.get(5, 0.0), 6),
            "recall_at_10": round(r.get(10, 0.0), 6),
            "recall_at_20": round(r.get(20, 0.0), 6),
            "precision_at_1": round(p.get(1, 0.0), 6),
            "precision_at_5": round(p.get(5, 0.0), 6),
            "precision_at_10": round(p.get(10, 0.0), 6),
            "precision_at_20": round(p.get(20, 0.0), 6),
            "map_at_10": round(metrics["map"], 6),
            "mrr": round(metrics["mrr"], 6),
            "ndcg_at_10": round(metrics["ndcg"], 6),
        },
        "performance": {
            "average_latency_ms": round(lat["average_ms"], 4),
            "median_latency_ms": round(lat["median_ms"], 4),
            "p95_latency_ms": round(lat["p95_ms"], 4),
            "indexing_time_seconds": round(indexing_time_s, 4),
        },
        "config": {
            "top_k": top_k,
            "map_k": MAP_K,
            "ndcg_k": NDCG_K,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Reproducibility ───────────────────────────────────────────────────────────
    _set_seeds(args.seed)

    dataset_dir = args.dataset if args.dataset is not None else DEFAULT_DATASET_DIR

    # ── Print run configuration ───────────────────────────────────────────────────
    # Ensure stdout can handle any Unicode characters on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print()
    print("Retrievr - Evaluation Module")
    print("-" * 40)
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  CLIP model   : {args.model}")
    print(f"  Limit        : {args.limit if args.limit else 'all'}")
    print(f"  Top-K        : {args.top_k}")
    print(f"  Seed         : {args.seed}")
    print("-" * 40)

    # ── Step 1: Load dataset ──────────────────────────────────────────────────────
    print("\n[1/5] Loading Flickr8k dataset ...")
    dataset: DatasetInfo = load_dataset(dataset_dir)

    # Apply --limit (on images; queries follow from that subset of images)
    if args.limit is not None and args.limit > 0:
        limited_images = set(dataset.image_filenames[: args.limit])
        dataset.image_filenames[:] = list(limited_images)
        dataset.queries[:] = [
            q for q in dataset.queries if q.image_filename in limited_images
        ]
        print(f"  Limited to {len(dataset.image_filenames)} images, "
              f"{len(dataset.queries)} queries.")
    else:
        print(f"  {len(dataset.image_filenames)} images, "
              f"{len(dataset.queries)} queries.")

    # ── Step 2: Load CLIP ─────────────────────────────────────────────────────────
    print("\n[2/5] Loading CLIP model ...")
    clip = CLIPAdapter(model_name=args.model, batch_size=args.batch_size)
    # Trigger model load now so the device is resolved
    _ = clip._load()
    device_str = str(clip.device)
    dim = clip.embedding_dim
    print(f"  Model: {args.model}  |  Device: {device_str.upper()}  |  Dim: {dim}")

    # ── Step 3: Encode images + build FAISS index ─────────────────────────────────
    print(f"\n[3/5] Encoding {len(dataset.image_filenames)} images and building FAISS index ...")
    image_paths = [dataset.image_dir / fname for fname in dataset.image_filenames]

    t_idx_start = time.perf_counter()
    image_embeddings = clip.encode_images(image_paths)
    faiss_index = _build_faiss_index(image_embeddings, dim)
    t_idx_end = time.perf_counter()
    indexing_time_s = t_idx_end - t_idx_start

    # id_map: FAISS position → image filename
    id_map: list[str] = list(dataset.image_filenames)

    print(f"  Indexed {faiss_index.ntotal} vectors in {indexing_time_s:.2f}s")

    # Validate that nothing was accidentally lost during encoding
    if faiss_index.ntotal != len(id_map):
        print(
            f"\n[WARNING] FAISS index has {faiss_index.ntotal} vectors "
            f"but id_map has {len(id_map)} entries. "
            f"Some images may have failed to load.",
            file=sys.stderr,
        )
        # Trim id_map to match what was actually indexed
        id_map = id_map[: faiss_index.ntotal]

    # ── Step 4: Run queries ───────────────────────────────────────────────────────
    print(f"\n[4/5] Running {len(dataset.queries)} caption queries (top-k={args.top_k}) ...")
    query_results = _run_queries(
        queries=dataset.queries,
        index=faiss_index,
        id_map=id_map,
        clip=clip,
        top_k=args.top_k,
    )

    # ── Step 5: Compute and report metrics ────────────────────────────────────────
    print("\n[5/5] Computing metrics ...")
    metrics = _compute_all_metrics(query_results, top_k=args.top_k)

    _print_report(
        metrics=metrics,
        dataset_info=dataset,
        n_images=len(id_map),
        n_queries=len(query_results),
        model_name=args.model,
        device_str=device_str,
        seed=args.seed,
        indexing_time_s=indexing_time_s,
        top_k=args.top_k,
    )

    # ── Optional: Save JSON ───────────────────────────────────────────────────────
    if args.save_results:
        _save_results(
            metrics=metrics,
            n_images=len(id_map),
            n_queries=len(query_results),
            indexing_time_s=indexing_time_s,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
