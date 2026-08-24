"""
metrics.py — Retrieval evaluation metrics for the Retrievr evaluation module.

All functions are implemented from scratch for clarity and interview-friendliness.
No opaque third-party evaluation libraries are used.

Metric signatures accept:
  retrieved     : list[str]       — ranked list of retrieved image IDs (best first)
  relevant      : set[str]        — set of ground-truth relevant image IDs
  k             : int             — cut-off rank

For nDCG, a relevance_scores dict maps image_id → int score (0–3).
In the Flickr8k benchmark, the associated image gets score 3, all others get 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────────
# Single-query metrics
# ─────────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K.

    Returns 0.0 if k <= 0 or retrieved is empty.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for img in top_k if img in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / |relevant|.

    Returns 0.0 if relevant is empty.
    Handles single-relevant-image case: returns 1.0 if the image appears in top-K.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for img in top_k if img in relevant)
    return hits / len(relevant)


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Average Precision@K (AP@K).

    Computes the mean of precision values at each rank where a relevant document
    is retrieved, up to rank K.  Rewards early retrieval of relevant items.

    Formula:
        AP@K = (1 / |relevant|) * Σ_{i=1}^{K} [r_i ∈ relevant] * P@i

    where P@i is precision at rank i and [·] is the Iverson bracket.

    Returns 0.0 if relevant is empty or no relevant item found in top-K.
    """
    if not relevant or k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, img in enumerate(retrieved[:k], start=1):
        if img in relevant:
            hits += 1
            precision_sum += hits / rank  # P@rank at the time of this hit

    # Normalise by the total number of relevant docs (not min(|relevant|, K))
    # This matches the standard IR definition of AP.
    return precision_sum / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_result.
    Returns 0.0 if no relevant result is found in the retrieved list.
    """
    for rank, img in enumerate(retrieved, start=1):
        if img in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain@K.

    Uses the standard logarithmic discount:
        DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)

    where rel_i is the relevance score of the item at rank i.

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved image IDs.
    relevance_scores:
        Mapping of image_id → relevance score (0 = not relevant, 3 = highly relevant).
    k:
        Rank cut-off.
    """
    dcg = 0.0
    for rank, img in enumerate(retrieved[:k], start=1):
        rel = relevance_scores.get(img, 0)
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain@K.

    nDCG@K = DCG@K / IDCG@K

    where IDCG@K is the DCG of the ideal (perfect) ranking.
    Returns 0.0 if the ideal DCG is 0 (no relevant items).

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved image IDs.
    relevance_scores:
        Mapping of image_id → relevance score.
    k:
        Rank cut-off.
    """
    dcg = dcg_at_k(retrieved, relevance_scores, k)

    # Ideal ranking: sort all relevant items by score descending
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    ideal_retrieved = [f"__ideal_{i}__" for i in range(len(ideal_scores))]
    ideal_rel_map = {img: score for img, score in zip(ideal_retrieved, ideal_scores)}
    idcg = dcg_at_k(ideal_retrieved, ideal_rel_map, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ─────────────────────────────────────────────────────────────────────────────────
# Per-query result container
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """All data needed to compute metrics for a single query."""
    query: str
    relevant: set[str]
    retrieved: list[str]          # ranked, best first
    relevance_scores: dict[str, int]  # image_id → grade
    latency_ms: float             # encode + FAISS search time


# ─────────────────────────────────────────────────────────────────────────────────
# Aggregate metrics over all queries
# ─────────────────────────────────────────────────────────────────────────────────

def mean_average_precision(results: list[QueryResult], k: int) -> float:
    """
    Mean Average Precision@K (mAP@K).

    mAP@K = mean(AP@K for all queries)
    """
    if not results:
        return 0.0
    aps = [average_precision_at_k(r.retrieved, r.relevant, k) for r in results]
    return sum(aps) / len(aps)


def mean_reciprocal_rank(results: list[QueryResult]) -> float:
    """
    Mean Reciprocal Rank (MRR).

    MRR = mean(RR for all queries)
    """
    if not results:
        return 0.0
    rrs = [reciprocal_rank(r.retrieved, r.relevant) for r in results]
    return sum(rrs) / len(rrs)


def mean_ndcg(results: list[QueryResult], k: int) -> float:
    """
    Mean nDCG@K across all queries.
    """
    if not results:
        return 0.0
    scores = [ndcg_at_k(r.retrieved, r.relevance_scores, k) for r in results]
    return sum(scores) / len(scores)


def mean_recall_at_k(results: list[QueryResult], k: int) -> float:
    """Mean Recall@K across all queries."""
    if not results:
        return 0.0
    return sum(recall_at_k(r.retrieved, r.relevant, k) for r in results) / len(results)


def mean_precision_at_k(results: list[QueryResult], k: int) -> float:
    """Mean Precision@K across all queries."""
    if not results:
        return 0.0
    return sum(precision_at_k(r.retrieved, r.relevant, k) for r in results) / len(results)


# ─────────────────────────────────────────────────────────────────────────────────
# Latency statistics
# ─────────────────────────────────────────────────────────────────────────────────

def latency_stats(results: list[QueryResult]) -> dict[str, float]:
    """
    Returns average, median, and P95 latency in milliseconds.

    Latency includes only text encoding + FAISS search time,
    NOT model loading or image indexing.
    """
    if not results:
        return {"average_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    latencies = sorted(r.latency_ms for r in results)
    n = len(latencies)
    average_ms = sum(latencies) / n

    # Median
    mid = n // 2
    if n % 2 == 0:
        median_ms = (latencies[mid - 1] + latencies[mid]) / 2
    else:
        median_ms = latencies[mid]

    # P95 — nearest-rank method
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95_ms = latencies[p95_idx]

    return {"average_ms": average_ms, "median_ms": median_ms, "p95_ms": p95_ms}
