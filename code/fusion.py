"""Score and rank fusion of a dense and a lexical result list (own implementation).

Conventions (kept identical to the production platform studied in the paper, which test_fusion_equivalence.py checks
whenever that platform's code is available locally):
  * min-max normalisation per list; an empty list stays empty; a constant list maps to 1.0 if it has a single element,
    to 0.0 otherwise;
  * ranks start at 1, higher score = better rank, tied scores share the mean of their ranks;
  * convex fusion: dense_weight * dense + (1 - dense_weight) * lexical, a document absent from a list scoring 0 there;
  * weighted RRF: 1 / (k + rank_dense) + sparse_weight / (k + rank_lexical), a document absent from a list taking the
    rank len(list) + 1 (rank 1 if the list is empty).
"""
from __future__ import annotations

Scores = dict[str, float]


def normalize_scores_minmax(scores: Scores) -> Scores:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        fill = 1.0 if len(scores) == 1 else 0.0
        return dict.fromkeys(scores, fill)
    span = hi - lo
    return {doc: (value - lo) / span for doc, value in scores.items()}


def compute_ranks_with_ties(scores: Scores) -> Scores:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranks: Scores = {}
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        shared = (start + 1 + end + 1) / 2          # mean of the 1-based ranks start+1 .. end+1
        for doc, _ in ordered[start:end + 1]:
            ranks[doc] = shared
        start = end + 1
    return ranks


def fuse_weighted_linear(dense_scores: Scores, bm25_scores: Scores, dense_weight: float = 0.6) -> Scores:
    lexical_weight = 1.0 - dense_weight
    return {doc: dense_weight * dense_scores.get(doc, 0.0) + lexical_weight * bm25_scores.get(doc, 0.0)
            for doc in dense_scores.keys() | bm25_scores.keys()}


def fuse_rrf(dense_scores: Scores, bm25_scores: Scores, k: int = 60, sparse_weight: float = 1.0) -> Scores:
    rank_d, rank_l = compute_ranks_with_ties(dense_scores), compute_ranks_with_ties(bm25_scores)
    missing_d = len(dense_scores) + 1 if dense_scores else 1
    missing_l = len(bm25_scores) + 1 if bm25_scores else 1
    return {doc: 1.0 / (k + rank_d.get(doc, missing_d)) + sparse_weight / (k + rank_l.get(doc, missing_l))
            for doc in dense_scores.keys() | bm25_scores.keys()}
