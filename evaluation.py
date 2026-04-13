"""
evaluation.py — Offline evaluation metrics for recommendation systems.
  - Precision@K
  - Recall@K
  - F1@K
  - NDCG@K  (Normalised Discounted Cumulative Gain)
  - Coverage (catalogue coverage %)
  - Diversity (intra-list category diversity)
"""

import numpy as np
import math
from typing import List, Dict


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Fraction of top-K recommended items that are relevant."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return round(hits / k, 4) if k else 0.0


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return round(hits / len(relevant), 4) if relevant else 0.0


def f1_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def dcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    rel_set = set(relevant)
    dcg = 0.0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Normalised DCG — ideal DCG is achieved when all relevant items are ranked first."""
    ideal = sorted([1] * min(len(relevant), k) + [0] * max(k - len(relevant), 0),
                   reverse=True)
    ideal_dcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    if ideal_dcg == 0:
        return 0.0
    actual_dcg = dcg_at_k(recommended, relevant, k)
    return round(actual_dcg / ideal_dcg, 4)


def catalogue_coverage(all_recommendations: List[List[int]], total_items: int) -> float:
    """Percentage of the catalogue that appears in at least one recommendation list."""
    unique_recommended = set(pid for recs in all_recommendations for pid in recs)
    return round(len(unique_recommended) / total_items * 100, 2)


def intra_list_diversity(recommended_products: List[Dict]) -> float:
    """
    Category diversity of a single recommendation list.
    = unique categories / total recommendations
    """
    if not recommended_products:
        return 0.0
    categories = [p["category"] for p in recommended_products]
    return round(len(set(categories)) / len(categories), 4)


def evaluate_algorithm(algo_name: str, recs_by_user: Dict[str, List[int]],
                        held_out: Dict[str, List[int]], k: int = 6) -> Dict:
    """
    Evaluate a recommender for all users and aggregate results.

    recs_by_user : {user_id: [recommended product ids]}
    held_out     : {user_id: [ground-truth relevant product ids]}
    """
    p_scores, r_scores, f1_scores, ndcg_scores = [], [], [], []

    for uid, recs in recs_by_user.items():
        relevant = held_out.get(uid, [])
        if not relevant:
            continue
        p_scores.append(precision_at_k(recs, relevant, k))
        r_scores.append(recall_at_k(recs, relevant, k))
        f1_scores.append(f1_at_k(recs, relevant, k))
        ndcg_scores.append(ndcg_at_k(recs, relevant, k))

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "algorithm":   algo_name,
        f"precision@{k}": avg(p_scores),
        f"recall@{k}":    avg(r_scores),
        f"f1@{k}":        avg(f1_scores),
        f"ndcg@{k}":      avg(ndcg_scores),
    }