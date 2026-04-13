"""
algorithms.py — Four recommendation algorithms:
  1. Collaborative Filtering  (user-based, cosine similarity)
  2. Content-Based Filtering  (tag + category feature matching)
  3. Trending                 (conversion-rate signal)
  4. Hybrid                   (weighted ensemble of CF + CB)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Dict

from data import PRODUCTS, USER_PROFILES, get_products_df, get_user_item_matrix


# ──────────────────────────────────────────────────────────────────────────────
# 1. COLLABORATIVE FILTERING
# ──────────────────────────────────────────────────────────────────────────────

class CollaborativeFilter:
    """
    User-based collaborative filtering using cosine similarity.
    Steps:
      1. Build binary user-item interaction matrix.
      2. Compute pairwise cosine similarity between users.
      3. For target user, find k most-similar neighbours.
      4. Score unseen items by weighted neighbour interactions.
      5. Return top-N scored items.
    """

    def __init__(self, k_neighbours: int = 3):
        self.k = k_neighbours
        self.matrix = get_user_item_matrix()
        self.users = list(self.matrix.index)
        sim_matrix = cosine_similarity(self.matrix.values)
        self.similarity = pd.DataFrame(sim_matrix, index=self.users, columns=self.users)

    def recommend(self, user_id: str, top_n: int = 6) -> List[Dict]:
        if user_id not in self.users:
            return []

        profile = USER_PROFILES[user_id]
        seen_ids = set(profile["history"] + profile.get("cart", []))

        # k nearest neighbours (excluding self)
        sims = self.similarity[user_id].drop(user_id).sort_values(ascending=False)
        neighbours = sims.head(self.k)

        # Score each unseen product
        scores: Dict[int, float] = {}
        for pid in self.matrix.columns:
            if pid in seen_ids:
                continue
            weighted_sum = sum(
                neighbours[n] * self.matrix.loc[n, pid]
                for n in neighbours.index
            )
            if weighted_sum > 0:
                scores[pid] = round(weighted_sum, 4)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
        products_map = {p["id"]: p for p in PRODUCTS}
        return [
            {**products_map[pid], "cf_score": scores[pid]}
            for pid in ranked_ids if pid in products_map
        ]

    def get_similar_users(self, user_id: str) -> pd.Series:
        """Return similarity scores for all other users."""
        return self.similarity[user_id].drop(user_id).sort_values(ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONTENT-BASED FILTERING
# ──────────────────────────────────────────────────────────────────────────────

class ContentBasedFilter:
    """
    Content-based filtering using product metadata (tags + category).
    Steps:
      1. One-hot encode product tags with MultiLabelBinarizer.
      2. Build a user profile vector by averaging vectors of viewed items.
      3. Compute cosine similarity between user vector and each product vector.
      4. Boost by category preference match and product rating.
      5. Return top-N scored unseen items.
    """

    def __init__(self):
        self.df = get_products_df()
        self.mlb = MultiLabelBinarizer()
        tag_matrix = self.mlb.fit_transform(self.df["tags"])
        self.tag_df = pd.DataFrame(tag_matrix, columns=self.mlb.classes_,
                                   index=self.df["id"])

    def _user_vector(self, user_id: str) -> np.ndarray:
        profile = USER_PROFILES[user_id]
        seen_ids = [pid for pid in profile["history"] if pid in self.tag_df.index]
        if not seen_ids:
            return np.zeros(len(self.mlb.classes_))
        return self.tag_df.loc[seen_ids].values.mean(axis=0)

    def recommend(self, user_id: str, top_n: int = 6) -> List[Dict]:
        if user_id not in USER_PROFILES:
            return []

        profile = USER_PROFILES[user_id]
        seen_ids = set(profile["history"] + profile.get("cart", []))
        preferred_categories = set(profile.get("preferences", []))

        user_vec = self._user_vector(user_id).reshape(1, -1)
        scores: Dict[int, float] = {}

        for _, row in self.df.iterrows():
            pid = row["id"]
            if pid in seen_ids:
                continue
            prod_vec = self.tag_df.loc[pid].values.reshape(1, -1)
            tag_sim = cosine_similarity(user_vec, prod_vec)[0][0]
            cat_bonus = 0.3 if row["category"] in preferred_categories else 0.0
            rating_bonus = (row["rating"] - 4.0) * 0.1   # small normalised boost
            scores[pid] = round(tag_sim + cat_bonus + rating_bonus, 4)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
        products_map = {p["id"]: p for p in PRODUCTS}
        return [
            {**products_map[pid], "cb_score": scores[pid]}
            for pid in ranked_ids if pid in products_map
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 3. TRENDING ALGORITHM
# ──────────────────────────────────────────────────────────────────────────────

class TrendingRecommender:
    """
    Surface products with the highest purchase conversion rate.
    conversion_rate = purchases / views
    An optional popularity_weight blends conversion rate with raw review count.
    """

    def recommend(self, top_n: int = 6, popularity_weight: float = 0.2) -> List[Dict]:
        df = get_products_df()
        df["trend_score"] = (
            (1 - popularity_weight) * df["conversion_rate"] / 100
            + popularity_weight * (df["reviews"] / df["reviews"].max())
        ).round(4)
        top = df.nlargest(top_n, "trend_score")
        result = []
        products_map = {p["id"]: p for p in PRODUCTS}
        for _, row in top.iterrows():
            product = dict(products_map[row["id"]])
            product["trend_score"] = row["trend_score"]
            result.append(product)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. HYBRID ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Weighted ensemble of Collaborative Filtering + Content-Based Filtering.
    cf_weight  : relative weight for CF scores  (default 1.2)
    cb_weight  : relative weight for CB scores  (default 1.0)
    Scores are normalised per algorithm before blending.
    """

    def __init__(self, cf_weight: float = 1.2, cb_weight: float = 1.0):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf = CollaborativeFilter()
        self.cb = ContentBasedFilter()

    def _normalise(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        max_val = max(scores.values()) or 1
        return {k: v / max_val for k, v in scores.items()}

    def recommend(self, user_id: str, top_n: int = 6) -> List[Dict]:
        cf_recs = self.cf.recommend(user_id, top_n=12)
        cb_recs = self.cb.recommend(user_id, top_n=12)

        cf_scores = self._normalise({r["id"]: r["cf_score"] for r in cf_recs})
        cb_scores = self._normalise({r["id"]: r["cb_score"] for r in cb_recs})

        all_ids = set(cf_scores) | set(cb_scores)
        blended: Dict[int, float] = {}
        for pid in all_ids:
            blended[pid] = (
                self.cf_weight * cf_scores.get(pid, 0)
                + self.cb_weight * cb_scores.get(pid, 0)
            )

        ranked_ids = sorted(blended, key=blended.get, reverse=True)[:top_n]
        products_map = {p["id"]: p for p in PRODUCTS}
        return [
            {**products_map[pid], "hybrid_score": round(blended[pid], 4)}
            for pid in ranked_ids if pid in products_map
        ]