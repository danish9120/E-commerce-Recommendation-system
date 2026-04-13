"""
demo.py — Non-interactive demo that showcases all features automatically.
Generates a full printed report. Run: python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data import PRODUCTS, USER_PROFILES, get_products_df
from algorithms import (
    CollaborativeFilter, ContentBasedFilter,
    TrendingRecommender, HybridRecommender,
)
from evaluation import (
    evaluate_algorithm, catalogue_coverage, intra_list_diversity
)
from display import (
    banner, section, separator,
    print_products, print_user_profile, print_evaluation_table,
    print_similar_users, print_analytics, print_category_summary,
    BOLD, RESET, CYAN, GREEN, YELLOW, RED, GREY, DIM,
)

def run_demo():
    banner()
    print(f"  {GREY}Running automated demo of all recommendation features…{RESET}\n")

    products_map = {p["id"]: p for p in PRODUCTS}

    # ── Init engines ─────────────────────────────────────────────────────────
    cf     = CollaborativeFilter(k_neighbours=3)
    cb     = ContentBasedFilter()
    trend  = TrendingRecommender()
    hybrid = HybridRecommender(cf_weight=1.2, cb_weight=1.0)

    # ── 1. User Profiles ─────────────────────────────────────────────────────
    section("👤  USER PROFILES", CYAN)
    for uid, profile in USER_PROFILES.items():
        history_prods = [products_map[p] for p in profile["history"] if p in products_map]
        print_user_profile(uid, profile, history_prods)
        separator()

    # ── 2. Trending ───────────────────────────────────────────────────────────
    trend_recs = trend.recommend(top_n=6)
    print_products(trend_recs, score_key="trend_score", label="🔥  Trending Products")

    # ── 3. Recommendations per user per algorithm ──────────────────────────────
    for uid in USER_PROFILES:
        name = USER_PROFILES[uid]["name"]

        h_recs = hybrid.recommend(uid, top_n=6)
        print_products(h_recs, score_key="hybrid_score",
                       label=f"⚡  Hybrid Recs → {name}")

        cf_recs = cf.recommend(uid, top_n=6)
        print_products(cf_recs, score_key="cf_score",
                       label=f"👥  Collaborative Recs → {name}")

        cb_recs = cb.recommend(uid, top_n=6)
        print_products(cb_recs, score_key="cb_score",
                       label=f"🧬  Content-Based Recs → {name}")

        sims = cf.get_similar_users(uid)
        print_similar_users(uid, sims)
        separator()

    # ── 4. Analytics ─────────────────────────────────────────────────────────
    df = get_products_df()
    print_analytics(df)
    print_category_summary(df)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    held_out = {uid: p.get("cart", []) for uid, p in USER_PROFILES.items() if p.get("cart")}
    eval_results = []
    for algo_name, engine in [("Hybrid", hybrid), ("Collaborative", cf), ("Content-Based", cb)]:
        recs_by_user = {uid: [r["id"] for r in engine.recommend(uid, top_n=6)] for uid in held_out}
        eval_results.append(evaluate_algorithm(algo_name, recs_by_user, held_out, k=6))

    trend_ids = [r["id"] for r in trend.recommend(top_n=6)]
    eval_results.append(evaluate_algorithm(
        "Trending", {uid: trend_ids for uid in held_out}, held_out, k=6
    ))
    print_evaluation_table(eval_results)

    # Coverage & Diversity
    all_rec_lists = [[r["id"] for r in hybrid.recommend(uid, top_n=6)] for uid in USER_PROFILES]
    cov = catalogue_coverage(all_rec_lists, len(PRODUCTS))
    div = intra_list_diversity(hybrid.recommend("alice", top_n=6))
    section("📐  Coverage & Diversity Metrics", GREEN)
    print(f"  Hybrid catalogue coverage (all users) : {GREEN}{cov}%{RESET}")
    print(f"  Intra-list diversity (alice, k=6)      : {GREEN}{div:.2%}{RESET}\n")

    print(f"\n{GREEN}{BOLD}  ✅  Demo complete. All algorithms executed successfully!{RESET}\n")


if __name__ == "__main__":
    run_demo()