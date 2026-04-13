"""
main.py — Interactive CLI for the E-Commerce Recommendation System.

Run:
    python main.py

Menu options:
  1. View personalised recommendations (choose user + algorithm)
  2. Browse full product catalogue
  3. View analytics & category summary
  4. Run algorithm evaluation
  5. Show user profiles
  6. Find similar users
  7. Exit
"""

import sys
from data import PRODUCTS, USER_PROFILES, get_products_df
from algorithms import (
    CollaborativeFilter,
    ContentBasedFilter,
    TrendingRecommender,
    HybridRecommender,
)
from evaluation import evaluate_algorithm
from display import (
    banner, section, separator, clear,
    print_products, print_user_profile, print_evaluation_table,
    print_similar_users, print_analytics, print_category_summary,
    BOLD, RESET,BLUE, CYAN, GREEN, YELLOW, RED, PURPLE, GREY, DIM, WHITE,
)

# ── Pre-init algorithms (avoids re-fitting on every call) ─────────────────────
cf_engine      = CollaborativeFilter(k_neighbours=3)
cb_engine      = ContentBasedFilter()
trend_engine   = TrendingRecommender()
hybrid_engine  = HybridRecommender(cf_weight=1.2, cb_weight=1.0)
products_df    = get_products_df()
products_map   = {p["id"]: p for p in PRODUCTS}

ALGO_MENU = {
    "1": ("Hybrid",        hybrid_engine),
    "2": ("Collaborative", cf_engine),
    "3": ("Content-Based", cb_engine),
    "4": ("Trending",      trend_engine),
}

USER_LIST = list(USER_PROFILES.keys())


def pick_user() -> str:
    print(f"\n{CYAN}  Select a user:{RESET}")
    for i, uid in enumerate(USER_LIST, 1):
        u = USER_PROFILES[uid]
        print(f"    {BOLD}{i}.{RESET} {u['name']} {DIM}({uid}){RESET}  — prefers: {', '.join(u['preferences'])}")
    choice = input(f"\n  Enter number [1-{len(USER_LIST)}]: ").strip()
    try:
        return USER_LIST[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"{RED}  Invalid choice, defaulting to alice.{RESET}")
        return "alice"


def pick_algorithm() -> tuple:
    print(f"\n{CYAN}  Select algorithm:{RESET}")
    descs = {
        "1": "Ensemble of CF + Content-Based (best accuracy)",
        "2": "User-similarity matrix, cosine k-NN",
        "3": "Tag / category feature matching",
        "4": "Highest purchase conversion rate",
    }
    for k, (name, _) in ALGO_MENU.items():
        print(f"    {BOLD}{k}.{RESET} {name:<18} {DIM}{descs[k]}{RESET}")
    choice = input("\n  Enter number [1-4]: ").strip()
    return ALGO_MENU.get(choice, ALGO_MENU["1"])


# ── Menu handlers ─────────────────────────────────────────────────────────────

def menu_recommendations():
    user_id   = pick_user()
    algo_name, engine = pick_algorithm()

    print(f"\n{GREY}  ⏳ Generating {algo_name} recommendations for {USER_PROFILES[user_id]['name']}...{RESET}")

    if algo_name == "Trending":
        recs = engine.recommend(top_n=6)
        score_key = "trend_score"
    else:
        recs = engine.recommend(user_id, top_n=6)
        score_key = {"Hybrid": "hybrid_score",
                     "Collaborative": "cf_score",
                     "Content-Based": "cb_score"}.get(algo_name)

    label = f"{algo_name} Recommendations for {USER_PROFILES[user_id]['name']}"
    print_products(recs, score_key=score_key, label=label)

    # Show explanation
    if algo_name == "Collaborative":
        sims = cf_engine.get_similar_users(user_id)
        print_similar_users(user_id, sims)
    elif algo_name == "Content-Based":
        profile = USER_PROFILES[user_id]
        section("🔍  Why these recommendations?", PURPLE)
        print(f"  Based on tags from your history, we matched products sharing tags")
        print(f"  with items you've viewed, boosted by preferred categories:")
        print(f"  {YELLOW}{', '.join(profile['preferences'])}{RESET}\n")


def menu_catalogue():
    section("📦  Full Product Catalogue", YELLOW)
    # Simple filter
    print(f"\n  Filter by category (leave blank for all):")
    cats = sorted(set(p["category"] for p in PRODUCTS))
    for i, c in enumerate(cats, 1):
        print(f"    {DIM}{i}.{RESET} {c}")
    raw = input("\n  Number (or Enter for all): ").strip()
    if raw:
        try:
            cat_filter = cats[int(raw) - 1]
            filtered = [p for p in PRODUCTS if p["category"] == cat_filter]
        except (ValueError, IndexError):
            filtered = PRODUCTS
    else:
        filtered = PRODUCTS
    print_products(filtered, label=f"Catalogue ({len(filtered)} products)")


def menu_analytics():
    print_analytics(products_df)
    print_category_summary(products_df)

    # Extra stats
    section("💡  Quick Stats", GREEN)
    total_rev = sum(p["price"] * p["purchases"] for p in PRODUCTS)
    avg_rating = sum(p["rating"] for p in PRODUCTS) / len(PRODUCTS)
    print(f"  Total simulated revenue : {GREEN}${total_rev:,.0f}{RESET}")
    print(f"  Average product rating  : {YELLOW}{avg_rating:.2f} / 5.0{RESET}")
    print(f"  Total products          : {CYAN}{len(PRODUCTS)}{RESET}")
    print(f"  Total users             : {CYAN}{len(USER_PROFILES)}{RESET}")
    best = max(PRODUCTS, key=lambda p: p["purchases"] / p["views"])
    print(f"  Best conversion product : {WHITE}{best['name']}{RESET} "
          f"({YELLOW}{best['purchases']/best['views']*100:.1f}%{RESET})\n")


def menu_evaluation():
    section("🧪  Running Offline Evaluation", GREEN)
    print(f"  {GREY}Using leave-one-out on each user's history as ground truth…{RESET}\n")

    # Simulate: held-out = items in cart; training = history minus cart
    held_out = {uid: p.get("cart", []) for uid, p in USER_PROFILES.items() if p.get("cart")}

    results = []
    for algo_name, engine in [
        ("Hybrid",        hybrid_engine),
        ("Collaborative", cf_engine),
        ("Content-Based", cb_engine),
    ]:
        recs_by_user = {}
        for uid in held_out:
            if algo_name == "Trending":
                recs_by_user[uid] = [r["id"] for r in engine.recommend(top_n=6)]
            else:
                recs = engine.recommend(uid, top_n=6)
                recs_by_user[uid] = [r["id"] for r in recs]
        results.append(evaluate_algorithm(algo_name, recs_by_user, held_out, k=6))

    # Trending (not user-specific — coverage matters more)
    trend_recs = [r["id"] for r in trend_engine.recommend(top_n=6)]
    trend_res = evaluate_algorithm(
        "Trending",
        {uid: trend_recs for uid in held_out},
        held_out, k=6
    )
    results.append(trend_res)

    print_evaluation_table(results)

    # Coverage
    all_rec_lists = []
    for uid in USER_PROFILES:
        all_rec_lists.append([r["id"] for r in hybrid_engine.recommend(uid, top_n=6)])
    from evaluation import catalogue_coverage, intra_list_diversity
    cov = catalogue_coverage(all_rec_lists, len(PRODUCTS))
    div = intra_list_diversity(hybrid_engine.recommend("alice", top_n=6))
    print(f"\n  {CYAN}Hybrid engine catalogue coverage : {GREEN}{cov}%{RESET}")
    print(f"  {CYAN}Intra-list diversity (alice)      : {GREEN}{div:.2%}{RESET}\n")


def menu_user_profiles():
    user_id = pick_user()
    profile = USER_PROFILES[user_id]
    history_prods = [products_map[pid] for pid in profile["history"] if pid in products_map]
    print_user_profile(user_id, profile, history_prods)

    cart_prods = [products_map[pid] for pid in profile.get("cart", []) if pid in products_map]
    if cart_prods:
        section("🛒  Cart", GREEN)
        for p in cart_prods:
            print(f"    {YELLOW}▸{RESET} {p['name']}  {DIM}(${p['price']}){RESET}")
    print()


def menu_similar_users():
    user_id = pick_user()
    sims = cf_engine.get_similar_users(user_id)
    print_similar_users(user_id, sims)
    section("🔍  Top Similar User's Recommendation Overlap", BLUE)
    top_sim_uid = sims.index[0]
    top_sim_profile = USER_PROFILES[top_sim_uid]
    print(f"  Most similar user: {BOLD}{top_sim_profile['name']}{RESET} ({top_sim_uid})")
    shared = set(USER_PROFILES[user_id]["history"]) & set(top_sim_profile["history"])
    print(f"  Shared history items: {YELLOW}{len(shared)}{RESET}")
    for pid in shared:
        p = products_map.get(pid)
        if p:
            print(f"    {GREY}· {p['name']}{RESET}")
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────

MENU_OPTIONS = {
    "1": ("🎯  Personalised Recommendations",  menu_recommendations),
    "2": ("📦  Browse Product Catalogue",       menu_catalogue),
    "3": ("📈  Analytics & Category Summary",   menu_analytics),
    "4": ("🧪  Algorithm Evaluation",           menu_evaluation),
    "5": ("👤  User Profiles",                  menu_user_profiles),
    "6": ("👥  Find Similar Users",             menu_similar_users),
    "7": ("🚪  Exit",                           None),
}


def main():
    clear()
    banner()
    print(f"  {GREY}Algorithms loaded: Hybrid · Collaborative · Content-Based · Trending{RESET}")
    print(f"  {GREY}Products: {len(PRODUCTS)}   Users: {len(USER_PROFILES)}   Categories: 7{RESET}\n")

    while True:
        section("📋  MAIN MENU", WHITE)
        for key, (label, _) in MENU_OPTIONS.items():
            print(f"    {BOLD}{key}.{RESET}  {label}")
        separator()
        choice = input(f"  {CYAN}Enter option [1-7]:{RESET} ").strip()

        if choice == "7":
            print(f"\n  {GREEN}Thank you for using RecoEngine. Goodbye! 👋{RESET}\n")
            sys.exit(0)

        handler = MENU_OPTIONS.get(choice, (None, None))[1]
        if handler:
            try:
                handler()
            except Exception as e:
                print(f"\n{RED}  Error: {e}{RESET}\n")
        else:
            print(f"\n{RED}  Invalid option. Please enter 1–7.{RESET}\n")

        input(f"\n  {GREY}Press Enter to return to menu…{RESET}")
        clear()
        banner()


if __name__ == "__main__":
    main()