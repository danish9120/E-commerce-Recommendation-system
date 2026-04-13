"""
display.py — Terminal display helpers (no external rich/colorama required).
Uses ANSI escape codes and tabulate for clean console output.
"""

import os
from tabulate import tabulate
from typing import List, Dict

# ── ANSI colours ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
PURPLE = "\033[95m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"
GREY   = "\033[90m"

# Category colour map
CAT_COLORS = {
    "Electronics": CYAN,
    "Kitchen":     YELLOW,
    "Fitness":     GREEN,
    "Home":        PURPLE,
    "Footwear":    BLUE,
    "Accessories": WHITE,
    "Furniture":   RED,
}

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def color_category(cat: str) -> str:
    c = CAT_COLORS.get(cat, WHITE)
    return f"{c}{cat}{RESET}"

def stars(rating: float) -> str:
    full  = int(rating)
    empty = 5 - full
    return f"{YELLOW}{'★' * full}{'☆' * empty}{RESET} {DIM}{rating}{RESET}"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║       🛍️  E-COMMERCE RECOMMENDATION SYSTEM  🛍️              ║
║              Python  ·  ML-Powered  ·  4 Algorithms          ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

def section(title: str, color: str = CYAN):
    width = 62
    bar = "─" * width
    print(f"\n{color}{BOLD}┌{bar}┐")
    print(f"│  {title:<{width-2}}│")
    print(f"└{bar}┘{RESET}")

def print_products(products: List[Dict], score_key: str = None, label: str = "RECOMMENDATIONS"):
    section(f"🎯  {label}", CYAN)
    rows = []
    for i, p in enumerate(products, 1):
        score_str = f"{p[score_key]:.4f}" if score_key and score_key in p else "—"
        conv = f"{p['purchases']/p['views']*100:.1f}%"
        rows.append([
            f"{BOLD}{i}{RESET}",
            f"{BOLD}{p['name']}{RESET}",
            color_category(p["category"]),
            f"{GREEN}${p['price']}{RESET}",
            stars(p["rating"]),
            f"{DIM}{p['reviews']:,}{RESET}",
            f"{YELLOW}{conv}{RESET}",
            f"{CYAN}{score_str}{RESET}",
            f"{DIM}{', '.join(p['tags'])}{RESET}",
        ])
    headers = ["#", "Product", "Category", "Price", "Rating", "Reviews", "Conv%", "Score", "Tags"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

def print_user_profile(user_id: str, profile: Dict, history_products: List[Dict]):
    section(f"👤  User Profile: {profile['name']}", PURPLE)
    info = [
        ["User ID",       user_id],
        ["Name",          profile["name"]],
        ["Preferences",   ", ".join(profile["preferences"])],
        ["Age Group",     profile.get("age_group", "N/A")],
        ["History Count", len(profile["history"])],
        ["Cart Items",    len(profile.get("cart", []))],
    ]
    print(tabulate(info, tablefmt="rounded_outline"))
    print(f"\n{GREY}  Purchase / View History:{RESET}")
    for p in history_products:
        print(f"    {YELLOW}▸{RESET} {p['name']}  {DIM}(${p['price']} · {p['category']}){RESET}")

def print_evaluation_table(results: List[Dict]):
    section("📊  Algorithm Evaluation (k=6)", GREEN)
    if not results:
        print("  No evaluation data.")
        return
    headers = list(results[0].keys())
    rows = []
    for r in results:
        row = []
        for k, v in r.items():
            if k == "algorithm":
                row.append(f"{BOLD}{v}{RESET}")
            else:
                pct = float(v) * 100
                if pct >= 80:
                    row.append(f"{GREEN}{v}{RESET}")
                elif pct >= 50:
                    row.append(f"{YELLOW}{v}{RESET}")
                else:
                    row.append(f"{RED}{v}{RESET}")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

def print_similar_users(user_id: str, sim_series):
    section(f"👥  Users Similar to '{user_id}'", BLUE)
    rows = [[f"{BOLD}{uid}{RESET}", f"{GREEN}{score:.4f}{RESET}"]
            for uid, score in sim_series.items()]
    print(tabulate(rows, headers=["User", "Cosine Similarity"], tablefmt="rounded_outline"))

def print_analytics(df):
    section("📈  Product Analytics — Top 10 by Conversion Rate", YELLOW)
    top10 = df.nlargest(10, "conversion_rate")[
        ["name", "category", "price", "rating", "reviews", "views", "purchases", "conversion_rate"]
    ]
    rows = []
    for _, r in top10.iterrows():
        rows.append([
            f"{BOLD}{r['name']}{RESET}",
            color_category(r["category"]),
            f"{GREEN}${r['price']}{RESET}",
            stars(r["rating"]),
            f"{DIM}{r['reviews']:,}{RESET}",
            f"{YELLOW}{r['conversion_rate']}%{RESET}",
        ])
    print(tabulate(rows,
                   headers=["Product", "Category", "Price", "Rating", "Reviews", "Conv%"],
                   tablefmt="rounded_outline"))

def print_category_summary(df):
    section("🗂️  Category Summary", PURPLE)
    cats = df.groupby("category").agg(
        products=("id", "count"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
        total_revenue=("purchases", lambda x: (x * df.loc[x.index, "price"]).sum()),
        avg_conv=("conversion_rate", "mean"),
    ).reset_index()
    rows = []
    for _, r in cats.iterrows():
        rows.append([
            color_category(r["category"]),
            r["products"],
            f"${r['avg_price']:.0f}",
            f"{r['avg_rating']:.2f}",
            f"${r['total_revenue']:,.0f}",
            f"{r['avg_conv']:.1f}%",
        ])
    print(tabulate(rows,
                   headers=["Category", "Products", "Avg Price", "Avg Rating", "Revenue", "Conv%"],
                   tablefmt="rounded_outline"))

def separator():
    print(f"\n{GREY}{'─' * 64}{RESET}\n")