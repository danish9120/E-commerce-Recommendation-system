"""
data.py — Product catalog and user profiles for the recommendation system.
"""

import pandas as pd
import numpy as np

# ─── PRODUCTS ──────────────────────────────────────────────────────────────────

PRODUCTS = [
    {"id": 1,  "name": "Arc Wireless Headphones",   "category": "Electronics", "price": 249, "rating": 4.8, "reviews": 1240, "tags": ["audio","wireless","premium"],     "views": 980,  "purchases": 320},
    {"id": 2,  "name": "Ember Smart Mug 2",          "category": "Kitchen",     "price": 149, "rating": 4.6, "reviews": 876,  "tags": ["coffee","smart","kitchen"],       "views": 740,  "purchases": 215},
    {"id": 3,  "name": "Ultralight Trail Runner",    "category": "Footwear",    "price": 189, "rating": 4.7, "reviews": 2100, "tags": ["running","outdoor","sport"],      "views": 1560, "purchases": 540},
    {"id": 4,  "name": "Minimal Desk Lamp",          "category": "Home",        "price": 89,  "rating": 4.5, "reviews": 430,  "tags": ["desk","lighting","minimal"],      "views": 610,  "purchases": 180},
    {"id": 5,  "name": "Mechanical Keyboard TKL",    "category": "Electronics", "price": 159, "rating": 4.9, "reviews": 3200, "tags": ["keyboard","mechanical","gaming"], "views": 2100, "purchases": 890},
    {"id": 6,  "name": "Leather Bifold Wallet",      "category": "Accessories", "price": 65,  "rating": 4.4, "reviews": 560,  "tags": ["leather","wallet","minimalist"],  "views": 420,  "purchases": 210},
    {"id": 7,  "name": "Noise-Cancel Earbuds Pro",   "category": "Electronics", "price": 199, "rating": 4.7, "reviews": 1870, "tags": ["audio","wireless","anc"],         "views": 1340, "purchases": 460},
    {"id": 8,  "name": "Stainless Steel Bottle",     "category": "Kitchen",     "price": 45,  "rating": 4.6, "reviews": 3400, "tags": ["water","outdoor","eco"],          "views": 1890, "purchases": 780},
    {"id": 9,  "name": "Ergonomic Office Chair",     "category": "Furniture",   "price": 599, "rating": 4.8, "reviews": 980,  "tags": ["office","ergonomic","work"],      "views": 880,  "purchases": 120},
    {"id": 10, "name": "4K Webcam Pro",              "category": "Electronics", "price": 129, "rating": 4.5, "reviews": 720,  "tags": ["webcam","video","streaming"],     "views": 960,  "purchases": 290},
    {"id": 11, "name": "Yoga Mat Elite",             "category": "Fitness",     "price": 78,  "rating": 4.7, "reviews": 1100, "tags": ["yoga","fitness","sport"],         "views": 730,  "purchases": 320},
    {"id": 12, "name": "Portable SSD 1TB",           "category": "Electronics", "price": 109, "rating": 4.9, "reviews": 2800, "tags": ["storage","portable","fast"],      "views": 1650, "purchases": 620},
    {"id": 13, "name": "French Press Copper",        "category": "Kitchen",     "price": 55,  "rating": 4.5, "reviews": 340,  "tags": ["coffee","kitchen","copper"],      "views": 380,  "purchases": 140},
    {"id": 14, "name": "Running GPS Watch",          "category": "Fitness",     "price": 329, "rating": 4.8, "reviews": 1560, "tags": ["running","gps","fitness"],        "views": 1100, "purchases": 280},
    {"id": 15, "name": "Linen Throw Blanket",        "category": "Home",        "price": 69,  "rating": 4.6, "reviews": 670,  "tags": ["home","comfort","linen"],         "views": 510,  "purchases": 230},
    {"id": 16, "name": "Electric Kettle Slim",       "category": "Kitchen",     "price": 79,  "rating": 4.4, "reviews": 480,  "tags": ["kettle","kitchen","electric"],    "views": 440,  "purchases": 165},
    {"id": 17, "name": "Bamboo Cutting Board",       "category": "Kitchen",     "price": 39,  "rating": 4.3, "reviews": 820,  "tags": ["kitchen","eco","bamboo"],         "views": 620,  "purchases": 280},
    {"id": 18, "name": "LED Monitor 27in 4K",        "category": "Electronics", "price": 449, "rating": 4.7, "reviews": 1450, "tags": ["monitor","4k","display"],         "views": 1200, "purchases": 340},
    {"id": 19, "name": "Foam Roller Set",            "category": "Fitness",     "price": 34,  "rating": 4.5, "reviews": 530,  "tags": ["fitness","recovery","sport"],     "views": 480,  "purchases": 190},
    {"id": 20, "name": "Scented Soy Candle Set",     "category": "Home",        "price": 42,  "rating": 4.4, "reviews": 390,  "tags": ["home","aromatherapy","comfort"],  "views": 360,  "purchases": 145},
]

# ─── USER PROFILES ─────────────────────────────────────────────────────────────

USER_PROFILES = {
    "alice": {
        "name": "Alice Chen",
        "preferences": ["Electronics", "Fitness"],
        "history": [1, 5, 7, 12, 10],   # product ids viewed/purchased
        "cart": [5, 12],
        "age_group": "25-34",
    },
    "bob": {
        "name": "Bob Rivera",
        "preferences": ["Kitchen", "Home"],
        "history": [2, 4, 8, 13, 15, 16],
        "cart": [4, 8],
        "age_group": "35-44",
    },
    "carol": {
        "name": "Carol Park",
        "preferences": ["Fitness", "Footwear"],
        "history": [3, 11, 14, 8, 19],
        "cart": [14],
        "age_group": "25-34",
    },
    "dave": {
        "name": "Dave Wilson",
        "preferences": ["Electronics", "Furniture"],
        "history": [9, 10, 18, 12, 5],
        "cart": [18],
        "age_group": "35-44",
    },
    "eva": {
        "name": "Eva Martinez",
        "preferences": ["Home", "Accessories", "Kitchen"],
        "history": [4, 6, 15, 20, 2],
        "cart": [6, 20],
        "age_group": "18-24",
    },
}

def get_products_df() -> pd.DataFrame:
    """Return products as a clean DataFrame."""
    df = pd.DataFrame(PRODUCTS)
    df["conversion_rate"] = (df["purchases"] / df["views"] * 100).round(2)
    df["tags_str"] = df["tags"].apply(lambda x: ", ".join(x))
    return df

def get_user_item_matrix() -> pd.DataFrame:
    """Build a binary user-item interaction matrix."""
    users = list(USER_PROFILES.keys())
    product_ids = [p["id"] for p in PRODUCTS]
    matrix = pd.DataFrame(0, index=users, columns=product_ids)
    for uid, profile in USER_PROFILES.items():
        for pid in profile["history"]:
            matrix.loc[uid, pid] = 1
        for pid in profile["cart"]:
            matrix.loc[uid, pid] = min(matrix.loc[uid, pid] + 0.5, 1.0)
    return matrix