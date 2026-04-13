"""
app.py — Flask REST API for the E-Commerce Recommendation System.

Endpoints:
  GET  /api/products                     → full product catalogue
  GET  /api/products/<id>                → single product
  GET  /api/users                        → list of users
  GET  /api/users/<user_id>              → user profile
  GET  /api/recommend/<user_id>          → hybrid recs (default)
  GET  /api/recommend/<user_id>?algo=cf  → collaborative
  GET  /api/recommend/<user_id>?algo=cb  → content-based
  GET  /api/trending                     → trending products
  GET  /api/similar-users/<user_id>      → cosine similarity scores
  GET  /api/analytics/summary            → category + KPI summary
  GET  /api/analytics/top-products       → top by conversion rate
  POST /api/cart/<user_id>/add           → add item to cart  {product_id}
  GET  /                                 → serves the Web UI
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template_string, abort
from flask import make_response
from data import PRODUCTS, USER_PROFILES, get_products_df
from algorithms import (
    CollaborativeFilter, ContentBasedFilter,
    TrendingRecommender, HybridRecommender,
)

app = Flask(__name__)

# ── Init engines once ──────────────────────────────────────────────────────────
cf_engine     = CollaborativeFilter(k_neighbours=3)
cb_engine     = ContentBasedFilter()
trend_engine  = TrendingRecommender()
hybrid_engine = HybridRecommender(cf_weight=1.2, cb_weight=1.0)

# In-memory cart state (per session; resets on restart)
_carts = {uid: list(p.get("cart", [])) for uid, p in USER_PROFILES.items()}

def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/products")
def get_products():
    category = request.args.get("category")
    search   = request.args.get("q", "").lower()
    result   = PRODUCTS
    if category:
        result = [p for p in result if p["category"] == category]
    if search:
        result = [p for p in result if search in p["name"].lower()
                  or any(search in t for t in p["tags"])]
    return _cors(jsonify({"count": len(result), "products": result}))


@app.route("/api/products/<int:pid>")
def get_product(pid):
    p = next((p for p in PRODUCTS if p["id"] == pid), None)
    if not p:
        abort(404)
    conv = round(p["purchases"] / p["views"] * 100, 2)
    return _cors(jsonify({**p, "conversion_rate": conv}))


@app.route("/api/categories")
def get_categories():
    cats = sorted(set(p["category"] for p in PRODUCTS))
    return _cors(jsonify({"categories": cats}))


# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/users")
def get_users():
    users = [
        {"id": uid, "name": p["name"],
         "preferences": p["preferences"],
         "history_count": len(p["history"]),
         "cart_count": len(_carts.get(uid, []))}
        for uid, p in USER_PROFILES.items()
    ]
    return _cors(jsonify({"users": users}))


@app.route("/api/users/<user_id>")
def get_user(user_id):
    if user_id not in USER_PROFILES:
        abort(404)
    profile = USER_PROFILES[user_id]
    products_map = {p["id"]: p for p in PRODUCTS}
    history_products = [products_map[i] for i in profile["history"] if i in products_map]
    cart_products    = [products_map[i] for i in _carts.get(user_id, []) if i in products_map]
    return _cors(jsonify({
        "id":              user_id,
        "name":            profile["name"],
        "preferences":     profile["preferences"],
        "age_group":       profile.get("age_group"),
        "history":         history_products,
        "cart":            cart_products,
    }))


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/recommend/<user_id>")
def recommend(user_id):
    if user_id not in USER_PROFILES:
        abort(404)
    algo   = request.args.get("algo", "hybrid").lower()
    top_n  = min(int(request.args.get("n", 6)), 20)

    if algo == "cf":
        recs      = cf_engine.recommend(user_id, top_n=top_n)
        score_key = "cf_score"
        algo_name = "Collaborative Filtering"
    elif algo == "cb":
        recs      = cb_engine.recommend(user_id, top_n=top_n)
        score_key = "cb_score"
        algo_name = "Content-Based Filtering"
    else:
        recs      = hybrid_engine.recommend(user_id, top_n=top_n)
        score_key = "hybrid_score"
        algo_name = "Hybrid"

    # normalise score key name for uniform API response
    for r in recs:
        r["score"] = r.pop(score_key, 0)
        r["conversion_rate"] = round(r["purchases"] / r["views"] * 100, 2)

    return _cors(jsonify({
        "user_id":   user_id,
        "algorithm": algo_name,
        "count":     len(recs),
        "recommendations": recs,
    }))


@app.route("/api/trending")
def trending():
    top_n = min(int(request.args.get("n", 6)), 20)
    recs  = trend_engine.recommend(top_n=top_n)
    for r in recs:
        r["score"] = r.pop("trend_score", 0)
        r["conversion_rate"] = round(r["purchases"] / r["views"] * 100, 2)
    return _cors(jsonify({"algorithm": "Trending", "count": len(recs), "recommendations": recs}))


@app.route("/api/similar-users/<user_id>")
def similar_users(user_id):
    if user_id not in USER_PROFILES:
        abort(404)
    sims = cf_engine.get_similar_users(user_id)
    result = [{"user_id": uid, "name": USER_PROFILES[uid]["name"],
               "similarity": round(float(score), 4)}
              for uid, score in sims.items()]
    return _cors(jsonify({"user_id": user_id, "similar_users": result}))


# ─────────────────────────────────────────────────────────────────────────────
# CART
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/cart/<user_id>/add", methods=["POST"])
def add_to_cart(user_id):
    if user_id not in USER_PROFILES:
        abort(404)
    data = request.get_json(silent=True) or {}
    pid  = data.get("product_id")
    if not pid or not any(p["id"] == pid for p in PRODUCTS):
        return _cors(jsonify({"error": "Invalid product_id"})), 400
    cart = _carts.setdefault(user_id, [])
    if pid not in cart:
        cart.append(pid)
    return _cors(jsonify({"user_id": user_id, "cart": cart, "cart_count": len(cart)}))


@app.route("/api/cart/<user_id>")
def get_cart(user_id):
    if user_id not in USER_PROFILES:
        abort(404)
    cart = _carts.get(user_id, [])
    products_map = {p["id"]: p for p in PRODUCTS}
    items = [products_map[i] for i in cart if i in products_map]
    total = sum(p["price"] for p in items)
    return _cors(jsonify({"user_id": user_id, "items": items, "total": total, "count": len(items)}))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analytics/summary")
def analytics_summary():
    df   = get_products_df()
    cats = df.groupby("category").agg(
        products=("id", "count"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
        total_revenue=("purchases", lambda x: int((x * df.loc[x.index, "price"]).sum())),
        avg_conversion=("conversion_rate", "mean"),
    ).reset_index().to_dict(orient="records")

    kpis = {
        "total_products":  len(PRODUCTS),
        "total_users":     len(USER_PROFILES),
        "total_revenue":   int(sum(p["price"] * p["purchases"] for p in PRODUCTS)),
        "avg_rating":      round(sum(p["rating"] for p in PRODUCTS) / len(PRODUCTS), 2),
        "avg_conversion":  round(sum(p["purchases"] / p["views"] for p in PRODUCTS) / len(PRODUCTS) * 100, 2),
    }
    return _cors(jsonify({"kpis": kpis, "by_category": cats}))


@app.route("/api/analytics/top-products")
def top_products():
    n    = min(int(request.args.get("n", 10)), 20)
    df   = get_products_df()
    top  = df.nlargest(n, "conversion_rate")[
        ["id","name","category","price","rating","reviews","views","purchases","conversion_rate"]
    ].to_dict(orient="records")
    return _cors(jsonify({"top_products": top}))


# ─────────────────────────────────────────────────────────────────────────────
# WEB UI  (served from the same Flask app)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        html = f.read()
    return html


if __name__ == "__main__":
    print("\n  🛍️  RecoEngine API starting…")
    print("  Web UI  → http://127.0.0.1:5000")
    print("  API     → http://127.0.0.1:5000/api/products\n")
    app.run(debug=True, port=5000)