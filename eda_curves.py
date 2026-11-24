"""Focused billing curves and a few data-quality checks."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/shaysilvas/Downloads/candidate_pack/data")
EXPORT_END = pd.Timestamp("2026-10-31 23:59:59")

orders = pd.read_csv(DATA / "orders.csv", low_memory=False)
tx = pd.read_csv(DATA / "transactions.csv")
products = pd.read_csv(DATA / "products.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
tx["ts"] = pd.to_datetime(tx["ts"])

prod = (
    products.sort_values("added_on")
    .drop_duplicates("product_id", keep="last")[
        ["product_id", "product_name", "product_group", "plan", "billing_cycle_days", "renewal_price_usd"]
    ]
)
o = orders[orders["is_test"] == 0].copy()
o = o.merge(prod, left_on="main_product_id", right_on="product_id", how="left")
o["age_days"] = (EXPORT_END - o["order_ts"]).dt.total_seconds() / 86400

t = tx[(tx["is_test"] == 0) & (tx["order_id"].isin(o["order_id"]))].copy()
t = t.merge(prod, on="product_id", how="left", suffixes=("", "_cat"))
t = t.merge(o[["order_id", "order_ts"]], on="order_id", how="left")
t["day"] = (t["ts"] - t["order_ts"]).dt.total_seconds() / 86400

approved = t[t["status"] == "approved"].copy()
sign = np.where(approved["type"].eq("charge"), 1.0, -1.0)
approved["signed"] = approved["amount_usd"] * sign

print("=== source by month ===")
o["month"] = o["order_ts"].dt.to_period("M").astype(str)
print(pd.crosstab(o["month"], o["traffic_source"]))

print("\n=== amount of addon and otp ===")
for g in ["checkout_addon", "upsell_one_time", "upsell_subscription", "main_subscription"]:
    sub = approved[(approved["product_group"] == g) & (approved["type"] == "charge")]
    print(g, sub.groupby("payment_num")["amount_usd"].agg(["count", "mean", "min", "max"]).head(5))

print("\n=== first attach day quantiles ===")
ch = approved[approved["type"] == "charge"]
for g in ["checkout_addon", "upsell_one_time", "upsell_subscription"]:
    first = ch[ch["product_group"] == g].groupby("order_id")["day"].min()
    print(g, "attach rate", len(first) / len(o), "quantiles", first.quantile([0.5, 0.9, 0.99, 1]).to_dict())

print("\n=== dup approved charges same payment ===")
ch = approved[approved["type"] == "charge"]
vc = ch.groupby(["order_id", "product_id", "payment_num"]).size()
print(vc.value_counts().head())

print("\n=== orders missing main p1 approved ===")
p1 = ch[(ch["product_group"] == "main_subscription") & (ch["payment_num"] == 1)]["order_id"]
missing = o[~o["order_id"].isin(p1)]
print(len(missing))
print(missing[["order_id", "order_ts", "plan", "intro_price_usd", "is_test"]].head(15).to_string(index=False))

print("\n=== MAIN curves brand x plan x discount ===")
main = approved[approved["product_group"] == "main_subscription"]
# theoretical schedule
def sched_day(plan, k):
    k = int(k)
    if plan == "1W":
        return 0.0 if k == 1 else 7.0 + 28.0 * (k - 2)
    if plan == "4W":
        return 28.0 * (k - 1)
    if plan == "12W":
        return 84.0 * (k - 1)
    return np.nan

net = main.groupby(["order_id", "payment_num"])["signed"].sum()

for plan in ["1W", "4W", "12W"]:
    print(f"\n----- {plan} -----")
    g = o[o["plan"] == plan]
    rows = []
    max_k = 30 if plan != "12W" else 12
    for k in range(1, max_k + 1):
        sd = sched_day(plan, k)
        if sd > 364:
            break
        elig = g[g["age_days"] >= sd + 40]
        ser = net.xs(k, level=1) if k in net.index.get_level_values(1) else pd.Series(dtype=float)
        vals = elig["order_id"].map(ser).fillna(0)
        rows.append((k, sd, len(elig), float(vals.mean()), float((vals > 0).mean()), float(vals.sum())))
    df = pd.DataFrame(rows, columns=["k", "day", "n", "mean", "hit", "total"])
    df["ratio"] = df["mean"] / df["mean"].shift(1)
    print(df.round(3).to_string(index=False))

print("\n=== 4W discount split early k ===")
g = o[o["plan"] == "4W"]
for disc, gg in g.groupby("discount_pct"):
    print("discount", disc, "n", len(gg))
    for k in range(1, 9):
        sd = sched_day("4W", k)
        elig = gg[gg["age_days"] >= sd + 40]
        ser = net.xs(k, level=1)
        vals = elig["order_id"].map(ser).fillna(0)
        print(f"  k={k} n={len(elig)} mean={vals.mean():.2f} hit={ (vals>0).mean():.3f}")

print("\n=== brand split 4W k=1..6 ===")
g = o[o["plan"] == "4W"]
for brand, gg in g.groupby("brand"):
    print(brand)
    for k in range(1, 7):
        sd = sched_day("4W", k)
        elig = gg[gg["age_days"] >= sd + 40]
        ser = net.xs(k, level=1)
        vals = elig["order_id"].map(ser).fillna(0)
        print(f"  k={k} n={len(elig)} mean={vals.mean():.2f} hit={(vals>0).mean():.3f}")

print("\n=== addon/otp net given purchase, age>=45 ===")
for gname in ["checkout_addon", "upsell_one_time"]:
    sub = approved[approved["product_group"] == gname]
    buyers = sub.loc[sub["type"] == "charge", "order_id"].unique()
    aged = o[o["order_id"].isin(buyers) & (o["age_days"] >= 45)]
    nets = sub.groupby("order_id")["signed"].sum()
    print(gname, "buyers", len(buyers), "mature", len(aged), "mean net", aged["order_id"].map(nets).mean())

print("\n=== upsell curve by brand, conditional on start ===")
us = approved[approved["product_group"] == "upsell_subscription"]
starters = set(us.loc[(us["type"] == "charge") & (us["payment_num"] == 1), "order_id"])
us_net = us.groupby(["order_id", "payment_num"])["signed"].sum()
base = o[o["order_id"].isin(starters)]
for brand, gg in base.groupby("brand"):
    print(brand, "starters", len(gg))
    for k in range(1, 16):
        sd = 0 if k == 1 else 7 + 28 * (k - 2)
        elig = gg[gg["age_days"] >= sd + 40]
        if k in us_net.index.get_level_values(1):
            ser = us_net.xs(k, level=1)
            vals = elig["order_id"].map(ser).fillna(0)
        else:
            vals = pd.Series(0, index=elig.index)
        if len(elig) == 0:
            continue
        print(f"  k={k} day={sd} n={len(elig)} mean={vals.mean():.3f} hit={(vals>0).mean():.3f}")

print("\n=== negative net orders share mature 180 ===")
# quick
print("age describe", o["age_days"].describe())
print("oct n", (o["order_ts"] >= "2026-10-01").sum())
print("campaigns oct", o.loc[o["order_ts"] >= "2026-10-01", "campaign_id"].nunique(dropna=True))
print(o.loc[o["order_ts"] >= "2026-10-01"].groupby(["brand", "traffic_source"]).size())
