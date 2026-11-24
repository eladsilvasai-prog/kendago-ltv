"""Data audit for the Kendago 2-year LTV task. Read-only diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/shaysilvas/Downloads/candidate_pack/data")
EXPORT_END = pd.Timestamp("2026-10-31 23:59:59")

orders = pd.read_csv(DATA / "orders.csv")
tx = pd.read_csv(DATA / "transactions.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
tx["ts"] = pd.to_datetime(tx["ts"])

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}")


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


section("SHAPE AND RANGE")
print("orders", orders.shape, orders["order_ts"].min(), orders["order_ts"].max())
print("tx", tx.shape, tx["ts"].min(), tx["ts"].max())
print("products", products.shape)
print("order is_test", orders["is_test"].value_counts(dropna=False).to_dict())
print("tx is_test", tx["is_test"].value_counts(dropna=False).to_dict())
print("dup order_id", orders["order_id"].duplicated().sum())
print("dup customer_id", orders["customer_id"].duplicated().sum())
print("orders missing from tx", (~orders["order_id"].isin(tx["order_id"])).sum())
print("tx order_id missing from orders", (~tx["order_id"].isin(orders["order_id"])).sum())

section("PRODUCTS")
print(products.sort_values(["brand", "product_group", "product_id", "geo"]).to_string(index=False))
print("product_id nunique", products["product_id"].nunique(), "rows", len(products))
print(products.groupby("product_id").size().sort_values(ascending=False).head(20))

section("ORDERS CATEGORICAL")
for col in [
    "brand",
    "funnel",
    "funnel_version",
    "main_product_id",
    "discount_pct",
    "traffic_source",
    "country",
    "device_type",
    "os",
    "device_tier",
    "payment_method",
    "processor",
    "email_domain_group",
    "gender",
    "age_group",
    "activity_level",
    "main_goal",
    "ab_bucket",
]:
    print(f"\n-- {col}")
    print(orders[col].value_counts(dropna=False).head(20).to_string())

section("NULL RATES")
null_rate = orders.isna().mean().sort_values(ascending=False)
print(null_rate[null_rate > 0].to_string())
print("empty campaign", (orders["campaign_id"].fillna("").eq("") | orders["campaign_id"].isna()).sum())

section("TX TYPES")
print(pd.crosstab(tx["type"], tx["status"]))
print("amount describe by type/status")
print(tx.groupby(["type", "status"])["amount_usd"].agg(["count", "sum", "mean", "min", "max"]))

section("JOIN PLAN")
prod = (
    products.sort_values("added_on")
    .drop_duplicates("product_id", keep="last")[
        ["product_id", "product_name", "brand", "product_group", "plan", "billing_cycle_days", "renewal_price_usd"]
    ]
)
print(prod.to_string(index=False))
orders = orders.merge(prod, left_on="main_product_id", right_on="product_id", how="left", suffixes=("", "_prod"))
print("unmatched main product", orders["product_group"].isna().sum())
print(orders.groupby(["brand", "plan", "main_product_id", "intro_price_usd"]).size())

section("TEST CONSISTENCY")
test_orders = set(orders.loc[orders["is_test"] == 1, "order_id"])
tx_test_orders = set(tx.loc[tx["is_test"] == 1, "order_id"])
print("test orders", len(test_orders), "tx-flagged orders", len(tx_test_orders))
print("in tx not orders", len(tx_test_orders - test_orders))
print("in orders not tx", len(test_orders - tx_test_orders))

# Work on non-test
o = orders[orders["is_test"] == 0].copy()
t = tx[(tx["is_test"] == 0) & (tx["order_id"].isin(o["order_id"]))].copy()
t = t.merge(prod, on="product_id", how="left", suffixes=("", "_p"))
print("tx unmatched product", t["product_group"].isna().sum())

section("PAYMENT NUM BY GROUP")
print(t.groupby(["product_group", "plan", "type", "status"])["payment_num"].agg(["count", "min", "max"]))

section("BILLING LAG days from order to charge")
t = t.merge(o[["order_id", "order_ts", "main_product_id", "intro_price_usd", "discount_pct"]], on="order_id", how="left")
t["day"] = (t["ts"] - t["order_ts"]).dt.total_seconds() / 86400
charges = t[(t["type"] == "charge") & (t["status"] == "approved")]
print(
    charges.groupby(["product_group", "plan", "payment_num"])["day"]
    .agg(["count", "median", "mean", "min", "max"])
    .head(80)
    .to_string()
)

section("ZERO AMOUNT CHARGES")
print(charges[charges["amount_usd"] == 0].groupby(["product_group", "payment_num"]).size())

section("REFUND / CB LAG")
rev = t[t["type"].isin(["refund", "chargeback"])].copy()
print(rev.groupby(["type", "status"]).size())
print(rev.groupby(["type", "status"])["day"].describe())
print("reversal payment_num vs charges")
print(rev.groupby(["product_group", "type", "status"])["payment_num"].value_counts().head(40))

section("MULTIPLE CHARGES SAME PAYMENT")
key = ["order_id", "product_id", "payment_num", "type", "status"]
print("dup charge keys", charges.duplicated(["order_id", "product_id", "payment_num"]).sum())
print(charges.groupby(["order_id", "product_id", "payment_num"]).size().value_counts().head())

section("MONTHLY COHORTS")
o["month"] = o["order_ts"].dt.to_period("M")
print(o.groupby(["month", "brand", "plan"]).size().unstack(fill_value=0).to_string())

section("OCTOBER 2026")
oct_mask = o["order_ts"] >= "2026-10-01"
print("oct orders", oct_mask.sum(), "campaigns", o.loc[oct_mask, "campaign_id"].nunique())
print(o.loc[oct_mask, "traffic_source"].value_counts())
print("organic oct", ((o["traffic_source"] == "organic") & oct_mask).sum())
print("cpa describe oct paid")
paid = o[oct_mask & (o["traffic_source"] != "organic")]
print(paid["acquisition_cost_usd"].describe())

# Net cash helper
approved = t[t["status"] == "approved"].copy()
sign = np.where(approved["type"].eq("charge"), 1.0, np.where(approved["type"].isin(["refund", "chargeback"]), -1.0, np.nan))
approved["signed"] = approved["amount_usd"] * sign
print("non charge/refund/cb approved", approved["signed"].isna().sum())
approved = approved.dropna(subset=["signed"])

section("OBSERVED NET BY HORIZON for orders aged enough")
# attach age at export
o["age_days"] = (EXPORT_END - o["order_ts"]).dt.total_seconds() / 86400
cash = approved.groupby("order_id")["signed"].sum().rename("net_all")
o = o.merge(cash, on="order_id", how="left")
o["net_all"] = o["net_all"].fillna(0)

def net_within(days):
    sub = approved[approved["day"] <= days]
    return sub.groupby("order_id")["signed"].sum()

for horizon in [0, 7, 30, 90, 180, 364]:
    col = f"net_{horizon}"
    s = net_within(horizon)
    o[col] = o["order_id"].map(s).fillna(0)
    mature = o[o["age_days"] >= horizon + 1]
    print(f"\nhorizon {horizon}d  mature orders {len(mature):,}")
    print(mature.groupby(["brand", "plan"])[col].agg(["count", "mean", "median"]).to_string())

section("MARGINAL NET BY PAYMENT NUM main subscription, mature")
main = approved[approved["product_group"] == "main_subscription"]
# scheduled day approx from median of approved charges
sched = (
    charges[charges["product_group"] == "main_subscription"]
    .groupby(["plan", "payment_num"])["day"]
    .median()
    .rename("sched_day")
)
print("schedule\n", sched.to_string())

main_net = main.groupby(["order_id", "plan", "payment_num"], as_index=False)["signed"].sum()
# need plan on orders
for plan, g in o.groupby("plan"):
    print(f"\n#### PLAN {plan}")
    sched_p = sched.loc[plan] if plan in sched.index.get_level_values(0) else None
    mx = int(main_net.loc[main_net["plan"] == plan, "payment_num"].max())
    rows = []
    for k in range(1, mx + 1):
        sd = float(sched.loc[(plan, k)]) if (plan, k) in sched.index else np.nan
        # mature if age > sched + 45 days for refund lag, else age > sched + 7
        lag = 45 if k <= 3 else 21
        eligible = g[g["age_days"] >= sd + lag] if pd.notna(sd) else g.iloc[0:0]
        nets = main_net[(main_net["plan"] == plan) & (main_net["payment_num"] == k)]
        m = eligible[["order_id"]].merge(nets[["order_id", "signed"]], on="order_id", how="left")
        m["signed"] = m["signed"].fillna(0)
        rows.append(
            {
                "k": k,
                "sched": sd,
                "n": len(m),
                "mean_net": m["signed"].mean() if len(m) else np.nan,
                "hit_rate": (m["signed"] > 0).mean() if len(m) else np.nan,
                "mean_if_hit": m.loc[m["signed"] > 0, "signed"].mean() if (m["signed"] > 0).any() else np.nan,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

section("UPSELL ATTACH day 0 and ever, by brand")
# day of attach
first_upsell = (
    charges[charges["product_group"].isin(["checkout_addon", "upsell_one_time", "upsell_subscription"])]
    .groupby(["order_id", "product_group"])["day"]
    .min()
    .unstack()
)
o2 = o.merge(first_upsell, on="order_id", how="left")
for col in ["checkout_addon", "upsell_one_time", "upsell_subscription"]:
    if col not in o2:
        continue
    print(col, "ever", o2[col].notna().mean(), "within 1 day", (o2[col] <= 1).mean())
print(o2.groupby("brand")[["checkout_addon", "upsell_one_time", "upsell_subscription"]].apply(lambda d: d.notna().mean()))

section("UPSELL SUB RENEWAL CURVE")
us = approved[approved["product_group"] == "upsell_subscription"]
print(
    charges[charges["product_group"] == "upsell_subscription"]
    .groupby("payment_num")["day"]
    .agg(["count", "median", "mean"])
    .head(20)
)
# among orders who started upsell (payment 1) and are old enough
starters = charges[(charges["product_group"] == "upsell_subscription") & (charges["payment_num"] == 1)]["order_id"].unique()
print("upsell starters", len(starters))
us_net = us.groupby(["order_id", "payment_num"])["signed"].sum()
sched_u = charges[charges["product_group"] == "upsell_subscription"].groupby("payment_num")["day"].median()
print("upsell sched\n", sched_u.to_string())
base = o[o["order_id"].isin(starters)][["order_id", "age_days", "brand"]]
for k, sd in sched_u.items():
    elig = base[base["age_days"] >= sd + (40 if k <= 3 else 14)]
    m = elig[["order_id"]].copy()
    m["signed"] = m["order_id"].map(us_net.xs(k, level="payment_num")).fillna(0) if k in us_net.index.get_level_values("payment_num") else 0
    # safer map
    if k in us_net.index.get_level_values(1):
        ser = us_net.xs(k, level=1)
        m["signed"] = m["order_id"].map(ser).fillna(0)
    else:
        m["signed"] = 0
    print(f"k={k} day={sd:.1f} n={len(m)} mean={m['signed'].mean():.3f} hit={(m['signed']>0).mean():.3f}")

section("SEGMENT SPREAD on 180d net, mature orders")
mature = o[o["age_days"] >= 181].copy()
print("n mature 180", len(mature), "mean", mature["net_180"].mean())
for col in ["brand", "plan", "traffic_source", "country", "discount_pct", "device_type", "payment_method", "gender", "age_group", "funnel", "funnel_version", "os", "device_tier", "email_domain_group", "activity_level", "main_goal", "ab_bucket"]:
    g = mature.groupby(col)["net_180"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    print(f"\n-- {col}")
    print(g.head(15).to_string())

section("PLAN MIX BY SOURCE among 180d mature")
print(pd.crosstab(mature["traffic_source"], mature["plan"], normalize="index").round(3))
print("mean net_180 by source x plan")
print(mature.pivot_table("net_180", "traffic_source", "plan", aggfunc="mean").round(2))
print("count")
print(mature.pivot_table("net_180", "traffic_source", "plan", aggfunc="count"))

section("DISCOUNT x PLAN net_180")
print(mature.pivot_table("net_180", ["brand", "plan"], "discount_pct", aggfunc=["mean", "count"]).round(2))

section("APP WORKOUTS vs AGE")
sub = o[o["brand"] == "FlexHer"]
print(sub[["app_workouts_completed", "age_days", "net_all"]].corr())
print(sub.groupby(pd.cut(sub["age_days"], [0, 30, 90, 180, 400]))["app_workouts_completed"].mean())

section("SAME DAY REFUND RATE")
day0_rev = rev[(rev["day"] <= 1) & (rev["status"] == "approved")]
print("reversals within 1d", len(day0_rev), "of orders", day0_rev["order_id"].nunique())
print(rev[rev["status"] == "approved"].groupby(pd.cut(rev.loc[rev["status"] == "approved", "day"], [-1, 1, 7, 14, 30, 60, 120, 400])).size())

section("PENDING AND DECLINED SHARE")
print(t.groupby(["type", "status"]).size())
print("declined amount sum", t.loc[t["status"] == "declined", "amount_usd"].sum())

section("INTRO vs FIRST CHARGE")
p1 = charges[(charges["product_group"] == "main_subscription") & (charges["payment_num"] == 1)][
    ["order_id", "amount_usd"]
]
chk = o.merge(p1, on="order_id", how="left")
print("missing p1 charge", chk["amount_usd"].isna().mean())
print("intro != charge", (chk["amount_usd"] - chk["intro_price_usd"]).abs().gt(0.01).mean())
print(chk.loc[(chk["amount_usd"] - chk["intro_price_usd"]).abs() > 0.01, ["intro_price_usd", "amount_usd", "plan"]].head(10))

section("CPA CONSTANT WITHIN CAMPAIGN-DAY?")
o["order_date"] = o["order_ts"].dt.date
paid = o[o["traffic_source"] != "organic"]
nunq = paid.groupby(["campaign_id", "order_date"])["acquisition_cost_usd"].nunique()
print("campaign-days with >1 cpa", (nunq > 1).sum(), "of", len(nunq))
print(paid.groupby("campaign_id")["acquisition_cost_usd"].agg(["count", "mean", "min", "max"]).sort_values("count", ascending=False).head(15))

section("FUNNEL VERSION OVER TIME")
print(pd.crosstab(o["month"], o["funnel_version"]))
print(pd.crosstab(o["month"], [o["brand"], o["funnel"]]))

section("QUIZ MISSINGNESS")
for col in ["age", "gender", "diet_type", "event_weeks", "device_tier", "us_state", "campaign_id"]:
    print(col, o[col].isna().mean())

print("\nDONE")
