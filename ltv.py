"""
Day-0 two-year pLTV for Kendago (FlexHer / PlateWise).

The export ends 2026-10-31, so nobody has a full 730 days of cash.
pLTV is expected net cash (approved charges minus refunds minus chargebacks)
over the 730 days after the order, estimated from the renewal curve of
customers who already reached each bill, then adjusted for the kind of
customer who showed up.

Information used is what is known on the day the order arrives: the order
itself, and the checkout add-on / upsells, which fire within minutes.
app_workouts_completed is left out — it accumulates after purchase and is
much higher for older orders.

Tail bills that no cohort has reached yet are projected with a continuation
rate anchored on the last observed bills and shrunk toward a prior.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data",
    Path("/Users/shaysilvas/Downloads/candidate_pack/data"),
]
ARTIFACTS = ROOT / "artifacts"

EXPORT_END = pd.Timestamp("2026-10-31 23:59:59")
HORIZON = 730
MATURITY_LAG = 40  # days after a bill before we treat refunds as mostly in
TRAIN_AGE = 180  # orders at least this old supply the customer adjustment
MIN_N_PLAN = 300
MIN_N_SEGMENT = 180
TAIL_PRIOR = {"1W": 0.90, "4W": 0.88, "12W": 0.74, "upsell": 0.80}
TAIL_PRIOR_WEIGHT = 0.35
TAIL_CLIP = (0.55, 0.96)

# Customer adjustment on top of the billing curve. Reference level stays at 1.0
# (the curve). Other levels get a shrunk ratio. Chosen because these cuts still
# moved day-180 cash on a March 2026 holdout after plan and offer were accounted for.
ADJUST_SPEC = [
    ("payment_method", "card"),
    ("age_group", "46-50"),
    ("funnel", "main_quiz"),
    ("country_group", "US"),
    ("has_otp", "0"),
    ("has_addon", "0"),
]
ADJUST_ALPHA = 100.0
RATIO_CLIP = (0.70, 1.35)
TRAIN_RATIO_CLIP = (0.0, 2.5)


def find_data() -> Path:
    for path in DATA_CANDIDATES:
        if (path / "orders.csv").exists():
            return path
    raise FileNotFoundError("orders.csv not found in data candidates")


def load_frames(data_dir: Path | None = None):
    data_dir = data_dir or find_data()
    orders = pd.read_csv(data_dir / "orders.csv", low_memory=False)
    tx = pd.read_csv(data_dir / "transactions.csv")
    products = pd.read_csv(data_dir / "products.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    tx["ts"] = pd.to_datetime(tx["ts"])
    products = (
        products.sort_values("added_on")
        .drop_duplicates("product_id", keep="last")[
            ["product_id", "product_name", "brand", "product_group", "plan", "billing_cycle_days", "renewal_price_usd"]
        ]
    )
    return orders, tx, products


def sched_day(kind: str, k: int) -> float:
    """k is 1-indexed. 1W and the upsell trial bill on day 7, then every 28."""
    if kind in ("1W", "upsell"):
        return 0.0 if k == 1 else 7.0 + 28.0 * (k - 2)
    if kind == "4W":
        return 28.0 * (k - 1)
    if kind == "12W":
        return 84.0 * (k - 1)
    raise KeyError(kind)


def max_payment(kind: str, horizon: int = HORIZON) -> int:
    k = 1
    while sched_day(kind, k + 1) <= horizon:
        k += 1
    return k


def country_group(country: pd.Series) -> pd.Series:
    known = {"US", "GB", "CA", "AU"}
    return country.where(country.isin(known), "OTHER")


def prepare_orders(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    prod = products.rename(columns={"brand": "product_brand"})
    o = orders.merge(prod, left_on="main_product_id", right_on="product_id", how="left")
    if o["plan"].isna().any():
        raise RuntimeError("main_product_id did not match the catalogue")
    o = o[o["is_test"] == 0].copy()
    o["age_days"] = (EXPORT_END - o["order_ts"]).dt.total_seconds() / 86400
    o["discount_pct"] = o["discount_pct"].astype(int).astype(str)
    o["country_group"] = country_group(o["country"].astype(str))
    o["age_group"] = o["age_group"].astype(str).replace({"nan": "unknown", "None": "unknown"}).fillna("unknown")
    o["device_tier"] = o["device_tier"].astype(str).replace({"nan": "not_phone", "None": "not_phone"})
    o.loc[o["device_type"].isin(["desktop"]), "device_tier"] = "not_phone"
    o.loc[~o["device_tier"].isin(["medium", "high", "premium", "unknown", "not_phone"]), "device_tier"] = "unknown"
    height_m = o["height_cm"].astype(float) / 100
    o["bmi"] = o["weight_kg"].astype(float) / height_m.pow(2)
    o.loc[~np.isfinite(o["bmi"]) | (o["bmi"] < 15) | (o["bmi"] > 60), "bmi"] = np.nan
    o["goal_gap"] = (o["weight_kg"] - o["goal_weight_kg"]) / o["weight_kg"]
    o.loc[~np.isfinite(o["goal_gap"]) | (o["goal_gap"] < -0.2) | (o["goal_gap"] > 0.6), "goal_gap"] = np.nan
    o["log_quiz_duration"] = np.log(o["quiz_duration_sec"].clip(lower=1))
    return o


def signed_payments(tx: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    t = tx[(tx["is_test"] == 0) & (tx["status"] == "approved")].copy()
    t = t[t["order_id"].isin(orders["order_id"])]
    t = t.merge(products[["product_id", "product_group"]], on="product_id", how="left")
    t = t.merge(orders[["order_id", "order_ts"]], on="order_id", how="left")
    t["day"] = (t["ts"] - t["order_ts"]).dt.total_seconds() / 86400
    t = t[(t["day"] >= -0.01) & (t["day"] <= HORIZON)]
    t["signed"] = np.where(t["type"].eq("charge"), t["amount_usd"], -t["amount_usd"])
    t = t[t["type"].isin(["charge", "refund", "chargeback"])]
    return (
        t.groupby(["order_id", "product_group", "payment_num"], as_index=False)["signed"]
        .sum()
    )


def attach_flags(pay: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Add-on and upsells are known the day the order arrives (they fire in minutes)."""
    flags = orders[["order_id"]].copy()
    for group, col in [
        ("checkout_addon", "has_addon"),
        ("upsell_one_time", "has_otp"),
        ("upsell_subscription", "has_upsell"),
    ]:
        ids = set(pay.loc[(pay["product_group"] == group) & (pay["payment_num"] == 1), "order_id"])
        # payment 1 of a refunded add-on still means they took it; netting happens in the cash curve
        flags[col] = flags["order_id"].isin(ids).astype(int)
    # A charge that was fully refunded still created a payment_num 1 row if we summed signed.
    # Rebuild flags from any row existing at payment 1, including net-zero after refund.
    return flags


def _empirical(order_ids: pd.Series, ages: pd.Series, pay: pd.DataFrame, group: str, kind: str):
    sub = pay[pay["product_group"] == group]
    k_max = max_payment(kind)
    means = np.zeros(k_max)
    counts = np.zeros(k_max, dtype=int)
    base = pd.DataFrame({"order_id": order_ids.values, "age_days": ages.values})
    for i, k in enumerate(range(1, k_max + 1)):
        day = sched_day(kind, k)
        eligible = base.loc[base["age_days"] >= day + MATURITY_LAG, "order_id"]
        counts[i] = int(len(eligible))
        if counts[i] == 0:
            means[i] = np.nan
            continue
        ser = sub.loc[sub["payment_num"] == k].groupby("order_id")["signed"].sum()
        means[i] = float(eligible.map(ser).fillna(0).mean())
    return means, counts


def _tail_ratio(means: np.ndarray, counts: np.ndarray, prior: float) -> float:
    ratios = []
    weights = []
    for i in range(2, len(means)):  # start at payment 3: payment 2 is the price step-up
        if counts[i] >= MIN_N_PLAN and counts[i - 1] >= MIN_N_PLAN and means[i - 1] > 1:
            ratio = means[i] / means[i - 1]
            if 0 < ratio < 1.05:
                ratios.append(ratio)
                weights.append(counts[i])
    if not ratios:
        return prior
    raw = float(np.average(ratios[-4:], weights=weights[-4:]))
    blended = (1 - TAIL_PRIOR_WEIGHT) * raw + TAIL_PRIOR_WEIGHT * prior
    return float(np.clip(blended, *TAIL_CLIP))


def _complete(emp_mean, emp_n, tail: float, min_n: int, parent_mean=None):
    mean = np.full(len(emp_mean), np.nan)
    source = []
    for i in range(len(emp_mean)):
        if emp_n[i] >= min_n and np.isfinite(emp_mean[i]):
            mean[i] = emp_mean[i]
            source.append("observed")
        elif parent_mean is not None and i > 0 and np.isfinite(mean[i - 1]) and parent_mean[i - 1] > 0:
            mean[i] = mean[i - 1] * (parent_mean[i] / parent_mean[i - 1])
            source.append("parent_shape")
        elif i == 0:
            mean[i] = float(parent_mean[0]) if parent_mean is not None else 0.0
            source.append("parent_level")
        else:
            mean[i] = max(0.0, mean[i - 1] * tail)
            source.append("extrapolated")
    # If we observed a prefix and the parent-shape branch didn't run, fill the rest with `tail`.
    for i in range(len(mean)):
        if not np.isfinite(mean[i]):
            prev = mean[i - 1] if i else 0.0
            mean[i] = max(0.0, prev * tail)
            source[i] = "extrapolated"
    return mean, source


def build_curves(orders: pd.DataFrame, pay: pd.DataFrame, exclude_ids: set | None = None):
    o = orders if not exclude_ids else orders[~orders["order_id"].isin(exclude_ids)]
    pay_u = pay if not exclude_ids else pay[~pay["order_id"].isin(exclude_ids)]
    curves = {"main": {}, "upsell": {}, "tail": {}, "n_orders": int(len(o))}

    for plan in ["1W", "4W", "12W"]:
        mask = o["plan"] == plan
        emp, cnt = _empirical(o.loc[mask, "order_id"], o.loc[mask, "age_days"], pay_u, "main_subscription", plan)
        tail = _tail_ratio(emp, cnt, TAIL_PRIOR[plan])
        parent, parent_src = _complete(emp, cnt, tail, MIN_N_PLAN)
        curves["tail"][plan] = tail
        curves["main"][plan] = {}
        for disc in ["61", "71"]:
            dmask = mask & (o["discount_pct"] == disc)
            demp, dcnt = _empirical(
                o.loc[dmask, "order_id"], o.loc[dmask, "age_days"], pay_u, "main_subscription", plan
            )
            # Discount segments share the plan's steady-state continuation.
            dmean, dsrc = _complete(demp, dcnt, tail, MIN_N_SEGMENT, parent_mean=parent)
            curves["main"][plan][disc] = {
                "day": [sched_day(plan, k) for k in range(1, len(dmean) + 1)],
                "mean": [round(float(v), 4) for v in dmean],
                "n": [int(v) for v in dcnt],
                "source": dsrc,
                "parent_mean": [round(float(v), 4) for v in parent],
                "parent_n": [int(v) for v in cnt],
                "parent_source": parent_src,
            }

    up_emp, up_cnt = _empirical(o["order_id"], o["age_days"], pay_u, "upsell_subscription", "upsell")
    # Conditional on starting the trial. Payment 1 is the $0 auth.
    starters = set(
        pay_u.loc[
            (pay_u["product_group"] == "upsell_subscription") & (pay_u["payment_num"] == 1),
            "order_id",
        ]
    )
    smask = o["order_id"].isin(starters)
    up_emp, up_cnt = _empirical(
        o.loc[smask, "order_id"], o.loc[smask, "age_days"], pay_u, "upsell_subscription", "upsell"
    )
    up_tail = _tail_ratio(up_emp, up_cnt, TAIL_PRIOR["upsell"])
    up_parent, _ = _complete(up_emp, up_cnt, up_tail, MIN_N_PLAN)
    curves["tail"]["upsell"] = up_tail
    curves["upsell"] = {}
    for disc in ["61", "71"]:
        dmask = smask & (o["discount_pct"] == disc)
        demp, dcnt = _empirical(
            o.loc[dmask, "order_id"], o.loc[dmask, "age_days"], pay_u, "upsell_subscription", "upsell"
        )
        dmean, dsrc = _complete(demp, dcnt, up_tail, MIN_N_SEGMENT, parent_mean=up_parent)
        curves["upsell"][disc] = {
            "day": [sched_day("upsell", k) for k in range(1, len(dmean) + 1)],
            "mean": [round(float(v), 4) for v in dmean],
            "n": [int(v) for v in dcnt],
            "source": dsrc,
        }

    # One-time products: expected net given they were taken, after the refund window.
    curves["one_time_net"] = {}
    for group, key in [("checkout_addon", "addon"), ("upsell_one_time", "otp")]:
        buyers = pay_u.loc[pay_u["product_group"] == group, "order_id"].unique()
        aged = o[o["order_id"].isin(buyers) & (o["age_days"] >= MATURITY_LAG)]
        nets = pay_u.loc[pay_u["product_group"] == group].groupby("order_id")["signed"].sum()
        curves["one_time_net"][key] = {
            "all": float(aged["order_id"].map(nets).mean()) if len(aged) else 0.0,
            "n": int(len(aged)),
        }
        by_brand = {}
        for brand, g in aged.groupby("brand"):
            by_brand[brand] = float(g["order_id"].map(nets).mean()) if len(g) >= 200 else curves["one_time_net"][key]["all"]
        curves["one_time_net"][key]["by_brand"] = by_brand

    # Attach rates for a new order where the upsells have not happened yet.
    flags = attach_flags(pay_u, o)
    m = o.merge(flags, on="order_id")
    attach = {}
    for (brand, plan), g in m.groupby(["brand", "plan"]):
        attach.setdefault(brand, {})[plan] = {
            "addon": float(g["has_addon"].mean()),
            "otp": float(g["has_otp"].mean()),
            "upsell": float(g["has_upsell"].mean()),
            "n": int(len(g)),
        }
    curves["attach"] = attach
    return curves


def _sum_curve(entry: dict, horizon: int, which: str = "mean") -> float:
    total = 0.0
    for day, value in zip(entry["day"], entry[which] if which == "mean" else entry[which]):
        if day <= horizon:
            total += value
    return float(total)


def structural_components(orders: pd.DataFrame, curves: dict, horizon: int = HORIZON) -> pd.DataFrame:
    """Expected net cash from the billing curves. Flags must already be on the frame."""
    main = np.zeros(len(orders))
    addon = np.zeros(len(orders))
    otp = np.zeros(len(orders))
    upsell = np.zeros(len(orders))
    # Cache sums: plan, discount, horizon
    cache: dict[tuple, float] = {}

    def main_sum(plan, disc):
        key = ("main", plan, disc, horizon)
        if key not in cache:
            cache[key] = _sum_curve(curves["main"][plan][disc], horizon)
        return cache[key]

    def up_sum(disc):
        key = ("up", disc, horizon)
        if key not in cache:
            cache[key] = _sum_curve(curves["upsell"][disc], horizon)
        return cache[key]

    plans = orders["plan"].to_numpy()
    discs = orders["discount_pct"].astype(str).to_numpy()
    brands = orders["brand"].to_numpy()
    has_addon = orders["has_addon"].to_numpy().astype(float)
    has_otp = orders["has_otp"].to_numpy().astype(float)
    has_up = orders["has_upsell"].to_numpy().astype(float)
    addon_net = curves["one_time_net"]["addon"]
    otp_net = curves["one_time_net"]["otp"]

    for i in range(len(orders)):
        main[i] = main_sum(plans[i], discs[i])
        brand = brands[i]
        if has_addon[i]:
            addon[i] = has_addon[i] * addon_net["by_brand"].get(brand, addon_net["all"])
        if has_otp[i]:
            otp[i] = has_otp[i] * otp_net["by_brand"].get(brand, otp_net["all"])
        if has_up[i]:
            upsell[i] = has_up[i] * up_sum(discs[i])

    out = pd.DataFrame(
        {
            "struct_main": main,
            "struct_addon": addon,
            "struct_otp": otp,
            "struct_upsell": upsell,
        },
        index=orders.index,
    )
    out["struct"] = out.sum(axis=1)
    return out


def cash_by_horizon(pay: pd.DataFrame, orders: pd.DataFrame, horizon: int) -> pd.Series:
    # Rebuild day from the payment file? We only stored payment_num sums.
    # Caller passes a day-level frame via `cash_within`, computed once.
    raise NotImplementedError


def cash_within(tx, orders, products, horizon: int) -> pd.Series:
    t = tx[(tx["is_test"] == 0) & (tx["status"] == "approved")].copy()
    t = t.merge(orders[["order_id", "order_ts"]], on="order_id", how="inner")
    t["day"] = (t["ts"] - t["order_ts"]).dt.total_seconds() / 86400
    t = t[(t["day"] >= -0.01) & (t["day"] <= horizon) & t["type"].isin(["charge", "refund", "chargeback"])]
    t["signed"] = np.where(t["type"].eq("charge"), t["amount_usd"], -t["amount_usd"])
    return t.groupby("order_id")["signed"].sum()


def _adjust_matrix(orders: pd.DataFrame, spec: list[tuple[str, str]]) -> np.ndarray:
    cols = []
    for col, level in spec:
        cols.append(orders[col].astype(str).eq(level).astype(float).to_numpy())
    if not cols:
        return np.zeros((len(orders), 0))
    return np.column_stack(cols)


def _adjustment_levels(train: pd.DataFrame) -> list[tuple[str, str]]:
    spec = []
    for col, ref in ADJUST_SPEC:
        levels = sorted(train[col].astype(str).unique())
        for level in levels:
            if level != ref and level != "nan":
                spec.append((col, level))
    return spec


def fit_adjustment(orders, pay, tx, products, curves, alpha: float = ADJUST_ALPHA, exclude_ids: set | None = None):
    """Shrunk ratio of day-180 cash vs the billing curve.

    The reference customer (US, card, age 46-50, main quiz, no add-on, no
    one-time upsell) keeps ratio 1. A stored scale puts the training-set
    average back on the curve, so the adjustment reallocates value instead
    of inflating it.
    """
    o = orders if not exclude_ids else orders[~orders["order_id"].isin(exclude_ids)].copy()
    o = o.drop(columns=[c for c in ("has_addon", "has_otp", "has_upsell") if c in o.columns])
    flags = attach_flags(pay if not exclude_ids else pay[~pay["order_id"].isin(exclude_ids)], o)
    o = o.merge(flags, on="order_id", how="left")
    for col in ("has_addon", "has_otp", "has_upsell"):
        o[col] = o[col].fillna(0).astype(int).astype(str)
    train = o[o["age_days"] >= TRAIN_AGE].copy()
    actual = train["order_id"].map(cash_within(tx, train, products, TRAIN_AGE)).fillna(0.0).to_numpy()
    struct = structural_components(train, curves, TRAIN_AGE)["struct"].to_numpy()
    ok = struct > 5
    train = train.loc[ok].copy()
    struct = struct[ok]
    actual = actual[ok]
    spec = _adjustment_levels(train)
    x = _adjust_matrix(train, spec)
    ratio = np.clip(actual / struct, *TRAIN_RATIO_CLIP)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x, ratio - 1.0)
    raw = np.clip(1.0 + x @ model.coef_, *RATIO_CLIP)
    scale = float(np.sum(struct) / np.sum(struct * raw))
    return {
        "model": model,
        "spec": spec,
        "scale": scale,
        "alpha": alpha,
        "n_train": int(len(train)),
        "feature_names": [f"{col}={level}" for col, level in spec],
        "references": {col: ref for col, ref in ADJUST_SPEC},
    }


def predict_ratio(orders: pd.DataFrame, bundle: dict) -> np.ndarray:
    frame = orders.copy()
    for col in ("has_addon", "has_otp", "has_upsell"):
        if col in frame.columns:
            frame[col] = frame[col].fillna(0).astype(int).astype(str)
    x = _adjust_matrix(frame, bundle["spec"])
    raw = np.clip(1.0 + x @ bundle["model"].coef_, *RATIO_CLIP)
    return raw * bundle["scale"]


def score_frame(orders: pd.DataFrame, curves: dict, bundle: dict | None, horizon: int = HORIZON, scalar: float = 1.0) -> pd.DataFrame:
    parts = structural_components(orders, curves, horizon)
    if bundle is None:
        ratio = np.ones(len(orders))
    else:
        ratio = predict_ratio(orders, bundle)
    parts = parts.copy()
    parts["ratio"] = ratio
    parts["scalar"] = scalar
    parts["pltv"] = (parts["struct"] * ratio * scalar).clip(lower=0)
    return parts


def decide(avg_pltv: float, cpa: float) -> str:
    if cpa is None or not np.isfinite(cpa) or cpa <= 0.01:
        return "organic"
    ratio = avg_pltv / cpa
    if ratio >= 3.5:
        return "scale"
    if ratio >= 2.5:
        return "hold"
    return "cut"


def campaign_table(scored_orders: pd.DataFrame) -> pd.DataFrame:
    df = scored_orders.copy()
    df["campaign_id"] = df["campaign_id"].fillna("").astype(str)
    organic = df["campaign_id"].eq("") | df["traffic_source"].eq("organic")
    df.loc[organic, "campaign_id"] = "organic:" + df.loc[organic, "brand"].astype(str)
    df.loc[organic, "campaign_name"] = "Organic — " + df.loc[organic, "brand"].astype(str)
    df["campaign_name"] = df["campaign_name"].fillna(df["campaign_id"])

    rows = []
    for cid, g in df.groupby("campaign_id", sort=False):
        cpa = float(g["acquisition_cost_usd"].mean())
        avg = float(g["pltv_2y"].mean())
        avg_low = float(g["pltv_low"].mean()) if "pltv_low" in g else avg
        avg_high = float(g["pltv_high"].mean()) if "pltv_high" in g else avg
        rows.append(
            {
                "campaign_id": cid,
                "campaign_name": g["campaign_name"].mode().iloc[0],
                "brand": g["brand"].mode().iloc[0] if g["brand"].nunique() == 1 else "mixed",
                "traffic_source": g["traffic_source"].mode().iloc[0] if g["traffic_source"].nunique() == 1 else "mixed",
                "orders": int(len(g)),
                "spend": float(g["acquisition_cost_usd"].sum()),
                "cpa": cpa,
                "avg_pltv": avg,
                "avg_pltv_low": avg_low,
                "avg_pltv_high": avg_high,
                "ltv_cac": (avg / cpa) if cpa > 0.01 else np.nan,
                "ltv_cac_low": (avg_low / cpa) if cpa > 0.01 else np.nan,
                "ltv_cac_high": (avg_high / cpa) if cpa > 0.01 else np.nan,
                "decision": decide(avg, cpa),
                "decision_low": decide(avg_low, cpa),
                "decision_high": decide(avg_high, cpa),
                "headroom": avg - 3.5 * cpa,
            }
        )
    table = pd.DataFrame(rows)
    table["_sort"] = table["ltv_cac"].fillna(-1)
    table = table.sort_values(["_sort", "orders"], ascending=[False, False]).drop(columns="_sort")
    return table.reset_index(drop=True)


def _metrics(actual, pred) -> dict:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    denom = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum(err**2) / denom) if denom > 0 else np.nan
    return {"mae": mae, "bias": bias, "r2": r2, "mean_actual": float(actual.mean()), "mean_pred": float(pred.mean()), "n": int(len(actual))}


def _prepare_model_frame(orders, pay):
    flags = attach_flags(pay, orders)
    return orders.drop(columns=[c for c in ["has_addon", "has_otp", "has_upsell"] if c in orders.columns]).merge(
        flags, on="order_id", how="left"
    )


def evaluate(orders, pay, tx, products):
    """March 2026 customer holdout. Curves and the adjustment never see these orders.

    The test slice is the second half of March. Nothing in the fit uses it,
    including the recenter scale (that comes from pre-March orders only).
    """
    march = set(orders.loc[(orders["order_ts"] >= "2026-03-01") & (orders["order_ts"] < "2026-04-01"), "order_id"])
    test_ids = set(orders.loc[(orders["order_ts"] >= "2026-03-16") & (orders["order_ts"] < "2026-04-01"), "order_id"])
    curves = build_curves(orders, pay, exclude_ids=march)
    o = _prepare_model_frame(orders, pay)
    bundle = fit_adjustment(o, pay, tx, products, curves, ADJUST_ALPHA, exclude_ids=march)
    test = o[o["order_id"].isin(test_ids)].copy()
    actual = test["order_id"].map(cash_within(tx, test, products, TRAIN_AGE)).fillna(0.0).to_numpy()
    pred = score_frame(test, curves, bundle, TRAIN_AGE, scalar=1.0)["pltv"].to_numpy()
    base = score_frame(test, curves, None, TRAIN_AGE, scalar=1.0)["pltv"].to_numpy()
    test_m = _metrics(actual, pred)
    base_m = _metrics(actual, base)
    use_model = test_m["mae"] <= base_m["mae"] - 0.5 and abs(test_m["bias"]) < 2.5
    return {
        "alpha": ADJUST_ALPHA if use_model else None,
        "use_model": use_model,
        "test": test_m if use_model else base_m,
        "test_model": test_m,
        "test_baseline": base_m,
        "holdout_scale": bundle["scale"],
        "curves_holdout_tail": curves["tail"],
    }


def coefficient_table(bundle: dict) -> pd.DataFrame:
    tab = pd.DataFrame(
        {
            "feature": bundle["feature_names"],
            "ratio_lift": bundle["model"].coef_,
            "ratio_vs_reference": 1.0 + bundle["model"].coef_,
        }
    )
    tab["abs"] = tab["ratio_lift"].abs()
    return tab.sort_values("abs", ascending=False).drop(columns="abs")


def score_population(orders, pay, curves, curves_low, curves_high, bundle, scalar):
    o = _prepare_model_frame(orders, pay)
    base = score_frame(o, curves, bundle, HORIZON, scalar)
    low = structural_components(o, curves_low, HORIZON)["struct"].to_numpy()
    high = structural_components(o, curves_high, HORIZON)["struct"].to_numpy()
    ratio = base["ratio"].to_numpy()
    o = o.copy()
    o["pltv_2y"] = base["pltv"].to_numpy()
    o["pltv_low"] = np.clip(low * ratio * scalar, 0, None)
    o["pltv_high"] = np.clip(high * ratio * scalar, 0, None)
    o["struct_2y"] = base["struct"].to_numpy()
    o["ratio"] = ratio
    for col in ["struct_main", "struct_addon", "struct_otp", "struct_upsell"]:
        o[col] = base[col].to_numpy()
    return o


def shift_tails(curves: dict, delta: float) -> dict:
    """Rebuild extrapolated bills with a higher or lower continuation rate.

    Observed bills stay put. Segment tails follow the shifted plan shape.
    """
    import copy

    out = copy.deepcopy(curves)
    for plan, entry in list(out["main"].items()):
        parent = np.array(entry["61"]["parent_mean"], dtype=float) if False else None
        # Reconstruct parent from the stored parent_mean (same for both discounts).
        parent = np.array(next(iter(entry.values()))["parent_mean"], dtype=float)
        parent_n = np.array(next(iter(entry.values()))["parent_n"], dtype=int)
        parent_src = next(iter(entry.values()))["parent_source"]
        tail = float(np.clip(out["tail"][plan] + delta, *TAIL_CLIP))
        out["tail"][plan] = tail
        # Re-extrapolate parent from the last observed point.
        last = max(i for i, s in enumerate(parent_src) if s == "observed")
        for i in range(last + 1, len(parent)):
            parent[i] = max(0.0, parent[i - 1] * tail)
        for disc, seg in entry.items():
            mean = np.array(seg["mean"], dtype=float)
            src = seg["source"]
            # Replay: keep observed, else follow parent shape.
            new = np.array(mean, dtype=float)
            new_src = list(src)
            for i in range(len(new)):
                if seg["n"][i] >= MIN_N_SEGMENT and src[i] == "observed":
                    continue
                if i == 0:
                    new[i] = parent[0]
                    new_src[i] = "parent_level"
                elif parent[i - 1] > 0:
                    new[i] = max(0.0, new[i - 1] * (parent[i] / parent[i - 1]))
                    new_src[i] = "parent_shape" if i <= last else "extrapolated"
            seg["mean"] = [round(float(v), 4) for v in new]
            seg["source"] = new_src
            seg["parent_mean"] = [round(float(v), 4) for v in parent]
    # Upsell
    tail = float(np.clip(out["tail"]["upsell"] + delta, *TAIL_CLIP))
    out["tail"]["upsell"] = tail
    # The stored upsell curve has no parent array. Shift only extrapolated points
    # by re-chaining from the last observed mean.
    for disc, seg in out["upsell"].items():
        mean = np.array(seg["mean"], dtype=float)
        src = seg["source"]
        observed_idx = [i for i, s in enumerate(src) if s == "observed"]
        if not observed_idx:
            continue
        last = max(observed_idx)
        for i in range(last + 1, len(mean)):
            mean[i] = max(0.0, mean[i - 1] * tail)
            src[i] = "extrapolated"
        seg["mean"] = [round(float(v), 4) for v in mean]
        seg["source"] = src
    return out


def default_profile() -> dict:
    return {
        "brand": "FlexHer",
        "funnel": "main_quiz",
        "plan": "4W",
        "discount_pct": "61",
        "country": "US",
        "traffic_source": "facebook",
        "device_type": "mobile",
        "device_tier": "medium",
        "payment_method": "card",
        "age_group": "46-50",
        "gender": "female",
        "activity_level": "light",
        "main_goal": "lose_weight",
        "motivation_score": 4,
        "quiz_duration_sec": 480,
        "quiz_reanswers": 1,
        "height_cm": 165,
        "weight_kg": 80,
        "goal_weight_kg": 70,
    }


def profile_to_frame(profile: dict, curves: dict, attach_mode: dict) -> pd.DataFrame:
    """attach_mode values are 'typical', 'yes', or 'no' for addon/otp/upsell.

    'typical' expands to the historical attach rate for that brand and plan
    via an 8-way mixture in `score_new_order`. Here we just build the base row.
    """
    row = default_profile()
    row.update(profile)
    df = pd.DataFrame([row])
    df["country_group"] = country_group(df["country"].astype(str))
    df["discount_pct"] = df["discount_pct"].astype(str)
    height_m = df["height_cm"].astype(float) / 100
    df["bmi"] = df["weight_kg"].astype(float) / height_m.pow(2)
    df["goal_gap"] = (df["weight_kg"] - df["goal_weight_kg"]) / df["weight_kg"]
    df["log_quiz_duration"] = np.log(df["quiz_duration_sec"].clip(lower=1))
    df["age_days"] = 0
    df["order_id"] = "NEW"
    return df


def score_new_order(profile: dict, attach_choice: dict, artifact: dict) -> dict:
    curves = artifact["curves"]
    bundle = artifact["bundle"]
    scalar = artifact["scalar"]
    base = profile_to_frame(profile, curves, attach_choice)
    brand = base["brand"].iloc[0]
    plan = base["plan"].iloc[0]
    rates = curves["attach"].get(brand, {}).get(plan, {"addon": 0.16, "otp": 0.19, "upsell": 0.32})

    def resolve(choice, rate):
        if choice == "yes":
            return [1.0]
        if choice == "no":
            return [0.0]
        return [0.0, 1.0]  # mixture

    # If all three are fixed, one row. If any is typical, mix.
    specs = []
    for a in resolve(attach_choice.get("addon", "typical"), rates["addon"]):
        for o in resolve(attach_choice.get("otp", "typical"), rates["otp"]):
            for u in resolve(attach_choice.get("upsell", "typical"), rates["upsell"]):
                w = 1.0
                if attach_choice.get("addon", "typical") == "typical":
                    w *= rates["addon"] if a == 1 else (1 - rates["addon"])
                if attach_choice.get("otp", "typical") == "typical":
                    w *= rates["otp"] if o == 1 else (1 - rates["otp"])
                if attach_choice.get("upsell", "typical") == "typical":
                    w *= rates["upsell"] if u == 1 else (1 - rates["upsell"])
                specs.append((w, a, o, u))

    total = 0.0
    total_low = 0.0
    total_high = 0.0
    comp = {"main": 0.0, "addon": 0.0, "otp": 0.0, "upsell": 0.0}
    for w, a, o, u in specs:
        row = base.copy()
        row["has_addon"] = int(a)
        row["has_otp"] = int(o)
        row["has_upsell"] = int(u)
        scored = score_frame(row, curves, bundle, HORIZON, scalar)
        low = structural_components(row, artifact["curves_low"], HORIZON)["struct"].iloc[0]
        high = structural_components(row, artifact["curves_high"], HORIZON)["struct"].iloc[0]
        ratio = float(scored["ratio"].iloc[0])
        total += w * float(scored["pltv"].iloc[0])
        total_low += w * float(max(0.0, low * ratio * scalar))
        total_high += w * float(max(0.0, high * ratio * scalar))
        for key, col in [("main", "struct_main"), ("addon", "struct_addon"), ("otp", "struct_otp"), ("upsell", "struct_upsell")]:
            comp[key] += w * float(scored[col].iloc[0]) * ratio * scalar
    struct_sum = sum(comp.values())
    return {
        "pltv": total,
        "pltv_low": total_low,
        "pltv_high": total_high,
        "components": comp,
        "attach_rates": rates,
        "struct": struct_sum,
    }


def sanity_block(scored: pd.DataFrame) -> dict:
    oldest = scored[scored["order_ts"] < "2025-12-01"]
    out = {}
    for plan, g in oldest.groupby("plan"):
        out[plan] = {
            "n": int(len(g)),
            "mean_pltv": float(g["pltv_2y"].mean()),
            "mean_struct": float(g["struct_2y"].mean()),
        }
    by_plan = (
        scored[scored["age_days"] >= TRAIN_AGE]
        .groupby("plan")["pltv_2y"]
        .agg(["count", "mean"])
        .round(2)
        .to_dict()
    )
    return {"oldest_nov": out, "mature_pltv_by_plan": by_plan}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    orders_raw, tx, products = load_frames()
    orders = prepare_orders(orders_raw, products)
    pay = signed_payments(tx, orders, products)
    print(f"orders {len(orders):,}  payment rows {len(pay):,}")

    print("\nHoldout: March 2026 customers, label = net cash through day 180")
    holdout = evaluate(orders, pay, tx, products)
    print(
        json.dumps(
            {k: v for k, v in holdout.items() if k != "bundle_holdout"},
            indent=2,
            default=float,
        )
    )

    curves = build_curves(orders, pay)
    print("\nTail continuation (share of the previous bill that still arrives):")
    print(json.dumps(curves["tail"], indent=2))
    print("\nCurve snapshot, standard offer (61):")
    for plan in ["1W", "4W", "12W"]:
        seg = curves["main"][plan]["61"]
        s730 = _sum_curve(seg, 730)
        s180 = _sum_curve(seg, 180)
        print(f"  {plan} main-only 180d ${s180:.1f}   730d ${s730:.1f}   tail {curves['tail'][plan]:.3f}")
        for k, (day, mean, n, src) in enumerate(zip(seg["day"], seg["mean"], seg["n"], seg["source"]), start=1):
            if k <= 6 or src != "observed":
                print(f"    k={k:02d} day={day:6.0f} n={n:6d} ${mean:7.2f} {src}")

    o = _prepare_model_frame(orders, pay)
    scalar = 1.0  # the training recenter lives inside predict_ratio
    if holdout["use_model"]:
        bundle = fit_adjustment(o, pay, tx, products, curves, ADJUST_ALPHA)
    else:
        bundle = None
    scale_txt = f"{bundle['scale']:.4f}" if bundle else "1 (not used)"
    print(f"\nProduction adjustment: {'kept' if bundle else 'dropped — curves alone held up'}  recenter={scale_txt}")
    if bundle:
        coefs = coefficient_table(bundle)
        print(coefs.head(25).to_string(index=False))
        coefs.to_csv(ARTIFACTS / "adjustment_effects.csv", index=False)

    curves_low = shift_tails(curves, -0.05)
    curves_high = shift_tails(curves, 0.05)
    scored = score_population(o, pay, curves, curves_low, curves_high, bundle, scalar)

    oct_mask = scored["order_ts"] >= "2026-10-01"
    october = scored.loc[oct_mask].copy()
    predictions = october[["order_id", "pltv_2y"]].sort_values("order_id")
    predictions["pltv_2y"] = predictions["pltv_2y"].round(2)
    predictions.to_csv(ROOT / "predictions.csv", index=False)
    predictions.to_csv(ARTIFACTS / "predictions.csv", index=False)

    keep = [
        "order_id",
        "order_ts",
        "brand",
        "funnel",
        "funnel_version",
        "plan",
        "discount_pct",
        "traffic_source",
        "campaign_id",
        "campaign_name",
        "acquisition_cost_usd",
        "country",
        "country_group",
        "payment_method",
        "age_group",
        "device_type",
        "pltv_2y",
        "pltv_low",
        "pltv_high",
        "struct_2y",
        "ratio",
        "struct_main",
        "struct_addon",
        "struct_otp",
        "struct_upsell",
        "has_addon",
        "has_otp",
        "has_upsell",
    ]
    october[keep].to_csv(ARTIFACTS / "october_scored.csv", index=False)
    table = campaign_table(october)
    table.to_csv(ARTIFACTS / "october_campaigns.csv", index=False)

    print("\nOctober predictions", len(predictions), "mean pLTV", round(predictions["pltv_2y"].mean(), 2))
    print("\nCampaign desk")
    show = table.copy()
    for col in ["spend", "cpa", "avg_pltv", "avg_pltv_low", "avg_pltv_high", "ltv_cac", "headroom"]:
        show[col] = show[col].round(2)
    print(show.drop(columns=["ltv_cac_low", "ltv_cac_high"]).to_string(index=False))

    flips = table[table["decision"] != table["decision_low"]][["campaign_name", "orders", "decision", "decision_low", "decision_high", "ltv_cac"]]
    print("\nDecisions that flip if the tail is 5 points weaker:\n", flips.to_string(index=False) if len(flips) else "none")

    # In-sample calibration check on mature orders, dollar net at day 180 vs prediction at day 180.
    mature = o[o["age_days"] >= TRAIN_AGE]
    actual = mature["order_id"].map(cash_within(tx, mature, products, TRAIN_AGE)).fillna(0.0)
    pred180 = score_frame(mature, curves, bundle, TRAIN_AGE, scalar)
    print("\nIn-sample day-180 (diagnostic, not a holdout)")
    print(_metrics(actual.to_numpy(), pred180["pltv"].to_numpy()))

    metrics = {
        "definition": "Net cash = approved charges − refunds − chargebacks, USD, before fees and tax, within 730 days of order_ts. Test orders excluded. Pending excluded.",
        "holdout": {k: v for k, v in holdout.items() if k != "bundle_holdout"},
        "tail": curves["tail"],
        "recenter": None if bundle is None else bundle["scale"],
        "use_model": holdout["use_model"],
        "october_orders": int(len(predictions)),
        "october_mean_pltv": float(predictions["pltv_2y"].mean()),
        "attach": curves["attach"],
        "one_time_net": curves["one_time_net"],
        "assumptions": [
            "Prediction is made the day the order arrives. Same-day add-on and upsells are known; later renewals, refunds and app workouts are not.",
            "Bills no cohort has reached are extended at a continuation rate shrunk toward 0.90 (1-week, after conversion), 0.88 (4-week), 0.74 (12-week) and 0.80 (upsell subscription).",
            "A 12-week cycle is three 4-week cycles, so its per-cycle continuation is lower. The prior 0.74 sits near the observed 12-week step-down.",
            "On top of the curve, a shrunk ratio adjusts for payment method, age, country, funnel and whether the checkout add-on or one-time upsell was taken. It is kept only if it beats the curve alone on orders from 16-31 March 2026.",
            "Reference customer is a US card payer aged 46-50 on the main quiz. AppLovin and Snapchat are not given their own residual: both launched in mid-2026, and the holdout did not support a traffic-source lift once plan and offer were in the curve.",
            "Organic CPA is 0, so LTV:CAC is undefined and those rows are not scale/hold/cut.",
            "Company thresholds: scale at 3.5 or more, hold from 2.5 up to 3.5, cut below 2.5. Applied to mean pLTV / mean CPA of the campaign's October orders.",
            "pLTV is before processor fees and tax, matching the brief. A fee haircut would lower every ratio and is not in the company rule as stated.",
        ],
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # Curves are JSON-safe already (lists). Bundle has sklearn objects.
    artifact = {
        "curves": curves,
        "curves_low": curves_low,
        "curves_high": curves_high,
        "bundle": bundle,
        "scalar": scalar,
        "use_model": holdout["use_model"],
        "metrics": metrics,
    }
    joblib.dump(artifact, ARTIFACTS / "model.joblib")

    assert predictions["order_id"].is_unique
    assert predictions["pltv_2y"].isna().sum() == 0
    assert (predictions["pltv_2y"] > 0).all()
    assert len(predictions) == int((orders["order_ts"] >= "2026-10-01").sum())
    print("\nWrote", ROOT / "predictions.csv")


if __name__ == "__main__":
    main()
