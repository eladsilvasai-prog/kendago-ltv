"""Media desk: October 2026 campaign calls, and a pLTV for an order that just landed."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import ltv

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="Kendago LTV desk", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.4rem; max-width: 1280px; }
  h1 { font-weight: 560; letter-spacing: -0.03em; font-size: 2rem; }
  h2, h3 { letter-spacing: -0.02em; }
  div[data-testid="stMetric"] {
    background: #efe8da;
    border: 1px solid #e0d6c4;
    padding: 0.7rem 0.9rem 0.5rem;
  }
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
  .call {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .scale { background: #d7efe3; color: #145c3a; }
  .hold { background: #f8e7c3; color: #7a4e08; }
  .cut { background: #f6d5dc; color: #8a1530; }
  .organic { background: #e6e1d6; color: #3f3a32; }
  .hero {
    font-size: 3.2rem;
    line-height: 1;
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
  }
  .sub { color: #5c564c; font-size: 0.95rem; }
  .rule { border-top: 1px solid #e0d6c4; margin: 0.6rem 0 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_october() -> pd.DataFrame:
    df = pd.read_csv(ART / "october_scored.csv")
    df["campaign_id"] = df["campaign_id"].fillna("").astype(str)
    return df


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


def money(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"${v:,.2f}"


def ratio_txt(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.2f}"


def badge(decision: str) -> str:
    label = {"scale": "Scale", "hold": "Hold", "cut": "Cut", "organic": "Organic"}.get(decision, decision)
    return f'<span class="call {decision}">{label}</span>'


def plan_mix(g: pd.DataFrame) -> str:
    share = g["plan"].value_counts(normalize=True)
    parts = [f"{share[p] * 100:.0f}% {p}" for p in ["4W", "1W", "12W"] if p in share.index and share[p] >= 0.05]
    return " · ".join(parts)


def desk_table(orders: pd.DataFrame) -> pd.DataFrame:
    table = ltv.campaign_table(orders)
    mixes = []
    framed = orders.copy()
    framed["campaign_id"] = framed["campaign_id"].fillna("").astype(str)
    organic = framed["campaign_id"].eq("") | framed["traffic_source"].eq("organic")
    framed.loc[organic, "campaign_id"] = "organic:" + framed.loc[organic, "brand"].astype(str)
    for cid, g in framed.groupby("campaign_id", sort=False):
        mixes.append({"campaign_id": cid, "plan_mix": plan_mix(g)})
    return table.merge(pd.DataFrame(mixes), on="campaign_id", how="left")


artifact = load_artifact()
october = load_october()
metrics = load_metrics()

st.title("Kendago LTV desk")
st.markdown(
    '<p class="sub">Two-year net cash, scored the day the order arrives. '
    "October 2026 book. Test orders are out. Scale at 3.5× CPA or better, "
    "hold from 2.5×, cut below.</p>",
    unsafe_allow_html=True,
)

desk, scorer, method = st.tabs(["October campaigns", "Score an order", "How the number is built"])

with desk:
    with st.sidebar:
        st.markdown("### Book")
        brands = ["All brands"] + sorted(october["brand"].unique())
        sources = ["All sources"] + sorted(october["traffic_source"].unique())
        brand_pick = st.selectbox("Brand", brands)
        source_pick = st.selectbox("Source", sources)
        paid_only = st.toggle("Paid media only", value=True)
        st.caption("CPA is the average acquisition cost of that campaign’s October orders. Organic has no CPA, so it is not a scale / hold / cut.")

    view = october
    if brand_pick != "All brands":
        view = view[view["brand"] == brand_pick]
    if source_pick != "All sources":
        view = view[view["traffic_source"] == source_pick]
    if paid_only:
        view = view[view["traffic_source"] != "organic"]

    table = desk_table(view) if len(view) else pd.DataFrame()
    paid = view[view["traffic_source"] != "organic"] if len(view) else view
    spend = float(paid["acquisition_cost_usd"].sum()) if len(paid) else 0.0
    book_pltv = float(paid["pltv_2y"].sum()) if len(paid) else 0.0
    book_ratio = book_pltv / spend if spend else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{len(view):,}")
    c2.metric("Spend", money(spend))
    c3.metric("Avg CPA", money(paid["acquisition_cost_usd"].mean()) if len(paid) else "—")
    c4.metric("Avg pLTV", money(view["pltv_2y"].mean()) if len(view) else "—")
    c5.metric("Book ratio", ratio_txt(book_ratio))

    if table.empty:
        st.info("No October orders in this filter.")
    else:
        calls = table[table["decision"] != "organic"]["decision"].value_counts()
        st.markdown(
            f'<p class="sub">{int(calls.get("scale", 0))} scale · {int(calls.get("hold", 0))} hold · '
            f'{int(calls.get("cut", 0))} cut. Book LTV:CAC is total pLTV divided by total spend, not an average of ratios.</p>',
            unsafe_allow_html=True,
        )
        show = table.copy()
        show["Campaign"] = show["campaign_name"]
        show["Call"] = show["decision"].map({"scale": "Scale", "hold": "Hold", "cut": "Cut", "organic": "Organic"})
        show["If tail weaker"] = show["decision_low"].map(
            {"scale": "Scale", "hold": "Hold", "cut": "Cut", "organic": "Organic"}
        )
        show["Orders"] = show["orders"]
        show["Plans"] = show["plan_mix"]
        show["Spend"] = show["spend"].map(money)
        show["CPA"] = show["cpa"].map(money)
        show["pLTV"] = show["avg_pltv"].map(money)
        show["pLTV range"] = [
            f"{money(lo)} – {money(hi)}" for lo, hi in zip(show["avg_pltv_low"], show["avg_pltv_high"])
        ]
        show["LTV:CAC"] = show["ltv_cac"].map(ratio_txt)
        show["vs 3.5×"] = show["headroom"].map(lambda v: "—" if pd.isna(v) else f"{v:+,.0f}")
        cols = [
            "Campaign",
            "Call",
            "LTV:CAC",
            "Orders",
            "CPA",
            "pLTV",
            "brand",
            "traffic_source",
            "If tail weaker",
            "Plans",
            "Spend",
            "pLTV range",
            "vs 3.5×",
        ]
        styled = show[cols].rename(columns={"brand": "Brand", "traffic_source": "Source"})

        def paint(val):
            tone = {"Scale": "#d7efe3", "Hold": "#f8e7c3", "Cut": "#f6d5dc", "Organic": "#e6e1d6"}.get(val)
            if not tone:
                return ""
            return f"background-color: {tone}"

        st.dataframe(
            styled.style.map(paint, subset=["Call", "If tail weaker"]),
            width="stretch",
            hide_index=True,
            height=640,
        )
        st.caption(
            "pLTV range moves only the bills no cohort has reached yet, by 5 points of continuation. "
            "“vs 3.5×” is dollars per order above or below the scale bar. "
            "One campaign, Dance Reels, is a hold that becomes a cut if that tail is weaker."
        )
        st.download_button(
            "Download predictions.csv",
            data=(ROOT / "predictions.csv").read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
        )

with scorer:
    st.markdown("A number for an order that just arrived. Quiz detail we don’t have is left at the typical customer; it is not in the adjustment.")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        brand = st.selectbox("Brand", ["FlexHer", "PlateWise"])
        funnel_options = ["main_quiz", "dance_quiz"] if brand == "FlexHer" else ["main_quiz"]
        funnel = st.selectbox("Funnel", funnel_options, format_func=lambda v: "Main quiz" if v == "main_quiz" else "Dance quiz")
        plan = st.selectbox("Plan", ["4W", "1W", "12W"], format_func=lambda v: {"4W": "4-week, $14.99 then $39", "1W": "1-week trial, $6.99 then $39", "12W": "12-week, $24.99 then $69"}[v])
        offer = st.selectbox("Offer", ["61", "71"], format_func=lambda v: "Standard, 61% off" if v == "61" else "Save offer, 71% off")
        country = st.selectbox("Country", ["US", "GB", "CA", "AU", "OTHER"], format_func=lambda v: "Other market" if v == "OTHER" else v)
        source = st.selectbox("Source", ["facebook", "google", "applovin", "snapchat", "organic"])
        pay = st.selectbox("Payment", ["card", "paypal", "apple_pay", "google_pay"], format_func=lambda v: v.replace("_", " "))
        age = st.selectbox(
            "Age",
            ["46-50", "18-30", "31-35", "36-40", "41-45", "51-55", "56-60", "61-65", "66-70", "71+", "unknown"],
        )
        st.markdown("**Taken at checkout**")
        a1, a2, a3 = st.columns(3)
        addon = a1.selectbox("Add-on $12", ["typical", "yes", "no"], format_func=str.title)
        otp = a2.selectbox("Bundle $29", ["typical", "yes", "no"], format_func=str.title)
        upsell = a3.selectbox("Coaching trial", ["typical", "yes", "no"], format_func=str.title)
        cpa = st.number_input("CPA, USD (0 if you only want pLTV)", min_value=0.0, value=0.0, step=1.0)

    profile = {
        "brand": brand,
        "funnel": funnel,
        "plan": plan,
        "discount_pct": offer,
        "country": country,
        "traffic_source": source,
        "payment_method": pay,
        "age_group": age,
        "gender": "female",
        "activity_level": "light",
        "main_goal": "lose_weight",
        "device_type": "mobile",
        "device_tier": "medium",
    }
    result = ltv.score_new_order(profile, {"addon": addon, "otp": otp, "upsell": upsell}, artifact)
    decision = ltv.decide(result["pltv"], cpa) if cpa > 0 else None
    rates = result["attach_rates"]

    with right:
        st.markdown('<p class="sub">2-year pLTV</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero">{money(result["pltv"])}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="sub">Tail range {money(result["pltv_low"])} – {money(result["pltv_high"])}. '
            f'Typical attach on this plan: add-on {rates["addon"]:.0%}, bundle {rates["otp"]:.0%}, '
            f'coaching {rates["upsell"]:.0%}.</p>',
            unsafe_allow_html=True,
        )
        if decision:
            cac = result["pltv"] / cpa
            st.markdown(badge(decision) + f' &nbsp; LTV:CAC **{cac:.2f}** at CPA {money(cpa)}', unsafe_allow_html=True)
        else:
            st.caption("Enter a CPA to get scale / hold / cut.")
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        comp = result["components"]
        labels = [
            ("Main plan", comp["main"]),
            ("Checkout add-on", comp["addon"]),
            ("One-time bundle", comp["otp"]),
            ("Coaching subscription", comp["upsell"]),
        ]
        for name, value in labels:
            st.write(f"{name}  ·  {money(value)}")

with method:
    test = metrics["holdout"]["test_model"]
    base = metrics["holdout"]["test_baseline"]
    st.markdown(
        f"""
The export stops on 31 Oct 2026, so the oldest customer has about a year of cash, not two.
pLTV is **expected net cash over the next 730 days**: approved charges, minus refunds, minus chargebacks.
Pending charges are ignored. Processor fees and tax are not taken off, because the 3.5 / 2.5 rule in the brief is on that same cash.

**Billing curve.** For each plan and each offer (61% or 71%), the value of bill number k is the average net cash from customers who were already old enough to have been billed and to have refunded. Bills nobody has reached — past about day 340 for the 4-week plan, past day 252 for the 12-week plan — continue at a rate anchored on the last observed bills and pulled toward a prior: **{metrics['tail']['4W']:.2f}** per 4-week cycle, **{metrics['tail']['1W']:.2f}** after the trial converts, **{metrics['tail']['12W']:.2f}** per 12-week cycle, **{metrics['tail']['upsell']:.2f}** on the coaching subscription. A 12-week cycle is three 4-week cycles of time in which to cancel, so its continuation is lower. The save offer has its own curve: those buyers pay less up front and renew worse, and the gap does not close.

**What is known on day 0.** The plan, the offer, the country, the payment method, the age, the funnel, and whether they took the add-on, the $29 bundle, or the coaching trial. Those three fire within minutes. Workouts completed in the app are not used: they pile up after purchase, and older orders have far more of them.

**Customer adjustment.** On top of the curve, a ridge model shifts the value for payment method, age, country, funnel, and the two one-time extras. The reference is a US card payer aged 46–50 on the main quiz. It was kept because, on orders from 16–31 March 2026 that the fit never saw, it lowered the average error from **${base['mae']:.2f}** to **${test['mae']:.2f}** and raised R² from **{base['r2']:.3f}** to **{test['r2']:.3f}**, with a bias of **${test['bias']:.2f}** on a mean of ${test['mean_actual']:.0f}. Traffic source did not earn a place: once plan and offer are in the curve, Facebook, Google and organic are close, and AppLovin and Snapchat are too new for a two-year residual of their own. They inherit the curve.

**Check.** The curve alone, with March held out, predicted day-180 cash of ${base['mean_pred']:.1f} against ${base['mean_actual']:.1f} actual.
"""
    )
    coef = ltv.coefficient_table(artifact["bundle"])
    nice = {
        "age_group": "Age",
        "payment_method": "Payment",
        "country_group": "Country",
        "funnel": "Funnel",
        "has_otp": "Bundle",
        "has_addon": "Add-on",
    }
    rows = []
    for _, r in coef.iterrows():
        col, level = r["feature"].split("=", 1)
        rows.append(
            {
                "Versus the reference": f"{nice.get(col, col)} · {level}",
                "Shift": f"{r['ratio_lift']:+.0%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Shifts stack, and they are versus the reference customer, not versus the average. "
        "A recenter of about 3.5% puts the average customer back on the curve, so the adjustment moves value between customers instead of inventing it. "
        "Under-40 and the dance quiz are the cuts that change a campaign call."
    )
