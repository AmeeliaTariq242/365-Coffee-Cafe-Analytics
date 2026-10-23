import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="365 Coffee Café | Guest Return Intelligence", page_icon="☕", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent
MODEL_PATH, DATA_PATH = BASE / "coffee_return_model.joblib", BASE / "coffee_visit_level_dataset.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 85% 0%, #422b23 0%, #171821 42%, #0d111a 100%);color:#f8f2ed;}
[data-testid="stSidebar"] {background:#17151a;border-right:1px solid #41302b;}
.hero {padding:28px 32px;border:1px solid #594036;border-radius:24px;background:linear-gradient(120deg,rgba(74,45,33,.97),rgba(32,29,37,.95));margin-bottom:22px;box-shadow:0 18px 50px #0004;}
.eyebrow {color:#f0bd8b;text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;}
.hero h1 {font-family:Manrope,sans-serif;font-size:34px;font-weight:800;margin:7px 0;color:#fff8f1;}
.hero p {color:#d6c2b6;margin:0;max-width:820px;}
.panel {padding:22px;border:1px solid #40323a;border-radius:18px;background:rgba(26,25,33,.88);margin-bottom:18px;}
.small-muted {color:#b9a9a4;font-size:13px;}
.result {padding:22px;border-radius:18px;background:linear-gradient(135deg,#4a3025,#33212a);border:1px solid #9b6a4e;margin-bottom:16px;}
.result-label {color:#f3c79d;font-size:12px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;}
.result-value {font-family:Manrope,sans-serif;font-size:34px;font-weight:800;color:#fff;margin:6px 0;}
div[data-testid="stMetric"] {background:#211e27;border:1px solid #49383b;padding:16px 18px;border-radius:16px;}
.stButton>button, .stFormSubmitButton>button {border-radius:12px;border:0;background:linear-gradient(90deg,#e8a76f,#f4cb91);color:#241610;font-weight:800;padding:10px 20px;}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#ffe0b4;color:#241610;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load(MODEL_PATH)
@st.cache_data
def load_data(): return pd.read_csv(DATA_PATH, low_memory=False)

def options(df, col, fallback):
    if col not in df: return fallback
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    return values or fallback

st.sidebar.markdown("## ☕ 365 COFFEE CAFÉ")
st.sidebar.caption("GUEST RETURN INTELLIGENCE")
page = st.sidebar.radio("WORKSPACE", ["Return Predictor", "Guest Insights", "About the Model"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Prediction guidance**")
st.sidebar.info("Use the score to support thoughtful retention decisions, not to label a guest permanently.")
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Place coffee_return_model.joblib and coffee_visit_level_dataset.csv beside this app.")
    st.stop()
model, df = load_model(), load_data()

if page == "Return Predictor":
    st.markdown('<div class="hero"><div class="eyebrow">Guest experience · Karachi</div><h1>Guest Return Intelligence</h1><p>Estimate whether a visit may be followed by another visit within 14 days. Explore service, loyalty and visit details to support more relevant retention decisions.</p></div>', unsafe_allow_html=True)
    left,right = st.columns([1.1,.9], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("01 · Visit profile")
        with st.form("visit_form"):
            st.markdown("**Guest & loyalty**")
            c1,c2 = st.columns(2)
            with c1: customer_age = st.number_input("Customer age", 18, 100, 30)
            with c2: customer_gender = st.selectbox("Gender", options(df,"customer_gender",["Unknown"]))
            c1,c2 = st.columns(2)
            with c1: visit_number = st.number_input("Visit number", 1, 1000, 1)
            with c2: days_since_previous_visit = st.number_input("Days since previous visit (-1 if first)", -1, 3650, -1)
            loyalty_tier = st.selectbox("Loyalty tier", options(df,"loyalty_tier",["Basic"]))
            st.markdown("**Order & spend**")
            c1,c2 = st.columns(2)
            with c1: store_location = st.selectbox("Branch", options(df,"store_location",["Unknown"]))
            with c2: order_channel = st.selectbox("Order channel", options(df,"order_channel",["In-store"]))
            c1,c2 = st.columns(2)
            with c1: transaction_qty = st.number_input("Items in basket", 1, 1000, 2)
            with c2: product_category = st.selectbox("Main product category", options(df,"product_category",["Other"]))
            c1,c2 = st.columns(2)
            with c1: revenue = st.number_input("Gross revenue", min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
            with c2: net_revenue = st.number_input("Net revenue", min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
            c1,c2 = st.columns(2)
            with c1: discount_pct = st.slider("Discount (%)", 0.0, 100.0, 0.0)
            with c2: discount_amount = st.number_input("Discount amount", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
            c1,c2 = st.columns(2)
            with c1: payment_method = st.selectbox("Payment method", options(df,"payment_method",["Other"]))
            with c2: promo_code = st.selectbox("Promotion code", options(df,"promo_code",["NONE"]))
            st.markdown("**Experience & timing**")
            c1,c2 = st.columns(2)
            with c1: wait_time_min = st.number_input("Wait time (minutes)", 0.0, 240.0, 5.0, step=1.0)
            with c2: service_rating = st.slider("Service rating", 1, 5, 4)
            c1,c2 = st.columns(2)
            with c1: transaction_hour = st.slider("Visit hour", 0, 23, 10)
            with c2: visit_month = st.selectbox("Month", list(range(1,13)), index=5)
            weekdays = options(df,"weekday",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            weekday = st.selectbox("Weekday", weekdays)
            weekday_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            visit_dayofweek = weekday_map.get(weekday, weekdays.index(weekday) if weekday in weekdays else 0)
            submitted = st.form_submit_button("✦  Estimate return likelihood", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="eyebrow">Café pulse</div><h3>Dataset snapshot</h3>', unsafe_allow_html=True)
        st.metric("Visits represented", f"{len(df):,}")
        if "returned_within_14_days" in df:
            rate = pd.to_numeric(df.returned_within_14_days,errors="coerce").mean()
            st.metric("Observed return rate", f"{rate:.1%}" if pd.notna(rate) else "—")
        if "wait_time_min" in df:
            st.metric("Average wait time", f"{pd.to_numeric(df.wait_time_min,errors='coerce').mean():.1f} min")
        st.markdown('<p class="small-muted">Statistics describe the dataset supplied with this app, not real-time café operations.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><h4>Retention mindset</h4><p class="small-muted">Prioritize service consistency, thoughtful loyalty offers and customer experience. A low score is not a reason to deny service or treat a customer differently.</p></div>', unsafe_allow_html=True)
    if submitted:
        values = {"customer_age":customer_age,"visit_number":visit_number,"days_since_previous_visit":days_since_previous_visit,"transaction_hour":transaction_hour,"transaction_qty":transaction_qty,"revenue":revenue,"net_revenue":net_revenue,"discount_pct":discount_pct,"discount_amount":discount_amount,"wait_time_min":wait_time_min,"service_rating":service_rating,"visit_month":visit_month,"visit_dayofweek":visit_dayofweek,"store_location":store_location,"customer_gender":customer_gender,"loyalty_tier":loyalty_tier,"order_channel":order_channel,"payment_method":payment_method,"promo_code":promo_code,"product_category":product_category,"weekday":weekday}
        try:
            X = pd.DataFrame([values])
            p = float(model.predict_proba(X)[0,1])
            label = "Higher estimated return likelihood" if p >= .5 else "Lower estimated return likelihood"
            st.markdown(f'<div class="result"><div class="result-label">Model estimate · next 14 days</div><div class="result-value">{p:.1%}</div><div style="color:#ead6c8">{label}</div></div>', unsafe_allow_html=True)
            a,b,c = st.columns(3)
            a.metric("Return probability", f"{p:.1%}"); b.metric("Confidence band", "Not calibrated"); c.metric("Decision horizon", "14 days")
            st.progress(min(max(p,0.0),1.0), text="Estimated probability of return")
            st.warning("Decision-support estimate only, not a guarantee. Model probability may not be calibrated and should not be used as the sole basis for decisions about an individual guest.")
        except Exception as e: st.error(f"Unable to generate prediction for these inputs: {e}")
elif page == "Guest Insights":
    st.markdown('<div class="hero"><div class="eyebrow">Hospitality analytics</div><h1>Guest Insights</h1><p>Review patterns in visits, service and customer returns to help frame practical experiments.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Visits", f"{len(df):,}")
    if "returned_within_14_days" in df: c2.metric("Observed return rate", f"{pd.to_numeric(df.returned_within_14_days,errors='coerce').mean():.1%}")
    if "service_rating" in df: c3.metric("Average rating", f"{pd.to_numeric(df.service_rating,errors='coerce').mean():.2f}/5")
    if "wait_time_min" in df: c4.metric("Average wait", f"{pd.to_numeric(df.wait_time_min,errors='coerce').mean():.1f} min")
    t1,t2,t3 = st.tabs(["Return by loyalty tier", "Service experience", "Visit patterns"])
    with t1:
        if {"loyalty_tier","returned_within_14_days"}.issubset(df.columns):
            agg = df.groupby("loyalty_tier").returned_within_14_days.agg(return_rate="mean", visits="count").sort_values("return_rate",ascending=False)
            st.bar_chart(agg[["return_rate"]]); st.dataframe(agg.style.format({"return_rate":"{:.1%}","visits":"{:,.0f}"}),use_container_width=True)
            st.caption("Associations are descriptive and do not prove that loyalty tier causes a return.")
    with t2:
        if {"wait_time_min","returned_within_14_days"}.issubset(df.columns):
            temp=df[["wait_time_min","returned_within_14_days"]].copy(); temp["wait_band"]=pd.cut(pd.to_numeric(temp.wait_time_min,errors="coerce"),[ -0.01,3,5,10,20,240],labels=["0–3","4–5","6–10","11–20","20+"])
            agg=temp.groupby("wait_band",observed=False).returned_within_14_days.agg(return_rate="mean",visits="count")
            st.bar_chart(agg[["return_rate"]]); st.dataframe(agg.style.format({"return_rate":"{:.1%}","visits":"{:,.0f}"}),use_container_width=True)
    with t3:
        col = "weekday" if "weekday" in df else "transaction_hour"
        if col in df and "returned_within_14_days" in df:
            agg=df.groupby(col).returned_within_14_days.agg(return_rate="mean",visits="count")
            st.bar_chart(agg[["return_rate"]]); st.dataframe(agg.style.format({"return_rate":"{:.1%}","visits":"{:,.0f}"}),use_container_width=True)
else:
    st.markdown('<div class="hero"><div class="eyebrow">Transparency</div><h1>About the model</h1><p>How to interpret a return estimate responsibly.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3>Model approach</h3><ul><li>Supervised binary classification using a Random Forest pipeline.</li><li>Inputs describe the visit, customer attributes available in the supplied data, channel, promotion and service experience.</li><li>The prediction target is whether the customer returns within 14 days.</li></ul></div>', unsafe_allow_html=True)
    st.warning("The model identifies statistical patterns, not causes. Predictions can be affected by class imbalance, data coverage, changes in customer behaviour and feature leakage risks. Do not use a prediction as a guarantee about an individual.")
    st.markdown('<div class="panel"><h3>Responsible use</h3><p class="small-muted">Use scores to design and evaluate broad retention experiments. Monitor outcomes across branches and customer groups, respect customer privacy, and continue to provide fair service regardless of a predicted score.</p></div>', unsafe_allow_html=True)
