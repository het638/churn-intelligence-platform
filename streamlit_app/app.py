"""
Customer Churn & Revenue Intelligence Platform — Streamlit app.

Reads a static snapshot (data/processed/churn_predictions_snapshot.csv) plus
the trained model artifacts (ml/models/*.joblib) that are committed to the
repo, so it runs standalone on Streamlit Community Cloud with no database
connection required. For live "what-if" predictions it re-runs the same
scikit-learn pipeline + SHAP explainer that were trained by ml/02_train_churn_models.py.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "..", "ml", "models")
SNAPSHOT_PATH = os.path.join(BASE, "..", "data", "processed", "churn_predictions_snapshot.csv")

st.set_page_config(page_title="Churn & Revenue Intelligence", layout="wide", page_icon="\U0001F4C9")


@st.cache_resource
def load_artifacts():
    churn_model = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return churn_model, meta, explainer, feature_names


@st.cache_data
def load_snapshot():
    return pd.read_csv(SNAPSHOT_PATH)


churn_model, meta, explainer, feature_names = load_artifacts()
df = load_snapshot()

st.title("Customer Churn & Revenue Intelligence Platform")
st.caption(
    "End-to-end analytics + ML product: MySQL -> SQL analytics -> scikit-learn churn/CLV models "
    "-> SHAP explainability -> this app. Data is synthetic, modeled on the IBM Telco Customer "
    "Churn dataset's structure and churn drivers."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001F4CA Overview", "\U0001F52E Predict Churn", "⚠️ High-Risk Customers", "\U0001F465 CLV & Segments"]
)

# ------------------------------------------------------------------ Overview
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    total_customers = len(df)
    mrr = df["monthly_charges"].sum()
    churn_rate = (df["risk_tier"] != "Low").mean()  # descriptive, not the literal churn rate
    revenue_at_risk = df.loc[df["risk_tier"] == "High", "monthly_charges"].sum()
    avg_clv = df["predicted_clv"].mean()

    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("MRR", f"€{mrr:,.0f}")
    c3.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.1%}")
    c4.metric("Revenue at Risk (High tier)", f"€{revenue_at_risk:,.0f}")
    c5.metric("Avg Predicted CLV", f"€{avg_clv:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.5,
                     title="Customers by Risk Tier",
                     color=risk_counts.index,
                     color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_contract = df.groupby("contract_type")["churn_probability"].mean().sort_values(ascending=False)
        fig = px.bar(x=by_contract.index, y=by_contract.values,
                     labels={"x": "Contract Type", "y": "Avg Churn Probability"},
                     title="Avg Churn Probability by Contract Type")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model performance")
    comp_path = os.path.join(BASE, "..", "ml", "outputs", "model_comparison.csv")
    if os.path.exists(comp_path):
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)
    roc_path = os.path.join(BASE, "..", "ml", "outputs", "roc_comparison.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC curves — Logistic Regression vs Random Forest vs XGBoost", width=500)

# ------------------------------------------------------------------ Predict
with tab2:
    st.subheader("Score a customer in real time")
    st.write("Enter a customer's profile to get a live churn probability, risk tier, and SHAP explanation.")

    colA, colB, colC = st.columns(3)
    with colA:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has partner", [0, 1], format_func=lambda x: "Yes" if x else "No")
        dependents = st.selectbox("Has dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")
        contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    with colB:
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        monthly_charges = st.slider("Monthly charges (EUR)", 18.0, 130.0, 65.0)
        tenure_months = st.slider("Tenure (months)", 1, 72, 12)
    with colC:
        paperless_billing = st.selectbox("Paperless billing", [0, 1], format_func=lambda x: "Yes" if x else "No")
        phone_service = st.selectbox("Phone service", [0, 1], format_func=lambda x: "Yes" if x else "No")
        multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        ticket_count = st.slider("Support tickets filed", 0, 10, 1)
        avg_satisfaction = st.slider("Avg support satisfaction (1-5)", 1.0, 5.0, 3.5)

    if st.button("Score this customer", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "senior_citizen": senior_citizen, "partner": partner,
            "dependents": dependents, "contract_type": contract_type, "payment_method": payment_method,
            "paperless_billing": paperless_billing, "internet_service": internet_service,
            "phone_service": phone_service, "multiple_lines": multiple_lines,
            "online_security": online_security, "online_backup": online_backup,
            "device_protection": device_protection, "tech_support": tech_support,
            "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
            "monthly_charges": monthly_charges, "tenure_months": tenure_months,
            "ticket_count": ticket_count, "avg_satisfaction": avg_satisfaction,
        }])
        X = row[meta["categorical"] + meta["numeric"]]
        proba = churn_model.predict_proba(X)[0, 1]
        risk = "Low" if proba < 0.3 else ("Medium" if proba < 0.6 else "High")

        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Risk tier", risk)

        st.subheader("Why? (SHAP explanation)")
        prep = churn_model.named_steps["prep"]
        transformed = prep.transform(X)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        X_row_df = pd.DataFrame(transformed, columns=feature_names)

        sv = explainer.shap_values(X_row_df)
        if isinstance(sv, list):
            sv = sv[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.shape(base_value)) else base_value

        explanation = shap.Explanation(
            values=sv[0], base_values=base_value, data=X_row_df.iloc[0].values,
            feature_names=list(feature_names),
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------------ Risk list
with tab3:
    st.subheader("High-risk customer explorer")
    c1, c2, c3 = st.columns(3)
    tier_filter = c1.multiselect("Risk tier", ["Low", "Medium", "High"], default=["High"])
    contract_filter = c2.multiselect("Contract type", df["contract_type"].unique().tolist(),
                                      default=df["contract_type"].unique().tolist())
    min_clv = c3.slider("Min predicted CLV (EUR)", 0, int(df["predicted_clv"].max()), 0)

    filtered = df[df["risk_tier"].isin(tier_filter) & df["contract_type"].isin(contract_filter)
                  & (df["predicted_clv"] >= min_clv)].sort_values("churn_probability", ascending=False)

    st.dataframe(
        filtered[["customer_id", "contract_type", "internet_service", "payment_method",
                  "monthly_charges", "tenure_months", "churn_probability", "risk_tier", "predicted_clv"]],
        use_container_width=True, height=400,
    )
    st.caption(f"{len(filtered):,} customers match — "
               f"EUR {filtered['monthly_charges'].sum():,.0f} monthly revenue represented.")

    fig = px.scatter(filtered, x="tenure_months", y="churn_probability", size="monthly_charges",
                      color="risk_tier", color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"},
                      hover_data=["customer_id", "predicted_clv"], title="Tenure vs Churn Probability")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ CLV
with tab4:
    st.subheader("Customer Lifetime Value & Segmentation")
    seg_summary = df.groupby("segment_label").agg(
        customers=("customer_id", "count"),
        avg_clv=("predicted_clv", "mean"),
        avg_tenure=("tenure_months", "mean"),
        avg_churn_prob=("churn_probability", "mean"),
    ).round(2).reset_index()
    st.dataframe(seg_summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_summary, x="segment_label", y="customers", title="Customers per Segment",
                      color="segment_label")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x="tenure_months", y="predicted_clv", color="segment_label",
                          size="monthly_charges", opacity=0.6, title="CLV vs Tenure by Segment")
        st.plotly_chart(fig, use_container_width=True)
