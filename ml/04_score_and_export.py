"""
Combines churn probability (from the best classifier) with CLV + segment
(from 03_segment_and_clv.py) into one scored table, and writes it back into
MySQL's ml_predictions table for the BI layer / Streamlit app to consume.
"""
import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from db_utils import get_engine

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
FEATURES_SCORED_PATH = os.path.join(BASE, "..", "data", "processed", "features_scored.csv")


def risk_tier(p):
    if p < 0.3:
        return "Low"
    elif p < 0.6:
        return "Medium"
    return "High"


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_SCORED_PATH)

    churn_model = joblib.load(os.path.join(MODELS_DIR, "churn_model.joblib"))
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        meta = json.load(f)

    X = df[meta["categorical"] + meta["numeric"]]
    df["churn_probability"] = churn_model.predict_proba(X)[:, 1].round(5)
    df["risk_tier"] = df["churn_probability"].apply(risk_tier)

    out = df[["customer_id", "churn_probability", "risk_tier", "predicted_clv", "segment_cluster"]].copy()
    out["model_version"] = meta["model_version"]
    out["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE ml_predictions"))
    out.to_sql("ml_predictions", engine, if_exists="append", index=False, chunksize=2000, method="multi")

    print(f"Wrote {len(out):,} rows to ml_predictions")
    print(out["risk_tier"].value_counts())
    print(f"\nTotal monthly revenue at risk (High tier): EUR "
          f"{df.loc[df['risk_tier']=='High', 'monthly_charges'].sum():,.2f}")

    # Static snapshot for the deployed Streamlit app (no MySQL access from the cloud)
    labels = {0: "High-Value Loyal", 1: "Growing / Mid-Value", 2: "Price-Sensitive", 3: "Low-Value / New"}
    snapshot_cols = ["customer_id", "contract_type", "internet_service", "payment_method",
                      "monthly_charges", "tenure_months", "ticket_count", "churn_probability",
                      "risk_tier", "predicted_clv", "segment_cluster"]
    snapshot = df[snapshot_cols].copy()
    snapshot["segment_label"] = snapshot["segment_cluster"].map(labels)
    snapshot_path = os.path.join(BASE, "..", "data", "processed", "churn_predictions_snapshot.csv")
    snapshot.to_csv(snapshot_path, index=False)
    print(f"Saved Streamlit snapshot -> {snapshot_path}")
