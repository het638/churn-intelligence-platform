"""
Two more ML components:
  1. K-Means customer segmentation on behavior/value features.
  2. CLV modeling: a regressor trained on churned customers' realized total
     tenure (months) predicts *expected total lifetime* for active customers;
     predicted_clv = monthly_charges * predicted_total_tenure_months.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.dirname(__file__)
FEATURES_PATH = os.path.join(BASE, "..", "data", "processed", "features.csv")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")

CATEGORICAL = ["gender", "contract_type", "payment_method", "internet_service",
               "multiple_lines", "online_security", "online_backup", "device_protection",
               "tech_support", "streaming_tv", "streaming_movies"]
NUMERIC_NO_TENURE = ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service",
                      "monthly_charges", "ticket_count", "avg_satisfaction"]

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)

    # ---------- 1. CLV regression: predict total lifetime (months) ----------
    churned = df[df["churned"] == 1].copy()
    active = df[df["churned"] == 0].copy()

    X_c = churned[CATEGORICAL + NUMERIC_NO_TENURE]
    y_c = churned["tenure_months"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    clv_preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC_NO_TENURE),
    ])
    clv_model = Pipeline([
        ("prep", clv_preprocess),
        ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)),
    ])
    clv_model.fit(Xc_train, yc_train)
    pred_test = clv_model.predict(Xc_test)
    mae = mean_absolute_error(yc_test, pred_test)
    r2 = r2_score(yc_test, pred_test)
    print(f"CLV lifetime-months regressor: MAE={mae:.2f} months, R2={r2:.3f}")

    # predicted total tenure for active customers must be >= their current tenure
    pred_total_tenure_active = clv_model.predict(active[CATEGORICAL + NUMERIC_NO_TENURE])
    pred_total_tenure_active = np.maximum(pred_total_tenure_active, active["tenure_months"].values)

    df["predicted_total_tenure_months"] = np.nan
    df.loc[df["churned"] == 0, "predicted_total_tenure_months"] = pred_total_tenure_active
    df.loc[df["churned"] == 1, "predicted_total_tenure_months"] = churned["tenure_months"].values
    df["predicted_clv"] = (df["predicted_total_tenure_months"] * df["monthly_charges"]).round(2)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clv_model, os.path.join(MODELS_DIR, "clv_model.joblib"))

    # ---------- 2. K-Means segmentation ----------
    seg_features = df[["tenure_months", "monthly_charges", "ticket_count", "predicted_clv"]].copy()
    scaler = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    raw_cluster = kmeans.fit_predict(seg_scaled)

    # relabel clusters by descending predicted_clv so labels are stable/meaningful
    centers = pd.DataFrame(seg_features).assign(cluster=raw_cluster).groupby("cluster")["predicted_clv"].mean()
    order = centers.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    df["segment_cluster"] = [remap[c] for c in raw_cluster]

    segment_labels = {
        0: "High-Value Loyal",
        1: "Growing / Mid-Value",
        2: "Price-Sensitive",
        3: "Low-Value / New",
    }
    df["segment_label"] = df["segment_cluster"].map(segment_labels)

    joblib.dump({"scaler": scaler, "kmeans": kmeans, "remap": remap, "labels": segment_labels},
                os.path.join(MODELS_DIR, "segmentation.joblib"))

    print("\nSegment sizes:\n", df["segment_label"].value_counts())
    print("\nSegment profile (means):\n",
          df.groupby("segment_label")[["tenure_months", "monthly_charges", "predicted_clv", "churned"]].mean().round(2))

    df.to_csv(os.path.join(BASE, "..", "data", "processed", "features_scored.csv"), index=False)
    print(f"\nSaved -> data/processed/features_scored.csv")
