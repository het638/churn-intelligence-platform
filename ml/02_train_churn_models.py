"""
Trains and compares three churn classifiers (Logistic Regression, Random Forest,
XGBoost), picks the best by ROC-AUC, saves it + a SHAP summary plot for explainability.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
FEATURES_PATH = os.path.join(BASE, "..", "data", "processed", "features.csv")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CATEGORICAL = ["gender", "contract_type", "payment_method", "internet_service",
               "multiple_lines", "online_security", "online_backup", "device_protection",
               "tech_support", "streaming_tv", "streaming_movies"]
NUMERIC = ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service",
           "monthly_charges", "tenure_months", "ticket_count", "avg_satisfaction"]
TARGET = "churned"

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
    }

    results = []
    fitted = {}
    plt.figure(figsize=(7, 6))
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        results.append({"model": name, "roc_auc": round(auc, 4), "precision": round(prec, 4),
                         "recall": round(rec, 4), "f1": round(f1, 4)})
        fitted[name] = pipe

        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        print(f"{name}: AUC={auc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn Model Comparison - ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_comparison.png"), dpi=150)
    plt.close()

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison:\n", results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "churn_model.joblib"))
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump({"best_model": best_name, "categorical": CATEGORICAL, "numeric": NUMERIC,
                    "model_version": "v1.0"}, f, indent=2)

    # --- SHAP explainability on the best model ---
    prep = best_pipe.named_steps["prep"]
    clf = best_pipe.named_steps["clf"]
    X_test_transformed = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()
    X_test_df = pd.DataFrame(
        X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
        columns=feature_names,
    )

    sample = X_test_df.sample(min(500, len(X_test_df)), random_state=42)
    if best_name == "logistic_regression":
        explainer = shap.LinearExplainer(clf, sample)
    else:
        explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    joblib.dump(list(feature_names), os.path.join(MODELS_DIR, "feature_names.joblib"))
    print(f"\nSaved model, ROC plot, SHAP summary plot -> {OUTPUTS_DIR}")
