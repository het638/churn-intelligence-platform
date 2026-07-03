"""
Pulls customers + subscriptions + aggregated support-ticket behavior from MySQL
and builds a single model-ready feature table, saved to data/processed/features.csv.
"""
import os
import pandas as pd
from db_utils import get_engine

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")

QUERY = """
SELECT
    c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
    s.contract_type, s.payment_method, s.paperless_billing, s.internet_service,
    s.phone_service, s.multiple_lines, s.online_security, s.online_backup,
    s.device_protection, s.tech_support, s.streaming_tv, s.streaming_movies,
    s.monthly_charges, s.tenure_months, s.is_active,
    COALESCE(t.ticket_count, 0) AS ticket_count,
    COALESCE(t.avg_satisfaction, 3.0) AS avg_satisfaction
FROM customers c
JOIN subscriptions s ON s.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS ticket_count, AVG(satisfaction_score) AS avg_satisfaction
    FROM support_tickets
    GROUP BY customer_id
) t ON t.customer_id = c.customer_id
"""

if __name__ == "__main__":
    engine = get_engine()
    df = pd.read_sql(QUERY, engine)
    df["churned"] = 1 - df["is_active"]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
