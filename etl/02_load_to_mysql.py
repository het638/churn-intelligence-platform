"""
Cleans/validates the generated CSVs and loads them into MySQL (churn_platform)
in FK-safe order: customers -> subscriptions -> transactions/support_tickets/churn_events.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DB_URL = (
    f"mysql+pymysql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)
engine = create_engine(DB_URL)


def validate(df, name, pk, not_null_cols):
    assert df[pk].is_unique, f"{name}: duplicate primary key {pk}"
    for col in not_null_cols:
        n_null = df[col].isna().sum()
        assert n_null == 0, f"{name}: {n_null} nulls in required column {col}"
    print(f"  [ok] {name}: {len(df):,} rows validated")


def load_table(df, table, chunksize=5000):
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    print(f"  [loaded] {table}: {len(df):,} rows")


if __name__ == "__main__":
    print("Reading CSVs...")
    customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"), parse_dates=["signup_date"])
    subscriptions = pd.read_csv(os.path.join(RAW_DIR, "subscriptions.csv"), parse_dates=["start_date", "end_date"])
    transactions = pd.read_csv(os.path.join(RAW_DIR, "transactions.csv"), parse_dates=["transaction_date"])
    tickets = pd.read_csv(os.path.join(RAW_DIR, "support_tickets.csv"), parse_dates=["opened_date", "closed_date"])
    churn_events = pd.read_csv(os.path.join(RAW_DIR, "churn_events.csv"), parse_dates=["churn_date"])

    print("Validating...")
    validate(customers, "customers", "customer_id", ["customer_id", "signup_date"])
    validate(subscriptions, "subscriptions", "subscription_id", ["customer_id", "monthly_charges", "tenure_months"])
    validate(transactions, "transactions", "transaction_id", ["customer_id", "subscription_id", "amount"])
    validate(tickets, "support_tickets", "ticket_id", ["customer_id", "opened_date"])
    validate(churn_events, "churn_events", "churn_id", ["customer_id", "churn_date"])

    # referential integrity checks
    assert set(subscriptions["customer_id"]) <= set(customers["customer_id"]), "orphan subscriptions"
    assert set(transactions["customer_id"]) <= set(customers["customer_id"]), "orphan transactions"
    assert set(churn_events["customer_id"]) <= set(customers["customer_id"]), "orphan churn_events"
    print("  [ok] referential integrity")

    print("Truncating existing rows...")
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in ["ml_predictions", "churn_events", "support_tickets", "transactions", "subscriptions", "customers"]:
            conn.execute(text(f"TRUNCATE TABLE {t}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    print("Loading...")
    load_table(customers, "customers")
    load_table(subscriptions, "subscriptions")
    load_table(transactions, "transactions", chunksize=10000)
    load_table(tickets, "support_tickets")
    load_table(churn_events, "churn_events")

    print("Done.")
