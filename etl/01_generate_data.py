"""
Generates a synthetic but realistic Telco-style churn dataset:
customers, subscriptions, transactions, support_tickets, churn_events.

Distributions and churn drivers are modeled after the well-known IBM Telco
Customer Churn dataset (contract type, tenure, monthly charges, tech support,
payment method), so downstream ML models learn a genuine, explainable signal
rather than noise. Output: CSVs in data/raw/.
"""
import numpy as np
import pandas as pd
from faker import Faker
from datetime import date, timedelta
import os

SEED = 42
np.random.seed(SEED)
fake = Faker("de_DE")
Faker.seed(SEED)

N_CUSTOMERS = 7043
TODAY = date(2026, 7, 1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)


def rand_choice(options, p, size):
    return np.random.choice(options, size=size, p=p)


def build_customers():
    customer_id = [f"CU-{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
    gender = rand_choice(["Male", "Female"], [0.5, 0.5], N_CUSTOMERS)
    senior_citizen = rand_choice([0, 1], [0.84, 0.16], N_CUSTOMERS)
    partner = rand_choice([0, 1], [0.52, 0.48], N_CUSTOMERS)
    dependents = rand_choice([0, 1], [0.70, 0.30], N_CUSTOMERS)

    # signup dates spread across the last 6 years
    days_back = np.random.randint(30, 6 * 365, size=N_CUSTOMERS)
    signup_date = [TODAY - timedelta(days=int(d)) for d in days_back]

    cities, states = [], []
    for _ in range(N_CUSTOMERS):
        cities.append(fake.city())
        states.append(fake.state())

    df = pd.DataFrame({
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "signup_date": signup_date,
        "city": cities,
        "state": states,
        "country": "Germany",
    })
    return df


def build_subscriptions(customers):
    n = len(customers)
    contract_type = rand_choice(
        ["Month-to-month", "One year", "Two year"], [0.55, 0.24, 0.21], n
    )
    payment_method = rand_choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21], n,
    )
    paperless_billing = rand_choice([0, 1], [0.41, 0.59], n)
    internet_service = rand_choice(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n)
    phone_service = rand_choice([0, 1], [0.10, 0.90], n)

    def dependent_yesno(has_internet, p_yes):
        out = np.where(
            has_internet == "No",
            "No internet service",
            np.where(np.random.random(n) < p_yes, "Yes", "No"),
        )
        return out

    multiple_lines = np.where(phone_service == 0, "No phone service",
                               rand_choice(["Yes", "No"], [0.42, 0.58], n))
    online_security = dependent_yesno(internet_service, 0.29)
    online_backup = dependent_yesno(internet_service, 0.34)
    device_protection = dependent_yesno(internet_service, 0.34)
    tech_support = dependent_yesno(internet_service, 0.29)
    streaming_tv = dependent_yesno(internet_service, 0.38)
    streaming_movies = dependent_yesno(internet_service, 0.39)

    # tenure: month-to-month customers skew shorter, longer contracts skew longer
    tenure_months = np.zeros(n, dtype=int)
    for ct, lam_range in [("Month-to-month", (1, 40)), ("One year", (6, 60)), ("Two year", (12, 72))]:
        mask = contract_type == ct
        tenure_months[mask] = np.random.randint(lam_range[0], lam_range[1], size=mask.sum())
    tenure_months = np.clip(tenure_months, 1, 72)

    # base monthly charge from services
    base = 18.0
    base += np.where(phone_service == 1, 5.0, 0.0)
    base += np.where(internet_service == "DSL", 24.0, 0.0)
    base += np.where(internet_service == "Fiber optic", 42.0, 0.0)
    for addon in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        base += np.where(addon == "Yes", np.random.uniform(4, 9, n), 0.0)
    monthly_charges = np.round(base + np.random.normal(0, 3.5, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 130.0)

    senior_citizen = customers["senior_citizen"].values
    partner = customers["partner"].values

    # --- churn signal: logistic function of realistic drivers ---
    z = (
        -0.06 * tenure_months
        + 0.032 * monthly_charges
        + np.where(contract_type == "Month-to-month", 1.35, 0.0)
        + np.where(contract_type == "Two year", -1.1, 0.0)
        + np.where(tech_support == "No", 0.55, 0.0)
        + np.where(payment_method == "Electronic check", 0.45, 0.0)
        + np.where(senior_citizen == 1, 0.35, 0.0)
        + np.where(internet_service == "Fiber optic", 0.35, 0.0)
        + np.where(partner == 0, 0.2, 0.0)
        - 3.15
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn_flag = (np.random.random(n) < churn_prob).astype(int)

    signup = customers["signup_date"].values
    start_date = signup
    end_date = []
    is_active = []
    for i in range(n):
        sd = customers["signup_date"].iloc[i]
        t = int(tenure_months[i])
        max_possible_tenure = (TODAY - sd).days // 30
        t = min(t, max(max_possible_tenure, 1))
        tenure_months[i] = t
        if churn_flag[i] == 1:
            ed = sd + timedelta(days=30 * t)
            if ed >= TODAY:
                ed = TODAY - timedelta(days=1)
            end_date.append(ed)
            is_active.append(0)
        else:
            end_date.append(pd.NaT)
            is_active.append(1)

    df = pd.DataFrame({
        "customer_id": customers["customer_id"].values,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "internet_service": internet_service,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "monthly_charges": monthly_charges,
        "tenure_months": tenure_months,
        "start_date": start_date,
        "end_date": end_date,
        "is_active": is_active,
        "churn_flag": churn_flag,  # kept for churn_events / transactions generation, dropped before load
    })
    df.insert(0, "subscription_id", range(1, n + 1))
    return df


def build_transactions(subs):
    rows = []
    txn_id = 1
    for _, s in subs.iterrows():
        n_months = int(s["tenure_months"])
        cur = s["start_date"]
        for m in range(n_months):
            txn_date = cur + timedelta(days=30 * m)
            amount = round(float(s["monthly_charges"]) + np.random.normal(0, 1.2), 2)
            roll = np.random.random()
            if roll < 0.03:
                status = "failed"
            elif roll < 0.05:
                status = "pending"
            else:
                status = "paid"
            rows.append((txn_id, s["customer_id"], s["subscription_id"], txn_date, amount, "charge", status))
            txn_id += 1
            if np.random.random() < 0.015:
                rows.append((txn_id, s["customer_id"], s["subscription_id"], txn_date, round(-amount * np.random.uniform(0.1, 1.0), 2), "refund", "paid"))
                txn_id += 1
    df = pd.DataFrame(rows, columns=["transaction_id", "customer_id", "subscription_id", "transaction_date", "amount", "transaction_type", "status"])
    return df


def build_support_tickets(subs):
    categories = ["billing", "technical", "cancellation", "general"]
    rows = []
    ticket_id = 1
    for _, s in subs.iterrows():
        base_rate = 0.35 if s["churn_flag"] == 1 else 0.15
        n_tickets = np.random.poisson(base_rate * (s["tenure_months"] / 12 + 1))
        n_tickets = min(n_tickets, 8)
        for _ in range(n_tickets):
            offset = np.random.randint(0, max(int(s["tenure_months"]) * 30, 1))
            opened = s["start_date"] + timedelta(days=int(offset))
            if opened >= TODAY:
                continue
            resolve_days = np.random.randint(0, 14)
            closed = opened + timedelta(days=resolve_days)
            if closed >= TODAY:
                closed = pd.NaT
            cat = np.random.choice(categories, p=[0.35, 0.4, 0.1, 0.15])
            priority = np.random.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15])
            sat = None if pd.isna(closed) else int(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.12, 0.2, 0.35, 0.25]))
            rows.append((ticket_id, s["customer_id"], opened, closed, cat, priority, sat))
            ticket_id += 1
    df = pd.DataFrame(rows, columns=["ticket_id", "customer_id", "opened_date", "closed_date", "category", "priority", "satisfaction_score"])
    return df


def build_churn_events(subs):
    reasons = ["Price", "Better offer", "Service quality", "Moved", "Other"]
    churned = subs[subs["churn_flag"] == 1].copy()
    df = pd.DataFrame({
        "churn_id": range(1, len(churned) + 1),
        "customer_id": churned["customer_id"].values,
        "churn_date": churned["end_date"].values,
        "churn_flag": 1,
        "churn_reason": np.random.choice(reasons, size=len(churned), p=[0.34, 0.22, 0.24, 0.1, 0.1]),
    })
    return df


if __name__ == "__main__":
    customers = build_customers()
    subs = build_subscriptions(customers)
    transactions = build_transactions(subs)
    tickets = build_support_tickets(subs)
    churn_events = build_churn_events(subs)

    subs_out = subs.drop(columns=["churn_flag"])

    customers.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
    subs_out.to_csv(os.path.join(OUT_DIR, "subscriptions.csv"), index=False)
    transactions.to_csv(os.path.join(OUT_DIR, "transactions.csv"), index=False)
    tickets.to_csv(os.path.join(OUT_DIR, "support_tickets.csv"), index=False)
    churn_events.to_csv(os.path.join(OUT_DIR, "churn_events.csv"), index=False)

    print(f"customers:       {len(customers):,}")
    print(f"subscriptions:   {len(subs_out):,}")
    print(f"transactions:    {len(transactions):,}")
    print(f"support_tickets: {len(tickets):,}")
    print(f"churn_events:    {len(churn_events):,}  (churn rate: {subs['churn_flag'].mean():.1%})")
