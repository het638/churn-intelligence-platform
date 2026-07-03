USE churn_platform;

-- 1. Monthly Recurring Revenue (MRR) per calendar month, from active subscriptions
CREATE OR REPLACE VIEW vw_mrr_by_month AS
SELECT
    DATE_FORMAT(m.month_start, '%Y-%m-01') AS month,
    SUM(s.monthly_charges) AS mrr
FROM subscriptions s
JOIN (
    SELECT DISTINCT DATE_FORMAT(txn_month, '%Y-%m-01') AS month_start
    FROM (
        SELECT DATE_SUB(transaction_date, INTERVAL DAY(transaction_date)-1 DAY) AS txn_month
        FROM transactions
    ) x
) m ON s.start_date <= m.month_start
   AND (s.end_date IS NULL OR s.end_date >= m.month_start)
GROUP BY m.month_start
ORDER BY m.month_start;

-- 2. Churn rate by month
CREATE OR REPLACE VIEW vw_churn_rate_by_month AS
SELECT
    DATE_FORMAT(churn_date, '%Y-%m-01') AS month,
    COUNT(*) AS churned_customers
FROM churn_events
GROUP BY DATE_FORMAT(churn_date, '%Y-%m-01')
ORDER BY month;

-- 3. Churn rate by segment: contract type
CREATE OR REPLACE VIEW vw_churn_rate_by_contract AS
SELECT
    s.contract_type,
    COUNT(*) AS total_customers,
    SUM(1 - s.is_active) AS churned_customers,
    ROUND(SUM(1 - s.is_active) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM subscriptions s
GROUP BY s.contract_type;

-- 4. Churn rate by segment: internet service
CREATE OR REPLACE VIEW vw_churn_rate_by_internet AS
SELECT
    s.internet_service,
    COUNT(*) AS total_customers,
    SUM(1 - s.is_active) AS churned_customers,
    ROUND(SUM(1 - s.is_active) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM subscriptions s
GROUP BY s.internet_service;

-- 5. Churn rate by payment method
CREATE OR REPLACE VIEW vw_churn_rate_by_payment AS
SELECT
    s.payment_method,
    COUNT(*) AS total_customers,
    SUM(1 - s.is_active) AS churned_customers,
    ROUND(SUM(1 - s.is_active) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM subscriptions s
GROUP BY s.payment_method;

-- 6. Tenure buckets vs churn
CREATE OR REPLACE VIEW vw_tenure_buckets AS
SELECT
    CASE
        WHEN tenure_months <= 6 THEN '0-6 mo'
        WHEN tenure_months <= 12 THEN '7-12 mo'
        WHEN tenure_months <= 24 THEN '13-24 mo'
        WHEN tenure_months <= 48 THEN '25-48 mo'
        ELSE '48+ mo'
    END AS tenure_bucket,
    COUNT(*) AS total_customers,
    SUM(1 - is_active) AS churned_customers,
    ROUND(SUM(1 - is_active) / COUNT(*) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM subscriptions
GROUP BY tenure_bucket
ORDER BY MIN(tenure_months);

-- 7. Customer Lifetime Value (simple realized CLV = total paid transactions per customer)
CREATE OR REPLACE VIEW vw_customer_clv AS
SELECT
    c.customer_id,
    s.contract_type,
    s.tenure_months,
    s.is_active,
    ROUND(SUM(CASE WHEN t.transaction_type = 'charge' AND t.status = 'paid' THEN t.amount ELSE 0 END), 2) AS lifetime_revenue,
    ROUND(SUM(CASE WHEN t.transaction_type = 'refund' THEN t.amount ELSE 0 END), 2) AS total_refunds
FROM customers c
JOIN subscriptions s ON s.customer_id = c.customer_id
LEFT JOIN transactions t ON t.customer_id = c.customer_id
GROUP BY c.customer_id, s.contract_type, s.tenure_months, s.is_active;

-- 8. Cohort retention: customers grouped by signup month, retained % over months since signup
CREATE OR REPLACE VIEW vw_cohort_retention AS
SELECT
    DATE_FORMAT(c.signup_date, '%Y-%m-01') AS cohort_month,
    TIMESTAMPDIFF(MONTH, c.signup_date, COALESCE(ce.churn_date, CURDATE())) AS months_since_signup,
    COUNT(*) AS customers_active_at_month
FROM customers c
JOIN subscriptions s ON s.customer_id = c.customer_id
LEFT JOIN churn_events ce ON ce.customer_id = c.customer_id
GROUP BY cohort_month, months_since_signup
ORDER BY cohort_month, months_since_signup;

-- 9. Revenue at risk: sum of monthly_charges for customers flagged High risk by the ML layer
CREATE OR REPLACE VIEW vw_revenue_at_risk AS
SELECT
    p.risk_tier,
    COUNT(*) AS customers,
    ROUND(SUM(s.monthly_charges), 2) AS monthly_revenue_at_risk
FROM ml_predictions p
JOIN subscriptions s ON s.customer_id = p.customer_id AND s.is_active = 1
GROUP BY p.risk_tier;

-- 10. Support ticket load vs churn
CREATE OR REPLACE VIEW vw_tickets_vs_churn AS
SELECT
    s.is_active,
    COUNT(DISTINCT s.customer_id) AS customers,
    ROUND(COUNT(t.ticket_id) / COUNT(DISTINCT s.customer_id), 2) AS avg_tickets_per_customer,
    ROUND(AVG(t.satisfaction_score), 2) AS avg_satisfaction
FROM subscriptions s
LEFT JOIN support_tickets t ON t.customer_id = s.customer_id
GROUP BY s.is_active;
