# Power BI Dashboard — Step-by-Step Build Guide

This is the one piece of the project you have to build yourself in the Power BI
Desktop GUI (no CLI/scripting API exists for it). Budget ~2-3 hours. Everything
it needs — schema, views, ML predictions — is already sitting in your
`churn_platform` MySQL database.

## 1. Connect Power BI to MySQL

1. Install the **MySQL Connector/NET** if you don't have it (Power BI needs it
   to talk to MySQL): https://dev.mysql.com/downloads/connector/net/
2. Open Power BI Desktop → **Get Data** → **More** → search "MySQL database".
3. Server: `localhost:3306`   Database: `churn_platform`
4. Auth: **Database** → Username `churn_app`, Password (from your `.env` file
   in the project root — do not commit that file).
5. In the Navigator, select these tables **and** views, then **Transform Data**
   (not Load, so you can rename/clean first):
   - `customers`, `subscriptions`, `transactions`, `support_tickets`,
     `churn_events`, `ml_predictions`
   - `vw_mrr_by_month`, `vw_churn_rate_by_contract`, `vw_churn_rate_by_internet`,
     `vw_churn_rate_by_payment`, `vw_tenure_buckets`, `vw_customer_clv`,
     `vw_cohort_retention`, `vw_revenue_at_risk`, `vw_tickets_vs_churn`
6. Click **Close & Apply**.

## 2. Build the data model (Model view)

Relationships to create (drag `customer_id` from one table onto the other):
- `customers[customer_id]` **1** → **\*** `subscriptions[customer_id]`
- `customers[customer_id]` **1** → **\*** `transactions[customer_id]`
- `customers[customer_id]` **1** → **\*** `support_tickets[customer_id]`
- `customers[customer_id]` **1** → **\*** `churn_events[customer_id]`
- `customers[customer_id]` **1** → **1** `ml_predictions[customer_id]`

Add a Date table for time intelligence (Modeling → New Table):
```
Date = CALENDAR(DATE(2020,1,1), DATE(2026,12,31))
```
Mark it as a Date Table (Modeling → Mark as Date Table), then relate
`Date[Date]` to `transactions[transaction_date]` (1-to-many).

## 3. Add the DAX measures

Create an empty table `_Measures` (Enter Data, no columns needed) to hold all
measures, then paste in every measure from `DAX_measures.txt` (Modeling → New
Measure, one at a time, or via Tabular Editor if you have it installed).

## 4. Build the 4 report pages

**Page 1 — Executive Overview**
- 5 KPI cards across the top: `[Total Customers]`, `[MRR (EUR)]`,
  `[Churn Rate %]`, `[Revenue at Risk (EUR)]`, `[Avg Predicted CLV (EUR)]`
- Line chart: `vw_mrr_by_month[month]` (axis) vs `vw_mrr_by_month[mrr]` (value)
- Donut chart: `ml_predictions[risk_tier]` vs `[High Risk Customers]`

**Page 2 — Churn Driver Deep Dive**
- Clustered bar: `vw_churn_rate_by_contract[contract_type]` vs `churn_rate_pct`
- Clustered bar: `vw_churn_rate_by_internet[internet_service]` vs `churn_rate_pct`
- Clustered bar: `vw_churn_rate_by_payment[payment_method]` vs `churn_rate_pct`
- Line/column combo: `vw_tenure_buckets[tenure_bucket]` vs `churn_rate_pct` +
  `avg_monthly_charges`
- Card: `[Avg Support Tickets per Customer]`, `[Avg Satisfaction Score]`

**Page 3 — High-Risk Customer List**
- Table visual: `customer_id`, `city`, `contract_type`, `monthly_charges`,
  `churn_probability`, `risk_tier`, `predicted_clv`
- Filter pane: `risk_tier = "High"`, sort by `churn_probability` descending
- Conditional formatting on `churn_probability` (red/yellow/green data bars)
- Slicers: `contract_type`, `internet_service`, `state`

**Page 4 — CLV & Segmentation**
- Scatter plot: `tenure_months` (x) vs `predicted_clv` (y), colored by
  `segment_cluster`, size by `monthly_charges`
- Bar chart: segment vs `[Customers per Segment]`
- Bar chart: segment vs `[Avg CLV by Segment (EUR)]`
- Table: `vw_cohort_retention` as a matrix (cohort_month rows,
  months_since_signup columns, values = customers_active_at_month) for a
  retention-curve heatmap.

## 5. Polish for the portfolio

- Theme: View → Themes → pick a clean corporate theme (avoid default colors).
- Add a title text box on each page and a page-navigation button bar.
- File → Export → Export to PDF, and also take a PNG screenshot of each page
  for the GitHub README.
- Publish to the Power BI Service (if you have a free account) so you can
  share a live link, or just ship the `.pbix` file + screenshots in the repo
  under `powerbi/screenshots/`.
