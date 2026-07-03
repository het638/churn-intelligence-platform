# Customer Churn - Statistical Analysis (base R only, no CRAN packages required)
# Complements the ML layer with inferential statistics: a logistic regression
# with odds ratios, chi-square tests of association, and t-tests on churn drivers.

# Run this script with the working directory set to the project root, e.g.:
#   Rscript r_stats/churn_statistical_analysis.R
data_path <- file.path("data", "processed", "features_scored.csv")
out_dir <- file.path("r_stats", "output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(data_path, stringsAsFactors = TRUE)
df$churned <- as.integer(df$churned)

report <- file(file.path(out_dir, "statistical_report.txt"), open = "w")
say <- function(...) {
  cat(..., "\n", file = report, append = TRUE)
  cat(..., "\n")
}

say("=== Customer Churn & Revenue Intelligence Platform ===")
say("=== Statistical Analysis Report (R) ===\n")

# ---------------------------------------------------------------
# 1. Logistic regression: churn ~ business drivers
# ---------------------------------------------------------------
say("--- 1. Logistic Regression: churned ~ drivers ---\n")

model <- glm(
  churned ~ contract_type + payment_method + internet_service + tech_support +
    monthly_charges + tenure_months + senior_citizen + ticket_count,
  data = df, family = binomial(link = "logit")
)
model_summary <- summary(model)
say(capture.output(print(model_summary)))

# odds ratios + 95% CI
say("\n-- Odds Ratios (exp(coef)) --")
or_table <- exp(cbind(OddsRatio = coef(model), confint.default(model)))
say(capture.output(print(round(or_table, 3))))

say(sprintf("\nMcFadden's pseudo R2: %.4f",
            1 - model$deviance / model$null.deviance))

# ---------------------------------------------------------------
# 2. Chi-square tests of association (categorical drivers vs churn)
# ---------------------------------------------------------------
say("\n--- 2. Chi-square tests of independence ---\n")

for (var in c("contract_type", "internet_service", "payment_method", "tech_support")) {
  tbl <- table(df[[var]], df$churned)
  test <- chisq.test(tbl)
  say(sprintf("churned x %s: X2 = %.2f, df = %d, p-value = %.6g",
              var, test$statistic, test$parameter, test$p.value))
}

# ---------------------------------------------------------------
# 3. t-tests: continuous drivers, churned vs retained
# ---------------------------------------------------------------
say("\n--- 3. Welch two-sample t-tests (churned vs retained) ---\n")

for (var in c("monthly_charges", "tenure_months", "ticket_count")) {
  t <- t.test(df[[var]] ~ df$churned)
  means <- tapply(df[[var]], df$churned, mean)
  say(sprintf(
    "%s: mean(retained)=%.2f, mean(churned)=%.2f, t=%.2f, df=%.1f, p-value=%.6g",
    var, means["0"], means["1"], t$statistic, t$parameter, t$p.value
  ))
}

close(report)

# ---------------------------------------------------------------
# Plot: churn rate by contract type (base R, no ggplot2 needed)
# ---------------------------------------------------------------
png(file.path(out_dir, "churn_rate_by_contract.png"), width = 700, height = 500)
rates <- tapply(df$churned, df$contract_type, mean) * 100
bp <- barplot(rates, col = "#2c7fb8", ylim = c(0, max(rates) * 1.25),
              main = "Churn Rate by Contract Type", ylab = "Churn rate (%)")
text(bp, rates, labels = sprintf("%.1f%%", rates), pos = 3)
dev.off()

cat("\nSaved report + plot to:", out_dir, "\n")
