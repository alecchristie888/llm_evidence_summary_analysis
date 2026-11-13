# ============================================================================
# LLM-as-Judge Full Dataset Analysis
# ----------------------------------------------------------------------------
# Inputs:
#   - summary_level.csv     : 1991 rows (one per summary = synopsis x question
#                             x summary_model). Pre-aggregated by the Python
#                             pipeline. This is the primary unit of analysis.
#   - statement_level.csv   : 23,641 rows (one per cited_statement). Useful
#                             for citation-behaviour diagnostics and for
#                             per-claim failure mode analysis.
#   - failure_reasons.csv   : 6,854 rows (one per faithfulness failure_reason
#                             tag). Long-format for tallying error types.
#
# Outputs (saved to data/derived/):
#   Tables (CSV):
#     - model_headline.csv          : model-level means + bootstrap 95% CIs
#     - synopsis_x_model.csv        : per-cell means for heatmaps
#     - summary_distribution.csv    : per-model performance bands
#     - variance_decomp.csv         : ANOVA: variance explained by model/topic
#     - pairwise_differences.csv    : bootstrap CIs for model-vs-model diffs
#     - failure_reason_profile.csv  : failure reason tallies per model
#     - outliers_top_bottom.csv     : best/worst 5 summaries per model x metric
#
#   Figures: none here; all figures are built by 05_analysis/figures.R from
#   these tables.
#
# Run from the repository root:  Rscript 05_analysis/analysis.R
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(stringr)
  library(forcats)
  library(purrr)
  library(tibble)
  library(scales)
  library(broom)
  library(viridisLite)
})

# Run from the repository root. If started from 05_analysis/ (or interactively from there), step up one level.
if (!dir.exists("data/derived") && dir.exists("../data/derived")) setwd("..")
if (!dir.exists("data/derived")) stop("Run from the repository root (the folder containing run_all.sh and data/); current directory: ", getwd())
read_derived <- function(name) {
  path <- file.path(Sys.getenv("IGF_DERIVED", "data/derived"), name)
  if (!file.exists(path) || file.size(path) < 10) stop(path, " is missing or empty: run `python 05_analysis/build_score_tables.py` (and the earlier stages in run_all.sh) first.")
  readr::read_csv(path, show_col_types = FALSE)
}

# viridis colour scale shorthand (we only use viridisLite)
scale_fill_viridis <- function(...) scale_fill_gradientn(colours = viridis(256), ...)

# Reproducibility
set.seed(42)

# Output theme — clean, journal-friendly
theme_set(
  theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.title         = element_text(face = "bold", size = 12),
      plot.subtitle      = element_text(colour = "grey30", size = 10),
      strip.text         = element_text(face = "bold"),
      legend.position    = "top"
    )
)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
dir.create("data/derived", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Friendly short model labels
model_relabel <- c(
  "anthropic/claude-sonnet-4" = "Claude Sonnet 4",
  "google/gemini-2.5-pro"     = "Gemini 2.5 Pro",
  "openai/gpt-5"              = "GPT-5",
  "moonshotai/kimi-k2-0905"   = "Kimi K2"
)
model_order <- c("Claude Sonnet 4", "Gemini 2.5 Pro", "GPT-5", "Kimi K2")

df_summ <- read_derived("summary_level.csv")
df_stmt <- read_derived("statement_level.csv")
df_fr   <- read_derived("failure_reasons.csv") |>
  mutate(model = factor(recode(summary_model, !!!model_relabel),
                        levels = model_order))

df_claims <- read_derived("claims_long.csv") |>
  mutate(model = factor(recode(summary_model, !!!model_relabel),
                        levels = model_order))

# Claim-level failure rate per model (no double-count from multi-tag claims)
claim_failure_rate <- df_claims |>
  group_by(model) |>
  summarise(
    n_claims        = n(),
    n_failed_claims = sum(support_assessment == "not_fully_supported"),
    claim_fail_rate = n_failed_claims / n_claims,
    .groups = "drop"
  )

write_csv(claim_failure_rate, "data/derived/claim_failure_rate.csv")


df_summ <- df_summ |>
  mutate(model = factor(recode(summary_model, !!!model_relabel),
                        levels = model_order))
df_stmt <- df_stmt |>
  mutate(model = factor(recode(summary_model, !!!model_relabel),
                        levels = model_order))
df_fr   <- df_fr |>
  mutate(model = factor(recode(summary_model, !!!model_relabel),
                        levels = model_order))

# Tidy metric column names
metric_levels <- c("relevance_score", "citation_rate",
                   "cc_all_score", "cc_cited_score", "faithfulness_score")
metric_labels <- c("Relevance", "Citation rate",
                   "CC (all stmts)", "CC (when cited)", "Faithfulness")

df_summ_long <- df_summ |>
  pivot_longer(all_of(metric_levels), names_to = "metric", values_to = "score") |>
  mutate(metric = factor(metric, levels = metric_levels, labels = metric_labels))

# ----------------------------------------------------------------------------
# Helper: bootstrap CI for a mean (resample summaries, the experimental unit)
# ----------------------------------------------------------------------------
boot_ci <- function(x, R = 2000, conf = 0.95) {
  x <- x[!is.na(x)]
  if (length(x) < 2) return(c(mean = mean(x, na.rm = TRUE), lo = NA, hi = NA))
  boots <- replicate(R, mean(sample(x, length(x), replace = TRUE)))
  q <- quantile(boots, c((1 - conf)/2, 1 - (1 - conf)/2))
  c(mean = mean(x), lo = unname(q[1]), hi = unname(q[2]))
}

# ============================================================================
# 1. HEADLINE TABLE — model-level means with bootstrap 95% CIs
# ============================================================================
cat("\n--- 1. Headline model-level table ---\n")

headline <- df_summ_long |>
  filter(!is.na(score)) |>
  group_by(model, metric) |>
  summarise(
    n_summaries = n(),
    ci = list(boot_ci(score)),
    .groups = "drop"
  ) |>
  unnest_wider(ci) |>
  mutate(across(c(mean, lo, hi), ~ round(.x, 4))) |>
  arrange(metric, model)

write_csv(headline, "data/derived/model_headline.csv")

# Print compact table
headline |>
  mutate(disp = sprintf("%.1f%% [%.1f, %.1f]", mean*100, lo*100, hi*100)) |>
  select(model, metric, disp) |>
  pivot_wider(names_from = metric, values_from = disp) |>
  print()

# ============================================================================
# 2. PER-SUMMARY DISTRIBUTION — variance and performance bands
# ============================================================================
cat("\n--- 2. Per-summary distributions ---\n")

# Distribution stats per model x metric
distr <- df_summ_long |>
  filter(!is.na(score)) |>
  group_by(model, metric) |>
  summarise(
    n       = n(),
    mean    = mean(score),
    median  = median(score),
    sd      = sd(score),
    iqr     = IQR(score),
    cv      = sd / mean,
    pct_ge_90 = mean(score >= 0.90),    # "really good" summaries
    pct_ge_80 = mean(score >= 0.80),
    pct_lt_70 = mean(score <  0.70),    # "poor" summaries
    pct_lt_50 = mean(score <  0.50),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), ~ round(.x, 4)))

write_csv(distr, "data/derived/summary_distribution.csv")
print(distr)

# ============================================================================
# 3. SYNOPSIS x MODEL HEATMAP DATA
# ============================================================================
cat("\n--- 3. Synopsis x Model cells ---\n")

synopsis_x_model <- df_summ_long |>
  filter(!is.na(score)) |>
  group_by(model, synopsis, metric) |>
  summarise(mean_score = mean(score),
            n_summaries = n(),
            .groups = "drop") |>
  mutate(mean_score = round(mean_score, 4))

write_csv(synopsis_x_model, "data/derived/synopsis_x_model.csv")

# ============================================================================
# 4. VARIANCE DECOMPOSITION — how much variance is model vs synopsis?
# ============================================================================
# Fit: score ~ model + synopsis + model:synopsis, on summary-level data.
# Report eta-squared per term.
# ============================================================================
cat("\n--- 4. Variance decomposition ---\n")

eta_sq_table <- function(df, metric_name) {
  d <- df |> filter(metric == metric_name, !is.na(score))
  fit <- aov(score ~ model + synopsis + model:synopsis, data = d)
  ss <- summary(fit)[[1]][, "Sum Sq"]
  terms <- rownames(summary(fit)[[1]])
  total <- sum(ss)
  tibble(metric = metric_name,
         term = trimws(terms),
         ss = ss,
         eta_sq = ss / total)
}

variance_decomp <- map_dfr(metric_labels, ~ eta_sq_table(df_summ_long, .x)) |>
  mutate(eta_sq = round(eta_sq, 4))

write_csv(variance_decomp, "data/derived/variance_decomp.csv")
print(variance_decomp |> pivot_wider(id_cols = term, names_from = metric, values_from = eta_sq))

# ============================================================================
# 5. PAIRWISE MODEL DIFFERENCES — bootstrap 95% CIs
# ============================================================================
cat("\n--- 5. Pairwise model differences ---\n")

boot_diff_ci <- function(x, y, R = 2000) {
  x <- x[!is.na(x)]; y <- y[!is.na(y)]
  if (length(x) < 2 || length(y) < 2) return(c(diff = NA, lo = NA, hi = NA))
  diffs <- replicate(R, mean(sample(x, length(x), replace = TRUE)) -
                        mean(sample(y, length(y), replace = TRUE)))
  q <- quantile(diffs, c(0.025, 0.975))
  c(diff = mean(x) - mean(y), lo = unname(q[1]), hi = unname(q[2]))
}

pair_rows <- list()
for (mt in metric_labels) {
  scores_by_model <- df_summ_long |>
    filter(metric == mt) |>
    group_by(model) |>
    summarise(scores = list(score), .groups = "drop")
  models <- as.character(scores_by_model$model)
  for (i in seq_along(models)) for (j in seq_along(models)) {
    if (i == j) next
    ci <- boot_diff_ci(scores_by_model$scores[[i]], scores_by_model$scores[[j]])
    pair_rows[[length(pair_rows)+1]] <- tibble(
      metric = mt, model_a = models[i], model_b = models[j],
      diff = unname(ci["diff"]), lo = unname(ci["lo"]), hi = unname(ci["hi"]),
      sig = sign(unname(ci["lo"])) == sign(unname(ci["hi"]))
    )
  }
}
pairwise <- bind_rows(pair_rows) |>
  mutate(across(c(diff, lo, hi), ~ round(.x, 4)))
write_csv(pairwise, "data/derived/pairwise_differences.csv")

# ============================================================================
# 6. FAILURE REASON PROFILE
# ============================================================================

# 1) Statements per summary (denominator for each summary's tag rate)
stmts_per_summary <- df_stmt |>
  filter(!is.na(faith_verdict)) |>
  count(model, synopsis, question, name = "n_stmts")

# 2) Tag counts per (summary, failure_reason)
tags_per_summary <- df_fr |>
  count(model, synopsis, question, failure_reason, name = "n_tags")

# 3) Full grid: every (summary x reason) combo, missing pairs filled with 0
all_reasons <- unique(df_fr$failure_reason)
fr_summary_rates <- stmts_per_summary |>
  tidyr::crossing(failure_reason = all_reasons) |>
  left_join(tags_per_summary,
            by = c("model", "synopsis", "question", "failure_reason")) |>
  mutate(n_tags = coalesce(n_tags, 0L),
         rate_in_summary = n_tags / n_stmts)

# 4) Summary-averaged rate per (model, failure_reason) with bootstrap 95% CI
boot_mean_ci <- function(x, R = 1000, alpha = 0.05) {
  if (length(x) < 2) return(c(mean(x), NA_real_, NA_real_))
  bs <- replicate(R, mean(sample(x, length(x), replace = TRUE)))
  c(mean(x), quantile(bs, alpha / 2), quantile(bs, 1 - alpha / 2))
}

fr_profile <- fr_summary_rates |>
  group_by(model, failure_reason) |>
  summarise(
    n_summaries = n(),
    n_tags_total = sum(n_tags),
    ci = list(boot_mean_ci(rate_in_summary)),
    .groups = "drop"
  ) |>
  mutate(mean_rate = vapply(ci, `[`, numeric(1), 1),
         lo        = vapply(ci, `[`, numeric(1), 2),
         hi        = vapply(ci, `[`, numeric(1), 3)) |>
  select(-ci) |>
  arrange(model, desc(mean_rate))

write_csv(fr_profile, "data/derived/failure_reason_profile.csv")
print(fr_profile |>
        group_by(model) |>
        slice_max(mean_rate, n = 6) |>
        select(model, failure_reason, mean_rate, lo, hi, n_tags_total))

# ============================================================================
# 7. OUTLIERS — best/worst 5 summaries per model x metric
# ============================================================================
cat("\n--- 7. Outlier summaries ---\n")

outliers <- df_summ_long |>
  filter(!is.na(score)) |>
  group_by(model, metric) |>
  arrange(score) |>
  mutate(rank_lo = row_number(),
         rank_hi = n() - row_number() + 1) |>
  filter(rank_lo <= 5 | rank_hi <= 5) |>
  mutate(direction = ifelse(rank_lo <= 5, "worst", "best")) |>
  select(model, metric, direction, synopsis, question, score, n_statements) |>
  arrange(model, metric, direction, score)

write_csv(outliers, "data/derived/outliers_top_bottom.csv")

# ============================================================================
# FIGURES
# ============================================================================

cat("\nTables written to data/derived/. Figures are built by 05_analysis/figures.R.\n")
