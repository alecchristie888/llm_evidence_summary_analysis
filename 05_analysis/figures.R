# ============================================================================
# All figures for "Faithful but incomplete" (main Figs 1–4 and Extended Data
# Figs 1–2), built from the derived tables written by build_score_tables.py,
# analysis.R and reframed_metrics.R. No API calls.
#
# Inputs (data/derived/)
#   summary_level.csv                    judge scores, one row per evaluated summary (build_score_tables.py)
#   statement_level.csv                  statements with judge verdicts (build_score_tables.py)
#   failure_reasons.csv                  one row per failure-reason tag (build_failure_reasons.py)
#   model_headline.csv                   judge metrics: mean + bootstrap 95% CI (analysis.R)
#   synopsis_x_model.csv                 per subject-area x model means (analysis.R)
#   attrition_by_model.csv               questions never completed (recheck_traces.py)
#   retrieval_per_summary.csv            per-summary judge scores + retrieval metrics (reframed_metrics.R)
#   retrieval_metrics_by_model_evaluated_run.csv  retrieval means + CIs, percent (reframed_metrics.R)
#   metric_ranks.csv                     rank of each model on every metric (reframed_metrics.R)
#
# Outputs (outputs/figures/)                             companion tables (data/derived/)
#   fig1_metric_profile.png     eight-metric profile        fig1_metric_profile_summary.csv
#   fig2_ranks_and_coverage.png ranks + faithfulness vs recall
#   fig3_distributions.png      ECDFs of six metrics        fig3_distribution_medians.csv
#   fig4_failure_profile.png    failure-reason sea stacks   fig4_failure_profile_summary.csv,
#                                                           fig4_tag_count_shares.csv, failure_rate_per_summary.csv
#   ed_fig1_cite_quality_scatter.png  citation rate vs correctness when cited, per summary
#   ed_fig2_synopsis_heatmap.png      subject area x model heatmap
#
# Each figure is built inside its own local() block so the sections are
# independent; comment one out to skip it. Run from the repository root:
#   Rscript 05_analysis/figures.R
# ============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(purrr); library(stringr); library(forcats)
  library(tibble); library(scales); library(ggplot2); library(patchwork)
})

# Run from the repository root. If started from 05_analysis/ (or interactively from there), step up one level.
if (!dir.exists("data/derived") && dir.exists("../data/derived")) setwd("..")
if (!dir.exists("data/derived")) stop("Run from the repository root (the folder containing run_all.sh and data/); current directory: ", getwd())
read_derived <- function(name) {
  path <- file.path(Sys.getenv("IGF_DERIVED", "data/derived"), name)
  if (!file.exists(path) || file.size(path) < 10) stop(path, " is missing or empty: run `python 05_analysis/build_score_tables.py` (and the earlier stages in run_all.sh) first.")
  readr::read_csv(path, show_col_types = FALSE)
}

D      <- Sys.getenv("IGF_DERIVED", "data/derived")
FIGDIR <- "outputs/figures"; dir.create(FIGDIR, recursive = TRUE, showWarnings = FALSE)
N_BOOT <- 1000

# ---- shared model palette (Okabe & Ito colour-blind-safe set) ---------------
MODEL_COLS <- c("Claude Sonnet 4" = "#D55E00",   # vermillion
                "Gemini 2.5 Pro"  = "#009E73",   # bluish green
                "GPT-5"           = "#0072B2",   # blue
                "Kimi K2"         = "#CC79A7")   # reddish purple
MODEL_ORDER <- names(MODEL_COLS)
MODEL_LABEL <- c("anthropic/claude-sonnet-4" = "Claude Sonnet 4", "google/gemini-2.5-pro" = "Gemini 2.5 Pro",
                 "openai/gpt-5" = "GPT-5", "moonshotai/kimi-k2-0905" = "Kimi K2")
LABEL <- MODEL_LABEL
COLS  <- MODEL_COLS

# ---- shared helpers ----------------------------------------------------------
boot_mean_ci <- function(x, n = N_BOOT) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(mean = NA, lo = NA, hi = NA))
  idx   <- matrix(sample.int(length(x), n * length(x), replace = TRUE), nrow = n)
  means <- rowMeans(matrix(x[idx], nrow = n))
  c(mean = mean(x), lo = unname(quantile(means, 0.025)), hi = unname(quantile(means, 0.975)))
}
wilson <- function(k, n, z = 1.96) {
  p <- k / n; den <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / den; half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  c(lo = centre - half, hi = centre + half)
}

# ============================================================================
# Figure 1: eight-metric profile with per-summary points
# The full multi-metric profile of the four summariser LLMs, grouped by the
# failure class each metric detects (Box 1). Every per-summary score is a point;
# mean and bootstrap 95% CI overlaid. Task completion (a per-question binary
# outcome with no per-summary distribution) is not shown here; it appears in
# Fig. 2a and Extended Data Table 2. Its place is taken by citation recall,
# the share of withheld relevant actions that the summary actually cites.
# ============================================================================
local({
  set.seed(7)  # jitter only
  
  # panel order and grouping follow Box 1 of the manuscript
  CLASS <- c("Relevance" = "Relevance", "Retrieval recall" = "Coverage:", "Retrieval precision" = "Coverage:",
             "Citation recall" = "Coverage:", "Citation rate" = "Attribution:",
             "Citation correctness (when cited)" = "Attribution:", "Cite-through rate" = "Attribution:",
             "Faithfulness" = "Faithfulness")
  METRIC_ORDER <- names(CLASS)
  panel_label <- function(metric) ifelse(CLASS[metric] == metric, paste0(metric), paste0(CLASS[metric], " ", metric))
  PANEL_LEVELS <- panel_label(METRIC_ORDER)
  as_panel <- function(metric) factor(panel_label(metric), levels = PANEL_LEVELS)
  
  # --- raw per-summary values (percent) -----------------------------------------
  per <- read_csv(file.path(D, "retrieval_per_summary.csv"), show_col_types = FALSE)
  raw <- per |>
    transmute(model,
              `Relevance` = 100 * relevance_score,
              `Retrieval recall` = 100 * recall_run, `Retrieval precision` = 100 * precision_run,
              `Citation recall` = 100 * cite_recall_run,
              `Citation rate` = 100 * citation_rate,
              `Citation correctness (when cited)` = 100 * cc_cited_score,
              `Cite-through rate` = 100 * cite_through_run,
              `Faithfulness` = 100 * faithfulness_score) |>
    pivot_longer(-model, names_to = "metric", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(model = factor(model, levels = MODEL_ORDER), panel = as_panel(metric))
  
  # --- means and 95% CIs --------------------------------------------------------
  judge <- read_csv(file.path(D, "model_headline.csv"), show_col_types = FALSE) |>
    filter(metric %in% c("Relevance", "Citation rate", "CC (when cited)", "Faithfulness")) |>
    transmute(model, metric = recode(metric, "CC (when cited)" = "Citation correctness (when cited)"),
              mean = 100 * mean, lo = 100 * lo, hi = 100 * hi, n = n_summaries)
  
  retr <- read_csv(file.path(D, "retrieval_metrics_by_model_evaluated_run.csv"), show_col_types = FALSE) |>
    transmute(model, n,
              `Retrieval recall__mean` = ret_recall, `Retrieval recall__lo` = ret_recall_lo, `Retrieval recall__hi` = ret_recall_hi,
              `Retrieval precision__mean` = ret_prec, `Retrieval precision__lo` = ret_prec_lo, `Retrieval precision__hi` = ret_prec_hi,
              `Citation recall__mean` = cite_recall, `Citation recall__lo` = cite_recall_lo, `Citation recall__hi` = cite_recall_hi,
              `Cite-through rate__mean` = cite_through, `Cite-through rate__lo` = cite_through_lo, `Cite-through rate__hi` = cite_through_hi) |>
    pivot_longer(-c(model, n), names_to = c("metric", ".value"), names_sep = "__")
  
  stats <- bind_rows(judge, retr) |>
    mutate(class = CLASS[metric], model = factor(model, levels = MODEL_ORDER), panel = as_panel(metric),
           label = sprintf("%.1f", mean))
  write_csv(stats |> arrange(panel, model) |> mutate(metric = metric, across(c(mean, lo, hi), ~ round(.x, 2))) |> select(class, metric, model, n, mean, lo, hi),
            file.path(D, "fig1_metric_profile_summary.csv"))
  
  # --- figure -------------------------------------------------------------------
  p <- ggplot() +
    geom_jitter(data = raw, aes(model, value, colour = model), width = 0.30, height = 0.9,
                size = 0.55, alpha = 0.28, shape = 16, show.legend = FALSE) +
    geom_errorbar(data = stats, aes(model, ymin = lo, ymax = hi), width = 0.28, linewidth = 0.55, colour = "grey10") +
    geom_point(data = stats, aes(model, mean, fill = model), shape = 21, size = 2.8, colour = "grey10", stroke = 0.6) +
    geom_text(data = stats, aes(model, y = 106, label = label, colour = model), size = 3.75, fontface = "bold",
              show.legend = FALSE) +
    facet_wrap(~ panel, nrow = 4) +
    scale_colour_manual(values = COLS, guide = "none") +
    scale_fill_manual(values = COLS, name = "Model:",
                      guide = guide_legend(override.aes = list(shape = 21, size = 3.2))) +
    scale_y_continuous(limits = c(-3, 113), breaks = seq(0, 100, 25), expand = c(0, 0),
                       labels = function(x) paste0(x, "%")) +
    labs(x = NULL, y = "Score per summary, mean and 95% CI")+
    theme_minimal(base_size = 15) +
    theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          strip.text = element_text(face = "bold", size = 14, hjust = 0, vjust = 0, margin = margin(t = 2, b = 2), lineheight = 1),
          legend.position = "bottom", legend.margin = margin(t = -4),
          panel.spacing.x = unit(1.0, "lines"), panel.spacing.y = unit(1.0, "lines"),
          plot.background = element_rect(fill = "white", colour = NA))
  
  ggsave(file.path(FIGDIR, "fig1_metric_profile.png"), p, width = 10, height = 10, dpi = 600, bg = "white")
  cat(sprintf("Saved %s/fig1_metric_profile.png: %d raw points, %d mean/CI markers\n", FIGDIR, nrow(raw), nrow(stats)))
})

# ============================================================================
# Figure 2: ranks by metric and faithfulness against retrieval recall
# (a) rank of each model on every metric (bump chart); (b) faithfulness against
# retrieval recall, one facet per model: its summaries as points, its mean with
# 95% CI as a large marker, other models' means as grey crosses.
# ============================================================================
local({
  set.seed(1)
  cols <- MODEL_COLS
  # rebuilt from the tables written by reframed_metrics.R
  per <- read_csv(file.path(D, "retrieval_per_summary.csv"), show_col_types = FALSE)
  rank_long <- read_csv(file.path(D, "metric_ranks.csv"), show_col_types = FALSE) |>
    select(model, ends_with("(rank)")) |>
    pivot_longer(-model, names_to = "metric", values_to = "rank") |>
    mutate(metric = sub(" \\(rank\\)$", "", metric))
  # metric order follows Fig. 1 (Box 1 classes), then task completion and the composite
  order <- c("Relevance", "Retrieval recall", "Retrieval precision", "Citation recall", "Citation rate", "CC (when cited)",
             "Cite-through rate", "Faithfulness","Composite (retrieval recall x faithfulness)")
  short <- c("Relevance", "Retrieval\nrecall", "Retrieval\nprecision", "Citation\nrecall", "Citation\nrate",
             "Citation\ncorrectness\n(when cited)", "Cite-through\nrate", "Faithfulness","Composite\n(retrieval recall\nx faithfulness)")
  classes <- tibble(class = c("Relevance", "Coverage", "Attribution", "Faithfulness","Composite"),
                    xmin = c(0.5, 1.5, 4.5, 7.5, 8.5), xmax = c(1.5, 4.5, 7.5, 8.5, 9.5))
  
  pa <- rank_long |>
    filter(metric %in% order) |>
    mutate(metric = factor(metric, levels = order, labels = short)) |>
    ggplot(aes(as.integer(metric), rank, group = model, colour = model)) +
    annotate("rect", xmin = length(order) - 0.5, xmax = length(order) + 0.5, ymin = 0.35, ymax = 4.3, fill = "grey92") +
    geom_vline(xintercept = classes$xmin[-1], colour = "grey85", linewidth = 0.4) +
    geom_text(data = classes, aes(x = (xmin + xmax) / 2, y = 0.5, label = class), inherit.aes = FALSE,
              size = 3.75, fontface = "bold", colour = "grey30") +
    geom_line(linewidth = 1) + geom_point(size = 2.6) +
    scale_x_continuous(breaks = seq_along(short), labels = short, expand = expansion(add = 0.5)) +
    scale_y_reverse(breaks = 1:4, limits = c(4.3, 0.35)) +
    scale_colour_manual(values = cols, name = "Model:", breaks = MODEL_ORDER) +
    labs(x = NULL, y = "Rank (1 = best)", title = "a.  The 'best' model depends on the metric") +
    theme_minimal(base_size = 15) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          plot.title = element_text(face = "bold"), legend.position = "bottom")
  
  pts <- per |>
    group_by(model) |>
    summarise(rx = list(boot_mean_ci(recall_run)), fy = list(boot_mean_ci(faithfulness_score)), .groups = "drop") |>
    mutate(recall = map_dbl(rx, "mean") * 100, recall_lo = map_dbl(rx, "lo") * 100, recall_hi = map_dbl(rx, "hi") * 100,
           faith = map_dbl(fy, "mean") * 100, faith_lo = map_dbl(fy, "lo") * 100, faith_hi = map_dbl(fy, "hi") * 100)
  
  raw_b <- per |>
    filter(!is.na(recall_run), !is.na(faithfulness_score)) |>
    transmute(model, recall = 100 * recall_run, faith = 100 * faithfulness_score)
  
  # one panel per model: its own summaries as points, its mean with 95% CI as a large marker,
  # and the other three models' means as small grey markers for reference
  others <- tidyr::crossing(panel_model = pts$model, pts |> rename(ref_model = model)) |>
    filter(panel_model != ref_model) |> rename(model = panel_model)
  pts_f  <- pts |> mutate(model = factor(model, levels = names(cols)))
  raw_b  <- raw_b |> mutate(model = factor(model, levels = names(cols)))
  others <- others |> mutate(model = factor(model, levels = names(cols)))
  
  pb <- ggplot() +
    geom_jitter(data = raw_b, aes(recall, faith, colour = model), width = 1.5, height = 1.5,
                size = 1.5, alpha = 0.3, shape = 16, show.legend = FALSE) +
    geom_point(data = others, aes(recall, faith), shape = 4, size =  2.8, colour = "grey45", stroke = 0.6) +
    geom_errorbar(data = pts_f, aes(xmin = recall_lo, xmax = recall_hi, y = faith), width = 2.5, orientation = "y",
                  colour = "grey10", linewidth = 0.5) +
    geom_errorbar(data = pts_f, aes(x = recall, ymin = faith_lo, ymax = faith_hi), width = 2.5,
                  colour = "grey10", linewidth = 0.5) +
    geom_point(data = pts_f, aes(recall, faith, fill = model), shape = 21, size = 3.5, colour = "grey10", stroke = 0.7,
               show.legend = FALSE) +
    geom_text(data = pts_f, aes(x = 2, y = 6, label = sprintf("recall %.1f%%, faithfulness %.1f%%", recall, faith)),
              hjust = 0, size = 3.75, colour = "grey25") +
    facet_wrap(~ model, nrow = 2) +
    scale_colour_manual(values = cols, guide = "none") +
    scale_fill_manual(values = cols, guide = "none") +
    scale_x_continuous(limits = c(-3, 103), breaks = seq(0, 100, 25), expand = c(0, 0)) +
    scale_y_continuous(limits = c(-3, 103), breaks = seq(0, 100, 25), expand = c(0, 0)) +
    labs(x = "Retrieval recall (% of relevant evidence sources opened)",
         y = "Faithfulness (% statements supported)",
         title = "b.  Faithfulness rewards omission")+
    theme_minimal(base_size = 15) +
    theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"),
          plot.subtitle = element_text(colour = "grey35", size = 10),
          strip.text = element_text(face = "bold", hjust = 0), panel.spacing = unit(1, "lines"))
  
  fig2 <- pa + pb + plot_layout(heights = c(1, 1.2))
  ggsave(file.path(FIGDIR, "fig2_ranks_and_coverage.png"), fig2, width = 10, height = 10, dpi = 600, bg = "white")
  cat(sprintf("\nSaved %s/fig2_ranks_and_coverage.png\n", FIGDIR))
})

# ============================================================================
# Figure 3: per-summary distributions (ECDFs)
# Empirical cumulative distributions of per-summary scores: one panel per metric
# (Box 1 order), one curve per model, medians as diamonds, thresholds (50, 70,
# 90%) as reference lines. Retrieval recall uses the evaluated-run definition.
# ============================================================================
local({
  METRICS <- c(`Relevance` = "relevance_score", `Retrieval recall` = "recall_run",
               `Citation rate` = "citation_rate", `Citation correctness (all statements)` = "cc_all_score",
               `Citation correctness (when cited)` = "cc_cited_score", `Faithfulness` = "faithfulness_score")
  THRESH <- c(50, 70, 90)
  
  per <- read_csv(file.path(D, "retrieval_per_summary.csv"), show_col_types = FALSE)
  long <- per |>
    select(model, all_of(unname(METRICS))) |>
    pivot_longer(-model, names_to = "col", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(value = 100 * value,
           metric = factor(names(METRICS)[match(col, METRICS)], levels = names(METRICS)),
           model = factor(model, levels = names(COLS)))
  
  meds <- long |>
    group_by(metric, model) |>
    summarise(n = n(), median = median(value), p_lt_50 = mean(value < 50), p_lt_70 = mean(value < 70),
              p_ge_90 = mean(value >= 90), .groups = "drop") |>
    mutate(ecdf_at_median = 0.5)
  write_csv(meds |> mutate(across(c(median, p_lt_50, p_lt_70, p_ge_90), ~ round(.x, 4))) |> select(-ecdf_at_median),
            file.path(D, "fig3_distribution_medians.csv"))
  
  thr <- tidyr::crossing(metric = factor(names(METRICS), levels = names(METRICS)), x = THRESH)
  
  p <- ggplot(long, aes(value, colour = model)) +
    #geom_vline(data = thr, aes(xintercept = x), colour = "grey80", linetype = "22", linewidth = 0.4) +
    #geom_text(data = thr |> filter(metric == levels(thr$metric)[1]), aes(x = x, y = 1.06, label = paste0(x, "%")),
    #          colour = "grey45", size = 2.5, inherit.aes = FALSE) +
    stat_ecdf(geom = "step", linewidth = 0.8, pad = TRUE) +
    geom_point(data = meds, aes(median, ecdf_at_median, fill = model), shape = 23, size = 2.8, colour = "grey15",
               stroke = 0.5, show.legend = FALSE) +
    facet_wrap(~ metric, nrow = 2) +
    scale_colour_manual(values = COLS, name = "Model:") +
    scale_fill_manual(values = COLS, guide = "none") +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20), labels = function(x) paste0(x, "%")) +
    scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2), labels = function(y) paste0(100 * y, "%"),
                       expand = c(0, 0)) +
    labs(x = "Per-summary score", y = "Cumulative proportion of summaries") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(), axis.title = element_text(size=12),
          strip.text = element_text(face = "bold", size=13, hjust = 0),
          legend.text = element_text(size=11),
          legend.position = "bottom", legend.margin = margin(t = -4), panel.spacing = unit(1.1, "lines"),
          plot.caption = element_text(hjust = 0, colour = "grey35", size = 7.5),
          plot.background = element_rect(fill = "white", colour = NA))
  
  ggsave(file.path(FIGDIR, "fig3_distributions.png"), p, width = 11, height = 6.4, dpi = 300, bg = "white")
  cat(sprintf("Saved %s/fig3_distributions.png (%d summaries x %d metrics)\n", FIGDIR, nrow(per), length(METRICS)))
})

# ============================================================================
# Figure 4: failure-reason sea stack plots
# Why statements were judged unfaithful. Sea stack plots (Stuart et al. 2024,
# Ecol. Evol., https://doi.org/10.1002/ece3.11208; design as in the Al-Stu/
# seastackplot package): failure reasons down the y-axis (ordered by overall
# frequency) and, within each reason, one horizontal sea stack per model: a
# histogram of the number of statements per summary tagged with that reason
# (bin width 1; 7 or more pooled) with the mean (diamond) and bootstrap 95% CI
# (ticks) on the baseline. Counts share one axis. Drawn in plain ggplot2 so the
# CI is the same summary-level bootstrap used elsewhere (the package's
# horizontal orientation is also broken in v0.0.1.0). Per-100-statement rates
# are still written to failure_rate_per_summary.csv and match
# failure_reason_profile.csv (analysis.R).
# ============================================================================
local({
  set.seed(11)
  
  REASON_LABEL <- c(absent_or_contradicted_fact = "Absent or contradicted fact",
                    overgeneralisation = "Overgeneralisation",
                    wrong_quantity = "Wrong quantity",
                    wrong_effect_direction_or_magnitude = "Wrong effect direction or magnitude",
                    unsupported_comparative_claim = "Unsupported comparative claim",
                    unsupported_evidence_summary = "Unsupported evidence summary",
                    mismerged_studies = "Mismerged studies")
  
  
  stmts <- read_csv(file.path(D, "statement_level.csv"), show_col_types = FALSE) |>
    filter(!is.na(faith_verdict)) |>
    mutate(question = str_trim(question)) |>
    count(summary_model, synopsis, question, name = "n_stmts")
  tags <- read_csv(file.path(D, "failure_reasons.csv"), show_col_types = FALSE) |>
    mutate(question = str_trim(question)) |>
    count(summary_model, synopsis, question, failure_reason, name = "n_tags")
  
  rates <- stmts |>
    tidyr::crossing(failure_reason = names(REASON_LABEL)) |>
    left_join(tags, by = c("summary_model", "synopsis", "question", "failure_reason")) |>
    mutate(n_tags = coalesce(n_tags, 0L), rate_per_100 = 100 * n_tags / n_stmts,
           model = factor(unname(MODEL_LABEL[summary_model]), levels = MODEL_ORDER))
  write_csv(rates |> select(model, synopsis, question, n_stmts, failure_reason, n_tags, rate_per_100),
            file.path(D, "failure_rate_per_summary.csv"))
  
  stats <- rates |>
    group_by(model, failure_reason) |>
    summarise(n_summaries = n(), n_tags_total = sum(n_tags), n_affected = sum(n_tags > 0),
              ci = list(boot_mean_ci(n_tags)), rate100_ci = list(boot_mean_ci(rate_per_100)), .groups = "drop") |>
    mutate(mean_tags_per_summary = sapply(ci, `[[`, "mean"), lo = sapply(ci, `[[`, "lo"), hi = sapply(ci, `[[`, "hi"),
           pct_affected = 100 * n_affected / n_summaries,
           pct_affected_lo = 100 * mapply(function(k, n) wilson(k, n)[["lo"]], n_affected, n_summaries),
           pct_affected_hi = 100 * mapply(function(k, n) wilson(k, n)[["hi"]], n_affected, n_summaries),
           rate_per_100_mean = sapply(rate100_ci, `[[`, "mean")) |>
    select(-ci, -rate100_ci)
  write_csv(stats |> mutate(across(where(is.double), ~ round(.x, 3))), file.path(D, "fig4_failure_profile_summary.csv"))
  
  # order panels by overall frequency (most common reason first)
  reason_order <- stats |> group_by(failure_reason) |> summarise(t = sum(n_tags_total)) |> arrange(desc(t)) |> pull(failure_reason)
  lab <- function(r) factor(REASON_LABEL[r], levels = REASON_LABEL[reason_order])
  
  CATS <- c("0", "1", "2", "3+")
  tint <- function(hex, f) {  # mix a colour with white; f = 0 white, 1 full colour
    rgb <- col2rgb(hex) / 255; rgb(1 - f * (1 - rgb[1]), 1 - f * (1 - rgb[2]), 1 - f * (1 - rgb[3]))
  }
  TINT_F <- c(`0` = 0.10, `1` = 0.42, `2` = 0.70, `3+` = 1.00)
  
  shares <- rates |>
    mutate(cat = factor(ifelse(n_tags >= 3, "3+", as.character(n_tags)), levels = CATS)) |>
    count(model, failure_reason, cat, name = "n") |>
    group_by(model, failure_reason) |> mutate(share = n / sum(n)) |> ungroup() |>
    mutate(reason = lab(failure_reason),
           fill = mapply(function(m, c) tint(MODEL_COLS[[as.character(m)]], TINT_F[[as.character(c)]]), model, cat),
           reason = fct_rev(reason), model = factor(model, levels = MODEL_ORDER), cat = factor(cat, levels = rev(CATS)))
  write_csv(shares |> select(model, failure_reason, cat, n, share) |> mutate(share = round(share, 4)),
            file.path(D, "fig4_tag_count_shares.csv"))
  
  stats <- stats |> mutate(reason = fct_rev(lab(failure_reason)), model = factor(model, levels = MODEL_ORDER),
                           label = sprintf("%.0f%%  |  %.1f (%.1f\u2013%.1f)", pct_affected, mean_tags_per_summary, lo, hi))
  
  # legend: one row of swatches per count category, drawn in neutral grey tints
  legend_df <- tibble(cat = factor(CATS, levels = CATS), fill = sapply(TINT_F, function(f) tint("#4D4D4D", f)))
  
  # --- sea stacks: rows = failure reasons, columns = models -----------------------
  XMAX <- 7
  ss <- rates |>
    mutate(tags = pmin(n_tags, XMAX), reason = lab(failure_reason), model = factor(model, levels = MODEL_ORDER)) |>
    count(reason, model, tags, name = "n")
  ymax <- max(ss$n)
  TICK <- ymax * 0.18   # height of the CI ticks, on the count scale
  st <- stats |> mutate(reason = lab(failure_reason), model = factor(model, levels = MODEL_ORDER),
                        cell_label = sprintf("%.0f%% \u22651", pct_affected))
  
  p <- ggplot() +
    geom_col(data = ss, aes(tags, n, fill = model), width = 0.9, show.legend = FALSE) +
    geom_hline(yintercept = 0, colour = "grey20", linewidth = 0.4) +
    geom_segment(data = st, aes(x = lo, xend = lo, y = 0, yend = TICK), colour = "grey10", linewidth = 0.7) +
    geom_segment(data = st, aes(x = hi, xend = hi, y = 0, yend = TICK), colour = "grey10", linewidth = 0.7) +
    geom_point(data = st, aes(mean_tags_per_summary, 0), shape = 23, size = 2.6, fill = "white", colour = "grey10", stroke = 0.7) +
    geom_text(data = st, aes(x = XMAX + 0.5, y = ymax * 0.2, label = cell_label), hjust = 1, vjust = 1,
              size = 4, colour = "grey25", lineheight = 0.95) +
    facet_grid(reason ~ model, switch = "y", labeller = labeller(reason = label_wrap_gen(18))) +
    scale_fill_manual(values = MODEL_COLS) +
    scale_x_continuous(breaks = 0:XMAX, labels = c(0:(XMAX - 1), paste0(XMAX, "+")), expand = expansion(add = 0.6)) +
    scale_y_continuous(breaks = seq(0, 500, 150), expand = expansion(mult = c(0.08, 0.05))) +
    labs(x = "Number of statements in the summary tagged with the failure reason", y = "Number of summaries") +
    theme_minimal(base_size = 15) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
          strip.text.x = element_text(face = "bold", size = 12),
          strip.text.y.left = element_text(angle = 0, hjust = 1, size = 15, colour = "grey15"),
          strip.placement = "outside", panel.spacing.x = unit(1.1, "lines"), panel.spacing.y = unit(0.5, "lines"),
          axis.text = element_text(size = 12),
          #plot.caption = element_text(hjust = 0, colour = "grey35", size = 7.5),
          plot.background = element_rect(fill = "white", colour = NA))
  fig <- p
  ggsave(file.path(FIGDIR, "fig4_failure_profile.png"), fig, width = 10, height = 10, dpi = 600, bg = "white")
  cat(sprintf("Saved %s/fig4_failure_profile.png (%d summaries x %d reasons)\n", FIGDIR, nrow(stmts), length(REASON_LABEL)))
})

# ============================================================================
# Extended Data Figure 1: citation rate against correctness when cited
# Citation rate against citation correctness when cited, one point per summary,
# model mean as a large marker. (Formerly fig4_cite_quality_scatter.png.)
# ============================================================================
local({
  scale_fill_viridis <- function(...) scale_fill_gradientn(colours = viridisLite::viridis(256), ...)
  theme_set(theme_minimal(base_size = 11) +
              theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
                    plot.title = element_text(face = "bold", size = 12), plot.subtitle = element_text(colour = "grey30", size = 10),
                    strip.text = element_text(face = "bold"), legend.position = "top"))
  df_summ <- read_csv(file.path(D, "summary_level.csv"), show_col_types = FALSE) |>
    mutate(model = factor(unname(MODEL_LABEL[summary_model]), levels = MODEL_ORDER))
  means <- df_summ |>
    filter(!is.na(citation_rate), !is.na(cc_cited_score)) |>
    group_by(model) |>
    summarise(citation_rate = mean(citation_rate),
              cc_cited_score = mean(cc_cited_score), .groups = "drop")
  
  p <- df_summ |>
    filter(!is.na(citation_rate), !is.na(cc_cited_score)) |>
    ggplot(aes(x = citation_rate, y = cc_cited_score, colour = model)) +
    geom_point(alpha = 0.4, size = 1.4) +
    geom_point(data = means, aes(fill = model),
               shape = 21, size = 4, colour = "black", stroke = 0.5) +
    facet_wrap(~ model, nrow = 1) +
    scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    scale_colour_manual(values = MODEL_COLS, guide = "none") +
    scale_fill_manual(values = MODEL_COLS, guide = "none") +
    labs(#title = "Citation behaviour: quantity vs quality",
      #subtitle = "Each dot = one summary. Diamond = model mean. x: how often the model cites. y: when it cites, how accurate.",
      x = "Citation rate", y = "Citation correctness when cited") +
    theme(axis.title = element_text(size=13),
          strip.text = element_text(size=13))
  
  ggsave("outputs/figures/ed_fig1_cite_quality_scatter.png", p, width = 12, height = 4.5, dpi = 300, bg = "white")
})

# ============================================================================
# Extended Data Figure 2: subject area x model heatmap
# Subject area x model heatmap of mean per-summary scores for four judge metrics.
# (Formerly fig3_synopsis_heatmap.png.)
# ============================================================================
local({
  scale_fill_viridis <- function(...) scale_fill_gradientn(colours = viridisLite::viridis(256), ...)
  theme_set(theme_minimal(base_size = 11) +
              theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
                    plot.title = element_text(face = "bold", size = 12), plot.subtitle = element_text(colour = "grey30", size = 10),
                    strip.text = element_text(face = "bold"), legend.position = "top"))
  df_summ <- read_csv(file.path(D, "summary_level.csv"), show_col_types = FALSE) |>
    mutate(model = factor(unname(MODEL_LABEL[summary_model]), levels = MODEL_ORDER))
  synopsis_x_model <- read_csv(file.path(D, "synopsis_x_model.csv"), show_col_types = FALSE) |>
    mutate(model = factor(model, levels = MODEL_ORDER))
  heat_metrics <- c("Relevance", "CC (when cited)", "Faithfulness", "Citation rate")
  p <- synopsis_x_model |>
    filter(metric %in% heat_metrics) |>
    mutate(metric = factor(metric, levels = heat_metrics),
           synopsis = str_replace_all(synopsis, "([a-z])([A-Z])", "\\1 \\2"),
           # Manual fixes for missing boundaries (e.g. 'Controlof' -> 'Control of')
           synopsis = str_replace_all(synopsis, "Controlof",   "Control of"),
           synopsis = str_replace_all(synopsis, "Biodiversityof", "Biodiversity of"),
           synopsis = str_replace_all(synopsis, "Managementof",   "Management of"),
           synopsis = str_replace_all(synopsis, "Shrublandand",   "Shrubland and"),
           synopsis = str_replace_all(synopsis, "Butterflyand",   "Butterfly and"),
           synopsis = str_replace_all(synopsis, "Marineand",      "Marine and"),
           synopsis = fct_reorder(synopsis, mean_score, .fun = mean)) |>
    ggplot(aes(x = model, y = synopsis, fill = mean_score)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.0f", mean_score*100)), size = 2.8, colour = "white") +
    facet_wrap(~ metric, nrow = 1) +
    scale_fill_viridis_c(labels = percent_format(accuracy = 1),begin=0, end = 0.92,
                         name = "Mean score", option="C",direction = -1) +
    labs(#title = "Per-synopsis performance by model",
      #subtitle = "Cells: mean of per-summary scores. Synopses ordered by overall mean.",
      x = "Model", y = "Subject area (synopsis)") +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          strip.text = element_text(size=12),
          legend.position = "right")
  
  ggsave("outputs/figures/ed_fig2_synopsis_heatmap.png", p, width = 14, height = 6.5, dpi = 300, bg = "white")
})
