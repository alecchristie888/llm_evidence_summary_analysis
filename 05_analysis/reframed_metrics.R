# ============================================================================
# Analyses added for the reframed manuscript ("Faithful but incomplete").
# Everything here is derived from files already in data/ — no API calls.
#
# Inputs
#   data/derived/summary_level.csv          judge scores, one row per evaluated summary
#   data/derived/retrieval_recall.csv       log-based retrieval metrics per (model, question)
#                                           (true_retrieval_recall.py; union of actions opened across all
#                                           completed runs of a question, incl. discarded duplicate runs).
#                                           Kept for comparison only.
#
# PRIMARY DEFINITION (reported in the paper): the evaluated-run definition. Retrieval recall,
# precision, actions opened and cite-through are computed from the tool-call trace of the single
# run that produced each judged summary, so every retrieval metric shares the denominator of the
# judge metrics (n = 515 / 520 / 467 / 489 summaries).
#   data/derived/attrition_by_model.csv     completion counts (recheck_traces.py)
#   data/summaries/<model>/orig_summaries/bg_km_<area>_summaries.json
#                                           the run that produced each evaluated summary, with its own
#                                           tool-call trace and the withheld reference pool
#
# Outputs (data/derived/)
#   retrieval_per_summary.csv            per-summary retrieval metrics under BOTH definitions, joined to
#                                        the judge scores (source for Fig. 2b and the correlations)
#   retrieval_metrics_by_model_evaluated_run.csv  model-level means (percent) with bootstrap CIs under the
#                                        evaluated-run definition: the retrieval numbers reported in the paper
#   retrieval_definition_comparison.csv  model-level means under the evaluated-run (paper) and the
#                                        log-based definition, side by side, with bootstrap CIs
#   metric_ranks.csv                     rank of each model on every metric (Fig. 2a)
#   composite_by_model.csv               mean of per-summary recall x faithfulness with bootstrap 95% CI
#   within_model_correlations.csv        Pearson r between retrieval effort and faithfulness within model
#   tail_probabilities.csv               P(score < 50), P(< 70), P(>= 90) per model x metric
# Figures are built from these tables by 05_analysis/figures.R.
#
# Run from the repository root:  Rscript 05_analysis/reframed_metrics.R
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(purrr); library(stringr)
  library(tibble); library(jsonlite)
})

# Run from the repository root. If started from 05_analysis/ (or interactively from there), step up one level.
if (!dir.exists("data/derived") && dir.exists("../data/derived")) setwd("..")
if (!dir.exists("data/derived")) stop("Run from the repository root (the folder containing run_all.sh and data/); current directory: ", getwd())
read_derived <- function(name) {
  path <- file.path(Sys.getenv("IGF_DERIVED", "data/derived"), name)
  if (!file.exists(path) || file.size(path) < 10) stop(path, " is missing or empty: run `python 05_analysis/build_score_tables.py` (and the earlier stages in run_all.sh) first.")
  readr::read_csv(path, show_col_types = FALSE)
}

set.seed(1)
N_BOOT <- 1000
D      <- Sys.getenv("IGF_DERIVED", "data/derived")

LABEL <- c("anthropic/claude-sonnet-4" = "Claude Sonnet 4",
           "google/gemini-2.5-pro"     = "Gemini 2.5 Pro",
           "openai/gpt-5"              = "GPT-5",
           "moonshotai/kimi-k2-0905"   = "Kimi K2")
SUMMARY_DIRS <- c("_claude-sonnet-4"       = "anthropic/claude-sonnet-4",
                  "_gemini-2-5-pro"        = "google/gemini-2.5-pro",
                  "_gpt-5"                 = "openai/gpt-5",
                  "fireworks_kimi-k2-0905" = "moonshotai/kimi-k2-0905")

boot_mean_ci <- function(x, n = N_BOOT) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(mean = NA, lo = NA, hi = NA))
  idx   <- matrix(sample.int(length(x), n * length(x), replace = TRUE), nrow = n)
  means <- rowMeans(matrix(x[idx], nrow = n))
  c(mean = mean(x), lo = unname(quantile(means, 0.025)), hi = unname(quantile(means, 0.975)))
}

# ----------------------------------------------------------------------------
# 1. Per-summary retrieval metrics from the run that produced each summary
# ----------------------------------------------------------------------------
as_chr_ids <- function(v) if (is.null(v)) character(0) else as.character(unlist(v))

parse_entry <- function(e, model, area) {
  if (is.null(e$relevant_summary) || !nzchar(e$relevant_summary)) return(NULL)  # not evaluated
  pool   <- as_chr_ids(e$all_relevant_action_ids)
  opened <- character(0); n_search <- 0L; max_off <- 0L; n_err <- 0L; n_iter <- 0L
  if (!is.null(e$tool_call_details)) {
    n_iter <- length(e$tool_call_details)
    for (it in e$tool_call_details) {
      for (t in it$tools_called) {
        rv <- t$return_val
        if (is.character(rv) && grepl('^\\s*(error|\\{"error)', tolower(rv))) n_err <- n_err + 1L
        args <- tryCatch(fromJSON(t$args, simplifyVector = TRUE), error = function(err) list())
        if (identical(t$function_name, "search_actions")) {
          n_search <- n_search + 1L
          off <- suppressWarnings(as.integer(args$offset)); if (length(off) && !is.na(off)) max_off <- max(max_off, off)
        } else if (identical(t$function_name, "get_action_details") && !is.null(args$action_id)) {
          opened <- union(opened, as.character(args$action_id))
        }
      }
    }
  }
  cited   <- as_chr_ids(e$summary_action_ids)
  hit     <- intersect(opened, pool)
  tibble(summary_model = model, synopsis = area, question = str_trim(e$query),
         n_pool = length(pool), n_opened_run = length(opened), n_searches_run = n_search,
         max_offset_run = max_off, n_iterations_run = n_iter, n_tool_errors_run = n_err,
         recall_run    = if (length(pool)) length(hit) / length(pool) else NA_real_,
         precision_run = if (length(opened)) length(hit) / length(opened) else NA_real_,
         cite_through_run = if (length(hit)) length(intersect(cited, hit)) / length(hit) else NA_real_,
         cite_recall_run  = if (length(pool)) length(intersect(cited, pool)) / length(pool) else NA_real_,
         n_cited_actions = length(cited))
}

run <- imap_dfr(SUMMARY_DIRS, function(model, d) {
  files <- list.files(file.path("data/summaries", d, "orig_summaries"),
                      pattern = "^bg_km_.*_summaries\\.json$", full.names = TRUE)
  map_dfr(files, function(f) {
    area <- str_match(basename(f), "^bg_km_(.*)_summaries\\.json$")[, 2]
    entries <- fromJSON(f, simplifyVector = FALSE)
    map_dfr(entries, parse_entry, model = model, area = area)
  })
})

# log-based definition (as reported in the paper)
logb <- read_csv(file.path(D, "retrieval_recall.csv"), show_col_types = FALSE) |>
  transmute(summary_model = model, question = str_trim(question),
            recall_log = ret_recall, precision_log = ret_prec, n_opened_log = n_opened,
            cite_through_log = conv, cite_recall_log = cite_recall)

summ <- read_derived("summary_level.csv") |>
  mutate(question = str_trim(question))

per <- summ |>
  left_join(select(run, -synopsis), by = c("summary_model", "question")) |>
  left_join(logb, by = c("summary_model", "question")) |>
  mutate(model = unname(LABEL[summary_model]))
write_csv(per, file.path(D, "retrieval_per_summary.csv"))

cat(sprintf("summaries: %d  matched to evaluated run: %d  matched to log-based recall: %d\n",
            nrow(summ), sum(!is.na(per$recall_run)), sum(!is.na(per$recall_log))))

# ----------------------------------------------------------------------------
# 2. Definition comparison (model level)
# ----------------------------------------------------------------------------
cmp_cols <- c(recall_log_based = "recall_log", recall_evaluated_run = "recall_run",
              precision_log_based = "precision_log", precision_evaluated_run = "precision_run",
              opened_log_based = "n_opened_log", opened_evaluated_run = "n_opened_run",
              cite_through_log_based = "cite_through_log", cite_through_evaluated_run = "cite_through_run",
              searches_evaluated_run = "n_searches_run", max_offset_evaluated_run = "max_offset_run",
              cite_recall_evaluated_run = "cite_recall_run")
cmp <- per |>
  group_by(model) |>
  group_modify(function(g, k) {
    out <- tibble(n_summaries = nrow(g), n_with_log_based = sum(!is.na(g$recall_log)))
    for (nm in names(cmp_cols)) {
      ci <- boot_mean_ci(g[[cmp_cols[[nm]]]])
      out[[nm]] <- round(ci[["mean"]], 4); out[[paste0(nm, "_lo")]] <- round(ci[["lo"]], 4); out[[paste0(nm, "_hi")]] <- round(ci[["hi"]], 4)
    }
    out
  }) |> ungroup()
write_csv(cmp, file.path(D, "retrieval_definition_comparison.csv"))
cat("\n=== Retrieval recall: evaluated-run (paper) vs log-based definition ===\n")
print(select(cmp, model, n_summaries, n_with_log_based, recall_evaluated_run, recall_log_based,
             opened_evaluated_run, opened_log_based), width = 200)

# the retrieval table reported in the paper (percent; counts for opened/searches/offset)
paper_cols <- c(ret_recall = "recall_run", cite_recall = "cite_recall_run", ret_prec = "precision_run",
                cite_through = "cite_through_run", n_opened = "n_opened_run", n_searches = "n_searches_run",
                max_offset = "max_offset_run")
pct_cols <- c("ret_recall", "cite_recall", "ret_prec", "cite_through")
retr_paper <- per |>
  group_by(model) |>
  group_modify(function(g, k) {
    out <- tibble(n = nrow(g))
    for (nm in names(paper_cols)) {
      ci <- boot_mean_ci(g[[paper_cols[[nm]]]]); f <- if (nm %in% pct_cols) 100 else 1
      out[[nm]] <- round(f * ci[["mean"]], 3); out[[paste0(nm, "_lo")]] <- round(f * ci[["lo"]], 3); out[[paste0(nm, "_hi")]] <- round(f * ci[["hi"]], 3)
    }
    out
  }) |> ungroup()
write_csv(retr_paper, file.path(D, "retrieval_metrics_by_model_evaluated_run.csv"))
cat("\n=== Retrieval metrics, evaluated-run definition (paper) ===\n")
print(select(retr_paper, model, n, ret_recall, ret_prec, cite_recall, cite_through, n_opened, n_searches, max_offset), width = 200)

# ----------------------------------------------------------------------------
# 3. Composite: mean of per-summary retrieval recall x faithfulness (bootstrap CI)
# ----------------------------------------------------------------------------
comp <- per |>
  group_by(model) |>
  group_modify(function(g, k) {
    map_dfr(c(evaluated_run_paper = "recall_run", log_based = "recall_log"), function(col) {
      prod <- g[[col]] * g$faithfulness_score
      ci <- boot_mean_ci(prod)
      tibble(n = sum(!is.na(prod)), composite_mean = round(ci[["mean"]], 4),
             lo = round(ci[["lo"]], 4), hi = round(ci[["hi"]], 4),
             model_level_product_illustrative = round(mean(g[[col]], na.rm = TRUE) * mean(g$faithfulness_score, na.rm = TRUE), 4))
    }, .id = "recall_definition")
  }) |> ungroup()
write_csv(comp, file.path(D, "composite_by_model.csv"))

# ----------------------------------------------------------------------------
# 4. Ranks on every metric (Fig. 2a)
# ----------------------------------------------------------------------------
att <- read_csv(file.path(D, "attrition_by_model.csv"), show_col_types = FALSE) |>
  mutate(model = unname(LABEL[model]), `Task completion` = 1 - never_completed / 520) |>
  select(model, `Task completion`)

means <- per |>
  group_by(model) |>
  summarise(Relevance = mean(relevance_score, na.rm = TRUE),
            `Citation rate` = mean(citation_rate, na.rm = TRUE),
            `CC (all statements)` = mean(cc_all_score, na.rm = TRUE),
            `CC (when cited)` = mean(cc_cited_score, na.rm = TRUE),
            Faithfulness = mean(faithfulness_score, na.rm = TRUE),
            `Retrieval recall` = mean(recall_run, na.rm = TRUE),
            `Citation recall` = mean(cite_recall_run, na.rm = TRUE),
            `Retrieval precision` = mean(precision_run, na.rm = TRUE),
            `Cite-through rate` = mean(cite_through_run, na.rm = TRUE), .groups = "drop") |>
  left_join(att, by = "model") |>
  left_join(comp |> filter(recall_definition == "evaluated_run_paper") |>
              transmute(model, `Composite (retrieval recall x faithfulness)` = composite_mean), by = "model")

rank_long <- means |>
  pivot_longer(-model, names_to = "metric", values_to = "mean") |>
  group_by(metric) |>
  mutate(rank = min_rank(desc(mean))) |>
  ungroup()
metric_ranks <- rank_long |>
  pivot_wider(id_cols = model, names_from = metric, values_from = c(mean, rank), names_glue = "{metric} ({.value})") |>
  mutate(across(where(is.numeric) & ends_with("(mean)"), ~ round(.x, 4)))
write_csv(metric_ranks, file.path(D, "metric_ranks.csv"))
cat("\n=== Ranks (1 = best) ===\n")
print(rank_long |> select(-mean) |> pivot_wider(names_from = model, values_from = rank), n = 20)
cat("\n=== Composite retrieval recall x faithfulness (summary-level mean, 95% CI) ===\n")
print(comp, width = 200)

# ----------------------------------------------------------------------------
# 5. Within-model correlations: retrieval effort vs faithfulness
# ----------------------------------------------------------------------------
effort_vars <- c("n_searches_run", "n_opened_run", "n_iterations_run", "recall_run", "recall_log", "n_statements")
cors <- per |>
  select(model, faithfulness_score, all_of(effort_vars)) |>
  pivot_longer(all_of(effort_vars), names_to = "x", values_to = "xval") |>
  group_by(model, x) |>
  summarise(y = "faithfulness_score",
            pearson_r = if (sum(complete.cases(xval, faithfulness_score)) > 2 && sd(xval, na.rm = TRUE) > 0)
              round(cor(xval, faithfulness_score, use = "complete.obs"), 3) else NA_real_,
            n = sum(complete.cases(xval, faithfulness_score)), .groups = "drop")
write_csv(cors, file.path(D, "within_model_correlations.csv"))
cat("\n=== Within-model Pearson r with faithfulness ===\n")
print(cors |> select(model, x, pearson_r) |> pivot_wider(names_from = model, values_from = pearson_r))

# ----------------------------------------------------------------------------
# 6. Tail probabilities (operational form of 'report distributions')
# ----------------------------------------------------------------------------
tail_vars <- c(Relevance = "relevance_score", `Citation rate` = "citation_rate",
               `CC (all statements)` = "cc_all_score", `CC (when cited)` = "cc_cited_score",
               Faithfulness = "faithfulness_score", `Retrieval recall` = "recall_run")
tails <- per |>
  select(model, all_of(unname(tail_vars))) |>
  pivot_longer(-model, names_to = "col", values_to = "x") |>
  filter(!is.na(x)) |>
  mutate(metric = names(tail_vars)[match(col, tail_vars)]) |>
  group_by(model, metric) |>
  summarise(n = n(), p_lt_50 = round(mean(x < 0.5), 4), p_lt_70 = round(mean(x < 0.7), 4),
            p_ge_90 = round(mean(x >= 0.9), 4), .groups = "drop")
write_csv(tails, file.path(D, "tail_probabilities.csv"))
