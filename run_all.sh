#!/usr/bin/env bash
# Regenerate every derived table and figure from the archived data. No API calls, no cost.
# Run from the repository root with the data archive unpacked into data/.
set -euo pipefail
mkdir -p data/derived outputs/figures

echo "== Stage 3 (offline part): judge validation from archived judge-vs-human comparisons"
python 03_llm_judge/build_judge_validation_table.py          # Table 1 / ED judge table  -> data/derived/judge_validation_table.csv
python 03_llm_judge/inter_rater_reliability.py > data/derived/inter_rater_reliability_report.txt   # ED human IRR table

echo "== Stage 5a: statement- and summary-level score tables (merges the per-model judge output)"
python 05_analysis/build_score_tables.py

echo "== Stage 4: retrieval and agent behaviour (from raw generation logs)"
python 04_retrieval_traces/parse_tool_traces.py
python 04_retrieval_traces/recheck_traces.py
python 04_retrieval_traces/true_retrieval_recall.py
python 04_retrieval_traces/retrieval_summary_stats.py

echo "== Stage 5b: failure tags and bootstrap statistics (tables)"
python 05_analysis/build_failure_reasons.py --results-dir data/judge_outputs/by_model --output-dir data/derived
Rscript 05_analysis/analysis.R                 # headline means, distributions, variance decomposition, failure-reason profile
Rscript 05_analysis/reframed_metrics.R         # evaluated-run retrieval metrics, ranks, composite, tails, correlations

echo "== Stage 5c: all figures (main Figs 1-4, Extended Data Figs 1-2)"
Rscript 05_analysis/figures.R

echo "== Stage 7: file manifest (sha256 + record counts for every data file)"
python make_manifest.py

echo "== Verify against the paper"
python verify_results.py
