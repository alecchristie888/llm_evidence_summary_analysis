"""
Cross-version comparison script.

Loads multiple llm_vs_human_comparison CSV files and produces a summary
table showing how agreement, Kappa, FP and FN counts change across prompt
versions for each judge model and metric.

Usage
-----
    python compare_versions.py

Configure RESULT_FILES below — add any CSV files you want to compare.
The key is a short label (e.g. "gemini-v7"); the value is the file path.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_llm_vs_human import _cohen_kappa, _kappa_label


# ===========================================================================
# CONFIGURATION — add/remove entries as needed
# ===========================================================================

RESULT_FILES = {
    # --- Gemini ---
    "gemini-v7":          "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v7.csv",
    "gemini-v7.1":        "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v7.1.csv",
    #"gemini-v8":          "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v8.csv",
    #"gemini-v9":          "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v9.csv",
    #"gemini-v9-concat":   "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v9-concat.csv",
    "gemini-v10-concat":  "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v10-concat.csv",
    #"gemini-v11-concat":  "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v11-concat.csv",
    "gemini-v12-concat":  "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v12-concat.csv",
    #"gemini-v13-concat":  "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v13-concat.csv",
    "gemini-v14-concat":  "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v14-concat.csv",
    "gemini-v14-conf":    "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v14-conf.csv",
    #"gemini-v14-conf-temp1": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v14-conf-temp1.csv",
    #"gemini-v14-conf-multivote": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v14-conf-multivote.csv",
    #"gemini-v15-concat":   "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v15-concat.csv",
    "gemini-v16-relevance": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v16-relevance.csv",
    "gemini-v17-cc":      "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v17-cc.csv",
    #"gemini-v18-ff":      "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v18-ff.csv",
    #"gemini-v19-cc":      "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v19-cc.csv",
    #"gemini-v20-rel":    "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v20-rel.csv",
    #"gemini-v21-rel":    "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-v21-rel.csv",
    "gemini-gepa1234": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-gepa1234.csv",
    "gemini-gepa5678": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-gepa5678.csv",
    "gemini-seed-v7": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-seed-v7.csv",

    # --- Claude Sonnet ---
    #"sonnet-v2":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v2.csv",
    #"sonnet-v3":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v3.csv",
    #"sonnet-v4":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v4.csv",
    #"sonnet-v5":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v5.csv",
    #"sonnet-v6":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v6.csv",
    "sonnet-v7":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v7.csv",
    "sonnet-v7.1":        "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v7.1.csv",
    #"sonnet-v8":          "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v8.csv",
    #"sonnet-v9-concat":   "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v9-concat.csv",
    "sonnet-v10-concat":  "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v10-concat.csv",
    #"sonnet-v11-concat":  "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v11-concat.csv",
    "sonnet-v12-concat":  "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v12-concat.csv",
    #"sonnet-v13-concat":  "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v13-concat.csv",
    "sonnet-v14-concat":  "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v14-concat.csv",
    #"sonnet-v14-conf-multivote": "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v14-conf-multivote.csv",
    "sonnet-v17-cc":      "llm_judge_results/llm_vs_human_comparison_claude-sonnet-4-6-v17-cc.csv",

    # --- Claude Opus ---
    #"opus-v8":            "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v8.csv",
    #"opus-v9":            "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v9.csv",
    #"opus-v9-concat":     "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v9-concat.csv",
    "opus-v10-concat":    "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v10-concat.csv",
    "opus-v12-concat":    "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v12-concat.csv",
    #"opus-v13-concat":    "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v13-concat.csv",
    "opus-v14-concat":    "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v14-concat.csv",
    "opus-v14-conf":      "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v14-conf.csv",
    "opus-v17-cc":        "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v17-cc.csv",
    #"opus-v19-cc":        "llm_judge_results/llm_vs_human_comparison_claude-opus-4-7-v19-cc.csv",
    "opus-gepa1234": "llm_judge_results/llm_vs_human_comparison_claude-opus-4.7-gepa1234.csv",
    "opus-gepa5678": "llm_judge_results/llm_vs_human_comparison_claude-opus-4.7-gepa5678.csv",
    "opus-seed-v7": "llm_judge_results/llm_vs_human_comparison_claude-opus-4.7-seed-v7.csv",

    # --- GPT-5.4 ---
    #"gpt5.4-v2":          "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v2.csv",
    "gpt5.4-v7":          "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v7.csv",
    "gpt5.4-v7.1":        "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v7.1.csv",
    #"gpt5.4-v8":          "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v8.csv",
    #"gpt5.4-v9":          "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v9.csv",
    #"gpt5.4-v9-concat":   "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v9-concat.csv",
    "gpt5.4-v10-concat":  "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v10-concat.csv",
    #"gpt5.4-v11-concat":  "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v11-concat.csv",
    "gpt5.4-v12-concat":  "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v12-concat.csv",
    #"gpt5.4-v13-concat":  "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v13-concat.csv",
    "gpt5.4-v14-concat":  "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v14-concat.csv",
    "gpt5.4-v17-cc":      "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v17-cc.csv",
    #"gpt5.4-v19-cc":      "llm_judge_results/llm_vs_human_comparison_gpt-5-4-v19-cc.csv",
    "gpt5.4-gepa1234": "llm_judge_results/llm_vs_human_comparison_gpt-5.4-gepa1234.csv",
    "gpt5.4-gepa5678": "llm_judge_results/llm_vs_human_comparison_gpt-5.4-gepa5678.csv",
    "gpt5.4-seed-v7": "llm_judge_results/llm_vs_human_comparison_gpt-5.4-seed-v7.csv",

}

# Which metrics to show. Subset of:
#   ["relevance", "citation_correctness", "citation_correctness_cited", "faithfulness"]
# "citation_correctness_cited" shows CC with auto-Nos excluded.
METRICS = [
    "relevance",
    "citation_correctness",
    "citation_correctness_cited",
    "faithfulness",
]

# Group entries by judge model prefix for cleaner output.
# Each entry is (display_name, list_of_key_prefixes_in_RESULT_FILES)
JUDGE_GROUPS = [
    ("Gemini 3.1 Pro",    ["gemini"]),
    ("Claude Sonnet 4.6", ["sonnet"]),
    ("Claude Opus 4.7",   ["opus"]),
    ("GPT-5.4",           ["gpt5.4"]),
]

# ===========================================================================


def get_stats(df: pd.DataFrame, metric: str, cited_only: bool = False) -> dict | None:
    h = f"human_{metric}"
    l = f"llm_{metric}"
    if metric == "citation_correctness_cited":
        h, l = "human_citation_correctness", "llm_citation_correctness"
        cited_only = True

    if h not in df.columns or l not in df.columns:
        return None

    paired = df[df[h].notna() & df[l].notna()]
    if cited_only and "cc_auto_filled" in paired.columns:
        paired = paired[~paired["cc_auto_filled"]]

    n = len(paired)
    if n == 0:
        return None

    agree = (paired[h] == paired[l]).sum()
    fp    = ((paired[h] == False) & (paired[l] == True)).sum()
    fn    = ((paired[h] == True)  & (paired[l] == False)).sum()
    tp    = ((paired[h] == True)  & (paired[l] == True)).sum()
    tn    = ((paired[h] == False) & (paired[l] == False)).sum()
    kappa, _ = _cohen_kappa(paired[h].tolist(), paired[l].tolist())
    prec  = tp / (tp + fp) if (tp + fp) > 0 else None
    rec   = tp / (tp + fn) if (tp + fn) > 0 else None
    spec  = tn / (tn + fp) if (tn + fp) > 0 else None
    f1    = 2 * prec * rec / (prec + rec) if prec and rec else None

    return dict(
        n=n, agree=int(agree), pct=agree / n,
        fp=int(fp), fn=int(fn), tp=int(tp), tn=int(tn),
        kappa=kappa, prec=prec, rec=rec, spec=spec, f1=f1,
    )


def fmt_kappa(k):
    return f"{k:.3f}" if k is not None else "N/A"


def fmt_pct(v):
    return f"{v:.1%}" if v is not None else "N/A"


def print_metric_table(
    dfs: dict,
    metric: str,
    judge_groups: list,
    show_detail: bool = True,
) -> None:
    """Print a table for one metric across all versions, grouped by judge."""
    is_cited = metric == "citation_correctness_cited"
    display_metric = "citation_correctness (cited only)" if is_cited else metric

    print(f"\n{'='*95}")
    print(f"METRIC: {display_metric.upper()}")
    print(f"{'='*95}")
    print(f"  {'Version':<22}  {'N':>5}  {'Agree':>8}  {'Kappa':>8}  {'Interpret':<18}  "
          f"{'FP':>4}  {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'Spec':>6}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*18}  {'-'*4}  {'-'*4}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}")

    for group_name, prefixes in judge_groups:
        # Find keys belonging to this group
        group_keys = [
            k for k in dfs
            if any(k.startswith(p) for p in prefixes)
        ]
        if not group_keys:
            continue

        printed_header = False
        for key in group_keys:
            df = dfs[key]
            s = get_stats(df, metric)
            if s is None:
                continue

            if not printed_header:
                print(f"\n  --- {group_name} ---")
                printed_header = True

            k_str    = fmt_kappa(s["kappa"])
            interp   = _kappa_label(s["kappa"]) if s["kappa"] is not None else ""
            prec_str = f"{s['prec']:.3f}" if s["prec"] is not None else "N/A"
            rec_str  = f"{s['rec']:.3f}"  if s["rec"]  is not None else "N/A"
            spec_str = f"{s['spec']:.3f}" if s["spec"] is not None else "N/A"

            print(f"  {key:<22}  {s['n']:>5}  {s['pct']:>7.1%}  {k_str:>8}  "
                  f"{interp:<18}  {s['fp']:>4}  {s['fn']:>4}  "
                  f"{prec_str:>6}  {rec_str:>6}  {spec_str:>6}")


def print_summary_table(dfs: dict, judge_groups: list) -> None:
    """Print a compact all-metrics-all-versions summary."""
    metrics_short = [
        ("relevance",                  "Relev"),
        ("citation_correctness",       "CC(all)"),
        ("citation_correctness_cited", "CC(cited)"),
        ("faithfulness",               "Faith"),
    ]

    print(f"\n{'='*95}")
    print("SUMMARY — ALL METRICS × ALL VERSIONS")
    print(f"{'='*95}")
    hdr = f"  {'Version':<22}  {'N':>5}"
    for _, short in metrics_short:
        hdr += f"  {short+' agree':>12}  {short+' κ':>9}"
    print(hdr)
    sep = f"  {'-'*22}  {'-'*5}"
    for _ in metrics_short:
        sep += f"  {'-'*12}  {'-'*9}"
    print(sep)

    for group_name, prefixes in judge_groups:
        group_keys = [k for k in dfs if any(k.startswith(p) for p in prefixes)]
        if not group_keys:
            continue
        print(f"\n  --- {group_name} ---")
        for key in group_keys:
            df = dfs[key]
            row = f"  {key:<22}  "
            # Use n from faithfulness as reference
            ref = get_stats(df, "faithfulness")
            row += f"{ref['n']:>5}" if ref else f"{'?':>5}"
            for metric, _ in metrics_short:
                s = get_stats(df, metric)
                if s:
                    row += f"  {s['pct']:>10.1%}  {fmt_kappa(s['kappa']):>9}"
                else:
                    row += f"  {'N/A':>12}  {'N/A':>9}"
            print(row)


def print_faithfulness_detail(dfs: dict, judge_groups: list) -> None:
    """Print FP/FN/Spec detail for faithfulness across all versions."""
    print(f"\n{'='*95}")
    print("FAITHFULNESS DETAIL — FP / FN / SPECIFICITY")
    print(f"{'='*95}")
    print(f"  {'Version':<22}  {'N':>5}  {'Agree':>8}  {'κ':>7}  "
          f"{'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'Spec':>6}  {'F1':>6}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for group_name, prefixes in judge_groups:
        group_keys = [k for k in dfs if any(k.startswith(p) for p in prefixes)]
        if not group_keys:
            continue
        print(f"\n  --- {group_name} ---")
        best_kappa = -99
        best_key = None
        for key in group_keys:
            s = get_stats(dfs[key], "faithfulness")
            if not s:
                continue
            if s["kappa"] and s["kappa"] > best_kappa:
                best_kappa = s["kappa"]
                best_key = key
            marker = ""
            print(f"  {key:<22}  {s['n']:>5}  {s['pct']:>7.1%}  "
                  f"{fmt_kappa(s['kappa']):>7}  "
                  f"{s['fp']:>5}  {s['fn']:>5}  "
                  f"{s['prec'] or 0:>6.3f}  {s['rec'] or 0:>6.3f}  "
                  f"{s['spec'] or 0:>6.3f}  {s['f1'] or 0:>6.3f}")
        if best_key:
            print(f"  ↑ Best κ: {best_key} ({best_kappa:.3f})")


def main():
    # Load all available files
    dfs = {}
    missing = []
    for key, path in RESULT_FILES.items():
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
        else:
            missing.append(f"{key}: {path}")

    print(f"Loaded {len(dfs)}/{len(RESULT_FILES)} result files.")
    if missing:
        print("Missing files (skipped):")
        for m in missing:
            print(f"  {m}")

    if not dfs:
        print("No files found — check paths in RESULT_FILES.")
        sys.exit(1)

    # Summary table
    print_summary_table(dfs, JUDGE_GROUPS)

    # Faithfulness detail
    print_faithfulness_detail(dfs, JUDGE_GROUPS)

    # Per-metric detailed tables
    for metric in METRICS:
        print_metric_table(dfs, metric, JUDGE_GROUPS)


if __name__ == "__main__":
    main()