"""
Judge model bias and performance analysis.

For each judge model, computes per-summary-model agreement, Kappa,
FP/FN rates, precision, recall, specificity, and a bias score showing
how much the judge deviates from human Yes rates for each candidate model.

Usage
-----
    python analyse_judge_bias.py

Add result CSV files to RESULT_FILES below, grouped by judge model.
Each entry maps a short label to a CSV path from compare_llm_vs_human.py.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_llm_vs_human import _cohen_kappa, _kappa_label


# ===========================================================================
# CONFIGURATION — add your result files here
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
    "gemini-final-prompt": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-finalprompts.csv",

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
    "opus-v10-concat":    "llm_judge_results/llm_vs_human_comparison_claude-opus-4.7-v10-concat.csv",
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

# Summary models to show (short display names)
SUMMARY_MODEL_LABELS = {
    "anthropic/claude-sonnet-4":  "claude-sonnet-4",
    "google/gemini-2.5-pro":      "gemini-2.5-pro",
    "moonshotai/kimi-k2-0905":    "kimi-k2",
    "openai/gpt-5":               "gpt-5",
}

METRICS = ["relevance", "citation_correctness", "faithfulness"]

# ===========================================================================


def get_stats(df: pd.DataFrame, metric: str, cited_only: bool = False) -> dict | None:
    h = f"human_{metric}"
    l = f"llm_{metric}"
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
    h_yes = paired[h].mean()
    l_yes = paired[l].mean()
    bias  = l_yes - h_yes  # positive = LLM too lenient, negative = too strict
    return dict(
        n=n, agree=int(agree), pct=agree/n,
        fp=int(fp), fn=int(fn), tp=int(tp), tn=int(tn),
        kappa=kappa, prec=prec, rec=rec, spec=spec, f1=f1,
        h_yes=h_yes, l_yes=l_yes, bias=bias,
    )


def fmt(v, fmt_str=".3f"):
    return f"{v:{fmt_str}}" if v is not None else "N/A"


def print_overall_summary(dfs: dict) -> None:
    """Overall agreement and Kappa per judge across all metrics."""
    print("\n" + "=" * 90)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 90)

    metrics_display = [
        ("relevance",              "Relevance"),
        ("citation_correctness",   "CC (all)"),
        ("citation_correctness",   "CC (cited)"),
        ("faithfulness",           "Faithfulness"),
    ]

    hdr = f"  {'Judge':<22}"
    for _, label in metrics_display:
        hdr += f"  {label+' agree':>13}  {label+' κ':>9}"
    print(hdr)
    print("  " + "-" * 22 + "  " + "  ".join(["-"*13 + "  " + "-"*9] * len(metrics_display)))

    for judge, df in dfs.items():
        row = f"  {judge:<22}"
        for metric, _ in metrics_display:
            cited = (metric == "citation_correctness" and _ == "CC (cited)")
            s = get_stats(df, metric, cited_only=cited)
            if s:
                k_str = fmt(s["kappa"])
                row += f"  {s['pct']:>11.1%}  {k_str:>9}"
            else:
                row += f"  {'N/A':>13}  {'N/A':>9}"
        print(row)


def print_metric_detail(dfs: dict, metric: str, cited_only: bool = False) -> None:
    """Detailed FP/FN/Prec/Rec/Spec per judge for one metric."""
    label = metric + (" (cited only)" if cited_only else "")
    print(f"\n{'='*90}")
    print(f"METRIC DETAIL: {label.upper()}")
    print(f"{'='*90}")
    print(f"  {'Judge':<22}  {'N':>5}  {'Agree':>8}  {'κ':>7}  "
          f"{'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'Spec':>6}  {'F1':>6}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for judge, df in dfs.items():
        s = get_stats(df, metric, cited_only=cited_only)
        if not s:
            print(f"  {judge:<22}  N/A")
            continue
        k_str = f"{s['kappa']:.3f}" if s['kappa'] is not None else "N/A"
        print(f"  {judge:<22}  {s['n']:>5}  {s['pct']:>7.1%}  {k_str:>7}  "
              f"{s['fp']:>5}  {s['fn']:>5}  "
              f"{fmt(s['prec']):>6}  {fmt(s['rec']):>6}  "
              f"{fmt(s['spec']):>6}  {fmt(s['f1']):>6}")


def print_bias_table(dfs: dict, metric: str, cited_only: bool = False) -> None:
    """
    Bias table: for each (judge, summary_model) pair, shows:
      - Human Yes rate
      - LLM Yes rate
      - Bias (diff)
      - Kappa
      - FP / FN counts
    """
    label = metric + (" (cited only)" if cited_only else "")
    print(f"\n{'='*90}")
    print(f"BIAS ANALYSIS: {label.upper()}")
    print(f"  Bias = LLM Yes rate - Human Yes rate")
    print(f"  Positive bias = judge too lenient on that model's summaries")
    print(f"  Negative bias = judge too strict on that model's summaries")
    print(f"{'='*90}")

    summary_models = list(SUMMARY_MODEL_LABELS.keys())
    col_w = 16

    # Header
    hdr = f"  {'Judge':<22}"
    for sm in summary_models:
        label_short = SUMMARY_MODEL_LABELS[sm]
        hdr += f"  {label_short:^{col_w}}"
    print(hdr)

    subhdr = f"  {'':22}"
    for _ in summary_models:
        subhdr += f"  {'agree% / bias / κ':^{col_w}}"
    print(subhdr)
    print("  " + "-"*22 + ("  " + "-"*col_w) * len(summary_models))

    for judge, df in dfs.items():
        row = f"  {judge:<22}"
        for sm in summary_models:
            mdf = df[df["summary_model"] == sm]
            s = get_stats(mdf, metric, cited_only=cited_only)
            if not s or s["n"] < 5:
                row += f"  {'N/A':^{col_w}}"
                continue
            bias_str = f"{s['bias']:+.0%}"
            k_str    = f"{s['kappa']:.2f}" if s['kappa'] is not None else "N/A"
            cell = f"{s['pct']:.0%}/{bias_str}/{k_str}"
            row += f"  {cell:^{col_w}}"
        print(row)

    # Bias magnitude summary — which judge is most neutral overall
    print(f"\n  Absolute bias (avg across summary models, lower = more neutral):")
    for judge, df in dfs.items():
        biases = []
        for sm in summary_models:
            mdf = df[df["summary_model"] == sm]
            s = get_stats(mdf, metric, cited_only=cited_only)
            if s and s["n"] >= 5:
                biases.append(abs(s["bias"]))
        if biases:
            avg_abs_bias = sum(biases) / len(biases)
            print(f"    {judge:<22}  avg |bias| = {avg_abs_bias:.1%}")


def print_self_favour(dfs: dict) -> None:
    """
    Specifically highlight whether the judge favours its own model's summaries
    vs other models.
    """
    # Map judge name fragments to summary model names
    judge_model_map = {
        "gemini":  "google/gemini-2.5-pro",
        "sonnet":  "anthropic/claude-sonnet-4",
        "opus":    "anthropic/claude-sonnet-4",   # opus judges claude summaries
        "gpt":     "openai/gpt-5",
        "deepseek": None,
    }

    print(f"\n{'='*90}")
    print("SELF-FAVOUR CHECK")
    print("  Is the judge biased toward its own model family's summaries?")
    print(f"{'='*90}")

    for metric in METRICS:
        cited_only = False
        print(f"\n  {metric}:")
        for judge, df in dfs.items():
            # Identify which summary model is the judge's "own"
            own_model = None
            for fragment, sm in judge_model_map.items():
                if fragment in judge.lower():
                    own_model = sm
                    break

            own_bias   = None
            other_biases = []

            for sm in SUMMARY_MODEL_LABELS:
                mdf = df[df["summary_model"] == sm]
                s = get_stats(mdf, metric, cited_only=cited_only)
                if not s or s["n"] < 5:
                    continue
                if sm == own_model:
                    own_bias = s["bias"]
                else:
                    other_biases.append(s["bias"])

            avg_other = sum(other_biases) / len(other_biases) if other_biases else None

            if own_bias is not None and avg_other is not None:
                self_favour = own_bias - avg_other
                direction = "MORE LENIENT on own model" if self_favour > 0.05 else (
                    "MORE STRICT on own model" if self_favour < -0.05 else "NEUTRAL"
                )
                print(f"    {judge:<22}  own bias: {own_bias:+.1%}  "
                      f"avg other bias: {avg_other:+.1%}  "
                      f"self-favour diff: {self_favour:+.1%}  → {direction}")
            else:
                print(f"    {judge:<22}  (own model not identified or insufficient data)")


def print_judge_recommendation(dfs: dict) -> None:
    """
    Summary recommendation table: score each judge on a composite
    of Kappa, bias neutrality, and FP/FN balance.
    """
    print(f"\n{'='*90}")
    print("JUDGE RECOMMENDATION SUMMARY")
    print("  Composite score = avg Kappa across metrics × (1 - avg absolute bias)")
    print("  Higher is better. Tiebreak by lowest FP rate on faithfulness.")
    print(f"{'='*90}")

    scores = {}
    for judge, df in dfs.items():
        kappas = []
        biases = []

        for metric in METRICS:
            cited = False
            s_all = get_stats(df, metric, cited_only=cited)
            if s_all and s_all["kappa"] is not None:
                kappas.append(s_all["kappa"])
            # Per-model bias
            for sm in SUMMARY_MODEL_LABELS:
                mdf = df[df["summary_model"] == sm]
                s = get_stats(mdf, metric, cited_only=cited)
                if s and s["n"] >= 5:
                    biases.append(abs(s["bias"]))

        if not kappas:
            continue

        avg_kappa    = sum(kappas) / len(kappas)
        avg_abs_bias = sum(biases) / len(biases) if biases else 0
        composite    = avg_kappa * (1 - avg_abs_bias)

        # FP rate on faithfulness for tiebreak
        s_faith = get_stats(df, "faithfulness")
        fp_rate = s_faith["fp"] / s_faith["n"] if s_faith else 1.0

        scores[judge] = {
            "avg_kappa": avg_kappa,
            "avg_abs_bias": avg_abs_bias,
            "composite": composite,
            "fp_rate_faith": fp_rate,
        }

    # Sort by composite score descending
    ranked = sorted(scores.items(), key=lambda x: (-x[1]["composite"], x[1]["fp_rate_faith"]))

    print(f"\n  {'Rank':<5}  {'Judge':<22}  {'Avg κ':>7}  "
          f"{'Avg |bias|':>10}  {'Composite':>10}  {'Faith FP%':>10}")
    print(f"  {'-'*5}  {'-'*22}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for rank, (judge, s) in enumerate(ranked, 1):
        marker = "  ← RECOMMENDED" if rank == 1 else ""
        print(f"  {rank:<5}  {judge:<22}  {s['avg_kappa']:>7.3f}  "
              f"{s['avg_abs_bias']:>10.1%}  {s['composite']:>10.3f}  "
              f"{s['fp_rate_faith']:>10.1%}{marker}")


def main():
    # Load available files
    dfs = {}
    for label, path in RESULT_FILES.items():
        if os.path.exists(path):
            dfs[label] = pd.read_csv(path)
            print(f"Loaded: {label} ({len(dfs[label])} rows)")
        else:
            print(f"MISSING: {label} — {path}")

    if not dfs:
        print("No files found. Check paths in RESULT_FILES.")
        sys.exit(1)

    print()

    # 1. Overall performance summary
    print_overall_summary(dfs)

    # 2. Per-metric detail
    for metric in METRICS:
        print_metric_detail(dfs, metric)
    print_metric_detail(dfs, "citation_correctness", cited_only=True)

    # 3. Bias tables per metric
    for metric in METRICS:
        print_bias_table(dfs, metric)
    print_bias_table(dfs, "citation_correctness", cited_only=True)

    # 4. Self-favour analysis
    print_self_favour(dfs)

    # 5. Overall recommendation
    print_judge_recommendation(dfs)


if __name__ == "__main__":
    main()
