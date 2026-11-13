"""Rebuild the judge-validation table (main-text Table 1 / Extended Data judge table)
from the archived judge-vs-human comparison files. No API calls.

Each row of the table is traced to exactly one file in data/judge_validation/
(the *_llm_vs_human_comparison_*.csv written by compare_llm_vs_human.py). Files are
listed in RUNS below so that every number in the table can be traced to its source.

Metrics per (metric, judge, prompt):
  n           statements with both a human and a judge label (citation correctness: cited statements only)
  agreement   % of statements where judge label == consensus human label
  kappa       Cohen's kappa
  mean_abs_bias  mean over the four summariser LLMs of |judge 'yes' rate - human 'yes' rate| (percentage points)
  bias_min / bias_max  lowest and highest signed per-summariser bias (negative = judge stricter than experts)

Output: data/derived/judge_validation_table.csv

Run from the repository root: python 03_llm_judge/build_judge_validation_table.py
"""
import glob
import os

import numpy as np
import pandas as pd

V = os.environ.get("IGF_JUDGE_VALIDATION", "data/judge_validation")
D = os.environ.get("IGF_DERIVED", "data/derived")

# (metric, judge, prompt label) -> comparison file. Refined manual prompts used in the paper:
# relevance v16, citation correctness v17, faithfulness v14-concat.
RUNS = [
    ("relevance", "Gemini 3.1 Pro Preview", "refined (v16)", "llm_vs_human_comparison_gemini-3.1-pro-preview-v16-relevance.csv"),
    ("relevance", "Claude Opus 4.7", "refined (v16)", "llm_vs_human_comparison_claude-opus-4-7-v16-relevance.csv"),
    ("relevance", "GPT-5.4", "refined (v16)", "llm_vs_human_comparison_gpt-5-4-v16-relevance.csv"),
    ("citation_correctness", "Gemini 3.1 Pro Preview", "refined (v17)", "llm_vs_human_comparison_gemini-3.1-pro-preview-v17-cc.csv"),
    ("citation_correctness", "Claude Opus 4.7", "refined (v17)", "llm_vs_human_comparison_claude-opus-4-7-v17-cc.csv"),
    ("citation_correctness", "GPT-5.4", "refined (v17)", "llm_vs_human_comparison_gpt-5-4-v17-cc.csv"),
    ("faithfulness", "Gemini 3.1 Pro Preview", "refined (v14)", "llm_vs_human_comparison_gemini-3.1-pro-preview-v14-concat.csv"),
    ("faithfulness", "Claude Opus 4.7", "refined (v14)", "llm_vs_human_comparison_claude-opus-4-7-v14-concat.csv"),
    ("faithfulness", "GPT-5.4", "refined (v14)", "llm_vs_human_comparison_gpt-5-4-v14-concat.csv"),
]
# GEPA-optimised prompts (two seeds) and the seed prompt (v7) for each judge
for judge, stem in [("Gemini 3.1 Pro Preview", "gemini-3.1-pro-preview"), ("Claude Opus 4.7", "claude-opus-4.7"), ("GPT-5.4", "gpt-5.4")]:
    for metric in ["relevance", "citation_correctness", "faithfulness"]:
        for seed in ["1234", "5678"]:
            RUNS.append((metric, judge, f"GEPA seed {seed}", f"llm_vs_human_comparison_{stem}-gepa{seed}.csv"))
        RUNS.append((metric, judge, "seed prompt (v7)", f"llm_vs_human_comparison_{stem}-seed-v7.csv"))


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "1.0"])


def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


rows = []
for metric, judge, prompt, fname in RUNS:
    path = os.path.join(V, fname)
    if not os.path.exists(path):
        rows.append({"metric": metric, "judge": judge, "prompt": prompt, "source_file": fname, "status": "FILE MISSING"})
        continue
    d = pd.read_csv(path)
    hcol, lcol = f"human_{metric}", f"llm_{metric}"
    if lcol not in d or d[lcol].notna().sum() == 0:
        rows.append({"metric": metric, "judge": judge, "prompt": prompt, "source_file": fname, "status": "metric not in file"})
        continue
    d = d[d[hcol].notna() & d[lcol].notna()]
    if metric == "citation_correctness":
        d = d[d["cited_action_id"].notna()]
    h, l = to_bool(d[hcol]), to_bool(d[lcol])
    bias = (l.groupby(d["summary_model"]).mean() - h.groupby(d["summary_model"]).mean()) * 100
    rows.append({
        "metric": metric, "judge": judge, "prompt": prompt, "source_file": fname, "status": "ok",
        "n": len(d), "agreement_pct": round(100 * (h == l).mean(), 1), "kappa": round(kappa(h, l), 3),
        "mean_abs_bias_pp": round(bias.abs().mean(), 1), "bias_min_pp": round(bias.min(), 1), "bias_max_pp": round(bias.max(), 1),
        "bias_by_summariser": "; ".join(f"{k.split('/')[-1]}={v:+.1f}" for k, v in bias.items()),
    })

tab = pd.DataFrame(rows)
os.makedirs(D, exist_ok=True)
tab.to_csv(f"{D}/judge_validation_table.csv", index=False)
pd.set_option("display.width", 250)
print(tab.drop(columns=["bias_by_summariser"]).to_string(index=False))
missing = tab[tab.status != "ok"]
if len(missing):
    print(f"\nWARNING: {len(missing)} table row(s) could not be rebuilt from the archive:")
    print(missing[["metric", "judge", "prompt", "source_file", "status"]].to_string(index=False))
