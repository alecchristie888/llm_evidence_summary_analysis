"""
General agreement analysis: human reviewer scores vs LLM judge scores.

Merges any number of human evaluation Excel files, matches each human-scored
statement against an LLM judge results JSON, and computes agreement statistics.

Usage
-----
    python compare_llm_vs_human.py

Configure the paths in CONFIG below. Add or remove files from
``reviewer_files`` as more reviewers complete their evaluations.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders.human_eval_loader import load_human_eval_records, _normalise_for_matching


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG = {
    # All human evaluation Excel files to include.
    # Group them by reviewer for clarity — order does not matter.
    "reviewer_files": [
        # Reviewer 1
        "human_data/sampled_subset_reviewer1_original.xlsx",
        "human_data/sampled_subset_reviewer1_extras_AT.xlsx",
        # Reviewer 2
        "human_data/sampled_subset_reviewer2_original.xlsx",
        "human_data/sampled_subset_reviewer2_extras_VC.xlsx",
        # Reviewer 3
        "human_data/sampled_subset_reviewer3_NT.xlsx",
        "human_data/sampled_subset_reviewer3_extras_NT.xlsx"
    ],

    # LLM judge results JSON (output of evaluate_human_eval_all_reviewers.py).
    # Change this filename to switch between judge models.
    "llm_results_json": "llm_judge_results/human_eval_all_reviewers_llm_scores_gemini-3.1-pro-preview-finalprompts.json",

    # Filter to answerable statements only.
    "answerable_only": True,

    # Where to save the full per-row comparison CSV (set to None to skip).
    "output_csv": "llm_judge_results/llm_vs_human_comparison_gemini-3.1-pro-preview-finalprompts.csv",
}

# ===========================================================================


def load_llm_results(path: str) -> tuple[dict, dict]:
    """
    Load LLM results JSON and build two indexes:
      - primary:  (question, norm_summary, model, statement)
      - fallback: (question, model, statement)  — used when summary differs slightly
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    primary = {}
    fallback = {}
    for r in data:
        pkey = (
            r.get("question", "").strip(),
            _normalise_for_matching(r.get("summary", "")),
            r.get("summary_model", "").strip(),
            r.get("statement", "").strip(),
        )
        fkey = (
            r.get("question", "").strip(),
            r.get("summary_model", "").strip(),
            r.get("statement", "").strip(),
        )
        primary[pkey] = r
        fallback[fkey] = r
    print(f"Loaded {len(data)} LLM results ({len(primary)} unique keys).")
    return primary, fallback


def build_comparison_df(reviewer_files, llm_index, llm_fallback=None, answerable_only=True) -> pd.DataFrame:
    """Load all reviewer files and match each scored row against LLM results."""
    rows = []
    seen_keys = {}  # dedup: key -> first source file
    fallback_used = 0

    for fpath in reviewer_files:
        records = load_human_eval_records(fpath, answerable_only=answerable_only)
        print(f"  {Path(fpath).name}: {len(records)} records")

        for rec in records:
            has_human = any([
                rec.human_relevance is not None,
                rec.human_citation_correctness is not None,
                rec.human_faithfulness is not None,
            ])
            if not has_human:
                continue

            match_key = (
                rec.question.strip(),
                _normalise_for_matching(rec.summary),
                rec.summary_model.strip(),
                rec.statement.strip(),
            )

            # Track which reviewer file first provided scores for this statement
            if match_key not in seen_keys:
                seen_keys[match_key] = Path(fpath).name

            llm = llm_index.get(match_key)

            # Fallback: match by (question, model, statement) if summary differs slightly
            if llm is None and llm_fallback is not None:
                fkey = (
                    rec.question.strip(),
                    rec.summary_model.strip(),
                    rec.statement.strip(),
                )
                llm = llm_fallback.get(fkey)
                if llm is not None:
                    fallback_used += 1

            def get_verdict(field):
                if llm is None:
                    return None
                v = llm.get(field)
                return v.get("verdict") if v else None

            def get_auto(field):
                if llm is None:
                    return False
                v = llm.get(field)
                return v.get("auto_filled", False) if v else False

            rows.append({
                "source_file":     Path(fpath).name,
                "synopsis":        rec.synopsis,
                "answerability":   rec.answerability,
                "question":        rec.question,
                "summary_model":   rec.summary_model,
                "statement":       rec.statement[:150],
                "cited_action_id": rec.cited_action_id,
                "reviewer":        rec.reviewer,
                "second_reviewer": rec.second_reviewer,

                "human_relevance":             rec.human_relevance,
                "human_citation_correctness":  rec.human_citation_correctness,
                "human_faithfulness":          rec.human_faithfulness,

                "llm_relevance":             get_verdict("llm_relevance"),
                "llm_citation_correctness":  get_verdict("llm_citation_correctness"),
                "llm_faithfulness":          get_verdict("llm_faithfulness"),
                "cc_auto_filled":            get_auto("llm_citation_correctness"),

                "llm_matched": llm is not None,
                # Faithfulness confidence (only present in new runs with confidence field)
                "llm_faithfulness_confidence": (
                    (llm.get("llm_faithfulness") or {}).get("confidence")
                    if llm else None
                ),
            })

    if fallback_used > 0:
        print(f"  ({fallback_used} records matched via fallback key — summary text differed slightly)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement statistics helpers
# ---------------------------------------------------------------------------

def _cohen_kappa(h_series, l_series):
    """
    Compute Cohen's Kappa for two boolean/None series.
    Returns (kappa, n_pairs) where n_pairs is the number of rows
    where both scores are non-null.
    """
    pairs = [
        (h, l) for h, l in zip(h_series, l_series)
        if h is not None and l is not None
        and str(h) not in ('nan', 'None')
        and str(l) not in ('nan', 'None')
    ]
    n = len(pairs)
    if n == 0:
        return None, 0

    agree = sum(1 for h, l in pairs if h == l)
    p_o = agree / n

    p1_yes = sum(1 for h, _ in pairs if h is True) / n
    p2_yes = sum(1 for _, l in pairs if l is True) / n
    p_e = (p1_yes * p2_yes) + ((1 - p1_yes) * (1 - p2_yes))

    if p_e >= 1.0:
        return 1.0, n
    return (p_o - p_e) / (1 - p_e), n


def _kappa_label(k):
    if k is None:   return "N/A"
    if k < 0:       return "worse than chance"
    if k < 0.20:    return "slight"
    if k < 0.40:    return "fair"
    if k < 0.60:    return "moderate"
    if k < 0.80:    return "substantial"
    return "almost perfect"


def _agreement_block(df, metric, label=None):
    h_col, l_col = f"human_{metric}", f"llm_{metric}"
    paired_all = df[df[h_col].notna() & df[l_col].notna()]

    tag = f"  [{label}]" if label else ""

    def _print_stats(paired, subtitle=None):
        n = len(paired)
        if n == 0:
            print(f"  {metric:25s}  No paired scores.")
            return
        agree = (paired[h_col] == paired[l_col]).sum()
        tp = int(((paired[h_col] == True)  & (paired[l_col] == True)).sum())
        tn = int(((paired[h_col] == False) & (paired[l_col] == False)).sum())
        fp = int(((paired[h_col] == False) & (paired[l_col] == True)).sum())
        fn = int(((paired[h_col] == True)  & (paired[l_col] == False)).sum())
        kappa, _ = _cohen_kappa(paired[h_col].tolist(), paired[l_col].tolist())
        k_str = f"{kappa:.3f} ({_kappa_label(kappa)})" if kappa is not None else "N/A"

        # Precision, Recall, F1, Specificity
        precision   = tp / (tp + fp) if (tp + fp) > 0 else None
        recall      = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None
              and (precision + recall) > 0 else None)

        def fmt(v): return f"{v:.3f}" if v is not None else "N/A"

        heading = f"  {metric}{tag}" + (f" ({subtitle})" if subtitle else "")
        print(heading)
        print(f"    Paired: {n}  |  Agreement: {agree}/{n} ({agree/n:.1%})  |  Kappa: {k_str}")
        print(f"    TP:{tp}  TN:{tn}  FP:{fp} (LLM too lenient)  FN:{fn} (LLM too strict)")
        print(f"    Precision: {fmt(precision)}  |  Recall: {fmt(recall)}  |  "
              f"F1: {fmt(f1)}  |  Specificity: {fmt(specificity)}")

    # Always print the full metric including auto-Nos
    _print_stats(paired_all, subtitle="all statements" if metric == "citation_correctness" else None)

    # For faithfulness, also print confidence breakdown if available
    if metric == "faithfulness" and "llm_faithfulness_confidence" in df.columns:
        conf_col = "llm_faithfulness_confidence"
        scored = df[df[h_col].notna() & df[l_col].notna()]
        for level in ("high", "medium", "low"):
            level_df = scored[scored[conf_col] == level]
            if len(level_df) == 0:
                continue
            n_l = len(level_df)
            agree_l = (level_df[h_col] == level_df[l_col]).sum()
            fp_l = ((level_df[h_col]==False)&(level_df[l_col]==True)).sum()
            fn_l = ((level_df[h_col]==True) &(level_df[l_col]==False)).sum()
            print(f"    confidence={level:<6}  n={n_l:>3}  "
                  f"agree:{agree_l/n_l:.0%}  FP:{fp_l}  FN:{fn_l}")

    # For CC, also print with auto-Nos excluded
    if metric == "citation_correctness" and "cc_auto_filled" in df.columns:
        auto_no_count = int(paired_all["cc_auto_filled"].sum())
        if auto_no_count > 0:
            paired_cited = paired_all[~paired_all["cc_auto_filled"]]
            _print_stats(paired_cited, subtitle=f"cited only, {auto_no_count} auto-No excluded")


def print_summary(df: pd.DataFrame, llm_results_path: str) -> None:
    judge_model = "unknown"
    try:
        with open(llm_results_path, encoding="utf-8") as f:
            sample = json.load(f)
        if sample:
            judge_model = sample[0].get("judge_model", "unknown")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print(f"LLM-AS-JUDGE vs HUMAN — AGREEMENT SUMMARY")
    print(f"Judge model : {judge_model}")
    print("=" * 70)
    print(f"Total human-scored rows : {len(df)}")
    print(f"Matched to LLM results  : {df['llm_matched'].sum()}")
    print(f"No LLM match found      : {(~df['llm_matched']).sum()}")
    print()

    # --- Overall ---
    print("--- Overall ---")
    for metric in ["relevance", "citation_correctness", "faithfulness"]:
        _agreement_block(df, metric)
    print()

    # --- Per summary model ---
    print("--- Per summary model ---")
    print("    CC shown as: all statements / cited only")
    for model in sorted(df["summary_model"].unique()):
        mdf = df[df["summary_model"] == model]
        parts = []
        for metric in ["relevance", "citation_correctness", "faithfulness"]:
            h, l = f"human_{metric}", f"llm_{metric}"
            paired_all = mdf[mdf[h].notna() & mdf[l].notna()]
            n_all = len(paired_all)
            if n_all == 0:
                continue
            a_all = (paired_all[h] == paired_all[l]).sum()
            fn_all = ((paired_all[h] == True) & (paired_all[l] == False)).sum()
            fp_all = ((paired_all[h] == False) & (paired_all[l] == True)).sum()
            kappa_all, _ = _cohen_kappa(paired_all[h].tolist(), paired_all[l].tolist())
            k_all = f"{kappa_all:.2f}" if kappa_all is not None else "N/A"

            if metric == "citation_correctness" and "cc_auto_filled" in paired_all.columns:
                paired_cited = paired_all[~paired_all["cc_auto_filled"]]
                n_c = len(paired_cited)
                if n_c > 0:
                    a_c = (paired_cited[h] == paired_cited[l]).sum()
                    kappa_c, _ = _cohen_kappa(paired_cited[h].tolist(), paired_cited[l].tolist())
                    k_c = f"{kappa_c:.2f}" if kappa_c is not None else "N/A"
                    parts.append(
                        f"citati:{a_all}/{n_all}({a_all/n_all:.0%}) κ={k_all} / "
                        f"cited:{a_c}/{n_c}({a_c/n_c:.0%}) κ={k_c} FP:{fp_all} FN:{fn_all}"
                    )
                    continue
            parts.append(f"{metric[:6]}:{a_all}/{n_all}({a_all/n_all:.0%}) κ={k_all} FP:{fp_all} FN:{fn_all}")
        print(f"  {model:<45s}  {', '.join(parts)}")
    print()

    # --- Per reviewer file ---
    print("--- Per reviewer file ---")
    for src in sorted(df["source_file"].unique()):
        sdf = df[df["source_file"] == src]
        parts = []
        for metric in ["relevance", "citation_correctness", "faithfulness"]:
            h, l = f"human_{metric}", f"llm_{metric}"
            paired = sdf[sdf[h].notna() & sdf[l].notna()]
            n = len(paired)
            if n > 0:
                a = (paired[h] == paired[l]).sum()
                parts.append(f"{metric[:6]}:{a}/{n}({a/n:.0%})")
        print(f"  {src:<50s}  {', '.join(parts)}")
    print()

    # --- Self-favour check ---
    print("--- Judge self-favour check (LLM Yes rate vs human Yes rate, with Kappa) ---")
    for metric in ["relevance", "citation_correctness", "faithfulness"]:
        h, l = f"human_{metric}", f"llm_{metric}"
        is_cc = metric == "citation_correctness" and "cc_auto_filled" in df.columns
        label = f"  {metric}" + (" (all / cited only):" if is_cc else ":")
        print(label)
        for model in sorted(df["summary_model"].unique()):
            mdf = df[df["summary_model"] == model]
            paired_all = mdf[mdf[h].notna() & mdf[l].notna()]
            if len(paired_all) == 0:
                continue
            is_judge = any(part in judge_model.lower()
                           for part in model.lower().replace("/", " ").split())
            marker = "  <-- JUDGE MODEL" if is_judge else ""

            # All-statements line
            h_yes = paired_all[h].mean()
            l_yes = paired_all[l].mean()
            kappa, _ = _cohen_kappa(paired_all[h].tolist(), paired_all[l].tolist())
            k_str = f"{kappa:.3f} ({_kappa_label(kappa)})" if kappa is not None else "N/A"
            print(f"    {model:<45s}  human:{h_yes:.1%}  llm:{l_yes:.1%}  "
                  f"diff:{l_yes - h_yes:+.1%}  κ:{k_str}{marker}")

            # CC: also show cited-only line
            if is_cc:
                paired_cited = paired_all[~paired_all["cc_auto_filled"]]
                if len(paired_cited) > 0:
                    h_yes_c = paired_cited[h].mean()
                    l_yes_c = paired_cited[l].mean()
                    kappa_c, _ = _cohen_kappa(paired_cited[h].tolist(), paired_cited[l].tolist())
                    k_str_c = f"{kappa_c:.3f} ({_kappa_label(kappa_c)})" if kappa_c is not None else "N/A"
                    print(f"    {'  cited only':<45s}  human:{h_yes_c:.1%}  llm:{l_yes_c:.1%}  "
                          f"diff:{l_yes_c - h_yes_c:+.1%}  κ:{k_str_c}")
    print("=" * 70)


def main():
    cfg = CONFIG

    llm_path = Path(cfg["llm_results_json"])
    if not llm_path.exists():
        print(f"ERROR: LLM results file not found: {llm_path}")
        print("Run evaluate_human_eval_all_reviewers.py first.")
        sys.exit(1)

    print(f"LLM results: {llm_path}")
    llm_index, llm_fallback = load_llm_results(str(llm_path))

    print("Loading reviewer files...")
    df = build_comparison_df(
        cfg["reviewer_files"],
        llm_index,
        llm_fallback=llm_fallback,
        answerable_only=cfg.get("answerable_only", True),
    )

    print_summary(df, str(llm_path))

    if cfg.get("output_csv"):
        out = Path(cfg["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nFull comparison saved to: {out}")


if __name__ == "__main__":
    main()
