"""
Inter-rater reliability analysis between human reviewers.

Finds statements that were scored by more than one reviewer and computes:
  - Raw agreement rate
  - Cohen's Kappa (accounts for chance agreement)
  - Per-metric and per-reviewer-pair breakdowns
  - A CSV of all disagreements for manual inspection

Usage
-----
    python inter_rater_reliability.py

Add reviewer files to REVIEWER_GROUPS in CONFIG below. Each group is a list
of Excel files belonging to the same reviewer — their scores are merged before
comparison. Add Reviewer 3 files when available.

How overlap is detected
-----------------------
Two records are considered the same statement if they share the same
(question, normalised_summary, summary_model, statement) key. This is the
same key used by the LLM judge for deduplication.
"""

import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders.human_eval_loader import load_human_eval_records, _normalise_for_matching


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG = {
    # Each entry is a (reviewer_name, [list of Excel files]) pair.
    # Files for the same reviewer are merged before comparison.
    "reviewer_groups": [
        ("Reviewer_1", [
            "data/human_labels/sampled_subset_reviewer1_original.xlsx",
            "data/human_labels/sampled_subset_reviewer1_extras_AT.xlsx",
        ]),
        ("Reviewer_2", [
            "data/human_labels/sampled_subset_reviewer2_original.xlsx",
            "data/human_labels/sampled_subset_reviewer2_extras_VC.xlsx",
        ]),
        ("Reviewer_3", [
            "data/human_labels/sampled_subset_reviewer3_NT.xlsx",
            "data/human_labels/sampled_subset_reviewer3_extras_NT.xlsx",
        ]),
    ],

    # Filter to answerable statements only.
    "answerable_only": True,

    # Where to save the disagreement detail CSV (set to None to skip).
    "output_disagreements_csv": "data/derived/inter_rater_disagreements.csv",

    # Where to save the full overlap CSV (all paired rows, agree and disagree).
    "output_overlap_csv": "data/derived/inter_rater_overlap.csv",
}

# ===========================================================================


def record_key(rec):
    # Reviewer workbooks can differ by a stray newline inside the same summary
    # (two of the 57 doubly-rated statements did), so strip all whitespace
    # before matching; the summary is only used here as a lookup key.
    summary = re.sub(r"\s+", "", _normalise_for_matching(rec.summary) or "")
    return (
        rec.question.strip(),
        summary,
        rec.summary_model.strip(),
        rec.statement.strip(),
    )


def load_reviewer(name: str, files: list, answerable_only: bool) -> dict:
    """
    Load all files for one reviewer and return a dict of key -> record.
    Where a key appears in multiple files, the last file wins (extras
    typically supersede originals for the same statements).
    """
    scored = {}
    for fpath in files:
        records = load_human_eval_records(fpath, answerable_only=answerable_only)
        for rec in records:
            has_score = any(x is not None for x in [
                rec.human_relevance,
                rec.human_citation_correctness,
                rec.human_faithfulness,
            ])
            if has_score:
                scored[record_key(rec)] = rec
    print(f"  {name}: {len(scored)} scored statements loaded")
    return scored


def fleiss_kappa(ratings_by_subject: list[list]) -> float | None:
    """
    Compute Fleiss' Kappa for multiple raters on binary (True/False) data.

    Parameters
    ----------
    ratings_by_subject : list of lists
        Each inner list contains the ratings from all raters who scored that
        subject (statement). Ratings must be True or False. None values are
        excluded. Subjects rated by fewer than 2 raters are skipped.

    Returns
    -------
    float or None
        Fleiss' Kappa, or None if there is insufficient data.

    Notes
    -----
    Fleiss' Kappa generalises Cohen's Kappa to any number of raters.
    Unlike Cohen's, it does not require each subject to be rated by the
    same number of raters, though the standard formula assumes a fixed
    number per subject. Here we use the general formulation that handles
    variable rater counts.
    """
    # Filter to subjects with at least 2 valid ratings
    subjects = [
        [r for r in ratings if r is not None]
        for ratings in ratings_by_subject
    ]
    subjects = [s for s in subjects if len(s) >= 2]

    if not subjects:
        return None

    n_subjects = len(subjects)
    # Total number of ratings across all subjects
    total_ratings = sum(len(s) for s in subjects)
    if total_ratings == 0:
        return None

    # Overall proportion of Yes ratings (used for expected agreement)
    total_yes = sum(sum(1 for r in s if r is True) for s in subjects)
    p_yes = total_yes / total_ratings
    p_no  = 1.0 - p_yes

    # Expected agreement by chance
    p_e = p_yes ** 2 + p_no ** 2

    # Observed agreement: for each subject, proportion of agreeing pairs
    p_o_sum = 0.0
    for s in subjects:
        n_i = len(s)
        n_yes_i = sum(1 for r in s if r is True)
        n_no_i  = n_i - n_yes_i
        # Number of agreeing pairs out of all pairs
        agreeing = n_yes_i * (n_yes_i - 1) + n_no_i * (n_no_i - 1)
        total_pairs = n_i * (n_i - 1)
        p_o_sum += agreeing / total_pairs if total_pairs > 0 else 0

    p_o = p_o_sum / n_subjects

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohen_kappa(v1_list, v2_list):
    """
    Compute Cohen's Kappa for two lists of binary (True/False) ratings.
    Returns None if there is insufficient data.
    """
    pairs = [(a, b) for a, b in zip(v1_list, v2_list)
             if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n  # observed agreement

    # Expected agreement by chance
    p1_yes = sum(1 for a, _ in pairs if a is True) / n
    p2_yes = sum(1 for _, b in pairs if b is True) / n
    p_e = (p1_yes * p2_yes) + ((1 - p1_yes) * (1 - p2_yes))

    if p_e == 1.0:
        return 1.0  # perfect agreement, no variance
    return (p_o - p_e) / (1 - p_e)


def kappa_interpretation(k):
    if k is None:
        return "N/A"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def analyse_pair(name1, scored1, name2, scored2):
    """Compare two reviewers on their overlapping statements."""
    overlap_keys = sorted(set(scored1.keys()) & set(scored2.keys()))  # sorted: stable row order across runs
    if not overlap_keys:
        print(f"  {name1} vs {name2}: NO overlapping statements found.")
        return pd.DataFrame()

    print(f"\n  {name1} vs {name2} — {len(overlap_keys)} overlapping statements")
    print(f"  {'Metric':<35s}  {'Agree':>6}  {'Total':>6}  {'Raw%':>6}  {'Kappa':>6}  Interpretation")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")

    rows = []
    metrics = ["human_relevance", "human_citation_correctness", "human_faithfulness"]

    for metric in metrics:
        # --- All statements (original) ---
        v1_all, v2_all = [], []
        v1_cited, v2_cited = [], []
        for key in overlap_keys:
            r1 = scored1[key]
            r2 = scored2[key]
            v1 = getattr(r1, metric)
            v2 = getattr(r2, metric)
            v1_all.append(v1)
            v2_all.append(v2)

            # For cited-only: exclude where both have no citation
            if metric == "human_citation_correctness":
                if r1.cited_action_id or r2.cited_action_id:
                    v1_cited.append(v1)
                    v2_cited.append(v2)

        def _print_row(v1_list, v2_list, label):
            paired = [(a, b) for a, b in zip(v1_list, v2_list)
                      if a is not None and b is not None]
            n = len(paired)
            if n == 0:
                return
            agree = sum(1 for a, b in paired if a == b)
            tp = sum(1 for a, b in paired if a is True  and b is True)
            tn = sum(1 for a, b in paired if a is False and b is False)
            fp = sum(1 for a, b in paired if a is False and b is True)
            fn = sum(1 for a, b in paired if a is True  and b is False)
            kappa = cohen_kappa(v1_list, v2_list)
            k_str = f"{kappa:.3f}" if kappa is not None else "N/A"
            precision   = tp / (tp + fp) if (tp + fp) > 0 else None
            recall      = tp / (tp + fn) if (tp + fn) > 0 else None
            specificity = tn / (tn + fp) if (tn + fp) > 0 else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None
                  and (precision + recall) > 0 else None)
            def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
            print(f"  {label:<35s}  {agree:>6}  {n:>6}  "
                  f"{agree/n:>5.1%}  {k_str:>6}  {kappa_interpretation(kappa)}")
            print(f"  {'':35s}  Prec:{fmt(precision)}  Rec:{fmt(recall)}  "
                  f"F1:{fmt(f1)}  Spec:{fmt(specificity)}")

        short = metric.replace("human_", "")
        _print_row(v1_all, v2_all, short)
        if metric == "human_citation_correctness" and v1_cited:
            _print_row(v1_cited, v2_cited, f"{short} (cited only)")

    # Build detailed rows for CSV
    for key in overlap_keys:
        r1 = scored1[key]
        r2 = scored2[key]
        for metric in metrics:
            v1 = getattr(r1, metric)
            v2 = getattr(r2, metric)
            if v1 is None and v2 is None:
                continue
            rows.append({
                "reviewer_1":    name1,
                "reviewer_2":    name2,
                "synopsis":      r1.synopsis,
                "question":      r1.question,
                "summary_model": r1.summary_model,
                "statement":     r1.statement[:150],
                "metric":        metric.replace("human_", ""),
                f"{name1}_score": v1,
                f"{name2}_score": v2,
                "agree":         v1 == v2 if (v1 is not None and v2 is not None) else None,
            })

    return pd.DataFrame(rows)


def print_overall_kappas(
    all_pairs_df: pd.DataFrame,
    reviewer_data: dict,
) -> None:
    """
    Print a summary Kappa table across all reviewer pairs, plus Fleiss' Kappa
    computed over all statements rated by 2+ reviewers simultaneously.
    """
    if all_pairs_df.empty:
        return

    metrics = ["relevance", "citation_correctness", "faithfulness"]

    # --- Pairwise Cohen's Kappa ---
    print("\n--- Pairwise Cohen's Kappa ---")
    pairs = all_pairs_df[["reviewer_1", "reviewer_2"]].drop_duplicates()

    for _, pair_row in pairs.iterrows():
        r1, r2 = pair_row["reviewer_1"], pair_row["reviewer_2"]
        pair_df = all_pairs_df[
            (all_pairs_df["reviewer_1"] == r1) &
            (all_pairs_df["reviewer_2"] == r2)
        ]
        print(f"  {r1} vs {r2}:")
        for metric in metrics:
            mdf = pair_df[pair_df["metric"] == metric].copy()
            s1_col = f"{r1}_score"
            s2_col = f"{r2}_score"
            if s1_col not in mdf.columns or s2_col not in mdf.columns:
                continue
            suffix = " (cited only)" if metric == "citation_correctness" else ""
            kappa = cohen_kappa(mdf[s1_col].tolist(), mdf[s2_col].tolist())
            k_str = f"{kappa:.3f}" if kappa is not None else "N/A"
            n = mdf[s1_col].notna().sum()
            agree = mdf["agree"].sum() if "agree" in mdf.columns else 0
            print(f"    {metric}{suffix:<25s}  n={n:>3}  agree:{agree/n:.1%}  "
                  f"kappa:{k_str}  ({kappa_interpretation(kappa)})")

    # --- Average pairwise Kappa ---
    print("\n--- Average pairwise Cohen's Kappa ---")
    for metric in metrics:
        kappas = []
        for _, pair_row in pairs.iterrows():
            r1, r2 = pair_row["reviewer_1"], pair_row["reviewer_2"]
            pair_df = all_pairs_df[
                (all_pairs_df["reviewer_1"] == r1) &
                (all_pairs_df["reviewer_2"] == r2)
            ]
            mdf = pair_df[pair_df["metric"] == metric]
            s1_col, s2_col = f"{r1}_score", f"{r2}_score"
            if s1_col not in mdf.columns or s2_col not in mdf.columns:
                continue
            k = cohen_kappa(mdf[s1_col].tolist(), mdf[s2_col].tolist())
            if k is not None:
                kappas.append(k)
        if kappas:
            avg_k = sum(kappas) / len(kappas)
            suffix = " (cited only)" if metric == "citation_correctness" else ""
            print(f"  {metric}{suffix:<25s}  avg kappa: {avg_k:.3f}  "
                  f"({kappa_interpretation(avg_k)})  "
                  f"[{len(kappas)} pairs: {', '.join(f'{k:.3f}' for k in kappas)}]")

    # --- Fleiss' Kappa ---
    print("\n--- Fleiss' Kappa (all reviewers simultaneously) ---")
    print("    Uses all statements rated by 2+ reviewers.")
    reviewer_names = list(reviewer_data.keys())

    for metric in metrics:
        attr = f"human_{metric}"

        # Collect all unique statement keys across all reviewers
        all_keys = set()
        for scored in reviewer_data.values():
            all_keys.update(scored.keys())

        # For each statement, gather ratings from all reviewers who scored it
        ratings_by_subject = []
        for key in sorted(all_keys):
            ratings = []
            for name in reviewer_names:
                scored = reviewer_data[name]
                if key in scored:
                    r = scored[key]
                    # For CC cited-only: skip if no citation
                    if metric == "citation_correctness":
                        v = getattr(r, attr)
                        if not r.cited_action_id:
                            continue  # exclude uncited
                    else:
                        v = getattr(r, attr)
                    if v is not None:
                        ratings.append(v)
            if len(ratings) >= 2:
                ratings_by_subject.append(ratings)

        fk = fleiss_kappa(ratings_by_subject)
        n_subjects = len(ratings_by_subject)
        suffix = " (cited only)" if metric == "citation_correctness" else ""
        fk_str = f"{fk:.3f}" if fk is not None else "N/A"
        print(f"  {metric}{suffix:<25s}  n={n_subjects:>3} subjects  "
              f"fleiss_kappa: {fk_str}  ({kappa_interpretation(fk)})")


def main():
    cfg = CONFIG
    answerable_only = cfg.get("answerable_only", True)

    print("Loading reviewer files...")
    reviewer_data = {}
    for name, files in cfg["reviewer_groups"]:
        reviewer_data[name] = load_reviewer(name, files, answerable_only)

    print()
    print("=" * 70)
    print("INTER-RATER RELIABILITY ANALYSIS")
    print("=" * 70)

    all_pair_dfs = []

    for (name1, scored1), (name2, scored2) in combinations(reviewer_data.items(), 2):
        pair_df = analyse_pair(name1, scored1, name2, scored2)
        if not pair_df.empty:
            all_pair_dfs.append(pair_df)

    if not all_pair_dfs:
        print("\nNo overlapping statements found between any reviewer pair.")
        return

    combined = pd.concat(all_pair_dfs, ignore_index=True)

    print_overall_kappas(combined, reviewer_data)

    # Save outputs
    if cfg.get("output_overlap_csv"):
        out = Path(cfg["output_overlap_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\nFull overlap data saved to: {out}")

    if cfg.get("output_disagreements_csv"):
        disagree = combined[combined["agree"] == False]
        out = Path(cfg["output_disagreements_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        disagree.to_csv(out, index=False)
        print(f"Disagreements only saved to: {out}  ({len(disagree)} rows)")

    print()


if __name__ == "__main__":
    main()
