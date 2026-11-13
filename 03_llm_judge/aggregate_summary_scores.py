"""
Summary-level score aggregation for the full LLM judge evaluation.

Aggregates statement-level LLM judge verdicts up to the summary level,
producing one row per (synopsis, question, summary_model) combination.

Metrics
-------
For each summary, the following scores are computed as proportions (0-1):

  relevance
      relevant statements / total statements

  citation_rate
      cited statements / total statements
      Pure descriptive metric — how often this model attaches a citation.
      Answers: "How often does this model bother to cite at all?"

  citation_correctness_all
      citation-correct statements / total statements
      Uncited statements count as incorrect (auto-No).
      Answers: "How well does this model support its statements with citations?"
      (Combines citation rate and citation accuracy.)

  citation_correctness_when_cited
      citation-correct statements / statements that have a citation
      Uncited statements are excluded from both numerator and denominator.
      Answers: "When this model cites, how accurate are those citations?"

  faithfulness
      faithful statements / total statements

Usage
-----
    python aggregate_summary_scores.py

Configure the paths in CONFIG below.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG = {
    # LLM judge results JSON (output of evaluate_human_eval_all_reviewers.py
    # or evaluate_full_dataset.py).
    "llm_results_json": "llm_judge_results/human_eval_all_reviewers_llm_scores_claude-sonnet-4-6-v7.json",

    # Where to save the summary-level scores CSV.
    "output_csv": "llm_judge_results/summary_level_scores_claude-sonnet-4-6-v7.csv",

    # Where to save a model-level aggregate CSV (mean across all summaries).
    "output_model_csv": "llm_judge_results/model_level_scores_claude-sonnet-4-6-v7.csv",

    # Minimum number of statements a summary must have to be included.
    # Summaries with very few statements give unreliable proportions.
    "min_statements": 1,
}

# ===========================================================================


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_to_summary_level(
    records: list[dict],
    min_statements: int = 1,
) -> pd.DataFrame:
    """
    Aggregate statement-level records to summary level.

    Groups by (synopsis, question, summary_model) and computes
    proportion scores for each metric.
    """
    # Group records by summary key
    groups = defaultdict(list)
    for r in records:
        key = (
            r.get("synopsis", ""),
            r.get("question", ""),
            r.get("summary_model", ""),
        )
        groups[key].append(r)

    rows = []
    for (synopsis, question, model), stmts in groups.items():
        n = len(stmts)
        if n < min_statements:
            continue

        # --- Relevance ---
        rel_verdicts = [
            s["llm_relevance"]["verdict"]
            for s in stmts
            if s.get("llm_relevance") and s["llm_relevance"].get("verdict") is not None
        ]
        rel_score = sum(rel_verdicts) / len(rel_verdicts) if rel_verdicts else None
        rel_n = len(rel_verdicts)

        # --- Citation Correctness (all statements) ---
        cc_verdicts_all = [
            s["llm_citation_correctness"]["verdict"]
            for s in stmts
            if s.get("llm_citation_correctness")
            and s["llm_citation_correctness"].get("verdict") is not None
        ]
        cc_score_all = sum(cc_verdicts_all) / len(cc_verdicts_all) if cc_verdicts_all else None
        cc_n_all = len(cc_verdicts_all)
        cc_auto_no = sum(
            1 for s in stmts
            if s.get("llm_citation_correctness")
            and s["llm_citation_correctness"].get("auto_filled")
        )

        # --- Citation rate (descriptive: did the model cite at all?) ---
        # cc_n_all is the denominator (statements with a CC verdict at all).
        # cc_auto_no is the count of those that were auto-No because no
        # citation was attached. So citation_rate = (cited) / (total scored).
        citation_rate = (
            (cc_n_all - cc_auto_no) / cc_n_all if cc_n_all > 0 else None
        )

        # --- Citation Correctness (cited statements only) ---
        # Use llm_citation_correctness_when_cited if available,
        # otherwise derive from llm_citation_correctness by excluding auto-filled
        cc_verdicts_cited = []
        for s in stmts:
            # Prefer the dedicated when_cited field
            wc = s.get("llm_citation_correctness_when_cited")
            if wc is not None:
                if wc.get("skipped_no_citation"):
                    continue  # uncited — exclude
                if wc.get("verdict") is not None:
                    cc_verdicts_cited.append(wc["verdict"])
            else:
                # Fall back: use cc field but skip auto-filled
                cc = s.get("llm_citation_correctness")
                if cc and not cc.get("auto_filled") and cc.get("verdict") is not None:
                    cc_verdicts_cited.append(cc["verdict"])

        cc_score_cited = sum(cc_verdicts_cited) / len(cc_verdicts_cited) if cc_verdicts_cited else None
        cc_n_cited = len(cc_verdicts_cited)

        # --- Faithfulness ---
        faith_verdicts = [
            s["llm_faithfulness"]["verdict"]
            for s in stmts
            if s.get("llm_faithfulness") and s["llm_faithfulness"].get("verdict") is not None
        ]
        faith_score = sum(faith_verdicts) / len(faith_verdicts) if faith_verdicts else None
        faith_n = len(faith_verdicts)

        rows.append({
            "synopsis":              synopsis,
            "question":              question,
            "summary_model":         model,
            "n_statements":          n,

            # Relevance
            "relevance_score":       rel_score,
            "relevance_n":           rel_n,

            # Citation rate (descriptive)
            "citation_rate":         citation_rate,

            # CC (all statements — uncited = incorrect)
            "cc_all_score":          cc_score_all,
            "cc_all_n":              cc_n_all,
            "cc_auto_no_count":      cc_auto_no,

            # CC (cited only — uncited excluded)
            "cc_cited_score":        cc_score_cited,
            "cc_cited_n":            cc_n_cited,

            # Faithfulness
            "faithfulness_score":    faith_score,
            "faithfulness_n":        faith_n,
        })

    return pd.DataFrame(rows)


def print_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return model-level aggregate scores."""
    print("\n" + "=" * 75)
    print("SUMMARY-LEVEL SCORES — PER CANDIDATE MODEL")
    print("(mean across all summaries for that model)")
    print("=" * 75)

    score_cols = [
        ("relevance_score",  "Relevance"),
        ("citation_rate",    "Cite rate"),
        ("cc_all_score",     "CC (all stmts)"),
        ("cc_cited_score",   "CC (cited only)"),
        ("faithfulness_score", "Faithfulness"),
    ]

    # Header
    hdr = f"  {'Model':<45s}"
    for _, label in score_cols:
        hdr += f"  {label:>15s}"
    hdr += f"  {'N summ':>7}  {'Avg stmts':>9}"
    print(hdr)
    print("  " + "-" * 45 + "  " + "  ".join(["-" * 15] * len(score_cols)) + "  " + "-" * 7 + "  " + "-" * 9)

    model_rows = []
    for model in sorted(df["summary_model"].unique()):
        mdf = df[df["summary_model"] == model]
        n_summ = len(mdf)
        avg_stmts = mdf["n_statements"].mean()

        row = {"summary_model": model, "n_summaries": n_summ, "avg_statements": round(avg_stmts, 1)}
        line = f"  {model:<45s}"
        for col, label in score_cols:
            mean_score = mdf[col].mean()
            row[col] = round(mean_score, 3) if pd.notna(mean_score) else None
            val_str = f"{mean_score:.1%}" if pd.notna(mean_score) else "N/A"
            line += f"  {val_str:>15s}"
        line += f"  {n_summ:>7}  {avg_stmts:>9.1f}"
        print(line)
        model_rows.append(row)

    # Overall
    print("  " + "-" * 45 + "  " + "  ".join(["-" * 15] * len(score_cols)) + "  " + "-" * 7 + "  " + "-" * 9)
    line = f"  {'OVERALL':<45s}"
    overall_row = {"summary_model": "OVERALL", "n_summaries": len(df), "avg_statements": round(df["n_statements"].mean(), 1)}
    for col, label in score_cols:
        mean_score = df[col].mean()
        overall_row[col] = round(mean_score, 3) if pd.notna(mean_score) else None
        val_str = f"{mean_score:.1%}" if pd.notna(mean_score) else "N/A"
        line += f"  {val_str:>15s}"
    line += f"  {len(df):>7}  {df['n_statements'].mean():>9.1f}"
    print(line)
    model_rows.append(overall_row)

    print()
    print("  Notes:")
    print("  - Cite rate:        proportion of statements that had any citation attached")
    print("  - CC (all stmts):   uncited statements scored as incorrect")
    print("  - CC (cited only):  uncited statements excluded from calculation")
    print("  - Scores are means of per-summary proportions (not statement-level rates)")
    print("=" * 75)

    return pd.DataFrame(model_rows)


def main():
    cfg = CONFIG

    llm_path = Path(cfg["llm_results_json"])
    if not llm_path.exists():
        print(f"ERROR: Results file not found: {llm_path}")
        sys.exit(1)

    print(f"Loading results from: {llm_path}")
    records = load_results(str(llm_path))
    print(f"Total statement records: {len(records)}")

    # Extract judge model name for display
    judge_model = records[0].get("judge_model", "unknown") if records else "unknown"
    print(f"Judge model: {judge_model}")

    # Aggregate to summary level
    df = aggregate_to_summary_level(records, min_statements=cfg.get("min_statements", 1))
    print(f"Total summaries: {len(df)}")

    # Print model-level summary and get aggregate df
    model_df = print_model_summary(df)

    # Save outputs
    if cfg.get("output_csv"):
        out = Path(cfg["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSummary-level scores saved to: {out}")

    if cfg.get("output_model_csv"):
        out = Path(cfg["output_model_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        model_df.to_csv(out, index=False)
        print(f"Model-level scores saved to:   {out}")


if __name__ == "__main__":
    main()
