"""
cleanup_failed_records.py

Removes records with any failed (verdict: None) LLM metric from the bake-off
output JSONs so that re-running the evaluation script will retry them.

Why "remove" instead of "patch in place"?
  evaluate_record() always re-runs every metric in metrics_to_run; it does NOT
  check whether a metric already has a value. And build_done_set() treats any
  record present in the output file as "done". So the only way to retry a
  failed metric on a record is to remove the entire record from the file.

Behaviour:
  - For each output file:
      1. Backs up to <file>.bak (only if no backup exists yet).
      2. Loads the JSON list.
      3. Drops any record where ANY llm_* field has verdict == None.
         (Also drops llm_faithfulness_per_doc / llm_faithfulness_majority_vote
         if their top-level verdict is None.)
      4. Writes the cleaned list back.
  - Prints a per-file summary: kept, dropped, unique-statements dropped.

Run: python cleanup_failed_records.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

OUTPUT_DIR = Path("llm_judge_results")

OUTPUT_FILES = [
    "human_eval_all_reviewers_llm_scores_claude-opus-4.7-gepa1234.json",
    "human_eval_all_reviewers_llm_scores_claude-opus-4.7-gepa5678.json",
    "human_eval_all_reviewers_llm_scores_claude-opus-4-7-seed-v7.json",
    #"human_eval_all_reviewers_llm_scores_gemini-3.1-pro-preview-seed-v7.json",
    #"human_eval_all_reviewers_llm_scores_gemini-3.1-pro-preview-gepa1234.json",
    #"human_eval_all_reviewers_llm_scores_gemini-3.1-pro-preview-gepa5678.json",
    #"human_eval_all_reviewers_llm_scores_gpt-5.4-seed-v7.json",
    #"human_eval_all_reviewers_llm_scores_gpt-5.4-gepa1234.json",
    #"human_eval_all_reviewers_llm_scores_gpt-5.4-gepa5678.json",
]

# Which top-level fields to inspect for verdict == None
METRIC_FIELDS = (
    "llm_relevance",
    "llm_citation_correctness",
    "llm_citation_correctness_when_cited",
    "llm_faithfulness",
    #"llm_faithfulness_per_doc", not run
    #"llm_faithfulness_majority_vote", not run
)


def has_failed_metric(record: dict) -> bool:
    """Return True if any populated llm_* field has verdict==None.

    Records where citation was legitimately skipped (no citation provided) are
    NOT flagged: those carry skipped_no_citation=True with verdict=None by design.
    """
    for field in METRIC_FIELDS:
        m = record.get(field)
        if not isinstance(m, dict):
            continue
        # Legitimate skip (no citation) — not a failure
        if m.get("skipped_no_citation"):
            continue
        if m.get("verdict") is None:
            return True
    return False


def clean_file(path: Path) -> dict:
    """Drop records with any failed metric. Returns a stats dict."""
    if not path.exists():
        return {"file": path.name, "status": "missing"}

    # Backup once
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        backup_action = "created"
    else:
        backup_action = "kept-existing"

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        return {"file": path.name, "status": "not-a-list"}

    kept, dropped = [], []
    for r in records:
        if has_failed_metric(r):
            dropped.append(r)
        else:
            kept.append(r)

    # Distinct dedup keys among dropped, so user knows true uniques to re-run
    def dedup_key(r: dict) -> tuple:
        return (
            r.get("synopsis", ""),
            r.get("question", ""),
            r.get("summary_model", ""),
            r.get("statement", ""),
        )

    unique_dropped = {dedup_key(r) for r in dropped}

    with path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    return {
        "file": path.name,
        "status": "cleaned",
        "backup": backup_action,
        "total_before": len(records),
        "kept": len(kept),
        "dropped": len(dropped),
        "unique_dropped_statements": len(unique_dropped),
    }


def main() -> None:
    print(f"\nCleaning files in: {OUTPUT_DIR.resolve()}\n")
    print(f"{'File':<70} {'Before':>7} {'Kept':>7} {'Dropped':>9} {'Unique':>7}")
    print("-" * 105)

    grand_dropped = 0
    grand_unique = 0

    for name in OUTPUT_FILES:
        stats = clean_file(OUTPUT_DIR / name)

        if stats["status"] == "missing":
            print(f"{name:<70} {'(missing)':>7}")
            continue
        if stats["status"] == "not-a-list":
            print(f"{name:<70} {'(not a JSON list, skipped)':>7}")
            continue

        print(
            f"{stats['file']:<70} "
            f"{stats['total_before']:>7} "
            f"{stats['kept']:>7} "
            f"{stats['dropped']:>9} "
            f"{stats['unique_dropped_statements']:>7}"
        )
        grand_dropped += stats["dropped"]
        grand_unique += stats["unique_dropped_statements"]

    print("-" * 105)
    print(
        f"{'TOTAL':<70} {'':>7} {'':>7} "
        f"{grand_dropped:>9} {grand_unique:>7}"
    )
    print(
        "\nBackups saved as <file>.bak (only on first cleanup run).\n"
        "Re-run your evaluation script with the same output_file values to\n"
        "retry the dropped records."
    )


if __name__ == "__main__":
    main()
