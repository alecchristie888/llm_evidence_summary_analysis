#!/usr/bin/env python3
"""
build_failure_reasons.py
------------------------
Reproducibly flatten per-claim faithfulness failure tags from the LLM-judge
output JSONs into a tidy CSV.

Reads every <synopsis>_llm_scores.json under FULL_RESULTS_DIR and emits two
files in OUTPUT_DIR:

  failure_reasons.csv          one row per (claim, failure_tag)
                               cols: summary_model, synopsis, question,
                                     statement_idx, claim_idx, claim_text,
                                     support_assessment, confidence,
                                     failure_reason

  claims_long.csv              one row per claim (including supported ones),
                               cols: summary_model, synopsis, question,
                                     statement_idx, claim_idx, claim_text,
                                     support_assessment, confidence,
                                     n_failure_reasons, failure_reasons_joined

The two files let you compute either:
  - failures per tag         (group failure_reasons.csv by failure_reason)
  - failures per claim       (filter claims_long.csv to support_assessment ==
                              'not_fully_supported')
  - failures per statement   (statement_idx groups inside claims_long.csv)
  - failures per summary     (summary_model + synopsis + question)

The judge prompt that produced the tags is `manual_faithfulness_v14-concat`,
which constrains the vocabulary to:
  - absent_or_contradicted_fact
  - overgeneralisation
  - wrong_quantity
  - unsupported_comparative_claim
  - unsupported_evidence_summary
  - wrong_effect_direction_or_magnitude
  - mismerged_studies

Usage:
    python build_failure_reasons.py \
        --results-dir full_results/gemini-3.1-pro-preview-finalprompts \
        --output-dir  analysis_outputs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from typing import Any


def iter_judge_files(results_dir: str):
    pattern = os.path.join(results_dir, "*", "*_llm_scores.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # Allow callers to point straight at a flat dir of JSONs too.
        files = sorted(glob.glob(os.path.join(results_dir, "*_llm_scores.json")))
    return files


def safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def build(results_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = iter_judge_files(results_dir)
    if not files:
        sys.exit(f"No judge JSONs found under {results_dir}")

    fr_path = os.path.join(output_dir, "failure_reasons.csv")
    cl_path = os.path.join(output_dir, "claims_long.csv")

    fr_cols = [
        "summary_model", "synopsis", "question",
        "statement_idx", "claim_idx", "claim_text",
        "support_assessment", "confidence", "failure_reason",
    ]
    cl_cols = [
        "summary_model", "synopsis", "question",
        "statement_idx", "claim_idx", "claim_text",
        "support_assessment", "confidence",
        "n_failure_reasons", "failure_reasons_joined",
    ]

    n_records = 0
    n_claims = 0
    n_claims_with_tags = 0
    n_tag_rows = 0
    tag_counts: Counter = Counter()
    model_counts: Counter = Counter()

    with open(fr_path, "w", newline="", encoding="utf-8") as fr_f, \
         open(cl_path, "w", newline="", encoding="utf-8") as cl_f:
        fr_w = csv.DictWriter(fr_f, fieldnames=fr_cols)
        cl_w = csv.DictWriter(cl_f, fieldnames=cl_cols)
        fr_w.writeheader()
        cl_w.writeheader()

        # Track statement indices per (model, synopsis, question) so multiple
        # statements within one summary get distinct ids.
        stmt_counter: dict[tuple, int] = {}

        for path in files:
            try:
                recs = json.load(open(path, encoding="utf-8"))
            except Exception as e:
                print(f"WARN: could not parse {path}: {e}", file=sys.stderr)
                continue
            if not isinstance(recs, list):
                continue

            for rec in recs:
                n_records += 1
                model = rec.get("summary_model")
                synopsis = rec.get("synopsis")
                question = rec.get("question")
                key = (model, synopsis, question)
                stmt_counter[key] = stmt_counter.get(key, -1) + 1
                statement_idx = stmt_counter[key]

                fobj = rec.get("llm_faithfulness") or {}
                confidence = safe_get(fobj, "confidence")
                claims = fobj.get("claims") or []

                for ci, claim in enumerate(claims):
                    n_claims += 1
                    support = safe_get(claim, "support_assessment")
                    tags = claim.get("failure_reasons") or []
                    # Normalise: drop empties, strip whitespace
                    tags = [str(t).strip() for t in tags if str(t).strip()]
                    if tags:
                        n_claims_with_tags += 1

                    cl_w.writerow({
                        "summary_model": model,
                        "synopsis": synopsis,
                        "question": question,
                        "statement_idx": statement_idx,
                        "claim_idx": ci,
                        "claim_text": safe_get(claim, "claim_text"),
                        "support_assessment": support,
                        "confidence": confidence,
                        "n_failure_reasons": len(tags),
                        "failure_reasons_joined": ";".join(tags),
                    })

                    for tag in tags:
                        n_tag_rows += 1
                        tag_counts[tag] += 1
                        model_counts[model] += 1
                        fr_w.writerow({
                            "summary_model": model,
                            "synopsis": synopsis,
                            "question": question,
                            "statement_idx": statement_idx,
                            "claim_idx": ci,
                            "claim_text": safe_get(claim, "claim_text"),
                            "support_assessment": support,
                            "confidence": confidence,
                            "failure_reason": tag,
                        })

    print(f"Wrote {fr_path}")
    print(f"Wrote {cl_path}")
    print()
    print(f"Records scanned:           {n_records:,}")
    print(f"Claims:                    {n_claims:,}")
    print(f"Claims with >=1 tag:       {n_claims_with_tags:,}")
    print(f"Failure-reason rows:       {n_tag_rows:,}")
    print()
    print("Top failure tags overall:")
    for tag, n in tag_counts.most_common():
        print(f"  {tag:42s} {n:>6,}")
    print()
    print("Tag rows per model:")
    for m, n in model_counts.most_common():
        print(f"  {m:35s} {n:>6,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir",
                   default="full_results/gemini-3.1-pro-preview-finalprompts",
                   help="Directory containing per-model subfolders of "
                        "<synopsis>_llm_scores.json (default: %(default)s)")
    p.add_argument("--output-dir", default="analysis_outputs",
                   help="Directory to write CSVs into (default: %(default)s)")
    args = p.parse_args()
    build(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
