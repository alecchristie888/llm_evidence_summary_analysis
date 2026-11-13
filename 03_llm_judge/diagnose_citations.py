"""
Diagnostic: investigate citation coverage across ALL summary-model folders
for the full-dataset run.

This script answers:

  1. Are citations being parsed correctly from the input JSON files?
     (i.e. is the empty `cited_action_ids` we see in some output records
     a real feature of the upstream pipeline, or a parsing bug?)

  2. Does citation coverage differ systematically between summary models?
     Some models may cite more aggressively than others — important context
     for interpreting the citation_correctness_all metric.

Run from the same directory you run evaluate_full_dataset.py from.

It does three checks:

  CHECK 1 — Inspect raw structure of one input cited_statements JSON file
            to confirm `citations` is the right field name to read.

  CHECK 2 — Per-statement: print sample rows showing each statement's
            `citations` field and any "Action XXX" mentions in the text.
            (In our pipeline the text never includes 'Action XXX' — they
            are stripped during cited-statement extraction. So mismatches
            here would be a red flag.)

  CHECK 3 — Coverage table: for every model_folder × synopsis-file,
            report total statements, statements with a non-empty citations
            field, and the citation-list length distribution.
"""

import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — set these to match your evaluate_full_dataset.py CONFIG
# ---------------------------------------------------------------------------
SUMMARIES_BASE_DIR = Path(
    "answerable_passed_qus_summaries/"
)
MODEL_FOLDERS = [
    "_claude-sonnet-4",
    "_gemini-2-5-pro",
    "_gpt-5",
    "fireworks_kimi-k2-0905",
]
# File used for deep-dive printout in CHECK 1 + CHECK 2
DEEP_DIVE_MODEL = "_claude-sonnet-4"
DEEP_DIVE_FILE = "bg_km_AmphibianConservation_cited_statements.json"

# Regex to find "Action 1234" or "Actions 1234, 5678" mentions in statement text
ACTION_RE = re.compile(r"Action[s]?\s+(\d+(?:\s*,\s*\d+)*)", re.IGNORECASE)


def extract_action_ids_from_text(text: str) -> list[str]:
    """Pull all 'Action 1234' style IDs out of a chunk of text."""
    ids: list[str] = []
    for m in ACTION_RE.finditer(text or ""):
        for piece in m.group(1).split(","):
            piece = piece.strip()
            if piece.isdigit():
                ids.append(piece)
    return ids


def show_keys(obj, prefix="", max_depth=4, depth=0):
    """Print the key structure of a nested dict/list to a limited depth."""
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            tname = type(v).__name__
            n_str = ""
            if isinstance(v, (list, dict)):
                n_str = f" (len={len(v)})"
            print(f"{prefix}{k}: {tname}{n_str}")
            if isinstance(v, dict):
                show_keys(v, prefix + "  ", max_depth, depth + 1)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"{prefix}  [list-of-dicts; first item keys:]")
                show_keys(v[0], prefix + "    ", max_depth, depth + 1)


# ---------------------------------------------------------------------------
# CHECK 1 — Structure of one cited_statements file
# ---------------------------------------------------------------------------

def check1_structure():
    print("=" * 78)
    print("CHECK 1 — Raw structure of one cited_statements file")
    print(f"  Model: {DEEP_DIVE_MODEL}")
    print(f"  File:  {DEEP_DIVE_FILE}")
    print("=" * 78)

    fpath = SUMMARIES_BASE_DIR / DEEP_DIVE_MODEL / "stmts_and_cited_stmts" / DEEP_DIVE_FILE
    if not fpath.exists():
        print(f"FILE NOT FOUND: {fpath}")
        return None

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level type: {type(data).__name__} (length {len(data)})")
    if not data:
        print("File is empty.")
        return data

    print("\nKeys of first entry:")
    show_keys(data[0])

    sd = data[0].get("summary_details", {})
    cs_list = sd.get("cited_statements", [])
    if cs_list:
        print(f"\nKeys of first cited_statement (out of {len(cs_list)} in entry 0):")
        show_keys(cs_list[0])
        print("\nFull dump of first cited_statement:")
        print(json.dumps(cs_list[0], indent=2)[:1500])
    else:
        print("No cited_statements found in summary_details of first entry!")

    return data


# ---------------------------------------------------------------------------
# CHECK 2 — Per-statement citations vs. text-mentioned action IDs
# ---------------------------------------------------------------------------

def check2_per_statement(data):
    print("\n" + "=" * 78)
    print("CHECK 2 — Per-statement: citations field vs. text mentions")
    print("=" * 78)

    if not data:
        print("Skipped — no data.")
        return

    n_total = 0
    n_with_citations = 0
    n_text_has_action = 0
    n_mismatch = 0  # text mentions Action(s) but citations field is empty

    for entry in data:
        sd = entry.get("summary_details", {})
        for cs in sd.get("cited_statements", []):
            n_total += 1
            stmt = cs.get("statement", "")
            cits = cs.get("citations") or []
            text_ids = extract_action_ids_from_text(stmt)

            if cits:
                n_with_citations += 1
            if text_ids:
                n_text_has_action += 1
            if text_ids and not cits:
                n_mismatch += 1

            if n_total <= 12:
                print(f"\n[{n_total}] statement: {stmt[:110]}")
                print(f"    citations field:      {cits}")
                print(f"    Action IDs in text:   {text_ids}")
                if cs.keys() - {"statement", "citations"}:
                    extra = list(cs.keys() - {"statement", "citations"})
                    print(f"    other keys on cs:     {extra}")

    print()
    print(f"  Total statements in this file:                   {n_total}")
    print(f"  ...with non-empty 'citations' field:             {n_with_citations} "
          f"({n_with_citations/max(n_total,1):.0%})")
    print(f"  ...whose statement text mentions 'Action XXX':   {n_text_has_action} "
          f"({n_text_has_action/max(n_total,1):.0%})")
    print(f"  ...mismatch (text has Action, citations empty):  {n_mismatch}")
    if n_mismatch > 0:
        print("  ⚠️  Mismatches detected — investigate parsing logic!")
    elif n_text_has_action == 0:
        print("  ✓ Text never includes 'Action XXX' (stripped during extraction). "
              "The 'citations' field is the ground truth.")


# ---------------------------------------------------------------------------
# CHECK 3 — Coverage per model across all files
# ---------------------------------------------------------------------------

def check3_coverage_per_model():
    print("\n" + "=" * 78)
    print("CHECK 3 — Citation coverage per summary model (all files)")
    print("=" * 78)

    rows = []
    cit_count_dist_per_model = {}
    file_breakdowns_per_model = {}
    summary_models_seen_per_folder = {}

    for model_folder in MODEL_FOLDERS:
        sac_dir = SUMMARIES_BASE_DIR / model_folder / "stmts_and_cited_stmts"
        if not sac_dir.exists():
            print(f"  Skipping {model_folder} — directory not found: {sac_dir}")
            continue

        n_files = 0
        n_total = 0
        n_with_cit = 0
        n_text_action = 0
        n_mismatch = 0
        files_with_zero_cited = 0
        cit_count_dist: Counter = Counter()
        file_breakdown = []
        summary_models_seen: Counter = Counter()

        for fpath in sorted(sac_dir.glob("*.json")):
            n_files += 1
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_total = 0
            file_with_cit = 0
            file_mismatch = 0
            for entry in data:
                sd = entry.get("summary_details", {})
                summary_models_seen[sd.get("summary_model", "unknown")] += 1
                for cs in sd.get("cited_statements", []):
                    n_total += 1
                    file_total += 1
                    cits = cs.get("citations") or []
                    cit_count_dist[len(cits)] += 1
                    text_ids = extract_action_ids_from_text(cs.get("statement", ""))
                    if cits:
                        n_with_cit += 1
                        file_with_cit += 1
                    if text_ids:
                        n_text_action += 1
                    if text_ids and not cits:
                        n_mismatch += 1
                        file_mismatch += 1

            if file_with_cit == 0 and file_total > 0:
                files_with_zero_cited += 1
            file_breakdown.append((fpath.name, file_total, file_with_cit, file_mismatch))

        coverage = n_with_cit / max(n_total, 1)
        rows.append({
            "model": model_folder,
            "files": n_files,
            "files_with_zero_cited": files_with_zero_cited,
            "statements": n_total,
            "with_citations": n_with_cit,
            "coverage": coverage,
            "text_mentions_action": n_text_action,
            "text_mentions_no_field": n_mismatch,
        })
        cit_count_dist_per_model[model_folder] = dict(cit_count_dist)
        file_breakdowns_per_model[model_folder] = file_breakdown
        summary_models_seen_per_folder[model_folder] = dict(summary_models_seen)

    # ---- Summary table ----
    print()
    print(f"  {'Folder':<28s}  {'Files':>6}  {'Stmts':>6}  {'WithCit':>8}  "
          f"{'CitRate':>7}  {'TxtMentions':>11}  {'Mismatch':>8}")
    print("  " + "-" * 86)
    for r in rows:
        print(
            f"  {r['model']:<28s}  {r['files']:>6}  {r['statements']:>6}  "
            f"{r['with_citations']:>8}  {r['coverage']:>6.1%}  "
            f"{r['text_mentions_action']:>11}  {r['text_mentions_no_field']:>8}"
        )

    # ---- Summary models seen inside each folder (sanity check) ----
    print()
    print("  summary_model field values seen inside each folder")
    print("  (each folder should contain exactly one summary_model):")
    for folder, models in summary_models_seen_per_folder.items():
        print(f"    {folder:<28s}  {models}")

    # ---- Citation list length distribution ----
    print()
    print("  Distribution of citation-list length per statement:")
    print(f"    {'folder':<28s}  {'length distribution (length: count)'}")
    for folder, dist in cit_count_dist_per_model.items():
        sorted_dist = dict(sorted(dist.items()))
        print(f"    {folder:<28s}  {sorted_dist}")

    # ---- Suspicious files (zero citations) ----
    print()
    suspicious = any(r["files_with_zero_cited"] for r in rows)
    if suspicious:
        print("  ⚠️  Files with ZERO statements having citations "
              "(possibly broken upstream extraction):")
        for r in rows:
            if r["files_with_zero_cited"]:
                print(f"    {r['model']:<28s}  "
                      f"{r['files_with_zero_cited']} / {r['files']} files")
                # Print which files
                for name, ft, fwc, fm in file_breakdowns_per_model[r["model"]]:
                    if ft > 0 and fwc == 0:
                        print(f"      - {name} ({ft} stmts, 0 cited)")
    else:
        print("  ✓ No files with zero citations across any model folder.")

    # ---- Mismatch summary ----
    print()
    total_mismatch = sum(r["text_mentions_no_field"] for r in rows)
    if total_mismatch > 0:
        print(f"  ⚠️  Total mismatches across all folders: {total_mismatch}")
        print("     (text mentions Action XXX but citations field is empty — "
              "investigate upstream extraction)")
    else:
        print("  ✓ Zero mismatches across all folders. The 'citations' field "
              "is the ground truth.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = check1_structure()
    check2_per_statement(data)
    check3_coverage_per_model()
    print("\nDone.")
