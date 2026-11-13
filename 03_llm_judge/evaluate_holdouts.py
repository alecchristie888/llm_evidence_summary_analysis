"""
Score one or more candidate prompts on the GEPA holdout sets (Option C).

Companion to optimise_faithfulness_prompt.py — uses the holdouts_<metric>.json
manifest written by that script to evaluate any prompt against the *exact*
Test-IID and Test-OOD statements that GEPA never saw during optimisation.

Why this script exists
----------------------
GEPA's val score is NOT an unbiased estimate of generalisation, because the
val set is used for Pareto candidate selection. To get a clean read on how
well an optimised prompt generalises:

  * Test-IID (random 15% from all reviewers) tells you whether the prompt
    overfits to val statements specifically. Healthy: Test-IID gain ≈ val gain.
  * Test-OOD (50% of Reviewer 3) tells you whether the prompt overfits to
    R1+R2 annotation style. Healthy: Test-OOD gain ≥ ~50% of val gain.

Usage
-----
1. Edit CONFIG below to point at your holdout manifest and the prompts you
   want to score (typically: the seed prompt and the optimised prompt).
2. Run:
       python evaluate_holdouts.py

Output
------
For each (prompt × split) combination:
  * Agreement rate with 95% Wilson confidence interval
  * Confusion matrix (TP / FP / TN / FN)
  * Per-reviewer breakdown (useful for the OOD set)

A side-by-side comparison table summarises seed vs. optimised on each split.
Detailed per-statement results are written to <output_dir>/holdout_scores.json.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Project root on path so we can import the same modules the optimiser uses
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the dataset builder and evaluator factory from the optimiser so the
# judge call is byte-identical to what GEPA used during optimisation.
from optimise_faithfulness_prompt import (  # noqa: E402
    REVIEWER_FILES,
    build_dataset,
    make_evaluator,
)


# ===========================================================================
# CONFIGURATION — edit these to match your run
# ===========================================================================

CONFIG = {
    # Which metric to evaluate. Must match the metric the holdouts were
    # built for — i.e. it must match the manifest filename.
    "metric": "citation_correctness",   # "faithfulness" | "relevance" | "citation_correctness"

    # Path to the holdout manifest written by optimise_faithfulness_prompt.py
    # The optimiser writes one manifest per split_seed:
    #   outputs/<run>/<metric>/holdouts_<metric>_seed<NNNN>.json
    # Test-IID and Test-OOD are identical across seeds (they're controlled
    # by holdout_seed, not split_seed) so either manifest works for scoring
    # the holdouts. Train/val membership IS seed-dependent — use the
    # matching manifest if you also score train/val.
    "holdouts_manifest": "outputs/gepa_v2.0_fullrun/citation_correctness/holdouts_citation_correctness_seed1234.json",

    # Prompts to compare. Each entry is (label, path-to-prompt-file).
    # With dual-seed runs we evaluate BOTH GEPA prompts on the holdouts so
    # we can spot seed-dependent overfit (large gap between seeds = unstable).
    # The seed prompt should be your v15 baseline (or whatever you started
    # GEPA from).
    "prompts": [
        ("seed",        "judge/seed_prompt_citation_correctness.txt"),
        ("gepa_v2.0_s1234",  "outputs/gepa_v2.0_fullrun/citation_correctness/optimised_citation_correctness_prompt_seed1234.txt"),
        ("gepa_v2.0_s5678",  "outputs/gepa_v2.0_fullrun/citation_correctness/optimised_citation_correctness_prompt_seed5678.txt"),
    ],

    # Which splits to score. Usually you want all of these so you can see
    # train/val/test_iid/test_ood side by side.
    "splits_to_score": ["test_iid", "test_ood"],

    # Same paths/judge config as optimise_faithfulness_prompt.py — keep in sync
    # to ensure the holdout judge call is identical to the optimisation one.
    "action_local_dir": "data/actions/background_key_messages/bg_km_all",
    "summaries_base_dir": (
        "answerable_passed_qus_summaries/answerable_passed_qus_summaries/"
        "hybrid_cross-encoder"
    ),
    "judge_model":       "google/gemini-3.1-pro-preview",
    "judge_provider":    None,
    "judge_temperature": 0.0,           # MUST match the optimiser

    # Where to save per-statement scores
    "output_path": "outputs/gepa_v2.0_fullrun/citation_correctness/holdout_scores.json",

    # If True (default), the saved JSON includes the full per-statement
    # `items` list (verdict, reasoning, cm_class, etc.) for downstream
    # analysis. Set False to keep only summary metrics + confusion matrices.
    "save_items": True,

    # Throttle (small delay between LLM calls; raise if you hit rate limits)
    "delay_seconds": 0.05,
}

# ===========================================================================


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(n_yes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = n_yes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(items: list[dict]) -> dict[str, int]:
    """Compute confusion matrix counts from per-statement score records."""
    cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for it in items:
        if it.get("error"):
            continue
        h = it["human_verdict"]
        l = it["llm_verdict"]
        if   h and     l:     cm["TP"] += 1
        elif (not h) and l:   cm["FP"] += 1
        elif (not h) and (not l): cm["TN"] += 1
        elif h and (not l):   cm["FN"] += 1
    return cm


def balanced_stats_from_cm(cm: dict[str, int]) -> dict[str, float | int | None]:
    """
    Derive sensitivity (recall on Yes), specificity (recall on No), balanced
    accuracy = 0.5 * (sens + spec), and a 95% CI on balanced accuracy.

    The CI is computed from the variance of the two independent sample
    proportions:
        Var(BA) = 0.25 * [ p_y(1-p_y)/n_pos + p_n(1-p_n)/n_neg ]
    where p_y = sensitivity, p_n = specificity. Falls back to None for the
    side that has zero examples (i.e. not estimable from this split).
    """
    n_pos = cm["TP"] + cm["FN"]      # human-Yes count
    n_neg = cm["TN"] + cm["FP"]      # human-No count
    sens = cm["TP"] / n_pos if n_pos else None
    spec = cm["TN"] / n_neg if n_neg else None

    if sens is None or spec is None:
        return {
            "n_pos": n_pos, "n_neg": n_neg,
            "sensitivity": sens, "specificity": spec,
            "bal_accuracy": None, "bal_ci_low": None, "bal_ci_high": None,
        }

    ba = 0.5 * (sens + spec)
    var = 0.25 * (sens * (1 - sens) / n_pos + spec * (1 - spec) / n_neg)
    half = 1.96 * math.sqrt(var)
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "sensitivity": sens, "specificity": spec,
        "bal_accuracy": ba,
        "bal_ci_low":  max(0.0, ba - half),
        "bal_ci_high": min(1.0, ba + half),
    }


# ---------------------------------------------------------------------------
# Hydrate manifest entries -> full examples (with action_docs_text etc.)
# ---------------------------------------------------------------------------

def _example_key(ex: dict) -> tuple[str, str, str]:
    return (ex["question"].strip(), ex["summary_model"].strip(), ex["statement"].strip())


def hydrate_split(
    manifest_split: list[dict],
    full_pool: list[dict],
    metric: str,
) -> list[dict]:
    """
    Match the lightweight manifest entries (just identifiers + label) against
    the full deduped pool produced by build_dataset(), so each holdout entry
    gains the heavy fields (all_action_docs_text or cited_action_text) needed
    by the evaluator.
    """
    by_key = {_example_key(ex): ex for ex in full_pool}
    label_key = f"human_{metric}"
    hydrated: list[dict] = []
    missing = 0
    for entry in manifest_split:
        k = (entry["question"].strip(), entry["summary_model"].strip(), entry["statement"].strip())
        full = by_key.get(k)
        if full is None:
            # Could happen if the action doc cache changed since the manifest
            # was written, or a citation_correctness statement lost its cite.
            missing += 1
            continue
        # Sanity check — manifest label should match the rebuilt label
        if full[label_key] != entry[label_key]:
            logger.warning(
                "Label mismatch for statement %r (manifest=%s, rebuilt=%s) — using rebuilt.",
                entry["statement"][:60], entry[label_key], full[label_key],
            )
        hydrated.append(full)
    if missing:
        logger.warning(
            "Could not hydrate %d/%d entries from manifest (missing from rebuild).",
            missing, len(manifest_split),
        )
    return hydrated


# ---------------------------------------------------------------------------
# Score one prompt on one split
# ---------------------------------------------------------------------------

def score_prompt_on_split(
    prompt_label: str,
    prompt_text: str,
    split_label: str,
    examples: list[dict],
    evaluator,
    delay: float,
) -> dict:
    """
    Run the evaluator (which already calls the judge with the candidate prompt
    injected) over each example, returning aggregate stats and per-statement
    detail records.
    """
    n = len(examples)
    logger.info("Scoring  prompt=%s  split=%s  n=%d", prompt_label, split_label, n)

    items: list[dict] = []
    n_correct = 0
    n_errors = 0
    for i, ex in enumerate(examples, 1):
        try:
            score, side = evaluator(prompt_text, ex)
        except Exception as e:  # belt-and-braces: evaluator is supposed to catch
            score, side = 0.0, {"Error": str(e)}

        err = side.get("Error")
        if err:
            n_errors += 1
            # Use the actual label key dynamically:
            human_field = next((k for k in ex if k.startswith("human_")), None)
            human_val = ex.get(human_field)
            items.append({
                "question":      ex["question"],
                "summary_model": ex["summary_model"],
                "statement":     ex["statement"],
                "reviewer":      ex.get("_reviewer"),
                "human_verdict": human_val,
                "llm_verdict":   None,
                "correct":       False,
                "cm_class":      "ERROR",
                "score":         0.0,
                "error":         err,
            })
        else:
            # Note: under balanced accuracy `score` is a class-weighted value
            # (e.g. 10.0 for a correctly classified No), not 0/1. Determine
            # "correct" from verdict equality, not score, so plain accuracy
            # remains comparable across runs with different scoring weights.
            human_yes = side["Human_verdict"] == "Yes"
            llm_yes   = side["LLM_verdict"]   == "Yes"
            is_correct = human_yes == llm_yes
            n_correct += int(is_correct)
            if   human_yes and llm_yes:        cm_class = "TP"
            elif (not human_yes) and llm_yes:  cm_class = "FP"
            elif (not human_yes) and (not llm_yes): cm_class = "TN"
            else:                              cm_class = "FN"
            items.append({
                "question":      ex["question"],
                "summary_model": ex["summary_model"],
                "statement":     ex["statement"],
                "reviewer":      ex.get("_reviewer"),
                "human_verdict": human_yes,
                "llm_verdict":   llm_yes,
                "error_type":    side.get("Error_type"),
                "reasoning":     side.get("Reasoning"),
                "correct":       is_correct,
                "cm_class":      cm_class,
                "score":         float(score),
                "error":         None,
            })

        if i % 25 == 0 or i == n:
            running = n_correct / max(1, i - n_errors) if (i - n_errors) > 0 else 0.0
            logger.info(
                "  [%d/%d] running agreement = %.3f  (errors so far: %d)",
                i, n, running, n_errors,
            )
        if delay > 0 and i < n:
            time.sleep(delay)

    n_scored = n - n_errors
    agreement = n_correct / n_scored if n_scored else 0.0
    lo, hi = wilson_ci(n_correct, n_scored)
    cm = confusion(items)
    bal = balanced_stats_from_cm(cm)

    return {
        "prompt":      prompt_label,
        "split":       split_label,
        "n_total":     n,
        "n_scored":    n_scored,
        "n_errors":    n_errors,
        "n_correct":   n_correct,
        "agreement":   agreement,
        "ci_low":      lo,
        "ci_high":     hi,
        "confusion":   cm,
        # Balanced accuracy + components (None when one class is absent).
        "n_pos":         bal["n_pos"],
        "n_neg":         bal["n_neg"],
        "sensitivity":   bal["sensitivity"],
        "specificity":   bal["specificity"],
        "bal_accuracy":  bal["bal_accuracy"],
        "bal_ci_low":    bal["bal_ci_low"],
        "bal_ci_high":   bal["bal_ci_high"],
        "items":       items,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_pct(p: float) -> str:
    return f"{p*100:5.1f}%"


def _fmt_pct_or_dash(p: float | None) -> str:
    return _fmt_pct(p) if p is not None else "   n/a"


def print_per_split_summary(result: dict) -> None:
    cm = result["confusion"]
    p  = result["agreement"]
    ba = result["bal_accuracy"]
    sens = result["sensitivity"]
    spec = result["specificity"]
    print(
        f"  {result['prompt']:<14}  acc={_fmt_pct(p)}  "
        f"95% CI=[{_fmt_pct(result['ci_low'])}, {_fmt_pct(result['ci_high'])}]  "
        f"TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}"
        + (f"  errors={result['n_errors']}" if result["n_errors"] else "")
    )
    if ba is not None:
        print(
            f"  {'':<14}  bal_acc={_fmt_pct(ba)}  "
            f"95% CI=[{_fmt_pct(result['bal_ci_low'])}, {_fmt_pct(result['bal_ci_high'])}]  "
            f"sens={_fmt_pct(sens)} (n={result['n_pos']})  "
            f"spec={_fmt_pct(spec)} (n={result['n_neg']})"
        )
    else:
        # One class absent — balanced accuracy not estimable on this split.
        print(
            f"  {'':<14}  bal_acc=  n/a  (n_pos={result['n_pos']}, n_neg={result['n_neg']})"
            f"  sens={_fmt_pct_or_dash(sens)}  spec={_fmt_pct_or_dash(spec)}"
        )


def print_per_reviewer_breakdown(result: dict) -> None:
    """For OOD diagnosis — show agreement broken down by source reviewer."""
    by_rev = defaultdict(
        lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0,
                 "correct": 0, "n": 0, "yes_h": 0, "yes_l": 0}
    )
    for it in result["items"]:
        if it.get("error"):
            continue
        d = by_rev[it.get("reviewer", "?")]
        d["n"] += 1
        d["correct"] += 1 if it["correct"] else 0
        d["yes_h"] += 1 if it["human_verdict"] else 0
        d["yes_l"] += 1 if it["llm_verdict"] else 0
        h, l = it["human_verdict"], it["llm_verdict"]
        if   h and     l:        d["TP"] += 1
        elif (not h) and l:      d["FP"] += 1
        elif (not h) and (not l):d["TN"] += 1
        elif h and (not l):      d["FN"] += 1
    if len(by_rev) > 1:
        for rev in sorted(by_rev):
            d = by_rev[rev]
            agree = d["correct"] / d["n"] if d["n"] else 0.0
            bal = balanced_stats_from_cm(d)
            ba_str = (
                f"  bal_acc={_fmt_pct(bal['bal_accuracy'])}"
                if bal["bal_accuracy"] is not None
                else "  bal_acc=  n/a"
            )
            print(
                f"      {rev}: n={d['n']:>3}  acc={_fmt_pct(agree)}{ba_str}  "
                f"human_yes={d['yes_h']}  llm_yes={d['yes_l']}"
            )


def _comparison_panel(
    title: str,
    metric_key: str,            # "agreement" or "bal_accuracy"
    n_keys: tuple[str, str] | None,  # (numerator_key, denom_key) for plain acc, else None
    results: list[dict],
    splits: list[str],
    prompts_in_order: list[str],
    idx: dict,
) -> None:
    print(title)
    header = f"{'Prompt':<16}" + "".join(f"{s:>22}" for s in splits)
    print(header)
    print("-" * len(header))
    for p in prompts_in_order:
        cells = [f"{p:<16}"]
        for s in splits:
            r = idx.get((p, s))
            if r is None or r.get(metric_key) is None:
                cells.append(f"{'—':>22}")
            else:
                if n_keys is not None:
                    num = r.get(n_keys[0])
                    den = r.get(n_keys[1])
                    cells.append(f"{_fmt_pct(r[metric_key]):>10}  [{num}/{den}]")
                else:
                    # bal_accuracy: show sens/spec component counts instead
                    cells.append(
                        f"{_fmt_pct(r[metric_key]):>10}  "
                        f"[{r.get('n_pos', 0)}+/{r.get('n_neg', 0)}-]"
                    )
        print("".join(cells))

    if len(prompts_in_order) >= 2:
        baseline = prompts_in_order[0]
        print("-" * len(header))
        print(f"{'Δ vs ' + baseline:<16}", end="")
        for s in splits:
            base = idx.get((baseline, s))
            best = idx.get((prompts_in_order[-1], s))
            if base is None or best is None or base.get(metric_key) is None or best.get(metric_key) is None:
                print(f"{'—':>22}", end="")
            else:
                delta = best[metric_key] - base[metric_key]
                arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
                print(f"{arrow + ' ' + (f'{delta*100:+.1f}pp'):>22}", end="")
        print()


def print_comparison_table(results: list[dict], splits: list[str]) -> None:
    """
    Side-by-side comparison table with TWO panels:
      1) Plain accuracy (agreement rate with human labels)
      2) Balanced accuracy (0.5*sensitivity + 0.5*specificity)

    Reports both because GEPA optimises balanced accuracy (where enabled)
    but readers also want to know the plain agreement rate.
    """
    prompts_in_order = []
    for r in results:
        if r["prompt"] not in prompts_in_order:
            prompts_in_order.append(r["prompt"])
    idx = {(r["prompt"], r["split"]): r for r in results}

    print()
    print("=" * 80)
    print("HOLDOUT COMPARISON — agreement with human labels")
    print("=" * 80)
    _comparison_panel(
        "\n[Plain accuracy]  cell = pct  [n_correct / n_scored]\n",
        metric_key="agreement",
        n_keys=("n_correct", "n_scored"),
        results=results, splits=splits,
        prompts_in_order=prompts_in_order, idx=idx,
    )
    _comparison_panel(
        "\n[Balanced accuracy]  cell = pct  [n_pos+ / n_neg-]   "
        "(equal weight to Yes recall and No recall)\n",
        metric_key="bal_accuracy",
        n_keys=None,
        results=results, splits=splits,
        prompts_in_order=prompts_in_order, idx=idx,
    )
    print("=" * 80)


def _gain(opt: dict, base: dict, key: str) -> float | None:
    if opt is None or base is None:
        return None
    if opt.get(key) is None or base.get(key) is None:
        return None
    return opt[key] - base[key]


def print_generalisation_diagnosis(results: list[dict]) -> None:
    """
    The headline check: does the optimised prompt generalise?

      * If Test-IID gain ≈ val gain → no IID overfitting
      * If Test-OOD gain ≥ ~50% of Test-IID gain → no reviewer-style overfitting

    Reports gains for BOTH plain accuracy and balanced accuracy. The verdict
    is driven by balanced accuracy when available (since that's what GEPA
    optimised under the new config); falls back to plain accuracy otherwise.
    """
    idx = {(r["prompt"], r["split"]): r for r in results}
    prompts = []
    for r in results:
        if r["prompt"] not in prompts:
            prompts.append(r["prompt"])
    if len(prompts) < 2:
        return
    baseline, optimised = prompts[0], prompts[-1]

    iid_opt, iid_base = idx.get((optimised, "test_iid")), idx.get((baseline, "test_iid"))
    ood_opt, ood_base = idx.get((optimised, "test_ood")), idx.get((baseline, "test_ood"))
    if not all([iid_opt, iid_base, ood_opt, ood_base]):
        return

    iid_acc = _gain(iid_opt, iid_base, "agreement")
    ood_acc = _gain(ood_opt, ood_base, "agreement")
    iid_bal = _gain(iid_opt, iid_base, "bal_accuracy")
    ood_bal = _gain(ood_opt, ood_base, "bal_accuracy")

    print()
    print("Generalisation diagnosis")
    print("-" * 80)
    print(f"  Comparing optimised={optimised!r}  vs  baseline={baseline!r}")
    print()
    print(f"  Plain accuracy gain   —  Test-IID: {iid_acc*100:+.1f}pp   Test-OOD: {ood_acc*100:+.1f}pp")
    if iid_bal is not None and ood_bal is not None:
        print(f"  Balanced acc gain     —  Test-IID: {iid_bal*100:+.1f}pp   Test-OOD: {ood_bal*100:+.1f}pp")
        primary_iid, primary_ood, primary_label = iid_bal, ood_bal, "balanced accuracy"
    else:
        print("  Balanced acc gain     —  not estimable on one or both splits (single class).")
        primary_iid, primary_ood, primary_label = iid_acc, ood_acc, "plain accuracy"

    print()
    print(f"  Verdict driven by {primary_label}:")
    if primary_iid <= 0:
        verdict = "❌ No IID gain — GEPA's val improvement did not transfer at all."
    elif primary_ood < 0:
        verdict = "❌ Negative OOD gain despite positive IID gain — clear style overfit."
    elif primary_ood < 0.5 * primary_iid and primary_iid > 0.01:
        verdict = ("⚠  OOD gain < 50% of IID gain — prompt likely overfits to "
                   "Reviewer 1+2 annotation style.")
    else:
        verdict = "✅ Both Test-IID and Test-OOD improved — healthy generalisation."
    print(f"    {verdict}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG) -> None:
    _setup_logging()

    metric = cfg["metric"]
    manifest_path = Path(cfg["holdouts_manifest"])
    if not manifest_path.exists():
        logger.error("Holdout manifest not found: %s", manifest_path)
        logger.error("Run optimise_faithfulness_prompt.py first to create it.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["metric"] != metric:
        logger.error(
            "Metric mismatch: CONFIG asks for %r but manifest is for %r.",
            metric, manifest["metric"],
        )
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("HOLDOUT EVALUATION  (metric=%s)", metric.upper())
    logger.info("Manifest:        %s", manifest_path)
    logger.info("Splits to score: %s", cfg["splits_to_score"])
    logger.info("Prompts:         %s", [p[0] for p in cfg["prompts"]])
    logger.info("Judge:           %s  (T=%.2f)", cfg["judge_model"], cfg["judge_temperature"])
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Rebuild the full pool so we can hydrate manifest entries with action
    # document text. This call goes through the SAME dedup logic as the
    # optimiser, so identifiers will line up exactly.
    # ------------------------------------------------------------------
    from data_loaders.action_loader import ActionLoader
    from data_loaders.summaries_loader import SummariesIndex

    logger.info("Loading summaries index + action loader...")
    summaries_index = SummariesIndex(cfg["summaries_base_dir"])
    action_loader = ActionLoader(
        local_base_dir=cfg.get("action_local_dir"),
        fetch_from_web=False,
    )

    logger.info("Rebuilding deduped pool (for action-doc hydration)...")
    full_pool = build_dataset(
        REVIEWER_FILES,
        summaries_index=summaries_index,
        action_loader=action_loader,
        metric=metric,
    )

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    prompt_texts: list[tuple[str, str]] = []
    for label, path in cfg["prompts"]:
        ppath = Path(path)
        if not ppath.exists():
            logger.error("Prompt file not found for %r: %s", label, ppath)
            sys.exit(1)
        prompt_texts.append((label, ppath.read_text(encoding="utf-8")))
        logger.info("Loaded prompt %-14s  (%d chars)  from %s",
                    label, len(prompt_texts[-1][1]), ppath)

    # ------------------------------------------------------------------
    # Build evaluator (same factory the optimiser used)
    # ------------------------------------------------------------------
    evaluator = make_evaluator(
        cfg["judge_model"],
        cfg.get("judge_provider"),
        judge_temperature=cfg["judge_temperature"],
        metric=metric,
    )

    # ------------------------------------------------------------------
    # Score every (prompt × split)
    # ------------------------------------------------------------------
    results: list[dict] = []
    for split_name in cfg["splits_to_score"]:
        if split_name not in manifest["splits"]:
            logger.warning("Split %r not in manifest — skipping.", split_name)
            continue
        examples = hydrate_split(manifest["splits"][split_name], full_pool, metric)
        if not examples:
            logger.warning("No hydrated examples for split %r — skipping.", split_name)
            continue
        for label, text in prompt_texts:
            r = score_prompt_on_split(
                label, text, split_name, examples, evaluator,
                delay=cfg.get("delay_seconds", 0.0),
            )
            results.append(r)
            print()
            print(f"--- {split_name} ---")
            print_per_split_summary(r)
            if split_name == "test_ood" or len({it.get("reviewer") for it in r["items"]}) > 1:
                print_per_reviewer_breakdown(r)

    # ------------------------------------------------------------------
    # Comparison + verdict
    # ------------------------------------------------------------------
    print_comparison_table(results, cfg["splits_to_score"])
    print_generalisation_diagnosis(results)

    # ------------------------------------------------------------------
    # Persist detailed results (per-statement scores included)
    # ------------------------------------------------------------------
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flat summary table for quick downstream loading:
    #   pandas.DataFrame(payload["summary_table"])
    summary_table = []
    for r in results:
        cm = r["confusion"]
        summary_table.append({
            "prompt":        r["prompt"],
            "split":         r["split"],
            "n_total":       r["n_total"],
            "n_scored":      r["n_scored"],
            "n_errors":      r["n_errors"],
            "n_correct":     r["n_correct"],
            "agreement":     r["agreement"],
            "acc_ci_low":    r["ci_low"],
            "acc_ci_high":   r["ci_high"],
            "bal_accuracy":  r["bal_accuracy"],
            "bal_ci_low":    r["bal_ci_low"],
            "bal_ci_high":   r["bal_ci_high"],
            "sensitivity":   r["sensitivity"],
            "specificity":   r["specificity"],
            "n_pos":         r["n_pos"],
            "n_neg":         r["n_neg"],
            "TP":            cm["TP"],
            "FP":            cm["FP"],
            "TN":            cm["TN"],
            "FN":            cm["FN"],
        })

    save_items = cfg.get("save_items", True)
    serialised_results = []
    for r in results:
        if save_items:
            serialised_results.append(r)
        else:
            r_no_items = {k: v for k, v in r.items() if k != "items"}
            serialised_results.append(r_no_items)

    payload = {
        "metric":             metric,
        "manifest":           str(manifest_path),
        "judge_model":        cfg["judge_model"],
        "judge_temperature":  cfg["judge_temperature"],
        "summary_table":      summary_table,
        "results":            serialised_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if save_items:
        print(f"\nDetailed per-statement scores saved to: {out_path}")
    else:
        print(f"\nSummary metrics saved to: {out_path}  (per-item details omitted)")


if __name__ == "__main__":
    main()

