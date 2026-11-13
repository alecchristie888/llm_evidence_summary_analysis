"""
LLM-as-Judge evaluation runner for all three reviewer Excel files.

This script supersedes ``evaluate_human_eval_set.py`` (which only covered the
46-statement reviewer3-extras file) and is the primary entry point for running
the LLM judge over the full human evaluation dataset.

Usage
-----
Configure the settings in ``CONFIG`` below, then run::

    python evaluate_human_eval_all_reviewers.py

What it does
------------
1. Loads answerable-only records from all three reviewer Excel files.
2. Deduplicates: the 57 statements that appear in two reviewer files
   (second-reviewer overlap) are evaluated *once* and the result is stored
   against the first file that contained the statement.  Both copies of the
   record in the final output therefore have identical LLM scores.
   (Set ``dedup_overlaps=True`` in CONFIG to enable; False keeps both copies.)
3. For each unique record, runs up to three LLM-as-Judge evaluations:
     a. Relevance        – question + statement only
     b. Citation Correctness – statement + cited action doc (or auto-No)
     c. Faithfulness     – statement + all relevant action docs
4. Saves results incrementally to a JSON file after every statement.
5. Resumes automatically from where it left off if interrupted.
6. Prints a summary of scores at the end.

Deduplication strategy
----------------------
A *dedup key* is ``synopsis|question|summary_model|statement`` (same as the
resumability key).  On first encounter a record is evaluated and its result
cached.  Any later copy of the same statement (from a different reviewer file)
receives the same LLM scores but retains its own ``reviewer``, ``second_reviewer``,
and ``source_file`` metadata.

Results file structure (one JSON object per record)
----------------------------------------------------
{
  "synopsis": "Bat Conservation",
  "answerability": "answerable",
  "question": "...",
  "summary": "...",
  "statement": "...",
  "summary_model": "anthropic/claude-sonnet-4",
  "summary_action_ids": ["2013", ...],
  "ce_action_ids": ["2013", "961", ...],
  "cited_action_id": "2013",
  "human_relevance": null,
  "human_citation_correctness": null,
  "human_faithfulness": null,
  "reviewer": "Reviewer_1",
  "second_reviewer": null,
  "source_file": "sampled_subset_reviewer1_all.xlsx",
  "llm_relevance": {
      "metric_name": "relevance",
      "verdict": true,
      "reasoning": "...",
      "raw_verdict": "Yes"
  },
  "llm_citation_correctness": { ... },
  "llm_faithfulness": { ... },
  "judge_model": "google/gemini-2.5-pro",
  "judge_provider": null
}
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is on the Python path when running as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders.action_loader import ActionLoader
from data_loaders.human_eval_loader import EvaluationRecord, load_human_eval_records
from judge import metrics as judge_metrics
from judge.metrics import faithfulness_majority_vote, faithfulness_per_document
from utils.io import build_done_set, load_results, results_as_unique_key, save_results


# ===========================================================================
# CONFIGURATION — edit these to match your setup
# ===========================================================================

CONFIG = {
    # ---------------- Reviewer Excel files ----------------
    # Paths to all three human-evaluation Excel files.
    # Each path may be absolute or relative to the working directory.
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

    # Filter to answerable questions only (recommended — unanswerable rows
    # crept into the subset by mistake and should be excluded).
    "answerable_only": True,

    # Deduplicate the 57 statements that appear in two reviewer files.
    # True  → evaluate each unique statement once; the second copy receives
    #          the same LLM scores as the first but keeps its own reviewer
    #          metadata.
    # False → evaluate every row independently, even duplicates.
    "dedup_overlaps": True,

    # ---------------- Paths ----------------
    # Directory containing action_{id}_clean.txt files.
    # Set to None to disable local loading (and enable web fallback instead).
    "action_local_dir": "data/actions/background_key_messages/bg_km_all",

    # Set to True to fetch action text from the CE website if a local file
    # is missing.
    "action_fetch_from_web": False,

    # Where to save LLM judge results (will be created if needed).
    "output_file": "llm_judge_results/human_eval_all_reviewers_llm_scores_gemini-3.1-pro-preview-finalprompts.json",

    # ---------------- Judge model ----------------
    # Any OpenRouter model identifier.
    # See https://openrouter.ai/models for available models.
    "judge_model": "google/gemini-3.1-pro-preview",

    # Optional provider pin. Set to None for OpenRouter's default routing.
    # Examples: "Google", "Anthropic", "OpenAI"
    "judge_provider": "Google",

    # ---------------- Run control ----------------
    # Maximum number of *new* statements to evaluate in this run.
    # Set to None to evaluate all remaining statements.  Useful for testing
    # (e.g. set to 10 to do a quick smoke test before a full run).
    "max_statements": None,

    # Random seed for shuffling. Set to an integer for reproducible ordering
    # (recommended when comparing prompt versions on the same sample).
    # Set to None for a different random order each run.
    "random_seed": 42,

    # Stratified sampling for test runs.
    # When max_statements is set, this ensures the sample is balanced across
    # summary models and high-error synopsis topics rather than purely random.
    # True  → sample proportionally from each (summary_model, synopsis) stratum.
    # False → simple random shuffle (original behaviour).
    "stratified_sample": True,

    # High-priority synopsis topics to oversample when stratified_sample=True.
    # These topics have historically had the most errors and benefit from
    # guaranteed representation in test runs.
    "priority_synopses": [
        "Marine Fish Conservation",
        "Soil Fertility",
        "Bee Conservation",
        "Mediterranean Farmland",
        "Bat Conservation",
        "Natural Pest Control",
    ],

    # Seconds to wait between statements (avoids rate-limit bursts).
    "delay_between_statements": 0.1,

    # Maximum LLM retry attempts per metric call.
    "max_retries": 3,

    # Temperature for all single-call LLM evaluations (relevance, CC, faithfulness).
    # 0.0 = fully deterministic (recommended for reproducibility).
    # Increase (e.g. 0.3) to introduce variation, but note this may reduce consistency.
    # The majority vote temperature (faithfulness_vote_temperature) overrides this
    # for majority vote runs.
    "judge_temperature": 0.0,

    # ---------------- Custom (e.g. GEPA-optimised) prompt templates ----------------
    # Optional paths to .txt files containing alternative prompt templates for
    # any of the three judge metrics. When set to None (default), the built-in
    # prompts in judge.prompts are used. When set to a path, the file is read
    # at the start of run_evaluation() and passed to the corresponding metric
    # via its ``prompt_template`` parameter.
    #
    # Required placeholders per metric (substituted with str.format_map; any
    # unrecognised ``{key}`` placeholders are left intact):
    #   relevance               : {question}, {statement}
    #   citation_correctness    : {statement}, {cited_action_id}, {cited_action_text}
    #   faithfulness            : {statement}, {all_action_docs_text}
    #
    # NOTE: when comparing prompt versions, also change ``output_file`` so the
    # dedup cache from previous runs does not contaminate the new results.
    "relevance_prompt_file":            "prompt_files/relevance/manual_relevance_v7.txt",
    "citation_correctness_prompt_file": "prompt_files/citation_correctness/manual_citation_correctness_v17.txt",
    "faithfulness_prompt_file":         "prompt_files/faithfulness/manual_faithfulness_v14-concat.txt",

    # Which metrics to run. Subset of:
    #   ["relevance", "citation_correctness", "faithfulness"]
    "metrics_to_run": ["relevance", "citation_correctness", "faithfulness"],

    # Per-document faithfulness mode.
    # True  → evaluate each action document independently (one LLM call per doc);
    #          overall verdict = Yes if ANY single document supports the statement.
    #          Results stored in llm_faithfulness_per_doc; llm_faithfulness is also
    #          populated with the aggregated verdict for backwards compatibility.
    # False → original behaviour: concatenate all docs into one prompt (single call).
    "faithfulness_per_document": False,

    # Majority vote faithfulness mode.
    # True  → run faithfulness N times at higher temperature and take majority vote.
    #          Results stored in llm_faithfulness_majority_vote; llm_faithfulness is
    #          also populated with the majority verdict for backwards compatibility.
    #          Cannot be used together with faithfulness_per_document=True.
    # False → original single-call behaviour.
    "faithfulness_majority_vote": False,

    # Number of votes for majority voting (odd numbers avoid ties).
    "faithfulness_n_votes": 3,

    # Temperature for majority vote runs (0.0 = deterministic, higher = more variation).
    # Recommended range: 0.3–0.6. Too high risks incoherent outputs.
    "faithfulness_vote_temperature": 0.3,
}

# ===========================================================================


def _stratified_sample(
    records: list,
    n: int,
    priority_synopses: list[str],
    seed: int | None = 42,
) -> list:
    """
    Draw a stratified sample of n records from the full list.

    Strategy
    --------
    1. Reserve half the budget (n//2) for priority synopsis topics — these
       are the high-error topics identified from previous runs. Sample
       proportionally within each (synopsis, summary_model) stratum.
    2. Fill the remaining budget from non-priority topics, again
       proportionally across (synopsis, summary_model) strata.
    3. If a stratum has fewer records than its allocation, take all of them
       and redistribute the remainder.

    The seed ensures the same sample is drawn on every run with the same
    seed, allowing fair comparison between prompt versions.
    """
    rng = random.Random(seed)

    # Separate into priority and non-priority
    priority = [r for r in records if r.synopsis in priority_synopses]
    other    = [r for r in records if r.synopsis not in priority_synopses]

    n_priority = min(n // 2, len(priority))
    n_other    = min(n - n_priority, len(other))

    def _proportional_sample(pool, k):
        """Sample k records proportionally from (synopsis, model) strata."""
        if k <= 0 or not pool:
            return []
        # Group by stratum
        strata: dict[tuple, list] = {}
        for r in pool:
            key = (r.synopsis, r.summary_model)
            strata.setdefault(key, []).append(r)

        # Allocate slots proportionally using floor allocation + top-up
        total = len(pool)
        keys = list(strata.keys())
        # Floor allocation
        allocations = {key: max(1, int(len(strata[key]) / total * k)) for key in keys}
        allocated = sum(allocations.values())
        # Top up to reach k by adding 1 to strata with largest fractional remainder
        if allocated < k:
            fracs = sorted(
                keys,
                key=lambda key: (len(strata[key]) / total * k) - allocations[key],
                reverse=True,
            )
            for key in fracs[:k - allocated]:
                allocations[key] += 1
        # Trim if over (can happen when k < num strata due to max(1,...) floor)
        allocated = sum(allocations.values())
        if allocated > k:
            by_size = sorted(keys, key=lambda key: allocations[key], reverse=True)
            for key in by_size:
                if allocated <= k:
                    break
                if allocations[key] > 1:
                    allocations[key] -= 1
                    allocated -= 1

        # Sample from each stratum
        sampled = []
        for key, members in strata.items():
            alloc = min(allocations.get(key, 0), len(members))
            if alloc > 0:
                sampled.extend(rng.sample(members, alloc))

        # Top up from leftover if short due to small strata
        sampled_ids = {id(r) for r in sampled}
        leftover = [r for r in pool if id(r) not in sampled_ids]
        shortfall = k - len(sampled)
        if shortfall > 0 and leftover:
            sampled.extend(rng.sample(leftover, min(shortfall, len(leftover))))

        return sampled

    sampled_priority = _proportional_sample(priority, n_priority)
    sampled_other    = _proportional_sample(other, n_other)
    result = sampled_priority + sampled_other
    rng.shuffle(result)  # mix priority and non-priority together
    return result


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation logic (shared with evaluate_human_eval_set.py)
# ---------------------------------------------------------------------------

def evaluate_record(
    record: EvaluationRecord,
    action_loader: ActionLoader,
    judge_model: str,
    judge_provider: Optional[str],
    metrics_to_run: list[str],
    max_retries: int,
    config: dict = None,
    custom_prompts: dict | None = None,
) -> dict:
    """
    Run all requested metric evaluations for a single statement record.

    Returns the record serialised as a dict, augmented with llm_* fields
    and ``judge_model`` / ``judge_provider``.

    Parameters
    ----------
    custom_prompts : dict | None
        Optional mapping of metric name → custom prompt template string.
        Recognised keys: ``"relevance"``, ``"citation_correctness"``,
        ``"faithfulness"``. Any metric with a non-None entry will use the
        provided template instead of the built-in prompt.
    """
    result = record.to_dict()
    result["judge_model"] = judge_model
    result["judge_provider"] = judge_provider

    statement = record.statement
    question = record.question
    judge_temperature = (config or {}).get("judge_temperature", 0.0)
    custom_prompts = custom_prompts or {}
    relevance_template = custom_prompts.get("relevance")
    citation_template = custom_prompts.get("citation_correctness")
    faithfulness_template = custom_prompts.get("faithfulness")

    # ---- 1. Relevance ----
    if "relevance" in metrics_to_run:
        logger.info("  [relevance] '%s...'", statement[:60])
        rel = judge_metrics.relevance(
            question=question,
            statement=statement,
            model=judge_model,
            provider=judge_provider,
            max_retries=max_retries,
            temperature=judge_temperature,
            prompt_template=relevance_template,
        )
        result["llm_relevance"] = rel
        logger.info("  [relevance] verdict=%s", rel.get("verdict"))

    # ---- 2. Citation Correctness ----
    if "citation_correctness" in metrics_to_run:
        # Use all cited IDs if available, falling back to single cited_action_id
        all_cited_ids = list(record.cited_action_ids) if record.cited_action_ids else []
        if not all_cited_ids and record.cited_action_id:
            all_cited_ids = [record.cited_action_id]

        logger.info(
            "  [citation_correctness] cited_ids=%s for '%s...'",
            all_cited_ids,
            statement[:60],
        )

        if not all_cited_ids:
            # No citation at all — auto-No
            cc = judge_metrics.citation_correctness(
                statement=statement,
                cited_action_id=None,
                cited_action_text=None,
                model=judge_model,
                provider=judge_provider,
                max_retries=max_retries,
                prompt_template=citation_template,
            )
        elif len(all_cited_ids) == 1:
            # Single citation — original behaviour
            cited_text = action_loader.get_action_text(all_cited_ids[0])
            cc = judge_metrics.citation_correctness(
                statement=statement,
                cited_action_id=all_cited_ids[0],
                cited_action_text=cited_text,
                model=judge_model,
                provider=judge_provider,
                max_retries=max_retries,
                temperature=judge_temperature,
                prompt_template=citation_template,
            )
        else:
            # Multiple citations — check each and return Yes if any supports it
            logger.info(
                "  [citation_correctness] Multiple cited IDs (%d) — checking each.",
                len(all_cited_ids),
            )
            per_id_results = []
            winning_cc = None
            for cid in all_cited_ids:
                cited_text = action_loader.get_action_text(cid)
                cc_i = judge_metrics.citation_correctness(
                    statement=statement,
                    cited_action_id=cid,
                    cited_action_text=cited_text,
                    model=judge_model,
                    provider=judge_provider,
                    max_retries=max_retries,
                    temperature=judge_temperature,
                    prompt_template=citation_template,
                )
                per_id_results.append(cc_i)
                if cc_i.get("verdict") is True and winning_cc is None:
                    winning_cc = cc_i
                    logger.info(
                        "  [citation_correctness] Supported by action %s — verdict=Yes.", cid
                    )

            if winning_cc is not None:
                cc = {
                    **winning_cc,
                    "checked_ids": all_cited_ids,
                    "per_id_results": per_id_results,
                    "multi_citation": True,
                }
            else:
                # None of the cited IDs supported it — use last result as base
                cc = {
                    **per_id_results[-1],
                    "checked_ids": all_cited_ids,
                    "per_id_results": per_id_results,
                    "multi_citation": True,
                    "verdict": False,
                }

        result["llm_citation_correctness"] = cc
        logger.info(
            "  [citation_correctness] verdict=%s  auto_filled=%s",
            cc.get("verdict"),
            cc.get("auto_filled", False),
        )

        # citation_correctness_when_cited: derive for free
        if cc.get("auto_filled"):
            result["llm_citation_correctness_when_cited"] = {
                "metric_name": "citation_correctness_when_cited",
                "statement": statement,
                "cited_action_id": None,
                "reasoning": "No citation provided — statement skipped for this metric.",
                "verdict": None,
                "skipped_no_citation": True,
            }
        else:
            result["llm_citation_correctness_when_cited"] = {
                **cc,
                "metric_name": "citation_correctness_when_cited",
            }

    # ---- 3. Faithfulness ----
    if "faithfulness" in metrics_to_run:
        action_ids_for_faith = list(record.summary_action_ids)
        if record.cited_action_id and record.cited_action_id not in action_ids_for_faith:
            action_ids_for_faith.append(record.cited_action_id)

        use_per_doc = config.get("faithfulness_per_document", False)
        use_majority = config.get("faithfulness_majority_vote", False) and not use_per_doc

        if use_majority:
            all_docs_text = action_loader.get_action_texts_combined(action_ids_for_faith)
            if all_docs_text is None:
                logger.warning(
                    "  [faithfulness_majority] No action docs for IDs %s. Skipping.",
                    action_ids_for_faith,
                )
                result["llm_faithfulness"] = {
                    "metric_name": "faithfulness",
                    "statement": statement,
                    "verdict": None,
                    "reasoning": None,
                    "error": f"No action docs available for IDs: {action_ids_for_faith}",
                }
            else:
                n_votes = config.get("faithfulness_n_votes", 3)
                vote_temp = config.get("faithfulness_vote_temperature", 0.4)
                logger.info(
                    "  [faithfulness_majority] %d votes at temp=%.1f for '%s...'",
                    n_votes, vote_temp, statement[:50],
                )
                faith_mv = faithfulness_majority_vote(
                    statement=statement,
                    all_action_docs_text=all_docs_text,
                    model=judge_model,
                    provider=judge_provider,
                    max_retries=max_retries,
                    n_votes=n_votes,
                    temperature=vote_temp,
                )
                result["llm_faithfulness_majority_vote"] = faith_mv
                # Populate llm_faithfulness for backwards compatibility
                result["llm_faithfulness"] = {
                    "metric_name": "faithfulness",
                    "statement": statement,
                    "verdict": faith_mv.get("verdict"),
                    "reasoning": faith_mv.get("reasoning"),
                    "confidence": faith_mv.get("confidence"),
                    "source": "majority_vote",
                    "vote_counts": faith_mv.get("vote_counts"),
                    "n_votes": faith_mv.get("n_votes"),
                }
                logger.info(
                    "  [faithfulness_majority] verdict=%s votes=%s confidence=%s",
                    faith_mv.get("verdict"),
                    faith_mv.get("vote_counts"),
                    faith_mv.get("confidence"),
                )

        elif use_per_doc:
            logger.info(
                "  [faithfulness_per_doc] Checking %d action IDs independently for '%s...'",
                len(action_ids_for_faith),
                statement[:60],
            )
            faith_pd = faithfulness_per_document(
                statement=statement,
                action_ids=action_ids_for_faith,
                action_loader=action_loader,
                model=judge_model,
                provider=judge_provider,
                max_retries=max_retries,
            )
            result["llm_faithfulness_per_doc"] = faith_pd
            # Also populate llm_faithfulness for backwards compatibility
            result["llm_faithfulness"] = {
                "metric_name": "faithfulness",
                "statement": statement,
                "verdict": faith_pd.get("verdict"),
                "reasoning": faith_pd.get("reasoning"),
                "source": "per_document_aggregated",
                "supporting_doc": faith_pd.get("supporting_doc"),
                "docs_checked": faith_pd.get("docs_checked"),
                "docs_supporting": faith_pd.get("docs_supporting"),
            }
            logger.info(
                "  [faithfulness_per_doc] verdict=%s (%d/%d docs supporting)",
                faith_pd.get("verdict"),
                faith_pd.get("docs_supporting", 0),
                faith_pd.get("docs_checked", 0),
            )
        else:
            all_docs_text = action_loader.get_action_texts_combined(action_ids_for_faith)
            if all_docs_text is None:
                logger.warning(
                    "  [faithfulness] No action documents available for IDs %s. Skipping.",
                    action_ids_for_faith,
                )
                result["llm_faithfulness"] = {
                    "metric_name": "faithfulness",
                    "statement": statement,
                    "verdict": None,
                    "reasoning": None,
                    "error": f"No action docs available for IDs: {action_ids_for_faith}",
                }
            else:
                logger.info(
                    "  [faithfulness] Using %d action IDs (concatenated) for '%s...'",
                    len(action_ids_for_faith),
                    statement[:60],
                )
                faith = judge_metrics.faithfulness(
                    statement=statement,
                    all_action_docs_text=all_docs_text,
                    model=judge_model,
                    provider=judge_provider,
                    max_retries=max_retries,
                    temperature=judge_temperature,
                    prompt_template=faithfulness_template,
                )
                result["llm_faithfulness"] = faith
                logger.info("  [faithfulness] verdict=%s", faith.get("verdict"))

    return result


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print a concise score summary to stdout."""
    metrics = ["relevance", "citation_correctness", "faithfulness"]
    metric_keys = {
        "relevance": "llm_relevance",
        "citation_correctness": "llm_citation_correctness",
        "faithfulness": "llm_faithfulness",
    }

    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE EVALUATION SUMMARY — ALL REVIEWERS")
    print("=" * 70)
    print(f"Total records in output: {len(results)}")
    print()

    for metric in metrics:
        key = metric_keys[metric]
        verdicts = [
            r[key]["verdict"]
            for r in results
            if key in r and r[key] is not None and r[key].get("verdict") is not None
        ]
        auto_filled = sum(
            1 for r in results
            if key in r and r[key] is not None and r[key].get("auto_filled")
        )
        n = len(verdicts)
        if n == 0:
            print(f"{metric:30s}  N/A (no results)")
            continue
        n_yes = sum(1 for v in verdicts if v is True)
        score = n_yes / n
        print(
            f"{metric:30s}  {n_yes:>3}/{n:<3} ({score:.1%})"
            + (f"  [{auto_filled} auto-No]" if auto_filled else "")
        )

        # Confidence breakdown — faithfulness only (only metric with confidence field)
        if metric == "faithfulness":
            conf_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
            for r in results:
                entry = r.get(key)
                if not entry or entry.get("verdict") is None:
                    continue
                conf = entry.get("confidence")
                if conf in conf_counts:
                    conf_counts[conf] += 1
                elif conf is None:
                    conf_counts["unknown"] += 1
            total_conf = sum(conf_counts.values())
            if total_conf > 0 and conf_counts["unknown"] < total_conf:
                parts = [
                    f"{level}: {conf_counts[level]} ({conf_counts[level]/total_conf:.0%})"
                    for level in ("high", "medium", "low")
                    if conf_counts[level] > 0
                ]
                print(f"  {'confidence breakdown:':<28}  {', '.join(parts)}")
                # Also show Yes/No rate per confidence level
                for level in ("high", "medium", "low"):
                    level_results = [
                        r for r in results
                        if r.get(key) and r[key].get("confidence") == level
                        and r[key].get("verdict") is not None
                    ]
                    if level_results:
                        n_yes_l = sum(1 for r in level_results if r[key]["verdict"] is True)
                        print(f"  {'  ' + level + ' → yes rate:':<28}  "
                              f"{n_yes_l}/{len(level_results)} ({n_yes_l/len(level_results):.0%})")

    print()

    # Per-model breakdown
    models = sorted({r.get("summary_model", "unknown") for r in results})
    if len(models) > 1:
        print("--- Per model ---")
        for model in models:
            model_results = [r for r in results if r.get("summary_model") == model]
            counts = []
            for metric, key in metric_keys.items():
                verdicts = [
                    r[key]["verdict"]
                    for r in model_results
                    if key in r and r[key] is not None and r[key].get("verdict") is not None
                ]
                if verdicts:
                    n_yes = sum(1 for v in verdicts if v)
                    counts.append(f"{metric[:6]}:{n_yes}/{len(verdicts)}")
            print(f"  {model:<45s}  {', '.join(counts)}")

    print()

    # Per-reviewer breakdown
    reviewers = sorted({r.get("source_file", "unknown") for r in results})
    if len(reviewers) > 1:
        print("--- Per reviewer file ---")
        for src in reviewers:
            src_results = [r for r in results if r.get("source_file") == src]
            counts = []
            for metric, key in metric_keys.items():
                verdicts = [
                    r[key]["verdict"]
                    for r in src_results
                    if key in r and r[key] is not None and r[key].get("verdict") is not None
                ]
                if verdicts:
                    n_yes = sum(1 for v in verdicts if v)
                    counts.append(f"{metric[:6]}:{n_yes}/{len(verdicts)}")
            print(f"  {src:<45s}  {', '.join(counts)}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(config: dict = CONFIG) -> list[dict]:
    """
    Run the full LLM-as-judge evaluation over all reviewer files.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see CONFIG at top of file).

    Returns
    -------
    list[dict]
        All evaluation results (including previously completed ones).
    """
    _setup_logging()

    judge_model = config["judge_model"]
    judge_provider = config.get("judge_provider")
    output_file = Path(config["output_file"])
    max_statements = config.get("max_statements")
    delay = config.get("delay_between_statements", 1.0)
    max_retries = config.get("max_retries", 3)
    metrics_to_run = config.get(
        "metrics_to_run", ["relevance", "citation_correctness", "faithfulness"]
    )
    dedup_overlaps = config.get("dedup_overlaps", True)
    answerable_only = config.get("answerable_only", True)

    # --- Load custom (e.g. GEPA-optimised) prompt templates if configured ---
    custom_prompts: dict[str, str] = {}
    prompt_file_keys = {
        "relevance":            config.get("relevance_prompt_file"),
        "citation_correctness": config.get("citation_correctness_prompt_file"),
        "faithfulness":         config.get("faithfulness_prompt_file"),
    }
    for metric_name, fpath in prompt_file_keys.items():
        if fpath:
            template_text = Path(fpath).read_text(encoding="utf-8")
            custom_prompts[metric_name] = template_text
            logger.info(
                "  Loaded custom %s prompt from %s (%d chars)",
                metric_name, fpath, len(template_text),
            )

    logger.info("=" * 70)
    logger.info("LLM-AS-JUDGE EVALUATION — ALL REVIEWERS")
    logger.info("Judge model : %s  (provider: %s)", judge_model, judge_provider)
    logger.info("Metrics     : %s", metrics_to_run)
    logger.info("Output      : %s", output_file)
    logger.info("Dedup overlaps: %s", dedup_overlaps)
    if custom_prompts:
        logger.info("Custom prompts: %s", sorted(custom_prompts.keys()))
    logger.info("=" * 70)

    # --- Load records from all reviewer files ---
    reviewer_files = config.get("reviewer_files", [])
    all_records: list[EvaluationRecord] = []
    for fpath in reviewer_files:
        records = load_human_eval_records(fpath, answerable_only=answerable_only)
        all_records.extend(records)
        logger.info("  Loaded %d records from %s", len(records), fpath)

    logger.info("Total records loaded: %d", len(all_records))

    # --- Load existing results (for resumability) ---
    results = load_results(output_file)
    done_keys = build_done_set(results)
    logger.info("%d records already evaluated (will skip).", len(done_keys))

    # --- Deduplication cache ---
    # Maps dedup_key → evaluated result dict (for re-use with duplicate records)
    dedup_cache: dict[str, dict] = {}
    if dedup_overlaps:
        # Populate cache from already-saved results so we benefit from previous
        # runs even when resuming.
        for r in results:
            key = results_as_unique_key(r)
            if key not in dedup_cache:
                dedup_cache[key] = r

    # --- Set up action loader ---
    action_loader = ActionLoader(
        local_base_dir=config.get("action_local_dir"),
        fetch_from_web=config.get("action_fetch_from_web", False),
    )

    # --- Build to-do list, apply sampling if requested ---
    to_do_records = [
        r for r in all_records
        if results_as_unique_key(r.to_dict()) not in done_keys
    ]

    seed = config.get("random_seed", 42)
    use_stratified = config.get("stratified_sample", False)

    if use_stratified and max_statements is not None and max_statements < len(to_do_records):
        priority_synopses = config.get("priority_synopses", [])
        to_do_records = _stratified_sample(
            to_do_records, max_statements, priority_synopses, seed=seed
        )
        logger.info(
            "Stratified sample: %d records drawn (seed=%s, %d priority synopses).",
            len(to_do_records), seed, len(priority_synopses),
        )
    else:
        rng = random.Random(seed)
        rng.shuffle(to_do_records)
        if seed is not None:
            logger.info("Records shuffled with seed=%s.", seed)

    total_to_do = len(to_do_records)
    if max_statements is not None and not use_stratified:
        total_to_do = min(total_to_do, max_statements)

    logger.info("Will evaluate up to %d new records this run.", total_to_do)

    evaluated_this_run = 0

    for record in to_do_records:
        record_key = results_as_unique_key(record.to_dict())

        # Enforce run limit (for non-stratified runs)
        if max_statements is not None and evaluated_this_run >= max_statements:
            logger.info(
                "Reached max_statements=%d limit. Stopping.", max_statements
            )
            break

        logger.info(
            "[%d/%d] %s | %s | '%s...'",
            evaluated_this_run + 1,
            total_to_do,
            record.synopsis,
            record.summary_model,
            record.statement[:50],
        )

        if dedup_overlaps and record_key in dedup_cache:
            # Re-use LLM scores from the first encounter of this statement.
            cached = dedup_cache[record_key]
            result = record.to_dict()  # keeps this record's reviewer metadata
            result["judge_model"] = cached.get("judge_model", judge_model)
            result["judge_provider"] = cached.get("judge_provider", judge_provider)
            result["llm_relevance"] = cached.get("llm_relevance")
            result["llm_citation_correctness"] = cached.get("llm_citation_correctness")
            result["llm_faithfulness"] = cached.get("llm_faithfulness")
            logger.info(
                "  [dedup] Reused cached LLM scores for duplicate statement."
            )
        else:
            result = evaluate_record(
                record=record,
                action_loader=action_loader,
                judge_model=judge_model,
                judge_provider=judge_provider,
                metrics_to_run=metrics_to_run,
                max_retries=max_retries,
                config=config,
                custom_prompts=custom_prompts,
            )
            if dedup_overlaps:
                dedup_cache[record_key] = result

        results.append(result)
        done_keys.add(record_key)
        evaluated_this_run += 1

        # Incremental save after every record
        save_results(results, output_file)
        logger.info(
            "  Saved. Progress: %d/%d records this run.",
            evaluated_this_run,
            total_to_do,
        )

        if delay > 0 and evaluated_this_run < total_to_do:
            time.sleep(delay)

    logger.info(
        "Evaluation complete. Evaluated %d records this run (%d total in file).",
        evaluated_this_run,
        len(results),
    )

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(CONFIG)
