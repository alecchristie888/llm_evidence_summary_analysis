"""
Automated faithfulness prompt optimisation using GEPA's optimize_anything.

This script is STANDALONE — it does not modify any existing code. It imports
your existing infrastructure (action_loader, human_eval_loader, summaries_loader)
but writes results to its own output files.

What it does
------------
Uses GEPA (Genetic-Pareto prompt optimiser) to automatically search for a
prompt (faithfulness, relevance, or citation correctness) that maximises
agreement with human reviewer scores.

Dataset partitioning — Option C: hybrid IID + reviewer-OOD holdout
------------------------------------------------------------------
From the full deduped pool of human-labelled statements, we carve out:

  * Test-OOD : 50% of Reviewer 3's statements (between-reviewer
               generalisation — GEPA never sees these)
  * Test-IID : random 15% slice from the remainder, stratified by Yes/No
               (i.i.d. holdout — GEPA never sees these)
  * GEPA pool: everything else, then split stratified into:
               - Train (40% of pool, capped per metric)  — used for reflection
               - Val   (60% of pool, capped per metric)  — used for Pareto
                                                            candidate selection

Class balance is preserved at every level via stratified sampling.
This makes test-vs-val gaps interpretable:
  * If Test-IID matches val → the prompt generalises to unseen statements
  * If Test-OOD lags Test-IID → the prompt overfits to R1+R2 annotation style

Why these settings
------------------
The budget, train/val sizes, judge temperature, and minibatch size below
follow the published GEPA ablations and the Decagon production tuning study
(decagon.ai/blog/optimizing-gepa-for-production):
  * 20–100 train examples is the sweet spot; >500 hurts generalisation
  * Judge temperature must be 0.0 so Pareto fitness signal is stable
  * Reflection minibatch >= 8 so heavily imbalanced classes (relevance: 7%
    No) reliably appear in each reflection round
  * Per-metric overrides applied (see PER_METRIC_OVERRIDES below)

Setup
-----
1. Install GEPA:
       pip install gepa

2. Your existing OPENROUTER_API_KEY in .env is all you need — no separate
   GEPA API key required. GEPA is fully open source.

3. Set OPTIMISE_METRIC below and (optionally) tune CONFIG.

4. Run:
       python optimise_faithfulness_prompt.py

Output
------
  outputs/<run>/<metric>/optimised_<metric>_prompt.txt
  outputs/<run>/<metric>/gepa_optimisation_history_<metric>.json
  outputs/<run>/<metric>/holdouts_<metric>.json   ← use this to score
                                                    Test-IID and Test-OOD
                                                    with the optimised prompt
"""

import json
import logging

# Module-level logger so helper functions (e.g. the evaluator's retry loop)
# can log even before main() configures the root handler.
logger = logging.getLogger(__name__)
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on Python path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Files are organised by reviewer so we can implement Option C:
#   * Test-IID: random ~15% drawn from across ALL reviewers (i.i.d. holdout)
#   * Test-OOD: half of Reviewer 3's statements only seen at evaluation time
#               (between-reviewer generalisation check)
#   * GEPA pool: everything else — split into train/val by `train_val_split`.
REVIEWER_FILES = {
    "reviewer1": [
        "human_data/sampled_subset_reviewer1_original.xlsx",
        "human_data/sampled_subset_reviewer1_extras_AT.xlsx",
    ],
    "reviewer2": [
        "human_data/sampled_subset_reviewer2_original.xlsx",
        "human_data/sampled_subset_reviewer2_extras_VC.xlsx",
    ],
    "reviewer3": [
        "human_data/sampled_subset_reviewer3_NT.xlsx",
        "human_data/sampled_subset_reviewer3_extras_NT.xlsx",
    ],
}

# Per-metric overrides applied on top of the base config below.
# Class imbalance differs sharply across metrics, so train size and
# minibatch are tuned per metric to ensure the minority class appears
# in each reflection minibatch with high probability.
PER_METRIC_OVERRIDES = {
    "faithfulness": {  # ~17% No
        "max_train_statements":      140,
        "max_val_statements":        400,
        "max_metric_calls":          3000,
        # Minibatch=16 gives ~5 expected No per round AFTER train
        # oversampling (which forces 30% No in train).
        "reflection_minibatch_size": 16,
        # Use balanced accuracy + oversampling so GEPA prioritises catching
        # unsupported claims (the No class). With 17% No, plain accuracy still
        # rewards Yes-leaning prompts more than is ideal for a strict judge;
        # balanced accuracy gives correct No predictions equal weight to Yes.
        # Oversampling (30% No in train) ensures reflection minibatches
        # reliably contain failures to learn from.
        "balanced_accuracy":         True,
        "train_no_class_target_frac": 0.30,
    },
    "relevance": {  # ~6.5% No after Option C splits — most imbalanced
        "max_train_statements":      140,
        "max_val_statements":        400,
        "max_metric_calls":          3000,
        # Minibatch=16 gives ~5 expected No per round AFTER train
        # oversampling (which forces 30% No in train).
        "reflection_minibatch_size": 16,
        # --- Imbalance handling (relevance only) ---
        # Use balanced accuracy as the GEPA score so candidates that correctly
        # identify the rare No class aren't penalised by tiny false-No costs
        # on the dominant Yes class. Without this, the seed's "always Yes"
        # behaviour scores 0.94 and is essentially unbeatable.
        "balanced_accuracy":         True,
        # Upsample No examples in TRAIN only (val/test untouched) so each
        # reflection minibatch contains plenty of failures to learn from.
        # 0.30 means "30% of train will be No examples" — at minibatch=16,
        # that gives ~5 expected No per reflection round vs ~1 without.
        "train_no_class_target_frac": 0.30,
    },
    "citation_correctness": {  # ~12% No (after auto-No filter)
        "max_train_statements":      140,
        "max_val_statements":        400,
        "max_metric_calls":          3000,
        # Minibatch=16 gives ~5 expected No per round AFTER train
        # oversampling (which forces 30% No in train).
        "reflection_minibatch_size": 16,
        # Use balanced accuracy + oversampling so GEPA learns to flag
        # uncited or weakly-cited claims. Pool is already filtered to
        # cited-only statements (auto-No statements are deterministic).
        "balanced_accuracy":         True,
        "train_no_class_target_frac": 0.30,
    },
}

# Dual-seed runs: GEPA is sensitive to the train/val split. Running with
# two independent seeds and comparing the resulting prompts lets us:
#   (a) spot instability — large gaps between seeds = overfit risk
#   (b) get a more honest performance estimate (average across seeds)
#   (c) pick the more robust prompt (winner on both seeds)
# Outputs from each seed are written to seed-suffixed filenames, e.g.
#   optimised_<metric>_prompt_seed1234.txt
#   gepa_optimisation_history_<metric>_seed1234.json
# The holdout manifest is identical across seeds (it's controlled by
# holdout_seed, not split_seed) so only one is written per metric.
SPLIT_SEEDS = [1234, 5678]

CONFIG = {
    "reviewer_files": REVIEWER_FILES,

    # --- Holdout strategy (Option C: hybrid IID + reviewer-OOD) ---
    # Fraction of the ENTIRE deduped pool reserved as Test-IID (random
    # mix from all 3 reviewers, never seen by GEPA).
    "test_iid_frac":   0.15,
    # Fraction of Reviewer 3's statements reserved as Test-OOD (between-
    # reviewer generalisation check). The remaining Reviewer 3 statements
    # flow into the GEPA train/val pool.
    "test_ood_r3_frac": 0.50,

    # Within the remaining GEPA pool, fraction used for training
    # (the rest becomes the Pareto/validation set).
    "train_val_split": 0.40,

    # Independent random seeds — separate so they don't accidentally collide
    # with the (now removed) seed=42 used inside build_dataset.
    # `split_seed` is overridden per-run inside main() — see SPLIT_SEEDS above.
    "holdout_seed": 7,
    "split_seed":   1234,

    # --- Action data ---
    "action_local_dir": "data/actions/background_key_messages/bg_km_all",

    # --- Summaries index ---
    "summaries_base_dir": (
        "answerable_passed_qus_summaries/answerable_passed_qus_summaries/"
        "hybrid_cross-encoder"
    ),

    # --- Judge model (used to evaluate each candidate prompt) ---
    "judge_model": "google/gemini-3.1-pro-preview",
    "judge_provider": None,
    # Temperature for the JUDGE during evaluation. Keep deterministic so
    # GEPA's Pareto frontier sees a stable fitness signal — the reflection
    # LM (proposer) can still be sampled at higher temperature.
    "judge_temperature": 0.0,

    # --- GEPA settings ---
    # Reflection model proposes improved candidates. Frontier-tier required
    # (Decagon ablations: small models fail at the meta-reasoning task).
    "gepa_reflection_model": "anthropic/claude-opus-4.7",

    # The four budget knobs below are populated by the active metric's
    # entry in PER_METRIC_OVERRIDES (see select_active_config()).
    "max_metric_calls":          None,
    "max_train_statements":      None,
    "max_val_statements":        None,
    "reflection_minibatch_size": None,

    # Parallel evaluation (does not affect quality, only wall-clock).
    "parallel":    True,
    "max_workers": 8,

    # --- Output ---
    "output_dir": "outputs/gepa_v2.0_fullrun",
}

# ---------------------------------------------------------------------------
# Seed prompts — set OPTIMISE_METRIC below to choose which to run
# ---------------------------------------------------------------------------

# Which metric to optimise: "faithfulness", "relevance", or "citation_correctness"
OPTIMISE_METRIC = "citation_correctness"

SEED_PROMPT_FAITHFULNESS = """You are evaluating whether a statement is supported by evidence from a set of conservation evidence documents.

Statement: {statement}

Available Evidence Documents:
---
{all_action_docs_text}
---

Task: Determine whether the specific claims in the statement are supported by the evidence documents shown above. The specific findings — numbers, outcomes, species, locations — must be traceable to the documents. It is not sufficient that the general topic is covered; the actual claim must be present. Mark No if the statement contains specific details that are absent from or contradict the documents.

Provide your reasoning first, then give a final verdict of 'Yes' (fully supported by the available evidence) or 'No' (not fully supported).

Respond ONLY as a JSON object with the following fields:
  "statement": the statement text
  "reasoning": your explanation
  "verdict": "Yes" or "No"
"""

SEED_PROMPT_RELEVANCE = """You are evaluating whether a statement is relevant to answering a specific question.

Question: {question}

Statement: {statement}

Task: Determine whether the statement is relevant to answering the question above.
A statement is relevant if it provides information that directly or indirectly helps answer the question. 
Do not require the statement to fully answer the question — contextual or supporting information counts as relevant. 
Only mark No if the statement has no meaningful connection to the question. 

Provide your reasoning first, then give a final verdict of 'Yes' (relevant) or 'No' (not relevant).

Respond ONLY as a JSON object with the following fields:
  "statement": the statement text
  "reasoning": your explanation
  "verdict": "Yes" or "No"
"""

SEED_PROMPT_CITATION_CORRECTNESS = """You are evaluating whether a statement is accurately supported by a specific cited document.

Statement: {statement}

Cited Action ID: {cited_action_id}

Content of Cited Document (Action {cited_action_id}):
---
{cited_action_text}
---

Task: Determine whether the core factual claim of the statement is supported by the cited document shown above. Reasonable paraphrasing is acceptable — do not require verbatim wording, only that the underlying finding or fact is present in the document. 
The statement should be marked No only if it contains specific claims that are absent from or contradict the cited document.
Provide your reasoning first, then give a final verdict of 'Yes' (fully supported by the cited document) or 'No' (not fully supported).

Respond ONLY as a JSON object with the following fields:
    "statement": the statement text
    "cited_action_id": the action ID
    "reasoning": your explanation
    "verdict": "Yes" or "No"
"""

# Select active seed prompt based on metric
if OPTIMISE_METRIC == "relevance":
    SEED_PROMPT = SEED_PROMPT_RELEVANCE
elif OPTIMISE_METRIC == "citation_correctness":
    SEED_PROMPT = SEED_PROMPT_CITATION_CORRECTNESS
else:
    SEED_PROMPT = SEED_PROMPT_FAITHFULNESS
# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

def _lazy_imports():
    """Import everything after path is set up."""
    global load_human_eval_records, SummariesIndex, ActionLoader, call_judge_llm
    from data_loaders.human_eval_loader import load_human_eval_records
    from data_loaders.summaries_loader import SummariesIndex
    from data_loaders.action_loader import ActionLoader
    from judge.llm_client import call_judge_llm


# ---------------------------------------------------------------------------
# Build dataset of human-scored examples
# ---------------------------------------------------------------------------

def build_dataset(
    reviewer_files: dict[str, list[str]],
    summaries_index,
    action_loader,
    metric: str = "faithfulness",
) -> list[dict]:
    """
    Load human-scored examples for the given metric, tagged by source reviewer.

    Each returned example carries a `_reviewer` key ("reviewer1"/"reviewer2"
    /"reviewer3") so we can build the Option C holdout (random IID + half of
    Reviewer 3 reserved as OOD).

    Deduplication uses (question, summary_model, statement). If the same
    statement appears under two reviewers (rare but possible), the first one
    encountered — in the order of `reviewer_files` — wins.
    """
    from data_loaders.human_eval_loader import load_human_eval_records

    human_field = f"human_{metric}"
    examples: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for reviewer_id, files in reviewer_files.items():
        for fpath in files:
            if not Path(fpath).exists():
                print(f"  WARNING: file not found — {fpath}")
                continue
            records = load_human_eval_records(fpath, answerable_only=True)
            for rec in records:
                if getattr(rec, human_field) is None:
                    continue
                key = (
                    rec.question.strip(),
                    rec.summary_model.strip(),
                    rec.statement.strip(),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                example = {
                    "statement":     rec.statement,
                    "question":      rec.question,
                    "synopsis":      rec.synopsis,
                    "summary_model": rec.summary_model,
                    human_field:     getattr(rec, human_field),
                    "_reviewer":     reviewer_id,
                }

                # Faithfulness needs all action document text
                if metric == "faithfulness":
                    action_ids = list(rec.summary_action_ids)
                    if rec.cited_action_id and rec.cited_action_id not in action_ids:
                        action_ids.append(rec.cited_action_id)
                    docs_text = action_loader.get_action_texts_combined(action_ids) or ""
                    if not docs_text.strip():
                        continue
                    example["all_action_docs_text"] = docs_text

                # Citation correctness needs the single cited document text.
                # Skip auto-No statements (no citation) — they're trivial and
                # would bias GEPA toward a lenient prompt.
                elif metric == "citation_correctness":
                    if not rec.cited_action_id:
                        continue  # skip uncited statements
                    cited_text = action_loader.get_action_text(rec.cited_action_id)
                    if not cited_text or not cited_text.strip():
                        continue  # skip if doc unavailable
                    example["cited_action_id"] = rec.cited_action_id
                    example["cited_action_text"] = cited_text

                examples.append(example)

    # Reviewer breakdown for visibility
    from collections import Counter
    by_reviewer = Counter(ex["_reviewer"] for ex in examples)
    print(f"  Loaded {len(examples)} scored examples ({metric})")
    for r, n in sorted(by_reviewer.items()):
        n_yes = sum(1 for ex in examples if ex["_reviewer"] == r and ex[human_field])
        print(f"    {r}: {n} ({n_yes} Yes / {n - n_yes} No)")
    return examples


# ---------------------------------------------------------------------------
# Option C: hybrid IID + reviewer-OOD holdout split
# ---------------------------------------------------------------------------

def _stratified_take(
    examples: list[dict],
    frac: float,
    label_key: str,
    rng,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified random split that preserves the Yes/No ratio.
    Returns (taken, remaining). `taken` has approximately `frac` of examples.
    """
    pos = [ex for ex in examples if ex[label_key]]
    neg = [ex for ex in examples if not ex[label_key]]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_take = int(round(len(pos) * frac))
    n_neg_take = int(round(len(neg) * frac))
    taken     = pos[:n_pos_take] + neg[:n_neg_take]
    remaining = pos[n_pos_take:] + neg[n_neg_take:]
    rng.shuffle(taken); rng.shuffle(remaining)
    return taken, remaining


def build_holdouts_and_pool(
    all_examples: list[dict],
    metric: str,
    test_iid_frac: float,
    test_ood_r3_frac: float,
    holdout_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Implements Option C:
      1. Reserve `test_ood_r3_frac` of Reviewer 3's examples as Test-OOD
         (between-reviewer generalisation check).
      2. From the remainder (R1 + R2 + half of R3), reserve `test_iid_frac`
         as Test-IID (random mix, stratified by Yes/No label).
      3. Whatever is left becomes the GEPA pool (later split into train/val).

    Both holdout slices are stratified by the human label so class balance
    is preserved. Reviewer 3's split is also stratified internally.
    """
    import random
    rng = random.Random(holdout_seed)

    label_key = f"human_{metric}"

    r3 = [ex for ex in all_examples if ex["_reviewer"] == "reviewer3"]
    other = [ex for ex in all_examples if ex["_reviewer"] != "reviewer3"]

    # Step 1: half of R3 → OOD test
    test_ood, r3_keep = _stratified_take(r3, test_ood_r3_frac, label_key, rng)

    # Step 2: random IID slice from (R1 + R2 + remaining R3)
    pool_pre = other + r3_keep
    test_iid, gepa_pool = _stratified_take(pool_pre, test_iid_frac, label_key, rng)

    return gepa_pool, test_iid, test_ood


def stratified_train_val_split(
    pool: list[dict],
    metric: str,
    train_val_split: float,
    split_seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified train/val split on the GEPA pool, preserving the Yes/No ratio.
    Train (a.k.a. "feedback" set) is `train_val_split` of the pool; val
    (a.k.a. "Pareto" set) is the remainder.
    """
    import random
    rng = random.Random(split_seed)
    label_key = f"human_{metric}"
    train, val = _stratified_take(pool, train_val_split, label_key, rng)
    return train, val


def oversample_minority_class(
    train: list[dict],
    metric: str,
    target_minority_frac: float,
    seed: int,
) -> list[dict]:
    """
    Upsample the minority (No) class in TRAIN ONLY so that No examples make
    up `target_minority_frac` of the returned list. Used on heavily imbalanced
    metrics (relevance) so each reflection minibatch contains enough failures
    for the reflection LM to actually learn from.

    Implementation: keep all Yes examples as-is; sample No examples *with
    replacement* until the target ratio is reached. Sampling with replacement
    is deliberate — it lets us exceed |No_unique| when the natural class is
    very small (e.g. only ~6 No examples in a 100-example train cap).

    The val and test splits are NOT modified. They keep the natural
    distribution so reported val/test numbers reflect real-world performance.
    """
    import random
    rng = random.Random(seed)
    label_key = f"human_{metric}"
    yes = [ex for ex in train if ex[label_key]]
    no  = [ex for ex in train if not ex[label_key]]
    if not no:
        logger.warning(
            "No minority-class examples in train — cannot oversample. "
            "Returning train unchanged."
        )
        return train
    if not yes:
        return train

    # Solve: target = no_count_after / (yes_count + no_count_after)
    # So:    no_count_after = yes_count * target / (1 - target)
    target = max(0.01, min(0.99, target_minority_frac))
    no_target = int(round(len(yes) * target / (1.0 - target)))
    no_target = max(no_target, len(no))  # never DOWN-sample the minority

    if no_target <= len(no):
        # Already at or above target — nothing to do
        oversampled_no = no
    else:
        # Keep all originals, then sample with replacement to fill up
        extras_needed = no_target - len(no)
        oversampled_no = list(no) + [rng.choice(no) for _ in range(extras_needed)]

    out = yes + oversampled_no
    rng.shuffle(out)
    actual_no_frac = len(oversampled_no) / len(out)
    logger.info(
        "Train oversampling: kept %d Yes + upsampled %d → %d No "
        "(unique %d) → final train size %d (%.1f%% No, target %.1f%%)",
        len(yes), len(no), len(oversampled_no), len(no),
        len(out), actual_no_frac * 100, target * 100,
    )
    return out


def _summarise_split(name: str, examples: list[dict], metric: str) -> None:
    """Pretty-print class balance and reviewer mix for a split."""
    from collections import Counter
    label_key = f"human_{metric}"
    n = len(examples)
    if n == 0:
        print(f"  {name}: 0 examples")
        return
    n_yes = sum(1 for ex in examples if ex[label_key])
    by_rev = Counter(ex["_reviewer"] for ex in examples)
    rev_str = ", ".join(f"{r}={c}" for r, c in sorted(by_rev.items()))
    print(
        f"  {name:>10}: n={n:>4}  Yes={n_yes:>4} ({n_yes/n:>5.1%})  "
        f"No={n-n_yes:>4} ({(n-n_yes)/n:>5.1%})  | {rev_str}"
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Transient OpenRouter / LiteLLM error patterns that warrant a local retry.
# We've observed `gemini-3.1-pro-preview` occasionally returning empty content,
# which surfaces as one of these strings somewhere in the exception message.
# Adding more patterns here is safe — they only trigger an extra retry, not a
# different scoring path.
_TRANSIENT_JUDGE_ERROR_PATTERNS = (
    "the JSON object must be str, bytes or bytearray, not NoneType",
    "Expecting value",            # plain JSONDecodeError on empty body
    "Expecting property name",    # truncated JSON
    "rate limit",                 # OpenRouter throttling
    "rate_limit",
    "429",
    "503",
    "502",
    "upstream connect error",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "Read timed out",
    "empty response",
    "None\" is not of type",      # schema-validation on null content
)


def _is_transient_judge_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p.lower() in msg for p in _TRANSIENT_JUDGE_ERROR_PATTERNS)


def make_evaluator(
    judge_model: str,
    judge_provider: str | None,
    judge_temperature: float = 0.0,
    metric: str = "faithfulness",
    max_local_retries: int = 4,
    base_backoff_seconds: float = 2.0,
    balanced_accuracy: bool = False,
    yes_weight: float = 1.0,
    no_weight: float = 1.0,
):
    """
    Returns an evaluator function for GEPA.

    Wraps `call_judge_llm` with a local retry loop on top of whatever retries
    the client already does, so that transient errors (empty responses, rate
    limits, brief upstream blips) don't poison GEPA's fitness signal or burn
    budget on what should really be a single successful call.

    Retry policy:
      * Transient error (see _TRANSIENT_JUDGE_ERROR_PATTERNS): wait with
        exponential backoff and try again, up to `max_local_retries` times.
      * Non-transient error (KeyError, schema mismatch, auth failure, etc.):
        give up immediately and return (0.0, {"Error": ...}) so GEPA can
        log the issue and move on.

    Scoring:
      * Default (balanced_accuracy=False): score = 1.0 if correct else 0.0.
      * balanced_accuracy=True: a correct prediction on a Yes example scores
        `yes_weight`, on a No example scores `no_weight`. Choose weights so
        that the aggregate score equals balanced accuracy:
           yes_weight = 0.5 / P(Yes),  no_weight = 0.5 / P(No).
        These should be derived from the *val set*'s class proportions and
        used everywhere (train, val, test) so all reported numbers live on
        the same scale.
    """
    import time

    from judge.llm_client import call_judge_llm

    _SCHEMA = {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "reasoning": {"type": "string"},
            "verdict":   {"type": "string", "enum": ["Yes", "No"]},
        },
        "required": ["statement", "reasoning", "verdict"],
        "additionalProperties": False,
    }

    human_field = f"human_{metric}"

    def _call_with_local_retry(prompt: str) -> dict:
        """Call the judge, retrying transient failures with backoff."""
        last_exc: Exception | None = None
        for attempt in range(1, max_local_retries + 1):
            try:
                result = call_judge_llm(
                    prompt=prompt,
                    response_schema=_SCHEMA,
                    schema_name="FaithfulnessVerdict",
                    model=judge_model,
                    provider=judge_provider,
                    max_retries=2,
                    temperature=judge_temperature,
                )
                if not isinstance(result, dict) or "verdict" not in result:
                    # Treat malformed/empty result as a transient error too
                    raise ValueError(
                        f"judge returned invalid payload: {type(result).__name__}"
                    )
                return result
            except Exception as exc:  # noqa: BLE001 — we re-raise non-transient
                last_exc = exc
                if not _is_transient_judge_error(exc):
                    raise
                if attempt >= max_local_retries:
                    break
                wait = base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient judge error (attempt %d/%d): %s — sleeping %.1fs",
                    attempt, max_local_retries, str(exc)[:160], wait,
                )
                time.sleep(wait)
        # Exhausted retries — propagate the last exception
        assert last_exc is not None
        raise last_exc

    def evaluator(candidate: str, example: dict) -> tuple[float, dict]:
        statement   = example["statement"]
        human_label = example[human_field]  # True/False

        # Inject placeholders into the candidate prompt
        try:
            if metric == "faithfulness":
                prompt = candidate.format(
                    statement=statement,
                    all_action_docs_text=example["all_action_docs_text"],
                )
            elif metric == "citation_correctness":
                cited_id = example["cited_action_id"]
                prompt = candidate.format(
                    statement=statement,
                    cited_action_id=cited_id,
                    cited_action_text=example["cited_action_text"],
                )
            else:  # relevance
                prompt = candidate.format(
                    question=example["question"],
                    statement=statement,
                )
        except (KeyError, IndexError) as e:
            return 0.0, {"Error": f"Prompt formatting failed: {e}"}

        try:
            result = _call_with_local_retry(prompt)
            raw_verdict = result.get("verdict", "No")
            llm_verdict = (raw_verdict == "Yes")
            reasoning   = result.get("reasoning", "")
        except Exception as e:
            # All retries exhausted (or non-transient error). Returning a
            # 0.0 score with an Error key keeps GEPA running, but log loudly
            # so we can spot persistent failures in the run log.
            logger.error(
                "Judge failed after retries on statement %r: %s",
                statement[:80], str(e)[:160],
            )
            return 0.0, {
                "Error": str(e),
                "Statement": statement[:100],
            }

        # ---- score ---------------------------------------------------
        is_correct = (llm_verdict == human_label)
        if balanced_accuracy:
            # Weight each example by class so the aggregate equals
            # 0.5*sensitivity + 0.5*specificity.
            class_weight = yes_weight if human_label else no_weight
            score = class_weight if is_correct else 0.0
        else:
            score = 1.0 if is_correct else 0.0

        # Diagnostic side information for GEPA's reflection step
        side_info = {
            "Statement":     statement[:150],
            "Question":      example.get("question", "")[:100],
            "LLM_verdict":   raw_verdict,
            "Human_verdict": "Yes" if human_label else "No",
            "Correct":       str(score == 1.0),
            "Error_type":    (
                "false_positive"  if (not human_label and llm_verdict) else
                "false_negative"  if (human_label and not llm_verdict) else
                "correct"
            ),
            "Reasoning":     reasoning[:300],
            "Synopsis":      example.get("synopsis", ""),
            "Model":         example.get("summary_model", ""),
        }
        return score, side_info

    return evaluator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # `logger` is already defined at module scope; basicConfig wires up the
    # root handler so its messages now get formatted/printed.

    # Check GEPA is installed
    try:
        import gepa.optimize_anything as oa
        from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig
    except ImportError:
        print("ERROR: GEPA is not installed. Run: pip install gepa")
        print("See: https://gepa-ai.github.io/gepa/docs/getting_started")
        sys.exit(1)

    # Lazy imports from local codebase
    _lazy_imports()
    from data_loaders.action_loader import ActionLoader
    from data_loaders.summaries_loader import SummariesIndex

    cfg = dict(CONFIG)  # shallow copy so per-metric overrides don't mutate global
    metric = OPTIMISE_METRIC

    # Apply per-metric overrides (train/val sizes, budget, minibatch)
    overrides = PER_METRIC_OVERRIDES.get(metric, {})
    for k, v in overrides.items():
        cfg[k] = v

    output_dir = Path(cfg["output_dir"]) / metric
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROMPT OPTIMISATION WITH GEPA — metric: %s", metric.upper())
    logger.info("Judge model:           %s  (T=%.2f)",
                cfg["judge_model"], cfg["judge_temperature"])
    logger.info("Reflection model:      %s", cfg["gepa_reflection_model"])
    logger.info("Max metric calls:      %s", cfg["max_metric_calls"])
    logger.info("Train cap / Val cap:   %s / %s",
                cfg["max_train_statements"], cfg["max_val_statements"])
    logger.info("Reflection minibatch:  %s", cfg["reflection_minibatch_size"])
    logger.info("Output dir:            %s", output_dir)
    logger.info("=" * 70)

    # Set up shared infrastructure
    logger.info("Loading summaries index...")
    summaries_index = SummariesIndex(cfg["summaries_base_dir"])

    logger.info("Setting up action loader...")
    action_loader = ActionLoader(
        local_base_dir=cfg.get("action_local_dir"),
        fetch_from_web=False,
    )

    # ------------------------------------------------------------------
    # Load + Option C holdouts
    # ------------------------------------------------------------------
    logger.info("Loading reviewer files (tagged by reviewer)...")
    all_data = build_dataset(
        cfg["reviewer_files"],
        summaries_index=summaries_index,
        action_loader=action_loader,
        metric=metric,
    )
    if not all_data:
        logger.error("No examples found. Check file paths and that human scores exist.")
        sys.exit(1)

    logger.info("Building Option C holdouts (Test-IID + Test-OOD R3)...")
    gepa_pool, test_iid, test_ood = build_holdouts_and_pool(
        all_data,
        metric=metric,
        test_iid_frac=cfg["test_iid_frac"],
        test_ood_r3_frac=cfg["test_ood_r3_frac"],
        holdout_seed=cfg["holdout_seed"],
    )

    # ------------------------------------------------------------------
    # Dual-seed loop: run GEPA twice with different train/val splits
    # ------------------------------------------------------------------
    seed_results: list[tuple[int, object, Path, Path]] = []
    for _seed_idx, _split_seed in enumerate(SPLIT_SEEDS):
        cfg["split_seed"] = _split_seed
        seed_suffix = f"_seed{_split_seed}"

        logger.info("=" * 70)
        logger.info(
            "SEED RUN %d/%d — split_seed=%d  (metric: %s)",
            _seed_idx + 1, len(SPLIT_SEEDS), _split_seed, metric,
        )
        logger.info("=" * 70)

        seed_result, seed_prompt_path, seed_history_path = _run_single_seed(
            cfg=cfg,
            metric=metric,
            output_dir=output_dir,
            seed_suffix=seed_suffix,
            gepa_pool=gepa_pool,
            test_iid=test_iid,
            test_ood=test_ood,
            objectives_in=None,  # populated inside helper
            backgrounds_in=None,
            seed_prompt=SEED_PROMPT,
        )
        seed_results.append((_split_seed, seed_result, seed_prompt_path, seed_history_path))

    # ------------------------------------------------------------------
    # Cross-seed summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DUAL-SEED SUMMARY — metric: %s" % metric.upper())
    print("=" * 70)
    print(f"{'Seed':<8} {'Seed val':<12} {'Best val':<12} {'Δ':<10} {'Prompt file'}")
    print("-" * 70)
    for _seed, _result, _prompt_path, _ in seed_results:
        _val_scores = getattr(_result, "val_aggregate_scores", None)
        _best_idx   = getattr(_result, "best_idx", None)
        if _val_scores and _best_idx is not None:
            _seed_score = _val_scores[0]
            _best_score = _val_scores[_best_idx]
            _delta = _best_score - _seed_score
            print(
                f"{_seed:<8} {_seed_score:<12.3f} {_best_score:<12.3f} "
                f"{_delta:<+10.3f} {_prompt_path.name}"
            )
        else:
            print(f"{_seed:<8} (no scores)  — see logs above")
    print("=" * 70)
    print("Next: run evaluate_holdouts.py with both prompt files in `prompts`")
    print("      to compare seed-1234 vs seed-5678 on Test-IID/Test-OOD.")
    print("=" * 70)

    # Return the result from the last seed (preserves old return contract)
    return seed_results[-1][1] if seed_results else None


# ---------------------------------------------------------------------------
# Per-seed worker (extracted from main() so we can run multiple seeds)
# ---------------------------------------------------------------------------

def _run_single_seed(
    cfg: dict,
    metric: str,
    output_dir: Path,
    seed_suffix: str,
    gepa_pool: list[dict],
    test_iid: list[dict],
    test_ood: list[dict],
    objectives_in,
    backgrounds_in,
    seed_prompt: str,
):
    """Run one GEPA optimisation pass for a single split_seed.

    Writes seed-suffixed output files:
        optimised_<metric>_prompt<seed_suffix>.txt
        gepa_optimisation_history_<metric><seed_suffix>.json
        holdouts_<metric><seed_suffix>.json
    Returns (gepa_result, prompt_path, history_path).
    """
    # Lazy GEPA imports (caller already verified availability)
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig

    # Stratified train/val split on the GEPA pool
    train_data, val_data = stratified_train_val_split(
        gepa_pool,
        metric=metric,
        train_val_split=cfg["train_val_split"],
        split_seed=cfg["split_seed"],
    )

    # Cap train/val while preserving stratification (sample within each class)
    def _cap_stratified(data: list[dict], cap: int | None, seed: int) -> list[dict]:
        if not cap or len(data) <= cap:
            return data
        import random
        rng = random.Random(seed)
        label_key = f"human_{metric}"
        pos = [ex for ex in data if ex[label_key]]
        neg = [ex for ex in data if not ex[label_key]]
        n = len(data)
        n_pos_keep = max(1, round(len(pos) * cap / n))
        n_neg_keep = max(1, cap - n_pos_keep)
        # Don't ask for more than we have
        n_pos_keep = min(n_pos_keep, len(pos))
        n_neg_keep = min(n_neg_keep, len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        capped = pos[:n_pos_keep] + neg[:n_neg_keep]
        rng.shuffle(capped)
        return capped

    train_data = _cap_stratified(
        train_data, cfg["max_train_statements"], seed=cfg["split_seed"] + 1
    )
    val_data = _cap_stratified(
        val_data, cfg["max_val_statements"], seed=cfg["split_seed"] + 2
    )

    # Report final splits
    logger.info("Final split summary:")
    _summarise_split("Train",    train_data, metric)
    _summarise_split("Val",      val_data,   metric)
    _summarise_split("Test-IID", test_iid,   metric)
    _summarise_split("Test-OOD", test_ood,   metric)

    # Persist the holdouts so evaluate_human_eval_all_reviewers.py (or any
    # downstream script) can score the optimised prompt on exactly these
    # statements without re-running the split.
    holdouts_path = output_dir / f"holdouts_{metric}{seed_suffix}.json"
    _holdout_payload = {
        "metric":            metric,
        "holdout_seed":      cfg["holdout_seed"],
        "split_seed":        cfg["split_seed"],
        "test_iid_frac":     cfg["test_iid_frac"],
        "test_ood_r3_frac":  cfg["test_ood_r3_frac"],
        "train_val_split":   cfg["train_val_split"],
        "counts": {
            "train":    len(train_data),
            "val":      len(val_data),
            "test_iid": len(test_iid),
            "test_ood": len(test_ood),
        },
        # Store identifiers only — not the full action_docs_text — to keep
        # files small and to make it cheap to re-load via the existing data
        # pipeline by matching on (question, summary_model, statement).
        "splits": {
            split_name: [
                {
                    "question":      ex["question"],
                    "summary_model": ex["summary_model"],
                    "statement":     ex["statement"],
                    "reviewer":      ex["_reviewer"],
                    f"human_{metric}": ex[f"human_{metric}"],
                }
                for ex in split_data
            ]
            for split_name, split_data in [
                ("train",    train_data),
                ("val",      val_data),
                ("test_iid", test_iid),
                ("test_ood", test_ood),
            ]
        },
    }
    with open(holdouts_path, "w", encoding="utf-8") as f:
        json.dump(_holdout_payload, f, indent=2)
    logger.info("Holdout manifest saved to: %s", holdouts_path)

    # ------------------------------------------------------------------
    # Imbalance handling (per-metric, configured in PER_METRIC_OVERRIDES)
    # ------------------------------------------------------------------
    label_key = f"human_{metric}"

    # 1. Balanced-accuracy weights derived from VAL proportions.
    #    Using val (not train) here is deliberate: train will be oversampled,
    #    so its class proportions don't reflect reality. Val keeps the natural
    #    distribution and is what we ultimately want the score to reflect.
    use_balanced = cfg.get("balanced_accuracy", False)
    if use_balanced:
        n_yes_val = sum(1 for ex in val_data if ex[label_key])
        n_no_val  = len(val_data) - n_yes_val
        if n_yes_val == 0 or n_no_val == 0:
            logger.warning(
                "Cannot use balanced accuracy: val has %d Yes / %d No. "
                "Falling back to plain accuracy.",
                n_yes_val, n_no_val,
            )
            use_balanced = False
            yes_weight = no_weight = 1.0
        else:
            p_yes = n_yes_val / len(val_data)
            p_no  = n_no_val  / len(val_data)
            yes_weight = 0.5 / p_yes
            no_weight  = 0.5 / p_no
            logger.info(
                "Balanced accuracy ENABLED. Val class balance: Yes=%.1f%%, No=%.1f%%. "
                "Per-example weights: Yes=%.3f, No=%.3f. "
                "Aggregate score = balanced accuracy = 0.5*sensitivity + 0.5*specificity.",
                p_yes * 100, p_no * 100, yes_weight, no_weight,
            )
    else:
        yes_weight = no_weight = 1.0

    # 2. Oversample the minority class in TRAIN ONLY (val/test untouched).
    train_target_no_frac = cfg.get("train_no_class_target_frac")
    if train_target_no_frac:
        train_data = oversample_minority_class(
            train_data,
            metric=metric,
            target_minority_frac=train_target_no_frac,
            seed=cfg["split_seed"] + 3,
        )
        # Re-summarise post-oversampling so the log is honest about what GEPA actually sees
        _summarise_split("Train (oversampled)", train_data, metric)

    # ------------------------------------------------------------------
    # GEPA optimisation
    # ------------------------------------------------------------------
    evaluator = make_evaluator(
        cfg["judge_model"],
        cfg.get("judge_provider"),
        judge_temperature=cfg.get("judge_temperature", 0.0),
        metric=metric,
        balanced_accuracy=use_balanced,
        yes_weight=yes_weight,
        no_weight=no_weight,
    )

    logger.info("Starting GEPA optimisation...")
    logger.info("(This will make up to %d LLM evaluator calls)", cfg["max_metric_calls"])

    # GEPA uses the OPENROUTER_API_KEY via the openrouter/ prefix
    reflection_lm = f"openrouter/{cfg['gepa_reflection_model']}"
    logger.info("GEPA reflection model: %s", reflection_lm)

    objectives = {
        "faithfulness": (
            "Optimise a faithfulness evaluation prompt for LLM-as-Judge evaluation "
            "of conservation evidence summaries. The prompt should maximise agreement "
            "with human expert reviewers on binary (Yes/No) faithfulness judgements. "
            "Key challenges: reducing false positives (LLM accepts unfaithful statements) "
            "while not over-penalising vague but accurate statements (false negatives). "
            "The prompt template must contain {statement} and {all_action_docs_text} placeholders."
        ),
        "relevance": (
            "Optimise a relevance evaluation prompt for LLM-as-Judge evaluation "
            "of conservation evidence summaries. The prompt should maximise agreement "
            "with human expert reviewers on binary (Yes/No) relevance judgements. "
            "Key challenges: the judge must distinguish between statements about the "
            "correct taxa/intervention (Yes) vs statements about adjacent topics or "
            "different species (No), without being too strict about incomplete or vague "
            "statements that are on-topic. "
            "The prompt template must contain {question} and {statement} placeholders."
        ),
        "citation_correctness": (
            "Optimise a citation correctness evaluation prompt for LLM-as-Judge "
            "evaluation of conservation evidence summaries. The prompt should maximise "
            "agreement with human expert reviewers on binary (Yes/No) citation "
            "correctness judgements. The judge receives a statement and the full text "
            "of the single action document it cites, and must decide whether the "
            "statement is supported by that document. "
            "Key challenges: allowing reasonable paraphrasing while catching statements "
            "that contain specific claims absent from or contradicting the document. "
            "The prompt template must contain {statement}, {cited_action_id}, and "
            "{cited_action_text} placeholders."
        ),
    }
    backgrounds = {
        "faithfulness": (
            "The judge LLM receives a statement extracted from a conservation evidence "
            "summary and the full text of the action documents the summary was based on. "
            "It must decide whether the statement is faithfully supported by those documents. "
            "Human reviewers apply these standards: (1) omission is not unfaithfulness; "
            "(2) specific numbers and effect directions must be correct; "
            "(3) comparative claims require explicit support; "
            "(4) vague statements that are not wrong should be accepted."
        ),
        "relevance": (
            "The judge LLM receives a conservation evidence question and a statement "
            "extracted from a summary. It must decide whether the statement is relevant "
            "to answering that question. Human reviewers apply these standards: "
            "(1) the statement must address the same taxa, species, or intervention as "
            "the question — adjacent topics (e.g. moths when question asks about bats) "
            "are not relevant; (2) vague, brief, or incomplete statements that are "
            "on-topic ARE relevant; (3) absence-of-evidence statements are relevant "
            "if they concern the question's topic; (4) one side of a comparison is "
            "relevant even if the question asks for a full comparison."
        ),
        "citation_correctness": (
            "The judge LLM receives a statement extracted from a conservation evidence "
            "summary and the full text of the single action document the statement "
            "cites. It must decide whether the statement is supported by that document. "
            "Human reviewers apply these standards: (1) reasonable paraphrasing is "
            "acceptable; (2) a statement that summarises a finding clearly present in "
            "the document should be Yes; (3) absence-of-evidence statements are Yes if "
            "the document explicitly states no studies exist; (4) a statement should "
            "only be No if it makes a specific claim that is clearly absent from or "
            "contradicts the document."
        ),
    }

    # Build the GEPA config with all the tuned knobs.
    # NOTE: track_stats=True populates result.detailed_results so we can
    # inspect every candidate later if needed.
    try:
        from gepa.optimize_anything import TrackingConfig
        tracking_cfg = TrackingConfig(track_stats=True)
    except Exception:
        tracking_cfg = None  # older GEPA versions — silently skip

    engine_kwargs = dict(
        max_metric_calls=cfg["max_metric_calls"],
        candidate_selection_strategy="pareto",
        parallel=cfg.get("parallel", True),
        max_workers=cfg.get("max_workers", 8),
        seed=cfg["split_seed"],
    )
    reflection_kwargs = dict(
        reflection_lm=reflection_lm,
        reflection_minibatch_size=cfg["reflection_minibatch_size"],
        # Skip reflection on minibatches where every example is already
        # correct (saves budget). The evaluator returns 1.0 for a correct
        # binary judgement, so perfect_score=1.0.
        skip_perfect_score=True,
        perfect_score=1.0,
    )

    gepa_config_kwargs = dict(
        engine=EngineConfig(**engine_kwargs),
        reflection=ReflectionConfig(**reflection_kwargs),
    )
    if tracking_cfg is not None:
        gepa_config_kwargs["tracking"] = tracking_cfg

    result = oa.optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=evaluator,
        dataset=train_data,
        valset=val_data if val_data else None,
        objective=objectives[metric],
        background=backgrounds[metric],
        config=GEPAConfig(**gepa_config_kwargs),
    )

    # Save results
    best_prompt = result.best_candidate
    best_prompt_path = output_dir / f"optimised_{metric}_prompt{seed_suffix}.txt"
    with open(best_prompt_path, "w", encoding="utf-8") as f:
        f.write(best_prompt)
    logger.info("Best prompt saved to: %s", best_prompt_path)

    # Save history
    try:
        history_path = output_dir / f"gepa_optimisation_history_{metric}{seed_suffix}.json"
        # Extract all available metrics from GEPAResult
        val_scores = getattr(result, "val_aggregate_scores", None)
        best_idx   = getattr(result, "best_idx", None)
        candidates = getattr(result, "candidates", None)
        history = {
            "metric":                metric,
            "split_seed":            cfg["split_seed"],
            "holdout_seed":          cfg["holdout_seed"],
            "best_candidate":        best_prompt,
            "best_idx":              best_idx,
            "n_candidates":          getattr(result, "num_candidates", len(candidates) if candidates else None),
            "total_metric_calls":    getattr(result, "total_metric_calls", None),
            "seed_val_score":        val_scores[0] if val_scores else None,
            "best_val_score":        val_scores[best_idx] if (val_scores and best_idx is not None) else None,
            "all_val_scores":        list(val_scores) if val_scores else None,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("History saved to: %s", history_path)
    except Exception as e:
        logger.warning("Could not save history: %s", e)

    # Print summary
    val_scores = getattr(result, "val_aggregate_scores", None)
    best_idx   = getattr(result, "best_idx", None)
    n_candidates = getattr(result, "num_candidates", "?")
    total_calls  = getattr(result, "total_metric_calls", "?")

    print("\n" + "=" * 65)
    print("OPTIMISATION COMPLETE — split_seed=%d" % cfg["split_seed"])
    print("=" * 65)
    print(f"Candidates explored:  {n_candidates}")
    print(f"Total metric calls:   {total_calls}")
    if val_scores:
        seed_score = val_scores[0]
        best_score = val_scores[best_idx] if best_idx is not None else None
        print(f"Seed prompt val score:  {seed_score:.3f} ({seed_score:.1%} agreement)")
        if best_score is not None:
            improvement = best_score - seed_score
            arrow = "↑" if improvement > 0.005 else ("↓" if improvement < -0.005 else "→")
            print(f"Best prompt val score:  {best_score:.3f} ({best_score:.1%} agreement)  {arrow} {improvement:+.3f}")
        if best_idx == 0:
            print("NOTE: Best candidate is the seed prompt — GEPA found no improvement.")
            print("      Try increasing max_metric_calls or reflection_minibatch_size,")
            print("      or check that the train set contains enough minority-class examples.")
    print(f"Output:           {best_prompt_path}")
    print(f"Holdout manifest: {holdouts_path}")
    print()
    print("--- Best optimised prompt (first 500 chars) ---")
    print(best_prompt[:500] + ("..." if len(best_prompt) > 500 else ""))
    print("=" * 70)

    return result, best_prompt_path, history_path


if __name__ == "__main__":
    main()
