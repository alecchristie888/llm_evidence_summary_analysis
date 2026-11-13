"""
LLM-as-Judge: Full dataset evaluation runner.

Runs the judge over ALL summaries across all candidate models, across all
synopsis topics, for the 'answerable / passed' subset.

This script mirrors the validated setup from
``evaluate_human_eval_all_reviewers.py`` so that scores produced here are
directly comparable to (and use the same prompts/model as) the human-eval
benchmark. Key parity features:

  * Same judge model + provider + temperature.
  * Custom prompt templates for each metric (relevance / citation correctness
    / faithfulness) loaded from .txt files, identical to the human-eval run.
  * Multi-citation handling: if a statement has multiple citations, each is
    checked and the verdict is Yes if any cited action supports it.
  * ``llm_citation_correctness_when_cited`` is populated alongside
    ``llm_citation_correctness`` so downstream aggregation works without
    fallback derivation.
  * ``synopsis`` is populated on every output record, derived from the input
    filename (e.g. ``bg_km_AmphibianConservation_cited_statements.json``
    → ``AmphibianConservation``) so ``aggregate_summary_scores.py`` groups
    correctly across topics.
  * Resumable: existing results are loaded and skipped per file.
  * Periodic incremental save (every N statements) and final save per file.
  * End-of-run summary printout.

Usage
-----
    python evaluate_full_dataset.py

Configure the settings in ``CONFIG`` below before running.

Output structure
----------------
    <output_dir>/
      <model_folder>/
        <synopsis>_llm_scores.json

Each file is a list of dicts, one per cited statement, with all requested
metric scores attached.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from data_loaders.action_loader import ActionLoader
from judge import metrics as judge_metrics
from utils.io import load_results, save_results


# ===========================================================================
# CONFIGURATION — mirror evaluate_human_eval_all_reviewers.py where possible
# ===========================================================================

CONFIG = {
    # ---------------- Input ----------------
    # Root directory of the summaries data (per-model subfolders below it).
    "summaries_base_dir": "answerable_passed_qus_summaries/",

    # Which model folders to evaluate.
    "model_folders": [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905",
    ],

    # ---------------- Action documents ----------------
    "action_local_dir": "data/actions/background_key_messages/bg_km_all",
    "action_fetch_from_web": False,

    # ---------------- Output ----------------
    # One <synopsis>_llm_scores.json per file is written under
    # <output_dir>/<model_folder>/. The judge_model is encoded into the
    # subdirectory name so multiple judge runs don't overwrite each other.
    "output_dir": "llm_judge_results/full_dataset/gemini-3.1-pro-preview-finalprompts",

    # ---------------- Judge model (mirrors human-eval CONFIG) ----------------
    "judge_model": "google/gemini-3.1-pro-preview",
    "judge_provider": "Google",
    "judge_temperature": 0.0,
    "max_retries": 3,

    # ---------------- Custom prompt templates (mirrors human-eval CONFIG) ----
    # If a path is None the built-in default in judge.prompts is used.
    "relevance_prompt_file":            "prompt_files/relevance/manual_relevance_v7.txt",
    "citation_correctness_prompt_file": "prompt_files/citation_correctness/manual_citation_correctness_v17.txt",
    "faithfulness_prompt_file":         "prompt_files/faithfulness/manual_faithfulness_v14-concat.txt",

    # ---------------- Run control ----------------
    # Maximum statements per cited-statements input file (None = all).
    # Useful for smoke-testing: e.g. set to 5 to evaluate 5 statements per
    # synopsis-file, ~20 files * 4 models = ~400 calls.
    "max_statements_per_file": None,

    # Seconds to wait between LLM calls (rate-limit guard).
    "delay_between_statements": 0.1,

    # Save the results JSON every N evaluated statements (full-file save also
    # happens at the end of every input file).
    "save_every_n": 5,

    # Which metrics to evaluate.
    "metrics_to_run": ["relevance", "citation_correctness", "faithfulness"],
}

# ===========================================================================


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNOPSIS_RE = re.compile(r"^bg_km_(?P<synopsis>.+?)_cited_statements$")


def synopsis_from_filename(fpath: Path) -> str:
    """
    Extract the synopsis name from an input filename.

    e.g. ``bg_km_AmphibianConservation_cited_statements.json`` →
    ``AmphibianConservation``.

    Falls back to the file stem if the expected pattern is not present.
    """
    stem = fpath.stem
    m = _SYNOPSIS_RE.match(stem)
    if m:
        return m.group("synopsis")
    # Fallback: strip a trailing _cited_statements if present
    return stem.replace("_cited_statements", "").replace("bg_km_", "")


def load_custom_prompts(config: dict) -> dict[str, str]:
    """Load .txt prompt templates for each metric (where configured)."""
    custom_prompts: dict[str, str] = {}
    keys = {
        "relevance":            config.get("relevance_prompt_file"),
        "citation_correctness": config.get("citation_correctness_prompt_file"),
        "faithfulness":         config.get("faithfulness_prompt_file"),
    }
    for metric_name, fpath in keys.items():
        if fpath:
            template_text = Path(fpath).read_text(encoding="utf-8")
            custom_prompts[metric_name] = template_text
            logger.info(
                "  Loaded custom %s prompt from %s (%d chars)",
                metric_name, fpath, len(template_text),
            )
    return custom_prompts


# ---------------------------------------------------------------------------
# Per-statement evaluation (mirrors human-eval evaluate_record)
# ---------------------------------------------------------------------------

def _evaluate_statement(
    *,
    synopsis: str,
    question: str,
    summary: str,
    summary_model: str,
    summary_action_ids: list[str],
    statement: str,
    citations: list[str],
    action_loader: ActionLoader,
    judge_model: str,
    judge_provider: Optional[str],
    judge_temperature: float,
    metrics_to_run: list[str],
    max_retries: int,
    custom_prompts: dict[str, str],
) -> dict:
    """Run all requested metric evaluations for a single statement."""

    # Normalise citations list (some entries may have stringy ids, ints, etc.)
    citations_list = [str(c) for c in (citations or []) if c is not None and str(c) != ""]
    cited_action_id = citations_list[0] if citations_list else None

    relevance_template = custom_prompts.get("relevance")
    citation_template = custom_prompts.get("citation_correctness")
    faithfulness_template = custom_prompts.get("faithfulness")

    result: dict = {
        "synopsis": synopsis,
        "question": question,
        "summary": summary,
        "summary_model": summary_model,
        "summary_action_ids": list(summary_action_ids or []),
        "ce_action_ids": list(citations_list),
        "cited_action_id": cited_action_id,
        "cited_action_ids": list(citations_list),
        "statement": statement,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
    }

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

    # ---- 2. Citation correctness ----
    if "citation_correctness" in metrics_to_run:
        all_cited_ids = list(citations_list)

        logger.info(
            "  [citation_correctness] cited_ids=%s for '%s...'",
            all_cited_ids, statement[:60],
        )

        if not all_cited_ids:
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
            cc.get("verdict"), cc.get("auto_filled", False),
        )

        # Mirror human-eval: derive citation_correctness_when_cited
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
        ids_for_faith = list(summary_action_ids or [])
        if cited_action_id and cited_action_id not in ids_for_faith:
            ids_for_faith.append(cited_action_id)

        all_docs_text = action_loader.get_action_texts_combined(ids_for_faith)
        if all_docs_text is None:
            logger.warning(
                "  [faithfulness] No action documents available for IDs %s. Skipping.",
                ids_for_faith,
            )
            result["llm_faithfulness"] = {
                "metric_name": "faithfulness",
                "statement": statement,
                "verdict": None,
                "reasoning": None,
                "error": f"No action docs available for IDs: {ids_for_faith}",
            }
        else:
            logger.info(
                "  [faithfulness] Using %d action IDs (concatenated) for '%s...'",
                len(ids_for_faith), statement[:60],
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
# Per-file runner
# ---------------------------------------------------------------------------

def evaluate_cited_statements_file(
    fpath: Path,
    *,
    synopsis: str,
    action_loader: ActionLoader,
    judge_model: str,
    judge_provider: Optional[str],
    judge_temperature: float,
    metrics_to_run: list[str],
    max_retries: int,
    delay: float,
    output_path: Path,
    max_statements: Optional[int],
    save_every_n: int,
    custom_prompts: dict[str, str],
) -> int:
    """Evaluate all cited_statements in one input JSON file. Returns count newly evaluated."""
    with open(fpath, "r", encoding="utf-8") as f:
        entries = json.load(f)

    existing = load_results(output_path)
    # Resumability key: (synopsis, question, summary_model, statement)
    done_set = {
        f"{r.get('synopsis','')}|{r.get('question','')}|{r.get('summary_model','')}|{r.get('statement','')}"
        for r in existing
    }

    results = list(existing)
    count_new = 0
    count_in_file = 0
    save_path = output_path

    for entry in entries:
        question = entry["question_details"]["query"]
        summary = entry["summary_details"].get("relevant_summary", "")
        summary_model = entry["summary_details"].get("summary_model", "")
        summary_action_ids = entry["summary_details"].get("summary_action_ids", [])
        cited_statements = entry["summary_details"].get("cited_statements", [])

        for cs in cited_statements:
            if max_statements is not None and count_new >= max_statements:
                save_results(results, save_path)
                logger.info(
                    "  Hit max_statements_per_file=%d for %s — stopping this file.",
                    max_statements, fpath.name,
                )
                return count_new

            statement = cs.get("statement", "")
            citations = cs.get("citations", [])

            key = f"{synopsis}|{question}|{summary_model}|{statement}"
            count_in_file += 1
            if key in done_set:
                continue

            logger.info(
                "  [%d] %s | %s | '%s...'",
                count_in_file, synopsis, summary_model, statement[:50],
            )

            result = _evaluate_statement(
                synopsis=synopsis,
                question=question,
                summary=summary,
                summary_model=summary_model,
                summary_action_ids=summary_action_ids,
                statement=statement,
                citations=citations,
                action_loader=action_loader,
                judge_model=judge_model,
                judge_provider=judge_provider,
                judge_temperature=judge_temperature,
                metrics_to_run=metrics_to_run,
                max_retries=max_retries,
                custom_prompts=custom_prompts,
            )

            results.append(result)
            done_set.add(key)
            count_new += 1

            if save_every_n > 0 and count_new % save_every_n == 0:
                save_results(results, save_path)
                logger.info("  Saved checkpoint after %d new statements.", count_new)

            if delay > 0:
                time.sleep(delay)

    # Final save for the file
    save_results(results, save_path)
    logger.info("File %s: evaluated %d new statements (file total now %d).",
                fpath.name, count_new, len(results))
    return count_new


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_run_summary(output_dir: Path, model_folders: list[str]) -> None:
    """Print a concise per-model verdict summary across all output files."""
    metric_keys = {
        "relevance": "llm_relevance",
        "citation_correctness": "llm_citation_correctness",
        "faithfulness": "llm_faithfulness",
    }

    print("\n" + "=" * 75)
    print("LLM-AS-JUDGE EVALUATION SUMMARY — FULL DATASET")
    print("=" * 75)

    for model_folder in model_folders:
        mdir = output_dir / model_folder
        if not mdir.exists():
            continue
        all_results: list[dict] = []
        for jf in sorted(mdir.glob("*_llm_scores.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
            except Exception as e:
                logger.warning("Could not read %s: %s", jf, e)

        if not all_results:
            continue

        print(f"\nModel: {model_folder}  ({len(all_results)} statements)")
        for metric, key in metric_keys.items():
            verdicts = [
                r[key]["verdict"]
                for r in all_results
                if key in r and r[key] and r[key].get("verdict") is not None
            ]
            if not verdicts:
                print(f"  {metric:25s}  N/A")
                continue
            n_yes = sum(1 for v in verdicts if v is True)
            print(f"  {metric:25s}  {n_yes:>4}/{len(verdicts):<4} ({n_yes/len(verdicts):.1%})")

    print("=" * 75)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_full_evaluation(config: dict = CONFIG) -> None:
    _setup_logging()

    base_dir = Path(config["summaries_base_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_model = config["judge_model"]
    judge_provider = config.get("judge_provider")
    judge_temperature = config.get("judge_temperature", 0.0)
    metrics_to_run = config.get(
        "metrics_to_run", ["relevance", "citation_correctness", "faithfulness"]
    )
    max_retries = config.get("max_retries", 3)
    max_statements_per_file = config.get("max_statements_per_file")
    delay = config.get("delay_between_statements", 0.1)
    save_every_n = config.get("save_every_n", 5)

    logger.info("=" * 70)
    logger.info("LLM-AS-JUDGE EVALUATION — FULL DATASET")
    logger.info("Judge model : %s  (provider: %s, temp: %s)",
                judge_model, judge_provider, judge_temperature)
    logger.info("Metrics     : %s", metrics_to_run)
    logger.info("Output dir  : %s", output_dir)
    if max_statements_per_file is not None:
        logger.info("Smoke-test mode: max_statements_per_file=%d", max_statements_per_file)
    logger.info("=" * 70)

    # Custom prompts (must match human-eval setup)
    custom_prompts = load_custom_prompts(config)

    action_loader = ActionLoader(
        local_base_dir=config.get("action_local_dir"),
        fetch_from_web=config.get("action_fetch_from_web", False),
    )

    grand_total = 0
    for model_folder in config["model_folders"]:
        sac_dir = base_dir / model_folder / "stmts_and_cited_stmts"
        if not sac_dir.exists():
            logger.warning("Directory not found: %s", sac_dir)
            continue

        model_output_dir = output_dir / model_folder
        model_output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(sac_dir.glob("*.json"))
        logger.info("Model %s: processing %d files.", model_folder, len(json_files))

        for fpath in json_files:
            synopsis = synopsis_from_filename(fpath)
            out_name = f"{synopsis}_llm_scores.json"
            output_path = model_output_dir / out_name
            logger.info("  File: %s → %s  (synopsis=%s)",
                        fpath.name, output_path.name, synopsis)

            try:
                n_new = evaluate_cited_statements_file(
                    fpath=fpath,
                    synopsis=synopsis,
                    action_loader=action_loader,
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    judge_temperature=judge_temperature,
                    metrics_to_run=metrics_to_run,
                    max_retries=max_retries,
                    delay=delay,
                    output_path=output_path,
                    max_statements=max_statements_per_file,
                    save_every_n=save_every_n,
                    custom_prompts=custom_prompts,
                )
                grand_total += n_new
            except Exception as e:
                logger.exception("Error while processing %s: %s", fpath.name, e)
                # continue to next file rather than abort the whole run

    logger.info("Full dataset evaluation complete. Newly evaluated this run: %d",
                grand_total)
    print_run_summary(output_dir, config["model_folders"])


if __name__ == "__main__":
    run_full_evaluation(CONFIG)
