"""
Utility functions for reading and writing evaluation results.

Results are stored as JSON (list of record dicts) to allow incremental
saving and resumability.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_results(filepath: str | Path) -> list[dict]:
    """
    Load existing results from a JSON file.
    Returns an empty list if the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning(
                "Results file %s contains %s, not a list. Starting fresh.",
                filepath,
                type(data).__name__,
            )
            return []
        logger.info("Loaded %d existing results from %s", len(data), filepath)
        return data
    except json.JSONDecodeError as e:
        logger.error("JSON decode error loading %s: %s. Starting fresh.", filepath, e)
        return []


def save_results(results: list[dict], filepath: str | Path) -> None:
    """
    Save results list to a JSON file, creating directories as needed.
    Writes atomically (to a temp file, then renames) to avoid corruption.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = filepath.with_suffix(".tmp.json")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, filepath)
        logger.debug("Saved %d results to %s", len(results), filepath)
    except Exception as e:
        logger.error("Error saving results to %s: %s", filepath, e)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def results_as_unique_key(result: dict) -> str:
    """
    Build a string key that uniquely identifies a statement-level result.
    Used to detect already-evaluated records and skip them (resumability).
    """
    return (
        f"{result.get('synopsis','')}"
        f"|{result.get('question','')}"
        f"|{result.get('summary_model','')}"
        f"|{result.get('statement','')}"
    )


def _is_successfully_scored(result: dict) -> bool:
    """
    Return True only if the record has real LLM verdicts (not null/error).

    A record saved with verdict=null means the LLM call failed after all
    retries — it should be retried on the next run, not skipped.
    Auto-filled citation_correctness (no citation → auto-No) counts as
    successfully scored since no LLM call is needed.
    """
    metrics_to_check = ["llm_relevance", "llm_citation_correctness", "llm_faithfulness"]

    for key in metrics_to_check:
        entry = result.get(key)
        if entry is None:
            # Metric not yet attempted at all
            return False
        # auto_filled or skipped_no_citation means deliberately skipped — counts as done
        if entry.get("auto_filled") or entry.get("skipped_no_citation"):
            continue
        # A real LLM call must have returned a non-null verdict
        if entry.get("verdict") is None:
            return False

    # citation_correctness_when_cited is derived for free from cc — if cc is
    # done, this is also done. No need to check separately.
    return True


def build_done_set(results: list[dict]) -> set[str]:
    """
    Return the set of unique keys for successfully completed results.

    Records with null verdicts (failed LLM calls) are excluded from the
    done set so they will be retried on the next run.
    """
    done = set()
    skipped = 0
    for r in results:
        if _is_successfully_scored(r):
            done.add(results_as_unique_key(r))
        else:
            skipped += 1
    if skipped:
        logger.info(
            "%d records with null/error verdicts will be retried this run.",
            skipped,
        )
    return done
