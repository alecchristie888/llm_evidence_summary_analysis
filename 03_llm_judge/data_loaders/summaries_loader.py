"""
Loader for the stmts_and_cited_stmts JSON data.

These files contain pre-generated statements with extracted citations, produced
by the student's pipeline. Each file contains a list of entries like:

  {
    "question_details": {
      "query": "...",
      "all_relevant_qu_ids": ["756", ...],
      "regenerated_qu_ids": [...]
    },
    "summary_details": {
      "summary_model": "openai/gpt-5",
      "summary_provider": null,
      "relevant_summary": "...",
      "summary_action_ids": ["842", ...],
      "summary_statements_model": "gemini-2.5-pro",
      "summary_statements": ["...", ...],
      "cited_statements_model": "...",
      "cited_statements": [
          {"statement": "...", "citations": ["842"]},
          ...
      ]
    }
  }

This module provides a lookup function: given a question + summary text,
return the list of cited_statements for that entry.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise(s: Optional[str]) -> Optional[str]:
    """
    Aggressively normalise a string for use as a lookup key.

    Collapses ALL runs of 2+ consecutive newlines to a single newline so that
    minor whitespace differences between the Excel source and the JSON source
    (e.g. ``\n\n- item`` in JSON vs ``\n\n\n- item`` in Excel) do not prevent
    a match.  This is used for key-building only.
    """
    if s is None:
        return None
    s = s.replace("_x000D_\n", "\n").replace("_x000D_", "\n")
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Model-to-folder mapping
# ---------------------------------------------------------------------------

# Maps the model string used in the Excel (as reported in summary_model field)
# to the folder name in the summaries directory.
MODEL_TO_FOLDER: dict[str, str] = {
    "anthropic/claude-sonnet-4": "_claude-sonnet-4",
    "google/gemini-2.5-pro": "_gemini-2-5-pro",
    "openai/gpt-5": "_gpt-5",
    "moonshotai/kimi-k2-0905": "fireworks_kimi-k2-0905",
}


class SummariesIndex:
    """
    Indexes all stmts_and_cited_stmts JSON files for fast lookup.

    Parameters
    ----------
    summaries_base_dir : str | Path
        Path to the directory containing the model sub-directories
        (e.g. `answerable_passed_qus_summaries/hybrid_cross-encoder/`).
    model_to_folder : dict | None
        Optional override for the model-to-folder mapping.
    """

    def __init__(
        self,
        summaries_base_dir: str | Path,
        model_to_folder: Optional[dict[str, str]] = None,
    ):
        self.base_dir = Path(summaries_base_dir)
        self.model_to_folder = model_to_folder or MODEL_TO_FOLDER
        # Index: (question, normalised_summary) → entry dict
        self._index: dict[tuple[str, str], dict] = {}
        self._loaded_models: set[str] = set()

    def _load_model(self, model_folder: str) -> None:
        if model_folder in self._loaded_models:
            return
        folder_path = self.base_dir / model_folder / "stmts_and_cited_stmts"
        if not folder_path.exists():
            logger.warning(
                "stmts_and_cited_stmts directory not found for model folder '%s': %s",
                model_folder,
                folder_path,
            )
            self._loaded_models.add(model_folder)
            return

        count = 0
        for fpath in sorted(folder_path.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data:
                    q = entry["question_details"]["query"]
                    raw_summary = entry["summary_details"].get("relevant_summary")

                    # Some entries store relevant_summary as a dict, e.g.
                    # {"erroneous_summary": "..."}  — extract the text value.
                    # The Excel file stores the Python repr of this dict as a
                    # plain string, so we index *both* the extracted text and
                    # the repr string to maximise matching coverage.
                    if isinstance(raw_summary, dict):
                        # Index using the repr string (matches Excel cell value)
                        repr_key = _normalise(str(raw_summary))
                        if q and repr_key:
                            self._index[(q, repr_key)] = entry
                            count += 1
                        # Also index using the inner text value if available
                        inner_text = next(iter(raw_summary.values()), None)
                        if inner_text and isinstance(inner_text, str):
                            s = _normalise(inner_text)
                            if q and s:
                                self._index[(q, s)] = entry
                                count += 1
                    else:
                        s = _normalise(raw_summary)
                        if q and s:
                            self._index[(q, s)] = entry
                            count += 1
            except Exception as e:
                logger.error("Error loading summaries file %s: %s", fpath, e)

        logger.info(
            "Indexed %d summaries from %s", count, folder_path
        )
        self._loaded_models.add(model_folder)

    def get_cited_statements(
        self,
        question: str,
        summary: str,
        model: str,
    ) -> Optional[list[dict]]:
        """
        Return the list of cited_statements for the given question/summary/model.

        Each item in the returned list has the form:
            {"statement": str, "citations": [str, ...]}

        Returns None if not found or if cited_statements field is missing.
        """
        model_folder = self.model_to_folder.get(model)
        if model_folder is None:
            logger.warning(
                "Unknown model '%s', not in model_to_folder mapping. "
                "Known models: %s",
                model,
                list(self.model_to_folder.keys()),
            )
            return None

        self._load_model(model_folder)

        norm_summary = _normalise(summary)
        entry = self._index.get((question, norm_summary))
        if entry is None:
            logger.warning(
                "No indexed entry found for question='%s...' / model='%s'",
                question[:60],
                model,
            )
            return None

        cited = entry["summary_details"].get("cited_statements")
        if not cited:
            logger.warning(
                "Entry found but 'cited_statements' is empty/missing for "
                "question='%s...'",
                question[:60],
            )
        return cited

    def get_entry(
        self, question: str, summary: str, model: str
    ) -> Optional[dict]:
        """Return the full indexed entry dict."""
        model_folder = self.model_to_folder.get(model)
        if not model_folder:
            return None
        self._load_model(model_folder)
        norm_summary = _normalise(summary)
        return self._index.get((question, norm_summary))
