"""
Human evaluation data loader.

Parses the Excel spreadsheet used by human evaluators into a list of
EvaluationRecord objects, each representing one statement to be judged.

Excel layout (one row per CE link, first row in a group carries all metadata):

  Col  | Name                | Notes
  -----|---------------------|----------------------------------------------
  A    | Synopsis            | Conservation topic (new group when non-null)
  B    | Answerability       | 'answerable' / 'unanswerable'
  C    | Question            | The question the summary was generated for
  D    | Summary             | Full summary text
  E    | Statement           | The extracted statement to evaluate
  F    | Relevant IDs        | Comma/semicolon-separated summary_action_ids
  G    | CE Link             | https://www.conservationevidence.com/actions/{id}
  H    | Summary Model       | Model that generated the summary
  I    | Cited IDs           | The specific action ID cited for this statement
  J    | Relevance           | Human score (Yes/No/None)
  K    | Citation Correctness| Human score (Yes/No/None)
  L    | Faithfulness        | Human score (Yes/No/None)
  M    | Comments            | Free text
  N    | Reviewer            | Primary reviewer
  O    | Second reviewer     | Secondary reviewer

The CE links column may have multiple rows per statement (one per linked
action). The first row in a group carries all metadata; subsequent rows
only have a CE link.

Normalisation notes
-------------------
* Excel encodes carriage returns as the literal string ``_x000D_`` followed
  by a newline (or alone).  These are collapsed back to ``\n``.
* Summaries in Excel sometimes have extra blank lines not present in the
  source JSON (e.g. ``\n\n\n- item`` vs ``\n- item``).  The function
  ``_normalise_for_matching()`` collapses all runs of 2+ newlines down to
  a single newline for the purposes of key-matching only (display text is
  kept as-is after basic CR cleanup).
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openpyxl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRecord:
    """
    One statement from the human evaluation dataset.
    Ready to be passed to the LLM judge.
    """
    synopsis: str
    answerability: str
    question: str
    summary: str
    statement: str
    summary_model: str

    # Action IDs the summary was based on (for faithfulness context)
    summary_action_ids: list[str]

    # CE links listed for this statement (all relevant action IDs)
    ce_action_ids: list[str]

    # The single action ID cited in-text for this statement (may be None)
    cited_action_id: Optional[str]

    # Human scores (may be None if not yet scored)
    human_relevance: Optional[bool]
    human_citation_correctness: Optional[bool]
    human_faithfulness: Optional[bool]

    reviewer: Optional[str] = None
    second_reviewer: Optional[str] = None

    # All cited action IDs (may be multiple, e.g. "1399, 1397" in the Excel)
    # The primary cited_action_id is cited_action_ids[0] for backwards compatibility.
    cited_action_ids: list = field(default_factory=list)

    # Which Excel file this record came from (populated by load_human_eval_records)
    source_file: Optional[str] = field(default=None)

    # Will be populated after LLM evaluation
    llm_relevance: Optional[dict] = field(default=None)
    llm_citation_correctness: Optional[dict] = field(default=None)
    llm_faithfulness: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "synopsis": self.synopsis,
            "answerability": self.answerability,
            "question": self.question,
            "summary": self.summary,
            "statement": self.statement,
            "summary_model": self.summary_model,
            "summary_action_ids": self.summary_action_ids,
            "ce_action_ids": self.ce_action_ids,
            "cited_action_id": self.cited_action_id,
            "cited_action_ids": self.cited_action_ids,
            "human_relevance": self.human_relevance,
            "human_citation_correctness": self.human_citation_correctness,
            "human_faithfulness": self.human_faithfulness,
            "reviewer": self.reviewer,
            "second_reviewer": self.second_reviewer,
            "source_file": self.source_file,
            "llm_relevance": self.llm_relevance,
            "llm_citation_correctness": self.llm_citation_correctness,
            "llm_faithfulness": self.llm_faithfulness,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_summary(s: Optional[str]) -> Optional[str]:
    """
    Normalise Excel carriage-return artefacts.

    Steps applied (for display / storage):
      1. Replace the literal ``_x000D_\n`` sequence with ``\n``.
      2. Replace any remaining ``_x000D_`` (lone CR placeholder) with ``\n``.
      3. Collapse runs of 3+ consecutive newlines down to 2 (preserve
         intentional paragraph breaks while removing accidental extras).
    """
    if s is None:
        return None
    s = s.replace("_x000D_\n", "\n").replace("_x000D_", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _normalise_for_matching(s: Optional[str]) -> Optional[str]:
    """
    Aggressively normalise a summary string for the sole purpose of
    key-matching against the JSON source data.

    The Excel version of some summaries has extra blank lines that are absent
    in the JSON (e.g. a double newline before a markdown bullet ``- ``
    becomes a triple newline in Excel).  Collapsing ALL runs of 2+ newlines
    to a single newline ensures these cases match correctly.

    This function should ONLY be used for building lookup keys — never for
    storing or displaying the actual summary text.
    """
    if s is None:
        return None
    # First apply the standard CR cleanup
    s = s.replace("_x000D_\n", "\n").replace("_x000D_", "\n")
    # Collapse ALL multi-newline runs to a single newline
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _parse_yes_no(value) -> Optional[bool]:
    """Convert a cell value (Yes/No/None) to bool or None."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _extract_action_id(ce_link: str) -> Optional[str]:
    """Extract numeric action ID from a CE URL."""
    match = re.search(r"/actions/(\d+)", ce_link)
    return match.group(1) if match else None


def _parse_id_list(raw: Optional[str]) -> list[str]:
    """
    Parse a delimited string of action IDs into a clean list.
    Handles comma, semicolon, and space separators.
    Ignores 'None', empty strings, and non-numeric entries.
    """
    if raw is None or str(raw).strip().lower() in ("none", ""):
        return []
    ids = re.findall(r"\d+", str(raw))
    return ids


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_human_eval_records(
    filepath: str | Path,
    answerable_only: bool = True,
) -> list[EvaluationRecord]:
    """
    Parse the human evaluation Excel file and return a list of EvaluationRecord.

    Parameters
    ----------
    filepath : str | Path
        Path to the .xlsx file.
    answerable_only : bool, default True
        When True, only records whose ``Answerability`` column equals
        ``'answerable'`` (case-insensitive) are returned.  Set to False
        to include unanswerable rows as well.

    Returns
    -------
    list[EvaluationRecord]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Human eval Excel file not found: {filepath}")

    wb = openpyxl.load_workbook(str(filepath), data_only=True)
    ws = wb.active

    headers = [cell.value for cell in ws[1]]
    logger.debug("Excel headers: %s", headers)

    # Column indices (0-based) – resolved from headers for robustness
    def col(name: str) -> int:
        try:
            return headers.index(name)
        except ValueError:
            raise ValueError(
                f"Expected column '{name}' not found in Excel. "
                f"Available columns: {headers}"
            )

    col_synopsis = col("Synopsis")
    col_answerability = col("Answerability")
    col_question = col("Question")
    col_summary = col("Summary")
    col_statement = col("Statement")
    col_relevant_ids = col("Relevant IDs")
    col_ce_link = col("CE Link")
    col_model = col("Summary Model")
    col_cited_ids = col("Cited IDs")
    col_relevance = col("Relevance")
    col_citation = col("Citation Correctness")
    col_faithfulness = col("Faithfulness")
    col_reviewer = col("Reviewer")
    col_second_reviewer = col("Second reviewer")

    records: list[EvaluationRecord] = []
    current: Optional[EvaluationRecord] = None

    for row in ws.iter_rows(min_row=2, values_only=True):
        synopsis = row[col_synopsis]

        if synopsis is not None:
            # --- New statement group starts ---
            if current is not None:
                records.append(current)

            ce_link = row[col_ce_link]
            first_ce_action_id = _extract_action_id(str(ce_link)) if ce_link else None

            # Parse cited_id – may be a single ID or multiple (e.g. "1399, 1397")
            cited_raw = row[col_cited_ids]
            cited_ids_parsed = _parse_id_list(cited_raw)
            # Keep first ID as primary for backwards compatibility,
            # but store all IDs for multi-citation CC checking.
            cited_action_id = cited_ids_parsed[0] if cited_ids_parsed else None
            cited_action_ids_all = cited_ids_parsed  # all cited IDs

            current = EvaluationRecord(
                synopsis=str(synopsis).strip(),
                answerability=str(row[col_answerability] or "").strip(),
                question=str(row[col_question] or "").strip(),
                summary=_normalise_summary(row[col_summary]),
                statement=str(row[col_statement] or "").strip(),
                summary_model=str(row[col_model] or "").strip(),
                summary_action_ids=_parse_id_list(row[col_relevant_ids]),
                ce_action_ids=[first_ce_action_id] if first_ce_action_id else [],
                cited_action_id=cited_action_id,
                cited_action_ids=cited_action_ids_all,
                human_relevance=_parse_yes_no(row[col_relevance]),
                human_citation_correctness=_parse_yes_no(row[col_citation]),
                human_faithfulness=_parse_yes_no(row[col_faithfulness]),
                reviewer=row[col_reviewer],
                second_reviewer=row[col_second_reviewer],
                source_file=filepath.name,
            )

        else:
            # --- Continuation row: additional CE link ---
            ce_link = row[col_ce_link]
            if ce_link and current is not None:
                action_id = _extract_action_id(str(ce_link))
                if action_id and action_id not in current.ce_action_ids:
                    current.ce_action_ids.append(action_id)

    # Don't forget the last group
    if current is not None:
        records.append(current)

    if answerable_only:
        before = len(records)
        records = [
            r for r in records
            if r.answerability.lower() == "answerable"
        ]
        skipped = before - len(records)
        if skipped:
            logger.info(
                "Filtered out %d unanswerable records from %s (%d answerable remain).",
                skipped,
                filepath.name,
                len(records),
            )

    logger.info(
        "Loaded %d evaluation records from %s", len(records), filepath.name
    )
    return records


# ---------------------------------------------------------------------------
# Public normalisation alias
# ---------------------------------------------------------------------------

def normalise_for_matching(s: Optional[str]) -> Optional[str]:
    """
    Public alias for :func:`_normalise_for_matching`.

    Aggressively collapse all multi-newline runs for the purpose of matching
    summary text from Excel against the source JSON data.  Do **not** use
    this for storing or displaying summary text — use
    :func:`_normalise_summary` for that.
    """
    return _normalise_for_matching(s)
