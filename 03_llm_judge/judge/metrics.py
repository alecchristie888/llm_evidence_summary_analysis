"""
Individual metric evaluators for the LLM-as-Judge pipeline.

Each function mirrors the human evaluation workflow:

  1. relevance()                    – binary: is the statement relevant to the question?
  2. citation_correctness()         – binary: is the statement supported by its cited doc?
  3. faithfulness()                 – binary: is the statement supported by any of the
                                      action docs the summary was built from?
                                      (concatenated mode — single LLM call)
  4. faithfulness_per_document()    – same verdict logic but evaluates each action document
                                      independently and returns Yes if ANY single document
                                      supports the statement on its own.
                                      Stores per-document results in llm_faithfulness_per_doc.

All return a dict with:
  {
      "verdict": True | False,
      "reasoning": str,
      "statement": str,
      ... (metric-specific extra fields)
  }

A verdict of None means the call failed and should be treated as missing data.
"""

import logging
from typing import Any

from .llm_client import call_judge_llm
from .prompts import (
    _render_custom_template,
    build_citation_correctness_prompt,
    build_faithfulness_prompt,
    build_faithfulness_single_doc_prompt,
    build_relevance_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Schemas for structured outputs
# ---------------------------------------------------------------------------

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["Yes", "No"]},
    },
    "required": ["statement", "reasoning", "verdict"],
    "additionalProperties": False,
}

# Extended schema for faithfulness.
# NOTE: Azure / OpenAI strict structured-outputs require every key in `properties`
# to also appear in `required`. To keep `confidence` and `claims` effectively
# optional we declare them as nullable (`["<type>", "null"]`) so the model can
# emit `null` (or an empty array) when the prompt does not request them.
_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "claims": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_text":          {"type": "string"},
                            "supporting_evidence": {"type": "string"},
                            "support_assessment":  {
                                "type": "string",
                                "enum": ["fully_supported", "not_fully_supported"],
                            },
                            "failure_reasons": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "wrong_quantity",
                                        "wrong_effect_direction_or_magnitude",
                                        "unsupported_comparative_claim",
                                        "mismerged_studies",
                                        "overgeneralisation",
                                        "absent_or_contradicted_fact",
                                        "unsupported_evidence_summary",
                                    ],
                                },
                            },
                        },
                        "required": ["claim_text", "supporting_evidence",
                                     "support_assessment", "failure_reasons"],
                        "additionalProperties": False,
                    },
                },
                {"type": "null"},
            ],
        },
        "reasoning":  {"type": "string"},
        "verdict":    {"type": "string", "enum": ["Yes", "No"]},
        # Nullable so the model can omit it (emit null) when not requested
        "confidence": {
            "anyOf": [
                {"type": "string", "enum": ["high", "medium", "low"]},
                {"type": "null"},
            ],
        },
    },
    "required": ["statement", "claims", "reasoning", "verdict", "confidence"],
    "additionalProperties": False,
}

_CITATION_CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "cited_action_id": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["Yes", "No"]},
    },
    "required": ["statement", "cited_action_id", "reasoning", "verdict"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_verdict(raw: str) -> bool | None:
    v = str(raw).strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    # Treat "unsure" (or anything unexpected) as missing for binary stats
    return None


# ---------------------------------------------------------------------------
# 1. Relevance
# ---------------------------------------------------------------------------

def relevance(
    question: str,
    statement: str,
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate whether the statement is relevant to the question.

    Parameters
    ----------
    prompt_template : str | None
        Optional custom prompt template containing ``{question}`` and
        ``{statement}`` placeholders. When provided, this template is rendered
        instead of the built-in ``build_relevance_prompt``. Useful for
        evaluating GEPA-optimised prompt variants without modifying source.

    Returns
    -------
    dict with keys: statement, reasoning, verdict (bool), metric_name
    """
    if prompt_template is not None:
        prompt = _render_custom_template(
            prompt_template, question=question, statement=statement
        )
    else:
        prompt = build_relevance_prompt(question=question, statement=statement)
    try:
        result = call_judge_llm(
            prompt=prompt,
            response_schema=_VERDICT_SCHEMA,
            schema_name="RelevanceVerdict",
            model=model,
            provider=provider,
            max_retries=max_retries,
            temperature=temperature,
        )
        return {
            "metric_name": "relevance",
            "statement": result.get("statement", statement),
            "reasoning": result.get("reasoning", ""),
            "verdict": _parse_verdict(result.get("verdict", "No")),
            "raw_verdict": result.get("verdict"),
        }
    except Exception as e:
        logger.error("Relevance evaluation failed for statement '%s': %s", statement[:60], e)
        return {
            "metric_name": "relevance",
            "statement": statement,
            "reasoning": None,
            "verdict": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 2. Citation Correctness
# ---------------------------------------------------------------------------

def citation_correctness(
    statement: str,
    cited_action_id: str | None,
    cited_action_text: str | None,
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate whether the statement is supported by its cited action document.

    If cited_action_id is None or empty, the verdict is automatically False
    (no citation means citation correctness cannot be established).
    See also: citation_correctness_when_cited() which only evaluates statements
    that actually provide a citation, skipping uncited statements entirely.

    Parameters
    ----------
    cited_action_id : str | None
        The single action ID cited for this statement, if any.
    cited_action_text : str | None
        The text content of the cited action document. None if unavailable.
    prompt_template : str | None
        Optional custom prompt template containing ``{statement}``,
        ``{cited_action_id}``, and ``{cited_action_text}`` placeholders.
        When provided, this template is rendered instead of the built-in
        ``build_citation_correctness_prompt``.
    """
    # Pre-fill rule: no citation → automatic No
    if not cited_action_id or cited_action_id.strip().lower() in ("none", ""):
        return {
            "metric_name": "citation_correctness",
            "statement": statement,
            "cited_action_id": None,
            "reasoning": "No citation ID was provided for this statement. Automatically scored as No.",
            "verdict": False,
            "raw_verdict": "No",
            "auto_filled": True,
        }

    if cited_action_text is None:
        logger.warning(
            "No action text available for cited action %s. Skipping citation correctness.",
            cited_action_id,
        )
        return {
            "metric_name": "citation_correctness",
            "statement": statement,
            "cited_action_id": cited_action_id,
            "reasoning": None,
            "verdict": None,
            "error": f"Action text not found for action ID {cited_action_id}",
        }

    if prompt_template is not None:
        prompt = _render_custom_template(
            prompt_template,
            statement=statement,
            cited_action_id=cited_action_id,
            cited_action_text=cited_action_text,
        )
    else:
        prompt = build_citation_correctness_prompt(
            statement=statement,
            cited_action_id=cited_action_id,
            cited_action_text=cited_action_text,
        )
    try:
        result = call_judge_llm(
            prompt=prompt,
            response_schema=_CITATION_CORRECTNESS_SCHEMA,
            schema_name="CitationCorrectnessVerdict",
            model=model,
            provider=provider,
            max_retries=max_retries,
            temperature=temperature,
        )
        return {
            "metric_name": "citation_correctness",
            "statement": result.get("statement", statement),
            "cited_action_id": result.get("cited_action_id", cited_action_id),
            "reasoning": result.get("reasoning", ""),
            "verdict": _parse_verdict(result.get("verdict", "No")),
            "raw_verdict": result.get("verdict"),
        }
    except Exception as e:
        logger.error(
            "Citation correctness evaluation failed for statement '%s': %s",
            statement[:60],
            e,
        )
        return {
            "metric_name": "citation_correctness",
            "statement": statement,
            "cited_action_id": cited_action_id,
            "reasoning": None,
            "verdict": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 2b. Citation Correctness (when cited only)
# ---------------------------------------------------------------------------

def citation_correctness_when_cited(
    statement: str,
    cited_action_id: str | None,
    cited_action_text: str | None,
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Variant of citation_correctness that only penalises incorrect citations.

    Statements with no citation are skipped entirely (verdict=None,
    skipped=True) rather than automatically scored as No.  This means
    the metric only measures whether citations that ARE given are correct,
    without penalising models that choose not to cite.

    Use this alongside the original citation_correctness() to separately
    quantify:
      - citation_correctness:            includes uncited statements as No
      - citation_correctness_when_cited: only scores statements that cite

    Parameters
    ----------
    prompt_template : str | None
        Optional custom prompt template (same placeholders as
        ``citation_correctness``).
    """
    # No citation → skip (not penalised)
    if not cited_action_id or cited_action_id.strip().lower() in ("none", ""):
        return {
            "metric_name": "citation_correctness_when_cited",
            "statement": statement,
            "cited_action_id": None,
            "reasoning": "No citation provided — statement skipped for this metric.",
            "verdict": None,
            "skipped_no_citation": True,
        }

    if cited_action_text is None:
        logger.warning(
            "No action text available for cited action %s. Skipping.",
            cited_action_id,
        )
        return {
            "metric_name": "citation_correctness_when_cited",
            "statement": statement,
            "cited_action_id": cited_action_id,
            "reasoning": None,
            "verdict": None,
            "error": f"Action text not found for action ID {cited_action_id}",
        }

    # Reuse the same prompt and LLM call as the original metric
    if prompt_template is not None:
        prompt = _render_custom_template(
            prompt_template,
            statement=statement,
            cited_action_id=cited_action_id,
            cited_action_text=cited_action_text,
        )
    else:
        prompt = build_citation_correctness_prompt(
            statement=statement,
            cited_action_id=cited_action_id,
            cited_action_text=cited_action_text,
        )
    try:
        result = call_judge_llm(
            prompt=prompt,
            response_schema=_CITATION_CORRECTNESS_SCHEMA,
            schema_name="CitationCorrectnessVerdict",
            model=model,
            provider=provider,
            max_retries=max_retries,
            temperature=temperature,
        )
        return {
            "metric_name": "citation_correctness_when_cited",
            "statement": result.get("statement", statement),
            "cited_action_id": result.get("cited_action_id", cited_action_id),
            "reasoning": result.get("reasoning", ""),
            "verdict": _parse_verdict(result.get("verdict", "No")),
            "raw_verdict": result.get("verdict"),
        }
    except Exception as e:
        logger.error(
            "Citation correctness (when cited) failed for '%s': %s",
            statement[:60], e,
        )
        return {
            "metric_name": "citation_correctness_when_cited",
            "statement": statement,
            "cited_action_id": cited_action_id,
            "reasoning": None,
            "verdict": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 3. Faithfulness (concatenated — single LLM call, original behaviour)
# ---------------------------------------------------------------------------

def faithfulness(
    statement: str,
    all_action_docs_text: str,
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate whether the statement is supported by any of the action documents
    that the summary was built from (summary_action_ids + cited action).

    Parameters
    ----------
    all_action_docs_text : str
        Concatenated text of all relevant action documents. If empty or None,
        verdict will be None (cannot evaluate).
    prompt_template : str | None
        Optional custom prompt template containing ``{statement}`` and
        ``{all_action_docs_text}`` placeholders. When provided, this template
        is rendered instead of the built-in ``build_faithfulness_prompt``.
    """
    if not all_action_docs_text or not all_action_docs_text.strip():
        return {
            "metric_name": "faithfulness",
            "statement": statement,
            "reasoning": None,
            "verdict": None,
            "error": "No action document text available for faithfulness evaluation.",
        }

    if prompt_template is not None:
        prompt = _render_custom_template(
            prompt_template,
            statement=statement,
            all_action_docs_text=all_action_docs_text,
        )
    else:
        prompt = build_faithfulness_prompt(
            statement=statement,
            all_action_docs_text=all_action_docs_text,
        )
    try:
        result = call_judge_llm(
            prompt=prompt,
            response_schema=_FAITHFULNESS_SCHEMA,
            schema_name="FaithfulnessVerdict",
            model=model,
            provider=provider,
            max_retries=max_retries,
            temperature=temperature,
        )
        raw = result.get("verdict", "No")
        parsed = _parse_verdict(raw)
        return {
            "metric_name": "faithfulness",
            "statement": result.get("statement", statement),
            "reasoning": result.get("reasoning", ""),
            "verdict": parsed,
            "raw_verdict": raw,
            # Optional fields — present only when schema includes them
            "confidence": result.get("confidence"),
            "claims": result.get("claims", []),
        }
    except Exception as e:
        logger.error(
            "Faithfulness evaluation failed for statement '%s': %s", statement[:60], e
        )
        return {
            "metric_name": "faithfulness",
            "statement": statement,
            "reasoning": None,
            "verdict": None,
            "confidence": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 3b. Faithfulness — majority vote (same model, N runs at higher temperature)
# ---------------------------------------------------------------------------

def faithfulness_majority_vote(
    statement: str,
    all_action_docs_text: str,
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    n_votes: int = 3,
    temperature: float = 0.4,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate faithfulness using majority voting.

    Runs the same judge N times at a higher temperature and takes the
    majority verdict. Ties (only possible with even N) fall back to No
    (conservative default).

    Parameters
    ----------
    n_votes : int
        Number of independent runs. Odd numbers avoid ties. Default 3.
    temperature : float
        Sampling temperature for each run. Higher values (e.g. 0.4-0.7)
        introduce variation between runs. Default 0.4.

    Returns
    -------
    dict with keys:
        metric_name     : "faithfulness_majority_vote"
        verdict         : True / False (majority result)
        votes           : list of individual run results
        vote_counts     : {"yes": n, "no": n}
        confidence      : "high" if unanimous, "medium" if majority,
                          "low" if near-tie
        reasoning       : reasoning from the winning-side runs
    """
    if not all_action_docs_text or not all_action_docs_text.strip():
        return {
            "metric_name": "faithfulness_majority_vote",
            "statement": statement,
            "verdict": None,
            "error": "No action document text available.",
        }

    if prompt_template is not None:
        prompt = _render_custom_template(
            prompt_template,
            statement=statement,
            all_action_docs_text=all_action_docs_text,
        )
    else:
        prompt = build_faithfulness_prompt(
            statement=statement,
            all_action_docs_text=all_action_docs_text,
        )

    votes: list[dict] = []
    for i in range(n_votes):
        try:
            result = call_judge_llm(
                prompt=prompt,
                response_schema=_FAITHFULNESS_SCHEMA,
                schema_name="FaithfulnessVerdict",
                model=model,
                provider=provider,
                max_retries=max_retries,
                temperature=temperature,
            )
            raw = result.get("verdict", "No")
            votes.append({
                "run": i + 1,
                "verdict": _parse_verdict(raw),
                "raw_verdict": raw,
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning", ""),
                "claims": result.get("claims", []),
            })
            logger.debug(
                "  [majority_vote] run %d/%d verdict=%s confidence=%s",
                i + 1, n_votes, raw, result.get("confidence"),
            )
        except Exception as e:
            logger.error("[majority_vote] run %d failed: %s", i + 1, e)
            votes.append({
                "run": i + 1,
                "verdict": None,
                "error": str(e),
            })

    # Count valid votes
    valid = [v for v in votes if v.get("verdict") is not None]
    n_yes = sum(1 for v in valid if v["verdict"] is True)
    n_no  = sum(1 for v in valid if v["verdict"] is False)
    n_valid = len(valid)

    if n_valid == 0:
        majority_verdict = None
        conf = None
    elif n_yes > n_no:
        majority_verdict = True
    elif n_no > n_yes:
        majority_verdict = False
    else:
        # Tie — default to No (conservative)
        majority_verdict = False

    # Confidence from vote distribution
    if n_valid > 0:
        majority_count = max(n_yes, n_no)
        if majority_count == n_valid:
            conf = "high"   # unanimous
        elif majority_count >= (n_valid * 2 / 3):
            conf = "medium" # clear majority
        else:
            conf = "low"    # near-tie
    else:
        conf = None

    # Collect reasoning from the majority-side runs
    majority_val = majority_verdict
    majority_reasonings = [
        v["reasoning"] for v in valid
        if v.get("verdict") == majority_val and v.get("reasoning")
    ]
    combined_reasoning = " | ".join(majority_reasonings[:2])  # top 2

    logger.info(
        "  [majority_vote] %d/%d Yes, %d/%d No. Verdict=%s confidence=%s",
        n_yes, n_valid, n_no, n_valid, majority_verdict, conf,
    )

    return {
        "metric_name": "faithfulness_majority_vote",
        "statement": statement,
        "verdict": majority_verdict,
        "vote_counts": {"yes": n_yes, "no": n_no},
        "n_votes": n_valid,
        "confidence": conf,
        "reasoning": combined_reasoning,
        "votes": votes,
    }


# ---------------------------------------------------------------------------
# 3c. Faithfulness per document (any-Yes logic)
# ---------------------------------------------------------------------------

def faithfulness_per_document(
    statement: str,
    action_ids: list[str],
    action_loader,  # ActionLoader instance
    model: str,
    provider: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate faithfulness by checking each action document independently.

    Makes one LLM call per action document. The overall verdict is True
    (faithful) if ANY single document supports the statement on its own,
    without needing to combine information across documents.

    This avoids the cross-document synthesis problem of the concatenated
    approach, where the judge can combine partial evidence from multiple
    documents to support a claim that no single document directly supports.

    Parameters
    ----------
    statement : str
        The statement to evaluate.
    action_ids : list[str]
        List of action IDs to check (summary_action_ids + cited_action_id).
        Deduplicated automatically.
    action_loader : ActionLoader
        Used to load action text for each ID.
    model : str
        OpenRouter model identifier.
    provider : str | None
        Optional provider pin.
    max_retries : int
        Retry attempts per document call.

    Returns
    -------
    dict with keys:
        metric_name         : "faithfulness_per_document"
        statement           : str
        verdict             : True if any document supports the statement
        supporting_doc      : action ID of the first document that supported it
                              (None if verdict is False)
        docs_checked        : number of documents evaluated
        docs_supporting     : number of documents that individually supported it
        per_doc_results     : list of per-document dicts
        reasoning           : summary of which documents supported/rejected
    """
    # Deduplicate action IDs while preserving order
    seen: set[str] = set()
    unique_ids: list[str] = []
    for aid in action_ids:
        aid = str(aid).strip()
        if aid and aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)

    if not unique_ids:
        return {
            "metric_name": "faithfulness_per_document",
            "statement": statement,
            "verdict": None,
            "supporting_doc": None,
            "docs_checked": 0,
            "docs_supporting": 0,
            "per_doc_results": [],
            "reasoning": "No action IDs provided.",
            "error": "No action IDs provided for faithfulness evaluation.",
        }

    per_doc_results: list[dict] = []
    first_supporting_doc: str | None = None
    docs_supporting = 0
    any_failed = False

    for action_id in unique_ids:
        doc_text = action_loader.get_action_text(action_id)

        if doc_text is None:
            logger.warning(
                "[faithfulness_per_doc] Action text unavailable for ID %s — skipping.",
                action_id,
            )
            per_doc_results.append({
                "action_id": action_id,
                "verdict": None,
                "reasoning": None,
                "error": f"Action text not found for ID {action_id}",
            })
            any_failed = True
            continue

        prompt = build_faithfulness_single_doc_prompt(
            statement=statement,
            action_id=action_id,
            action_doc_text=doc_text,
        )

        try:
            result = call_judge_llm(
                prompt=prompt,
                response_schema=_VERDICT_SCHEMA,
                schema_name="FaithfulnessSingleDocVerdict",
                model=model,
                provider=provider,
                max_retries=max_retries,
            )
            verdict_bool = _parse_verdict(result.get("verdict", "No"))
            per_doc_results.append({
                "action_id": action_id,
                "verdict": verdict_bool,
                "reasoning": result.get("reasoning", ""),
                "raw_verdict": result.get("verdict"),
            })
            if verdict_bool:
                docs_supporting += 1
                if first_supporting_doc is None:
                    first_supporting_doc = action_id
                    logger.info(
                        "  [faithfulness_per_doc] Statement supported by action %s — "
                        "continuing to check remaining docs.",
                        action_id,
                    )

        except Exception as e:
            logger.error(
                "[faithfulness_per_doc] LLM call failed for action %s: %s",
                action_id, e,
            )
            per_doc_results.append({
                "action_id": action_id,
                "verdict": None,
                "reasoning": None,
                "error": str(e),
            })
            any_failed = True

    # Overall verdict: True if any document supported the statement
    docs_checked = len(per_doc_results)
    any_supported = docs_supporting > 0

    # Build a concise reasoning summary
    supporting_ids = [r["action_id"] for r in per_doc_results if r.get("verdict") is True]
    rejecting_ids  = [r["action_id"] for r in per_doc_results if r.get("verdict") is False]
    failed_ids     = [r["action_id"] for r in per_doc_results if r.get("verdict") is None]

    reasoning_parts = []
    if supporting_ids:
        reasoning_parts.append(f"Supported by: {', '.join(supporting_ids)}")
    if rejecting_ids:
        reasoning_parts.append(f"Not supported by: {', '.join(rejecting_ids)}")
    if failed_ids:
        reasoning_parts.append(f"Failed/skipped: {', '.join(failed_ids)}")
    reasoning_summary = "; ".join(reasoning_parts) or "No documents evaluated."

    overall_verdict: bool | None
    if docs_checked == 0:
        overall_verdict = None
    elif any_supported:
        overall_verdict = True
    elif any_failed and not rejecting_ids:
        # All docs either failed or were skipped — can't determine verdict
        overall_verdict = None
    else:
        overall_verdict = False

    logger.info(
        "  [faithfulness_per_doc] %d/%d docs supported statement. Overall: %s",
        docs_supporting,
        docs_checked,
        overall_verdict,
    )

    return {
        "metric_name": "faithfulness_per_document",
        "statement": statement,
        "verdict": overall_verdict,
        "supporting_doc": first_supporting_doc,
        "docs_checked": docs_checked,
        "docs_supporting": docs_supporting,
        "per_doc_results": per_doc_results,
        "reasoning": reasoning_summary,
    }
