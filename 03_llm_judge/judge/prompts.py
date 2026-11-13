"""
Prompt builders for LLM-as-Judge evaluation.

Three metrics are evaluated, mirroring the human evaluation workflow:

1. Relevance      – Is the statement relevant to the question?
   Inputs: question + statement only.
   Output: binary Yes/No + reasoning.

2. Citation Correctness – Does the statement come from the specific action ID it cites?
   Inputs: statement + cited action document (one doc only).
   Output: binary Yes/No + reasoning.
   Special rule: statements with NO citation automatically receive 'No'.

3. Faithfulness   – Is the statement supported by evidence from ANY of the action
                    documents that the summary was based on (summary_action_ids) plus
                    the citation document?
   Inputs: statement + all relevant action documents.
   Output: binary Yes/No + reasoning.

3b. Faithfulness (single-doc) – Same question but evaluated against ONE document at a time.
   Used by faithfulness_per_document() in metrics.py.
   The caller aggregates: if any single document returns Yes, the overall verdict is Yes.
   Inputs: statement + one action document.
   Output: binary Yes/No + reasoning.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: render a custom prompt template loaded from disk
# ---------------------------------------------------------------------------

class _SafeFormatDict(defaultdict):
    """Dict that returns ``{key}`` literal for missing keys instead of raising."""
    def __missing__(self, key):
        return "{" + key + "}"


def _render_custom_template(template: str, **fields) -> str:
    """
    Render a custom prompt template using ``str.format_map`` with safe defaults.

    Any placeholder in ``template`` matching a provided field is substituted;
    unknown placeholders are left intact (e.g. ``{foo}`` stays ``{foo}``).
    This makes it safe to load externally-authored prompt files (e.g. from
    GEPA optimisation) without crashing on stray braces.

    Parameters
    ----------
    template : str
        The prompt template containing Python ``str.format``-style placeholders.
    **fields
        Keyword arguments used as substitution values (e.g. ``question=...``,
        ``statement=...``).

    Returns
    -------
    str
        The rendered prompt.
    """
    safe = _SafeFormatDict()
    safe.update(fields)
    return template.format_map(safe)


# ---------------------------------------------------------------------------
# 1. RELEVANCE
# ---------------------------------------------------------------------------

def build_relevance_prompt(question: str, statement: str) -> str:
    # GEPA-optimised prompt (v21 relevance — best val score 0.980 vs v2 seed 0.910).
    # Key advances over previous v20: Rule 4 expanded to cover cross-region/trade-off
    # comparisons with explicit examples; ecosystem exclusion rule tightened with a
    # concrete example; ecosystem class examples added inline; double-quote style
    # throughout; reasoning instruction clarified to "in which case answer Yes".
    return (
        "You are evaluating whether a statement is relevant to answering a specific "
        "conservation evidence question.\n\n"
        "Question: {question}\n\n"
        "Statement: {statement}\n\n"
        "Task: Decide whether the statement is relevant to the question using the "
        "standards below.\n\n"
        "RELEVANCE STANDARDS:\n\n"
        "1. TOPIC MATCH (required, but interpreted VERY PERMISSIVELY): The statement "
        "should plausibly concern the same broad subject as the question. Interpret "
        '"same" at a GENEROUS level of generality \u2014 sub-types, specific examples, '
        "closely analogous forms, near-synonyms, or specific instances of a broader "
        "category all count as on-topic.\n"
        "   - Clear mismatch example: Question about bats, statement about moths/birds "
        "with no tie-back to bats \u2192 NOT relevant.\n"
        "   - Taxa generality: A specific species/genus counts as relevant to a question "
        'about its broader group (e.g. a statement about "cattle" is relevant to a '
        'question about "livestock"; a specific bat species is relevant to a question '
        'about "bats").\n'
        "   - Intervention generality and ANALOGY: A specific intervention counts as "
        "relevant to a question about a broader, related, or analogous class. Treat "
        "structurally or functionally similar interventions as relevant even if the "
        "wording differs.\n"
        '     * Example: "tree plantations next to remnant native vegetation" IS '
        'relevant to "hedgerows or tree lines in agricultural land".\n'
        '     * Example: "community-based wetland management" IS relevant to "legal '
        'frameworks and public engagement" \u2014 even if the question is framed around '
        "forest conservation, because community-based management is a direct instance "
        "of the intervention class.\n"
        '     * Example: "pit habitats on artificial structures" IS relevant to '
        '"habitat creation methods for species richness."\n'
        "   - Ecosystem/context generality: Do NOT reject a statement merely because it "
        "describes a different ecosystem, habitat, country, or adjacent land-use than "
        "the question implies. Even when the question mentions a specific ecosystem "
        '(e.g. "forest conservation"), a statement from a different ecosystem (e.g. '
        "wetland) is still RELEVANT if it concerns the same intervention/mechanism "
        "class (e.g. community-based management, legal frameworks, public engagement). "
        "The intervention/mechanism match takes priority over ecosystem match.\n"
        "   - Only treat ecosystem/region restriction as disqualifying if the question "
        "EXPLICITLY and narrowly excludes other contexts "
        '(e.g. "in forests only, excluding other ecosystems") '
        "AND the statement's content cannot transfer as evidence.\n"
        "   - If the question's scope (taxa, intervention, region, ecosystem) is "
        "ambiguous or broad, default to RELEVANT.\n\n"
        "2. PARTIAL / INCOMPLETE INFO IS OK: Vague, brief, or incomplete statements are "
        "RELEVANT as long as they are on-topic. The statement does not need to fully "
        "answer the question.\n\n"
        "3. ABSENCE OF EVIDENCE: Statements saying no effect was found, no studies "
        "exist, or results were inconclusive ARE relevant if they concern the "
        "question's topic.\n\n"
        "4. ONE SIDE OF A COMPARISON: If the question asks to compare A vs B, across "
        "regions, or about trade-offs between X and Y, a statement addressing only ONE "
        "side (just A, just one region, just one outcome dimension) IS relevant.\n\n"
        '5. MARK "No" ONLY IF AT LEAST ONE OF THESE CLEARLY HOLDS:\n'
        "   - The statement is about a clearly different taxon/species group than the "
        "question (e.g. moths vs bats, fish vs amphibians) with no mechanistic or "
        "contextual tie-back.\n"
        "   - The statement is about a fundamentally different intervention with NO "
        "meaningful structural, functional, or analogical overlap.\n"
        "   - The statement has no meaningful connection to the question's subject "
        "matter.\n"
        '   Do NOT mark "No" merely because:\n'
        "   - The ecosystem/habitat/region/land-use context differs from what the "
        "question seems focused on (this alone is NEVER sufficient grounds for No).\n"
        "   - The statement is vague, brief, or only partially answers the question.\n"
        "   - The statement covers only one side of a requested comparison.\n"
        "   - The statement is a specific instance, analogue, or close variant of the "
        "broader category in the question.\n"
        "   - The wording of the intervention differs from the question's wording but "
        "refers to a structurally/functionally similar practice.\n\n"
        "DECISION PROCEDURE:\n"
        "Step 1: Identify the question's (a) subject/taxa, (b) "
        "intervention/mechanism class, and (c) any contextual framing (ecosystem, "
        "region). Note that (b) is the primary criterion; (c) is secondary and rarely "
        "disqualifying.\n"
        "Step 2: Identify what the statement is about (taxa/subject + "
        "intervention/mechanism + context).\n"
        "Step 3: Ask: \"Is the statement plausibly an example, instance, sub-type, "
        "analogue, or close variant of the question's intervention/mechanism or "
        "taxa?\" If yes \u2192 Yes.\n"
        "Step 4: Before answering No, explicitly check:\n"
        "   (a) Could this statement be viewed as a functional analogue or specific "
        "case of the question's intervention/mechanism?\n"
        "   (b) Am I rejecting solely because of ecosystem/habitat/region mismatch? "
        "If so, reconsider \u2014 this alone is not sufficient for No.\n"
        "   If any reasonable interpretation makes it on-topic, answer Yes.\n"
        "Step 5: Only answer No if there is (i) a clear, unambiguous mismatch in taxa, "
        "OR (ii) a fundamental mismatch in intervention/mechanism with no analogy. "
        "Context/ecosystem mismatch alone does NOT justify No.\n"
        "Step 6: When genuinely uncertain after applying the above, default to Yes.\n\n"
        "Respond ONLY as a JSON object with the following fields:\n"
        '  "statement": the statement text\n'
        '  "reasoning": brief explanation applying the steps above, explicitly considering '
        "whether the statement could be an analogue/instance of the question's topic, "
        "and explicitly noting if ecosystem mismatch was the only concern "
        '(in which case answer Yes)\n'
        '  "verdict": "Yes" or "No"\n'
    ).format(question=question, statement=statement)


# ---------------------------------------------------------------------------
# 2. CITATION CORRECTNESS
# ---------------------------------------------------------------------------

def build_citation_correctness_prompt(
    statement: str,
    cited_action_id: str,
    cited_action_text: str,
) -> str:
    # GEPA-optimised prompt (v18 CC — best val score 0.960 vs seed 0.915, +4.5%).
    # Key advance: generous interpretation of paraphrasing, effectiveness language,
    # numerical ranges, and aggregated claims. Mark No ONLY for clear contradictions
    # or specific claims wholly absent from the document.
    return (
        "You are evaluating whether a statement from a conservation evidence summary "
        "is supported by the single cited action document.\n\n"
        "Statement: {statement}\n\n"
        "Cited Action ID: {cited_action_id}\n\n"
        "Content of Cited Document (Action {cited_action_id}):\n"
        "---\n"
        "{cited_action_text}\n"
        "---\n\n"
        "Task: Decide whether the statement is supported by the cited document.\n\n"
        "Apply these standards, which mirror how human expert reviewers judge these "
        "cases:\n\n"
        "1. **Reasonable paraphrasing is acceptable.** Do not require verbatim wording. "
        "If the underlying finding, fact, or study result is present in the document, "
        "the statement is supported even if phrasing differs.\n\n"
        "2. **Summary statements are supported** if the finding they summarise is "
        "clearly present in the document. A statement that accurately describes what "
        "one or more studies in the document found should be marked Yes, even if the "
        "document also contains additional caveats, effectiveness ratings, or contrary "
        "studies that the statement does not mention. Statements do not need to reproduce "
        "the document\'s overall effectiveness category (e.g. \'Unknown effectiveness\', "
        "\'Beneficial\') to be supported \u2014 they only need to accurately reflect "
        "specific findings or content that the document contains.\n\n"
        "3. **Phrases like \'has proven effective\', \'was effective\', \'worked\', or "
        "\'promoted [outcome]\' ** should be interpreted as referring to the outcomes "
        "observed in the studies described, not as a claim about the document\'s overall "
        "effectiveness rating. If the document describes studies where the intervention "
        "produced the stated outcome, such a statement is supported \u2014 even if the "
        "document\'s overall effectiveness category is \'Unknown\' or mixed. Only mark No "
        "if the statement makes a quantitative or categorical claim about overall "
        "effectiveness that directly contradicts the document (e.g. claiming an "
        "intervention is \'proven effective across all studies\' when the document shows "
        "mostly negative results).\n\n"
        "4. **Absence-of-evidence statements** (e.g. \'no studies were found\', \'no "
        "evidence available\') are supported (Yes) if the document explicitly states that "
        "no studies or no evidence exist for the action.\n\n"
        "5. **Numerical ranges, durations, and quantitative descriptors** (e.g. \'one to "
        "27 years\', \'across multiple sites\', \'several studies\') should be evaluated "
        "generously. If the statement\'s range or descriptor plausibly encompasses or "
        "summarises the specific values found in the document, treat it as supported. "
        "Only mark No if the document clearly contradicts the range (e.g. statement says "
        "\'over 20 years\' but document only describes a 2-year study) or if the number "
        "is a specific verifiable claim that the document flatly refutes.\n\n"
        "6. **Aggregated or generalised claims** (e.g. \'studies showed\', \'areas with X "
        "had Y\') should be marked Yes if at least one study in the document supports the "
        "claim and the document does not contradict it with counter-evidence that would "
        "make the generalisation misleading. The statement does not need to cite every "
        "study or match the exact count of studies.\n\n"
        "7. **Mark No only if** the statement makes a specific factual claim (about "
        "species, location, magnitude, direction of effect, methodology, or outcome) that "
        "is clearly absent from the document or directly contradicts what the document "
        "says. Do not mark No merely because:\n"
        "   - the statement omits nuance, caveats, or effectiveness ratings;\n"
        "   - the statement\'s numerical range is broader than the specific values in the "
        "document but plausibly encompasses them;\n"
        "   - the statement generalises across studies when the document contains at least "
        "one supporting study;\n"
        "   - the statement paraphrases findings in slightly different terms.\n\n"
        "Procedure:\n"
        "- Identify the core factual claim(s) in the statement.\n"
        "- Locate supporting content in the document for each claim.\n"
        "- For numerical or quantitative elements, check whether the document\'s values "
        "are consistent with (not necessarily identical to) the statement\'s claim.\n"
        "- Check for any element that directly contradicts the document.\n"
        "- Apply the standards above, erring toward Yes when the core finding is present "
        "and the statement is a reasonable paraphrase or summary. Reserve No for clear "
        "contradictions or specific claims wholly absent from the document.\n\n"
        "Provide your reasoning first, then give a final verdict of \'Yes\' (supported by "
        "the cited document) or \'No\' (not supported).\n\n"
        "Respond ONLY as a JSON object with the following fields:\n"
        '  "statement": the statement text\n'
        '  "cited_action_id": the action ID\n'
        '  "reasoning": your explanation\n'
        '  "verdict": "Yes" or "No"\n'
    ).format(
        statement=statement,
        cited_action_id=cited_action_id,
        cited_action_text=cited_action_text,
    )


# ---------------------------------------------------------------------------
# 3. FAITHFULNESS
# ---------------------------------------------------------------------------

def build_faithfulness_single_doc_prompt(
    statement: str,
    action_id: str,
    action_doc_text: str,
) -> str:
    """
    Prompt for faithfulness evaluated against a single action document.

    Used by faithfulness_per_document() in metrics.py — called once per
    action document. The caller aggregates results with any-Yes logic.
    """
    return (
        "You are evaluating whether a statement is directly supported by a single "
        "conservation evidence document.\n\n"
        "Statement: {statement}\n\n"
        "Evidence Document (Action ID: {action_id}):\n"
        "---\n"
        "{action_doc_text}\n"
        "---\n\n"
        "Task: Determine whether the specific claims in the statement are supported "
        "by this single document alone.\n\n"
        "Mark Yes if:\n"
        "- The specific finding in the statement — including any numbers, outcomes, "
        "species, or locations — is present in this document\n"
        "- Reasonable paraphrasing of the document's content is acceptable\n\n"
        "Mark No if:\n"
        "- The statement makes a specific claim that is absent from this document\n"
        "- The statement combines information from this document with something not "
        "present here (the claim must be supportable by this document alone)\n"
        "- The statement makes a comparative claim (e.g. 'X is better than Y') that "
        "is not directly stated in this document\n\n"
        "Provide your reasoning first, then give a final verdict of 'Yes' (supported "
        "by this document) or 'No' (not supported by this document alone).\n\n"
        "Respond ONLY as a JSON object with the following fields:\n"
        '  "statement": the statement text\n'
        '  "reasoning": your explanation\n'
        '  "verdict": "Yes" or "No"\n'
    ).format(
        statement=statement,
        action_id=action_id,
        action_doc_text=action_doc_text,
    )



def build_faithfulness_prompt(
    statement: str,
    all_action_docs_text: str,
) -> str:
    # GEPA-optimised prompt (v18 — best val score 0.855 vs seed 0.845).
    # Key advance: strict action-attribution requiring the judge to identify
    # which specific action the statement names, then verify ALL claims
    # against ONLY that action's document. Handles compound claims,
    # qualifier co-location, and evidence-characterisation accuracy.
    return (
        "You are evaluating whether a statement extracted from a conservation evidence "
        "summary is faithfully supported by a set of source action documents.\n\n"
        "Statement: {statement}\n\n"
        "Available Evidence Documents:\n"
        "---\n"
        "{all_action_docs_text}\n"
        "---\n\n"
        "## Evaluation Standards\n\n"
        "Apply these rules in order:\n\n"
        "1. **Omission is NOT unfaithfulness.** If the statement makes a narrower or less "
        "complete claim than the documents, but what it says is correct, mark Yes. Do not "
        "penalise a statement for leaving out caveats, additional studies, or extra "
        "detail.\n\n"
        "2. **Subject\u2013action attribution must be strict.** Every claim in the statement "
        "must be attributable to the SAME action/intervention that the documents describe "
        "for that subject.\n"
        "   - Identify the action the statement is about from the NAMED materials, methods, "
        "or interventions in the statement itself (e.g., \'cotton gin compost and poultry "
        "manure\' \u2192 organic processing wastes; \'municipal waste compost\' \u2192 municipal "
        "waste compost; \'prescribed fire\' \u2192 prescribed fire).\n"
        "   - Findings from Action A cannot support a statement attributed to Action B, "
        "EVEN IF A and B are related, adjacent, or concern similar materials/outcomes.\n"
        "   - Be especially suspicious when the statement\'s materials/methods appear as "
        "examples under one action heading but the claim\'s framing (e.g. specific soil "
        "type, specific comparison) is drawn from a different action\'s evidence. Check "
        "that every qualifier (soil type, region, outcome) is stated in the SAME passage "
        "as the named materials.\n\n"
        "3. **Specific facts must be correct.** Any specific number (counts of studies, "
        "sample sizes, percentages, years, locations, species, action names, soil types, "
        "materials) stated must match the documents. If a number, named entity, or "
        "qualifier (e.g. \'sandy loam soils\') in the statement is wrong, missing from, or "
        "not explicitly co-located with the named finding in the documents, mark No.\n"
        "   - A qualifier like a soil type is only supported if the document explicitly "
        "ties that qualifier to the specific finding stated \u2014 not merely lists it "
        "elsewhere in the same action\'s coverage.\n\n"
        "4. **Effect directions must be correct.** If the statement asserts an "
        "increase/decrease/improvement/reduction/no effect, the documents must explicitly "
        "support that direction for the named subject under the named action. Reversed or "
        "unsupported directional claims \u2192 No.\n\n"
        "5. **Comparative claims require explicit support.** Claims of the form \'X more "
        "than Y\', \'better than\', \'most effective\', \'compared to\' must be explicitly "
        "stated or directly derivable from a single passage in the documents. If the "
        "comparison is not made in the documents, mark No \u2014 even if each side is "
        "individually discussed.\n\n"
        "6. **Characterisations of evidence (\'mixed\', \'inconsistent\', \'limited\', "
        "\'no evidence\') must match the documents\' actual pattern of findings for the "
        "specific action named.**\n"
        "   - \'No evidence\' / \'lack of evidence\' statements are supported if the "
        "documents explicitly say no studies were found for that action.\n"
        "   - \'Mixed\' / \'inconsistent\' requires the documents to show genuinely "
        "conflicting results on the SAME outcome for the SAME action.\n"
        "   - If the documents show mostly one result with only a minor variation, this "
        "is NOT \'mixed\' \u2014 mark No.\n\n"
        "7. **Compound claims require ALL parts to be supported.** If the statement "
        "asserts multiple properties joined by \'and\', \'with\', commas, or lists (e.g. "
        "\'cotton gin compost and poultry manure reduced soil loss in sandy loam soils\'), "
        "every component \u2014 every named material, every named qualifier, every outcome "
        "\u2014 must be explicitly supported in a single coherent passage for the named "
        "action. An unsupported component makes the whole statement unfaithful.\n\n"
        "8. **Vague but accurate statements are acceptable.** General statements (e.g. "
        "\'some studies have examined X\', \'evidence exists for Y\', \'there was a lack of "
        "sufficient evidence\') that are not wrong and are consistent with the documents "
        "should be marked Yes, even if imprecise. Vagueness is only acceptable when no "
        "specific claim is smuggled in alongside it.\n\n"
        "9. **The claim must be traceable to the documents.** Specific findings \u2014 "
        "numbers, outcomes, species, locations, action names, qualifiers \u2014 must appear "
        "in the documents, tied to the correct action. General topical overlap is not "
        "sufficient.\n\n"
        "## Procedure\n\n"
        "Work through these steps explicitly in your reasoning:\n\n"
        "**Step 1 \u2014 Identify the action.** From the NAMED materials/methods/interventions "
        "in the statement, determine which action the statement is making claims about.\n\n"
        "**Step 2 \u2014 Locate the action document.** Find the document for that specific "
        "action. If multiple actions are plausible, list them.\n\n"
        "**Step 3 \u2014 Decompose the statement.** List every specific claim separately: "
        "named materials/methods, named qualifiers (soil type, location, region), numbers, "
        "effect directions, comparisons, evidence-strength characterisations, and every "
        "clause in compound claims.\n\n"
        "**Step 4 \u2014 Verify each claim.** For each, quote or paraphrase the supporting "
        "passage from the correct action\'s document. Flag any claim where the passage is "
        "from a different action, a qualifier is not tied to the finding in the same "
        "passage, or a compound component is missing.\n\n"
        "**Step 5 \u2014 Verdict.** Mark Yes only if every specific claim is explicitly "
        "supported by a passage in the document for the correctly identified action, with "
        "all qualifiers co-located with the finding. Otherwise mark No.\n\n"
        "Respond ONLY as a JSON object with the following fields:\n"
        '  "statement": the statement text\n'
        '  "reasoning": your step-by-step analysis\n'
        '  "verdict": "Yes" or "No"\n'
    ).format(
        statement=statement,
        all_action_docs_text=all_action_docs_text,
    )
