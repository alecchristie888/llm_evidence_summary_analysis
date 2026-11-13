"""Build the statement- and summary-level score tables from the full-corpus judge output.

Input : data/judge_outputs/merged_all_records.json   (one record per statement)
Output: data/derived/statement_level.csv   (23,641 rows)
        data/derived/summary_level.csv     (1,991 rows; one per model x question)

Per-summary scores are proportions over that summary's statements:
  relevance_score  = share of statements judged relevant
  citation_rate    = share of statements carrying >=1 citation
  cc_all_score     = share judged citation-correct (uncited statements count as incorrect)
  cc_cited_score   = share judged citation-correct among cited statements only
  faithfulness_score = share judged faithful to the summary's cited actions
"""
import glob, json, os
import pandas as pd

J = os.environ.get("IGF_JUDGE", "data/judge_outputs/merged_all_records.json")
D = os.environ.get("IGF_DERIVED", "data/derived")
os.makedirs(D, exist_ok=True)

def v(r, k):
    m = r.get(k) or {}
    return m.get("verdict")

# Scripts assume the working directory is the repository root (the folder holding run_all.sh and data/).
if not os.path.isdir(os.path.dirname(J)):
    raise SystemExit(f"Cannot find {os.path.dirname(J)!r} from working directory {os.getcwd()!r}. "
                     "Run from the repository root with the data archive unpacked so that data/judge_outputs/ exists.")

if not os.path.exists(J):  # build the merged file from the per-model judge outputs
    files = sorted(glob.glob(os.path.join(os.path.dirname(J), "by_model", "*", "*_llm_scores.json")))
    if not files:
        raise SystemExit(f"No judge output files found under {os.path.dirname(J)}/by_model/. Nothing written.")
    recs = []
    for f in files:
        recs.extend(json.load(open(f, encoding="utf-8")))
    json.dump(recs, open(J, "w", encoding="utf-8"))

records = json.load(open(J, encoding="utf-8"))
if not records:
    raise SystemExit(f"{J} is empty. Delete it and re-run so it is rebuilt from data/judge_outputs/by_model/. Nothing written.")

rows = []
for r in records:
    ff = r.get("llm_faithfulness") or {}
    cc = r.get("llm_citation_correctness") or {}
    claims = ff.get("claims") or []
    reasons = [x for c in claims for x in (c.get("failure_reasons") or [])]  # one entry per failing claim
    rows.append(dict(
        synopsis=r["synopsis"], summary_model=r["summary_model"], question=r["question"],
        statement=r["statement"], n_cited=len(r.get("cited_action_ids") or []),
        rel_verdict=v(r, "llm_relevance"), cc_verdict=cc.get("verdict"),
        cc_auto_no=bool(cc.get("auto_filled", False)),
        cc_when_cited_verdict=v(r, "llm_citation_correctness_when_cited"),
        faith_verdict=ff.get("verdict"), n_claims=len(claims),
        failure_reasons="|".join(reasons)))
st = pd.DataFrame(rows)
st.to_csv(f"{D}/statement_level.csv", index=False)

f = lambda s: pd.to_numeric(s.map({True: 1.0, False: 0.0}), errors="coerce").mean()
sm = st.groupby(["summary_model", "synopsis", "question"], sort=False).apply(lambda g: pd.Series(dict(
    n_statements=float(len(g)), n_cited_statements=float((g.n_cited > 0).sum()),
    citation_rate=(g.n_cited > 0).mean(), relevance_score=f(g.rel_verdict),
    cc_all_score=f(g.cc_verdict), cc_cited_score=f(g.cc_when_cited_verdict),
    faithfulness_score=f(g.faith_verdict)))).reset_index()
sm.to_csv(f"{D}/summary_level.csv", index=False)
print(f"statements: {len(st):,}  summaries: {len(sm):,}  questions: {st.question.nunique()}")
