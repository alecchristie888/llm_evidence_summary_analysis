"""Check that the regenerated tables reproduce the numbers reported in the paper.

Run from the repository root after run_all.sh. Point estimates must match to the
reported precision; bootstrap CIs are checked separately (see README).
"""
import sys
import pandas as pd

D = "data/derived"
fails = 0

def check(label, got, want, tol=0.05):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'OK ' if ok else 'FAIL'}  {label:55} got {got:8.2f}  paper {want:8.2f}")

# Corpus size
st = pd.read_csv(f"{D}/statement_level.csv"); sm = pd.read_csv(f"{D}/summary_level.csv")
check("statements", len(st), 23641, 0); check("summaries", len(sm), 1991, 0)
check("questions", st.question.nunique(), 520, 0); check("subject areas", st.synopsis.nunique(), 19, 0)

# Headline means (%), judge = Gemini 3.1 Pro Preview
PAPER = {
    ("Relevance", "GPT-5"): 96.5, ("Relevance", "Kimi K2"): 94.4, ("Relevance", "Claude Sonnet 4"): 93.1, ("Relevance", "Gemini 2.5 Pro"): 88.2,
    ("Citation rate", "GPT-5"): 97.8, ("Citation rate", "Gemini 2.5 Pro"): 96.2, ("Citation rate", "Claude Sonnet 4"): 85.8, ("Citation rate", "Kimi K2"): 42.5,
    ("CC (when cited)", "GPT-5"): 96.9, ("CC (when cited)", "Gemini 2.5 Pro"): 96.9, ("CC (when cited)", "Claude Sonnet 4"): 93.3, ("CC (when cited)", "Kimi K2"): 83.3,
    ("Faithfulness", "Gemini 2.5 Pro"): 91.5, ("Faithfulness", "GPT-5"): 90.3, ("Faithfulness", "Claude Sonnet 4"): 81.1, ("Faithfulness", "Kimi K2"): 73.7,
}
h = pd.read_csv(f"{D}/model_headline.csv")
for (metric, model), want in PAPER.items():
    got = 100 * h.loc[(h.metric == metric) & (h.model == model), "mean"].item()
    check(f"{metric} — {model}", got, want)

# Retrieval (%), evaluated-run definition (Results; Extended Data retrieval table)
RET = {"Gemini 2.5 Pro": 49.2, "GPT-5": 65.2, "Kimi K2": 68.4, "Claude Sonnet 4": 72.9}
r = pd.read_csv(f"{D}/retrieval_metrics_by_model_evaluated_run.csv").set_index("model")
for m, want in RET.items():
    check(f"retrieval recall — {m}", r.loc[m, "ret_recall"], want)
PREC = {"GPT-5": 70.3, "Gemini 2.5 Pro": 67.6, "Kimi K2": 57.6, "Claude Sonnet 4": 53.7}
for m, want in PREC.items():
    check(f"retrieval precision — {m}", r.loc[m, "ret_prec"], want)
CT = {"Gemini 2.5 Pro": 98.5, "GPT-5": 96.6, "Kimi K2": 96.0, "Claude Sonnet 4": 94.7}
for m, want in CT.items():
    check(f"cite-through — {m}", r.loc[m, "cite_through"], want)
check("actions opened — Gemini 2.5 Pro", r.loc["Gemini 2.5 Pro", "n_opened"], 3.07, 0.005)
check("actions opened — Claude Sonnet 4", r.loc["Claude Sonnet 4", "n_opened"], 6.82, 0.005)
check("actions opened — GPT-5", r.loc["GPT-5", "n_opened"], 4.42, 0.005)
check("actions opened — Kimi K2", r.loc["Kimi K2", "n_opened"], 6.03, 0.005)
ATT = {"openai/gpt-5": 38, "moonshotai/kimi-k2-0905": 13, "anthropic/claude-sonnet-4": 2, "google/gemini-2.5-pro": 0}
a = pd.read_csv(f"{D}/attrition_by_model.csv").set_index("model")
for m, want in ATT.items():
    check(f"questions never completed — {m}", a.loc[m, "never_completed"], want, 0)

# Agent behaviour (Results 'Faithfulness rewards omission')
bh = pd.read_csv(f"{D}/agent_behaviour_by_model.csv").set_index("model")
check("searches per question — Gemini 2.5 Pro", bh.loc["google/gemini-2.5-pro", "searches"], 1.40, 0.005)
check("max search offset — GPT-5 (evaluated-run)", r.loc["GPT-5", "max_offset"], 4.68, 0.005)
check("max search offset — Gemini 2.5 Pro (evaluated-run)", r.loc["Gemini 2.5 Pro", "max_offset"], 0.24, 0.005)
er = pd.read_csv(f"{D}/agent_errors_by_model.csv").set_index("model")
check("tool-error rate % — Kimi K2", er.loc["moonshotai/kimi-k2-0905", "pct_toolerr"], 11.2)

# Distribution tails (Results 'Averages conceal the summaries decision-makers actually read')
sd = pd.read_csv(f"{D}/summary_distribution.csv").set_index(["model", "metric"])
check("Kimi K2 citation rate: % summaries < 50", 100 * sd.loc[("Kimi K2", "Citation rate"), "pct_lt_50"], 49.7)
check("Kimi K2 citation rate: % summaries >= 90", 100 * sd.loc[("Kimi K2", "Citation rate"), "pct_ge_90"], 20.4, 0.06)  # CSV is rounded to 4 dp (0.2045)
check("Claude Sonnet 4 faithfulness: % < 70", 100 * sd.loc[("Claude Sonnet 4", "Faithfulness"), "pct_lt_70"], 22.3)
check("Kimi K2 faithfulness: % < 70", 100 * sd.loc[("Kimi K2", "Faithfulness"), "pct_lt_70"], 37.2)
check("GPT-5 faithfulness: % < 70", 100 * sd.loc[("GPT-5", "Faithfulness"), "pct_lt_70"], 6.9)
check("Gemini 2.5 Pro faithfulness: % >= 90", 100 * sd.loc[("Gemini 2.5 Pro", "Faithfulness"), "pct_ge_90"], 69.2)

# Variance decomposition (eta-squared, %)
vd = pd.read_csv(f"{D}/variance_decomp.csv").set_index(["metric", "term"])
check("eta2 model — citation rate", 100 * vd.loc[("Citation rate", "model"), "eta_sq"], 48.6)
check("eta2 model — faithfulness", 100 * vd.loc[("Faithfulness", "model"), "eta_sq"], 17.9, 0.06)  # CSV is rounded to 4 dp (0.1795)
check("eta2 residual — relevance", 100 * vd.loc[("Relevance", "Residuals"), "eta_sq"], 88.4)

# Judge validation (Table 1)
jv = pd.read_csv(f"{D}/judge_validation_table.csv")
jv = jv[jv.status == "ok"].set_index(["metric", "judge", "prompt"])
g = ("Gemini 3.1 Pro Preview",)
check("judge agreement — relevance, Gemini refined", jv.loc[("relevance", *g, "refined (v16)"), "agreement_pct"], 94.1)
check("judge kappa — faithfulness, Gemini refined", jv.loc[("faithfulness", *g, "refined (v14)"), "kappa"], 0.40, 0.005)
check("judge |bias| — faithfulness, Gemini refined", jv.loc[("faithfulness", *g, "refined (v14)"), "mean_abs_bias_pp"], 4.3)
check("judge kappa — faithfulness, Claude Opus 4.7 refined", jv.loc[("faithfulness", "Claude Opus 4.7", "refined (v14)"), "kappa"], 0.43, 0.005)
check("judge |bias| — faithfulness, Claude Opus 4.7 refined", jv.loc[("faithfulness", "Claude Opus 4.7", "refined (v14)"), "mean_abs_bias_pp"], 5.1)
check("judge agreement — faithfulness, GPT-5.4 refined", jv.loc[("faithfulness", "GPT-5.4", "refined (v14)"), "agreement_pct"], 72.7)
check("judge bias min — faithfulness, GPT-5.4 refined", jv.loc[("faithfulness", "GPT-5.4", "refined (v14)"), "bias_min_pp"], -24.6)
check("all judge-validation rows rebuilt from a source file", int((pd.read_csv(f"{D}/judge_validation_table.csv").status != "ok").sum()), 0)
o = ("Claude Opus 4.7", "refined (v16)")
check("judge agreement — relevance, Claude Opus 4.7 refined", jv.loc[("relevance", *o), "agreement_pct"], 93.8)
check("judge kappa — relevance, Claude Opus 4.7 refined", jv.loc[("relevance", *o), "kappa"], 0.38, 0.005)
check("judge |bias| — relevance, Claude Opus 4.7 refined", jv.loc[("relevance", *o), "mean_abs_bias_pp"], 2.3)
check("judge bias range — relevance, Claude Opus 4.7 refined (min)", jv.loc[("relevance", *o), "bias_min_pp"], 1.0)
check("judge bias range — relevance, Claude Opus 4.7 refined (max)", jv.loc[("relevance", *o), "bias_max_pp"], 3.3)

# Human inter-rater reliability (ED Table 9): 57 statements, 19 per reviewer pair
irr = pd.read_csv(f"{D}/inter_rater_overlap.csv")
pairs = irr[irr.metric == "faithfulness"].groupby(["reviewer_1", "reviewer_2"]).size()
check("IRR: three reviewer pairs", len(pairs), 3)
check("IRR: 19 dual-scored statements per pair (min)", int(pairs.min()), 19)
check("IRR: 19 dual-scored statements per pair (max)", int(pairs.max()), 19)
f = irr[irr.metric == "faithfulness"].groupby(["reviewer_1", "reviewer_2"]).agree.mean() * 100
check("IRR: faithfulness raw agreement range (min, 84.2)", round(f.min(), 1), 84.2)
check("IRR: faithfulness raw agreement range (max, 89.5)", round(f.max(), 1), 89.5)

# Reframed-manuscript additions (05_analysis/reframed_metrics.R, figures.R)
wc = pd.read_csv(f"{D}/within_model_correlations.csv")
eff = wc[wc.x.isin(["n_searches_run", "n_opened_run", "n_iterations_run", "recall_run", "recall_log"])]
check("max |r| retrieval effort vs faithfulness within model (< 0.13)", eff.pearson_r.abs().max(), 0.0, 0.13)
comp = pd.read_csv(f"{D}/composite_by_model.csv")
comp = comp[comp.recall_definition == "evaluated_run_paper"].set_index("model")
for m, want in {"Claude Sonnet 4": 0.591, "GPT-5": 0.589, "Kimi K2": 0.508, "Gemini 2.5 Pro": 0.453}.items():
    check(f"composite recall x faithfulness (summary-level mean) — {m}", comp.loc[m, "composite_mean"], want, 0.0006)
f4 = pd.read_csv(f"{D}/fig4_failure_profile_summary.csv").set_index(["model", "failure_reason"])
for m, want in {"Kimi K2": 80, "Claude Sonnet 4": 69, "GPT-5": 40, "Gemini 2.5 Pro": 33}.items():
    check(f"% summaries with >=1 absent/contradicted-fact statement — {m}", f4.loc[(m, "absent_or_contradicted_fact"), "pct_affected"], want, 0.5)
for m, want in {"Kimi K2": 54, "Claude Sonnet 4": 46, "GPT-5": 19, "Gemini 2.5 Pro": 11}.items():
    check(f"% summaries with >=1 overgeneralisation — {m}", f4.loc[(m, "overgeneralisation"), "pct_affected"], want, 0.5)
rk = pd.read_csv(f"{D}/metric_ranks.csv", index_col=0)
firsts = {c: rk[c].idxmin() for c in rk.columns if c.endswith("(rank)")}
check("number of distinct models ranked first across metrics", len(set(firsts.values())), 3, 0)

print(f"\n{'All checks passed' if not fails else f'{fails} check(s) failed'}")
sys.exit(bool(fails))
