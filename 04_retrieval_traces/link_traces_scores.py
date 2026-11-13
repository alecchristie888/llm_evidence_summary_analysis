import json, glob, pandas as pd

MAIN = ["openai/gpt-5", "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro", "moonshotai/kimi-k2-0905"]

tr = pd.DataFrame(json.load(open("data/derived/tool_traces.json", encoding="utf-8")))
tr = tr[tr.model.isin(MAIN)].copy()
tr["docs_seen"] = tr.details

sl = pd.read_csv("data/derived/summary_level.csv")
sl = sl.rename(columns={"summary_model": "model", "question": "query"})

df = tr.merge(sl, on=["model", "query"], how="inner")
print(f"Traces: {len(tr)} | Scored summaries: {len(sl)} | Matched: {len(df)}\n")

# gold pool recall
pool = {}
for f in glob.glob("data/questions/bg_km_*_cited_statements.json"):
    for e in json.load(open(f, encoding="utf-8")):
        pool[e["question_details"]["query"]] = set(e["question_details"].get("all_relevant_qu_ids") or [])

sa = {}
for r in json.load(open("data/judge_outputs/merged_all_records.json", encoding="utf-8")):
    sa[(r["summary_model"], r["question"])] = set(r.get("summary_action_ids") or [])

rows = []
for _, r in df.iterrows():
    p = pool.get(r["query"])
    u = sa.get((r["model"], r["query"]))
    if p and u is not None and len(p) > 0:
        rows.append({"model": r["model"], "recall": len(u & p) / len(p),
                     "precision": len(u & p) / max(len(u), 1)})
rec = pd.DataFrame(rows)

print("=== Agentic effort vs outcome (per-summary means) ===")
out = df.groupby("model").agg(
    n=("query", "size"), iters=("iters", "mean"), searches=("searches", "mean"),
    docs=("docs_seen", "mean"), maxed=("maxed", "mean"),
    relevance=("relevance_score", "mean"), cite_rate=("citation_rate", "mean"),
    cc_cited=("cc_cited_score", "mean"), faith=("faithfulness_score", "mean"),
).round(3)
out["maxed"] = (100 * out["maxed"]).round(1)
for c in ["relevance", "cite_rate", "cc_cited", "faith"]:
    out[c] = (100 * out[c]).round(1)
print(out.to_string())

print("\n=== Gold-pool retrieval recall / precision ===")
print((100 * rec.groupby("model")[["recall", "precision"]].mean()).round(1).to_string())

print("\n=== Within-model: does more searching help? (Pearson r) ===")
for m in MAIN:
    d = df[df.model == m]
    r1 = d["searches"].corr(d["faithfulness_score"])
    r2 = d["docs_seen"].corr(d["faithfulness_score"])
    r3 = d["searches"].corr(d["relevance_score"])
    print(f"{m:26} searches~faith {r1:+.3f}   docs~faith {r2:+.3f}   searches~relev {r3:+.3f}")

print("\n=== Loop-exhausted vs completed summaries ===")
for m in MAIN:
    d = df[df.model == m]
    a, b = d[d.maxed], d[~d.maxed]
    if len(a) >= 5:
        print(f"{m:26} maxed n={len(a):3d} faith {100*a.faithfulness_score.mean():5.1f}% "
              f"cite {100*a.citation_rate.mean():5.1f}%  |  ok n={len(b):3d} "
              f"faith {100*b.faithfulness_score.mean():5.1f}% cite {100*b.citation_rate.mean():5.1f}%")

df.to_csv("data/derived/traces_x_scores.csv", index=False)
print("\nSaved merged data to traces_x_scores.csv")
