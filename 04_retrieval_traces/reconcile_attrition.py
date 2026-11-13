import json, glob
import pandas as pd

MAIN = ["openai/gpt-5", "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro", "moonshotai/kimi-k2-0905"]

runs = pd.DataFrame(json.load(open("data/derived/tool_traces.json", encoding="utf-8")))
runs = runs[runs.model.isin(MAIN)]

# The 520 evaluated questions
evalq = set()
for f in glob.glob("data/questions/bg_km_*_cited_statements.json"):
    for e in json.load(open(f, encoding="utf-8")):
        evalq.add(e["question_details"]["query"].strip())
print(f"Questions in evaluated corpus files: {len(evalq)}")
print(f"Unique query strings in logs:        {runs['query'].nunique()}\n")

runs["in_eval"] = runs["query"].str.strip().isin(evalq)
print("=== Are logged questions part of the evaluated corpus? ===")
print(runs.groupby("model")["in_eval"].agg(n="size", in_corpus="sum").to_string())

print("\n=== Cap rate: ALL logged questions vs EVALUATED-CORPUS questions only ===")
hdr = f"{'model':26} {'n_all':>6} {'cap_all':>8} {'n_eval':>7} {'cap_eval':>9}"
print(hdr); print("-" * len(hdr))
for m in MAIN:
    d = runs[runs.model == m]
    e = d[d.in_eval]
    print(f"{m:26} {len(d):6d} {100*d.maxed.mean():7.1f}% {len(e):7d} "
          f"{100*e.maxed.mean() if len(e) else float('nan'):8.1f}%")

# Per-question outcome among evaluated corpus: did the model ever succeed?
print("\n=== Per-question outcome, evaluated corpus only ===")
hdr2 = (f"{'model':26} {'qs_seen':>8} {'ever_done':>10} {'only_maxed':>11} "
        f"{'never_done':>11} {'missing/520':>12}")
print(hdr2); print("-" * len(hdr2))
summ = {"openai/gpt-5": 467, "anthropic/claude-sonnet-4": 515,
        "google/gemini-2.5-pro": 520, "moonshotai/kimi-k2-0905": 489}
for m in MAIN:
    e = runs[(runs.model == m) & runs.in_eval]
    g = e.groupby("query").agg(done=("complete", "max"), maxed=("maxed", "max"))
    only_maxed = ((~g.done) & g.maxed).sum()
    print(f"{m:26} {len(g):8d} {g.done.sum():10d} {only_maxed:11d} "
          f"{(~g.done).sum():11d} {520 - summ[m]:12d}")
