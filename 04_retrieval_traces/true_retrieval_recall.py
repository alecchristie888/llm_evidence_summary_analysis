import re, glob, json, ast
import pandas as pd, numpy as np

LOGDIR = "data/raw/generation_logs/"
MAIN = ["openai/gpt-5", "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro", "moonshotai/kimi-k2-0905"]

def parse_args(s):
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, dict): return v
        except Exception: pass
    return {}

re_start = re.compile(r"Starting conversation with query: (.+)$")
re_model = re.compile(r" - INFO - Model: (\S+)")
re_tool  = re.compile(r"Executing tool: (\w+) with args: (.*)$")
re_done  = re.compile(r"Conversation complete, with formatting tool call")

def norm_ids(v):
    out = []
    if isinstance(v, (list, tuple, set)):
        for x in v: out += norm_ids(x)
    elif v is not None:
        out.append(str(v).strip())
    return out

runs, cur = [], None
for path in sorted(glob.glob(LOGDIR + "*.log")):
    if "statements_gen" in path or path.endswith("rag.log"):
        continue
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re_start.search(line)
        if m:
            if cur: runs.append(cur)
            cur = dict(query=m.group(1).strip(), model=None,
                       opened=set(), complete=False)
            continue
        if cur is None: continue
        m = re_model.search(line)
        if m and cur["model"] is None: cur["model"] = m.group(1); continue
        m = re_tool.search(line)
        if m and m.group(1) == "get_action_details":
            a = parse_args(m.group(2))
            for key in ("action_id", "action_ids", "ids", "id"):
                if key in a: cur["opened"].update(norm_ids(a[key]))
            continue
        if re_done.search(line): cur["complete"] = True; continue
if cur: runs.append(cur)

runs = [r for r in runs if r["model"] in MAIN]
print(f"Runs parsed: {len(runs)}")
print(f"Runs where action IDs were extracted: {sum(1 for r in runs if r['opened'])}")

# reference pools
pool = {}
for f in glob.glob("data/questions/bg_km_*_cited_statements.json"):
    for e in json.load(open(f, encoding="utf-8")):
        pool[e["question_details"]["query"].strip()] = set(
            map(str, e["question_details"].get("all_relevant_qu_ids") or []))

# cited actions
sa = {}
for r in json.load(open("data/judge_outputs/merged_all_records.json", encoding="utf-8")):
    sa.setdefault((r["summary_model"], r["question"].strip()), set()).update(
        map(str, r.get("summary_action_ids") or []))

# union opened across completed runs per (model, question)
opened = {}
for r in runs:
    if not r["complete"]: continue
    opened.setdefault((r["model"], r["query"]), set()).update(r["opened"])

rows = []
for (m, q), op in opened.items():
    p = pool.get(q)
    if not p or not op: continue
    cited = sa.get((m, q), set())
    rows.append({
        "model": m,
        "question": q,  # added so per-summary retrieval can be joined to judge scores
        "ret_recall": len(op & p) / len(p),
        "ret_prec": len(op & p) / len(op),
        "n_opened": len(op),
        "cite_recall": len(cited & p) / len(p),
        "n_cited": len(cited),
        # of the relevant actions it opened, how many did it go on to cite?
        "conv": len(cited & op & p) / len(op & p) if (op & p) else np.nan,
    })
df = pd.DataFrame(rows)
print(f"Summaries with both pool and opened IDs: {len(df)}\n")

print("=== TRUE retrieval (actions opened) vs citation recall ===")
out = df.groupby("model").agg(
    n=("n_opened", "size"), opened=("n_opened", "mean"),
    ret_recall=("ret_recall", "mean"), ret_prec=("ret_prec", "mean"),
    cite_recall=("cite_recall", "mean"), cited=("n_cited", "mean"),
    conversion=("conv", "mean"))
for c in ["ret_recall", "ret_prec", "cite_recall", "conversion"]:
    out[c] = (100 * out[c]).round(1)
print(out.round(2).to_string())
df.to_csv("data/derived/retrieval_recall.csv", index=False)
print("\nSaved to retrieval_recall.csv")
