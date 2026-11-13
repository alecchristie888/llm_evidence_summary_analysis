import re, glob, json, ast
import pandas as pd

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
re_iter  = re.compile(r"Iteration (\d+): Calling LLM")
re_tool  = re.compile(r"Executing tool: (\w+) with args: (.*)$")
re_done  = re.compile(r"Conversation complete, with formatting tool call")
re_max   = re.compile(r"Maximum iterations reached")
re_fail  = re.compile(r"Tool execution failed\. Tool name (\w+)")
re_nf    = re.compile(r"Action ID (\S+) not found")
re_llmerr= re.compile(r"Error occurred while calling LLM")

# ---- keep ALL runs (no dedupe) ----
allruns, cur = [], None
for path in sorted(glob.glob(LOGDIR + "*.log")):
    if "statements_gen" in path or path.endswith("rag.log"):
        continue
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re_start.search(line)
        if m:
            if cur: allruns.append(cur)
            cur = dict(query=m.group(1).strip(), model=None, iters=0, searches=0,
                       details=0, toolfail=0, notfound=0, llmerr=0, complete=False,
                       maxed=False, offsets=[], ks=[], file=path.split("/")[-1])
            continue
        if cur is None: continue
        m = re_model.search(line)
        if m and cur["model"] is None: cur["model"] = m.group(1); continue
        m = re_iter.search(line)
        if m: cur["iters"] = max(cur["iters"], int(m.group(1))); continue
        m = re_tool.search(line)
        if m:
            name, a = m.group(1), parse_args(m.group(2))
            if name == "search_actions":
                cur["searches"] += 1
                off, k = a.get("offset", 0), a.get("k")
                cur["offsets"].append(off if isinstance(off, int) else 0)
                if isinstance(k, int): cur["ks"].append(k)
            elif name == "get_action_details":
                cur["details"] += 1
            continue
        if re_done.search(line):   cur["complete"] = True; continue
        if re_max.search(line):    cur["maxed"] = True; continue
        if re_fail.search(line):   cur["toolfail"] += 1; continue
        if re_nf.search(line):     cur["notfound"] += 1; continue
        if re_llmerr.search(line): cur["llmerr"] += 1; continue
if cur: allruns.append(cur)

df = pd.DataFrame([r for r in allruns if r["model"] in MAIN])
df["max_offset"] = df.offsets.apply(lambda o: max(o) if o else 0)
print(f"Total runs parsed (with retries): {len(df)}")

# ---- restrict to evaluated corpus ----
evalq = set()
for f in glob.glob("data/judge_outputs/merged_all_records.json"):
    for r in json.load(open(f, encoding="utf-8")):
        evalq.add(r["question"].strip())
print(f"Evaluated-corpus questions: {len(evalq)}")

df["in_eval"] = df["query"].isin(evalq)
ev = df[df.in_eval].copy()
print(f"Runs on evaluated questions: {len(ev)} "
      f"({ev.groupby('model')['query'].nunique().to_dict()})\n")

# ---- select the run that produced the scored summary: last completed run ----
ev["_ord"] = range(len(ev))
done = ev[ev.complete].sort_values("_ord").groupby(["model", "query"]).tail(1)
print(f"Completed runs (one per model-question): {len(done)}\n")

print("=== CORRECTED behavioural table (evaluated corpus, completed runs) ===")
t = done.groupby("model").agg(
    n=("query", "size"), iters=("iters", "mean"), searches=("searches", "mean"),
    docs=("details", "mean"), max_off=("max_offset", "mean"),
).round(2)
# error rates: over ALL runs on evaluated questions (errors can end a run)
err = ev.groupby("model").agg(
    runs=("query", "size"),
    pct_toolerr=("toolfail", lambda s: 100 * (s > 0).mean()),
    pct_badid=("notfound", lambda s: 100 * (s > 0).mean()),
    pct_llmerr=("llmerr", lambda s: 100 * (s > 0).mean()),
).round(2)
print(t.join(err).to_string())

print("\n=== Cap rate per question (evaluated corpus) ===")
g = ev.groupby(["model", "query"]).agg(done=("complete", "max"), maxed=("maxed", "max"))
for m in MAIN:
    d = g.loc[m]
    print(f"{m:26} questions {len(d):4d}  never completed {int((~d.done).sum()):3d} "
          f"({100*(~d.done).mean():4.1f}%)   hit cap at least once {int(d.maxed.sum()):3d}")

done.to_csv("data/derived/traces_eval_completed.csv", index=False)
t.join(err).to_csv("data/derived/agent_errors_by_model.csv")          # ED behaviour table: tool-error rate
att = g.reset_index().groupby("model").agg(
    questions=("query", "size"),
    never_completed=("done", lambda s: int((~s).sum())),
    pct_never_completed=("done", lambda s: round(100 * (~s).mean(), 1)),
    hit_cap_at_least_once=("maxed", "sum"))
att.to_csv("data/derived/attrition_by_model.csv")                      # Extended Data Table 9
print("\nSaved to traces_eval_completed.csv")
