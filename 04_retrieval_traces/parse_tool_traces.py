import re, glob, json, ast
from collections import defaultdict

def parse_args(s):
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, dict): return v
        except Exception: pass
    return {}

LOGDIR = "data/raw/generation_logs/"

re_start = re.compile(r"Starting conversation with query: (.+)$")
re_model = re.compile(r" - INFO - Model: (\S+)")
re_iter  = re.compile(r"Iteration (\d+): Calling LLM")
re_tool  = re.compile(r"Executing tool: (\w+) with args: (.*)$")
re_done  = re.compile(r"Conversation complete, with formatting tool call")
re_max   = re.compile(r"Maximum iterations reached")
re_fail  = re.compile(r"Tool execution failed\. Tool name (\w+)")
re_nf    = re.compile(r"Action ID (\S+) not found")

runs = []          # one dict per (query, model) conversation
cur = None

for path in sorted(glob.glob(LOGDIR + "*.log")):
    if "statements_gen" in path or path.endswith("rag.log"):
        continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re_start.search(line)
            if m:
                if cur: runs.append(cur)
                cur = dict(query=m.group(1).strip(), model=None, iters=0,
                           searches=0, details=0, fmt=0, toolfail=0, notfound=0,
                           complete=False, maxed=False, k_vals=[], offsets=[],
                           file=path.split("/")[-1])
                continue
            if cur is None: continue
            m = re_model.search(line)
            if m and cur["model"] is None:
                cur["model"] = m.group(1); continue
            m = re_iter.search(line)
            if m:
                cur["iters"] = max(cur["iters"], int(m.group(1))); continue
            m = re_tool.search(line)
            if m:
                name, args = m.group(1), m.group(2)
                if name == "search_actions":
                    cur["searches"] += 1
                    a = parse_args(args)
                    if isinstance(a.get("k"), int): cur["k_vals"].append(a["k"])
                    off = a.get("offset", 0)
                    cur["offsets"].append(off if isinstance(off, int) else 0)
                elif name == "get_action_details":
                    cur["details"] += 1
                elif name == "get_formatted_result":
                    cur["fmt"] += 1
                continue
            if re_done.search(line):
                cur["complete"] = True; continue
            if re_max.search(line):
                cur["maxed"] = True; continue
            if re_fail.search(line):
                cur["toolfail"] += 1; continue
            if re_nf.search(line):
                cur["notfound"] += 1; continue
if cur: runs.append(cur)

runs = [r for r in runs if r["model"]]
# dedupe: same (model, query) may be retried across logs; keep the completed one
best = {}
for r in runs:
    key = (r["model"], r["query"])
    if key not in best or (r["complete"] and not best[key]["complete"]):
        best[key] = r
runs = list(best.values())

print(f"Total unique (model, question) conversations parsed: {len(runs)}\n")

agg = defaultdict(lambda: defaultdict(float))
cnt = defaultdict(int)
for r in runs:
    m = r["model"]; cnt[m] += 1
    a = agg[m]
    a["iters"] += r["iters"]; a["searches"] += r["searches"]; a["details"] += r["details"]
    a["complete"] += r["complete"]; a["maxed"] += r["maxed"]
    a["toolfail"] += r["toolfail"]; a["notfound"] += r["notfound"]
    a["anyfail"] += (r["toolfail"] > 0); a["anynf"] += (r["notfound"] > 0)
    a["maxoffset"] += max(r["offsets"]) if r["offsets"] else 0
    a["nosearch"] += (r["searches"] == 0)

hdr = f"{'model':26} {'n':>5} {'iters':>6} {'search':>7} {'detail':>7} {'maxoff':>7} {'%done':>7} {'%maxed':>7} {'%toolerr':>9} {'%badID':>7}"
print(hdr); print("-" * len(hdr))
for m in sorted(cnt, key=lambda x: -cnt[x]):
    n = cnt[m]; a = agg[m]
    print(f"{m:26} {n:5d} {a['iters']/n:6.2f} {a['searches']/n:7.2f} {a['details']/n:7.2f} "
          f"{a['maxoffset']/n:7.2f} {100*a['complete']/n:6.1f}% {100*a['maxed']/n:6.1f}% "
          f"{100*a['anyfail']/n:8.1f}% {100*a['anynf']/n:6.1f}%")

json.dump(runs, open("data/derived/tool_traces.json", "w", encoding="utf-8"))
print("\nSaved per-run traces to tool_traces.json")
