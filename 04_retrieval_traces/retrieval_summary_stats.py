"""Per-model retrieval metrics and agentic-behaviour summaries with 95% bootstrap CIs.

Inputs (from 04_retrieval_traces/):
  data/derived/retrieval_recall.csv        one row per summary (true_retrieval_recall.py)
  data/derived/traces_eval_completed.csv   one row per completed, evaluated run (recheck_traces.py)
Outputs:
  data/derived/retrieval_metrics_by_model.csv   (Fig. 2; Results 'retrieval' subsection)
  data/derived/agent_behaviour_by_model.csv     (Extended Data behaviour table)
"""
import os
import numpy as np
import pandas as pd

D = os.environ.get("IGF_DERIVED", "data/derived")
B, SEED = 1000, 1

def boot(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    rng = np.random.default_rng(SEED)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)

def table(df, cols, scale):
    rows = []
    for model, g in df.groupby("model"):
        r = {"model": model}
        for c in cols:
            mu, lo, hi, n = boot(g[c])
            k = scale.get(c, 1)
            r.update({c: mu * k, f"{c}_lo": lo * k, f"{c}_hi": hi * k})
        r["n"] = len(g)
        rows.append(r)
    return pd.DataFrame(rows).round(3)

rr = pd.read_csv(f"{D}/retrieval_recall.csv")
ret = table(rr, ["ret_recall", "cite_recall", "ret_prec", "conv", "n_opened"],
            {"ret_recall": 100, "cite_recall": 100, "ret_prec": 100, "conv": 100})
ret.to_csv(f"{D}/retrieval_metrics_by_model.csv", index=False)

tr = pd.read_csv(f"{D}/traces_eval_completed.csv")
tr = tr[tr["in_eval"] & tr["complete"]]
# Actions opened (unique) comes from retrieval_recall.csv (n_opened); tool-error and
# attrition rates are computed over ALL runs incl. retries in recheck_traces.py.
beh = table(tr, ["iters", "searches", "max_offset"], {})
beh.to_csv(f"{D}/agent_behaviour_by_model.csv", index=False)

print(ret.to_string(index=False)); print(); print(beh.to_string(index=False))
