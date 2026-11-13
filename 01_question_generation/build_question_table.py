"""Build the canonical question table from the archived summary files.

Every summary file (data/summaries/<model>/orig_summaries/bg_km_<area>_summaries.json)
carries, for each question, the withheld reference pool (`all_relevant_action_ids`) that
was fixed at question generation. The pools are identical across the four models, so the
summary files are a complete and consistent record of the 523 filter-passing questions.
The 19 `bg_km_*_cited_statements.json` files in data/questions/ are per-model statement
snapshots and are missing 5 evaluated questions (3 Bat Conservation, 1 Bee Conservation,
1 Shrubland and Heathland Conservation); do not use them as the question list.

Output: data/questions/questions.csv with one row per question:
  question_id        Q0001... (stable, assigned by sorted synopsis then question text)
  synopsis           subject area
  question           question text (as posed to the summariser LLMs)
  n_reference_actions, reference_action_ids   withheld pool
  n_models_with_summary   how many of the 4 summariser LLMs produced a summary (0-4)
  evaluated          True for the 520 questions with at least one scored summary

Run from the repository root: python 01_question_generation/build_question_table.py
"""
import glob
import json
import os
import re

import pandas as pd

MODELS = {"_claude-sonnet-4": "anthropic/claude-sonnet-4", "_gemini-2-5-pro": "google/gemini-2.5-pro",
          "_gpt-5": "openai/gpt-5", "fireworks_kimi-k2-0905": "moonshotai/kimi-k2-0905"}

rec = {}
conflicts = 0
for d, model in MODELS.items():
    for f in sorted(glob.glob(f"data/summaries/{d}/orig_summaries/bg_km_*_summaries.json")):
        area = re.search(r"bg_km_(.*)_summaries\.json$", os.path.basename(f)).group(1)
        for e in json.load(open(f, encoding="utf-8")):
            q = e["query"].strip()
            pool = tuple(sorted(map(str, e.get("all_relevant_action_ids") or [])))
            r = rec.setdefault((area, q), {"pool": pool, "models": set()})
            if r["pool"] != pool:
                conflicts += 1
            if e.get("relevant_summary"):
                r["models"].add(model)

scored = set()
if os.path.exists("data/derived/summary_level.csv"):
    scored = set(pd.read_csv("data/derived/summary_level.csv")["question"].str.strip())

rows = []
for i, ((area, q), r) in enumerate(sorted(rec.items()), 1):
    rows.append({"question_id": f"Q{i:04d}", "synopsis": area, "question": q,
                 "n_reference_actions": len(r["pool"]), "reference_action_ids": ";".join(r["pool"]),
                 "n_models_with_summary": len(r["models"]), "evaluated": q in scored})
df = pd.DataFrame(rows)
os.makedirs("data/questions", exist_ok=True)
df.to_csv("data/questions/questions.csv", index=False)
print(f"questions: {len(df)}  evaluated (>=1 scored summary): {df.evaluated.sum()}  "
      f"pool conflicts across models: {conflicts}")
print(df.groupby("synopsis").size().to_string())
