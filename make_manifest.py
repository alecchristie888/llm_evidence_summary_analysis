"""Write MANIFEST.csv: one row per file under data/ and outputs/ with size, sha256 and,
for tabular/JSON files, the number of records. Re-run after any change to data/.

The manifest is the provenance anchor: a reader (or a co-author) can check that the
files they hold are byte-identical to the ones the paper's numbers were computed from.

Run from the repository root: python make_manifest.py
"""
import csv
import glob
import hashlib
import json
import os

ROOTS = ["data", "outputs"]
SKIP_DIRS = {"data/derived/tool_traces.json"}  # regenerated; large; hashed anyway


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def n_records(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return max(sum(1 for _ in csv.reader(f)) - 1, 0)
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            return len(obj) if isinstance(obj, (list, dict)) else 1
        if ext == ".log" or ext == ".txt":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
    except Exception:
        return ""
    return ""


rows = []
for root in ROOTS:
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isdir(path):
            continue
        rel = path.replace(os.sep, "/")
        stage = rel.split("/")[1] if rel.count("/") >= 1 else ""
        rows.append({
            "path": rel,
            "folder": "/".join(rel.split("/")[:2]),
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
            "n_records": n_records(path),
        })

with open("MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path", "folder", "bytes", "sha256", "n_records"])
    w.writeheader()
    w.writerows(rows)

by_folder = {}
for r in rows:
    d = by_folder.setdefault(r["folder"], [0, 0])
    d[0] += 1; d[1] += r["bytes"]
print(f"{len(rows)} files hashed -> MANIFEST.csv")
for k, (n, b) in sorted(by_folder.items()):
    print(f"  {k:40} {n:6} files  {b/1e6:8.1f} MB")
