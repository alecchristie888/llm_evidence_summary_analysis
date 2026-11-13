"""Verify that the files under data/ and outputs/ match MANIFEST.csv (sha256).
Run from the repository root: python check_manifest.py [--derived-only]
Exit code 1 if any file is missing or differs."""
import csv, hashlib, os, sys
only_derived = "--derived-only" in sys.argv
def sha256(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(block), b""): h.update(c)
    return h.hexdigest()
missing = changed = ok = 0
for r in csv.DictReader(open("MANIFEST.csv", encoding="utf-8")):
    if only_derived and not r["path"].startswith("data/derived"): continue
    if not os.path.exists(r["path"]): missing += 1; print("MISSING ", r["path"]); continue
    if sha256(r["path"]) != r["sha256"]: changed += 1; print("CHANGED ", r["path"]); continue
    ok += 1
print(f"\n{ok} files match, {changed} differ, {missing} missing")
sys.exit(1 if (changed or missing) else 0)
