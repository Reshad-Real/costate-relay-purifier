"""Inspect the downloaded outdoor PM2.5 files (data only; nothing in them is executed)."""
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TS = os.path.join(ROOT, "data", "external", "airnow_dhaka")
EPA = os.path.join(ROOT, "data", "external", "epa_dhaka")

rows = []
for f in sorted(os.listdir(TS)):
    if f.endswith(".json"):
        d = json.load(open(os.path.join(TS, f), encoding="utf-8"))
        print(f, "channel fields:", {k: d["channel"].get(k) for k in ("field1", "field2", "field3", "field4")}, "n =", len(d["feeds"]))
        rows += d["feeds"]
df = pd.DataFrame(rows)
print(df.head(3).to_string()); print(df.tail(3).to_string())
print("rows", len(df), "unique timestamps", df.created_at.nunique())
for f in sorted(os.listdir(EPA)):
    x = pd.ExcelFile(os.path.join(EPA, f))
    for sh in x.sheet_names:
        t = x.parse(sh, header=None, nrows=8)
        n = len(x.parse(sh, header=None))
        print(f"--- {f} / {sh}: {n} rows"); print(t.to_string()[:900])
