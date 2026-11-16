#!/usr/bin/env python3
"""Build the screening table: python3 mk_table.py <metrics_dir> <deepseek_results.json> <specs.json> > table.md
Rates are refusal rates. DeepSeek rates are over classified items only (content-filter blocks are left out).
"""
import json
import sys
from pathlib import Path

mdir, ds_path, specs_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ds = {r["config_name"]: r for r in json.load(open(ds_path))["rows"]}
specs = json.load(open(specs_path))
f = lambda x, d=3: "–" if x is None or x != x else f"{x:.{d}f}"
print("| spec | method | DS harmful refusal | DS over-refusal | string harmful | string over | MMLU | KL | apply s | peak GPU GB |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in specs:
    p = mdir / f"{s['name']}.json"
    if not p.exists():
        print(f"| {s['name']} | {s['method']} | not run | | | | | | | |")
        continue
    m, r = json.load(open(p)), ds.get(s["name"], {})
    hb, xb = r.get("n_harmful_blocked") or 0, r.get("n_xstest_blocked") or 0
    h = f(r.get("harmful_refusal_rate")) + (f" (n={r['n_harmful_classified']})" if hb else "")
    x = f(r.get("over_refusal_rate")) + (f" (n={r['n_xstest_classified']})" if xb else "")
    print(f"| {s['name']} | {s['method']} | {h} | {x} | {f(m['string_harmful_refusal'])} | {f(m['string_xstest_refusal'])} | "
          f"{f(m['mmlu'], 2)} | {f(m['kl'])} | {f(m['apply_s'], 1)} | {f(m['gpu_gb'], 1)} |")
