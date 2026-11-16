#!/usr/bin/env python3
"""Second, independent check of both rubrics against human labels, run once with the rubrics frozen (no tuning on these sets). No GPU.
  python3 judge_bench2.py SET.json OUT.json [RUBRIC.txt]
SET.json: [{"id", "prompt", "response", "gold"}]. With RUBRIC.txt: two-label harm rubric, gold 1 = unsafe (BeaverTails). Without: our
three-label rubric, gold = XSTest-v2 human label (1_full_compliance / 2_full_refusal / 3_partial_refusal).
Two passes + a third on disagreement, majority vote. OUT.json holds labels only, no text."""
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
RUBRIC = sys.argv[3] if len(sys.argv) > 3 else None
if RUBRIC:
    os.environ["RUBRIC_FILE"] = RUBRIC
else:
    os.environ["LABELS3"] = "1"
import ds_judge as D

rows = json.load(open(sys.argv[1]))
client = D._client(model=D.MODEL, reasoning_effort="high", timeout_s=180.0)


def one(r):
    try:
        return client.score_one(r["prompt"], r["response"])[0]
    except Exception:
        return None


with ThreadPoolExecutor(10) as ex:
    p1, p2 = list(ex.map(one, rows)), list(ex.map(one, rows))
    dis = [i for i, (a, b) in enumerate(zip(p1, p2)) if a != b]
    p3 = dict(zip(dis, ex.map(one, [rows[i] for i in dis])))
vote = []
for i, (a, b) in enumerate(zip(p1, p2)):
    v = [int(x) for x in (a, b, p3.get(i)) if x is not None]
    top = max(set(v), key=v.count) if v else None
    vote.append(top if v and v.count(top) * 2 > len(v) else (None if not v or RUBRIC else 2))
table = Counter((str(r["gold"]), v) for r, v in zip(rows, vote))
out = {"set": sys.argv[1], "rubric": RUBRIC or "three-label", "n": len(rows), "blocked": vote.count(None), "passes_disagreed": len(dis),
       "gold_x_judge": {f"{g} -> {v}": n for (g, v), n in sorted(table.items(), key=str)}, "labels": {"ids": [r["id"] for r in rows], "vote": vote}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "labels"}, indent=1))
