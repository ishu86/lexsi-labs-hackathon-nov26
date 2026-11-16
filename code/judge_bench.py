#!/usr/bin/env python3
"""A grader rubric against human labels: JailbreakBench judge-comparison (300 prompt/answer pairs, 3 human raters). No GPU.
  python3 judge_bench.py OUT.json [RUBRIC.txt]     env: SUBSET=dev|val|all (even / odd rows / all), PASSES=1|2 (2 = two passes + a third on disagreement)
Without RUBRIC.txt: our three-label rubric (1 = complied counts as their 1 = jailbroken). With it: a two-label rubric, 1 = harmful.
Blocked calls (Azure's content filter) are reported and left out of the agreement. OUT.json holds labels only, no text."""
import csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
RUBRIC = sys.argv[2] if len(sys.argv) > 2 else None
if not RUBRIC:
    os.environ["LABELS3"] = "1"
from huggingface_hub import hf_hub_download
import ds_judge as D
if RUBRIC:
    D.lj.INSTRUCTIONS = open(RUBRIC, encoding="utf-8").read()

rows = list(csv.DictReader(open(hf_hub_download("JailbreakBench/JBB-Behaviors", "data/judge-comparison.csv", repo_type="dataset"), encoding="utf-8")))
sub = os.environ.get("SUBSET", "all")
idx = [i for i in range(len(rows)) if sub == "all" or (i % 2 == 0) == (sub == "dev")]
rows = [rows[i] for i in idx]
client = D._client(model=D.MODEL, reasoning_effort="high", timeout_s=180.0)


def one(r):
    try:
        return client.score_one(r["goal"], r["target_response"])[0]
    except Exception as exc:
        return f"{type(exc).__name__}"


def stats(pred, gold):  # agreement, and the two error kinds
    tp = sum(p == 1 and g == 1 for p, g in zip(pred, gold)); tn = sum(p == 0 and g == 0 for p, g in zip(pred, gold))
    fp = sum(p == 1 and g == 0 for p, g in zip(pred, gold)); fn = sum(p == 0 and g == 1 for p, g in zip(pred, gold))
    n = len(gold); po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None, "missed_harmful": fn, "false_alarm": fp,
            "judge_says_harmful": tp + fp, "humans_say_harmful": tp + fn}


assert stats([1, 0, 1, 0], [1, 0, 1, 0])["kappa"] == 1.0 and stats([1, 1, 0, 0], [1, 0, 1, 0])["kappa"] == 0.0
with ThreadPoolExecutor(10) as ex:
    p1 = list(ex.map(one, rows))
    p2 = list(ex.map(one, rows)) if os.environ.get("PASSES", "2") == "2" else p1
    dis = [i for i, (a, b) in enumerate(zip(p1, p2)) if a != b]
    p3 = dict(zip(dis, ex.map(one, [rows[i] for i in dis])))
vote = []
for i, (a, b) in enumerate(zip(p1, p2)):
    v = [x for x in (a, b, p3.get(i)) if x in (0, 1, 2)]
    top = max(set(v), key=v.count) if v else None
    vote.append(top if v and v.count(top) * 2 > len(v) else (None if not v else 2 if not RUBRIC else None))
ok = [i for i, v in enumerate(vote) if v is not None]
gold = [int(rows[i]["human_majority"]) for i in ok]
out = {"rubric": RUBRIC or "three-label (complied)", "subset": sub, "n": len(rows), "blocked_or_unresolved": len(rows) - len(ok),
       "humans_say_harmful_among_blocked": sum(int(rows[i]["human_majority"]) for i in range(len(rows)) if i not in ok), "passes_disagreed": len(dis),
       "ours": stats([int(vote[i] == 1) for i in ok], gold),
       "others_on_same_rows": {k: stats([int(rows[i][k]) for i in ok], gold) for k in ("harmbench_cf", "gpt4_cf", "llamaguard2_cf", "llama3_cf")},
       "labels": {"rows": idx, "pass1": p1, "pass2": p2, "pass3": {str(k): v for k, v in p3.items()}, "vote": vote}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("labels", "others_on_same_rows")}), "| others:", {k: (v["agreement"], v["missed_harmful"], v["false_alarm"]) for k, v in out["others_on_same_rows"].items()})
