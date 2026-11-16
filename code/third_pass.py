#!/usr/bin/env python3
"""Third grading pass only where passes 1 and 2 disagree (or one of them has no label).
  LABELS3=1 python3 third_pass.py ROOT1 ROOT2 ROOT3 [CONFIG,CONFIG...]
A majority of three is already decided when two passes agree, so this gives the same vote as three full passes at about two thirds
of the cost. Writes ROOT3/judge_deepseek3/<config>/<split>/chunk_0000.json in the layout vote.py reads. Text is never logged."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if not os.environ.get("RUBRIC_FILE"):  # with RUBRIC_FILE the kind is that rubric's (see ds_judge.py)
    os.environ["LABELS3"] = "1"
import ds_judge as D
import vote as V

KIND = D.lj.JUDGE_KIND

r1, r2, r3 = sys.argv[1:4]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
p1, p2 = V.scores(r1, KIND), V.scores(r2, KIND)
client = D._client(model=D.MODEL, reasoning_effort="high", timeout_s=180.0)
todo = {}
for cdir in sorted(Path(r1, "completions").iterdir()):
    for split in (("harmful",) if os.environ.get("SKIP_XSTEST") == "1" else ("harmful", "xstest")):  # SKIP_XSTEST=1 when passes 1-2 skipped the safe prompts
        f, out = cdir / f"{split}.json", Path(r3, KIND, cdir.name, split, "chunk_0000.json")
        if not f.exists() or out.exists() or (only and cdir.name not in only):
            continue
        d = json.loads(f.read_text())
        idx = [i for i in range(len(d["responses"])) if p1.get((cdir.name, split, i)) is None or p1.get((cdir.name, split, i)) != p2.get((cdir.name, split, i))]
        todo[(cdir.name, split)] = (d, idx, out)


def one(job):
    (name, split), (d, idx, out) = job
    scores, usd, errors = [None] * len(d["responses"]), 0.0, 0
    for i in idx:
        try:
            s, u = client.score_one(d["prompts"][i], d["responses"][i])  # a content-filter block returns None: a real "no label"
            scores[i], usd = (float(s) if s is not None else None), usd + u.get("usd", 0.0)
        except Exception as exc:
            errors += 1
            print(f"  {name}/{split} item {i}: {type(exc).__name__}", flush=True)
    if errors:  # network or parse errors: write nothing, so the next run grades this file again instead of freezing the gap
        print(f"  {name}/{split}: {errors} errors, file NOT written; rerun third_pass.py", flush=True)
        return 0, usd
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config_name": name, "split": split, "start": 0, "scores": scores, "third_pass_only_on_disagreement": True}))
    return len(idx), usd


with ThreadPoolExecutor(int(os.environ.get("CONC", "8"))) as ex:
    res = list(ex.map(one, todo.items()))
print(f"third pass: {sum(n for n, _ in res)} items regraded over {len(res)} files, usd={sum(u for _, u in res):.3f}")
