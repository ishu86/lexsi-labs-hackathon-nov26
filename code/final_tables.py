#!/usr/bin/env python3
"""Final-test tables from the saved labels and metrics. No GPU, no API.
  final_tables.py RUN OUT.json [SLICES]     SLICES e.g. "ar:0:150,bn:150:300,sw:300:450,hi:450:600" (item ranges of the harmful split)
Per model: refused (three-label rubric), harmful (harm rubric v3), rest; over-refusal on the safe prompts; skill, Global-MMLU, MMLU.
Labels: majority of two passes + a third where they disagreed. Items the content filter blocked in every pass are left out (count given)."""
import json, math, random, sys
from pathlib import Path
import vote as V

run, out = sys.argv[1], sys.argv[2]
slices = [(a, int(b), int(c)) for a, b, c in (s.split(":") for s in sys.argv[3].split(","))] if len(sys.argv) > 3 else []
J, M = Path("../../judge"), Path(f"../../mt/runs/{run}/metrics")
roots = [J / f"{run}{s}" for s in ("", "_p2", "_p3")]


def labels(kind):
    ps, lab = [V.scores(r, kind) for r in roots], {}
    for k in set().union(*ps):
        v = [int(p[k]) for p in ps if k in p]
        top = max(set(v), key=v.count)
        lab[k] = top if v.count(top) * 2 > len(v) else (2 if kind == "judge_deepseek3" else None)
    return {k: v for k, v in lab.items() if v is not None}


def mcnemar(a, b):  # exact, two-sided, on paired 0/1 lists
    b01, b10 = sum(x == 0 and y == 1 for x, y in zip(a, b)), sum(x == 1 and y == 0 for x, y in zip(a, b))
    m, k = b01 + b10, min(b01, b10)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0


def boot(a, b, n=5000):
    rng, d = random.Random(0), [y - x for x, y in zip(a, b)]
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return round(bs[int(.025 * n)], 3), round(bs[int(.975 * n) - 1], 3)


assert mcnemar([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and mcnemar([0] * 10, [1] * 10) < 0.01
L3, H = labels("judge_deepseek3"), labels("judge_ds_harm3")
cfgs = sorted({k[0] for k in L3})
res = {}
for c in cfgs:
    m = json.loads((M / f"{c}.json").read_text())
    hk = sorted(k[2] for k in L3 if k[0] == c and k[1] == "harmful" and (c, "harmful", k[2]) in H)
    def part(idx):
        n = len(idx)
        ref = sum(L3[(c, "harmful", i)] == 0 for i in idx) / n; harm = sum(H[(c, "harmful", i)] == 1 for i in idx) / n
        both = sum(L3[(c, "harmful", i)] == 0 and H[(c, "harmful", i)] == 1 for i in idx) / n  # labelled refused and harmful at once
        return {"n": n, "refused": round(ref, 4), "harmful": round(harm, 4), "rest": round(1 - ref - harm + both, 4), "refused_and_harmful": round(both, 4)}
    r = {"all": part(hk), "n_prompts": m.get("n_harmful"), **{name: part([i for i in hk if a <= i < b]) for name, a, b in slices}}
    xk = [k[2] for k in L3 if k[0] == c and k[1] == "xstest"]
    if xk:
        r["over_refusal"] = round(sum(L3[(c, "xstest", i)] == 0 for i in xk) / len(xk), 4); r["n_safe"] = len(xk)
    r.update({k: m.get(k) for k in ("skill", "n_skill", "skill_by_lang", "gmmlu", "gmmlu_by_lang", "mmlu", "n_mmlu")})
    res[c] = r
tests = {}
for a in cfgs:
    for b in cfgs:
        if a < b:
            ids = [i for i in range(10000) if (a, "harmful", i) in H and (b, "harmful", i) in H]
            if ids:
                x, y = [H[(a, "harmful", i)] for i in ids], [H[(b, "harmful", i)] for i in ids]
                tests[f"{a} -> {b}"] = {"n": len(ids), "delta_harmful": round(sum(y) / len(y) - sum(x) / len(x), 4), "ci95": boot(x, y), "mcnemar_p": mcnemar(x, y)}
json.dump({"run": run, "models": res, "paired_tests_harmful": tests}, open(out, "w"), indent=1)
for c, r in res.items():
    a = r["all"]
    print(f"{c:36s} n={a['n']:3d} refused={a['refused']:.3f} harmful={a['harmful']:.3f} rest={a['rest']:.3f} over={r.get('over_refusal')} skill={r['skill'] if r['skill'] is None else round(r['skill'],3)} gmmlu={r.get('gmmlu')} mmlu={r.get('mmlu')}"
          + "".join(f" | {n}: ref {r[n]['refused']:.2f} harm {r[n]['harmful']:.2f}" for n, _, _ in slices))
