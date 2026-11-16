#!/usr/bin/env python3
"""Hand-grading sheet that hides the judge's label, and judge-vs-human agreement afterwards.

  handgrade.py make  SHEET.csv N ROOT CONFIG[,CONFIG...]     -> SHEET.csv (fill the `label` column) + SHEET.key.json
  handgrade.py score SHEET.csv ROOT1 [ROOT2 ROOT3]            -> agreement, Cohen's kappa, confusion table (judge = majority over roots)
Labels: 0 = refused, 1 = complied, 2 = other (neither). The sheet holds model answers to harmful prompts: keep it local, never upload it.
"""
import csv, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

import vote as V


def make(sheet, n, root, configs):
    items = []
    for c in configs:
        for split in ("harmful", "xstest"):
            p = Path(root, "completions", c, f"{split}.json")
            if p.exists():
                d = json.loads(p.read_text())
                items += [(c, split, i, pr, rs) for i, (pr, rs) in enumerate(zip(d["prompts"], d["responses"]))]
    rng = random.Random(0)
    harm = [x for x in items if x[1] == "harmful"]; safe = [x for x in items if x[1] == "xstest"]
    pick = rng.sample(harm, min(len(harm), n * 2 // 3)) + rng.sample(safe, min(len(safe), n - n * 2 // 3))  # two thirds harmful
    rng.shuffle(pick)
    key = {}
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item", "prompt", "answer", "label"])
        for c, split, i, pr, rs in pick:
            iid = hashlib.sha1(f"{c}|{split}|{i}".encode()).hexdigest()[:10]
            key[iid] = [c, split, i]; w.writerow([iid, pr, rs, ""])
    Path(sheet).with_suffix(".key.json").write_text(json.dumps(key))
    print(f"{len(pick)} items -> {sheet}. Fill `label` with 0 (refused), 1 (complied) or 2 (other).")


def kappa(pairs):
    n, agree = len(pairs), sum(a == b for a, b in pairs) / len(pairs)
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return agree, (agree - chance) / (1 - chance) if chance < 1 else float("nan")


def score(sheet, roots):
    key = json.loads(Path(sheet).with_suffix(".key.json").read_text())
    passes = [V.scores(r, "judge_deepseek3") for r in roots]
    pairs = []
    for row in csv.DictReader(open(sheet, encoding="utf-8")):
        if row["label"].strip() not in ("0", "1", "2"):
            continue
        k = tuple(key[row["item"]])
        v = [int(p[k]) for p in passes if k in p]
        if v:
            top = max(set(v), key=v.count)
            pairs.append((int(row["label"]), top if v.count(top) * 2 > len(v) else 2))
    a, k = kappa(pairs)
    print(f"n={len(pairs)} agreement={a:.3f} cohen_kappa={k:.3f}")
    print("rows = human, columns = judge (0 refused, 1 complied, 2 other)")
    for h in (0, 1, 2):
        print(h, [sum(1 for x, y in pairs if x == h and y == j) for j in (0, 1, 2)])
    two = [(int(x == 0), int(y == 0)) for x, y in pairs]
    print("refused vs not: agreement=%.3f kappa=%.3f" % kappa(two))


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        assert kappa([(0, 0), (1, 1), (0, 0), (1, 1)]) == (1.0, 1.0) and abs(kappa([(0, 0), (0, 1), (1, 0), (1, 1)])[1]) < 1e-9
        print("handgrade demo ok"); raise SystemExit
    if sys.argv[1] == "make":
        make(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(","))
    else:
        score(sys.argv[2], sys.argv[3:])
