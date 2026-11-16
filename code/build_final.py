#!/usr/bin/env python3
"""Fresh held-out items for the final evaluation. No GPU, no paid calls.

  uv run -q --no-project --python 3.12 --with huggingface_hub --with pyarrow --with transformers --with jinja2 --with <CuratorKIT clone> python build_final.py [OUT_DIR]

Writes OUT_DIR (default data/final_v1/):
  skill_final.jsonl       360 MASSIVE intent items from the TEST split, 60 per language of build_data.LANGS (prompt, completion, lang, gid, source).
                          Excludes every MASSIVE id used in skill_test.jsonl or any training file (ids are shared across languages, so by id),
                          then drops any item that exact/near-duplicates (0.8, message text only) a training row or an old skill-test item.
  global_mmlu_lite.jsonl  40 questions x {en, ar, bn, sw, hi} from CohereLabs/Global-MMLU-Lite test: the same 40 sample_ids in every language
                          (id, lang, question, choices[4], answer A-D, subject, cultural_sensitivity_label).
  manifest.json           seed, counts, sources/licences, exclusion counts, leak-check numbers, sha256 (first 16 hex) per file.
"""
import gzip, hashlib, json, os, random, sys
from pathlib import Path

import pyarrow.parquet as pq

import build_data as B

SEED, N_INTENT, N_MMLU, MMLU_LANGS = 4242, 60, 40, ("en", "ar", "bn", "sw", "hi")
D = Path(os.environ.get("DATA_DIR", "data"))
TRAIN = ("ft_v1/train_massive.jsonl", "ft_v2/train_massive.jsonl", "mix_v1/intent.jsonl")
rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
msg = lambda r: r["prompt"].split("Message: ", 1)[-1]


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    old = rd(D / "mix_v1/skill_test.jsonl"); train = [r for f in TRAIN for r in rd(D / f)]
    used = {r["gid"] for r in old + train}
    # candidates: test split minus used ids, shuffled per language
    cand, excl = {}, {}
    for lang in B.LANGS:
        te = B.jsonl(B.get("mteb/amazon_massive_intent", f"test/{lang}.json.gz"), gzip.open)
        cand[lang] = [r for r in te if f"massive-{r['id']}" not in used]; excl[lang] = {"test_rows": len(te), "excluded_by_id": len(te) - len(cand[lang])}
        rng.shuffle(cand[lang])
    # leak check the whole candidate pool once, then walk each shuffled list taking clean items until 60
    from curatorkit.schema import DataSample
    from curatorkit.splits import leakage_report
    S = lambda texts, tag: [DataSample(id=f"{tag}:{i}", source_uri=tag, task_type="instruction_following", instruction=t, output="") for i, t in enumerate(texts)]
    flat = [(l, r) for l in B.LANGS for r in cand[l]]
    held = [msg(r) for r in old + train]
    rep = leakage_report({"new": S([r["text"] for _, r in flat], "new"), "old": S(held, "old")}, near_dup_threshold=0.8)
    cross = lambda ps: [p for p in ps if str(p[0]).split(":")[0] != str(p[1]).split(":")[0]]
    ex, nd = cross(rep["exact_duplicates"]), cross(rep["near_duplicates"])
    bad = {int(i.split(":")[1]) for p in ex + nd for i in p[:2] if str(i).startswith("new:")}
    rows, dropped, k = [], {}, 0
    for lang in B.LANGS:
        got, dropped[lang] = [], []
        for r in cand[lang]:
            if len(got) == N_INTENT:
                k += 1; continue
            if k in bad:
                dropped[lang].append(f"massive-{r['id']}")
            else:
                got.append({"prompt": B.PROMPT.format(text=r["text"]), "completion": r["label_text"], "lang": lang, "gid": f"massive-{r['id']}", "source": "massive"})
            k += 1
        assert len(got) == N_INTENT, lang
        rows += got
    gids = [r["gid"] for r in rows]
    assert not set(gids) & used and len(set(r["gid"] for r in rows if True)) == len(set(gids))
    labels = {r["completion"] for r in rows}; all_labels = {r["completion"] for r in old + train}
    cov = {l: len({r["completion"] for r in rows if r["lang"] == l}) for l in B.LANGS}

    # Global-MMLU-Lite: parallel sample_ids across languages
    g = lambda l: pq.read_table(B.get("CohereLabs/Global-MMLU-Lite", f"{l}/test-00000-of-00001.parquet")).to_pylist()
    m = {l: {r["sample_id"]: r for r in g(l)} for l in MMLU_LANGS}
    common = sorted(set.intersection(*(set(v) for v in m.values())))
    ids = sorted(random.Random(SEED).sample(common, N_MMLU))
    mm = [{"id": i, "lang": l, "question": m[l][i]["question"], "choices": [m[l][i][f"option_{c}"] for c in "abcd"], "answer": m[l][i]["answer"],
           "subject": m[l][i]["subject"], "cultural_sensitivity_label": m[l][i]["cultural_sensitivity_label"]} for l in MMLU_LANGS for i in ids]
    parallel_answers = all(m[l][i]["answer"] == m["en"][i]["answer"] for l in MMLU_LANGS for i in ids)

    dump = lambda name, rs: (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    dump("skill_final.jsonl", rows); dump("global_mmlu_lite.jsonl", mm)
    sha = lambda n: hashlib.sha256((out / n).read_bytes()).hexdigest()[:16]
    cnt = lambda rs: {l: sum(r["lang"] == l for r in rs) for l in sorted({r["lang"] for r in rs})}
    manifest = {"seed": SEED, "counts": {"skill_final": cnt(rows), "global_mmlu_lite": cnt(mm)},
                "sources": {"skill_final": "mteb/amazon_massive_intent test split (MASSIVE, CC BY 4.0)", "global_mmlu_lite": "CohereLabs/Global-MMLU-Lite test split (Apache-2.0)"},
                "exclusions": {"excluded_ids_total": len(used), "per_language": excl, "training_rows_checked": len(train), "old_skill_test_rows": len(old)},
                "leak_check": {"tool": "curatorkit.splits.leakage_report, exact + MinHash >= 0.8, message text only, candidate pool vs training rows + old skill test",
                               "candidates_checked": len(flat), "held_out_texts": len(held), "exact_pairs": len(ex), "near_pairs": len(nd), "candidates_flagged": len(bad),
                               "drawn_items_dropped_and_replaced": dropped},
                "label_coverage": {"labels_in_skill_final": len(labels), "labels_in_old_skill_plus_train": len(all_labels), "per_language": cov},
                "mmlu": {"n_questions": N_MMLU, "question_ids": ids, "parallel_ids": True, "answers_identical_across_languages": parallel_answers,
                         "cs_ca_counts": {x: sum(r["cultural_sensitivity_label"] == x for r in mm if r["lang"] == "en") for x in {r["cultural_sensitivity_label"] for r in mm}}},
                "sha256_16": {n: sha(n) for n in ("skill_final.jsonl", "global_mmlu_lite.jsonl")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("counts", "label_coverage", "leak_check", "sha256_16")} | {"excl": excl, "mmlu": {k: v for k, v in manifest["mmlu"].items() if k != "question_ids"}}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(D / "final_v1"))
