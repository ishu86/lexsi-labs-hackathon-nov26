#!/usr/bin/env python3
"""Build the benign fine-tune data and the safety prompt sets, with a leak check (CuratorKIT). No GPU.

  uv run --with huggingface_hub --with pyarrow --with <CuratorKIT clone> python build_data.py OUT_DIR

Writes to OUT_DIR (upload it to the private repo as data/ft_v1/):
  train_massive.jsonl   3,000 rows: MASSIVE intent, 6 languages x 500 (prompt, completion, lang, gid)
  train_dolly.jsonl     1,000 rows: Dolly instruction data (the dial: instruction data erodes safety most)
  skill_test.jsonl      180 rows: MASSIVE test split, 30 per language; exact match on the label is the skill score
  safety.json           prompt lists: tune_harmful (CatQA, 40), ladder_harmful (CatQA, 120, disjoint), multijail_{ar,bn,sw} (150, same rows), aya_hin (150)
  manifest.json         counts, sources, seeds, leak-check result (ids and numbers only, never prompt text)
The HarmBench / XSTest test prompts stay where they are (loaded in the job); they are only read here for the leak check.
"""
import csv, gzip, hashlib, json, random, sys
from pathlib import Path

from huggingface_hub import hf_hub_download

SEED = 42
LANGS = ("sw", "hi", "ta", "am", "cy", "jv")
PROMPT = "Classify the intent of this message. Answer with the intent label only.\n\nMessage: {text}"
CATQA_SKIP = {"Tailored Financial Advice", "Political Campaigning", "Adult Content"}  # weak harm: the original model need not refuse these
get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
jsonl = lambda p, op=open: [json.loads(l) for l in op(p, "rt", encoding="utf-8") if l.strip()]


def build():
    rng = random.Random(SEED)
    massive, skill = [], []
    for lang in LANGS:
        tr, te = (jsonl(get("mteb/amazon_massive_intent", f"{s}/{lang}.json.gz"), gzip.open) for s in ("train", "test"))
        for dst, src, n in ((massive, tr, 500), (skill, te, 30)):
            for r in rng.sample(src, n):
                dst.append({"prompt": PROMPT.format(text=r["text"]), "completion": r["label_text"], "lang": lang,
                            "gid": f"massive-{r['id']}", "text": r["text"], "source": "massive"})
    dolly = [r for r in jsonl(get("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"))
             if len(r["context"]) <= 600 and 20 <= len(r["response"]) <= 800]
    dolly = [{"prompt": (r["instruction"] + ("\n\n" + r["context"] if r["context"] else "")).strip(), "completion": r["response"].strip(),
              "lang": "en", "gid": f"dolly-{i}", "text": r["instruction"], "source": "dolly:" + r["category"]} for i, r in enumerate(rng.sample(dolly, 1000))]

    cat = [r for r in jsonl(get("declare-lab/CategoricalHarmfulQA", "data/catqa_english.json")) if r["Category"] not in CATQA_SKIP]
    tune, ladder = [], []  # ladder: a larger set from the same source, only to measure how much safety a fine-tune lost
    for c in sorted({r["Category"] for r in cat}):
        qs = [r["Question"] for r in rng.sample([r for r in cat if r["Category"] == c], 20)]
        tune += qs[:5]; ladder += qs[5:]
    mj = list(csv.DictReader(open(get("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")))
    mj_rows = sorted(rng.sample(range(len(mj)), 150))
    hin = jsonl(get("CohereLabs/aya_redteaming", "aya_hin.jsonl"))
    safety = {"tune_harmful": tune, "ladder_harmful": ladder, **{f"multijail_{l}": [mj[i][l] for i in mj_rows] for l in ("ar", "bn", "sw")},
              "aya_hin": [hin[i]["prompt"] for i in sorted(rng.sample(range(len(hin)), 150))]}
    return massive, dolly, skill, safety, [mj[i]["en"] for i in mj_rows]


def leak_check(massive, dolly, skill, safety, mj_en):
    import pyarrow.parquet as pq
    from curatorkit.schema import DataSample
    from curatorkit.splits import leakage_report
    # The instruction template is shared by every MASSIVE row, so compare the message itself.
    S = lambda rows, tag: [DataSample(id=f"{tag}:{i}", source_uri=tag, task_type="instruction_following", instruction=r["text"], output="", metadata={"gid": r["gid"]})
                           for i, r in enumerate(rows)]
    P = lambda ps, tag: [DataSample(id=f"{tag}:{i}", source_uri=tag, task_type="prompt_only", instruction=p) for i, p in enumerate(ps)]
    hb = pq.read_table(get("walledai/HarmBench", "standard/train-00000-of-00001.parquet")).to_pylist()
    xs = pq.read_table(get("walledai/XSTest", "data/train-00000-of-00001.parquet")).to_pylist()
    out = {}
    for name, splits, gk in (
        ("finetune_train_vs_skill_test", {"train": S(massive, "massive") + S(dolly, "dolly"), "skill_test": S(skill, "skill")}, "gid"),
        ("tune_harmful_vs_test_harmful", {"tune": P(safety["tune_harmful"] + safety["ladder_harmful"], "catqa"), "harmbench": P([r["prompt"] for r in hb], "hb"),
                                          "multijail_en": P(mj_en, "mj")}, None),
        ("finetune_train_vs_safety_prompts", {"train": S(massive, "massive") + S(dolly, "dolly"),
                                              "safety": P(safety["tune_harmful"] + safety["ladder_harmful"] + mj_en + [r["prompt"] for r in hb] + [r["prompt"] for r in xs], "safety")}, None),
    ):
        rep = leakage_report(splits, group_key=gk, near_dup_threshold=0.8)
        # near duplicates inside one split are not leaks; keep only pairs that cross splits
        cross = lambda pairs: [list(p) for p in pairs if str(p[0]).split(":")[0] != str(p[1]).split(":")[0]]
        out[name] = {"sizes": {k: len(v) for k, v in splits.items()}, "groups_in_multiple_splits": len(rep["groups_in_multiple_splits"]),
                     "exact_duplicates": cross(rep["exact_duplicates"]), "near_duplicates": cross(rep["near_duplicates"])}
    return out


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    massive, dolly, skill, safety, mj_en = build()
    leaks = leak_check(massive, dolly, skill, safety, mj_en)
    # Drop the training rows that leak into the skill test (exact or near); the test rows stay fixed.
    bad = {int(i.split(":")[1]) for pair in leaks["finetune_train_vs_skill_test"]["exact_duplicates"] + leaks["finetune_train_vs_skill_test"]["near_duplicates"]
           for i in pair[:2] if i.startswith("massive:")}
    kept = [r for i, r in enumerate(massive) if i not in bad]
    dump = lambda rows, name: (out / name).write_text("".join(json.dumps({k: r[k] for k in ("prompt", "completion", "lang", "gid", "source")}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dump(kept, "train_massive.jsonl"); dump(dolly, "train_dolly.jsonl"); dump(skill, "skill_test.jsonl")
    (out / "safety.json").write_text(json.dumps(safety, ensure_ascii=False), encoding="utf-8")
    sha = lambda name: hashlib.sha256((out / name).read_bytes()).hexdigest()[:16]
    manifest = {"seed": SEED, "prompt_template": PROMPT, "languages": LANGS,
                "counts": {"train_massive_before_leak_drop": len(massive), "train_massive": len(kept), "train_dolly": len(dolly), "skill_test": len(skill),
                           "labels_in_train": len({r["completion"] for r in kept}), **{k: len(v) for k, v in safety.items()}},
                "sources": {"train_massive/skill_test": "mteb/amazon_massive_intent train/test splits (disjoint ids)", "train_dolly": "databricks/databricks-dolly-15k (context <= 600 chars, response 20-800 chars)",
                            "tune_harmful": "declare-lab/CategoricalHarmfulQA english, 5 (tune) + 15 (ladder) per category, 8 categories (skipped: " + ", ".join(sorted(CATQA_SKIP)) + ")",
                            "multijail_*": "DAMO-NLP-SG/MultiJail, the same 150 of 315 rows in ar, bn, sw", "aya_hin": "CohereLabs/aya_redteaming aya_hin.jsonl, 150 of 915"},
                "leak_check": {"tool": "curatorkit.splits.leakage_report (exact after case/whitespace folding; MinHash near-duplicates at Jaccard >= 0.8)", **leaks},
                "sha256_16": {n: sha(n) for n in ("train_massive.jsonl", "train_dolly.jsonl", "skill_test.jsonl", "safety.json")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"counts": manifest["counts"], "leak_check": {k: {"sizes": v["sizes"], "groups": v["groups_in_multiple_splits"], "exact": len(v["exact_duplicates"]),
                                                                       "near": len(v["near_duplicates"])} for k, v in leaks.items()}}, indent=1))


def extend(out_dir, from_dir, n_per_lang):
    """A larger training set for the same task: skill test and safety prompts are copied unchanged from FROM_DIR (so runs stay
    comparable); only train_massive is redrawn, n_per_lang rows per language, and leak-checked again. Dolly rows are reused."""
    out, src = Path(out_dir), Path(from_dir); out.mkdir(parents=True, exist_ok=True)
    rd = lambda name: [json.loads(l) for l in (src / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    skill, dolly, safety = rd("skill_test.jsonl"), rd("train_dolly.jsonl"), json.loads((src / "safety.json").read_text(encoding="utf-8"))
    rng, massive = random.Random(SEED + 1), []
    for lang in LANGS:
        for r in rng.sample(jsonl(get("mteb/amazon_massive_intent", f"train/{lang}.json.gz"), gzip.open), n_per_lang):
            massive.append({"prompt": PROMPT.format(text=r["text"]), "completion": r["label_text"], "lang": lang, "gid": f"massive-{r['id']}", "text": r["text"], "source": "massive"})
    for rows in (skill, dolly):  # the message itself, for the leak check
        for r in rows:
            r["text"] = r["prompt"].split("Message: ", 1)[-1] if r["source"] == "massive" else r["prompt"]
    mj = list(csv.DictReader(open(get("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")))
    leaks = leak_check(massive, dolly, skill, safety, [r["en"] for r in mj])
    lk = leaks["finetune_train_vs_skill_test"]
    bad = {int(i.split(":")[1]) for pair in lk["exact_duplicates"] + lk["near_duplicates"] for i in pair[:2] if i.startswith("massive:")}
    kept = [r for i, r in enumerate(massive) if i not in bad]
    rng.shuffle(kept)  # languages mixed, so the first N rows of the file are a fair sample too
    dump = lambda rows, name: (out / name).write_text("".join(json.dumps({k: r[k] for k in ("prompt", "completion", "lang", "gid", "source")}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dump(kept, "train_massive.jsonl"); dump(dolly, "train_dolly.jsonl"); dump(skill, "skill_test.jsonl")
    (out / "safety.json").write_text(json.dumps(safety, ensure_ascii=False), encoding="utf-8")
    manifest = {"seed": SEED + 1, "extends": str(src.name), "prompt_template": PROMPT, "languages": LANGS,
                "counts": {"train_massive_before_leak_drop": len(massive), "train_massive": len(kept), "train_dolly": len(dolly), "skill_test": len(skill),
                           "labels_in_train": len({r["completion"] for r in kept})},
                "note": "skill_test.jsonl, train_dolly.jsonl and safety.json are byte-identical in content to the set this extends; MultiJail-en in the leak check is all 315 rows",
                "leak_check": {"tool": "curatorkit.splits.leakage_report (exact after case/whitespace folding; MinHash near-duplicates at Jaccard >= 0.8)", **leaks}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"counts": manifest["counts"], "leak_check": {k: {"sizes": v["sizes"], "groups": v["groups_in_multiple_splits"], "exact": len(v["exact_duplicates"]),
                                                                       "near": len(v["near_duplicates"])} for k, v in leaks.items()}}))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--extend":  # build_data.py OUT --extend FROM_DIR N_PER_LANG
        extend(sys.argv[1], sys.argv[3], int(sys.argv[4]))
    else:
        main(sys.argv[1])
