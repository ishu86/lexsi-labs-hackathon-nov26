#!/usr/bin/env python3
"""Pools of harmless training rows for screening fine-tune mixes. No GPU.

  uv run --with huggingface_hub --with pyarrow --with transformers --with <CuratorKIT clone> python build_pools.py OUT_DIR FT_V2_DIR

Writes OUT_DIR/{intent,dolly,alpaca,aya}.jsonl (prompt, completion, lang, gid, source), each shuffled with a fixed seed so
"the first N rows" is a fair sample and a larger run is a superset of a smaller one; skill_test.jsonl and safety.json are copied
unchanged from FT_V2_DIR. Every row fits the 512-token training limit (the chat template's safety preamble alone is ~360 tokens)
and has passed the leak check against the skill test and every safety prompt.
  intent   MASSIVE intent rows of ft_v2 (6 languages; answer = one label)                       the task skill
  dolly    databricks-dolly-15k: human-written instruction/answer pairs, English
  alpaca   tatsu-lab/alpaca: synthetic instruction/answer pairs, English (CC BY-NC 4.0)
  aya      CohereLabs/aya_dataset: human-written prompt/answer pairs in Tamil, Amharic, Hindi, Swahili, Javanese
"""
import csv, json, random, shutil, sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import build_data as B

SEED, MAX_LEN, CAP = 44, 512, 6000
get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
ok_len = lambda a: 20 <= len(a) <= 800


def main(out_dir, ft_dir):
    out, src = Path(out_dir), Path(ft_dir); out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(SEED)
    pools = {"intent": rd(src / "train_massive.jsonl")}
    d = [r for r in B.jsonl(get("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")) if len(r["context"]) <= 600 and ok_len(r["response"])]
    pools["dolly"] = [{"prompt": (r["instruction"] + ("\n\n" + r["context"] if r["context"] else "")).strip(), "completion": r["response"].strip(), "lang": "en",
                       "gid": f"dolly-{i}", "source": "dolly:" + r["category"]} for i, r in enumerate(d)]
    a = [r for r in pq.read_table(get("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")).to_pylist() if ok_len(r["output"]) and len(r["input"]) <= 600]
    pools["alpaca"] = [{"prompt": (r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")).strip(), "completion": r["output"].strip(), "lang": "en",
                        "gid": f"alpaca-{i}", "source": "alpaca"} for i, r in enumerate(a)]
    y = pq.read_table(get("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet")).to_pylist()
    langs, per = {"tam": "ta", "amh": "am", "hin": "hi", "swh": "sw", "jav": "jv"}, {}
    for i, r in enumerate(y):
        if r["language_code"] in langs and ok_len(r["targets"]) and len(r["inputs"]) <= 600:
            per.setdefault(r["language_code"], []).append({"prompt": r["inputs"].strip(), "completion": r["targets"].strip(), "lang": langs[r["language_code"]],
                                                           "gid": f"aya-{i}", "source": "aya:" + r["language_code"]})
    pools["aya"] = [r for code in sorted(per) for r in rng.sample(per[code], min(len(per[code]), 1500))]  # Tamil has 14k rows: cap it so one language does not swamp the pool

    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    n_tok = lambda r: len(tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False),
                              add_special_tokens=False)["input_ids"])
    counts = {}
    for name in ("dolly", "alpaca", "aya"):
        rows = pools[name]; rng.shuffle(rows)
        kept = []
        for r in rows:  # token filter only until the cap is reached (tokenising 50k rows is slow and unneeded)
            if n_tok(r) <= MAX_LEN:
                kept.append(r)
            if len(kept) >= CAP + 200:
                break
        counts[name] = {"after_char_filter": len(rows), "kept_before_leak_check": len(kept)}
        pools[name] = kept

    # leak check: every pool against the skill test (by message text) and against every safety prompt
    from curatorkit.schema import DataSample
    from curatorkit.splits import leakage_report
    skill, safety = rd(src / "skill_test.jsonl"), json.loads((src / "safety.json").read_text(encoding="utf-8"))
    hb = pq.read_table(get("walledai/HarmBench", "standard/train-00000-of-00001.parquet")).to_pylist()
    xs = pq.read_table(get("walledai/XSTest", "data/train-00000-of-00001.parquet")).to_pylist()
    mj = list(csv.DictReader(open(get("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")))
    held = [r["prompt"].split("Message: ", 1)[-1] for r in skill] + [p for v in safety.values() for p in v] + [r["prompt"] for r in hb] + [r["prompt"] for r in xs] + [r["en"] for r in mj]
    S = lambda texts, tag: [DataSample(id=f"{tag}:{i}", source_uri=tag, task_type="prompt_only", instruction=t) for i, t in enumerate(texts)]
    leaks = {}
    for name in ("dolly", "alpaca", "aya"):
        rep = leakage_report({"pool": S([r["prompt"] for r in pools[name]], "pool"), "held": S(held, "held")}, near_dup_threshold=0.8)
        cross = [p for p in rep["exact_duplicates"] + rep["near_duplicates"] if str(p[0]).split(":")[0] != str(p[1]).split(":")[0]]
        bad = {int(i.split(":")[1]) for p in cross for i in p[:2] if str(i).startswith("pool:")}
        pools[name] = [r for i, r in enumerate(pools[name]) if i not in bad][:CAP]
        leaks[name] = {"checked": counts[name]["kept_before_leak_check"], "held_out_texts": len(held), "overlaps_dropped": len(bad)}
        counts[name]["final"] = len(pools[name])
    counts["intent"] = {"final": len(pools["intent"])}
    for name, rows in pools.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("prompt", "completion", "lang", "gid", "source")}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for f in ("skill_test.jsonl", "safety.json"):
        shutil.copy(src / f, out / f)
    by_lang = {}
    for r in pools["aya"]:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
    manifest = {"seed": SEED, "max_tokens": MAX_LEN, "counts": counts, "aya_rows_by_language": by_lang,
                "sources": {"intent": "data/ft_v2/train_massive.jsonl (mteb/amazon_massive_intent, leak-checked there)", "dolly": "databricks/databricks-dolly-15k (cc-by-sa-3.0)",
                            "alpaca": "tatsu-lab/alpaca (cc-by-nc-4.0)", "aya": "CohereLabs/aya_dataset train (apache-2.0), languages tam/amh/hin/swh/jav, at most 1,500 per language"},
                "leak_check": {"tool": "curatorkit.splits.leakage_report, exact + MinHash >= 0.8, each pool vs skill-test messages + CatQA + MultiJail (10 languages' English column and our subsets) + Aya Hindi + HarmBench + XSTest", **leaks}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"counts": counts, "aya_by_language": by_lang, "leaks": leaks}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
