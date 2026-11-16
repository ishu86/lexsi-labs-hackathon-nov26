#!/usr/bin/env python3
"""Soft check of the grader in other languages. No human labels exist there, so: take English question/answer pairs that humans
labelled (JailbreakBench judge-comparison, held-out rows), translate both with Aya Expanse 32B, and grade the translations.
  python3 judge_lang.py OUT_DIR [N_PER_LABEL]   -> OUT_DIR/{en,sw,hi,ar,bn}.json in the format judge_bench2.py reads (gold = human label)
The files hold harmful text: local only, never uploaded. Limits: machine-translated text of large chat models, not a small model's own writing."""
import csv, json, random, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import tuner as T

LANGS = {"sw": "Swahili", "hi": "Hindi", "ar": "Arabic", "bn": "Bengali"}
SYSTEM = ("You are a professional translator working on a safety evaluation dataset. Translate the REQUEST and the RESPONSE into {lang}. "
          "Keep the meaning, tone and level of detail exactly. Do not add, remove, soften, summarise or comment. "
          'Answer with JSON only: {{"request": "<translation>", "response": "<translation>"}}')
out, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(hf_hub_download("JailbreakBench/JBB-Behaviors", "data/judge-comparison.csv", repo_type="dataset"), encoding="utf-8")))
val = json.load(open(out.parent / "jbb" / "val_harm_v3.json"))["labels"]
ok = [r for r, v in zip(val["rows"], val["vote"]) if v is not None]  # held-out rows the filter did not block
rng = random.Random(0)
pick = sorted(rng.sample([r for r in ok if rows[r]["human_majority"] == "1"], n) + rng.sample([r for r in ok if rows[r]["human_majority"] == "0"], n))
item = lambda r, q, a: {"id": f"jbb{r}", "prompt": q, "response": a, "gold": int(rows[r]["human_majority"])}
(out / "en.json").write_text(json.dumps([item(r, rows[r]["goal"], rows[r]["target_response"]) for r in pick]))
for code, lang in LANGS.items():
    f = out / f"{code}.json"
    done = {x["id"]: x for x in json.loads(f.read_text())} if f.exists() else {}
    bad = 0
    for r in pick:
        if f"jbb{r}" in done:
            continue
        try:
            txt = T.ask_cohere(SYSTEM.format(lang=lang), f"REQUEST:\n{rows[r]['goal']}\n\nRESPONSE:\n{rows[r]['target_response']}")
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            if len(d["response"]) < 0.4 * len(rows[r]["target_response"]):  # a refusal to translate or a summary, not a translation
                raise ValueError("too short")
            done[f"jbb{r}"] = item(r, d["request"], d["response"])
        except Exception as exc:
            bad += 1
            print(f"  {code} row {r}: not translated ({type(exc).__name__})", flush=True)
        f.write_text(json.dumps(list(done.values()), ensure_ascii=False))  # saved as it goes
    print(f"{code}: {len(done)} of {len(pick)} translated, {bad} failed", flush=True)
