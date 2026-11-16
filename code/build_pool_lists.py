# Adds data/mix_v1/lists.jsonl: harmless instruction rows (Alpaca + Dolly) whose ANSWER is a list (3+ numbered or bulleted lines).
# Same filters as build_pools.py (512 tokens, leak check against every held-out text). Rows may also be in the alpaca/dolly pools.
import csv, json, os, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as B
get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
out = Path(os.environ.get("DATA_DIR", "data")) / "mix_v1"
is_list = lambda a: sum(bool(re.match(r"\s*(\d+[.)]|[-*•])\s+\S", l)) for l in a.splitlines()) >= 3
ok = lambda a: 20 <= len(a) <= 800
rows = []
for i, r in enumerate(pq.read_table(get("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")).to_pylist()):
    if ok(r["output"]) and len(r["input"]) <= 600 and is_list(r["output"]):
        rows.append({"prompt": (r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")).strip(), "completion": r["output"].strip(), "lang": "en", "gid": f"alpaca-{i}", "source": "lists:alpaca"})
d = [r for r in B.jsonl(get("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")) if len(r["context"]) <= 600 and ok(r["response"])]
for i, r in enumerate(d):
    if is_list(r["response"]):
        rows.append({"prompt": (r["instruction"] + ("\n\n" + r["context"] if r["context"] else "")).strip(), "completion": r["response"].strip(), "lang": "en", "gid": f"dolly-{i}", "source": "lists:dolly"})
n_all = len(rows)
random.Random(45).shuffle(rows)
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
n_tok = lambda r: len(tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False), add_special_tokens=False)["input_ids"])
kept = []
for r in rows:
    if n_tok(r) <= 512:
        kept.append(r)
    if len(kept) >= 3200:
        break
from curatorkit.schema import DataSample
from curatorkit.splits import leakage_report
rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
skill, safety = rd(out / "skill_test.jsonl"), json.loads((out / "safety.json").read_text(encoding="utf-8"))
hb = pq.read_table(get("walledai/HarmBench", "standard/train-00000-of-00001.parquet")).to_pylist()
xs = pq.read_table(get("walledai/XSTest", "data/train-00000-of-00001.parquet")).to_pylist()
mj = list(csv.DictReader(open(get("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")))
held = [r["prompt"].split("Message: ", 1)[-1] for r in skill] + [p for v in safety.values() for p in v] + [r["prompt"] for r in hb] + [r["prompt"] for r in xs] + [r["en"] for r in mj]
S = lambda texts, tag: [DataSample(id=f"{tag}:{i}", source_uri=tag, task_type="prompt_only", instruction=t) for i, t in enumerate(texts)]
rep = leakage_report({"pool": S([r["prompt"] for r in kept], "pool"), "held": S(held, "held")}, near_dup_threshold=0.8)
bad = {int(i.split(":")[1]) for p in rep["exact_duplicates"] + rep["near_duplicates"] if str(p[0]).split(":")[0] != str(p[1]).split(":")[0] for i in p[:2] if str(i).startswith("pool:")}
kept = [r for i, r in enumerate(kept) if i not in bad][:3000]
(out / "lists.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
m = json.loads((out / "manifest.json").read_text())
m["counts"]["lists"] = {"list_format_rows_found": n_all, "final": len(kept), "from_alpaca": sum(r["source"] == "lists:alpaca" for r in kept), "from_dolly": sum(r["source"] == "lists:dolly" for r in kept)}
m["leak_check"]["lists"] = {"checked": len(kept) + len(bad), "held_out_texts": len(held), "overlaps_dropped": len(bad)}
m["sources"]["lists"] = "rows of tatsu-lab/alpaca and databricks-dolly-15k whose answer has 3+ numbered or bulleted lines"
(out / "manifest.json").write_text(json.dumps(m, indent=1))
print(m["counts"]["lists"], m["leak_check"]["lists"])
