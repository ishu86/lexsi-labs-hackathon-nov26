# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "safetensors",
#   "transformers>=5.15",
#   "torch>=2.7",
#   "accelerate",
#   "peft>=0.18",
#   "datasets",
#   "huggingface_hub",
#   "numpy",
#   "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git",
#   "auditkit @ git+https://github.com/Lexsi-Labs/AuditKIT.git",
# ]
# ///
"""Screen harmless fine-tune mixes: short LoRA runs, one after another, on one loaded model.

For each candidate in SPECS_FILE (a JSON list in the repo):
  {"name", "parts": {"intent": 1200, "dolly": 1200, ...}, "lora_r": 16, "lr": 2e-4, "steps": 300}   ("parts": {} = the original model, no training)
it trains a fresh adapter on the first N rows of each pool (data/<DATA>/<pool>.jsonl), merges it, and measures
  skill           exact match on the 180 intent items
  kw_refusal      keyword-flagged refusals on the 160 CatQA harmful prompts, SCREEN_TOKENS new tokens (a refusal shows in the first words)
  kw_over         the same on 50 safe XSTest prompts
  *_actadd        all three again with the hand-picked repair (refusal direction added at layer 16, strength 5)
then restores the original weights exactly (checked) and moves on. Metrics and answers are committed per candidate; finished
candidates are skipped on relaunch. This is a screen: keyword grades, short answers, one seed. The mix that is chosen gets a full
fine-tune and the model-graded evaluation.
Env: HF_TOKEN, HF_CKPT_REPO, RUN_NAME, SPECS_FILE, DATA (mix_v1), MICRO, SCREEN_TOKENS (48), SCREEN_BATCH (16).
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("FULL", "1")
RUN = os.environ.setdefault("RUN_NAME", "mix_screen1")
os.environ.setdefault("CASE", "mix_screen")
REPO = os.environ["HF_CKPT_REPO"]
DATA = os.environ.setdefault("DATA", "mix_v1")
TOK, GB = int(os.environ.get("SCREEN_TOKENS", "48")), int(os.environ.get("SCREEN_BATCH", "16"))
T0 = time.time()

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, str(Path(hf_hub_download(REPO, "code/exp1_job.py", repo_type="dataset")).parent))
hf_hub_download(REPO, "code/finetune_job.py", repo_type="dataset")
import exp1_job as E  # noqa: E402
import finetune_job as F  # noqa: E402  (helpers only: encode, epoch_order, per_row_loss, apply_factors, skill_score)
import torch  # noqa: E402

log = E.log
BATCH, MICRO = F.BATCH, F.MICRO


def main():
    E.log_environment(); E.hf_login(); E.ensure_ckpt_repo(); E._seed_all()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: stopping in the first minute")
    api = HfApi()
    specs = json.loads(Path(hf_hub_download(REPO, os.environ["SPECS_FILE"], repo_type="dataset")).read_text())
    done = {f.rsplit("/", 1)[1][:-5] for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith(f"runs/{RUN}/metrics/")}
    todo = [s for s in specs if s["name"] not in done]
    log(f"RUN={RUN} candidates={len(specs)} todo={[s['name'] for s in todo]}")
    if not todo:
        return
    from peft import LoraConfig, get_peft_model
    from safetune.steer import RefusalDirectionModel

    model, tok = E.load_model_and_tok()
    rd = lambda name: [json.loads(l) for l in Path(hf_hub_download(REPO, f"data/{DATA}/{name}", repo_type="dataset")).read_text(encoding="utf-8").splitlines() if l.strip()]
    pools = {p: rd(f"{p}.jsonl") for p in sorted({p for s in todo for p in s["parts"]})}
    skill_rows = rd("skill_test.jsonl")
    safety = json.loads(Path(hf_hub_download(REPO, f"data/{DATA}/safety.json", repo_type="dataset")).read_text(encoding="utf-8"))
    harmful = safety["tune_harmful"] + safety["ladder_harmful"]
    loaded = E.load_data()
    xsafe, cap = [x for i, x in enumerate(loaded[3]) if i % 5 == 0], loaded[4]
    direction = E.load_direction()[0]
    pristine = {k: p.detach().clone().cpu() for k, p in model.named_parameters() if p.dim() == 2 and "embed" not in k}  # exact restore after each candidate
    probe = sorted(pristine)[::max(1, len(pristine) // 8)]

    def intact():
        """Are the base weights exactly the original ones? Compares 8 whole matrices spread over the model."""
        params = dict(model.named_parameters())
        return all(torch.equal(params[k].detach().cpu(), pristine[k]) for k in probe)
    gen = lambda m, ps: E.generate_completions(m, tok, ps, max_new_tokens=TOK, batch_size=GB)
    rate = lambda flags: sum(bool(f) for f in flags) / max(1, len(flags))

    def measure(m, name, rec, suffix=""):
        files = []
        rec["skill" + suffix], rec["skill_by_lang" + suffix] = F.skill_score(m, tok, skill_rows)
        for split, prompts, key in (("harmful", harmful, "kw_refusal"), ("xstest", xsafe, "kw_over")):
            resp = gen(m, prompts)
            flags = [bool(f) for f in E.string_refusal(resp)]
            rec[key + suffix] = rate(flags)
            p = E.RUN_LOG_DIR / "completions" / (name + suffix.replace("_", "__")) / f"{split}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"config_name": p.parent.name, "split": split, "n": len(prompts), "prompts": prompts, "responses": resp, "string_refusal": flags,
                                     "max_new_tokens": TOK, "gen_batch": GB, "seed": E.SEED, "dtype": str(E.DTYPE)}))
            files.append(p)
        return files

    def train(spec, rec):
        rows = [r for pool, n in spec["parts"].items() for r in pools[pool][:n]]
        enc, dropped = F.encode(tok, rows)
        r = int(spec.get("lora_r", 16))
        alpha, lr = float(spec.get("lora_alpha", 2 * r)), float(spec.get("lr", 2e-4))
        steps = int(spec.get("steps") or math.ceil(len(enc) * float(spec.get("epochs", 1)) / BATCH))
        pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
        try:
            tp = [p for p in pm.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(tp, lr=lr, weight_decay=0.0)
            perms, lm, ld, t = {}, [], [], time.time()

            def row(i):
                e = i // len(enc)
                if e not in perms:
                    perms[e] = F.epoch_order([len(x[0]) for x in enc], E.SEED, e, BATCH)
                return enc[perms[e][i % len(enc)]]

            pm.train()
            for step in range(steps):
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, (step + 1) / 30)
                full = [row(step * BATCH + j) for j in range(BATCH)]
                for j in range(0, BATCH, MICRO):
                    batch = full[j:j + MICRO]
                    width = max(len(b[0]) for b in batch)
                    ids = torch.tensor([b[0] + [tok.pad_token_id] * (width - len(b[0])) for b in batch], device=E.DEVICE)
                    lab = torch.tensor([b[1] + [-100] * (width - len(b[1])) for b in batch], device=E.DEVICE)
                    att = torch.tensor([[1] * len(b[0]) + [0] * (width - len(b[0])) for b in batch], device=E.DEVICE)
                    with torch.autocast("cuda", dtype=E.DTYPE):
                        out = pm(input_ids=ids, attention_mask=att, labels=lab)
                        loss = out.loss * len(batch) / BATCH
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"loss is {loss.item()} at step {step}")
                    with torch.no_grad():
                        for b, v in zip(batch, F.per_row_loss(out.logits, lab)):
                            (ld if b[2] else lm).append(v)
                    del out
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(tp, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step % 50 == 0 or step == steps - 1:
                    el = time.time() - t
                    log(f"  {spec['name']} step {step + 1}/{steps} intent={avg(lm[-150:])} free_text={avg(ld[-150:])} {el:.0f}s {(step + 1) * BATCH / max(el, 1e-9):.1f} seq/s "
                        f"gpu={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
            sd, factors = pm.state_dict(), {}
            for k in sd:
                if ".lora_A." in k:
                    name = k.split("base_model.model.", 1)[1].split(".lora_A.")[0] + ".weight"
                    factors[name] = {"A": sd[k].detach().float().cpu(), "B": sd[k.replace(".lora_A.", ".lora_B.")].detach().float().cpu()}
            rec.update(rows=len(enc), dropped_too_long=dropped, rows_by_pool={p: min(n, len(pools[p])) for p, n in spec["parts"].items()}, steps=steps, lora_r=r,
                       lora_alpha=alpha, lr=lr, train_s=round(time.time() - t), seq_per_s=round(steps * BATCH / max(time.time() - t, 1e-9), 1),
                       loss_intent_last=avg(lm[-200:]), loss_free_text_last=avg(ld[-200:]))
            return {"scaling": alpha / r, "factors": factors}
        finally:
            pm.unload()
            del pm
            torch.cuda.empty_cache()

    avg = lambda v: round(sum(v) / len(v), 4) if v else None
    for spec in todo:
        name, t, rec = spec["name"], time.time(), {"name": spec["name"], "spec": spec, "run": RUN, "screen_tokens": TOK, "gen_batch": GB, "n_harmful": len(harmful), "n_xstest": len(xsafe)}
        try:
            if spec["parts"]:
                payload = train(spec, rec)
                if not intact():
                    raise RuntimeError("the base weights changed during training")
                F.apply_factors(model, payload)
            files = measure(model, name, rec)
            os.environ["EVAL_STAGE_TAG"] = f"{RUN}:{name}"  # the MMLU cache is keyed on the model name
            rec["mmlu"] = float(E.eval_capability_checkpointed(model, tok, cap)[0])
            if os.environ.get("ACTADD", "1") == "1":  # the hand-picked steering repair on the same model: reported, not a selection criterion
                w = RefusalDirectionModel(model, direction, mode="steer", strength=5.0, layers=[16])
                w.install()
                try:
                    files += measure(model, name, rec, "_actadd")
                finally:
                    w.remove()
            rec["total_s"] = round(time.time() - t)
            p = E.RUN_LOG_DIR / "metrics" / f"{name}.json"
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(rec, indent=1))
            api.create_commit(REPO, [CommitOperationAdd(path_in_repo=f.relative_to(E.RUN_DIR).as_posix(), path_or_fileobj=str(f)) for f in files + [p]],
                              commit_message=f"{RUN} {name}", repo_type="dataset")
            log(f"CAND {name} rows={rec.get('rows_by_pool')} r={rec.get('lora_r')} lr={rec.get('lr')} steps={rec.get('steps')} | skill={rec['skill']:.3f} kw_refusal={rec['kw_refusal']:.3f} "
                f"kw_over={rec['kw_over']:.3f} mmlu={rec['mmlu']:.2f} | with actadd: " + (f"skill={rec['skill_actadd']:.3f} kw_refusal={rec['kw_refusal_actadd']:.3f} " if "skill_actadd" in rec else "not run ") +
                f"| loss intent={rec.get('loss_intent_last')} free_text={rec.get('loss_free_text_last')} {rec.get('seq_per_s')} seq/s total {rec['total_s']}s")
        except Exception as exc:  # one bad candidate must not sink the screen
            log(f"CAND {name} FAILED {type(exc).__name__}: {str(exc)[:300]}")
            p = E.RUN_LOG_DIR / "errors" / f"{name}.json"
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({"name": name, "spec": spec, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}))
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"runs/{RUN}/errors/{name}.json", repo_id=REPO, repo_type="dataset")
        finally:
            with torch.no_grad():
                params = dict(model.named_parameters())
                for k, v in pristine.items():
                    params[k].copy_(v.to(params[k].device))
            torch.cuda.empty_cache()
            if not intact():
                raise SystemExit("could not restore the original weights exactly: stopping, later candidates would be wrong")
    log(f"mix_screen_job done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
