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
"""Benign LoRA fine-tune of the aligned model -> a "case" the repair job can load.

Trains on data/<DATA>/train_massive.jsonl (+ DOLLY_N rows of train_dolly.jsonl), loss on the answer tokens only.
Saves to the private repo:
  cases/<CASE>/lora_factors.pt   {"scaling", "factors": {weight name: {"A", "B"}}}; the case is aligned + scaling * B @ A
  cases/<CASE>/meta.json         settings, loss curve, skill before/after (exact match on skill_test.jsonl), MMLU, timings
  runs/<RUN_NAME>/completions/<CASE>/{harmful,xstest}.json (+ aligned/ with ALIGNED_EVAL=1), same layout the grader reads
Env: HF_TOKEN, HF_CKPT_REPO, CASE, RUN_NAME, DATA (ft_v1), N_MASSIVE (0 = all), DOLLY_N, EPOCHS, MAX_STEPS (8 = smoke), LR, LORA_R,
     CKPT_S (seconds between checkpoints), EVAL_EVERY (steps between mid-run measurements + snapshots cases/<CASE>_s<step>; 0 = none), RESUME_FROM (an earlier case to continue training from),
     LORA_ALPHA, BATCH (sequences per optimizer step), MICRO (sequences per forward pass; 8 at once does not fit an L4), MAX_LEN, EVAL_SETS (names in safety.json, joined into the harmful split), ALIGNED_EVAL, BATCH_CHECK.
ponytail: plain peft + a hand-written loop (AlignTune was read and not used: full-text loss by default, ~50 dependencies).
Reuses exp1_job.py from the repo for loading, generation, MMLU and uploads, like repair_job.py.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("FULL", "1")
RUN = os.environ.setdefault("RUN_NAME", "ft_smoke")
REPO = os.environ["HF_CKPT_REPO"]
CASE = os.environ.get("CASE", "ft_smoke")
DATA = os.environ.get("DATA", "ft_v1")
env = lambda k, d, t=int: t(os.environ.get(k, d))
N_MASSIVE, DOLLY_N, EPOCHS, MAX_STEPS = env("N_MASSIVE", 0), env("DOLLY_N", 1000), env("EPOCHS", 3, float), env("MAX_STEPS", 0)
LR, LORA_R, LORA_ALPHA, BATCH, MAX_LEN = env("LR", 2e-4, float), env("LORA_R", 16), env("LORA_ALPHA", 32), env("BATCH", 8), env("MAX_LEN", 512)
MICRO, CKPT_S, EVAL_EVERY = env("MICRO", 2), env("CKPT_S", 300), env("EVAL_EVERY", 0)
EVAL_SETS = [s for s in os.environ.get("EVAL_SETS", "tune_harmful,ladder_harmful").split(",") if s]
MID_GEN = os.environ.get("MID_GEN", "1") == "1"  # 0 = mid-run points measure skill only and save the snapshot (about 5 min cheaper each)
PARTS = json.loads(os.environ.get("PARTS", "{}"))  # {"intent": 6400, "dolly": 5600}: first N rows of data/<DATA>/<pool>.jsonl each (pools as in mix_screen_job.py)
T0 = time.time()

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, str(Path(hf_hub_download(REPO, "code/exp1_job.py", repo_type="dataset")).parent))
import exp1_job as E  # noqa: E402
import torch  # noqa: E402

log = E.log
data_file = lambda name: Path(hf_hub_download(REPO, f"data/{DATA}/{name}", repo_type="dataset"))
jsonl = lambda name: [json.loads(l) for l in data_file(name).read_text(encoding="utf-8").splitlines() if l.strip()]


def skill_score(model, tok, rows):
    """Exact match of the first line of the answer against the intent label."""
    out = E.generate_completions(model, tok, [r["prompt"] for r in rows], max_new_tokens=16)
    hit = [(o.strip().splitlines() or [""])[0].strip().lower() == r["completion"].lower() for o, r in zip(out, rows)]
    by = {}
    for h, r in zip(hit, rows):
        by.setdefault(r["lang"], []).append(h)
    return sum(hit) / len(hit), {k: round(sum(v) / len(v), 3) for k, v in by.items()}


def encode(tok, rows):
    """-> [(input_ids, labels)] with labels = -100 on everything before the answer. Rows longer than MAX_LEN are dropped."""
    out, dropped = [], 0
    for r in rows:
        pre = E.format_chat(tok, r["prompt"])
        full = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False)
        assert full.startswith(pre), "chat template: the prompt is not a prefix of prompt + answer"
        a, b = tok(pre, add_special_tokens=False)["input_ids"], tok(full, add_special_tokens=False)["input_ids"]
        assert b[:len(a)] == a and len(b) > len(a)
        if len(b) > MAX_LEN:
            dropped += 1
            continue
        out.append((b, [-100] * len(a) + b[len(a):], r["source"] != "massive"))  # third field: a Dolly (free text) row
    return out, dropped


def epoch_order(lengths, seed, epoch, batch):
    """Row order for one epoch. Shuffled, then rows of similar length are put in the same batch (less padding, so faster),
    then the batches are shuffled. Depends only on (seed, epoch), so a resumed run sees the same order."""
    rng = random.Random(f"{seed}:{epoch}")
    idx = rng.sample(range(len(lengths)), len(lengths))
    groups = []
    for i in range(0, len(idx), batch * 32):
        chunk = sorted(idx[i:i + batch * 32], key=lambda j: lengths[j])
        groups += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(groups)
    return [j for g in groups for j in g]


def per_row_loss(logits, lab):
    """Mean answer-token loss of each row of a batch (for the log only: intent rows and free-text rows have very different floors).
    One row at a time: the vocabulary is 262k wide, so all rows at once in fp32 costs gigabytes."""
    out = []
    for i in range(lab.shape[0]):
        tg = lab[i, 1:]
        m = tg != -100
        out.append(float(torch.nn.functional.cross_entropy(logits[i, :-1][m].float(), tg[m])))
    return out


def factors_of(pm):
    sd, factors = pm.state_dict(), {}
    for k in sd:
        if ".lora_A." in k:
            name = k.split("base_model.model.", 1)[1].split(".lora_A.")[0] + ".weight"
            factors[name] = {"A": sd[k].detach().float().cpu(), "B": sd[k.replace(".lora_A.", ".lora_B.")].detach().float().cpu()}
    return {"scaling": LORA_ALPHA / LORA_R, "factors": factors}


def apply_factors(model, payload):
    """aligned -> case, the same arithmetic repair_job.py uses: W := (W + scaling * B @ A) in fp32, cast back."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, f in payload["factors"].items():
            p = params[k]
            p.copy_((p.float() + payload["scaling"] * (f["B"].to(p.device).float() @ f["A"].to(p.device).float())).to(p.dtype))
    return len(payload["factors"])


def commit(api, ops, msg):
    """Commit with retries: a network blip at the end of a paid run must not lose the result."""
    for attempt in range(4):
        try:
            return api.create_commit(REPO, ops, commit_message=msg, repo_type="dataset")
        except Exception as exc:
            log(f"WARNING commit failed ({attempt + 1}/4) {type(exc).__name__}: {str(exc)[:200]}")
            time.sleep(20 * (attempt + 1))
    raise SystemExit(f"cannot save to the repo: {msg}")


def main():
    E.log_environment(); E.hf_login(); E.ensure_ckpt_repo(); E._seed_all()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: stopping in the first minute")
    api, rng = HfApi(), random.Random(E.SEED)
    model, tok = E.load_model_and_tok()
    skill_rows, safety = jsonl("skill_test.jsonl"), json.loads(data_file("safety.json").read_text(encoding="utf-8"))
    if PARTS:
        train = [r for pool, n in PARTS.items() for r in jsonl(f"{pool}.jsonl")[:n]]
    else:
        train = jsonl("train_massive.jsonl")
        train = (train[:N_MASSIVE] if N_MASSIVE else train) + jsonl("train_dolly.jsonl")[:DOLLY_N]
    meta = {"case": CASE, "run": RUN, "model_id": E.MODEL_ID, "data": DATA, "n_massive": PARTS.get("intent", 0) if PARTS else len(train) - DOLLY_N, "n_dolly": len(train) - PARTS.get("intent", 0) if PARTS else DOLLY_N, "epochs": EPOCHS, "max_steps": MAX_STEPS,
            "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "batch": BATCH, "micro_batch": MICRO, "max_len": MAX_LEN, "seed": E.SEED, "dtype": str(E.DTYPE), "target_modules": "all-linear", "parts": PARTS}

    _, calib_s, harmful_hb, xsafe, cap, _ = E.load_data()
    xsafe = [x for i, x in enumerate(xsafe) if i % 5 == 0]  # the tune split of XSTest, as in repair_job.py
    harmful = [p for s in EVAL_SETS for p in safety[s]]
    sets, n = {}, 0
    for s in EVAL_SETS:
        sets[s] = [n, n + len(safety[s])]; n += len(safety[s])

    def gen_eval(m, name):
        files = []
        for split, prompts in (("harmful", harmful), ("xstest", xsafe)):
            resp = E.generate_completions(m, tok, prompts)
            flags = [bool(f) for f in E.string_refusal(resp)]
            p = E.RUN_LOG_DIR / "completions" / name / f"{split}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"config_name": name, "split": split, "n": len(prompts), "prompts": prompts, "responses": resp, "string_refusal": flags,
                                     "sets": sets if split == "harmful" else {"xstest_tune": [0, len(prompts)]}, "seed": E.SEED,
                                     "max_new_tokens": E.MAX_NEW_TOKENS, "dtype": str(E.DTYPE), "gen_batch": E.GEN_BATCH}))
            files.append(p)
            meta[f"string_{split}_refusal__{name}"] = sum(flags) / max(1, len(flags))
        return files

    files = []
    meta["skill_aligned"], meta["skill_aligned_by_lang"] = skill_score(model, tok, skill_rows)
    log(f"skill of the original model: {meta['skill_aligned']:.3f} {meta['skill_aligned_by_lang']}")
    if os.environ.get("ALIGNED_EVAL") == "1" and not api.file_exists(REPO, f"runs/{RUN}/completions/aligned/harmful.json", repo_type="dataset"):
        fs = gen_eval(model, "aligned")
        meta["mmlu_aligned"] = float(E.eval_capability_checkpointed(model, tok, cap)[0])
        m = E.RUN_LOG_DIR / "metrics" / "aligned.json"; m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({k: v for k, v in meta.items() if "aligned" in k or k in ("model_id", "data", "seed", "dtype")}, indent=1))
        commit(api, [CommitOperationAdd(path_in_repo=p.relative_to(E.RUN_DIR).as_posix(), path_or_fileobj=str(p)) for p in fs + [m]], f"{RUN} aligned answers")  # saved now: a later crash must not cost these 4 minutes again
    if os.environ.get("BATCH_CHECK") == "1":  # is a larger generation batch safe? Same answers, how much faster.
        ps, res = harmful[:16] + xsafe[:16], {}
        for bs in (4, 16):
            t = time.time(); res[bs] = (E.generate_completions(model, tok, ps, batch_size=bs), time.time() - t)
        meta["batch_check"] = {"n": len(ps), "identical_answers": sum(a == b for a, b in zip(res[4][0], res[16][0])), "s_batch4": round(res[4][1], 1), "s_batch16": round(res[16][1], 1)}
        log(f"batch check: {meta['batch_check']}")

    # ---- train
    from peft import LoraConfig, get_peft_model
    out_dir = E.RUN_LOG_DIR / "case"; out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: meta[k] for k in ("model_id", "data", "n_massive", "n_dolly", "lr", "lora_r", "lora_alpha", "batch", "max_len", "seed")}
    if PARTS:
        cfg["parts"] = PARTS  # only when set, so checkpoints of older runs still match
    enc, meta["dropped_too_long"] = encode(tok, train)
    meta["rows_trained"] = {"intent": sum(not x[2] for x in enc), "free_text": sum(x[2] for x in enc)}  # n_massive / n_dolly above are counts before the length drop
    meta["note_micro_batch"] = "the loss is a token mean inside each forward pass, weighted by rows across passes: MICRO slightly changes how long answers are weighted"
    pm = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    tp = [p for p in pm.parameters() if p.requires_grad]
    steps = MAX_STEPS or math.ceil(len(enc) * EPOCHS / BATCH)
    opt = torch.optim.AdamW(tp, lr=LR, weight_decay=0.0)
    named = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    # Checkpoint + resume: the LoRA weights, optimizer state and step go to the repo every CKPT_S seconds, so a dead job loses
    # minutes, not the run. The learning rate is constant after a 30-step warm-up and the data order depends only on the seed and
    # the epoch, so "train 1 epoch, look, then continue to 3" (RESUME_FROM=<earlier case>) is the same run as 3 epochs in one go.
    start, losses, ck = 0, [], out_dir / "ckpt.pt"
    for src in (CASE, os.environ.get("RESUME_FROM")):
        if src and api.file_exists(REPO, f"cases/{src}/ckpt.pt", repo_type="dataset"):
            st = torch.load(hf_hub_download(REPO, f"cases/{src}/ckpt.pt", repo_type="dataset"), map_location="cpu", weights_only=True)
            if st["cfg"] != cfg:
                raise SystemExit(f"checkpoint cases/{src}/ckpt.pt was made with other settings {st['cfg']} != {cfg}")
            with torch.no_grad():
                for n, p in named.items():
                    p.copy_(st["lora"][n].to(p.device))
            opt.load_state_dict(st["opt"])
            start, losses = min(st["step"], steps), list(st["losses"])
            meta["resumed_from"] = {"case": src, "step": st["step"]}
            log(f"resumed from cases/{src}/ckpt.pt at step {st['step']}")
            del st
            break

    def save_ckpt(step):
        try:
            torch.save({"cfg": cfg, "step": step, "losses": losses, "lora": {n: p.detach().cpu() for n, p in named.items()}, "opt": opt.state_dict()}, ck)
            api.upload_file(path_or_fileobj=str(ck), path_in_repo=f"cases/{CASE}/ckpt.pt", repo_id=REPO, repo_type="dataset", commit_message=f"{CASE} checkpoint step {step}")
            log(f"checkpoint saved at step {step}")
        except Exception as exc:  # training goes on; the next checkpoint or the final save will retry
            log(f"WARNING checkpoint upload failed at step {step}: {type(exc).__name__}: {str(exc)[:200]}")

    perms = {}

    def row(i):  # the i-th training sequence of the whole run
        e = i // len(enc)
        if e not in perms:
            perms[e] = epoch_order([len(x[0]) for x in enc], E.SEED, e, BATCH)
        return enc[perms[e][i % len(enc)]]

    def mid_eval(step):
        """Skill and safety answers part-way through training, and the model at this point saved as its own case.
        Measured on the adapter as it trains (not merged); a failure here is logged and training goes on."""
        t = time.time()
        try:
            name = f"{CASE}_s{step}"
            pm.eval()
            with torch.autocast("cuda", dtype=E.DTYPE):
                sk, by = skill_score(pm, tok, skill_rows)
                fs = gen_eval(pm, name) if MID_GEN else []
            point = {"step": step, "rows_seen": step * BATCH, "skill": sk, "skill_by_lang": by, "string_harmful_refusal": meta.get(f"string_harmful_refusal__{name}"),
                     "string_xstest_refusal": meta.get(f"string_xstest_refusal__{name}"), "loss_intent": avg(loss_m[-200:]), "loss_free_text": avg(loss_d[-200:])}
            meta.setdefault("curve", []).append(point)
            d = out_dir / name; d.mkdir(parents=True, exist_ok=True)
            torch.save(factors_of(pm), d / "lora_factors.pt"); (d / "meta.json").write_text(json.dumps({**cfg, **point, "parent_case": CASE}, indent=1))
            api.create_commit(REPO, [CommitOperationAdd(path_in_repo=f"cases/{name}/{n}", path_or_fileobj=str(d / n)) for n in ("lora_factors.pt", "meta.json")]
                              + [CommitOperationAdd(path_in_repo=p.relative_to(E.RUN_DIR).as_posix(), path_or_fileobj=str(p)) for p in fs],
                              commit_message=f"{RUN} {name}: snapshot + answers", repo_type="dataset")
            log(f"MID step {step}: skill={sk:.3f} keyword_refusal={point['string_harmful_refusal']} loss intent={point['loss_intent']} free_text={point['loss_free_text']} ({time.time() - t:.0f}s)")
        except Exception as exc:
            log(f"WARNING mid-run measurement at step {step} failed: {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            pm.train(); torch.cuda.empty_cache()
        return time.time() - t

    avg = lambda v: round(sum(v) / len(v), 4) if v else None
    loss_m, loss_d, t_eval = [], [], 0.0

    log(f"train rows={len(enc)} dropped={meta['dropped_too_long']} steps={start}->{steps} trainable={sum(p.numel() for p in tp) / 1e6:.1f}M lora dtype={tp[0].dtype}")
    pm.train()
    t_train = t_ck = time.time()
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / 30)
        full, total = [row(step * BATCH + j) for j in range(BATCH)], 0.0
        for j in range(0, BATCH, MICRO):  # gradient accumulation: the same update as one batch of BATCH, in pieces that fit the GPU
            batch = full[j:j + MICRO]
            width = max(len(b[0]) for b in batch)
            ids = torch.tensor([b[0] + [tok.pad_token_id] * (width - len(b[0])) for b in batch], device=E.DEVICE)
            lab = torch.tensor([b[1] + [-100] * (width - len(b[1])) for b in batch], device=E.DEVICE)
            att = torch.tensor([[1] * len(b[0]) + [0] * (width - len(b[0])) for b in batch], device=E.DEVICE)
            with torch.autocast("cuda", dtype=E.DTYPE):
                out = pm(input_ids=ids, attention_mask=att, labels=lab)
                loss = out.loss * len(batch) / BATCH
            with torch.no_grad():
                for b, v in zip(batch, per_row_loss(out.logits, lab)):
                    (loss_d if b[2] else loss_m).append(v)
            del out
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at step {step}: stopping")
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(tp, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(total)
        if step % 25 == 0 or step == steps - 1:
            el = time.time() - t_train - t_eval
            log(f"step {step + 1}/{steps} loss={sum(losses[-25:]) / len(losses[-25:]):.4f} intent={avg(loss_m[-150:])} free_text={avg(loss_d[-50:])} "
                f"{el:.0f}s {(step + 1 - start) * BATCH / max(el, 1e-9):.1f} seq/s gpu={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        mid = EVAL_EVERY and (step + 1) % EVAL_EVERY == 0 and steps - (step + 1) >= EVAL_EVERY // 2
        if (time.time() - t_ck > CKPT_S or mid) and step + 1 < steps:
            save_ckpt(step + 1); t_ck = time.time()
        if mid:
            t_eval += mid_eval(step + 1)
    if start < steps:
        save_ckpt(steps)
    meta.update(steps=steps, train_s=round(time.time() - t_train - t_eval), loss_intent_last=avg(loss_m[-200:]), loss_free_text_last=avg(loss_d[-200:]), loss_first=round(sum(losses[:10]) / len(losses[:10]), 4), loss_last=round(sum(losses[-10:]) / len(losses[-10:]), 4),
                epochs_done=round(steps * BATCH / len(enc), 3), loss_curve=[round(sum(losses[i:i + 25]) / len(losses[i:i + 25]), 4) for i in range(0, len(losses), 25)])

    # ---- factors out, peft off, case rebuilt from the factors (so what we measure is exactly what repair_job.py will load)
    payload = factors_of(pm)
    factors = payload["factors"]
    model = pm.unload()
    del pm, opt, tp
    model.eval(); torch.cuda.empty_cache()
    missing = [k for k in factors if k not in dict(model.named_parameters())]
    if missing or not factors:
        raise SystemExit(f"{len(missing)} of {len(factors)} LoRA weight names do not match a model parameter, e.g. {missing[:2]}")
    out = out_dir
    torch.save(payload, out / "lora_factors.pt"); (out / "meta.json").write_text(json.dumps(meta, indent=1))
    commit(api, [CommitOperationAdd(path_in_repo=f"cases/{CASE}/{n}", path_or_fileobj=str(out / n)) for n in ("lora_factors.pt", "meta.json")],
           f"{RUN} case {CASE}: factors (evaluation follows)")  # the case exists from here on, whatever happens to the evaluation
    meta["n_factors"] = apply_factors(model, payload)
    meta["skill"], meta["skill_by_lang"] = skill_score(model, tok, skill_rows)
    meta["mmlu"] = float(E.eval_capability_checkpointed(model, tok, cap)[0])
    files += gen_eval(model, CASE)
    meta.update(wall_s=round(time.time() - T0), gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    log(f"skill {meta['skill_aligned']:.3f} -> {meta['skill']:.3f} {meta['skill_by_lang']} mmlu={meta['mmlu']:.3f} loss {meta['loss_first']} -> {meta['loss_last']}")

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    ops = [CommitOperationAdd(path_in_repo=f"cases/{CASE}/meta.json", path_or_fileobj=str(out / "meta.json"))]
    ops += [CommitOperationAdd(path_in_repo=p.relative_to(E.RUN_DIR).as_posix(), path_or_fileobj=str(p)) for p in files]
    ops.append(CommitOperationAdd(path_in_repo=f"runs/{RUN}/metrics/{CASE}.json", path_or_fileobj=str(out / "meta.json")))
    commit(api, ops, f"{RUN} case {CASE}")
    log(f"finetune_job done in {time.time() - T0:.0f}s: cases/{CASE} saved")


if __name__ == "__main__":
    main()
