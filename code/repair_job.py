# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "safetensors",
#   "transformers>=5.15",
#   "torch>=2.7",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "numpy",
#   "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git",
#   "auditkit @ git+https://github.com/Lexsi-Labs/AuditKIT.git",
# ]
# ///
"""Run a list of repair specs on one broken model. One spec = one method + its settings.

Every finished spec uploads its completions + metrics to the private repo right away and is
skipped on relaunch, so a crashed or timed-out job loses at most the spec it was on.

Env: HF_TOKEN, HF_CKPT_REPO, RUN_NAME, SPECS_FILE (repo path to a JSON list),
     CASE (abl_safetune | any name with cases/<CASE>/deltas.pt in the repo),
     SPLIT (tune | test | all), DEADLINE_S (stop starting new specs after this), MODEL_ID (default tiny-aya-global),
     GEN_BATCH (generation batch size; keep it the same for runs you compare),
     HARMFUL_SET (harmbench = HarmBench cut by SPLIT; or names in data/<DATA>/safety.json joined with "+", used whole, e.g. tune_harmful).
A spec may carry its own "case" (default: CASE), so one job can cover several broken models. Cases:
  abl_safetune          SafeTune abliteration (repo deltas for tiny-aya-global, redone in-job for another MODEL_ID)
  dense_t<t>            every weight moved the fraction t from the aligned model toward the unaligned base (no single direction)
  ft_<name>             a LoRA fine-tune from finetune_job.py: cases/ft_<name>/lora_factors.pt; adds a "skill" score per spec
                        (exact match on data/<DATA>/skill_test.jsonl), the thing a repair must keep
  <name>                cases/<name>/deltas.pt in the repo (e.g. heretic_global)
Method "diagnose" runs no generation: it measures broken - aligned (how many tensors, rank, layers, overlap with the refusal direction).
ponytail: reuses exp1_job.py (fetched from the repo) for loading, generation, KL, MMLU, uploads.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("FULL", "1")
os.environ.setdefault("N_CALIB", "128")
RUN = os.environ.setdefault("RUN_NAME", "screen")
REPO = os.environ["HF_CKPT_REPO"]
CASE = os.environ.get("CASE", "abl_safetune")
SPLIT = os.environ.get("SPLIT", "tune")
DEADLINE_S = float(os.environ.get("DEADLINE_S", "3000"))
HARMFUL_SET = os.environ.get("HARMFUL_SET", "harmbench")
DATA = os.environ.get("DATA", "ft_v1")
SKILL_FILE = os.environ.get("SKILL_FILE", "")  # repo path under data/, e.g. final_v1/skill_final.jsonl; default data/<DATA>/skill_test.jsonl
GMMLU_FILE = os.environ.get("GMMLU_FILE", "")  # repo path under data/, e.g. final_v1/global_mmlu_lite.jsonl; empty = not measured
QUANT4 = os.environ.get("QUANT4") == "1"  # every non-original spec is evaluated after 4-bit quantization (bitsandbytes NF4); launch with --with bitsandbytes


def by_lang(rows, hits):
    """Share of hits per row["lang"], with the count, e.g. {"sw": [0.5, 2]}."""
    out = {}
    for r, h in zip(rows, hits):
        a = out.setdefault(r.get("lang", "?"), [0, 0]); a[0] += bool(h); a[1] += 1
    return {k: [round(a / n, 4), n] for k, (a, n) in sorted(out.items())}


assert by_lang([{"lang": "sw"}, {"lang": "sw"}, {"lang": "hi"}], [1, 0, 1]) == {"hi": [1.0, 1], "sw": [0.5, 2]}
T0 = time.time()

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, str(Path(hf_hub_download(REPO, "code/exp1_job.py", repo_type="dataset")).parent))
import exp1_job as E  # noqa: E402
import torch  # noqa: E402

log = E.log
REFUSAL = "I'm sorry, but I can't help with that request."


def pick(items, split=SPLIT):
    """Deterministic split: every 5th item tunes, the rest test."""
    if split == "all":
        return list(items)
    return [x for i, x in enumerate(items) if (i % 5 == 0) == (split == "tune")]


def main():
    E.log_environment(); E.hf_login(); E.ensure_ckpt_repo(); E._seed_all()
    api = HfApi()
    kinds = ("metrics",) if os.environ.get("RETRY_ERRORS") == "1" else ("metrics", "errors")  # a failed spec is not retried by default
    done = {f.rsplit("/", 1)[1] for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith(tuple(f"runs/{RUN}/{k}/" for k in kinds))}
    # A run name stands for one set of prompts and one generation setting: finished specs are skipped by name only, so a relaunch
    # under the same name with another SPLIT / HARMFUL_SET would silently reuse the old results.
    cfg = {"split": SPLIT, "harmful_set": HARMFUL_SET, "data": DATA, "gen_batch": E.GEN_BATCH, "max_new_tokens": E.MAX_NEW_TOKENS, "model_id": E.MODEL_ID}
    cfg.update({k: v for k, v in (("skill_file", SKILL_FILE), ("gmmlu_file", GMMLU_FILE), ("quant4", QUANT4)) if v})  # only when set, so older runs still match
    if api.file_exists(REPO, f"runs/{RUN}/run_config.json", repo_type="dataset"):
        old = json.loads(Path(hf_hub_download(REPO, f"runs/{RUN}/run_config.json", repo_type="dataset")).read_text())
        if old != cfg:
            raise SystemExit(f"run {RUN} was started with {old}; this launch has {cfg}. Use a new RUN_NAME.")
    else:
        api.upload_file(path_or_fileobj=json.dumps(cfg, indent=1).encode(), path_in_repo=f"runs/{RUN}/run_config.json", repo_id=REPO, repo_type="dataset")
    specs = json.loads(Path(hf_hub_download(REPO, os.environ["SPECS_FILE"], repo_type="dataset")).read_text())
    todo = [s for s in specs if f"{s['name']}.json" not in done]
    if os.environ.get("SHARD"):  # "i/n": this job takes every n-th spec of the full list, so n jobs under one RUN_NAME split the work without overlap
        i, n = map(int, os.environ["SHARD"].split("/"))
        mine = {s["name"] for k, s in enumerate(specs) if k % n == i}
        todo = [s for s in todo if s["name"] in mine]
    log(f"RUN={RUN} CASE={CASE} SPLIT={SPLIT} specs={len(specs)} todo={len(todo)} skipped_done={len(specs) - len(todo)}")
    if not todo:
        return

    aligned, tok = E.load_model_and_tok()
    calib_h, calib_s, harmful, xsafe, cap, klp = E.load_data()
    harmful, xsafe = pick(harmful), pick(xsafe)
    data_file = lambda name: Path(hf_hub_download(REPO, f"data/{DATA}/{name}", repo_type="dataset"))
    read_rows = lambda path: [json.loads(l) for l in Path(hf_hub_download(REPO, f"data/{path}", repo_type="dataset")).read_text(encoding="utf-8").splitlines() if l.strip()]
    if HARMFUL_SET != "harmbench":
        safety, calib = json.loads(data_file("safety.json").read_text(encoding="utf-8")), {E.norm_prompt(p) for p in calib_h}
        harmful = [p for name in HARMFUL_SET.split("+") for p in safety[name] if E.norm_prompt(p) not in calib]
        log(f"HARMFUL_SET={HARMFUL_SET}: {len(harmful)} prompts (after dropping any that are also calibration prompts)")
    skill_rows = None
    if any(s.get("case", CASE).startswith("ft_") for s in todo):
        skill_rows = read_rows(SKILL_FILE or f"{DATA}/skill_test.jsonl")
    gm_rows = read_rows(GMMLU_FILE) if GMMLU_FILE else None
    if gm_rows:  # Global-MMLU Lite: the letter with the highest next-token logit among A-D
        gm_prompts = [r["question"].strip() + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", r["choices"])) + "\nAnswer with the letter only." for r in gm_rows]
        gm_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in "ABCD"]
        log(f"GMMLU {len(gm_rows)} rows from {GMMLU_FILE}; letter token ids {gm_ids}")
    fmt = lambda ps: [E.format_chat(tok, p) for p in ps]
    base_logits = E.last_token_logits(aligned, tok, klp)
    work = E.load_model(E.MODEL_ID, dtype=E.DTYPE, device=E.DEVICE).eval()

    DEFAULT_MODEL = E.MODEL_ID == "CohereLabs/tiny-aya-global"
    state = {"base": None, "direction": None, "caa": {}, "deltas": {}, "case": None, "broken_logits": {}}

    def case_deltas(case):
        if case not in state["deltas"]:
            if case.startswith("ft_"):
                state["deltas"][case] = torch.load(hf_hub_download(REPO, f"cases/{case}/lora_factors.pt", repo_type="dataset"), map_location="cpu", weights_only=True)
            elif case == "abl_safetune":
                E.pull_stage_from_hf("broken")
                state["deltas"][case] = E.load_edited_deltas(E.stage_dir("broken") / "deltas.pt")
            else:
                state["deltas"][case] = torch.load(hf_hub_download(REPO, f"cases/{case}/deltas.pt", repo_type="dataset"), map_location="cpu", weights_only=True)
        return state["deltas"][case]

    def reset(case):
        """work := the broken model of this case, rebuilt from the aligned weights."""
        state["case"] = case
        src = dict(aligned.named_parameters())
        with torch.no_grad():
            if case.startswith("dense_t"):
                t, b = float(case[len("dense_t"):]), dict(base_cpu().named_parameters())
                for k, p in work.named_parameters():
                    p.copy_(torch.lerp(src[k], b[k].to(device=p.device, dtype=p.dtype), t))
                return
            for k, p in work.named_parameters():
                p.copy_(src[k])
            if case.startswith("ft_"):  # same arithmetic as finetune_job.apply_factors: W := (W + scaling * B @ A) in fp32, cast back
                lf, params = case_deltas(case), dict(work.named_parameters())
                for k, f in lf["factors"].items():
                    p = params[k]
                    p.copy_((p.float() + lf["scaling"] * (f["B"].to(p.device).float() @ f["A"].to(p.device).float())).to(p.dtype))
                return
            if case == "abl_safetune" and not DEFAULT_MODEL:  # same break, made from this model's own direction
                from safetune.steer import orthogonalize_weights
                if len(orthogonalize_weights(work, direction(), include_attn=True, include_mlp=True)) < 2:
                    raise RuntimeError("orthogonalize_weights edited nothing")
                return
            deltas = case_deltas(case)
            params = {**dict(work.named_parameters()), **E.collect_editable_params(work)}
            n = 0
            for k, d in deltas.items():
                if k in params:
                    params[k].add_(d.to(device=params[k].device, dtype=params[k].dtype)); n += 1
        if n != len(deltas):
            raise RuntimeError(f"only {n}/{len(deltas)} case deltas matched a parameter")

    def base_cpu():
        if state["base"] is None:
            state["base"] = E.load_model(E.BASE_UNALIGNED_ID, dtype=E.DTYPE, device="cpu").eval()
        return state["base"]

    def base():
        return base_cpu().to(E.DEVICE)

    def park():
        if state["base"] is not None:
            state["base"].to("cpu")
        torch.cuda.empty_cache()

    def quant4():
        """work (repair already applied, in bf16) -> saved to disk -> loaded again in 4-bit NF4, as a deployment would load it.
        Returns (4-bit model, undo). work waits on the CPU meanwhile, so two bf16 models and the 4-bit one never share the GPU.
        ponytail: weight-only specs (none, wise_ft, ...); a spec with a hook or an override is refused by the caller."""
        import shutil
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        from safetune._refusal_helpers import _load_pretrained_lm
        t, d, bf16_gb = time.time(), "/tmp/q4src", work.get_memory_footprint() / 2**30
        work.save_pretrained(d, safe_serialization=True)
        work.to("cpu"); torch.cuda.empty_cache()
        try:
            q = _load_pretrained_lm(d, dtype=E.DTYPE, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)).eval()
            n4, nl = (sum(isinstance(x, c) for x in q.modules()) for c in (bnb.nn.Linear4bit, torch.nn.Linear))
            log(f"QUANT4 nf4, compute bf16, no double quant: {n4} of {nl} linear modules are 4-bit; footprint {q.get_memory_footprint() / 2**30:.2f} GB "
                f"(bf16: {bf16_gb:.2f} GB); save + load {time.time() - t:.0f}s")
            if not n4:
                raise RuntimeError("no 4-bit modules after loading")
        except BaseException:
            work.to(E.DEVICE)
            raise
        return q, lambda: (work.to(E.DEVICE), shutil.rmtree(d, ignore_errors=True))

    def direction():
        if state["direction"] is None:
            if DEFAULT_MODEL:
                state["direction"] = E.load_direction()[0]
            else:  # the refusal direction of this model's own original
                from safetune.steer import RefusalDirectionConfig, extract_refusal_direction
                d, layer, _ = extract_refusal_direction(aligned, tok, harmful_prompts=list(calib_h), harmless_prompts=list(calib_s),
                                                        config=RefusalDirectionConfig(normalize=True))
                log(f"refusal direction for {E.MODEL_ID}: picked layer {layer}")
                state["direction"], state["direction_layer"] = d.detach().cpu(), int(layer)
        return state["direction"]

    def diagnose(case):
        """Cheap weight-space facts about broken - aligned for the tuner. No generation."""
        import re
        import statistics as st
        reset(case)
        a, d = dict(aligned.named_parameters()), direction().to(E.DEVICE).float()
        d, rows, n2d = d / d.norm(), [], 0
        with torch.no_grad():
            for k, p in work.named_parameters():
                if p.dim() != 2:
                    continue
                n2d += 1
                diff = p.float() - a[k].float()
                fn = float(diff.norm())
                if fn == 0:
                    continue
                m = re.search(r"layers\.(\d+)\.", k)
                row = {"name": k, "layer": int(m.group(1)) if m else -1, "rel_norm": fn / float(a[k].float().norm())}
                if max(diff.shape) <= 8192:
                    e = torch.linalg.svdvals(diff) ** 2
                    row["top1_energy"], row["eff_rank"] = float(e[0] / e.sum()), float(e.sum() ** 2 / (e ** 2).sum())
                    if diff.shape[0] == d.numel():  # how much of the change writes along the refusal direction
                        row["dir_energy"] = float((d @ diff).norm() ** 2 / fn ** 2)
                rows.append(row)
                del diff
        med = lambda key: (lambda v: round(st.median(v), 5) if v else None)([r[key] for r in rows if key in r])
        kinds = {}
        for r in rows:
            kind = r["name"].split(".")[-2] if "." in r["name"] else r["name"]
            kinds[kind] = kinds.get(kind, 0) + 1
        layers = sorted({r["layer"] for r in rows if r["layer"] >= 0})
        try:
            circuit = {"refusal_circuit": circuit_diag()}
        except Exception as exc:  # the weight facts above must survive a circuit-finder failure
            circuit = {"refusal_circuit_error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            log(f"WARNING circuit diagnostic failed: {circuit['refusal_circuit_error']}")
        return {"n_matrices": n2d, "n_changed": len(rows), "frac_changed": round(len(rows) / max(1, n2d), 4), "changed_by_module": kinds,
                "n_layers_touched": len(layers), "layers_touched": layers, "median_rel_norm": med("rel_norm"),
                "max_rel_norm": round(max((r["rel_norm"] for r in rows), default=0.0), 5), "median_top1_energy": med("top1_energy"),
                "median_eff_rank": med("eff_rank"), "median_dir_energy": med("dir_energy"),
                "direction_layer": state.get("direction_layer", 18), **circuit}

    def circuit_nodes(model, n=32):
        """Refusal-circuit scores per block from SafeTune's EAP-IG, as {"A<L>.0": s, "MLP <L>": s} (the names layer_mass reads).
        ponytail: block granularity (one attention node per layer); switch to "head" if head-level picks are ever needed."""
        from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit
        cfg = EAPSafetyCircuitConfig(method="eap-ig", ig_steps=3, top_k_edges=10**6, batch_size=4, max_seq_len=96, granularity="block")
        req = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)  # the input embeddings carry the gradient; this skips 6.6 GB of parameter grads
        try:
            info = eap_safety_circuit(model, fmt(calib_h[:n]), fmt(calib_s[:n]), cfg, tokenizer=tok)
        finally:
            for p, r in zip(model.parameters(), req):
                p.requires_grad_(r)
            model.zero_grad(set_to_none=True)
        out = {}
        for name, sc in info.safety_units.activation_correlation.items():  # blocks.L.attn | blocks.L.mlp
            parts = name.split(".")
            out[f"A{parts[1]}.0" if parts[2] == "attn" else f"MLP {parts[1]}"] = float(sc)
        if len({k.split()[-1].split(".")[0].lstrip("A") for k in out}) < 4:
            raise RuntimeError(f"circuit finder returned too few layers ({len(out)} nodes)")
        return out

    def aligned_nodes():
        if "ns_aligned" not in state:
            state["ns_aligned"] = circuit_nodes(aligned)
        return state["ns_aligned"]

    def circuit_diag():
        """Where refusal is computed in the original vs in the current broken model: share of attribution mass per layer."""
        from safetune.core.circuit_kit.adapter import layer_mass
        ma, mw = layer_mass(aligned_nodes()), layer_mass(circuit_nodes(work))
        ta, tw = sum(ma.values()), sum(mw.values())
        pa, pw = {k: v / ta for k, v in ma.items()}, {k: v / tw for k, v in mw.items()}
        top = lambda m: [[int(k), round(m[k], 4)] for k in sorted(m, key=lambda k: -m[k])[:8]]
        diff = {int(k): round(pw.get(k, 0) - pa.get(k, 0), 4) for k in set(pa) | set(pw)}
        return {"aligned_top8_layers_share": top(pa), "broken_top8_layers_share": top(pw),
                "largest_share_changes_broken_minus_aligned": [list(x) for x in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:8]],
                "mass_total_aligned": round(ta, 4), "mass_total_broken": round(tw, 4)}

    def caa_vectors(layers):
        key = tuple(layers)
        if key not in state["caa"]:
            from safetune.interventions.steer.caa import CAAConfig, extract_caa_vectors
            # 32 prompts at a time: the vector is a difference of means, so the mean over equal chunks is the same vector,
            # and one 128-prompt batch needs 26 GB (does not fit an L4).
            h, s_, parts = fmt(calib_h), fmt(calib_s), []
            n = min(len(h), len(s_)) // 32 * 32 or min(len(h), len(s_))
            c = 8 if n % 8 == 0 else n  # 8 at a time over the same n prompts: same mean; 32 at a time ran out of memory on an L4 once a fine-tune case was loaded
            for i in range(0, n, c):
                parts.append(extract_caa_vectors(aligned, tok, h[i:i + c], s_[i:i + c], CAAConfig(target_layers=list(layers))))
                torch.cuda.empty_cache()
            state["caa"][key] = {L: sum(p[L] for p in parts) / len(parts) for L in parts[0]}
        return state["caa"][key]

    def apply(spec, _reset=True):
        """Return (gen_model, cleanup, override) for one spec. override(prompts, responses) may rewrite responses."""
        m, kw = spec["method"], dict(spec.get("kw") or {})
        if m == "aligned":
            return aligned, None, None
        case = spec.get("case", CASE)
        if _reset:  # False only inside a "stack": the next step builds on the previous one
            reset(case)
        if case not in state["broken_logits"]:  # reference for "how much benign behaviour of the broken model did the repair keep"
            state["broken_logits"][case] = E.last_token_logits(work, tok, klp)
        if m == "none":
            pass
        elif m == "stack":  # several repairs on top of each other, in the order given: {"steps": [{"method", "kw"}, ...]} (weight changes first, a steering hook or bias last)
            undo = []
            for step in kw["steps"]:
                _, c, _ = apply({**step, "case": case}, _reset=False)
                if c:
                    undo.append(c)
            return work, (lambda: [c() for c in reversed(undo)]) if undo else None, None
        elif m == "svr":
            from safetune.interventions.recover.safety_vector_restore import apply_safety_vector_restore
            apply_safety_vector_restore(work, aligned, **kw)
        elif m == "wise_ft":
            from safetune.interventions.recover.wise_ft import apply_wise_ft
            if case.startswith("ft_"):  # LoRA never touches the embeddings; skipping them saves 8 GB of fp32 temporaries (fits an L4), same result
                kw.setdefault("param_filter", lambda name, t: "embed" not in name)
            apply_wise_ft(work, aligned, **kw)
        elif m == "safe_delta":
            from safetune.interventions.recover.safe_delta import apply_safe_delta
            apply_safe_delta(work, aligned=aligned, **kw)
        elif m in ("task_arith", "lssf", "safemerge", "somf", "lox", "mscp"):
            b = base()
            try:
                if m == "task_arith":
                    from safetune.interventions.recover.merge import task_arithmetic
                    task_arithmetic(work, base=b, aligned=aligned, **kw)
                elif m == "lssf":
                    from safetune.interventions.recover.lssf import apply_lssf
                    apply_lssf(work, base=b, aligned=aligned, **kw)
                elif m == "safemerge":
                    from safetune.interventions.recover.safemerge import apply_safemerge
                    apply_safemerge(work, base=b, aligned=aligned, **kw)
                elif m == "somf":
                    from safetune.interventions.recover.merge import somf_merge
                    somf_merge(work, aligned=aligned, base=b, **kw)
                elif m == "lox":
                    from safetune.interventions.recover.lox import apply_lox
                    apply_lox(work, base=b, aligned=aligned, **kw)
                elif m == "mscp":
                    from safetune.interventions.recover.mscp import apply_mscp
                    apply_mscp(work, aligned_state=aligned.state_dict(), base_state=b.state_dict(), **kw)
            finally:
                park()
        elif m == "ctheta":  # restore only the layers the refusal circuit lives in
            from safetune.core.circuit_kit.adapter import circuit_info_from_node_scores
            from safetune.interventions.recover.ctheta import apply_ctheta
            ci = circuit_info_from_node_scores(aligned_nodes(), top_fraction=float(kw.get("top_fraction", 0.5)), layer_fraction=float(kw.get("layer_fraction", 0.2)))
            neg = base_cpu() if kw.get("negative") == "base" else work  # work: move circuit layers back toward the original; base: SafeTune's alignment task vector
            apply_ctheta(work, positive=aligned, negative=neg, circuit_info=ci, strength=float(kw.get("strength", 1.0)))
            log(f"ctheta layers={ci.layer_suggestions.layer_subset} modules={ci.layer_suggestions.target_modules}")
        elif m == "grad_selective":
            from safetune.interventions.recover.grad_selective_recover import apply_grad_selective_recover
            n = int(kw.pop("n", 16))
            ids = [tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(E.DEVICE) for p in fmt(calib_h[:n])]
            with torch.enable_grad():
                apply_grad_selective_recover(work, aligned, ids, max_samples=n, **{"target_modules": ["o_proj", "down_proj"], **kw})
            work.zero_grad(set_to_none=True)
        elif m == "oneshot":
            from safetune.interventions.recover.oneshot_safety_patch import apply_oneshot_safety_patch
            with torch.enable_grad():
                apply_oneshot_safety_patch(work, calib_h[0], REFUSAL, tok, **kw)
            work.zero_grad(set_to_none=True)
        elif m == "actadd":
            from safetune.steer import RefusalDirectionModel
            w = RefusalDirectionModel(work, direction(), mode="steer", strength=float(kw["strength"]), layers=list(kw["layers"]))
            w.install()
            return work, w.remove, None
        elif m == "steer_bias":  # weight-space twin of actadd: Cohere2 blocks are parallel (h = x + attn + mlp, nothing after), so a bias on mlp.down_proj shifts the block output by exactly s*d
            from safetune._refusal_helpers import _get_decoder_layers
            blocks, saved = _get_decoder_layers(work), []
            d = direction()
            for L in kw["layers"]:
                lin = blocks[int(L)].mlp.down_proj
                saved.append((lin, lin.bias))
                shift = float(kw["strength"]) * d.to(device=lin.weight.device, dtype=lin.weight.dtype)  # cast first, like the hook, so bf16 rounding matches
                lin.bias = torch.nn.Parameter(shift if lin.bias is None else lin.bias.detach() + shift, requires_grad=lin.weight.requires_grad)

            def undo():  # the old Parameter object (or None) goes back, so named_parameters() is as before
                for lin, old in saved:
                    lin.bias = old
            if os.environ.get("STEER_BIAS_CHECK") == "1":  # compare with the hook on the unpatched model; log only
                patched = [lin.bias for lin, _ in saved]
                undo()
                from safetune.steer import RefusalDirectionModel
                w = RefusalDirectionModel(work, d, mode="steer", strength=float(kw["strength"]), layers=list(kw["layers"])).install()
                try:
                    ref = E.last_token_logits(work, tok, klp)
                finally:
                    w.remove()
                for (lin, _), pb in zip(saved, patched):
                    lin.bias = pb
                log(f"STEER_BIAS_CHECK max_abs_logit_diff(bias vs hook)={float((E.last_token_logits(work, tok, klp) - ref).abs().max()):.5f}")
            return work, undo, None
        elif m == "caa":
            from safetune.interventions.steer.caa import CAAModel
            w = CAAModel(work, caa_vectors(kw["layers"]), strength=float(kw["strength"]))
            w.install()
            return work, w.remove, None
        elif m == "cast":
            from safetune.interventions.steer.cast import CASTModel, fit_cast_condition
            cond = fit_cast_condition(work, list(calib_h), list(calib_s), tok, device=E.DEVICE)
            w = CASTModel(work, caa_vectors(kw["layers"]), condition=cond, alpha=float(kw["strength"]))
            return w, w.remove, None  # generate() is gated per prompt; forward passes (KL, MMLU) are not
        elif m == "probe_guard":
            from safetune.interventions.steer.probe_guard import LinearProbeConfig, LinearProbeGuardModel, fit_linear_probe
            probe = fit_linear_probe(work, tok, fmt(calib_h), fmt(calib_s), LinearProbeConfig(**kw))
            g = LinearProbeGuardModel(work, tok, probe)

            def override(prompts, responses):
                flags = [bool(s.item() >= probe.threshold) for s in g.score(fmt(prompts))]
                return [REFUSAL if f else r for f, r in zip(flags, responses)]
            return work, None, override
        elif m == "safe_lora":  # project the LoRA update onto the span of (aligned - base), then merge the projected update into the aligned weights
            if not case.startswith("ft_"):
                raise ValueError("safe_lora needs an ft_* case (LoRA factors)")
            from peft import LoraConfig, get_peft_model
            from safetune.interventions.recover.safe_lora import apply_safe_lora
            lf, src, kw = case_deltas(case), dict(aligned.named_parameters()), dict(kw)
            if "frac" in kw:  # share of LoRA layers to project, least-aligned first: a dial that works whatever the real cosines are
                kw.update(select_layers_type="number", num_proj_layers=max(1, round(float(kw.pop("frac")) * len(lf["factors"]))))
            flags = {k: p.requires_grad for k, p in work.named_parameters()}  # get_peft_model freezes everything; put it back after
            with torch.no_grad():
                wp = dict(work.named_parameters())
                for k in lf["factors"]:  # undo reset()'s merge
                    wp[k].copy_(src[k])
            r = next(iter(lf["factors"].values()))["A"].shape[0]
            pm = get_peft_model(work, LoraConfig(r=r, lora_alpha=lf["scaling"] * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
            mods = {}
            with torch.no_grad():
                for k, f in lf["factors"].items():  # inverse of finetune_job.factors_of
                    mod = pm.get_submodule("base_model.model." + k.removesuffix(".weight"))
                    mod.lora_A["default"].weight.copy_(f["A"]); mod.lora_B["default"].weight.copy_(f["B"])
                    mods[k] = mod
                before = {k: mod.lora_B["default"].weight.detach().clone() for k, mod in mods.items()}
                # Safe LoRA builds its projection from aligned - base (the safety direction), moves one layer's pair to the GPU at a time; base stays on CPU
                apply_safe_lora(pm, aligned_state_dict=aligned.state_dict(), base_state_dict=base_cpu().state_dict(), **kw)
                proj = {k: (mod.lora_A["default"].weight.detach().clone(), mod.lora_B["default"].weight.detach().clone()) for k, mod in mods.items()}
            pm.unload()  # removes the LoRA wrappers without merging; returns the same object as work
            if hasattr(work, "peft_config"):
                del work.peft_config
            n_proj = sum(not torch.equal(before[k], proj[k][1]) for k in proj)
            with torch.no_grad():
                params = dict(work.named_parameters())
                for k, (A, B) in proj.items():
                    p = params[k]
                    p.copy_((p.float() + lf["scaling"] * (B.to(p.device).float() @ A.to(p.device).float())).to(p.dtype))
                for k, p in work.named_parameters():
                    p.requires_grad_(flags[k])
            log(f"safe_lora: projected {n_proj}/{len(proj)} lora layers kw={kw}")
        else:
            raise ValueError(f"unknown method {m}")
        return work, None, None

    def evaluate(spec):
        name, t = spec["name"], time.time()
        if spec["method"] == "diagnose":
            case = spec.get("case", CASE)
            return {"name": name, "spec": spec, "case": case, "model_id": E.MODEL_ID, "run": RUN, "diagnostics": diagnose(case),
                    "total_s": round(time.time() - t, 1), "_files": []}
        gen_model, cleanup, override = apply(spec)
        if QUANT4 and spec["method"] != "aligned":
            if gen_model is not work or cleanup or override:
                raise ValueError("QUANT4 covers weight-only specs; this one adds a hook or rewrites answers")
            gen_model, cleanup = quant4()
        t_apply = time.time() - t
        try:
            fwd = aligned if spec["method"] == "aligned" else gen_model if QUANT4 else work
            out, rec = {}, {"name": name, "spec": spec, "case": spec.get("case", CASE), "split": SPLIT, "harmful_set": HARMFUL_SET, "run": RUN, "model_id": E.MODEL_ID, "gen_batch": E.GEN_BATCH}
            for split, prompts in (("harmful", harmful), ("xstest", xsafe)):
                tg = time.time()
                resp = E.generate_completions(gen_model, tok, prompts)
                if QUANT4:
                    rec["quant"] = "bitsandbytes 4-bit nf4, bf16 compute, no double quant (original model not quantized)"
                    log(f"QUANT4 {name} {split}: {len(prompts)} answers in {time.time() - tg:.0f}s")
                if override:
                    resp = override(prompts, resp)
                flags = [bool(f) for f in E.string_refusal(resp)]
                out[split] = {"config_name": name, "split": split, "n": len(prompts), "prompts": prompts, "responses": resp,
                              "string_refusal": flags, "seed": E.SEED, "max_new_tokens": E.MAX_NEW_TOKENS, "dtype": str(E.DTYPE)}
                rec[f"string_{split}_refusal"] = sum(flags) / max(1, len(flags))
                rec[f"n_{split}"] = len(prompts)
            if skill_rows and (spec["method"] == "aligned" or rec["case"].startswith("ft_")):
                ans = E.generate_completions(gen_model, tok, [r["prompt"] for r in skill_rows], max_new_tokens=16)
                hits = [(a.strip().splitlines() or [""])[0].strip().lower() == r["completion"].lower() for a, r in zip(ans, skill_rows)]
                rec["skill"], rec["n_skill"], rec["skill_by_lang"] = sum(hits) / len(hits), len(hits), by_lang(skill_rows, hits)
            if gm_rows:  # through the same model object as generation, so a steering hook counts
                pick_ = E.last_token_logits(fwd, tok, gm_prompts)[:, gm_ids].argmax(-1).tolist()
                hits = ["ABCD"[i] == r["answer"] for i, r in zip(pick_, gm_rows)]
                rec["gmmlu"], rec["n_gmmlu"], rec["gmmlu_by_lang"] = sum(hits) / len(hits), len(hits), by_lang(gm_rows, hits)
            logits = E.last_token_logits(fwd, tok, klp)
            rec["kl"] = float(E.mean_kl(base_logits, logits))
            bl = state["broken_logits"].get(spec.get("case", CASE))
            if bl is not None:
                rec["kl_to_broken"] = float(E.mean_kl(bl, logits))
            # AuditKIT caches answers under the model's name, which encodes only the weights. A steering hook changes the answers
            # but not the weights, so without a per-spec tag a steering try got the cached MMLU of the unrepaired model.
            os.environ["EVAL_STAGE_TAG"] = f"{RUN}:{name}"
            acc, n_cap, note = E.eval_capability_checkpointed(fwd, tok, cap)
            rec.update(mmlu=float(acc), n_mmlu=int(n_cap), mmlu_note=note, apply_s=round(t_apply, 1), total_s=round(time.time() - t, 1),
                       gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0)
        finally:
            if cleanup:
                cleanup()
        for split, data in out.items():
            p = E.RUN_LOG_DIR / "completions" / name / f"{split}.json"
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(data))
            rec.setdefault("_files", []).append(p)
        return rec

    def commit(paths, msg):
        """One commit for all files of a spec (the repo allows 128 commits per hour). If it cannot be saved, stop the job:
        GPU time on results that are lost is wasted money."""
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=p.relative_to(E.RUN_DIR).as_posix(), path_or_fileobj=str(p)) for p in paths]
        for attempt in range(4):
            try:
                api.create_commit(REPO, ops, commit_message=msg, repo_type="dataset")
                return
            except Exception as exc:
                log(f"WARNING commit failed ({attempt + 1}/4) {type(exc).__name__}: {str(exc)[:200]}")
                time.sleep(30 * (attempt + 1))
        raise SystemExit(f"cannot save to the repo ({msg}); stopping so no GPU time is spent on unsaved results")

    def save(kind, name, obj, extra=()):
        p = E.RUN_LOG_DIR / kind / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=1))
        commit([*extra, p], f"{RUN} {kind} {name}")

    for i, spec in enumerate(todo):
        if time.time() - T0 > DEADLINE_S:
            log(f"DEADLINE reached; {len(todo) - i} specs left for the next launch")
            break
        gc.collect()  # a failed spec leaves temporaries alive through its traceback; free them before the next one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            rec = evaluate(spec)
            files = rec.pop("_files")
            save("metrics", spec["name"], rec, files)  # same commit as the completions: its presence means the spec is complete
            log(f"SPEC {spec['name']} " + " ".join(f"{k}={rec[k]}" for k in ("case", "string_harmful_refusal", "string_xstest_refusal", "skill", "mmlu", "kl", "kl_to_broken", "apply_s", "total_s", "gpu_gb", "diagnostics") if k in rec)[:900])
        except Exception as exc:  # one bad method must not sink the batch
            log(f"SPEC {spec['name']} FAILED {type(exc).__name__}: {str(exc)[:300]}")
            save("errors", spec["name"], {"name": spec["name"], "spec": spec, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
            work.zero_grad(set_to_none=True)
            park()
    E.upload_run_artifacts("repair_job end")
    log(f"repair_job done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
