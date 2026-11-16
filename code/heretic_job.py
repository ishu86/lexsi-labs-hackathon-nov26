# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "heretic-llm @ git+https://github.com/p-e-w/heretic.git@a14b8543537b50ce8db4db7eaae4eb295d5d5142",
#   "safetensors",
#   "huggingface_hub",
# ]
# ///
"""Unattended Heretic abliteration -> cases/<CASE>/deltas.pt in the private repo.

Phase 1 runs the Optuna search (study journal uploaded every 2 min, resumed on relaunch).
Phase 2 re-enters Heretic to merge the chosen trial; we keep only the weight deltas vs the
original model (o_proj / down_proj), never the merged model.
Env: HF_TOKEN, HF_CKPT_REPO, CASE, MODEL_ID, N_TRIALS, N_STARTUP, KL_CAP, MAX_RESP.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ["HF_CKPT_REPO"]
CASE = os.environ.get("CASE", "heretic_global")
MODEL_ID = os.environ.get("MODEL_ID", "CohereLabs/tiny-aya-global")
N_TRIALS = int(os.environ.get("N_TRIALS", "60"))
N_STARTUP = int(os.environ.get("N_STARTUP", "20"))
KL_CAP = float(os.environ.get("KL_CAP", "1.0"))
MAX_RESP = int(os.environ.get("MAX_RESP", "64"))
WORK = Path("/tmp/heretic_work"); WORK.mkdir(parents=True, exist_ok=True)
CKPT = WORK / "checkpoints"
LOG = WORK / "heretic.log"
api = HfApi()
T0 = time.time()


def say(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def upload(files: dict, tries=1):
    """files: {repo-relative name: local path}, sent as ONE commit (the repo allows 128 commits per hour)."""
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=f"cases/{CASE}/{rel}", path_or_fileobj=str(p)) for rel, p in files.items() if Path(p).is_file()]
    for attempt in range(tries):
        try:
            api.create_commit(REPO, ops, commit_message=f"{CASE} {' '.join(files)}"[:120], repo_type="dataset")
            return True
        except Exception as exc:
            say(f"WARNING upload {list(files)} failed ({attempt + 1}/{tries}): {type(exc).__name__}: {str(exc)[:160]}")
            if attempt + 1 < tries:
                time.sleep(60)
    return False


def sync_ckpt():
    upload({**{f"checkpoints/{p.name}": p for p in CKPT.glob("*.jsonl")}, "heretic.log": LOG})


def write_config():
    (WORK / "config.toml").write_text(f'''
model = "{MODEL_ID}"
max_response_length = {MAX_RESP}
modifiers = [{{ plugin = "heretic.modifiers.abliteration.Abliteration" }}]
n_trials = {N_TRIALS}
n_startup_trials = {N_STARTUP}
checkpoint_action = "continue"
trial_index = 0
model_action = "save"
export_strategy = "adapter"
save_directory = "{WORK / 'out_adapter'}"
''')


def run_heretic(tag):
    say(f"heretic start ({tag})")
    stop = threading.Event()

    def ticker():
        while not stop.wait(300):
            sync_ckpt()
    threading.Thread(target=ticker, daemon=True).start()
    proc = subprocess.Popen(["heretic"], cwd=WORK, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line.strip():
            say(f"H| {line[:300]}")
    rc = proc.wait()
    stop.set()
    say(f"heretic end ({tag}) rc={rc}")
    if rc != 0:
        raise SystemExit(f"heretic {tag} failed rc={rc}")


def study():
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    f = next(CKPT.glob("*.jsonl"))
    return optuna.load_study(study_name="heretic", storage=JournalStorage(JournalFileBackend(str(f), lock_obj=JournalFileOpenLock(str(f)))))


def choose(st):
    """Same order Heretic shows (scores ascending, refusals first); take the fewest refusals with KL under the cap."""
    def vals(t):
        return tuple(s["score"]["value"] for s in t.user_attrs["scores"])
    front = sorted(st.best_trials, key=vals)
    table = [{"front_index": i, "trial": t.user_attrs.get("index"), "scores": {s["name"]: s["score"]["value"] for s in t.user_attrs["scores"]}} for i, t in enumerate(front)]
    for row in table:
        say(f"front {row}")
    ok = [i for i, t in enumerate(front) if vals(t)[1] <= KL_CAP]
    idx = ok[0] if ok else max(range(len(front)), key=lambda i: -vals(front[i])[1])
    return idx, front[idx], table


def deltas_from(merged_dir: Path):
    import torch
    from safetensors import safe_open
    src = Path(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"]))
    orig = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if "o_proj" in k or "down_proj" in k:
                    orig[k] = h.get_tensor(k)
    out, norms = {}, {}
    for f in sorted(merged_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k in orig:
                    m = h.get_tensor(k)
                    d = m.float() - orig[k].to(m.dtype).float()  # same dtype first, so rounding is not counted as an edit
                    if float(d.abs().max()) > 0:
                        out[k] = d.half()
                        norms[k] = [float(d.norm()), float(orig[k].float().norm())]
    if not out:
        raise SystemExit("merged model equals the original on every o_proj/down_proj: Heretic edited nothing")
    return out, norms


def preflight():
    """Fail in the first minute if torch cannot use the GPU (Heretic's pyproject pins torch to the CPU wheel index;
    launch with -e UV_NO_SOURCES=1 so uv ignores that and takes the CUDA build from PyPI)."""
    import torch
    ok = torch.cuda.is_available()
    say(f"preflight torch={torch.__version__} cuda={ok} UV_NO_SOURCES={os.environ.get('UV_NO_SOURCES')}")
    if not ok:
        raise SystemExit("preflight: no CUDA in this torch build; relaunch with -e UV_NO_SOURCES=1")
    x = torch.ones(8, device="cuda", dtype=torch.bfloat16)
    say(f"preflight gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.mem_get_info()[1] / 2**30:.1f} op={float((x @ x).item())}")


def main():
    say(f"CASE={CASE} MODEL={MODEL_ID} trials={N_TRIALS} startup={N_STARTUP} kl_cap={KL_CAP}")
    preflight()
    try:  # resume an interrupted search
        snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"cases/{CASE}/checkpoints/*"], local_dir=str(WORK / "pull"))
        for p in (WORK / "pull" / "cases" / CASE / "checkpoints").glob("*.jsonl"):
            CKPT.mkdir(exist_ok=True); (CKPT / p.name).write_bytes(p.read_bytes()); say(f"resumed study file {p.name}")
    except Exception as exc:
        say(f"no checkpoint to resume ({type(exc).__name__})")
    write_config()
    run_heretic("search")
    st = study()
    idx, trial, table = choose(st)
    say(f"chosen front_index={idx} trial={trial.user_attrs.get('index')}")
    # A resumed run reads its settings from the study, not config.toml: point the stored settings at our pick.
    s = json.loads(st.user_attrs["settings"])
    s.update(trial_index=idx, export_strategy="merge", model_action="save")
    st.set_user_attr("settings", json.dumps(s))
    cfg = (WORK / "config.toml").read_text().replace("out_adapter", "out_merged")
    (WORK / "config.toml").write_text(cfg)
    run_heretic("merge")
    import torch
    deltas, norms = deltas_from(WORK / "out_merged")
    torch.save(deltas, WORK / "deltas.pt")
    meta = {"case": CASE, "model_id": MODEL_ID, "tool": "heretic a14b854 Abliteration", "n_trials": N_TRIALS, "kl_cap": KL_CAP,
            "chosen_front_index": idx, "chosen_trial": trial.user_attrs.get("index"), "chosen_scores": table[idx]["scores"],
            "parameters": trial.user_attrs.get("parameters"), "pareto_front": table, "n_tensors": len(deltas),
            "delta_norm_and_weight_norm": norms, "wall_s": round(time.time() - T0)}
    (WORK / "meta.json").write_text(json.dumps(meta, indent=1))
    if not upload({"deltas.pt": WORK / "deltas.pt", "meta.json": WORK / "meta.json", "heretic.log": LOG,
                   **{f"checkpoints/{p.name}": p for p in CKPT.glob("*.jsonl")}}, tries=12):  # the result: keep trying for 12 minutes
        raise SystemExit("deltas could not be uploaded")
    say(f"done: {len(deltas)} tensors changed, scores={table[idx]['scores']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sync_ckpt()
