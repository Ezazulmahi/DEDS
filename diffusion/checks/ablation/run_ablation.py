r"""Ablation driver (option A, 5,000 steps). Runs every step in order and skips what is already done, so it can be
restarted at any time; a STOP file in F:\DEDS_extra_checks makes it exit before the next step.
  1. preprocess.py on the filtered set (data_filtered) and on the matching unfiltered set (data_unfiltered)
  2. finetune.py (Imraj's script, unchanged) for four variants, 5,000 steps each, same settings otherwise:
       full          lambda_mid 0.005, anchor 1      filtered data    (the paper's objective)
       reward_only   lambda_mid 0.005, no anchor     filtered data
       filter_only   lambda_mid 0,     anchor 1      filtered data
       ordinary      lambda_mid 0,     anchor 1      unfiltered data
  3. eval_variant.py for each variant, plus the reference covers (baseline and the paper's 80,000-step model)."""
import os, subprocess, sys
from pathlib import Path

A = Path(r"F:\DEDS_extra_checks\ablation")
STOP = Path(r"F:\DEDS_extra_checks\STOP")
PY = sys.executable
VARIANTS = [("full", "data_filtered", ["--lambda_mid", "0.005", "--lambda_anchor", "1.0"]),
            ("reward_only", "data_filtered", ["--lambda_mid", "0.005", "--no_anchor"]),
            ("filter_only", "data_filtered", ["--lambda_mid", "0", "--lambda_anchor", "1.0"]),
            ("ordinary", "data_unfiltered", ["--lambda_mid", "0", "--lambda_anchor", "1.0"])]


def run(cmd, cwd, log):
    if STOP.exists():
        print("STOP file found, exiting", flush=True); sys.exit(1)
    print(">>", " ".join(map(str, cmd)), "in", cwd, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"FAILED ({r.returncode}), see {log}", flush=True); sys.exit(r.returncode)


env_cache = {**os.environ, "HF_HUB_CACHE": str(A / "hf_cache")}
# 1. data
run([PY, "-I", "-X", "utf8", A / "code" / "preprocess.py"], A / "data_filtered", A / "data_filtered" / "preprocess.log")
if not (A / "data_unfiltered" / "filtered_dataset" / "index.csv").exists() or \
        len(list((A / "data_unfiltered" / "filtered_dataset").glob("img_*.jpg"))) < 4246:
    print(">> make_unfiltered.py", flush=True)
    r = subprocess.run([PY, "-I", "-X", "utf8", str(A / "make_unfiltered.py")], env=env_cache,
                       stdout=open(A / "make_unfiltered.log", "a"), stderr=subprocess.STDOUT)
    if r.returncode:
        print("make_unfiltered failed", flush=True); sys.exit(1)
run([PY, "-I", "-X", "utf8", A / "code" / "preprocess.py"], A / "data_unfiltered", A / "data_unfiltered" / "preprocess.log")
for d in ("data_filtered", "data_unfiltered"):
    n = sum(1 for _ in open(A / d / "processed_dataset" / "manifest.csv", encoding="utf-8")) - 1
    print(f"{d}: {n} training images", flush=True)

# 2. train every variant first, so an evaluation problem cannot hold up the GPU training
for name, data, args in VARIANTS:
    out = A / "runs" / name
    if not (out / "final" / "unet.pt").exists():
        # checkpoints every 1,000 steps; after an interruption training resumes from the latest complete one
        done = sorted(p for p in out.glob("checkpoint_step*") if (p / "ema.pt").exists()) if out.exists() else []
        resume = ["--resume", done[-1]] if done else []
        run([PY, "-I", "-X", "utf8", A / "code" / "finetune.py", "--steps", "5000", "--checkpoint_every", "1000",
             "--output_dir", out, *resume, *args], A / data, A / "runs" / f"{name}.log")
# 3. evaluate each variant, then the reference covers
for name, _, _ in VARIANTS:
    run([PY, "-I", "-X", "utf8", A / "eval_variant.py", name, A / "runs" / name / "final"], A, A / "runs" / f"{name}_eval.log")
run([PY, "-I", "-X", "utf8", A / "eval_variant.py", "--reference"], A, A / "runs" / "reference_eval.log")
print("ablation complete", flush=True)
