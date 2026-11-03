r"""Ablation evaluation. For one trained variant (a finetune.py checkpoint folder), generate the covers of corpus
prompts 0, 5, ..., 995 with the corpus sampler and seeds (validate.py loader, 256x256, 50 DDIM steps, no CFG),
then measure them next to the stored corpus baseline covers and the paper's 80,000-step covers of the same pairs:
eligible count, fallback, raw and eligible-slot capacity, spectrum, CLIP similarity to the baseline cover, and
matched-payload embedding (same K and message on every cover type) with exact recovery, PSNR and SSIM.

usage:  python eval_variant.py NAME CHECKPOINT_DIR        (generation on the GPU, then CPU evaluation)
        python eval_variant.py --reference                (baseline and 80k corpus covers only, no GPU)
Resumable; rows go to F:\DEDS_extra_checks\results\ablation_eval.csv keyed by variant:index."""
import os, sys, io, math, random, string, contextlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, cv2

sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

HERE = Path(__file__).resolve().parent
OUT = r"F:\DEDS_extra_checks\results\ablation_eval.csv"
IDX = list(range(0, 1000, 5))
ALPHABET = string.ascii_letters + string.digits + " .,!?"
D = None


def generate(name, ckpt):
    import torch
    spec = importlib.util.spec_from_file_location("gen", r"F:\Steganalyzer_UniversalDCT\01_generate_covers.py")
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    V = G.V
    out = HERE / "covers" / name; out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in IDX if not (out / f"{i:03d}.png").exists()]
    if not todo:
        return
    tok, enc, vae, unet, sched = V.load_pipeline(G.dev, G.dt, finetuned=True, finetuned_dir=Path(ckpt))
    # validate.py loads with strict=False, so confirm that every LoRA tensor in the checkpoint reached the model
    sd = torch.load(Path(ckpt) / "unet.pt", map_location="cpu")
    model = dict(unet.named_parameters())
    lora = [k for k in sd if "lora_" in k]
    hit = [k for k in lora if k in model and torch.equal(model[k].detach().float().cpu(), sd[k].float())]
    assert lora and len(hit) == len(lora), f"LoRA weights not loaded: {len(hit)} of {len(lora)} matched"
    assert any(sd[k].abs().sum() > 0 for k in lora if "lora_B" in k), "lora_B all zero: adapter untrained"
    print(f"{name}: {len(hit)} LoRA tensors loaded and checked", flush=True)
    sched.set_timesteps(V.DDIM_STEPS)
    prompts = V.PROMPTS + Path(r"F:\scirep results\09_expansion_1000\prompts_400_999.txt").read_text(encoding="utf-8").splitlines()
    for b in range(0, len(todo), 10):
        ks = todo[b:b + 10]
        imgs = G.generate_batch(tok, enc, vae, unet, sched, [prompts[i] for i in ks], [42 + i for i in ks])
        for i, im in zip(ks, imgs):
            cv2.imwrite(str(out / f"{i:03d}.png"), V.float_rgb_to_uint8_bgr(im))
        print(f"{name}: {min(b + 10, len(todo))}/{len(todo)} generated", flush=True)
    del tok, enc, vae, unet, sched; torch.cuda.empty_cache()


def clip_similarity(name):
    """Cosine similarity of CLIP image embeddings between each cover and its baseline cover."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def emb(paths):
        ims = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in paths]
        with torch.no_grad():
            e = m.get_image_features(**p(images=ims, return_tensors="pt"))
        e = e if hasattr(e, "norm") else e.pooler_output   # transformers >= 5 returns an output object
        return torch.nn.functional.normalize(e, dim=-1)
    sims = {}
    for b in range(0, len(IDX), 20):
        ks = IDX[b:b + 20]
        a = emb([cover_path("baseline", i) for i in ks]); c = emb([cover_path(name, i) for i in ks])
        for i, s in zip(ks, (a * c).sum(-1).tolist()):
            sims[i] = s
    return sims


def cover_path(name, i):
    if name == "baseline":
        return os.path.join(C.CORP, "baseline", f"{i:03d}.png")
    if name == "full_80k":
        return os.path.join(C.CORP, "finetuned", f"{i:03d}.png")
    return str(HERE / "covers" / name / f"{i:03d}.png")


def _init():
    global D
    cv2.setNumThreads(1)
    D = C.load_v5()


def work(task):
    key, name, i, clip = task
    K, n_bits = TASKS[i]
    msg = "".join(random.Random(620000 + i).choice(ALPHABET) for _ in range(n_bits // 8))
    cover = cv2.imread(cover_path(name, i))
    st = C.cover_stats(D, cover)
    own = st["K"] if st["K"] is not None else D.FALLBACK_K
    I, J, U, V, Q = D._slot_arrays(D._build_slot_order(own, 32, 32))
    blocks, _, _ = D._to_blocks(C.luma(cover))
    elig = int((np.abs(np.round(blocks[I, J, U, V] / Q)) >= 2).sum())
    row = dict(key=key, variant=name, index=i, e_bar=st["e_bar"], fallback=st["K"] is None, own_K=own,
               raw_bpp=1022 * (own + 1) / 65536, elig_slot_bpp=elig / 65536,
               ge030=st["K"] is not None and st["capacity_bpp"] >= 0.30, clip_to_baseline=clip,
               **C.spectrum_stats(cover), K=K, bpp=n_bits / 65536)
    o = D._compute_K; D._compute_K = lambda b, K=K: K
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stego = D.embed(cover, msg)
        row["error"] = ""
    except Exception as e:
        stego, row["error"] = None, repr(e)
    finally:
        D._compute_K = o
    if stego is not None:
        stego = C.png_roundtrip(stego, str(HERE / "stego_tmp" / f"{name}_{i:03d}.png"))
        with contextlib.redirect_stdout(io.StringIO()):
            row["exact"] = D.extract(stego) == msg
        row.update(C.quality(cover, stego))
    return [row]


TASKS = C.matched_tasks()

if __name__ == "__main__":
    (HERE / "stego_tmp").mkdir(exist_ok=True)
    if sys.argv[1] == "--reference":
        names = ["baseline", "full_80k"]
    else:
        names = [sys.argv[1]]
        generate(sys.argv[1], sys.argv[2])
    for name in names:
        sims = {i: 1.0 for i in IDX} if name == "baseline" else clip_similarity(name)
        ok = C.run_resumable([(f"{name}:{i}", name, i, sims[i]) for i in IDX], work, OUT, init=_init, workers=3)
        if not ok:
            sys.exit(1)
