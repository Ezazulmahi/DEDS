r"""Naturalness and prompt alignment (reviewer comments R1.4, R2.2, R1.3). No downloads: CLIP and the pytorch-fid
Inception weights are already cached.
Per image: CLIPScore (100 x cosine of the image and its own prompt), CLIP-IQA quality P("Good photo." vs "Bad photo.")
and naturalness P("A natural photo." vs "A synthetic, artificial image.").
Per set: FID and KID (polynomial kernel, x 1000) of Inception pool features against two real references, the DTD
texture photographs and the Picsum photographs. Writes results/rev_natural_images.csv and rev_natural_sets.csv.
usage: python rev_natural.py [extra set names to add later, e.g. full_run2 full_run3 reward_unfiltered]"""
import os, sys, glob
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
from scipy import linalg

torch.set_num_threads(2)
OUT = r"F:\DEDS_extra_checks\results"
CORP = r"F:\scirep results\09_expansion_1000\corpus1000"
ABL = r"F:\DEDS_extra_checks\ablation\covers"
DOM = r"F:\DEDS_extra_checks\test4_domains"


def corpus_prompts():
    ns = {}
    exec(open(r"F:\DEDS_diffusion_codebase\2_generation\prompts.py", encoding="utf-8").read(), ns)
    first = next(v for v in ns.values() if isinstance(v, list) and len(v) == 400)
    rest = open(r"F:\DEDS_diffusion_codebase\2_generation\prompts_400_999.txt", encoding="utf-8").read().splitlines()
    return first + rest


P = corpus_prompts(); assert len(P) == 1000
dom = pd.read_csv(os.path.join(DOM, "prompts.tsv"), sep="\t")
SETS = {"baseline": [(os.path.join(CORP, "baseline", f"{i:03d}.png"), P[i]) for i in range(1000)],
        "finetuned": [(os.path.join(CORP, "finetuned", f"{i:03d}.png"), P[i]) for i in range(1000)],
        "domains_baseline": [(os.path.join(DOM, "baseline", f"{r.index:03d}.png"), r.prompt) for r in dom.itertuples()],
        "domains_finetuned": [(os.path.join(DOM, "finetuned", f"{r.index:03d}.png"), r.prompt) for r in dom.itertuples()],
        "dtd": [(p, None) for p in sorted(glob.glob(r"F:\DEDS_extra_checks\test5_textures\covers\*.png"))],
        "picsum": [(p, None) for p in sorted(glob.glob(r"F:\DEDS_project\finetune_extra\covers\*.png"))[:195]]}
for v in ["full", "reward_only", "filter_only", "ordinary"] + sys.argv[1:]:
    SETS[f"abl_{v}"] = [(os.path.join(ABL, v, f"{i:03d}.png"), P[i]) for i in range(0, 1000, 5)]
SETS["abl_baseline"] = [(os.path.join(CORP, "baseline", f"{i:03d}.png"), P[i]) for i in range(0, 1000, 5)]
SETS["abl_full_80k"] = [(os.path.join(CORP, "finetuned", f"{i:03d}.png"), P[i]) for i in range(0, 1000, 5)]

from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def feats(out):
    return torch.nn.functional.normalize(out if hasattr(out, "norm") else out.pooler_output, dim=-1)


with torch.no_grad():
    ANT = feats(clip.get_text_features(**proc(text=["Good photo.", "Bad photo.", "A natural photo.",
                                                    "A synthetic, artificial image."], return_tensors="pt", padding=True)))

from pytorch_fid.inception import InceptionV3
inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()

rows, F = [], {}
for name, items in SETS.items():
    paths = [p for p, _ in items]
    assert all(os.path.exists(p) for p in paths), f"missing images in {name}"
    fs = []
    for b in range(0, len(items), 32):
        chunk = items[b:b + 32]
        ims = [Image.open(p).convert("RGB") for p, _ in chunk]
        with torch.no_grad():
            fi = feats(clip.get_image_features(**proc(images=ims, return_tensors="pt")))
            lg = 100 * fi @ ANT.T
            q = torch.softmax(lg[:, :2], 1)[:, 0]; nat = torch.softmax(lg[:, 2:], 1)[:, 0]
            tx = [t for _, t in chunk]
            cs = [float("nan")] * len(chunk)
            if tx[0] is not None:
                ft = feats(clip.get_text_features(**proc(text=tx, return_tensors="pt", padding=True, truncation=True)))
                cs = (100 * (fi * ft).sum(1)).tolist()
            x = torch.from_numpy(np.stack([np.asarray(im.resize((299, 299), Image.BILINEAR)) for im in ims])).permute(0, 3, 1, 2).float() / 255
            fs.append(inc(x)[0].squeeze(-1).squeeze(-1).numpy())
        for (p, _), a, b_, c in zip(chunk, cs, q.tolist(), nat.tolist()):
            rows.append(dict(set=name, image=os.path.basename(p), clipscore=a, quality=b_, natural=c))
    F[name] = np.concatenate(fs)
    print(name, len(items), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev_natural_images.csv"), index=False)


def fid(a, b):
    m1, m2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def kid(a, b, n=100, reps=50, seed=0):
    rng, d, v = np.random.default_rng(seed), a.shape[1], []
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    for _ in range(reps):
        x = a[rng.choice(len(a), min(n, len(a)), replace=False)]; y = b[rng.choice(len(b), min(n, len(b)), replace=False)]
        m = len(x)
        v.append((k(x, x).sum() - np.trace(k(x, x))) / (m * (m - 1)) + (k(y, y).sum() - np.trace(k(y, y))) / (m * (m - 1)) - 2 * k(x, y).mean())
    return 1000 * float(np.mean(v)), 1000 * float(np.std(v))


img = pd.DataFrame(rows); srows = []
for name in SETS:
    g = img[img.set == name]
    r = dict(set=name, n=len(g), clipscore=g.clipscore.mean(), quality=g.quality.mean(), natural=g.natural.mean())
    for ref in ("dtd", "picsum"):
        if name != ref:
            r[f"fid_{ref}"] = fid(F[name], F[ref]); r[f"kid_{ref}"], r[f"kid_{ref}_sd"] = kid(F[name], F[ref])
    srows.append(r)
pd.DataFrame(srows).to_csv(os.path.join(OUT, "rev_natural_sets.csv"), index=False)
print(pd.DataFrame(srows).round(3).to_string())
