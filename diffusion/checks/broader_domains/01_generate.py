r"""Test 4, step 1: broader image domains. 300 baseline / fine-tuned cover pairs from six domains that are neither
in the evaluation prompt bank nor texture close-ups, generated with the existing models and the corpus sampler
(256x256, 50 DDIM steps, no classifier-free guidance). Seeds 300000 + index, used for no other image.
No training. Resumable: existing PNGs are skipped; a STOP file in F:\DEDS_extra_checks stops it after the batch."""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, torch, cv2

HERE = Path(__file__).resolve().parent
STOP = HERE.parent / "STOP"
spec = importlib.util.spec_from_file_location("gen", r"F:\Steganalyzer_UniversalDCT\01_generate_covers.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
V = G.V

DOMAINS = {
    "landscape": (["a mountain range at sunrise", "a desert with sand and rocks", "a waterfall in a forest", "a calm beach at sunset",
                   "a field of sunflowers", "a snowy pine forest", "a river valley with hills", "a rocky coastline",
                   "a meadow with wildflowers", "a volcano under a cloudy sky"],
                  ["wide angle", "golden hour light", "overcast day", "aerial view", "photographed in spring"]),
    "urban": (["a busy city intersection", "an old stone cathedral", "a modern glass skyscraper", "a narrow cobblestone alley",
               "a train station platform", "a row of colourful houses", "a harbour with fishing boats", "a suspension bridge",
               "a parking garage", "a rooftop with water tanks"],
              ["street photography", "at dusk", "symmetrical composition", "rainy weather", "midday sun"]),
    "animals": (["a cat sleeping on a sofa", "a horse in a paddock", "a flock of sheep on a hill", "a parrot on a branch",
                 "a fox in the snow", "an elephant at a waterhole", "a goldfish in a bowl", "a cow in a green field",
                 "a pair of ducks on a pond", "a butterfly on a leaf"],
                ["wildlife photograph", "close portrait", "soft natural light", "telephoto lens", "shallow depth of field"]),
    "food_objects": (["a plate of pasta", "a loaf of bread on a board", "a stack of pancakes", "a glass of orange juice",
                      "a pair of headphones", "an old typewriter", "a ceramic teapot", "a wristwatch on a table",
                      "a basket of eggs", "a set of kitchen knives"],
                     ["product photograph", "studio lighting", "top-down view", "on a white background", "warm light"]),
    "indoor": (["a living room with a fireplace", "a classroom with desks", "a hospital corridor", "a library reading room",
                "a small bathroom", "an office with computers", "a restaurant dining room", "a bedroom with a large window",
                "a workshop with tools", "a museum gallery"],
               ["interior photograph", "natural window light", "wide angle", "evening lamps", "minimalist style"]),
    "styles": (["a lighthouse by the sea", "a bowl of fruit", "a quiet village street", "a sailing ship", "a mountain cabin",
                "a cup of tea", "a city skyline", "a garden with a bench", "a vintage car", "a flower in a vase"],
               ["oil painting", "watercolour painting", "pencil sketch", "cartoon illustration", "3D render"]),
}
ITEMS = []
for dom, (subjects, mods) in DOMAINS.items():
    for s in subjects:
        for m in mods:
            ITEMS.append((dom, f"{m} of {s}" if dom == "styles" else f"a photograph of {s}, {m}"))
PROMPTS = [p for _, p in ITEMS]
assert len(PROMPTS) == 300 and len(set(PROMPTS)) == 300 and not (set(PROMPTS) & set(V.PROMPTS))
SEED0, BATCH = 300000, 10

if __name__ == "__main__":
    (HERE / "prompts.tsv").write_text("index\tdomain\tprompt\n" + "\n".join(f"{i}\t{d}\t{p}" for i, (d, p) in enumerate(ITEMS)),
                                      encoding="utf-8")
    for finetuned, tag in ((True, "finetuned"), (False, "baseline")):
        out = HERE / tag; out.mkdir(exist_ok=True)
        todo = [b for b in range(0, 300, BATCH) if not all((out / f"{k:03d}.png").exists() for k in range(b, b + BATCH))]
        if not todo:
            print(tag, "already complete", flush=True); continue
        tok, enc, vae, unet, sched = V.load_pipeline(G.dev, G.dt, finetuned=finetuned, finetuned_dir=Path(r"F:\Results"))
        sched.set_timesteps(V.DDIM_STEPS)
        for b in todo:
            if STOP.exists():
                print("STOP file found, exiting; finished batches are saved", flush=True); sys.exit(1)
            imgs = G.generate_batch(tok, enc, vae, unet, sched, PROMPTS[b:b + BATCH], [SEED0 + k for k in range(b, b + BATCH)])
            for k, im in zip(range(b, b + BATCH), imgs):
                cv2.imwrite(str(out / f"{k:03d}.png"), V.float_rgb_to_uint8_bgr(im))
            print(f"{tag} {b + BATCH}/300", flush=True)
        del tok, enc, vae, unet, sched; torch.cuda.empty_cache()
    print("generation done", flush=True)
