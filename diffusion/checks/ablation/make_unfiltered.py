r"""Unfiltered training set for the 'ordinary fine-tuning' ablation: 4,246 images (the size of the filtered set) taken evenly (every 34th row)
from the same ImageNet-21K-Recaption shards that the filtered set came from, with no texture filter, written in
the exact layout filter_dataset.py produces (img_XXXXXX.jpg + .txt + index.csv) so preprocess.py runs unchanged.
Reads the shards from the local Hugging Face cache (HF_HUB_CACHE); downloads nothing new."""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from huggingface_hub import hf_hub_download, list_repo_files

REPO, TARGET, STRIDE, N_SHARDS = "gmongaras/Imagenet21K_Recaption", 4_246, 34, 87   # same count as the filtered set
OUT = Path(r"F:\DEDS_extra_checks\ablation\data_unfiltered\filtered_dataset")   # name preprocess.py expects
OUT.mkdir(parents=True, exist_ok=True)
shards = sorted(p for p in list_repo_files(REPO, repo_type="dataset") if p.endswith(".parquet"))[:N_SHARDS]
n = 0
with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["filename", "caption", "good_block_ratio"])
    for s in shards:
        df = pd.read_parquet(hf_hub_download(REPO, s, repo_type="dataset", local_files_only=True))
        for k, row in enumerate(df.to_dict("records")):
            if k % STRIDE:
                continue
            v = row.get("image"); b = v.get("bytes") if isinstance(v, dict) else v
            if b is None:
                continue
            fname = f"img_{n:06d}.jpg"
            if b[:2] == b"\xff\xd8":
                (OUT / fname).write_bytes(b)
            else:
                bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                cv2.imwrite(str(OUT / fname), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cap = str(row.get("recaption_short") or "").strip().replace("\n", " ").replace("\t", " ")[:220]
            (OUT / f"img_{n:06d}.txt").write_text(cap, encoding="utf-8")
            w.writerow([fname, cap, ""])
            n += 1
            if n >= TARGET:
                break
        print(f"{s}: total {n}", flush=True)
        if n >= TARGET:
            break
print("unfiltered images:", n)
sys.exit(0 if n >= TARGET else 1)
