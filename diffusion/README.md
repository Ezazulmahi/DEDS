# Diffusion code

Everything that trains the LoRA adapter, plus the later checks that generate new covers with it.
The code that generated the paper's evaluation corpus is in `../generation/` (`validate.py` loads Stable
Diffusion v1.5 with the adapter in `../lora_weights/`), so it is not repeated here.

## training/
The fine-tuning pipeline, also published at https://github.com/Imraj-Rabbani/P3-Image-Steganography-Pipeline
(commit `83f98cf`, MIT licence, same authors).

    filter_dataset.py → preprocess.py → check.py → finetune.py → validate.py (+ sweep_checkpoints.py)

* `filter_dataset.py` scans `gmongaras/Imagenet21K_Recaption` and keeps images in which more than 30% of the
  8×8 luminance blocks have all of their first twenty AC coefficients non-zero after quantisation.
* `preprocess.py` centre-crops and resizes to 256×256 and stores the block DCT.
* `finetune.py` trains a rank-4 LoRA on the U-Net attention projections with
  `L = L_diff + λ_mid·L_mid + λ_anchor·L_anchor` (defaults λ_mid = 0.005, λ_anchor = 1, batch 2 × 4,
  learning rate 1e-5, 80,000 steps). Options: `--steps --lambda_mid --lambda_anchor --no_anchor --output_dir`.
  The default `--output_dir` points to a D: drive; pass your own folder.
* This folder's `validate.py` is the pipeline's own validation stage. The paper's covers were generated with
  `../generation/validate.py`.

## checks/
Later checks that use the trained adapter. Folder paths at the top of each script are those of our machine.

* `broader_domains/`: 300 cover pairs from six domains outside the texture prompt bank (landscapes, urban
  scenes, animals, food and objects, indoor scenes, and non-photographic styles), generated with the corpus
  sampler and seeds 300000 + i, then measured and embedded with the matched-payload protocol.
  `results_test4_domains.csv` has one row per cover.
* `eligible_slot_capacity.py`: for every corpus cover, how many of its raw slots at its own K hold an
  eligible coefficient (|q| ≥ 2). Per-image results in `eligible_slot_capacity.csv`.
* `ablation/`: 5,000-step ablation of the training objective on one shared training set: the full objective,
  reward without the anchor, anchor without the reward, and ordinary fine-tuning on unfiltered images.
  `run_ablation.py` runs every step; `eval_variant.py` evaluates a trained variant on prompts 0, 5, …, 995.
* `common.py`: shared helpers (cover statistics, spectra, embedding wrappers).
