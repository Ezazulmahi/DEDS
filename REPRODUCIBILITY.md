# Reproducibility manifest

This file lists what is needed to repeat every result of the DEDS paper, what is released here, and what could not
be recorded. Folder names refer to this repository.

## Training data
* Source: the public Hugging Face dataset `gmongaras/Imagenet21K_Recaption` (captions from the dataset, images from
  ImageNet-21K). The dataset page states no licence; the images remain subject to the ImageNet terms of access, so they
  are not redistributed here.
* Filtering: `diffusion/training/filter_dataset.py` reads the Parquet shards in sorted order and keeps an image when more
  than 30% of its 8×8 luminance blocks have all of their first twenty AC coefficients non-zero after quantisation. It
  stopped at 50,000 accepted images. The filter is deterministic, so the same shard order gives the same images.
* Standardisation: `diffusion/training/preprocess.py` drops images whose shorter side is under 250 pixels, centre-crops
  and resizes the rest to 256×256 (Lanczos), leaving 40,101 training images.
* No duplicate or near-duplicate removal was performed.
* There is no held-out split of the training images: the model is evaluated on generated covers from a separate prompt
  bank (`prompts/`), not on training images.

## Training
* Script: `diffusion/training/finetune.py`, unchanged. Rank-4 LoRA (alpha 8, dropout 0.05) on the query, key and value
  projections of every U-Net attention layer; loss weights lambda_mid = 0.005 and lambda_anchor = 1; batch 2 with 4
  accumulation steps; AdamW, learning rate 1e-5 with 1,000 warm-up steps; 80,000 steps.
* The training script sets no random seed, so repeated runs differ; the ablation and the first two repeated runs of
  the revision are independent runs of the same script.
* Two further repeats of the full objective (5,000 steps) used recorded seeds 2026 and 2027, set in a copy of the
  script that fixes the Python, NumPy and PyTorch generators and the data order before the adapter is created
  (`diffusion/checks/ablation/seeded/`). A second start of seed 2026 matched the first training steps to four decimal
  places; the remaining differences come from non-deterministic GPU arithmetic.
* Released weights: the final adapter in `lora_weights/`. Intermediate checkpoints and the moving-average weights were
  not released; the covers were generated from the final-step weights, not the moving average.

## Generation and evaluation seeds
| Images | Prompts | Seeds |
|---|---|---|
| Evaluation corpus, 1,000 pairs | `prompts/` 0 to 999 | 42 + i |
| Unseen-prompt check, 200 pairs | `prompts/prompts_unseen.txt` | 200000 + i |
| Broader domains, 300 pairs | `diffusion/checks/broader_domains/prompts.tsv` | 300000 + i |
| Steganalysis training covers, 4,000 | first 400 prompts of the bank | 100000 upwards |
| Messages, matched-payload sweep | 67-character alphabet | 424242 + i |

Sampler for every cover: DDIM, 50 steps, no classifier-free guidance, 256×256, except where a check states otherwise.

## Software
* Evaluation, embedding, steganalysis, the 600 later corpus pairs and every later check: NVIDIA GeForce RTX 5060, AMD
  Ryzen 5 7500F, Python 3.14, PyTorch 2.11.0 (CUDA 12.8), Diffusers 0.39.0, Transformers 5.13.0, PEFT 0.19.1,
  NumPy 2.4.4, SciPy 1.17.1, OpenCV 4.13.0, scikit-image 0.26.0, pandas 3.0.5. Full list in `requirements.txt`.
* Training and the first 400 corpus pairs: NVIDIA RTX 4080 under Windows 11 with Python 3.10 or later. The library
  versions of that machine were not recorded. The first twelve covers of each set regenerate pixel for pixel from the
  released weights with the code here.

## Third-party data and terms
* Stable Diffusion v1.5: CreativeML OpenRAIL-M licence.
* Describable Textures Dataset (texture-photograph check): released by its authors for research use.
* Reference photographs (spectral comparison): Lorem Picsum service, images under the Unsplash licence; identifiers listed
  in the repository.
* Our code: MIT licence (`LICENSE`).
