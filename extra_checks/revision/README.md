# Checks added in revision

Scripts and results of the analyses added while revising the paper. Every script saves its results as it goes and
skips finished items when run again. Folder paths near the top of each script point to our machine.

| Script | Paper | Results |
|---|---|---|
| `code/rev_cpu.py` (gain split) | Results, Embedding distortion; Supplementary Table S19 | `results/rev_gain_split.csv`: 200 pairs embedded with each combination of margin (0.5 or 0.15) and the old $|q'|\ge2$ preference, PSNR after the first write, after the verify-fix loop and in the final image |
| `code/rev_cpu.py` (offsets) | Supplementary Table S20 | `results/rev_offsets.csv`: within-bin offsets of the used slots per pair, cover type and class, with the measured first-write change |
| `code/rev_cpu.py` (sweep) and `code/rev_equal_quality.py` | Results, Embedding capacity; Supplementary Table S22 | `results/rev_rd.csv` (1,195 covers at 20 payloads, 23,900 embeddings, all recovered exactly) and `results/rev_capacity_equal_quality.csv` (largest payload meeting PSNR 38 or 40 dB, SSIM 0.98 or 0.99) |
| `code/rev_natural.py` | Supplementary Table S21 | `results/rev_natural_images.csv` (CLIPScore and CLIP quality per image), `results/rev_natural_sets.csv` (FID and KID of each cover set to the DTD and Lorem Picsum photographs) |
| `code/rev_abl_detect.py` | Supplementary Table S18, last column | `results/rev_abl_detect.csv`: SRM AUC of DEDS at matched payload on the covers of every ablation variant, five-fold cross-validation over pairs |
| `code/srm_sanity.py` | Results, Detectability; Methods | controls for the matched SRM result: true labels AUC 1.000, shuffled labels 0.487, two halves of the covers 0.521 |
| `code/fig2_larger_labels.py`, `code/fig3_s1_larger_labels.py` | Figs. 2, 3 and S1 | the same figures with larger labels |
| `code/crypto_roundtrip.py` | Methods, Encrypted messages; Supplementary Table S23 | `results/crypto_roundtrip.csv`: AES-256-GCM with the key wrapped by RSA-2048 OAEP, embedded in 100 covers at each of 0.30, 0.40 and 0.60 bits per pixel, extracted from the reloaded PNG and decrypted (300 of 300 exact), with pixel and block tamper tests (never a wrong message). Needs the Python `cryptography` package; test keys are generated per run and not saved |

The matched steganalysis of the paper's embedder (margin 0.15, 0.4 bits per pixel in training and testing) is in
`steganalysis/` (scripts `07c`, `10c` and `14c`, results `srnet_kv_matched-deds_v5m*` and `srm_fld/deds_v5m_*`).
The two repeat runs of the full objective and the run on unfiltered data are in `diffusion/checks/ablation/`.
