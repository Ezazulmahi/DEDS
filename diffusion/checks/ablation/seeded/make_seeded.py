"""Builds finetune_seeded.py: the ablation copy of finetune.py with an explicit, recorded seed (reviewer 2, comment 7).
The seed fixes Python, NumPy and PyTorch (CPU and CUDA) before the LoRA adapter is created, and the DataLoader order.
Nothing else changes. The original finetune.py is not touched."""
from pathlib import Path

src = Path(r"F:\DEDS_extra_checks\ablation\code\finetune.py").read_text(encoding="utf-8")
s = src
reps = [
    ('    parser.add_argument("--lr",            type=float, default=LR)',
     '    parser.add_argument("--lr",            type=float, default=LR)\n'
     '    parser.add_argument("--seed",          type=int,   required=True)'),
    ('def run_training(args: argparse.Namespace) -> None:\n',
     'def run_training(args: argparse.Namespace) -> None:\n'
     '    import random\n'
     '    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)\n'
     '    Path(args.output_dir).mkdir(parents=True, exist_ok=True)\n'
     '    (Path(args.output_dir) / "seed.txt").write_text(f"{args.seed}\\n")\n'
     '    print(f"[seeded] seed {args.seed}", flush=True)\n'),
    ('        persistent_workers = (args.num_workers > 0),\n    )',
     '        persistent_workers = (args.num_workers > 0),\n'
     '        generator          = torch.Generator().manual_seed(args.seed),\n    )'),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
Path(r"F:\DEDS_extra_checks\seeded\finetune_seeded.py").write_text(s, encoding="utf-8")
print("written")
