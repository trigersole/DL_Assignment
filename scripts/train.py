from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai6103.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S for one AI6103 experiment")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=6103)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        ExperimentConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            epochs=args.epochs,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
            mixup_alpha=args.mixup_alpha,
            batch_size=args.batch_size,
            momentum=args.momentum,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            resume=args.resume,
        )
    )
    print(f"Saved results to {args.output_dir.resolve()}")
    print(summary["final"])


if __name__ == "__main__":
    main()

