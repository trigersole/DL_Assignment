from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai6103.experiment import evaluate_test


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time evaluation on the held-out test split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "final_test.json")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate_test(args.checkpoint, args.data_root, args.output, args.num_workers, args.device)
    print(result)


if __name__ == "__main__":
    main()

