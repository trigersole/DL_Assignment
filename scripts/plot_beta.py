from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai6103.plots import plot_beta_density


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the mixup Beta density")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mixup_beta_density.png")
    args = parser.parse_args()
    plot_beta_density(args.alpha, args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()

