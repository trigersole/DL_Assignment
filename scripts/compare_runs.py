from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai6103.plots import plot_experiment_comparison, read_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay learning curves from multiple runs")
    parser.add_argument("--run", action="append", type=Path, required=True, help="Run directory; repeat this flag")
    parser.add_argument("--label", action="append", required=True, help="Matching legend label; repeat this flag")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Experiment comparison")
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        parser.error("Each --run requires one matching --label")
    experiments = []
    for label, run_dir in zip(args.label, args.run):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            parser.error(f"Missing {metrics_path}")
        experiments.append((label, read_metrics(metrics_path)))
    plot_experiment_comparison(experiments, args.output, args.title)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
