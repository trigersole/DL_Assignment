from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final metrics from completed experiments")
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or args.outputs / "experiment_summary.csv"
    rows = []
    for path in sorted(args.outputs.glob("**/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        final = payload["final"]
        rows.append(
            {
                "experiment": str(path.parent.relative_to(args.outputs)),
                "lr": config["learning_rate"],
                "scheduler": config["scheduler"],
                "weight_decay": config["weight_decay"],
                "mixup_alpha": config["mixup_alpha"],
                "epochs": config["epochs"],
                "train_loss": final["train_loss"],
                "train_accuracy": final["train_accuracy"],
                "validation_loss": final["validation_loss"],
                "validation_accuracy": final["validation_accuracy"],
            }
        )
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.outputs}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()

