from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm

from .data import prepare_datasets
from .plots import plot_training_curves


@dataclass
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    learning_rate: float
    epochs: int
    scheduler: str = "none"
    weight_decay: float = 0.0
    mixup_alpha: float = 0.0
    batch_size: int = 128
    momentum: float = 0.9
    seed: int = 6103
    num_workers: int = 4
    device: str = "auto"
    resume: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed = coefficient * images + (1.0 - coefficient) * images[permutation]
    return mixed, targets, targets[permutation], coefficient


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, mixup_alpha=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "evaluate")
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_a = target_b = targets
        coefficient = 1.0
        if training and mixup_alpha > 0:
            images, target_a, target_b, coefficient = _mixup(images, targets, mixup_alpha)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = coefficient * criterion(logits, target_a) + (1.0 - coefficient) * criterion(logits, target_b)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)
        correct = coefficient * predictions.eq(target_a).sum().item()
        correct += (1.0 - coefficient) * predictions.eq(target_b).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_examples += batch_size
        iterator.set_postfix(loss=f"{total_loss / total_examples:.4f}")
    return total_loss / total_examples, total_correct / total_examples


def _write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: ExperimentConfig) -> dict:
    if config.scheduler not in {"none", "cosine"}:
        raise ValueError("scheduler must be 'none' or 'cosine'")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = config.output_dir.parent.parent / "data_artifacts"
    datasets, class_to_index, mean, std = prepare_datasets(
        config.data_root, artifacts_dir, config.seed, config.batch_size, config.num_workers
    )
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
        if split != "test"  # The test set is intentionally unavailable during tuning.
    }

    model = efficientnet_v2_s(weights=None, num_classes=len(class_to_index)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.0)
        if config.scheduler == "cosine"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 1
    metrics: list[dict[str, float]] = []
    checkpoint_path = config.output_dir / "last_checkpoint.pt"
    if config.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        metrics = checkpoint["metrics"]
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            generator.set_state(checkpoint["loader_generator_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    started = time.time()
    best_validation_loss = min((row["validation_loss"] for row in metrics), default=float("inf"))
    for epoch in range(start_epoch, config.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixup_alpha=config.mixup_alpha,
        )
        with torch.inference_mode():
            validation_loss, validation_accuracy = _run_epoch(
                model, loaders["validation"], criterion, device
            )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        metrics.append(row)
        if scheduler is not None:
            scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "class_to_index": class_to_index,
            "mean": mean,
            "std": std,
            "config": {**asdict(config), "data_root": str(config.data_root), "output_dir": str(config.output_dir)},
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator_state": generator.get_state(),
        }
        torch.save(checkpoint, checkpoint_path)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(checkpoint, config.output_dir / "best_checkpoint.pt")
        _write_metrics(config.output_dir / "metrics.csv", metrics)
        print(
            f"epoch={epoch:03d} lr={learning_rate:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )

    plot_training_curves(metrics, config.output_dir / "training_curves.png", config.output_dir.name)
    summary = {
        "config": {**asdict(config), "data_root": str(config.data_root), "output_dir": str(config.output_dir)},
        "device": str(device),
        "classes": len(class_to_index),
        "normalization": {"mean": mean, "std": std},
        "final": metrics[-1],
        "best_validation_epoch": int(min(metrics, key=lambda row: row["validation_loss"])["epoch"]),
        "elapsed_seconds": time.time() - started,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_test(checkpoint_path: Path, data_root: Path, output_path: Path, num_workers: int, device_name: str):
    """Evaluate a selected checkpoint on the hold-out test set exactly when explicitly requested."""
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    artifacts_dir = checkpoint_path.parent.parent.parent / "data_artifacts"
    datasets, class_to_index, _, _ = prepare_datasets(
        data_root,
        artifacts_dir,
        int(config["seed"]),
        int(config["batch_size"]),
        num_workers,
    )
    if class_to_index != checkpoint["class_to_index"]:
        raise ValueError("Checkpoint class mapping does not match the dataset")
    model = efficientnet_v2_s(weights=None, num_classes=len(class_to_index)).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        datasets["test"],
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        loss, accuracy = _run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    result = {"checkpoint": str(checkpoint_path), "test_loss": loss, "test_accuracy": accuracy}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
