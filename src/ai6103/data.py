from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_TOTAL = 10_222
SPLIT_SIZES = {"train": 7_155, "validation": 1_533, "test": 1_534}
PRE_SPLIT_LABEL_STEMS = {
    "train_labels": "train",
    "val_labels": "validation",
    "validation_labels": "validation",
    "test_labels": "test",
}
SPLIT_DIRECTORY_NAMES = {
    "train": {"train"},
    "validation": {"val", "validation"},
    "test": {"test"},
}


@dataclass(frozen=True)
class Record:
    image_id: str
    breed: str
    path: Path
    split: str


def _find_labels(data_root: Path) -> Path:
    candidates = [data_root / "labels.csv", data_root / "train" / "labels.csv"]
    candidates.extend(sorted(data_root.glob("**/labels.csv")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find labels.csv under {data_root}. Run scripts/download_data.py "
        "or place the extracted assignment dataset there."
    )


def _index_images(data_root: Path) -> dict[str, Path]:
    # Prefer the labelled-image directories so an original Kaggle test folder is ignored.
    search_roots = [data_root / "train", data_root / "images", data_root]
    for root in search_roots:
        if not root.is_dir():
            continue
        index = {
            path.stem: path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if len(index) >= EXPECTED_TOTAL:
            return index
    raise FileNotFoundError(
        f"Expected at least {EXPECTED_TOTAL} labelled images under {data_root}, but none "
        "of the conventional train/ or images/ layouts matched."
    )


def _read_label_rows(labels_path: Path) -> list[tuple[str, str]]:
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_path} has no header")
        normalized = {name.lower().strip(): name for name in reader.fieldnames}
        id_column = next(
            (normalized[name] for name in ("id", "image_id", "image", "filename", "file") if name in normalized),
            None,
        )
        breed_column = next(
            (normalized[name] for name in ("breed", "label", "class", "target") if name in normalized),
            None,
        )
        if id_column is None or breed_column is None:
            raise ValueError(
                f"{labels_path} must contain an image column (id/image_id/image/filename/file) "
                "and a class column (breed/label/class/target)"
            )
        return [(row[id_column].strip(), row[breed_column].strip()) for row in reader]


def _find_pre_split_labels(data_root: Path) -> dict[str, Path] | None:
    matches: dict[str, list[Path]] = {split: [] for split in SPLIT_SIZES}
    for path in data_root.rglob("*.csv"):
        split = PRE_SPLIT_LABEL_STEMS.get(path.stem.lower())
        if split is not None:
            matches[split].append(path)
    found_count = sum(bool(paths) for paths in matches.values())
    if found_count == 0:
        return None
    missing = [split for split, paths in matches.items() if not paths]
    duplicates = {split: paths for split, paths in matches.items() if len(paths) > 1}
    if missing or duplicates:
        raise ValueError(
            "Pre-split data requires exactly one train_labels.csv, val_labels.csv "
            f"(or validation_labels.csv), and test_labels.csv. Missing={missing}, duplicates={duplicates}"
        )
    return {split: paths[0] for split, paths in matches.items()}


def _find_split_image_directory(data_root: Path, split: str) -> Path:
    names = SPLIT_DIRECTORY_NAMES[split]
    candidates = [
        path
        for path in data_root.rglob("*")
        if path.is_dir() and path.name.lower() in names
    ]
    expected = SPLIT_SIZES[split]
    viable = []
    for directory in candidates:
        image_count = sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count >= expected:
            viable.append((directory, image_count))
    if len(viable) != 1:
        raise FileNotFoundError(
            f"Expected one '{split}' directory containing at least {expected} images under "
            f"{data_root}; found {viable}. Images may be nested in an image/ or images/ subfolder."
        )
    return viable[0][0]


def _index_images_in_directory(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            raise ValueError(f"Duplicate image id {path.stem!r} under {directory}")
        index[path.stem] = path
    return index


def _load_pre_split_records(data_root: Path) -> list[Record] | None:
    label_paths = _find_pre_split_labels(data_root)
    if label_paths is None:
        return None
    records: list[Record] = []
    seen_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        label_rows = _read_label_rows(label_paths[split])
        expected = SPLIT_SIZES[split]
        if len(label_rows) != expected:
            raise ValueError(
                f"{label_paths[split]} contains {len(label_rows)} labels; expected {expected} for {split}"
            )
        image_directory = _find_split_image_directory(data_root, split)
        image_index = _index_images_in_directory(image_directory)
        for image_id, breed in label_rows:
            lookup_id = Path(image_id).stem
            if image_id in seen_ids:
                raise ValueError(f"Image id {image_id!r} occurs in more than one split")
            seen_ids.add(image_id)
            try:
                image_path = image_index[lookup_id]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"No image matching id {image_id!r} from {label_paths[split]} "
                    f"was found under {image_directory}"
                ) from exc
            records.append(Record(image_id, breed, image_path, split))
    _validate_records(records)
    return records


def _write_manifest(manifest_path: Path, records: Iterable[Record]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "breed", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record.image_id, "breed": record.breed, "split": record.split})


def create_or_load_manifest(data_root: Path, manifest_path: Path, seed: int) -> list[Record]:
    """Use provided splits when present; otherwise create the required stratified split."""
    data_root = data_root.resolve()
    pre_split_records = _load_pre_split_records(data_root)
    if pre_split_records is not None:
        records_by_id = {record.image_id: record for record in pre_split_records}
        if manifest_path.is_file():
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != EXPECTED_TOTAL:
                raise ValueError(f"Existing manifest has {len(rows)} rows; expected {EXPECTED_TOTAL}")
            records = []
            for row in rows:
                source = records_by_id.get(row["id"])
                if source is None or source.breed != row["breed"] or source.split != row["split"]:
                    raise ValueError(
                        "Existing split manifest does not match the supplied train/val/test label files. "
                        "Remove outputs/data_artifacts/split_manifest.csv and run again."
                    )
                records.append(source)
            _validate_records(records)
            return records
        _write_manifest(manifest_path, pre_split_records)
        return pre_split_records

    labels_path = _find_labels(data_root)
    image_index = _index_images(data_root)
    label_rows = _read_label_rows(labels_path)

    if len(label_rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} labelled rows, found {len(label_rows)}")

    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != EXPECTED_TOTAL:
            raise ValueError(f"Existing manifest has {len(rows)} rows; expected {EXPECTED_TOTAL}")
        records = []
        known_labels = dict(label_rows)
        for row in rows:
            image_id = row["id"]
            if known_labels.get(image_id) != row["breed"]:
                raise ValueError("Existing split manifest does not match labels.csv")
            if image_id not in image_index:
                raise FileNotFoundError(f"Image for labelled id {image_id!r} is missing")
            records.append(Record(image_id, row["breed"], image_index[image_id], row["split"]))
        _validate_records(records)
        return records

    ids = [image_id for image_id, _ in label_rows]
    breeds = [breed for _, breed in label_rows]
    train_ids, remaining_ids, train_y, remaining_y = train_test_split(
        ids,
        breeds,
        train_size=SPLIT_SIZES["train"],
        random_state=seed,
        stratify=breeds,
    )
    validation_ids, test_ids, validation_y, test_y = train_test_split(
        remaining_ids,
        remaining_y,
        train_size=SPLIT_SIZES["validation"],
        random_state=seed,
        stratify=remaining_y,
    )
    split_by_id = {image_id: "train" for image_id in train_ids}
    split_by_id.update({image_id: "validation" for image_id in validation_ids})
    split_by_id.update({image_id: "test" for image_id in test_ids})

    records = [
        Record(image_id, breed, image_index[image_id], split_by_id[image_id])
        for image_id, breed in label_rows
    ]
    _validate_records(records)
    _write_manifest(manifest_path, records)
    return records


def _validate_records(records: Iterable[Record]) -> None:
    records = list(records)
    counts = Counter(record.split for record in records)
    if dict(counts) != SPLIT_SIZES:
        raise ValueError(f"Split counts must be {SPLIT_SIZES}; got {dict(counts)}")
    missing = [str(record.path) for record in records if not record.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing path: {missing[0]}")


class DogBreedDataset(Dataset):
    def __init__(self, records: list[Record], class_to_index: dict[str, int], transform) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.class_to_index[record.breed]


def compute_or_load_stats(
    train_records: list[Record],
    class_to_index: dict[str, int],
    stats_path: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[list[float], list[float]]:
    """Compute channel statistics only from resized training images."""
    if stats_path.is_file():
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        return payload["mean"], payload["std"]

    dataset = DogBreedDataset(
        train_records,
        class_to_index,
        transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in loader:
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.square()
    std = variance.clamp_min(0).sqrt()
    result = {"mean": mean.tolist(), "std": std.tolist(), "pixel_count": pixel_count}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["mean"], result["std"]


def build_transforms(mean: list[float], std: list[float]):
    """Return the exact training augmentation and deterministic evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((100, 100), padding=12, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, evaluation_transform


def prepare_datasets(data_root: Path, artifacts_dir: Path, seed: int, batch_size: int, num_workers: int):
    manifest_path = artifacts_dir / "split_manifest.csv"
    records = create_or_load_manifest(data_root, manifest_path, seed)
    breeds = sorted({record.breed for record in records})
    class_to_index = {breed: index for index, breed in enumerate(breeds)}
    train_records = [record for record in records if record.split == "train"]
    mean, std = compute_or_load_stats(
        train_records,
        class_to_index,
        artifacts_dir / "normalization.json",
        batch_size,
        num_workers,
    )
    train_transform, evaluation_transform = build_transforms(mean, std)
    datasets = {
        "train": DogBreedDataset(train_records, class_to_index, train_transform),
        "validation": DogBreedDataset(
            [record for record in records if record.split == "validation"],
            class_to_index,
            evaluation_transform,
        ),
        "test": DogBreedDataset(
            [record for record in records if record.split == "test"],
            class_to_index,
            evaluation_transform,
        ),
    }
    return datasets, class_to_index, mean, std
