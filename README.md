# AI6103 Dog Breed Assignment

This repository implements the experiments specified in `AI6103_Assignment.pdf` without fabricating any results. It uses torchvision's EfficientNetV2-S with random initialization, SGD with momentum 0.9, 100 x 100 images, the required augmentation, and a fixed stratified 7,155/1,533/1,534 split. The test split is inaccessible to the training command and is evaluated only by the separate final-evaluation script.

## Kaggle notebook

For a notebook-based workflow, upload this project and the image dataset as Kaggle inputs, enable a GPU, and open `notebooks/ai6103_kaggle.ipynb`. Its first code cell selects a single experiment (`smoke`, `lr_sweep`, `constant`, `cosine`, either weight-decay run, or `mixup`). It discovers the attached inputs, writes checkpoints and plots to `/kaggle/working/outputs`, displays results inline, and can recover a prior notebook output for `--resume`.

Start with `EXPERIMENT = "smoke"`. After that succeeds, use a fresh saved notebook version for each long 300-epoch configuration. Do not execute the final test snippet until all hyperparameters have been selected from validation results.

## 1. Environment and data

Use Python 3.10 or newer. A CUDA-capable GPU is strongly recommended because the full suite includes multiple 300-epoch EfficientNetV2-S runs.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/download_data.py
```

The downloader uses the Google Drive file from the assignment. Alternatively, extract the supplied data manually under `data/`. The loader accepts either `labels.csv` plus `train/` or `images/`, or a pre-split layout like the following (the `image/` level may also be named `images/` or omitted):

```text
data/
├── train_labels.csv
├── val_labels.csv
├── test_labels.csv
├── train/image/*.jpg
├── val/image/*.jpg
└── test/image/*.jpg
```

For pre-split data, the supplied split is preserved rather than generated again. Label CSVs may use `id`, `image_id`, `image`, `filename`, or `file` for the image column and `breed`, `label`, `class`, or `target` for the class column. On the first run the code creates:

- `outputs/data_artifacts/split_manifest.csv`: the persistent stratified split.
- `outputs/data_artifacts/normalization.json`: RGB mean and standard deviation computed from resized training images only.

Keep both files unchanged so every experiment uses exactly the same split and normalization.

## 2. Learning-rate experiments (15 epochs)

```powershell
python scripts/train.py --lr 0.5   --epochs 15 --output-dir outputs/lr/lr_0.5
python scripts/train.py --lr 0.005 --epochs 15 --output-dir outputs/lr/lr_0.005
python scripts/train.py --lr 0.001 --epochs 15 --output-dir outputs/lr/lr_0.001
python scripts/compare_runs.py --run outputs/lr/lr_0.5 --label "lr=0.5" --run outputs/lr/lr_0.005 --label "lr=0.005" --run outputs/lr/lr_0.001 --label "lr=0.001" --output outputs/lr/comparison.png --title "Initial learning rate"
python scripts/summarize_results.py
```

Use the learning rate with the smallest **final training loss** in the next sections, exactly as requested. Set it as `$BEST_LR` below; do not choose it using test performance.

```powershell
$BEST_LR = 0.005  # Replace after inspecting your actual learning-rate results.
```

## 3. Constant versus cosine learning rate (300 epochs)

```powershell
python scripts/train.py --lr $BEST_LR --epochs 300 --scheduler none   --output-dir outputs/schedule/constant
python scripts/train.py --lr $BEST_LR --epochs 300 --scheduler cosine --output-dir outputs/schedule/cosine
python scripts/compare_runs.py --run outputs/schedule/constant --label "Constant" --run outputs/schedule/cosine --label "Cosine" --output outputs/schedule/comparison.png --title "Learning-rate schedule"
```

The cosine run uses PyTorch's `CosineAnnealingLR` with `T_max=300` and `eta_min=0`. For epoch/update index `t`, this corresponds to

`eta_t = 0.5 * eta_0 * (1 + cos(pi * t / T))`, for `0 <= t <= T`.

## 4. Weight decay (300 epochs)

```powershell
python scripts/train.py --lr $BEST_LR --epochs 300 --scheduler cosine --weight-decay 0.0005 --output-dir outputs/weight_decay/wd_5e-4
python scripts/train.py --lr $BEST_LR --epochs 300 --scheduler cosine --weight-decay 0.0001 --output-dir outputs/weight_decay/wd_1e-4
python scripts/compare_runs.py --run outputs/schedule/cosine --label "No weight decay" --run outputs/weight_decay/wd_5e-4 --label "weight decay=5e-4" --run outputs/weight_decay/wd_1e-4 --label "weight decay=1e-4" --output outputs/weight_decay/comparison.png --title "Weight decay"
```

Choose the best setup using validation performance only. Keep the cosine run with zero weight decay in the comparison as a baseline. Set `$BEST_WD` to the chosen value.

```powershell
$BEST_WD = 0.0001  # Replace after inspecting your actual validation results.
```

## 5. Mixup and Beta density (300 epochs)

```powershell
python scripts/plot_beta.py --alpha 0.2
python scripts/train.py --lr $BEST_LR --epochs 300 --scheduler cosine --weight-decay $BEST_WD --mixup-alpha 0.2 --output-dir outputs/mixup/alpha_0.2
python scripts/summarize_results.py
```

For mixup, training loss and accuracy are coefficient-weighted against both mixed labels. Validation remains unmodified and is therefore directly comparable across runs.

## 6. Final test evaluation - run only once

After all choices are frozen using validation results, evaluate the selected checkpoint. The best checkpoint means the epoch with minimum validation loss; `last_checkpoint.pt` is also retained when the report specifically needs final-epoch metrics.

```powershell
python scripts/evaluate_test.py --checkpoint outputs/mixup/alpha_0.2/best_checkpoint.pt
```

This writes `outputs/final_test.json`. Do not rerun the test to tune any choice.

## Outputs and recovery

Every experiment directory contains `metrics.csv`, `training_curves.png`, `summary.json`, `last_checkpoint.pt`, and `best_checkpoint.pt`. Add `--resume` with the same arguments and output directory to continue an interrupted run. The report must be written in the required AAAI double-column format, stay within six pages excluding references, describe the numerical observations in your own words, and **must not answer Section 7 (“The Takeaway Message”)**.
