#!/usr/bin/env python3
"""
Kaggle-ready Focus-RCNet two-stage KD experiment.

This script tests whether the paper-style follow-up protocol also helps
two-stage knowledge distillation:
    - Stage 1: load Focus-RCNet CE baseline trained with 200 epochs, batch 16
    - Stage 2: fine-tune with KD for 50 epochs, batch 16
    - KD temperature: 4
    - KD alpha: 0.5

The dataset, seed, augmentation, architecture, input size, optimizer, and
teacher model match Results/notebook2-phase1-focus-rcnet.ipynb. The 200e-bs16
protocol matches Results/focus-rcnet-baseline-ce-e200-bs16.ipynb.

Expected Kaggle inputs:
    Dataset:
        /kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized

    Required teacher checkpoint for EfficientNet-B4 KD and split indices:
        /kaggle/input/notebooks/hamzapratama/notebook1-teacher-training/
        efficientnet_b4_teacher_best.pth

    Required Stage 1 checkpoint:
        focus_rcnet_baseline_ce_e200_bs16_best.pth

Attach the teacher checkpoint before running; KD fine-tuning cannot train
without teacher weights. Attach the Stage 1 baseline checkpoint before running;
this experiment does not retrain the CE baseline.
"""

import copy
import csv
import os
import random
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import matplotlib
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Config:
    EXPERIMENT_NAME = "Focus-RCNet Two-Stage KD - 200 Epochs, Batch Size 16"
    EXPERIMENT_ID = "focus-rcnet-twostage-kd-e200-bs16-t4-a05"

    SEED = 42
    DATASET_NAME = "TrashNet"
    NUM_CLASSES = 6
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3

    IMG_SIZE = 380
    STAGE1_EPOCHS = 200
    STAGE2_EPOCHS = 50
    EPOCHS = STAGE2_EPOCHS
    BATCH_SIZE = 16
    STRICT_STAGE1_SPLIT_CHECK = True

    OPTIMIZER = "SGD"
    STAGE1_LR = 0.05
    STAGE2_LR = 0.005
    LR = STAGE2_LR
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = EPOCHS
    USE_AMP = True

    KD_TEMPERATURE = 4
    KD_ALPHA = 0.5
    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"
    STUDENT_MODEL_NAME = "Focus-RCNet"

    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    TEACHER_CHECKPOINT_PATH = (
        "/kaggle/input/notebooks/hamzapratama/notebook1-teacher-training/"
        "efficientnet_b4_teacher_best.pth"
    )
    BASELINE_CHECKPOINT_FILENAME = "focus_rcnet_baseline_ce_e200_bs16_best.pth"
    BASELINE_CHECKPOINT_CANDIDATES = (
        "/kaggle/input/notebooks/hamzapratama/focus-rcnet-baseline-ce-e200-bs16/"
        "focus_rcnet_baseline_ce_e200_bs16_best.pth",
        "/kaggle/input/focus-rcnet-baseline-ce-e200-bs16/"
        "focus_rcnet_baseline_ce_e200_bs16_best.pth",
        "/kaggle/working/focus_rcnet_baseline_ce_e200_bs16_best.pth",
    )
    OUTPUT_DIR = "/kaggle/working"

    CHECKPOINT_FILENAME = "focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth"
    HISTORY_FILENAME = "training_history_focus_rcnet_twostage_kd_e200_bs16_t4_a05.csv"
    SUMMARY_FILENAME = "summary_focus_rcnet_twostage_kd_e200_bs16_t4_a05.csv"
    CURVES_FILENAME = "training_curves_focus_rcnet_twostage_kd_e200_bs16_t4_a05.png"

    ORIGINAL_LOCAL_BASELINE_ACC = 0.8524
    FOLLOWUP_BASELINE_CE_ACC = 0.8630
    DIRECT_KD_100E_BS8_ACC = 0.8564
    TWOSTAGE_KD_100E_BS8_ACC = 0.8630
    DIRECT_KD_200E_BS16_ACC = 0.8841
    PAPER_FOCUS_RCNET_ACC = 0.9000
    PAPER_FOCUS_RCNET_KD_ACC = 0.9200


cfg = Config()


def seed_everything(seed: int) -> None:
    """Set random seeds using the same deterministic protocol as Notebook 2."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_checkpoint_path(candidates, filename: str, label: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path

    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"{label} checkpoint not found. Expected filename: {filename}\n"
        f"Checked candidates:\n{candidate_text}\n"
        "Attach the checkpoint as a Kaggle input or place it in /kaggle/working."
    )


class AlbumentationsDataset(torch.utils.data.Dataset):
    """Wrap an ImageFolder-compatible dataset with Albumentations transforms."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


class Focus(nn.Module):
    """YOLOv5-style Focus module: spatial slicing followed by CBS."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            dim=1,
        )
        return self.conv(x)


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        _, _, height, width = x.size()
        n = height * width - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_sq
            / (
                4
                * (
                    x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n
                    + self.e_lambda
                )
            )
            + 0.5
        )
        return x * torch.sigmoid(y)


class SandglassBlock(nn.Module):
    """MobileNeXt-style sandglass block used by the replication notebook."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 2,
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = max(in_channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        output = self.layers(x)
        return x + output if self.use_residual else output


class FocusRCNet(nn.Module):
    """Focus-RCNet architecture copied from Notebook 2."""

    STAGE_CONFIG = [
        (48, 4, 2, 2),
        (96, 3, 2, 2),
        (192, 2, 2, 2),
        (384, 2, 2, 2),
    ]

    def __init__(self, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.focus = Focus(in_channels=3, out_channels=24, kernel_size=1)

        stages = []
        in_channels = 24
        for out_channels, num_blocks, stride, reduction in self.STAGE_CONFIG:
            blocks = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(
                    SandglassBlock(
                        in_channels,
                        out_channels,
                        stride=block_stride,
                        reduction=reduction,
                    )
                )
                in_channels = out_channels
            blocks.append(SimAM())
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.focus(x)
        x = self.stages(x)
        x = self.conv5(x)
        return self.classifier(x)


def create_focus_rcnet(num_classes: int = 6) -> FocusRCNet:
    return FocusRCNet(num_classes=num_classes)


def make_transforms():
    """Create the exact augmentation pipeline used by Notebook 2."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = A.Compose(
        [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(
                    int(cfg.IMG_SIZE * 0.05),
                    int(cfg.IMG_SIZE * 0.2),
                ),
                hole_width_range=(
                    int(cfg.IMG_SIZE * 0.05),
                    int(cfg.IMG_SIZE * 0.2),
                ),
                fill=0,
                p=0.5,
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def load_or_create_split(full_dataset):
    """
    Reuse Notebook 1 indices when available.

    The fallback reproduces Notebook 1's stratified split for cases where the
    teacher checkpoint is not attached as a Kaggle input.
    """
    checkpoint_path = Path(cfg.TEACHER_CHECKPOINT_PATH)
    if checkpoint_path.is_file():
        teacher_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        train_indices = teacher_checkpoint["train_indices"]
        val_indices = teacher_checkpoint["val_indices"]
        print(f"Split source: teacher checkpoint ({checkpoint_path})")
        print(
            "Teacher checkpoint best validation accuracy: "
            f"{teacher_checkpoint['best_val_acc']:.4f}"
        )
        del teacher_checkpoint
    else:
        targets = [sample[1] for sample in full_dataset.samples]
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=cfg.VAL_SPLIT,
            random_state=cfg.SEED,
            stratify=targets,
        )
        print("Split source: regenerated stratified 70/30 split with seed 42")
        print(f"Teacher checkpoint not found at: {checkpoint_path}")
        print("Two-stage KD fine-tuning still requires this checkpoint for teacher weights.")

    all_indices = list(train_indices) + list(val_indices)
    assert len(all_indices) == len(full_dataset)
    assert not set(train_indices).intersection(val_indices)
    assert max(all_indices) < len(full_dataset)
    return train_indices, val_indices


def prepare_data():
    train_transform, val_transform = make_transforms()
    full_dataset = datasets.ImageFolder(root=cfg.DATASET_PATH, transform=None)
    assert full_dataset.classes == cfg.CLASS_NAMES, (
        f"Expected classes {cfg.CLASS_NAMES}, found {full_dataset.classes}"
    )
    train_indices, val_indices = load_or_create_split(full_dataset)

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    train_dataset = AlbumentationsDataset(train_subset, transform=train_transform)
    val_dataset = AlbumentationsDataset(val_subset, transform=val_transform)

    def worker_init_fn(worker_id):
        np.random.seed(cfg.SEED + worker_id)
        random.seed(cfg.SEED + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    targets = [sample[1] for sample in full_dataset.samples]
    train_dist = Counter(targets[index] for index in train_indices)
    val_dist = Counter(targets[index] for index in val_indices)

    print(f"Dataset path: {cfg.DATASET_PATH}")
    print(f"Total images: {len(full_dataset)}")
    print(f"Train images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("\nClass distribution:")
    print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 42)
    for class_idx, class_name in enumerate(full_dataset.classes):
        train_count = train_dist[class_idx]
        val_count = val_dist[class_idx]
        print(
            f"{class_name:<15} {train_count:>8} "
            f"{val_count:>8} {train_count + val_count:>8}"
        )
    print("-" * 42)
    return train_loader, val_loader, train_indices, val_indices


def verify_architecture():
    model = create_focus_rcnet(cfg.NUM_CLASSES)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    sample = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    output = model(sample)
    assert total_params == 520_630, f"Unexpected parameter count: {total_params}"
    assert output.shape == (1, cfg.NUM_CLASSES), f"Unexpected shape: {output.shape}"
    print(f"Focus-RCNet parameters: {total_params:,}")
    print(f"Input shape: {list(sample.shape)}")
    print(f"Output shape: {list(output.shape)}")


def create_teacher_model():
    return timm.create_model(
        cfg.TEACHER_TIMM_NAME,
        pretrained=False,
        num_classes=cfg.NUM_CLASSES,
    )


def load_teacher_model(device):
    checkpoint_path = resolve_checkpoint_path(
        (cfg.TEACHER_CHECKPOINT_PATH,),
        Path(cfg.TEACHER_CHECKPOINT_PATH).name,
        "Teacher",
    )

    teacher_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_model = create_teacher_model()
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    teacher_params = sum(parameter.numel() for parameter in teacher_model.parameters())
    print(f"Teacher model loaded: {cfg.TEACHER_MODEL_NAME}")
    print(f"Teacher timm name: {cfg.TEACHER_TIMM_NAME}")
    print(f"Teacher checkpoint: {checkpoint_path}")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Teacher best epoch: {teacher_checkpoint.get('best_epoch', 'n/a')}")
    print(f"Teacher best val accuracy: {teacher_checkpoint.get('best_val_acc', float('nan')):.4f}")
    print("Teacher mode: eval, frozen")
    del teacher_checkpoint
    return teacher_model


def _same_indices(left, right) -> bool:
    return list(left) == list(right)


def load_stage1_model(device, train_indices, val_indices):
    checkpoint_path = resolve_checkpoint_path(
        cfg.BASELINE_CHECKPOINT_CANDIDATES,
        cfg.BASELINE_CHECKPOINT_FILENAME,
        "Stage 1 baseline",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "train_indices" in checkpoint and "val_indices" in checkpoint:
        train_match = _same_indices(checkpoint["train_indices"], train_indices)
        val_match = _same_indices(checkpoint["val_indices"], val_indices)
        if cfg.STRICT_STAGE1_SPLIT_CHECK and not (train_match and val_match):
            raise ValueError(
                "Stage 1 checkpoint split does not match the current split. "
                "Attach the matching focus_rcnet_baseline_ce_e200_bs16 checkpoint "
                "or disable STRICT_STAGE1_SPLIT_CHECK after documenting the mismatch."
            )
        if not (train_match and val_match):
            print("Warning: Stage 1 checkpoint split differs from the current split.")

    model = create_focus_rcnet(cfg.NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("best_epoch", "n/a")
    best_val_acc = checkpoint.get("best_val_acc", float("nan"))
    print("Stage 1 baseline checkpoint loaded")
    print(f"Stage 1 checkpoint: {checkpoint_path}")
    print(f"Stage 1 best epoch: {best_epoch}")
    print(f"Stage 1 best val accuracy: {best_val_acc:.4f}")
    del checkpoint
    return model, checkpoint_path


def train_one_epoch_kd(student, teacher, loader, optimizer, scaler, device):
    student.train()
    teacher.eval()
    running_loss_total = 0.0
    running_loss_soft = 0.0
    running_loss_hard = 0.0
    correct = 0
    total = 0
    ce_criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=cfg.USE_AMP):
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

        with autocast(enabled=False):
            temperature = cfg.KD_TEMPERATURE
            alpha = cfg.KD_ALPHA
            student_log_soft = F.log_softmax(student_logits.float() / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits.float() / temperature, dim=1)
            loss_soft = F.kl_div(
                student_log_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (temperature ** 2)
            loss_hard = ce_criterion(student_logits.float(), labels)
            loss = alpha * loss_soft + (1.0 - alpha) * loss_hard

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite KD loss detected: "
                    f"total={loss.item()}, soft={loss_soft.item()}, hard={loss_hard.item()}"
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss_total += loss.item() * images.size(0)
        running_loss_soft += loss_soft.item() * images.size(0)
        running_loss_hard += loss_hard.item() * images.size(0)
        predictions = student_logits.max(1).indices
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

    return (
        running_loss_total / total,
        correct / total,
        running_loss_soft / total,
        running_loss_hard / total,
    )


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=cfg.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        predictions = outputs.max(1).indices
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()
    return running_loss / total, correct / total


def run_experiment(model, teacher_model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_MAX)
    scaler = GradScaler(enabled=cfg.USE_AMP)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "train_loss_soft": [],
        "train_loss_hard": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    total_start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, train_loss_soft, train_loss_hard = train_one_epoch_kd(
            model,
            teacher_model,
            train_loader,
            optimizer,
            scaler,
            device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        history["train_loss_soft"].append(train_loss_soft)
        history["train_loss_hard"].append(train_loss_hard)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        best_marker = " BEST" if is_best else ""
        print(
            f"Epoch [{epoch:3d}/{cfg.EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"(Soft: {train_loss_soft:.4f}, Hard: {train_loss_hard:.4f}) "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {epoch_time:.1f}s{best_marker}",
            flush=True,
        )

    total_time = time.time() - total_start
    return history, best_val_acc, best_epoch, best_model_state, total_time


def save_history(history, output_dir: Path):
    path = output_dir / cfg.HISTORY_FILENAME
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss_total",
                "train_loss_soft",
                "train_loss_hard",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
                "epoch_time",
            ]
        )
        for index in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][index],
                    f"{history['train_loss'][index]:.6f}",
                    f"{history['train_loss_soft'][index]:.6f}",
                    f"{history['train_loss_hard'][index]:.6f}",
                    f"{history['train_acc'][index]:.6f}",
                    f"{history['val_loss'][index]:.6f}",
                    f"{history['val_acc'][index]:.6f}",
                    f"{history['lr'][index]:.8f}",
                    f"{history['epoch_time'][index]:.2f}",
                ]
            )
    return path


def save_checkpoint(
    model_state,
    history,
    best_epoch,
    best_val_acc,
    train_indices,
    val_indices,
    stage1_checkpoint_path,
    output_dir: Path,
):
    path = output_dir / cfg.CHECKPOINT_FILENAME
    checkpoint = {
        "model_state_dict": model_state,
        "experiment_id": cfg.EXPERIMENT_ID,
        "experiment_name": cfg.EXPERIMENT_NAME,
        "model_name": f"{cfg.STUDENT_MODEL_NAME}-TwoStage-KD",
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "config": {
            "img_size": cfg.IMG_SIZE,
            "stage1_epochs": cfg.STAGE1_EPOCHS,
            "stage1_lr": cfg.STAGE1_LR,
            "stage2_epochs": cfg.STAGE2_EPOCHS,
            "stage2_lr": cfg.STAGE2_LR,
            "epochs": cfg.STAGE2_EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.STAGE2_LR,
            "momentum": cfg.MOMENTUM,
            "weight_decay": cfg.WEIGHT_DECAY,
            "optimizer": cfg.OPTIMIZER,
            "scheduler": cfg.SCHEDULER,
            "t_max": cfg.T_MAX,
            "seed": cfg.SEED,
            "train_split": cfg.TRAIN_SPLIT,
            "val_split": cfg.VAL_SPLIT,
            "use_amp": cfg.USE_AMP,
            "training_mode": "two_stage_kd",
            "knowledge_distillation": True,
            "kd_temperature": cfg.KD_TEMPERATURE,
            "kd_alpha": cfg.KD_ALPHA,
            "teacher_model": cfg.TEACHER_MODEL_NAME,
            "teacher_timm_name": cfg.TEACHER_TIMM_NAME,
            "teacher_checkpoint_path": cfg.TEACHER_CHECKPOINT_PATH,
            "stage1_checkpoint_path": str(stage1_checkpoint_path),
        },
        "dataset_name": cfg.DATASET_NAME,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(checkpoint, path)
    return path


def save_summary(best_val_acc, best_epoch, total_time, output_dir: Path):
    path = output_dir / cfg.SUMMARY_FILENAME
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["experiment", cfg.EXPERIMENT_NAME])
        writer.writerow(["best_val_accuracy", f"{best_val_acc:.6f}"])
        writer.writerow(["best_val_accuracy_percent", f"{best_val_acc * 100:.2f}"])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["total_time_minutes", f"{total_time / 60:.2f}"])
        writer.writerow(["stage1_epochs", cfg.STAGE1_EPOCHS])
        writer.writerow(["stage1_lr", cfg.STAGE1_LR])
        writer.writerow(["stage2_epochs", cfg.STAGE2_EPOCHS])
        writer.writerow(["stage2_lr", cfg.STAGE2_LR])
        writer.writerow(["epochs", cfg.STAGE2_EPOCHS])
        writer.writerow(["batch_size", cfg.BATCH_SIZE])
        writer.writerow(["kd_temperature", cfg.KD_TEMPERATURE])
        writer.writerow(["kd_alpha", cfg.KD_ALPHA])
        writer.writerow(["teacher_model", cfg.TEACHER_MODEL_NAME])
        writer.writerow(["original_local_baseline_accuracy", cfg.ORIGINAL_LOCAL_BASELINE_ACC])
        writer.writerow(["followup_baseline_ce_accuracy", cfg.FOLLOWUP_BASELINE_CE_ACC])
        writer.writerow(["direct_kd_100e_bs8_accuracy", cfg.DIRECT_KD_100E_BS8_ACC])
        writer.writerow(["twostage_kd_100e_bs8_accuracy", cfg.TWOSTAGE_KD_100E_BS8_ACC])
        writer.writerow(["direct_kd_200e_bs16_accuracy", cfg.DIRECT_KD_200E_BS16_ACC])
        writer.writerow(["paper_focus_rcnet_accuracy", cfg.PAPER_FOCUS_RCNET_ACC])
        writer.writerow(["paper_focus_rcnet_kd_accuracy", cfg.PAPER_FOCUS_RCNET_KD_ACC])
        writer.writerow(
            [
                "delta_vs_original_local_baseline_pp",
                f"{(best_val_acc - cfg.ORIGINAL_LOCAL_BASELINE_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_followup_baseline_ce_pp",
                f"{(best_val_acc - cfg.FOLLOWUP_BASELINE_CE_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_direct_kd_100e_bs8_pp",
                f"{(best_val_acc - cfg.DIRECT_KD_100E_BS8_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_twostage_kd_100e_bs8_pp",
                f"{(best_val_acc - cfg.TWOSTAGE_KD_100E_BS8_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_direct_kd_200e_bs16_pp",
                f"{(best_val_acc - cfg.DIRECT_KD_200E_BS16_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_paper_focus_rcnet_baseline_pp",
                f"{(best_val_acc - cfg.PAPER_FOCUS_RCNET_ACC) * 100:+.2f}",
            ]
        )
        writer.writerow(
            [
                "delta_vs_paper_focus_rcnet_kd_pp",
                f"{(best_val_acc - cfg.PAPER_FOCUS_RCNET_KD_ACC) * 100:+.2f}",
            ]
        )
    return path


def save_curves(history, output_dir: Path):
    path = output_dir / cfg.CURVES_FILENAME
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train KD total loss")
    axes[0].plot(history["epoch"], history["train_loss_soft"], label="Train soft loss")
    axes[0].plot(history["epoch"], history["train_loss_hard"], label="Train hard CE loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation CE loss")
    axes[0].set_title("Focus-RCNet Two-Stage KD Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_acc"], label="Validation accuracy")
    axes[1].axhline(
        cfg.FOLLOWUP_BASELINE_CE_ACC,
        color="tab:green",
        linestyle="--",
        label="200e-bs16 CE baseline (86.30%)",
    )
    axes[1].axhline(
        cfg.DIRECT_KD_100E_BS8_ACC,
        color="tab:purple",
        linestyle="--",
        label="100e-bs8 direct KD (85.64%)",
    )
    axes[1].axhline(
        cfg.TWOSTAGE_KD_100E_BS8_ACC,
        color="tab:cyan",
        linestyle="--",
        label="100e-bs8 two-stage KD (86.30%)",
    )
    axes[1].axhline(
        cfg.DIRECT_KD_200E_BS16_ACC,
        color="tab:brown",
        linestyle="--",
        label="200e-bs16 direct KD (88.41%)",
    )
    axes[1].axhline(
        cfg.ORIGINAL_LOCAL_BASELINE_ACC,
        color="tab:orange",
        linestyle="--",
        label="Original local baseline (85.24%)",
    )
    axes[1].axhline(
        cfg.PAPER_FOCUS_RCNET_ACC,
        color="tab:red",
        linestyle="--",
        label="Paper Focus-RCNet (90.00%)",
    )
    axes[1].set_title("Focus-RCNet Two-Stage KD Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def print_configuration(device):
    print("=" * 76)
    print(cfg.EXPERIMENT_NAME)
    print("=" * 76)
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Dataset      : {cfg.DATASET_NAME}")
    print(f"Image size   : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print(f"Stage 1      : load CE baseline ({cfg.STAGE1_EPOCHS} epochs, lr={cfg.STAGE1_LR})")
    print(f"Stage 2      : KD fine-tuning ({cfg.STAGE2_EPOCHS} epochs, lr={cfg.STAGE2_LR})")
    print(f"Batch size   : {cfg.BATCH_SIZE}")
    print(f"Optimizer    : {cfg.OPTIMIZER}")
    print(f"Learning rate: {cfg.STAGE2_LR}")
    print(f"Momentum     : {cfg.MOMENTUM}")
    print(f"Weight decay : {cfg.WEIGHT_DECAY}")
    print(f"Scheduler    : {cfg.SCHEDULER} (T_max={cfg.T_MAX})")
    print("Loss         : alpha * KLDiv(T) * T^2 + (1-alpha) * CrossEntropyLoss")
    print(f"KD           : enabled, T={cfg.KD_TEMPERATURE}, alpha={cfg.KD_ALPHA}")
    print(f"Teacher      : {cfg.TEACHER_MODEL_NAME} ({cfg.TEACHER_TIMM_NAME})")
    print(f"AMP          : {cfg.USE_AMP}")
    print(f"Seed         : {cfg.SEED}")
    print("=" * 76)


def main():
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_configuration(device)

    teacher_model = load_teacher_model(device)
    train_loader, val_loader, train_indices, val_indices = prepare_data()
    verify_architecture()

    model, stage1_checkpoint_path = load_stage1_model(device, train_indices, val_indices)
    print("\nStarting two-stage KD fine-tuning...", flush=True)
    history, best_val_acc, best_epoch, best_model_state, total_time = run_experiment(
        model,
        teacher_model,
        train_loader,
        val_loader,
        device,
    )

    checkpoint_path = save_checkpoint(
        best_model_state,
        history,
        best_epoch,
        best_val_acc,
        train_indices,
        val_indices,
        stage1_checkpoint_path,
        output_dir,
    )
    history_path = save_history(history, output_dir)
    summary_path = save_summary(best_val_acc, best_epoch, total_time, output_dir)
    curves_path = save_curves(history, output_dir)

    print("\n" + "=" * 76)
    print("Experiment complete")
    print("=" * 76)
    print(f"Best validation accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Best epoch               : {best_epoch}")
    print(f"Total time               : {total_time / 60:.1f} minutes")
    print(
        "Delta vs baseline 100e-bs8 CE : "
        f"{(best_val_acc - cfg.ORIGINAL_LOCAL_BASELINE_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs baseline 200e-bs16 CE: "
        f"{(best_val_acc - cfg.FOLLOWUP_BASELINE_CE_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs direct KD 100e-bs8   : "
        f"{(best_val_acc - cfg.DIRECT_KD_100E_BS8_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs two-stage KD 100e-bs8: "
        f"{(best_val_acc - cfg.TWOSTAGE_KD_100E_BS8_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs direct KD 200e-bs16  : "
        f"{(best_val_acc - cfg.DIRECT_KD_200E_BS16_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs paper baseline       : "
        f"{(best_val_acc - cfg.PAPER_FOCUS_RCNET_ACC) * 100:+.2f} pp"
    )
    print(
        "Delta vs paper KD             : "
        f"{(best_val_acc - cfg.PAPER_FOCUS_RCNET_KD_ACC) * 100:+.2f} pp"
    )
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"History CSV              : {history_path}")
    print(f"Summary CSV              : {summary_path}")
    print(f"Training curves          : {curves_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
