#!/usr/bin/env python
# =============================================================================
# Notebook 3 - Phase 2: EfficientNet-Lite0 Experiments + Final Evaluation
# =============================================================================
# Experiment 5: Train EfficientNet-Lite0 (Baseline)
# Experiment 6: KD From Scratch (EfficientNet-B4 -> EfficientNet-Lite0)
# Experiment 7: Two-Stage KD (load Exp 5 -> fine-tune with KD)
# Final evaluation: compare all 7 core experiments
#
# Controlled-comparison protocol:
# - IMG_SIZE: 380x380
# - LR: 0.05
# - EPOCHS: 100
# - BATCH_SIZE: 8
# - Student pretrained weights: disabled
# - Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - KD: T=4, alpha=0.5
# - Two-Stage: Stage2 epochs=50, lr=0.005
#
# The optional EfficientNet-Lite0 Native Profile is intentionally excluded.
# Run that as a separate follow-up only if the controlled results need it.
# =============================================================================

# %% [markdown]
# # Notebook 3 - Phase 2: EfficientNet-Lite0 + Final Evaluation
#
# This notebook runs Experiments 5, 6, and 7 using the same controlled protocol
# as Experiments 2, 3, and 4. It then evaluates all seven core checkpoints.

# %% [markdown]
# ## 0. Setup

# %%
# ============================================================
# 0.1 - Imports
# ============================================================
import os
import random
import time
import copy
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("All imports successful")

# %%
# ============================================================
# 0.2 - Configuration
# ============================================================

class Config:
    """Centralized controlled-comparison configuration for Phase 2."""

    # ----- Experiment Metadata -----
    NOTEBOOK_ID = 3
    PHASE = 2

    # ----- Compute Environment -----
    PLATFORM = "Kaggle Notebooks"
    FRAMEWORK = "PyTorch"
    USE_AMP = True

    # ----- Reproducibility -----
    SEED = 42

    # ----- Dataset -----
    DATASET_NAME = "TrashNet"
    NUM_CLASSES = 6
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3

    # ----- Controlled Input -----
    IMG_SIZE = 380

    # ----- Training (Experiments 5 & 6) -----
    EPOCHS = 100
    BATCH_SIZE = 8

    # ----- Optimization -----
    OPTIMIZER = "SGD"
    LR = 0.05
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # ----- Scheduler -----
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = EPOCHS

    # ----- Knowledge Distillation -----
    KD_TEMPERATURE = 4
    KD_ALPHA = 0.5

    # ----- Two-Stage KD (Experiment 7, Stage 2) -----
    STAGE2_EPOCHS = 50
    STAGE2_LR = 0.005
    STAGE2_FREEZE_LAYERS = False
    STAGE2_T_MAX = STAGE2_EPOCHS

    # ----- Models -----
    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"
    STUDENT_MODEL_NAME = "EfficientNet-Lite0"
    STUDENT_PRETRAINED = False
    # timm used tf_efficientnet_lite0 historically and may expose a tagged
    # variant in newer releases. Resolve the available registry name at runtime.
    STUDENT_TIMM_CANDIDATES = (
        "tf_efficientnet_lite0",
        "tf_efficientnet_lite0.in1k",
        "efficientnet_lite0",
    )

    # ----- Notebook 3 Output Artifacts -----
    EXP5_CHECKPOINT = "efficientnet_lite0_baseline_best.pth"
    EXP6_CHECKPOINT = "efficientnet_lite0_kd_scratch_best.pth"
    EXP7_CHECKPOINT = "efficientnet_lite0_kd_twostage_best.pth"
    EXP5_HISTORY = "training_history_exp5.csv"
    EXP6_HISTORY = "training_history_exp6.csv"
    EXP7_HISTORY = "training_history_exp7.csv"
    FINAL_COMPARISON_CSV = "final_comparison_table.csv"

    # ----- Earlier Checkpoint Filenames -----
    TEACHER_CHECKPOINT = "efficientnet_b4_teacher_best.pth"
    EXP2_CHECKPOINT = "focus_rcnet_baseline_best.pth"
    EXP3_CHECKPOINT = "focus_rcnet_kd_scratch_best.pth"
    EXP4_CHECKPOINT = "focus_rcnet_kd_twostage_best.pth"

    # ----- Paths -----
    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    TEACHER_INPUT_DIR = "/kaggle/input/notebooks/hamzapratama/notebook1-teacher-training"
    PHASE1_INPUT_DIR = "/kaggle/input/notebooks/hamzapratama/notebook2-phase1-focus-rcnet"
    OUTPUT_DIR = "/kaggle/working/"

    # ----- Efficiency Evaluation -----
    INFERENCE_WARMUP_RUNS = 10
    INFERENCE_MEASURE_RUNS = 50


cfg = Config()

print("=" * 72)
print("  Notebook 3 - Phase 2: EfficientNet-Lite0 Controlled Comparison")
print("=" * 72)
print(f"  Teacher        : {cfg.TEACHER_MODEL_NAME}")
print(f"  Student        : {cfg.STUDENT_MODEL_NAME}")
print(f"  Student weights: pretrained={cfg.STUDENT_PRETRAINED}")
print(f"  Dataset        : {cfg.DATASET_NAME} ({cfg.NUM_CLASSES} classes)")
print(f"  Image Size     : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"  Epochs         : {cfg.EPOCHS} (Stage2: {cfg.STAGE2_EPOCHS})")
print(f"  Batch Size     : {cfg.BATCH_SIZE}")
print(f"  Optimizer      : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM})")
print(f"  Scheduler      : {cfg.SCHEDULER}")
print(f"  KD             : T={cfg.KD_TEMPERATURE}, alpha={cfg.KD_ALPHA}")
print(f"  AMP            : {cfg.USE_AMP}")
print(f"  Seed           : {cfg.SEED}")
print("=" * 72)

# %%
# ============================================================
# 0.3 - Seed, Device, and Kaggle Input Resolver
# ============================================================

def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_input_checkpoint(filename: str, preferred_dirs=()) -> str:
    """Resolve a Kaggle input artifact while tolerating notebook slug changes."""
    for directory in preferred_dirs:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return str(candidate)

    input_root = Path("/kaggle/input")
    matches = sorted(input_root.rglob(filename)) if input_root.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under /kaggle/input. "
            "Attach the required Notebook 1 and Notebook 2 outputs as Kaggle inputs."
        )

    if len(matches) > 1:
        print(f"Warning: multiple matches found for {filename}; using {matches[0]}")
    return str(matches[0])


seed_everything(cfg.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

teacher_checkpoint_path = resolve_input_checkpoint(
    cfg.TEACHER_CHECKPOINT,
    preferred_dirs=(cfg.TEACHER_INPUT_DIR,),
)
print(f"Teacher checkpoint: {teacher_checkpoint_path}")

# %% [markdown]
# ## 1. Dataset & DataLoader

# %%
# ============================================================
# 1.1 - Data Augmentation
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(int(cfg.IMG_SIZE * 0.05), int(cfg.IMG_SIZE * 0.2)),
        hole_width_range=(int(cfg.IMG_SIZE * 0.05), int(cfg.IMG_SIZE * 0.2)),
        fill=0,
        p=0.5,
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

print("Augmentation pipelines created")
print("   Train: Resize -> HFlip -> VFlip -> BrightnessContrast -> CoarseDropout -> Normalize")
print("   Val  : Resize -> Normalize")

# %%
# ============================================================
# 1.2 - Albumentations Dataset Wrapper
# ============================================================

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

# %%
# ============================================================
# 1.3 - Load Dataset and Reuse Exact Teacher Split
# ============================================================

teacher_ckpt = torch.load(teacher_checkpoint_path, map_location="cpu")
train_indices = teacher_ckpt["train_indices"]
val_indices = teacher_ckpt["val_indices"]
teacher_val_acc = teacher_ckpt["best_val_acc"]
print("Loaded split indices from teacher checkpoint")
print(f"   Teacher best val acc: {teacher_val_acc:.4f}")

full_dataset = datasets.ImageFolder(root=cfg.DATASET_PATH, transform=None)
print(f"Dataset loaded from: {cfg.DATASET_PATH}")
print(f"   Total images : {len(full_dataset)}")
print(f"   Classes found: {full_dataset.classes}")
assert full_dataset.classes == cfg.CLASS_NAMES, (
    f"Expected classes {cfg.CLASS_NAMES}, found {full_dataset.classes}"
)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)
train_dataset = AlbumentationsDataset(train_subset, transform=train_transform)
val_dataset = AlbumentationsDataset(val_subset, transform=val_transform)

print(f"Split (from teacher checkpoint, seed={cfg.SEED}):")
print(f"   Train: {len(train_indices)} images ({len(train_indices) / len(full_dataset) * 100:.1f}%)")
print(f"   Val  : {len(val_indices)} images ({len(val_indices) / len(full_dataset) * 100:.1f}%)")

# %%
# ============================================================
# 1.4 - DataLoaders
# ============================================================

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

print("DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches  : {len(val_loader)}")

# %%
# ============================================================
# 1.5 - Verify Class Distribution
# ============================================================

targets = [sample[1] for sample in full_dataset.samples]
train_dist = Counter(targets[i] for i in train_indices)
val_dist = Counter(targets[i] for i in val_indices)

print("Class Distribution:")
print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
print("-" * 42)
for idx, class_name in enumerate(full_dataset.classes):
    train_count = train_dist[idx]
    val_count = val_dist[idx]
    print(f"{class_name:<15} {train_count:>8} {val_count:>8} {train_count + val_count:>8}")
print("-" * 42)
print(f"{'Total':<15} {len(train_indices):>8} {len(val_indices):>8} {len(full_dataset):>8}")

# %% [markdown]
# ## 2. Teacher and Student Models

# %%
# ============================================================
# 2.1 - Load Frozen EfficientNet-B4 Teacher
# ============================================================

def create_teacher_model(model_name: str, num_classes: int, pretrained: bool = False):
    """Create the EfficientNet-B4 teacher structure using timm."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


teacher_model = create_teacher_model(cfg.TEACHER_TIMM_NAME, cfg.NUM_CLASSES, pretrained=False)
teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
teacher_model = teacher_model.to(device)
teacher_model.eval()
for parameter in teacher_model.parameters():
    parameter.requires_grad = False

teacher_params = sum(parameter.numel() for parameter in teacher_model.parameters())
print(f"Teacher loaded: {cfg.TEACHER_MODEL_NAME}")
print(f"   Parameters  : {teacher_params:,}")
print(f"   Best epoch  : {teacher_ckpt['best_epoch']}")
print(f"   Best val acc: {teacher_ckpt['best_val_acc']:.4f}")
print("   Mode        : eval (frozen)")
del teacher_ckpt
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
# ============================================================
# 2.2 - Resolve and Create EfficientNet-Lite0 Student
# ============================================================

def resolve_lite0_timm_name() -> str:
    """Return the EfficientNet-Lite0 registry name supported by installed timm."""
    errors = []
    for model_name in cfg.STUDENT_TIMM_CANDIDATES:
        try:
            test_model = timm.create_model(model_name, pretrained=False, num_classes=cfg.NUM_CLASSES)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            continue
        del test_model
        return model_name
    details = "\n".join(errors)
    raise RuntimeError(f"Unable to create EfficientNet-Lite0 with installed timm:\n{details}")


student_timm_name = resolve_lite0_timm_name()
print(f"Resolved student timm registry name: {student_timm_name}")


def create_lite0_model(num_classes: int = 6, pretrained: bool = False):
    """Create a fresh EfficientNet-Lite0 student model."""
    return timm.create_model(student_timm_name, pretrained=pretrained, num_classes=num_classes)


_test_lite0 = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED)
_lite0_params = sum(parameter.numel() for parameter in _test_lite0.parameters())
_test_input = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
_test_output = _test_lite0(_test_input)
print("EfficientNet-Lite0 architecture verified")
print(f"   timm registry name: {student_timm_name}")
print(f"   Parameters        : {_lite0_params:,}")
print(f"   Pretrained        : {cfg.STUDENT_PRETRAINED}")
print(f"   Input shape       : {list(_test_input.shape)}")
print(f"   Output shape      : {list(_test_output.shape)}")
assert _test_output.shape == (1, cfg.NUM_CLASSES), (
    f"Expected output shape (1, {cfg.NUM_CLASSES}), got {_test_output.shape}"
)
del _test_lite0, _test_input, _test_output

# %% [markdown]
# ## 3. Corrected Focus-RCNet Definition for Final Evaluation
#
# This copy matches Notebook 2 and is used only to load Phase 1 checkpoints.

# %%
# ============================================================
# 3.1 - Focus, SimAM, Sandglass, and Focus-RCNet
# ============================================================

class Focus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        sliced_channels = in_channels * 4
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(sliced_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1)
        return self.conv(x)


class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        _, _, h, w = x.size()
        n = h * w - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class SandglassBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 2):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = max(in_channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        output = self.layers(x)
        return x + output if self.use_residual else output


class FocusRCNet(nn.Module):
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
                blocks.append(SandglassBlock(in_channels, out_channels, stride=block_stride, reduction=reduction))
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
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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


def create_focus_rcnet(num_classes: int = 6):
    """Create the corrected Focus-RCNet structure used by Notebook 2."""
    return FocusRCNet(num_classes=num_classes)


_test_focus = create_focus_rcnet(cfg.NUM_CLASSES)
_focus_params = sum(parameter.numel() for parameter in _test_focus.parameters())
print(f"Focus-RCNet evaluation structure verified: {_focus_params:,} parameters")
assert _focus_params == 520630, f"Expected corrected Focus-RCNet params 520630, got {_focus_params}"
del _test_focus

# %% [markdown]
# ## 4. Training Utilities

# %%
# ============================================================
# 4.1 - CE, KD, Validation, Runner, and Save Helpers
# ============================================================

def train_one_epoch_ce(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train one epoch using standard cross entropy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        predicted = outputs.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_one_epoch_kd(student, teacher, loader, optimizer, scaler, device, use_amp, temperature, alpha):
    """Train one epoch using CE plus temperature-scaled KL distillation."""
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

        with autocast(enabled=use_amp):
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

        # Keep KD soft-target operations in FP32 for numerical stability.
        with autocast(enabled=False):
            student_log_soft = F.log_softmax(student_logits.float() / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits.float() / temperature, dim=1)
            loss_soft = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
            loss_hard = ce_criterion(student_logits.float(), labels)
            loss = alpha * loss_soft + (1.0 - alpha) * loss_hard

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite KD loss: total={loss.item()}, soft={loss_soft.item()}, hard={loss_hard.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss_total += loss.item() * images.size(0)
        running_loss_soft += loss_soft.item() * images.size(0)
        running_loss_hard += loss_hard.item() * images.size(0)
        predicted = student_logits.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return (
        running_loss_total / total,
        correct / total,
        running_loss_soft / total,
        running_loss_hard / total,
    )


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    """Validate and return average CE loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        predicted = outputs.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def run_experiment(model, train_loader, val_loader, device, experiment_name, epochs, lr, t_max,
                   use_amp=True, teacher_model=None, kd_temperature=None, kd_alpha=None):
    """Run one complete CE or KD experiment and retain the best validation checkpoint."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scaler = GradScaler(enabled=use_amp)
    use_kd = teacher_model is not None

    print("=" * 72)
    print(f"Starting training: {experiment_name}")
    print(f"   Mode: {'KD' if use_kd else 'CE'} | Epochs: {epochs} | LR: {lr}")
    if use_kd:
        print(f"   KD: T={kd_temperature}, alpha={kd_alpha}")
    print("=" * 72)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [], "epoch_time": [],
    }
    if use_kd:
        history["train_loss_soft"] = []
        history["train_loss_hard"] = []

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    total_train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        if use_kd:
            train_loss, train_acc, train_loss_soft, train_loss_hard = train_one_epoch_kd(
                model, teacher_model, train_loader, optimizer, scaler, device, use_amp,
                kd_temperature, kd_alpha,
            )
        else:
            train_loss, train_acc = train_one_epoch_ce(
                model, train_loader, criterion, optimizer, scaler, device, use_amp,
            )

        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        if use_kd:
            history["train_loss_soft"].append(train_loss_soft)
            history["train_loss_hard"].append(train_loss_hard)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        best_marker = " BEST" if is_best else ""
        if use_kd:
            print(
                f"Epoch [{epoch:3d}/{epochs}] | Train Loss: {train_loss:.4f} "
                f"(Soft: {train_loss_soft:.4f}, Hard: {train_loss_hard:.4f}) "
                f"| Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} "
                f"| Time: {epoch_time:.1f}s{best_marker}"
            )
        else:
            print(
                f"Epoch [{epoch:3d}/{epochs}] | Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} "
                f"| Time: {epoch_time:.1f}s{best_marker}"
            )

    total_train_time = time.time() - total_train_start
    print("=" * 72)
    print(f"Training complete: {experiment_name}")
    print(f"   Total time  : {total_train_time / 60:.1f} minutes")
    print(f"   Best epoch  : {best_epoch}")
    print(f"   Best val acc: {best_val_acc:.4f}")
    print("=" * 72)
    return history, best_val_acc, best_epoch, best_model_state, total_train_time


def save_checkpoint(model_state, experiment_id, experiment_name, model_name, best_epoch,
                    best_val_acc, filename, config_override=None):
    """Save a standardized model checkpoint for later evaluation."""
    checkpoint = {
        "model_state_dict": model_state,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "config": config_override or {},
        "dataset_name": cfg.DATASET_NAME,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def save_history(history, filename):
    """Save per-epoch training metrics to CSV."""
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    is_kd_history = "train_loss_soft" in history
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        if is_kd_history:
            writer.writerow([
                "epoch", "train_loss_total", "train_loss_soft", "train_loss_hard",
                "train_acc", "val_loss", "val_acc", "lr", "epoch_time",
            ])
        else:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"])
        for idx in range(len(history["epoch"])):
            base_values = [history["epoch"][idx]]
            if is_kd_history:
                base_values.extend([
                    f"{history['train_loss'][idx]:.6f}",
                    f"{history['train_loss_soft'][idx]:.6f}",
                    f"{history['train_loss_hard'][idx]:.6f}",
                ])
            else:
                base_values.append(f"{history['train_loss'][idx]:.6f}")
            base_values.extend([
                f"{history['train_acc'][idx]:.6f}",
                f"{history['val_loss'][idx]:.6f}",
                f"{history['val_acc'][idx]:.6f}",
                f"{history['lr'][idx]:.8f}",
                f"{history['epoch_time'][idx]:.2f}",
            ])
            writer.writerow(base_values)
    print(f"History saved: {path}")
    return path


def experiment_config(**overrides):
    """Return the common controlled protocol plus experiment-specific fields."""
    config = {
        "img_size": cfg.IMG_SIZE,
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "lr": cfg.LR,
        "momentum": cfg.MOMENTUM,
        "weight_decay": cfg.WEIGHT_DECAY,
        "optimizer": cfg.OPTIMIZER,
        "scheduler": cfg.SCHEDULER,
        "seed": cfg.SEED,
        "train_split": cfg.TRAIN_SPLIT,
        "val_split": cfg.VAL_SPLIT,
        "use_amp": cfg.USE_AMP,
        "student_timm_name": student_timm_name,
        "student_pretrained": cfg.STUDENT_PRETRAINED,
    }
    config.update(overrides)
    return config

# %% [markdown]
# ## 5. Experiment 5 - EfficientNet-Lite0 Baseline

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 72)
print("EXPERIMENT 5 - EfficientNet-Lite0 Baseline")
print("=" * 72)
exp5_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED).to(device)
exp5_params = sum(parameter.numel() for parameter in exp5_model.parameters())
print(f"Model     : {cfg.STUDENT_MODEL_NAME} ({student_timm_name})")
print(f"Params    : {exp5_params:,}")
print(f"Pretrained: {cfg.STUDENT_PRETRAINED}")

exp5_history, exp5_best_acc, exp5_best_epoch, exp5_best_state, exp5_time = run_experiment(
    model=exp5_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 5 - EfficientNet-Lite0 Baseline",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
)
save_checkpoint(
    exp5_best_state, 5, "Experiment 5 - EfficientNet-Lite0 Baseline", cfg.STUDENT_MODEL_NAME,
    exp5_best_epoch, exp5_best_acc, cfg.EXP5_CHECKPOINT, experiment_config(),
)
save_history(exp5_history, cfg.EXP5_HISTORY)
del exp5_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Experiment 6 - KD From Scratch

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 72)
print("EXPERIMENT 6 - EfficientNet-Lite0 KD From Scratch")
print("=" * 72)
exp6_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED).to(device)

exp6_history, exp6_best_acc, exp6_best_epoch, exp6_best_state, exp6_time = run_experiment(
    model=exp6_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 6 - EfficientNet-Lite0 KD Scratch",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)
save_checkpoint(
    exp6_best_state, 6, "Experiment 6 - EfficientNet-Lite0 KD Scratch", f"{cfg.STUDENT_MODEL_NAME}-KD",
    exp6_best_epoch, exp6_best_acc, cfg.EXP6_CHECKPOINT,
    experiment_config(
        kd_temperature=cfg.KD_TEMPERATURE,
        kd_alpha=cfg.KD_ALPHA,
        teacher_model=cfg.TEACHER_MODEL_NAME,
    ),
)
save_history(exp6_history, cfg.EXP6_HISTORY)
del exp6_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Experiment 7 - Two-Stage KD

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 72)
print("EXPERIMENT 7 - EfficientNet-Lite0 Two-Stage KD")
print("=" * 72)
exp7_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED).to(device)
exp7_model.load_state_dict(exp5_best_state)
print(f"Loaded Experiment 5 baseline state with val acc {exp5_best_acc:.4f}")

exp7_history, exp7_best_acc, exp7_best_epoch, exp7_best_state, exp7_time = run_experiment(
    model=exp7_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 7 - EfficientNet-Lite0 Two-Stage KD (Stage 2)",
    epochs=cfg.STAGE2_EPOCHS,
    lr=cfg.STAGE2_LR,
    t_max=cfg.STAGE2_T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)
save_checkpoint(
    exp7_best_state, 7, "Experiment 7 - EfficientNet-Lite0 Two-Stage KD", f"{cfg.STUDENT_MODEL_NAME}-KD-TwoStage",
    exp7_best_epoch, exp7_best_acc, cfg.EXP7_CHECKPOINT,
    experiment_config(
        stage1_epochs=cfg.EPOCHS,
        stage1_lr=cfg.LR,
        stage2_epochs=cfg.STAGE2_EPOCHS,
        stage2_lr=cfg.STAGE2_LR,
        kd_temperature=cfg.KD_TEMPERATURE,
        kd_alpha=cfg.KD_ALPHA,
        teacher_model=cfg.TEACHER_MODEL_NAME,
        stage1_checkpoint=cfg.EXP5_CHECKPOINT,
    ),
)
save_history(exp7_history, cfg.EXP7_HISTORY)
del exp7_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 8. Phase 2 Results Summary

# %%
print("\n" + "=" * 76)
print("Phase 2 Results Summary")
print("=" * 76)
print(f"{'Experiment':<45} {'Best Acc':>10} {'Best Ep':>8} {'Time':>10}")
print("-" * 76)
print(f"{'Exp 5: EfficientNet-Lite0 Baseline':<45} {f'{exp5_best_acc * 100:.2f}%':>10} {exp5_best_epoch:>8} {f'{exp5_time / 60:.1f} min':>10}")
print(f"{'Exp 6: EfficientNet-Lite0 KD Scratch':<45} {f'{exp6_best_acc * 100:.2f}%':>10} {exp6_best_epoch:>8} {f'{exp6_time / 60:.1f} min':>10}")
print(f"{'Exp 7: EfficientNet-Lite0 Two-Stage KD':<45} {f'{exp7_best_acc * 100:.2f}%':>10} {exp7_best_epoch:>8} {f'{exp7_time / 60:.1f} min':>10}")
print("-" * 76)
print(f"KD Scratch improvement over baseline: {(exp6_best_acc - exp5_best_acc) * 100:+.2f} pp")
print(f"Two-Stage improvement over baseline : {(exp7_best_acc - exp5_best_acc) * 100:+.2f} pp")
print(f"Two-Stage vs KD Scratch             : {(exp7_best_acc - exp6_best_acc) * 100:+.2f} pp")
print("=" * 76)

# %%
# ============================================================
# 8.1 - Phase 2 Training Curves
# ============================================================

phase2_experiments = [
    ("Exp 5: Baseline", exp5_history, "tab:blue"),
    ("Exp 6: KD Scratch", exp6_history, "tab:orange"),
    ("Exp 7: Two-Stage KD", exp7_history, "tab:green"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for col_idx, (name, history, color) in enumerate(phase2_experiments):
    axes[0][col_idx].plot(history["epoch"], history["train_loss"], label="Train Loss", color=color)
    axes[0][col_idx].plot(history["epoch"], history["val_loss"], label="Val Loss", color=color, linestyle="--")
    axes[0][col_idx].set_title(f"{name} - Loss")
    axes[0][col_idx].legend()
    axes[0][col_idx].grid(True, alpha=0.3)
    axes[1][col_idx].plot(history["epoch"], history["train_acc"], label="Train Acc", color=color)
    axes[1][col_idx].plot(history["epoch"], history["val_acc"], label="Val Acc", color=color, linestyle="--")
    axes[1][col_idx].set_title(f"{name} - Accuracy")
    axes[1][col_idx].legend()
    axes[1][col_idx].grid(True, alpha=0.3)
plt.suptitle("Phase 2 - EfficientNet-Lite0 Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curves_phase2.png"), dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for name, history, color in phase2_experiments:
    ax.plot(history["epoch"], history["val_acc"], label=name, color=color, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Phase 2 - Validation Accuracy Comparison", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "val_accuracy_comparison_phase2.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Verify Notebook 3 Checkpoints

# %%
for checkpoint_name, experiment_name in [
    (cfg.EXP5_CHECKPOINT, "Experiment 5 - Baseline"),
    (cfg.EXP6_CHECKPOINT, "Experiment 6 - KD Scratch"),
    (cfg.EXP7_CHECKPOINT, "Experiment 7 - Two-Stage KD"),
]:
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_name)
    loaded = torch.load(checkpoint_path, map_location="cpu")
    verify_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=False)
    verify_model.load_state_dict(loaded["model_state_dict"])
    print(f"Verified {experiment_name}: {checkpoint_name} | best={loaded['best_val_acc']:.4f}")
    del loaded, verify_model

# %% [markdown]
# ## 10. Final Evaluation Across All 7 Core Experiments

# %%
# ============================================================
# 10.1 - Evaluation Utilities
# ============================================================

def create_eval_model(model_kind: str):
    if model_kind == "teacher":
        return create_teacher_model(cfg.TEACHER_TIMM_NAME, cfg.NUM_CLASSES, pretrained=False)
    if model_kind == "focus_rcnet":
        return create_focus_rcnet(cfg.NUM_CLASSES)
    if model_kind == "lite0":
        return create_lite0_model(cfg.NUM_CLASSES, pretrained=False)
    raise ValueError(f"Unknown model kind: {model_kind}")


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
        probabilities = F.softmax(logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)
        all_labels.append(labels.numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def estimate_flops(model, device, input_size: int) -> int:
    """Estimate Conv2d and Linear FLOPs using forward hooks (multiply-add = 2 FLOPs)."""
    macs = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        kernel_height, kernel_width = module.kernel_size
        kernel_ops = kernel_height * kernel_width * (module.in_channels // module.groups)
        macs += output.numel() * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    sample = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    with torch.no_grad():
        with autocast(enabled=cfg.USE_AMP):
            model(sample)
    for hook in hooks:
        hook.remove()
    return macs * 2


@torch.no_grad()
def measure_inference_time_ms(model, device, input_size: int, warmup_runs: int, measure_runs: int) -> float:
    """Measure batch-size-one inference latency on the current device."""
    sample = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    for _ in range(warmup_runs):
        with autocast(enabled=cfg.USE_AMP):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure_runs):
        with autocast(enabled=cfg.USE_AMP):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / measure_runs


def calculate_macro_roc(y_true, probabilities):
    y_true_binary = label_binarize(y_true, classes=np.arange(cfg.NUM_CLASSES))
    per_class_curves = []
    for class_idx in range(cfg.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_binary[:, class_idx], probabilities[:, class_idx])
        per_class_curves.append((fpr, tpr))
    all_fpr = np.unique(np.concatenate([curve[0] for curve in per_class_curves]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in per_class_curves:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= cfg.NUM_CLASSES
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def evaluate_checkpoint(spec):
    checkpoint = torch.load(spec["path"], map_location="cpu")
    model = create_eval_model(spec["model_kind"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    y_true, y_pred, probabilities = collect_predictions(model, val_loader, device, cfg.USE_AMP)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    roc_auc_macro = roc_auc_score(
        label_binarize(y_true, classes=np.arange(cfg.NUM_CLASSES)),
        probabilities,
        average="macro",
        multi_class="ovr",
    )
    macro_fpr, macro_tpr, plotted_macro_auc = calculate_macro_roc(y_true, probabilities)
    result = {
        "experiment_id": spec["experiment_id"],
        "experiment": spec["experiment"],
        "model": spec["model"],
        "validation_accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
        "auc_macro_ovr": roc_auc_macro,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "flops": estimate_flops(model, device, cfg.IMG_SIZE),
        "inference_time_ms": measure_inference_time_ms(
            model, device, cfg.IMG_SIZE, cfg.INFERENCE_WARMUP_RUNS, cfg.INFERENCE_MEASURE_RUNS,
        ),
        "best_epoch": checkpoint.get("best_epoch", ""),
        "checkpoint": os.path.basename(spec["path"]),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(cfg.NUM_CLASSES))
    roc_data = (macro_fpr, macro_tpr, plotted_macro_auc)
    del model, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, matrix, roc_data

# %%
# ============================================================
# 10.2 - Resolve Checkpoints and Evaluate
# ============================================================

phase1_checkpoint_paths = {
    2: resolve_input_checkpoint(cfg.EXP2_CHECKPOINT, preferred_dirs=(cfg.PHASE1_INPUT_DIR,)),
    3: resolve_input_checkpoint(cfg.EXP3_CHECKPOINT, preferred_dirs=(cfg.PHASE1_INPUT_DIR,)),
    4: resolve_input_checkpoint(cfg.EXP4_CHECKPOINT, preferred_dirs=(cfg.PHASE1_INPUT_DIR,)),
}

checkpoint_specs = [
    {"experiment_id": 1, "experiment": "Exp 1: Teacher", "model": "EfficientNet-B4", "model_kind": "teacher", "path": teacher_checkpoint_path},
    {"experiment_id": 2, "experiment": "Exp 2: Baseline", "model": "Focus-RCNet", "model_kind": "focus_rcnet", "path": phase1_checkpoint_paths[2]},
    {"experiment_id": 3, "experiment": "Exp 3: KD Scratch", "model": "Focus-RCNet", "model_kind": "focus_rcnet", "path": phase1_checkpoint_paths[3]},
    {"experiment_id": 4, "experiment": "Exp 4: Two-Stage KD", "model": "Focus-RCNet", "model_kind": "focus_rcnet", "path": phase1_checkpoint_paths[4]},
    {"experiment_id": 5, "experiment": "Exp 5: Baseline", "model": "EfficientNet-Lite0", "model_kind": "lite0", "path": os.path.join(cfg.OUTPUT_DIR, cfg.EXP5_CHECKPOINT)},
    {"experiment_id": 6, "experiment": "Exp 6: KD Scratch", "model": "EfficientNet-Lite0", "model_kind": "lite0", "path": os.path.join(cfg.OUTPUT_DIR, cfg.EXP6_CHECKPOINT)},
    {"experiment_id": 7, "experiment": "Exp 7: Two-Stage KD", "model": "EfficientNet-Lite0", "model_kind": "lite0", "path": os.path.join(cfg.OUTPUT_DIR, cfg.EXP7_CHECKPOINT)},
]

final_results = []
confusion_matrices = {}
roc_curves = {}
for spec in checkpoint_specs:
    print(f"Evaluating {spec['experiment']} - {spec['model']}")
    result, matrix, roc_data = evaluate_checkpoint(spec)
    final_results.append(result)
    confusion_matrices[spec["experiment_id"]] = matrix
    roc_curves[spec["experiment_id"]] = roc_data
    print(
        f"   Acc={result['validation_accuracy']:.4f} | F1={result['f1_macro']:.4f} "
        f"| AUC={result['auc_macro_ovr']:.4f} | Params={result['parameters']:,} "
        f"| FLOPs={result['flops'] / 1e9:.3f}G | Latency={result['inference_time_ms']:.3f}ms"
    )

# %%
# ============================================================
# 10.3 - Save Final Comparison CSV and Print Table
# ============================================================

comparison_path = os.path.join(cfg.OUTPUT_DIR, cfg.FINAL_COMPARISON_CSV)
fieldnames = [
    "experiment_id", "experiment", "model", "validation_accuracy", "precision_macro",
    "recall_macro", "f1_macro", "auc_macro_ovr", "parameters", "flops",
    "inference_time_ms", "best_epoch", "checkpoint",
]
with open(comparison_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_results)
print(f"Final comparison saved: {comparison_path}")

print("\n" + "=" * 132)
print("Final Controlled Comparison - All 7 Core Experiments")
print("=" * 132)
print(f"{'ID':>2} {'Experiment':<24} {'Model':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Params':>12} {'GFLOPs':>10} {'Latency':>10}")
print("-" * 132)
for result in final_results:
    print(
        f"{result['experiment_id']:>2} {result['experiment']:<24} {result['model']:<20} "
        f"{result['validation_accuracy'] * 100:>7.2f}% {result['f1_macro']:>8.4f} "
        f"{result['auc_macro_ovr']:>8.4f} {result['parameters']:>12,} "
        f"{result['flops'] / 1e9:>10.3f} {result['inference_time_ms']:>8.3f}ms"
    )
print("=" * 132)

# %%
# ============================================================
# 10.4 - Save Confusion Matrices
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()
for axis, result in zip(axes, final_results):
    matrix = confusion_matrices[result["experiment_id"]]
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(f"{result['experiment']}\n{result['model']}", fontsize=10)
    axis.set_xticks(np.arange(cfg.NUM_CLASSES))
    axis.set_yticks(np.arange(cfg.NUM_CLASSES))
    axis.set_xticklabels(cfg.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(cfg.CLASS_NAMES, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    threshold = matrix.max() / 2
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(
                col_idx, row_idx, str(matrix[row_idx, col_idx]),
                ha="center", va="center",
                color="white" if matrix[row_idx, col_idx] > threshold else "black",
                fontsize=8,
            )
axes[-1].axis("off")
fig.colorbar(image, ax=axes.tolist(), shrink=0.72)
plt.suptitle("Confusion Matrices - All 7 Core Experiments", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# ============================================================
# 10.5 - Save Macro ROC Curves
# ============================================================

fig, axis = plt.subplots(figsize=(11, 8))
for result in final_results:
    fpr, tpr, plotted_auc = roc_curves[result["experiment_id"]]
    axis.plot(fpr, tpr, linewidth=1.8, label=f"{result['experiment_id']}: {result['model']} (AUC={plotted_auc:.3f})")
axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
axis.set_xlabel("False Positive Rate")
axis.set_ylabel("True Positive Rate")
axis.set_title("Macro-Average ROC Curves - All 7 Core Experiments", fontweight="bold")
axis.legend(fontsize=9)
axis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Output Artifacts
#
# - efficientnet_lite0_baseline_best.pth
# - efficientnet_lite0_kd_scratch_best.pth
# - efficientnet_lite0_kd_twostage_best.pth
# - training_history_exp5.csv
# - training_history_exp6.csv
# - training_history_exp7.csv
# - training_curves_phase2.png
# - val_accuracy_comparison_phase2.png
# - final_comparison_table.csv
# - confusion_matrices.png
# - roc_curves.png

# %%
print("\n" + "=" * 72)
print("Notebook 3 complete - Phase 2 controlled comparison and final evaluation done")
print("=" * 72)
print(f"Exp 5 (Lite0 Baseline)    : {exp5_best_acc:.4f} ({exp5_best_acc * 100:.2f}%)")
print(f"Exp 6 (Lite0 KD Scratch)  : {exp6_best_acc:.4f} ({exp6_best_acc * 100:.2f}%)")
print(f"Exp 7 (Lite0 Two-Stage KD): {exp7_best_acc:.4f} ({exp7_best_acc * 100:.2f}%)")
print("Analyze these controlled results before deciding whether to run the optional Lite0 Native Profile.")
print("=" * 72)
