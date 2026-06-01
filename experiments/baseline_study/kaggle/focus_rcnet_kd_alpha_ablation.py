#!/usr/bin/env python
# =============================================================================
# Notebook 2B — Focus-RCNet KD Scratch Alpha Ablation
# =============================================================================
# Ablation A1: Focus-RCNet KD Scratch with alpha=0.10
# Ablation A2: Focus-RCNet KD Scratch with alpha=0.05
#
# This notebook implements the Focus-RCNet architecture from:
#   "Focus-RCNet: A Lightweight Recyclable Waste Classification Algorithm
#    Based on Focus and Knowledge Distillation" (Zheng et al., 2023)
#
# Protocol: Phase 1 configuration
# - IMG_SIZE: 380x380
# - LR: 0.05
# - EPOCHS: 100
# - BATCH_SIZE: 8
# - Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - KD: T=4, alpha in {0.10, 0.05}
#
# This script is separate from Notebook 2 and must not overwrite the original
# controlled-comparison artifacts for Experiments 2, 3, and 4.
# =============================================================================

# %% [markdown]
# # Notebook 2B — Focus-RCNet KD Scratch Alpha Ablation
#
# Follow-up ini menguji apakah bobot soft-target yang lebih kecil memperbaiki
# KD Scratch Focus-RCNet. Teacher tetap aktif sejak batch pertama dan model
# student selalu dibuat ulang dari random initialization.
#
# Teacher model (EfficientNet-B4) di-load dari checkpoint Notebook 1.

# %% [markdown]
# ## 0. Setup

# %%
# ============================================================
# 0.1 — Imports
# ============================================================
import os
import random
import time
import copy
import csv
import math
from pathlib import Path

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
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("✅ All imports successful")

# %%
# ============================================================
# 0.2 — Configuration
# ============================================================

class Config:
    """Centralized Focus-RCNet KD Scratch alpha-ablation configuration."""

    # ----- Experiment Metadata -----
    NOTEBOOK_ID = "2B"
    PHASE = "1-focus-rcnet-kd-alpha-ablation"

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

    # ----- Input -----
    IMG_SIZE = 380

    # ----- Training (Experiments 2 & 3) -----
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

    # ----- Loss -----
    LOSS_CE = "CrossEntropyLoss"

    # ----- Knowledge Distillation -----
    KD_TEMPERATURE = 4
    KD_ALPHA_VARIANTS = (
        {"ablation_id": "A1", "alpha": 0.10},
        {"ablation_id": "A2", "alpha": 0.05},
    )
    REFERENCE_KD_ALPHA = 0.50
    REFERENCE_BASELINE_ACC = 0.8524
    REFERENCE_KD_SCRATCH_ACC = 0.8564
    REFERENCE_TWOSTAGE_ACC = 0.8630

    # ----- Models -----
    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"
    STUDENT_MODEL_NAME = "Focus-RCNet"

    # ----- Output Filenames -----
    TEACHER_CHECKPOINT = "efficientnet_b4_teacher_best.pth"
    ABLATION_COMPARISON_CSV = "focus_rcnet_kd_scratch_alpha_ablation_table.csv"

    # ----- Paths -----
    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    TEACHER_CHECKPOINT_PATH = "/kaggle/input/notebooks/hamzapratama/notebook1-teacher-training/efficientnet_b4_teacher_best.pth"
    OUTPUT_DIR = "/kaggle/working/"


cfg = Config()

# Print configuration summary
print("=" * 60)
print(f"  Notebook 2B — Focus-RCNet KD Scratch Alpha Ablation")
print("=" * 60)
print(f"  Teacher      : {cfg.TEACHER_MODEL_NAME}")
print(f"  Student      : {cfg.STUDENT_MODEL_NAME}")
print(f"  Dataset      : {cfg.DATASET_NAME} ({cfg.NUM_CLASSES} classes)")
print(f"  Image Size   : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"  Epochs       : {cfg.EPOCHS}")
print(f"  Batch Size   : {cfg.BATCH_SIZE}")
print(f"  Optimizer    : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM})")
print(f"  Scheduler    : {cfg.SCHEDULER}")
print(f"  KD           : T={cfg.KD_TEMPERATURE}, alpha variants={[item['alpha'] for item in cfg.KD_ALPHA_VARIANTS]}")
print(f"  AMP          : {cfg.USE_AMP}")
print(f"  Seed         : {cfg.SEED}")
print("=" * 60)

# %%
# ============================================================
# 0.3 — Seed Everything & Device Setup
# ============================================================

def seed_everything(seed: int) -> None:
    """Set seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(cfg.SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

# %% [markdown]
# ## 1. Dataset & DataLoader

# %%
# ============================================================
# 1.1 — Data Augmentation (Albumentations)
# ============================================================

# ImageNet normalization constants
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

print("✅ Augmentation pipelines created")
print(f"   Train transforms: Resize→HFlip→VFlip→BrightnessContrast→CoarseDropout→Normalize")
print(f"   Val transforms  : Resize→Normalize")

# %%
# ============================================================
# 1.2 — Albumentations Dataset Wrapper
# ============================================================

class AlbumentationsDataset(torch.utils.data.Dataset):
    """Wraps an ImageFolder dataset with Albumentations transforms."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # PIL Image → numpy array
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

# %%
# ============================================================
# 1.3 — Load Dataset & Create Train/Val Split
# ============================================================
# We load the split indices from the teacher checkpoint
# to guarantee identical train/val partitions.

# Load teacher checkpoint for split indices
teacher_ckpt = torch.load(cfg.TEACHER_CHECKPOINT_PATH, map_location="cpu")
train_indices = teacher_ckpt["train_indices"]
val_indices = teacher_ckpt["val_indices"]
teacher_val_acc = teacher_ckpt["best_val_acc"]
print(f"📂 Loaded split indices from teacher checkpoint")
print(f"   Teacher best val acc: {teacher_val_acc:.4f}")

# Load full dataset
full_dataset = datasets.ImageFolder(
    root=cfg.DATASET_PATH,
    transform=None,
)

print(f"📁 Dataset loaded from: {cfg.DATASET_PATH}")
print(f"   Total images : {len(full_dataset)}")
print(f"   Classes found: {full_dataset.classes}")
assert len(full_dataset.classes) == cfg.NUM_CLASSES, \
    f"Expected {cfg.NUM_CLASSES} classes, found {len(full_dataset.classes)}"

# Create subsets using loaded indices
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# Wrap with Albumentations
train_dataset = AlbumentationsDataset(train_subset, transform=train_transform)
val_dataset = AlbumentationsDataset(val_subset, transform=val_transform)

print(f"\n📊 Split (from teacher checkpoint, seed={cfg.SEED}):")
print(f"   Train : {len(train_indices)} images ({len(train_indices)/len(full_dataset)*100:.1f}%)")
print(f"   Val   : {len(val_indices)} images ({len(val_indices)/len(full_dataset)*100:.1f}%)")

# %%
# ============================================================
# 1.4 — DataLoaders
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

print(f"✅ DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches  : {len(val_loader)}")

# %%
# ============================================================
# 1.5 — Verify Class Distribution in Split
# ============================================================

from collections import Counter

targets = [s[1] for s in full_dataset.samples]
train_labels = [targets[i] for i in train_indices]
val_labels = [targets[i] for i in val_indices]

train_dist = Counter(train_labels)
val_dist = Counter(val_labels)

print("📊 Class Distribution:")
print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
print("-" * 42)
for idx, class_name in enumerate(full_dataset.classes):
    t = train_dist[idx]
    v = val_dist[idx]
    print(f"{class_name:<15} {t:>8} {v:>8} {t+v:>8}")
print("-" * 42)
print(f"{'Total':<15} {len(train_indices):>8} {len(val_indices):>8} {len(full_dataset):>8}")

# %% [markdown]
# ## 2. Load Teacher Model

# %%
# ============================================================
# 2.1 — Load Pre-trained Teacher (EfficientNet-B4)
# ============================================================

def create_teacher_model(model_name: str, num_classes: int, pretrained: bool = False):
    """Create EfficientNet-B4 teacher model using timm."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


# Build teacher and load trained weights
teacher_model = create_teacher_model(
    cfg.TEACHER_TIMM_NAME,
    cfg.NUM_CLASSES,
    pretrained=False,
)
teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
teacher_model = teacher_model.to(device)
teacher_model.eval()

# Freeze all teacher parameters
for param in teacher_model.parameters():
    param.requires_grad = False

teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"✅ Teacher model loaded: {cfg.TEACHER_MODEL_NAME}")
print(f"   Parameters  : {teacher_params:,}")
print(f"   Best epoch  : {teacher_ckpt['best_epoch']}")
print(f"   Best val acc: {teacher_ckpt['best_val_acc']:.4f}")
print(f"   Mode        : eval (frozen)")

# Free the checkpoint from memory (keep only the model)
del teacher_ckpt
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 3. Focus-RCNet Model Definition
#
# Custom implementation based on:
# - **Focus Module**: From YOLOv5, reduces spatial dimensions while preserving info
# - **Sandglass Block**: From MobileNeXt, reversed inverted residual
# - **SimAM**: Parameter-free attention mechanism
#
# Reference: Zheng et al. (2023), "Focus-RCNet: A Lightweight Recyclable Waste
# Classification Algorithm Based on Focus and Knowledge Distillation"

# %%
# ============================================================
# 3.1 — Focus Module (from YOLOv5)
# ============================================================

class Focus(nn.Module):
    """
    Focus module from YOLOv5.

    Performs spatial slicing to reduce resolution by 2x while expanding channels 4x,
    then applies a CBS (Conv + BatchNorm + SiLU) block.

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # After slicing, channels are 4x the input
        sliced_channels = in_channels * 4
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(sliced_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Slice operation: take every other pixel in both spatial dimensions
        # This creates 4 slices and concatenates along channel dimension
        x = torch.cat([
            x[..., ::2, ::2],   # top-left
            x[..., 1::2, ::2],  # bottom-left
            x[..., ::2, 1::2],  # top-right
            x[..., 1::2, 1::2], # bottom-right
        ], dim=1)
        return self.conv(x)


# %%
# ============================================================
# 3.2 — SimAM Attention Module (Parameter-Free)
# ============================================================

class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for CNNs.

    Computes 3D attention weights based on energy function without
    any learnable parameters. Each neuron's importance is determined
    by how distinguishable it is from other neurons in the same channel.

    Reference: Yang et al. (2021), ICML.
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1

        # Channel-wise mean and variance
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * torch.sigmoid(y)


# %%
# ============================================================
# 3.3 — Sandglass Block (from MobileNeXt)
# ============================================================

class SandglassBlock(nn.Module):
    """
    Sandglass block (reversed inverted residual) from MobileNeXt.

    Structure:
        DWConv 3x3 → PW 1x1 (reduce) → PW 1x1 (expand) → DWConv 3x3
        with residual connection at high dimensions.

    Unlike inverted residuals (narrow→wide→narrow), sandglass blocks
    perform identity mapping and spatial transforms in high-dimensional
    space (wide→narrow→wide), reducing information loss.

    Args:
        in_channels: input channel count
        out_channels: output channel count
        stride: spatial stride (1 or 2)
        reduction: channel reduction ratio for bottleneck
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = in_channels // reduction

        # Ensure at least 1 channel in bottleneck
        mid_channels = max(mid_channels, 1)

        self.layers = nn.Sequential(
            # 1. Depthwise conv 3x3 (high-dim spatial transform)
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),

            # 2. Pointwise 1x1 (reduce channels — bottleneck)
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # No activation here (linear bottleneck)

            # 3. Pointwise 1x1 (expand channels)
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),

            # 4. Depthwise conv 3x3 (high-dim spatial transform)
            nn.Conv2d(out_channels, out_channels, 3, stride=1,
                      padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            # No activation before residual add
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


# %%
# ============================================================
# 3.4 — Focus-RCNet Full Architecture
# ============================================================

class FocusRCNet(nn.Module):
    """
    Focus-RCNet: Lightweight recyclable waste classification network.

    Architecture:
        Focus(3→24) → [Sandglass stages with SimAM] → Conv5(384→512)
        → GAP → Classifier

    Stage configuration from Table 1 of the paper:
        Stage 1: 24→48,   n=4, first block stride=2
        Stage 2: 48→96,   n=3, first block stride=2
        Stage 3: 96→192,  n=2, first block stride=2
        Stage 4: 192→384, n=2, first block stride=2

    Args:
        num_classes: number of output classes
        dropout: dropout rate before classifier
    """

    # Stage config: (out_channels, num_blocks, stride, reduction)
    STAGE_CONFIG = [
        (48,  4, 2, 2),   # Stage 1
        (96,  3, 2, 2),   # Stage 2
        (192, 2, 2, 2),   # Stage 3
        (384, 2, 2, 2),   # Stage 4
    ]

    def __init__(self, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()

        # Focus module: 3 → 24 channels, spatial /2
        self.focus = Focus(in_channels=3, out_channels=24, kernel_size=1)

        # Build sandglass stages with SimAM attention
        stages = []
        in_ch = 24
        for out_ch, num_blocks, stride, reduction in self.STAGE_CONFIG:
            stage_blocks = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                stage_blocks.append(SandglassBlock(in_ch, out_ch, stride=s, reduction=reduction))
                in_ch = out_ch
            stage_blocks.append(SimAM())
            stages.append(nn.Sequential(*stage_blocks))

        self.stages = nn.Sequential(*stages)

        # Final 1x1 convolution from Table 1 of the paper
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming Normal for Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.focus(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


# %%
# ============================================================
# 3.5 — Verify Focus-RCNet Architecture
# ============================================================

def create_focus_rcnet(num_classes: int = 6) -> FocusRCNet:
    """Create a fresh Focus-RCNet model."""
    return FocusRCNet(num_classes=num_classes)


# Verify architecture
_test_model = create_focus_rcnet(cfg.NUM_CLASSES)
_total_params = sum(p.numel() for p in _test_model.parameters())
_trainable_params = sum(p.numel() for p in _test_model.parameters() if p.requires_grad)

print(f"✅ Focus-RCNet architecture verified")
print(f"   Total params    : {_total_params:,}")
print(f"   Trainable params: {_trainable_params:,}")
print(f"   Paper target    : ~525,802")
print(f"   Match           : {'✅ Close' if abs(_total_params - 525802) / 525802 < 0.15 else '⚠️ Needs tuning'}")

# Verify forward pass shape
_test_input = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
_test_output = _test_model(_test_input)
print(f"   Input shape     : {list(_test_input.shape)}")
print(f"   Output shape    : {list(_test_output.shape)}")
assert _test_output.shape == (1, cfg.NUM_CLASSES), \
    f"Expected output shape (1, {cfg.NUM_CLASSES}), got {_test_output.shape}"

# Print per-stage feature map sizes
print(f"\n   Feature map progression:")
_x = _test_input
_x = _test_model.focus(_x)
print(f"     Focus  : {list(_x.shape)}")
for i, stage in enumerate(_test_model.stages):
    _x = stage(_x)
    print(f"     Stage {i+1}: {list(_x.shape)}")
_x = _test_model.conv5(_x)
print(f"     Conv5  : {list(_x.shape)}")

del _test_model, _test_input, _test_output, _x

# %% [markdown]
# ## 3b. Training Utilities

# %%
# ============================================================
# 3.6 — Training & Validation Functions
# ============================================================

def train_one_epoch_ce(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train one epoch with standard CrossEntropy loss."""
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_one_epoch_kd(student, teacher, loader, optimizer, scaler, device, use_amp,
                       temperature, alpha):
    """
    Train one epoch with Knowledge Distillation loss.

    KD Loss = α * KL_div(student_soft, teacher_soft) * T²
            + (1-α) * CE(student_logits, true_labels)
    """
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
            # Student forward
            student_logits = student(images)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(images)

        # Compute soft-target operations in FP32 for numerical stability.
        with autocast(enabled=False):
            # Soft targets
            student_log_soft = F.log_softmax(student_logits.float() / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits.float() / temperature, dim=1)

            # KD loss components
            loss_soft = F.kl_div(student_log_soft, teacher_soft,
                                 reduction="batchmean") * (temperature ** 2)
            loss_hard = ce_criterion(student_logits.float(), labels)

            # Combined loss
            loss = alpha * loss_soft + (1.0 - alpha) * loss_hard

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite KD loss detected: total={loss.item()}, "
                f"soft={loss_soft.item()}, hard={loss_hard.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss_total += loss.item() * images.size(0)
        running_loss_soft += loss_soft.item() * images.size(0)
        running_loss_hard += loss_hard.item() * images.size(0)
        _, predicted = student_logits.max(1)
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
    """Validate and return average loss and accuracy."""
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# %%
# ============================================================
# 3.7 — Generic Experiment Runner
# ============================================================

def run_experiment(
    model,
    train_loader,
    val_loader,
    device,
    experiment_name,
    epochs,
    lr,
    t_max,
    use_amp=True,
    # KD arguments (if None, use standard CE training)
    teacher_model=None,
    kd_temperature=None,
    kd_alpha=None,
):
    """
    Run a complete training experiment (either CE or KD).

    Returns:
        history: dict with per-epoch metrics
        best_val_acc: float
        best_epoch: int
        best_model_state: OrderedDict
        total_train_time: float (seconds)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scaler = GradScaler(enabled=use_amp)

    use_kd = teacher_model is not None
    mode_str = "KD" if use_kd else "CE"

    print("=" * 60)
    print(f"  🚀 Starting Training: {experiment_name}")
    print(f"  Mode: {mode_str} | Epochs: {epochs} | LR: {lr}")
    if use_kd:
        print(f"  KD: T={kd_temperature}, α={kd_alpha}")
    print("=" * 60)

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

        # Train
        if use_kd:
            train_loss, train_acc, train_loss_soft, train_loss_hard = train_one_epoch_kd(
                model, teacher_model, train_loader, optimizer, scaler,
                device, use_amp, kd_temperature, kd_alpha,
            )
        else:
            train_loss, train_acc = train_one_epoch_ce(
                model, train_loader, criterion, optimizer, scaler,
                device, use_amp,
            )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Record history
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

        # Check for best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        # Logging
        best_marker = " ⭐ BEST" if is_best else ""
        if use_kd:
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} "
                f"(Soft: {train_loss_soft:.4f}, Hard: {train_loss_hard:.4f}) "
                f"| Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
                f"| LR: {current_lr:.6f} "
                f"| Time: {epoch_time:.1f}s{best_marker}"
            )
        else:
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
                f"| LR: {current_lr:.6f} "
                f"| Time: {epoch_time:.1f}s{best_marker}"
            )

    total_train_time = time.time() - total_train_start

    print("=" * 60)
    print(f"  ✅ Training Complete: {experiment_name}")
    print(f"  Total time  : {total_train_time/60:.1f} minutes")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print("=" * 60)

    return history, best_val_acc, best_epoch, best_model_state, total_train_time


# %%
# ============================================================
# 3.8 — Checkpoint Save/Load Helpers
# ============================================================

def save_checkpoint(model_state, experiment_id, experiment_name, model_name,
                    best_epoch, best_val_acc, history, filename, config_override=None):
    """Save a standardized checkpoint."""
    ckpt_config = config_override or {
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
    }

    checkpoint = {
        "model_state_dict": model_state,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "config": ckpt_config,
        "dataset_name": cfg.DATASET_NAME,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    path = os.path.join(cfg.OUTPUT_DIR, filename)
    torch.save(checkpoint, path)
    fsize = os.path.getsize(path) / 1e6
    print(f"✅ Checkpoint saved: {path} ({fsize:.1f} MB)")
    return path


def save_history(history, filename):
    """Save training history to CSV."""
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    is_kd_history = "train_loss_soft" in history
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if is_kd_history:
            writer.writerow([
                "epoch", "train_loss_total", "train_loss_soft", "train_loss_hard",
                "train_acc", "val_loss", "val_acc", "lr", "epoch_time",
            ])
        else:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"])
        for i in range(len(history["epoch"])):
            if is_kd_history:
                writer.writerow([
                    history["epoch"][i],
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['train_loss_soft'][i]:.6f}",
                    f"{history['train_loss_hard'][i]:.6f}",
                    f"{history['train_acc'][i]:.6f}",
                    f"{history['val_loss'][i]:.6f}",
                    f"{history['val_acc'][i]:.6f}",
                    f"{history['lr'][i]:.8f}",
                    f"{history['epoch_time'][i]:.2f}",
                ])
            else:
                writer.writerow([
                    history["epoch"][i],
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['train_acc'][i]:.6f}",
                    f"{history['val_loss'][i]:.6f}",
                    f"{history['val_acc'][i]:.6f}",
                    f"{history['lr'][i]:.8f}",
                    f"{history['epoch_time'][i]:.2f}",
                ])
    print(f"✅ History saved: {path}")
    return path


# %% [markdown]
# ## 4. Run Focus-RCNet KD Scratch Alpha Ablations

# %%
def ablation_artifacts(ablation_id: str):
    suffix = ablation_id.lower()
    return {
        "checkpoint": f"focus_rcnet_kd_scratch_{suffix}_best.pth",
        "history": f"training_history_focus_rcnet_kd_scratch_{suffix}.csv",
    }


ablation_records = []
for variant in cfg.KD_ALPHA_VARIANTS:
    seed_everything(cfg.SEED)
    ablation_id = variant["ablation_id"]
    kd_alpha = variant["alpha"]
    artifacts = ablation_artifacts(ablation_id)
    experiment_name = f"{ablation_id} — Focus-RCNet KD Scratch (alpha={kd_alpha:.2f})"

    print("\n" + "=" * 72)
    print(f"  {experiment_name}")
    print("=" * 72)
    model = create_focus_rcnet(cfg.NUM_CLASSES).to(device)
    history, best_acc, best_epoch, best_state, train_time = run_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        experiment_name=experiment_name,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        t_max=cfg.T_MAX,
        use_amp=cfg.USE_AMP,
        teacher_model=teacher_model,
        kd_temperature=cfg.KD_TEMPERATURE,
        kd_alpha=kd_alpha,
    )
    assert not teacher_model.training, "Teacher must remain in eval mode"
    assert all(not parameter.requires_grad for parameter in teacher_model.parameters())
    assert all(parameter.grad is None for parameter in teacher_model.parameters())
    checkpoint_path = save_checkpoint(
        model_state=best_state,
        experiment_id=f"3-{ablation_id}",
        experiment_name=experiment_name,
        model_name=f"{cfg.STUDENT_MODEL_NAME}-KD-Scratch-alpha{kd_alpha:.2f}",
        best_epoch=best_epoch,
        best_val_acc=best_acc,
        history=history,
        filename=artifacts["checkpoint"],
        config_override={
            "track": "focus_rcnet_kd_scratch_alpha_ablation",
            "variant": "kd_scratch_from_random_initialization",
            "ablation_id": ablation_id,
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
            "kd_temperature": cfg.KD_TEMPERATURE,
            "kd_alpha": kd_alpha,
            "teacher_model": cfg.TEACHER_MODEL_NAME,
        },
    )
    save_history(history, artifacts["history"])
    ablation_records.append({
        "ablation_id": ablation_id,
        "alpha": kd_alpha,
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time": train_time,
        "checkpoint_path": checkpoint_path,
    })
    del model, best_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 5. Results Summary and Curves

# %%
print("\n" + "=" * 100)
print("  Focus-RCNet KD Scratch Alpha Ablation")
print("=" * 100)
print(f"{'Run':>4} {'Alpha':>7} {'Best Acc':>10} {'Best Ep':>8} {'Time':>10} {'Delta vs Exp 3':>15} {'Delta vs Base':>14}")
print("-" * 100)
for record in ablation_records:
    delta_exp3 = (record["best_acc"] - cfg.REFERENCE_KD_SCRATCH_ACC) * 100
    delta_base = (record["best_acc"] - cfg.REFERENCE_BASELINE_ACC) * 100
    print(
        f"{record['ablation_id']:>4} {record['alpha']:>7.2f} "
        f"{record['best_acc'] * 100:>9.2f}% {record['best_epoch']:>8} "
        f"{record['train_time'] / 60:>8.1f}m {delta_exp3:>+14.2f}pp {delta_base:>+13.2f}pp"
    )
print("-" * 100)
print(f"Reference Exp 2 baseline       : {cfg.REFERENCE_BASELINE_ACC * 100:.2f}%")
print(f"Reference Exp 3 alpha={cfg.REFERENCE_KD_ALPHA:.2f}: {cfg.REFERENCE_KD_SCRATCH_ACC * 100:.2f}%")
print(f"Reference Exp 4 Two-Stage      : {cfg.REFERENCE_TWOSTAGE_ACC * 100:.2f}%")
print("=" * 100)

colors = {"A1": "tab:blue", "A2": "tab:orange"}
fig, axes = plt.subplots(2, len(ablation_records), figsize=(13, 8), squeeze=False)
for col_idx, record in enumerate(ablation_records):
    history = record["history"]
    color = colors[record["ablation_id"]]
    axes[0][col_idx].plot(history["epoch"], history["train_loss"], label="Train Total", color=color)
    axes[0][col_idx].plot(history["epoch"], history["val_loss"], label="Val CE", color=color, linestyle="--")
    axes[0][col_idx].set_title(f"{record['ablation_id']} alpha={record['alpha']:.2f} — Loss")
    axes[0][col_idx].legend()
    axes[0][col_idx].grid(True, alpha=0.3)
    axes[1][col_idx].plot(history["epoch"], history["train_acc"], label="Train Acc", color=color)
    axes[1][col_idx].plot(history["epoch"], history["val_acc"], label="Val Acc", color=color, linestyle="--")
    axes[1][col_idx].set_title(f"{record['ablation_id']} alpha={record['alpha']:.2f} — Accuracy")
    axes[1][col_idx].legend()
    axes[1][col_idx].grid(True, alpha=0.3)
plt.suptitle("Focus-RCNet KD Scratch Alpha Ablation Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curves_focus_rcnet_kd_scratch_alpha_ablation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axis = plt.subplots(figsize=(10, 6))
for record in ablation_records:
    axis.plot(
        record["history"]["epoch"],
        record["history"]["val_acc"],
        label=f"{record['ablation_id']}: alpha={record['alpha']:.2f}",
        color=colors[record["ablation_id"]],
        linewidth=2,
    )
axis.axhline(cfg.REFERENCE_BASELINE_ACC, color="tab:gray", linestyle=":", label="Exp 2 baseline")
axis.axhline(cfg.REFERENCE_KD_SCRATCH_ACC, color="tab:red", linestyle="--", label="Exp 3 alpha=0.50 reference")
axis.axhline(cfg.REFERENCE_TWOSTAGE_ACC, color="tab:green", linestyle="--", label="Exp 4 Two-Stage reference")
axis.set_xlabel("Epoch")
axis.set_ylabel("Validation Accuracy")
axis.set_title("Focus-RCNet KD Scratch Validation Accuracy", fontweight="bold")
axis.legend()
axis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "val_accuracy_focus_rcnet_kd_scratch_alpha_ablation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Verify Checkpoints and Save Comparison

# %%
comparison_rows = []
for record in ablation_records:
    loaded = torch.load(record["checkpoint_path"], map_location="cpu")
    verify_model = create_focus_rcnet(cfg.NUM_CLASSES)
    verify_model.load_state_dict(loaded["model_state_dict"])
    print(
        f"Verified {record['ablation_id']}: {os.path.basename(record['checkpoint_path'])} "
        f"| best={loaded['best_val_acc']:.4f}"
    )
    comparison_rows.append({
        "ablation_id": record["ablation_id"],
        "alpha": record["alpha"],
        "temperature": cfg.KD_TEMPERATURE,
        "best_val_accuracy": loaded["best_val_acc"],
        "best_epoch": loaded["best_epoch"],
        "checkpoint": os.path.basename(record["checkpoint_path"]),
    })
    del loaded, verify_model

comparison_path = os.path.join(cfg.OUTPUT_DIR, cfg.ABLATION_COMPARISON_CSV)
fieldnames = ["ablation_id", "alpha", "temperature", "best_val_accuracy", "best_epoch", "checkpoint"]
with open(comparison_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Ablation comparison saved: {comparison_path}")

# %% [markdown]
# ## Output Artifacts
#
# - focus_rcnet_kd_scratch_a1_best.pth
# - focus_rcnet_kd_scratch_a2_best.pth
# - training_history_focus_rcnet_kd_scratch_a1.csv
# - training_history_focus_rcnet_kd_scratch_a2.csv
# - focus_rcnet_kd_scratch_alpha_ablation_table.csv
# - training_curves_focus_rcnet_kd_scratch_alpha_ablation.png
# - val_accuracy_focus_rcnet_kd_scratch_alpha_ablation.png

# %%
print("\n" + "=" * 76)
print("  Notebook 2B complete — Focus-RCNet KD Scratch alpha ablation done")
print("=" * 76)
for row in comparison_rows:
    print(f"  {row['ablation_id']} (alpha={row['alpha']:.2f}): {row['best_val_accuracy']:.4f} ({row['best_val_accuracy'] * 100:.2f}%)")
print(f"  Reference Exp 3 (alpha={cfg.REFERENCE_KD_ALPHA:.2f}): {cfg.REFERENCE_KD_SCRATCH_ACC:.4f} ({cfg.REFERENCE_KD_SCRATCH_ACC * 100:.2f}%)")
print("=" * 76)
