#!/usr/bin/env python
# =============================================================================
# Notebook 2 — Phase 1: Focus-RCNet Experiments
# =============================================================================
# Experiment 2: Train Focus-RCNet (Baseline)
# Experiment 3: KD From Scratch (EfficientNet-B4 → Focus-RCNet)
# Experiment 4: Two-Stage KD (EfficientNet-B4 → Focus-RCNet)
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
# - KD: T=4, α=0.5
# - Two-Stage: Stage2 epochs=50, lr=0.005
# =============================================================================

# %% [markdown]
# # Notebook 2 — Phase 1: Focus-RCNet Experiments
#
# **Experiments 2, 3, 4** dari 7 eksperimen total.
#
# Notebook ini mengimplementasikan:
# - **Exp 2**: Focus-RCNet baseline (tanpa KD)
# - **Exp 3**: KD from scratch (EfficientNet-B4 → Focus-RCNet)
# - **Exp 4**: Two-stage KD (load Exp 2 → fine-tune dengan KD)
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
    """Centralized experiment configuration — Phase 1 protocol."""

    # ----- Experiment Metadata -----
    NOTEBOOK_ID = 2
    PHASE = 1

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
    KD_ALPHA = 0.5

    # ----- Two-Stage KD (Experiment 4, Stage 2) -----
    STAGE2_EPOCHS = 50
    STAGE2_LR = 0.005
    STAGE2_FREEZE_LAYERS = False
    STAGE2_T_MAX = 50  # = STAGE2_EPOCHS

    # ----- Models -----
    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"
    STUDENT_MODEL_NAME = "Focus-RCNet"

    # ----- Checkpoint Filenames -----
    TEACHER_CHECKPOINT = "efficientnet_b4_teacher_best.pth"
    EXP2_CHECKPOINT = "focus_rcnet_baseline_best.pth"
    EXP3_CHECKPOINT = "focus_rcnet_kd_scratch_best.pth"
    EXP4_CHECKPOINT = "focus_rcnet_kd_twostage_best.pth"
    EXP2_HISTORY = "training_history_exp2.csv"
    EXP3_HISTORY = "training_history_exp3.csv"
    EXP4_HISTORY = "training_history_exp4.csv"

    # ----- Paths -----
    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    TEACHER_CHECKPOINT_PATH = "/kaggle/input/notebook1-output/efficientnet_b4_teacher_best.pth"
    OUTPUT_DIR = "/kaggle/working/"


cfg = Config()

# Print configuration summary
print("=" * 60)
print(f"  Notebook 2 — Phase 1: Focus-RCNet Experiments")
print("=" * 60)
print(f"  Teacher      : {cfg.TEACHER_MODEL_NAME}")
print(f"  Student      : {cfg.STUDENT_MODEL_NAME}")
print(f"  Dataset      : {cfg.DATASET_NAME} ({cfg.NUM_CLASSES} classes)")
print(f"  Image Size   : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"  Epochs       : {cfg.EPOCHS} (Stage2: {cfg.STAGE2_EPOCHS})")
print(f"  Batch Size   : {cfg.BATCH_SIZE}")
print(f"  Optimizer    : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM})")
print(f"  Scheduler    : {cfg.SCHEDULER}")
print(f"  KD           : T={cfg.KD_TEMPERATURE}, α={cfg.KD_ALPHA}")
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
        max_holes=1,
        max_height=int(cfg.IMG_SIZE * 0.2),
        max_width=int(cfg.IMG_SIZE * 0.2),
        min_holes=1,
        min_height=int(cfg.IMG_SIZE * 0.05),
        min_width=int(cfg.IMG_SIZE * 0.05),
        fill_value=0,
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
        Focus(3→32) → [Sandglass stages with SimAM] → GAP → Classifier

    Stage configuration (reconstructed from paper, ~525K params):
        Stage 1: 32→64,   n=2, stride=2
        Stage 2: 64→96,   n=3, stride=2
        Stage 3: 96→160,  n=4, stride=2
        Stage 4: 160→256, n=3, stride=2
        Stage 5: 256→320, n=2, stride=1

    Args:
        num_classes: number of output classes
        dropout: dropout rate before classifier
    """

    # Stage config: (out_channels, num_blocks, stride, reduction)
    STAGE_CONFIG = [
        (64,  2, 2, 2),   # Stage 1
        (96,  3, 2, 2),   # Stage 2
        (160, 4, 2, 2),   # Stage 3
        (256, 3, 2, 2),   # Stage 4
        (320, 2, 1, 2),   # Stage 5
    ]

    def __init__(self, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()

        # Focus module: 3 → 32 channels, spatial /2
        self.focus = Focus(in_channels=3, out_channels=32, kernel_size=3)

        # Build sandglass stages with SimAM attention
        stages = []
        in_ch = 32
        for out_ch, num_blocks, stride, reduction in self.STAGE_CONFIG:
            stage_blocks = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                stage_blocks.append(SandglassBlock(in_ch, out_ch, stride=s, reduction=reduction))
                in_ch = out_ch
            stage_blocks.append(SimAM())
            stages.append(nn.Sequential(*stage_blocks))

        self.stages = nn.Sequential(*stages)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_ch, num_classes),
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
    running_loss = 0.0
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

            # Soft targets
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            # KD loss components
            loss_soft = F.kl_div(student_log_soft, teacher_soft,
                                 reduction="batchmean") * (temperature ** 2)
            loss_hard = ce_criterion(student_logits, labels)

            # Combined loss
            loss = alpha * loss_soft + (1.0 - alpha) * loss_hard

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


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

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    total_train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        if use_kd:
            train_loss, train_acc = train_one_epoch_kd(
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

        # Check for best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        # Logging
        best_marker = " ⭐ BEST" if is_best else ""
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
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"])
        for i in range(len(history["epoch"])):
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
# ## 4. Experiment 2 — Train Focus-RCNet Baseline

# %%
# ============================================================
# 4.1 — Experiment 2: Focus-RCNet Baseline (CE Training)
# ============================================================

print("\n" + "🔬" * 30)
print("  EXPERIMENT 2 — Focus-RCNet Baseline")
print("🔬" * 30 + "\n")

# Reset seed for reproducibility
seed_everything(cfg.SEED)

# Create fresh model
exp2_model = create_focus_rcnet(cfg.NUM_CLASSES).to(device)

exp2_params = sum(p.numel() for p in exp2_model.parameters())
print(f"📋 Model: {cfg.STUDENT_MODEL_NAME}")
print(f"   Params: {exp2_params:,}")
print(f"   Training: Standard CE, {cfg.EPOCHS} epochs, LR={cfg.LR}")

# Run training
exp2_history, exp2_best_acc, exp2_best_epoch, exp2_best_state, exp2_time = run_experiment(
    model=exp2_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 2 — Focus-RCNet Baseline",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
)

# Save checkpoint and history
save_checkpoint(
    model_state=exp2_best_state,
    experiment_id=2,
    experiment_name="Experiment 2 — Focus-RCNet Baseline",
    model_name=cfg.STUDENT_MODEL_NAME,
    best_epoch=exp2_best_epoch,
    best_val_acc=exp2_best_acc,
    history=exp2_history,
    filename=cfg.EXP2_CHECKPOINT,
)
save_history(exp2_history, cfg.EXP2_HISTORY)

# Free memory
del exp2_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 5. Experiment 3 — KD From Scratch (EfficientNet-B4 → Focus-RCNet)

# %%
# ============================================================
# 5.1 — Experiment 3: KD From Scratch
# ============================================================

print("\n" + "🔬" * 30)
print("  EXPERIMENT 3 — Focus-RCNet KD From Scratch")
print("🔬" * 30 + "\n")

# Reset seed for reproducibility
seed_everything(cfg.SEED)

# Create fresh student model
exp3_model = create_focus_rcnet(cfg.NUM_CLASSES).to(device)

print(f"📋 Student: {cfg.STUDENT_MODEL_NAME}")
print(f"   Teacher: {cfg.TEACHER_MODEL_NAME} (frozen)")
print(f"   KD: T={cfg.KD_TEMPERATURE}, α={cfg.KD_ALPHA}")
print(f"   Training: KD from scratch, {cfg.EPOCHS} epochs, LR={cfg.LR}")

# Run KD training
exp3_history, exp3_best_acc, exp3_best_epoch, exp3_best_state, exp3_time = run_experiment(
    model=exp3_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 3 — Focus-RCNet KD Scratch",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)

# Save checkpoint and history
save_checkpoint(
    model_state=exp3_best_state,
    experiment_id=3,
    experiment_name="Experiment 3 — Focus-RCNet KD Scratch",
    model_name=f"{cfg.STUDENT_MODEL_NAME}-KD",
    best_epoch=exp3_best_epoch,
    best_val_acc=exp3_best_acc,
    history=exp3_history,
    filename=cfg.EXP3_CHECKPOINT,
    config_override={
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
        "kd_alpha": cfg.KD_ALPHA,
        "teacher_model": cfg.TEACHER_MODEL_NAME,
    },
)
save_history(exp3_history, cfg.EXP3_HISTORY)

# Free memory
del exp3_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 6. Experiment 4 — Two-Stage KD (EfficientNet-B4 → Focus-RCNet)
#
# **Stage 1**: Load checkpoint from Experiment 2 (skip re-training)
# **Stage 2**: Fine-tune with KD, 50 epochs, LR=0.005

# %%
# ============================================================
# 6.1 — Experiment 4: Two-Stage KD
# ============================================================

print("\n" + "🔬" * 30)
print("  EXPERIMENT 4 — Focus-RCNet Two-Stage KD")
print("🔬" * 30 + "\n")

# Reset seed
seed_everything(cfg.SEED)

# Stage 1: Load pre-trained baseline from Experiment 2
exp4_model = create_focus_rcnet(cfg.NUM_CLASSES).to(device)
exp4_model.load_state_dict(exp2_best_state)

print(f"📋 Stage 1: Loaded Experiment 2 baseline checkpoint")
print(f"   Baseline best val acc: {exp2_best_acc:.4f}")
print(f"\n📋 Stage 2: Fine-tuning with KD")
print(f"   Teacher: {cfg.TEACHER_MODEL_NAME} (frozen)")
print(f"   KD: T={cfg.KD_TEMPERATURE}, α={cfg.KD_ALPHA}")
print(f"   Epochs: {cfg.STAGE2_EPOCHS}, LR: {cfg.STAGE2_LR}")

# Stage 2: Fine-tune with KD
exp4_history, exp4_best_acc, exp4_best_epoch, exp4_best_state, exp4_time = run_experiment(
    model=exp4_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_name="Experiment 4 — Focus-RCNet Two-Stage KD (Stage 2)",
    epochs=cfg.STAGE2_EPOCHS,
    lr=cfg.STAGE2_LR,
    t_max=cfg.STAGE2_T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)

# Save checkpoint and history
save_checkpoint(
    model_state=exp4_best_state,
    experiment_id=4,
    experiment_name="Experiment 4 — Focus-RCNet Two-Stage KD",
    model_name=f"{cfg.STUDENT_MODEL_NAME}-KD-TwoStage",
    best_epoch=exp4_best_epoch,
    best_val_acc=exp4_best_acc,
    history=exp4_history,
    filename=cfg.EXP4_CHECKPOINT,
    config_override={
        "img_size": cfg.IMG_SIZE,
        "stage1_epochs": cfg.EPOCHS,
        "stage1_lr": cfg.LR,
        "stage2_epochs": cfg.STAGE2_EPOCHS,
        "stage2_lr": cfg.STAGE2_LR,
        "batch_size": cfg.BATCH_SIZE,
        "momentum": cfg.MOMENTUM,
        "weight_decay": cfg.WEIGHT_DECAY,
        "optimizer": cfg.OPTIMIZER,
        "scheduler": cfg.SCHEDULER,
        "seed": cfg.SEED,
        "train_split": cfg.TRAIN_SPLIT,
        "val_split": cfg.VAL_SPLIT,
        "use_amp": cfg.USE_AMP,
        "kd_temperature": cfg.KD_TEMPERATURE,
        "kd_alpha": cfg.KD_ALPHA,
        "teacher_model": cfg.TEACHER_MODEL_NAME,
        "stage1_checkpoint": cfg.EXP2_CHECKPOINT,
    },
)
save_history(exp4_history, cfg.EXP4_HISTORY)

# Free memory
del exp4_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 7. Phase 1 Results Summary

# %%
# ============================================================
# 7.1 — Comparison Table
# ============================================================

print("=" * 70)
print("  📊 Phase 1 Results Summary")
print("=" * 70)
print(f"{'Experiment':<40} {'Best Acc':>10} {'Best Ep':>8} {'Time':>10}")
print("-" * 70)
print(f"{'Exp 1: EfficientNet-B4 (Teacher)':<40} {'95.52%':>10} {'61':>8} {'72.1 min':>10}")
print(f"{'Exp 2: Focus-RCNet (Baseline)':<40} {f'{exp2_best_acc*100:.2f}%':>10} {exp2_best_epoch:>8} {f'{exp2_time/60:.1f} min':>10}")
print(f"{'Exp 3: Focus-RCNet KD Scratch':<40} {f'{exp3_best_acc*100:.2f}%':>10} {exp3_best_epoch:>8} {f'{exp3_time/60:.1f} min':>10}")
print(f"{'Exp 4: Focus-RCNet Two-Stage KD':<40} {f'{exp4_best_acc*100:.2f}%':>10} {exp4_best_epoch:>8} {f'{exp4_time/60:.1f} min':>10}")
print("-" * 70)
print(f"  KD Scratch improvement over baseline: {(exp3_best_acc - exp2_best_acc)*100:+.2f}%")
print(f"  Two-Stage improvement over baseline : {(exp4_best_acc - exp2_best_acc)*100:+.2f}%")
print(f"  Two-Stage vs KD Scratch             : {(exp4_best_acc - exp3_best_acc)*100:+.2f}%")
print("=" * 70)

# %%
# ============================================================
# 7.2 — Training Curves (All Experiments)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

all_experiments = [
    ("Exp 2: Baseline", exp2_history, "tab:blue"),
    ("Exp 3: KD Scratch", exp3_history, "tab:orange"),
    ("Exp 4: Two-Stage KD", exp4_history, "tab:green"),
]

# --- Row 1: Individual loss curves ---
for col_idx, (name, hist, color) in enumerate(all_experiments):
    ax = axes[0][col_idx]
    ax.plot(hist["epoch"], hist["train_loss"], label="Train Loss", linewidth=1.5, color=color)
    ax.plot(hist["epoch"], hist["val_loss"], label="Val Loss", linewidth=1.5, color=color, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

# --- Row 2: Individual accuracy curves ---
for col_idx, (name, hist, color) in enumerate(all_experiments):
    ax = axes[1][col_idx]
    ax.plot(hist["epoch"], hist["train_acc"], label="Train Acc", linewidth=1.5, color=color)
    ax.plot(hist["epoch"], hist["val_acc"], label="Val Acc", linewidth=1.5, color=color, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} — Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Phase 1 — Focus-RCNet Experiments Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curves_phase1.png"), dpi=150, bbox_inches="tight")
plt.show()
print("📊 Training curves saved: training_curves_phase1.png")

# %%
# ============================================================
# 7.3 — Overlaid Validation Accuracy Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for name, hist, color in all_experiments:
    ax.plot(hist["epoch"], hist["val_acc"], label=name, linewidth=2, color=color)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Accuracy", fontsize=12)
ax.set_title("Phase 1 — Validation Accuracy Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "val_accuracy_comparison_phase1.png"), dpi=150, bbox_inches="tight")
plt.show()
print("📊 Comparison plot saved: val_accuracy_comparison_phase1.png")

# %% [markdown]
# ## 8. Verify All Checkpoints

# %%
# ============================================================
# 8.1 — Verify All Saved Checkpoints
# ============================================================

print("=" * 60)
print("  🔍 Checkpoint Verification")
print("=" * 60)

checkpoints_to_verify = [
    (cfg.EXP2_CHECKPOINT, "Experiment 2 — Baseline"),
    (cfg.EXP3_CHECKPOINT, "Experiment 3 — KD Scratch"),
    (cfg.EXP4_CHECKPOINT, "Experiment 4 — Two-Stage KD"),
]

for ckpt_name, exp_name in checkpoints_to_verify:
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, ckpt_name)
    loaded = torch.load(ckpt_path, map_location="cpu")

    verify_model = create_focus_rcnet(cfg.NUM_CLASSES)
    verify_model.load_state_dict(loaded["model_state_dict"])

    print(f"  ✅ {exp_name}")
    print(f"     File: {ckpt_name}")
    print(f"     Best epoch: {loaded['best_epoch']}")
    print(f"     Best val acc: {loaded['best_val_acc']:.4f}")

    del loaded, verify_model

torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("=" * 60)

# %% [markdown]
# ## Output Artifacts
#
# | Artifact | Description |
# |----------|-------------|
# | `focus_rcnet_baseline_best.pth` | Experiment 2: Focus-RCNet baseline checkpoint |
# | `focus_rcnet_kd_scratch_best.pth` | Experiment 3: Focus-RCNet KD from scratch |
# | `focus_rcnet_kd_twostage_best.pth` | Experiment 4: Focus-RCNet two-stage KD |
# | `training_history_exp2.csv` | Per-epoch metrics for Experiment 2 |
# | `training_history_exp3.csv` | Per-epoch metrics for Experiment 3 |
# | `training_history_exp4.csv` | Per-epoch metrics for Experiment 4 |
# | `training_curves_phase1.png` | Training curves for all Phase 1 experiments |
# | `val_accuracy_comparison_phase1.png` | Overlaid validation accuracy comparison |
#
# **Next Step:** Use all checkpoints in Notebook 3 (Phase 2: EfficientNet-Lite0
# experiments + Final Evaluation across all 7 experiments).

# %%
print("\n" + "=" * 60)
print("  🎉 Notebook 2 Complete — Phase 1 Focus-RCNet Experiments Done!")
print("=" * 60)
print(f"  Exp 2 (Baseline)    : {exp2_best_acc:.4f} ({exp2_best_acc*100:.2f}%)")
print(f"  Exp 3 (KD Scratch)  : {exp3_best_acc:.4f} ({exp3_best_acc*100:.2f}%)")
print(f"  Exp 4 (Two-Stage KD): {exp4_best_acc:.4f} ({exp4_best_acc*100:.2f}%)")
print(f"  Next: Notebook 3 — Phase 2 EfficientNet-Lite0 + Final Evaluation")
print("=" * 60)
