#!/usr/bin/env python
# =============================================================================
# Notebook 1 — Teacher Model Training (EfficientNet-B4)
# =============================================================================
# Experiment 1: Train EfficientNet-B4 as Teacher Model
#
# This notebook trains the EfficientNet-B4 teacher model on TrashNet dataset.
# The trained teacher will be used in Notebook 2 (Phase 1) and Notebook 3 (Phase 2)
# for knowledge distillation experiments.
#
# Protocol: Phase 1 configuration
# - IMG_SIZE: 380x380
# - LR: 0.05
# - EPOCHS: 100
# - BATCH_SIZE: 8
# - Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - Loss: CrossEntropyLoss
# - AMP: Enabled
# =============================================================================

# %% [markdown]
# # Notebook 1 — Teacher Model Training (EfficientNet-B4)
#
# **Experiment 1** dari 7 eksperimen total.
#
# Notebook ini melatih EfficientNet-B4 sebagai teacher model pada dataset TrashNet.
# Checkpoint yang dihasilkan akan digunakan oleh Notebook 2 dan Notebook 3
# untuk knowledge distillation.

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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    EXPERIMENT_NAME = "Experiment 1 — EfficientNet-B4 Teacher"
    EXPERIMENT_ID = 1
    NOTEBOOK_ID = 1

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

    # ----- Training -----
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
    LOSS = "CrossEntropyLoss"

    # ----- Model -----
    MODEL_NAME = "EfficientNet-B4"
    MODEL_TIMM_NAME = "efficientnet_b4"
    PRETRAINED = True

    # ----- Checkpoint -----
    CHECKPOINT_METRIC = "best_val_accuracy"
    EARLY_STOPPING = False
    CHECKPOINT_FILENAME = "efficientnet_b4_teacher_best.pth"
    HISTORY_FILENAME = "training_history_exp1.csv"

    # ----- Paths -----
    # Update DATASET_PATH to match your Kaggle input directory
    DATASET_PATH = "/kaggle/input/trashnet/dataset-resized/dataset-resized"
    OUTPUT_DIR = "/kaggle/working"


cfg = Config()

# Print configuration summary
print("=" * 60)
print(f"  {cfg.EXPERIMENT_NAME}")
print("=" * 60)
print(f"  Model        : {cfg.MODEL_NAME} (timm: {cfg.MODEL_TIMM_NAME})")
print(f"  Dataset      : {cfg.DATASET_NAME} ({cfg.NUM_CLASSES} classes)")
print(f"  Image Size   : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"  Epochs       : {cfg.EPOCHS}")
print(f"  Batch Size   : {cfg.BATCH_SIZE}")
print(f"  Optimizer    : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM})")
print(f"  Scheduler    : {cfg.SCHEDULER} (T_max={cfg.T_MAX})")
print(f"  Loss         : {cfg.LOSS}")
print(f"  AMP          : {cfg.USE_AMP}")
print(f"  Seed         : {cfg.SEED}")
print(f"  Split        : Train {cfg.TRAIN_SPLIT} / Val {cfg.VAL_SPLIT}")
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
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
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

# Load full dataset (without transforms — we apply via wrapper)
full_dataset = datasets.ImageFolder(
    root=cfg.DATASET_PATH,
    transform=None,  # Raw PIL images
)

# Verify classes
print(f"📁 Dataset loaded from: {cfg.DATASET_PATH}")
print(f"   Total images : {len(full_dataset)}")
print(f"   Classes found: {full_dataset.classes}")
print(f"   Class to idx : {full_dataset.class_to_idx}")
assert len(full_dataset.classes) == cfg.NUM_CLASSES, \
    f"Expected {cfg.NUM_CLASSES} classes, found {len(full_dataset.classes)}"

# Stratified split using fixed seed
from sklearn.model_selection import train_test_split

targets = [s[1] for s in full_dataset.samples]
indices = list(range(len(full_dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=cfg.VAL_SPLIT,
    random_state=cfg.SEED,
    stratify=targets,
)

print(f"\n📊 Split (seed={cfg.SEED}):")
print(f"   Train : {len(train_indices)} images ({len(train_indices)/len(full_dataset)*100:.1f}%)")
print(f"   Val   : {len(val_indices)} images ({len(val_indices)/len(full_dataset)*100:.1f}%)")

# Create subsets
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# Wrap with Albumentations
train_dataset = AlbumentationsDataset(train_subset, transform=train_transform)
val_dataset = AlbumentationsDataset(val_subset, transform=val_transform)

# %%
# ============================================================
# 1.4 — DataLoaders
# ============================================================

# Worker seed function for reproducibility
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
# ## 2. Model Definition

# %%
# ============================================================
# 2.1 — Create EfficientNet-B4 Model
# ============================================================

def create_teacher_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Create EfficientNet-B4 teacher model using timm.

    Args:
        model_name: timm model identifier
        num_classes: number of output classes
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        model: nn.Module ready for training
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


model = create_teacher_model(
    model_name=cfg.MODEL_TIMM_NAME,
    num_classes=cfg.NUM_CLASSES,
    pretrained=cfg.PRETRAINED,
)
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created: {cfg.MODEL_NAME}")
print(f"   Source       : timm ({cfg.MODEL_TIMM_NAME})")
print(f"   Pretrained   : {cfg.PRETRAINED} (ImageNet)")
print(f"   Total params : {total_params:,}")
print(f"   Trainable    : {trainable_params:,}")
print(f"   Output classes: {cfg.NUM_CLASSES}")

# %% [markdown]
# ## 3. Training Loop

# %%
# ============================================================
# 3.1 — Loss, Optimizer, Scheduler
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=cfg.LR,
    momentum=cfg.MOMENTUM,
    weight_decay=cfg.WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.T_MAX,
)

# AMP GradScaler
scaler = GradScaler(enabled=cfg.USE_AMP)

print(f"✅ Training components initialized")
print(f"   Loss      : {cfg.LOSS}")
print(f"   Optimizer : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM}, wd={cfg.WEIGHT_DECAY})")
print(f"   Scheduler : {cfg.SCHEDULER} (T_max={cfg.T_MAX})")
print(f"   AMP       : {cfg.USE_AMP}")

# %%
# ============================================================
# 3.2 — Training & Validation Functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# %%
# ============================================================
# 3.3 — Main Training Loop
# ============================================================

print("=" * 60)
print(f"  🚀 Starting Training: {cfg.EXPERIMENT_NAME}")
print(f"  Epochs: {cfg.EPOCHS} | Batch Size: {cfg.BATCH_SIZE}")
print("=" * 60)

# Training history
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
    "epoch_time": [],
}

best_val_acc = 0.0
best_epoch = 0
best_model_state = None

total_train_start = time.time()

for epoch in range(1, cfg.EPOCHS + 1):
    epoch_start = time.time()
    current_lr = optimizer.param_groups[0]["lr"]

    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, cfg.USE_AMP
    )

    # Validate
    val_loss, val_acc = validate(
        model, val_loader, criterion, device, cfg.USE_AMP
    )

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

    # Check for best model (best_val_accuracy strategy)
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())

    # Logging
    best_marker = " ⭐ BEST" if is_best else ""
    print(
        f"Epoch [{epoch:3d}/{cfg.EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
        f"| LR: {current_lr:.6f} "
        f"| Time: {epoch_time:.1f}s{best_marker}"
    )

total_train_time = time.time() - total_train_start

print("=" * 60)
print(f"  ✅ Training Complete!")
print(f"  Total time  : {total_train_time/60:.1f} minutes")
print(f"  Best epoch  : {best_epoch}")
print(f"  Best val acc: {best_val_acc:.4f}")
print("=" * 60)

# %% [markdown]
# ## 4. Training Results

# %%
# ============================================================
# 4.1 — Training Curves
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=1.5)
axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=1.5)
axes[0].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Epoch ({best_epoch})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", linewidth=1.5)
axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", linewidth=1.5)
axes[1].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Epoch ({best_epoch})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curves")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate schedule
axes[2].plot(history["epoch"], history["lr"], label="Learning Rate", linewidth=1.5, color="green")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule (CosineAnnealing)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"{cfg.EXPERIMENT_NAME} — Training Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curves_exp1.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"📊 Training curves saved")

# %%
# ============================================================
# 4.2 — Best Model Metrics Summary
# ============================================================

print("=" * 60)
print(f"  📋 Training Summary — {cfg.EXPERIMENT_NAME}")
print("=" * 60)
print(f"  Model          : {cfg.MODEL_NAME}")
print(f"  Total Epochs   : {cfg.EPOCHS}")
print(f"  Best Epoch     : {best_epoch}")
print(f"  Best Val Acc   : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"  Final Train Acc: {history['train_acc'][-1]:.4f}")
print(f"  Final Val Acc  : {history['val_acc'][-1]:.4f}")
print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"  Final Val Loss  : {history['val_loss'][-1]:.4f}")
print(f"  Total Time     : {total_train_time/60:.1f} minutes")
print(f"  Avg Epoch Time : {np.mean(history['epoch_time']):.1f}s")
print(f"  Total Params   : {total_params:,}")
print("=" * 60)

# %% [markdown]
# ## 5. Save Checkpoint

# %%
# ============================================================
# 5.1 — Save Best Model Checkpoint
# ============================================================

checkpoint = {
    # Model state
    "model_state_dict": best_model_state,

    # Training metadata
    "experiment_id": cfg.EXPERIMENT_ID,
    "experiment_name": cfg.EXPERIMENT_NAME,
    "model_name": cfg.MODEL_NAME,
    "model_timm_name": cfg.MODEL_TIMM_NAME,
    "num_classes": cfg.NUM_CLASSES,
    "class_names": cfg.CLASS_NAMES,

    # Best metrics
    "best_epoch": best_epoch,
    "best_val_acc": best_val_acc,

    # Configuration for reproducibility
    "config": {
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
        "pretrained": cfg.PRETRAINED,
        "use_amp": cfg.USE_AMP,
    },

    # Dataset info
    "dataset_name": cfg.DATASET_NAME,
    "train_size": len(train_indices),
    "val_size": len(val_indices),
    "train_indices": train_indices,
    "val_indices": val_indices,
}

checkpoint_path = os.path.join(cfg.OUTPUT_DIR, cfg.CHECKPOINT_FILENAME)
torch.save(checkpoint, checkpoint_path)
print(f"✅ Best model checkpoint saved: {checkpoint_path}")
print(f"   File size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")

# %%
# ============================================================
# 5.2 — Save Training History CSV
# ============================================================

history_path = os.path.join(cfg.OUTPUT_DIR, cfg.HISTORY_FILENAME)

with open(history_path, "w", newline="") as f:
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

print(f"✅ Training history saved: {history_path}")

# %%
# ============================================================
# 5.3 — Verify Saved Checkpoint
# ============================================================

# Quick verification: load checkpoint and check model loads correctly
loaded_ckpt = torch.load(checkpoint_path, map_location="cpu")
verify_model = create_teacher_model(cfg.MODEL_TIMM_NAME, cfg.NUM_CLASSES, pretrained=False)
verify_model.load_state_dict(loaded_ckpt["model_state_dict"])

print(f"✅ Checkpoint verification passed")
print(f"   Model loads correctly from: {cfg.CHECKPOINT_FILENAME}")
print(f"   Best epoch: {loaded_ckpt['best_epoch']}")
print(f"   Best val acc: {loaded_ckpt['best_val_acc']:.4f}")

del loaded_ckpt, verify_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## Output Artifacts
#
# | Artifact | Description |
# |----------|-------------|
# | `efficientnet_b4_teacher_best.pth` | Best model checkpoint (by val accuracy) |
# | `training_history_exp1.csv` | Per-epoch training metrics |
# | `training_curves_exp1.png` | Loss, accuracy, and LR curves |
#
# **Next Step:** Use the teacher checkpoint in Notebook 2 (Phase 1: Focus-RCNet experiments)
# and Notebook 3 (Phase 2: EfficientNet-Lite0 experiments).

# %%
print("\n" + "=" * 60)
print("  🎉 Notebook 1 Complete — Teacher Model Training Done!")
print("=" * 60)
print(f"  Checkpoint : {cfg.CHECKPOINT_FILENAME}")
print(f"  History    : {cfg.HISTORY_FILENAME}")
print(f"  Best Acc   : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"  Next       : Notebook 2 — Phase 1 Focus-RCNet Experiments")
print("=" * 60)
