#!/usr/bin/env python
# =============================================================================
# Notebook 3B - Conditional Follow-Up: EfficientNet-Lite0 Native Profile
# =============================================================================
# Native A: EfficientNet-Lite0 baseline at 224x224 with ImageNet initialization
# Native B: Direct KD training from ImageNet initialization
# Native C: Two-Stage KD (load Native A -> fine-tune with KD)
#
# Follow-up protocol:
# - Student input: 224x224 (EfficientNet-Lite0 native resolution)
# - Teacher input during KD: 380x380 (same resolution as Notebook 1 teacher)
# - Student pretrained weights: ImageNet
# - LR: 0.01
# - EPOCHS: 100
# - BATCH_SIZE: 8
# - Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - KD: T=4, alpha=0.5
# - Two-Stage: Stage2 epochs=50, lr=0.001
#
# This file is intentionally separate from Notebook 3. Its artifacts and result
# table must not replace the controlled-comparison outputs from Experiments 5/6/7.
# =============================================================================

# %% [markdown]
# # Notebook 3B - EfficientNet-Lite0 Native Profile
#
# This optional deployment-oriented follow-up evaluates Lite0 under its native
# input resolution and ImageNet initialization. During KD, the teacher retains
# its original 380x380 input while the Lite0 student receives a 224x224 view of
# the same augmented image.

# %%
# ============================================================
# 0.1 - Imports
# ============================================================
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
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("All imports successful")

# %%
# ============================================================
# 0.2 - Configuration
# ============================================================

class Config:
    """Centralized optional Native-profile configuration."""

    NOTEBOOK_ID = "3B"
    PHASE = "2-native-followup"

    PLATFORM = "Kaggle Notebooks"
    FRAMEWORK = "PyTorch"
    USE_AMP = True
    SEED = 42

    DATASET_NAME = "TrashNet"
    NUM_CLASSES = 6
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3

    # Student deployment input and teacher training-time KD input.
    STUDENT_IMG_SIZE = 224
    TEACHER_IMG_SIZE = 380

    EPOCHS = 100
    BATCH_SIZE = 8

    OPTIMIZER = "SGD"
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = EPOCHS

    KD_TEMPERATURE = 4
    KD_ALPHA = 0.5

    STAGE2_EPOCHS = 50
    STAGE2_LR = 0.001
    STAGE2_FREEZE_LAYERS = False
    STAGE2_T_MAX = STAGE2_EPOCHS

    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"
    STUDENT_MODEL_NAME = "EfficientNet-Lite0 Native"
    STUDENT_PRETRAINED = True
    STUDENT_WEIGHTS = "ImageNet"
    STUDENT_TIMM_CANDIDATES = (
        "tf_efficientnet_lite0",
        "tf_efficientnet_lite0.in1k",
        "efficientnet_lite0",
    )

    NATIVE_BASELINE_CHECKPOINT = "efficientnet_lite0_native_baseline_best.pth"
    NATIVE_KD_CHECKPOINT = "efficientnet_lite0_native_kd_scratch_best.pth"
    NATIVE_TWOSTAGE_CHECKPOINT = "efficientnet_lite0_native_kd_twostage_best.pth"
    NATIVE_BASELINE_HISTORY = "training_history_lite0_native_baseline.csv"
    NATIVE_KD_HISTORY = "training_history_lite0_native_direct_kd.csv"
    NATIVE_TWOSTAGE_HISTORY = "training_history_lite0_native_kd_twostage.csv"
    NATIVE_COMPARISON_CSV = "lite0_native_comparison_table.csv"

    TEACHER_CHECKPOINT = "efficientnet_b4_teacher_best.pth"

    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    TEACHER_INPUT_DIR = "/kaggle/input/notebooks/hamzapratama/notebook1-teacher-training"
    OUTPUT_DIR = "/kaggle/working/"

    INFERENCE_WARMUP_RUNS = 10
    INFERENCE_MEASURE_RUNS = 50


cfg = Config()

print("=" * 76)
print("  Notebook 3B - EfficientNet-Lite0 Native Profile")
print("=" * 76)
print(f"  Teacher        : {cfg.TEACHER_MODEL_NAME} @ {cfg.TEACHER_IMG_SIZE}x{cfg.TEACHER_IMG_SIZE}")
print(f"  Student        : {cfg.STUDENT_MODEL_NAME} @ {cfg.STUDENT_IMG_SIZE}x{cfg.STUDENT_IMG_SIZE}")
print(f"  Student weights: pretrained={cfg.STUDENT_PRETRAINED} ({cfg.STUDENT_WEIGHTS})")
print(f"  Dataset        : {cfg.DATASET_NAME} ({cfg.NUM_CLASSES} classes)")
print(f"  Epochs         : {cfg.EPOCHS} (Stage2: {cfg.STAGE2_EPOCHS})")
print(f"  Batch Size     : {cfg.BATCH_SIZE}")
print(f"  Optimizer      : {cfg.OPTIMIZER} (lr={cfg.LR}, momentum={cfg.MOMENTUM})")
print(f"  KD             : T={cfg.KD_TEMPERATURE}, alpha={cfg.KD_ALPHA}")
print(f"  AMP            : {cfg.USE_AMP}")
print(f"  Seed           : {cfg.SEED}")
print("=" * 76)

# %%
# ============================================================
# 0.3 - Seed, Device, AMP, and Kaggle Input Resolver
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
            "Attach the Notebook 1 output as a Kaggle input."
        )
    if len(matches) > 1:
        print(f"Warning: multiple matches found for {filename}; using {matches[0]}")
    return str(matches[0])


def amp_autocast(enabled: bool):
    """Use the current torch.amp API while disabling AMP cleanly on CPU."""
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def create_grad_scaler(enabled: bool):
    """Create a CUDA scaler using the non-deprecated torch.amp API."""
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


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
# ## 1. Dataset and Dual-Resolution DataLoaders

# %%
# ============================================================
# 1.1 - Shared Augmentation and View Transforms
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Apply stochastic augmentation once before branching into teacher and student
# views. This keeps KD pairs semantically aligned while preserving each model's
# intended input resolution.
shared_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(0.05, 0.2),
        hole_width_range=(0.05, 0.2),
        fill=0,
        p=0.5,
    ),
])

student_train_view = A.Compose([
    A.Resize(cfg.STUDENT_IMG_SIZE, cfg.STUDENT_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

teacher_train_view = A.Compose([
    A.Resize(cfg.TEACHER_IMG_SIZE, cfg.TEACHER_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

student_val_transform = A.Compose([
    A.Resize(cfg.STUDENT_IMG_SIZE, cfg.STUDENT_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

teacher_val_transform = A.Compose([
    A.Resize(cfg.TEACHER_IMG_SIZE, cfg.TEACHER_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

print("Augmentation pipelines created")
print("   Shared train: HFlip -> VFlip -> BrightnessContrast -> CoarseDropout")
print(f"   Student view : Resize {cfg.STUDENT_IMG_SIZE} -> Normalize")
print(f"   Teacher view : Resize {cfg.TEACHER_IMG_SIZE} -> Normalize (KD only)")

# %%
# ============================================================
# 1.2 - Dataset Wrappers
# ============================================================

class SingleViewDataset(torch.utils.data.Dataset):
    """Return one transformed image view for CE training or validation."""

    def __init__(self, dataset, view_transform, shared_transform=None):
        self.dataset = dataset
        self.shared_transform = shared_transform
        self.view_transform = view_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.shared_transform:
            image = self.shared_transform(image=image)["image"]
        image = self.view_transform(image=image)["image"]
        return image, label


class DualResolutionDataset(torch.utils.data.Dataset):
    """Return aligned student-224 and teacher-380 views for KD training."""

    def __init__(self, dataset, shared_transform, student_transform, teacher_transform):
        self.dataset = dataset
        self.shared_transform = shared_transform
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        shared_image = self.shared_transform(image=image)["image"]
        student_image = self.student_transform(image=shared_image)["image"]
        teacher_image = self.teacher_transform(image=shared_image)["image"]
        return student_image, teacher_image, label

# %%
# ============================================================
# 1.3 - Load Dataset and Reuse Exact Teacher Split
# ============================================================

teacher_ckpt = torch.load(teacher_checkpoint_path, map_location="cpu")
train_indices = teacher_ckpt["train_indices"]
val_indices = teacher_ckpt["val_indices"]
print("Loaded split indices from teacher checkpoint")
print(f"   Teacher recorded best val acc: {teacher_ckpt['best_val_acc']:.4f}")

full_dataset = datasets.ImageFolder(root=cfg.DATASET_PATH, transform=None)
print(f"Dataset loaded from: {cfg.DATASET_PATH}")
print(f"   Total images : {len(full_dataset)}")
print(f"   Classes found: {full_dataset.classes}")
assert full_dataset.classes == cfg.CLASS_NAMES, (
    f"Expected classes {cfg.CLASS_NAMES}, found {full_dataset.classes}"
)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

train_student_dataset = SingleViewDataset(
    train_subset,
    shared_transform=shared_train_transform,
    view_transform=student_train_view,
)
train_kd_dataset = DualResolutionDataset(
    train_subset,
    shared_transform=shared_train_transform,
    student_transform=student_train_view,
    teacher_transform=teacher_train_view,
)
val_student_dataset = SingleViewDataset(val_subset, view_transform=student_val_transform)
val_teacher_dataset = SingleViewDataset(val_subset, view_transform=teacher_val_transform)

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


def make_loader(dataset, shuffle: bool):
    generator = torch.Generator()
    generator.manual_seed(cfg.SEED)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=False,
    )


val_student_loader = make_loader(val_student_dataset, shuffle=False)
val_teacher_loader = make_loader(val_teacher_dataset, shuffle=False)

print("DataLoaders ready")
print(f"   Train batches: {len(make_loader(train_student_dataset, shuffle=True))}")
print(f"   Val batches  : {len(val_student_loader)}")

# %%
# ============================================================
# 1.5 - Verify Class Distribution and Dual Views
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

_sample_student, _sample_teacher, _sample_label = train_kd_dataset[0]
print("Dual-resolution KD sample verified")
print(f"   Student shape: {list(_sample_student.shape)}")
print(f"   Teacher shape: {list(_sample_teacher.shape)}")
print(f"   Label        : {_sample_label}")
assert _sample_student.shape == (3, cfg.STUDENT_IMG_SIZE, cfg.STUDENT_IMG_SIZE)
assert _sample_teacher.shape == (3, cfg.TEACHER_IMG_SIZE, cfg.TEACHER_IMG_SIZE)
del _sample_student, _sample_teacher, _sample_label

# %% [markdown]
# ## 2. Frozen Teacher and Pretrained Native Student

# %%
# ============================================================
# 2.1 - Load Frozen EfficientNet-B4 Teacher
# ============================================================

def create_teacher_model(model_name: str, num_classes: int, pretrained: bool = False):
    """Create the EfficientNet-B4 teacher structure using timm."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def verify_teacher_frozen(model) -> None:
    """Fail fast if KD accidentally mutates the teacher setup."""
    assert not model.training, "Teacher must remain in eval mode"
    assert all(not parameter.requires_grad for parameter in model.parameters()), (
        "Every teacher parameter must have requires_grad=False"
    )
    assert all(parameter.grad is None for parameter in model.parameters()), (
        "Frozen teacher parameters must not accumulate gradients"
    )


teacher_model = create_teacher_model(cfg.TEACHER_TIMM_NAME, cfg.NUM_CLASSES, pretrained=False)
teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
teacher_model = teacher_model.to(device)
teacher_model.eval()
for parameter in teacher_model.parameters():
    parameter.requires_grad = False
verify_teacher_frozen(teacher_model)

teacher_params = sum(parameter.numel() for parameter in teacher_model.parameters())
print(f"Teacher loaded: {cfg.TEACHER_MODEL_NAME}")
print(f"   Parameters  : {teacher_params:,}")
print(f"   Best epoch  : {teacher_ckpt['best_epoch']}")
print(f"   Recorded acc: {teacher_ckpt['best_val_acc']:.4f}")
print(f"   KD input    : {cfg.TEACHER_IMG_SIZE}x{cfg.TEACHER_IMG_SIZE}")
print("   Mode        : eval (frozen)")

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
    """Create an EfficientNet-Lite0 student model."""
    return timm.create_model(student_timm_name, pretrained=pretrained, num_classes=num_classes)


# Creating the first pretrained model may download ImageNet weights. Enable
# Kaggle Internet or attach the corresponding timm weights if this cache is cold.
_test_lite0 = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED)
_lite0_params = sum(parameter.numel() for parameter in _test_lite0.parameters())
_test_input = torch.randn(1, 3, cfg.STUDENT_IMG_SIZE, cfg.STUDENT_IMG_SIZE)
_test_output = _test_lite0(_test_input)
print("EfficientNet-Lite0 Native architecture verified")
print(f"   timm registry name: {student_timm_name}")
print(f"   Parameters        : {_lite0_params:,}")
print(f"   Pretrained        : {cfg.STUDENT_PRETRAINED} ({cfg.STUDENT_WEIGHTS})")
print(f"   Input shape       : {list(_test_input.shape)}")
print(f"   Output shape      : {list(_test_output.shape)}")
assert _test_output.shape == (1, cfg.NUM_CLASSES), (
    f"Expected output shape (1, {cfg.NUM_CLASSES}), got {_test_output.shape}"
)
del _test_lite0, _test_input, _test_output

# %% [markdown]
# ## 3. Training Utilities and Teacher Preflight

# %%
# ============================================================
# 3.1 - CE, KD, Validation, Runner, and Save Helpers
# ============================================================

def train_one_epoch_ce(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train one student epoch using standard cross entropy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for student_images, labels in loader:
        student_images = student_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with amp_autocast(use_amp):
            outputs = model(student_images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * student_images.size(0)
        predicted = outputs.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_one_epoch_kd(student, teacher, loader, optimizer, scaler, device, use_amp, temperature, alpha):
    """Train student-224 with CE plus KL soft targets from frozen teacher-380."""
    student.train()
    teacher.eval()
    running_loss_total = 0.0
    running_loss_soft = 0.0
    running_loss_hard = 0.0
    correct = 0
    total = 0
    ce_criterion = nn.CrossEntropyLoss()

    for student_images, teacher_images, labels in loader:
        student_images = student_images.to(device, non_blocking=True)
        teacher_images = teacher_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with amp_autocast(use_amp):
            student_logits = student(student_images)
            with torch.no_grad():
                teacher_logits = teacher(teacher_images)

        # Keep soft-target operations in FP32 for numerical stability.
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
        running_loss_total += loss.item() * student_images.size(0)
        running_loss_soft += loss_soft.item() * student_images.size(0)
        running_loss_hard += loss_hard.item() * student_images.size(0)
        predicted = student_logits.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    verify_teacher_frozen(teacher)
    return (
        running_loss_total / total,
        correct / total,
        running_loss_soft / total,
        running_loss_hard / total,
    )


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    """Validate one model with a single-resolution loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_autocast(use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        predicted = outputs.max(1).indices
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def run_experiment(model, train_loader, val_loader, device, experiment_name, epochs, lr, t_max,
                   use_amp=True, teacher_model=None, kd_temperature=None, kd_alpha=None):
    """Run one complete CE or dual-resolution KD experiment."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scaler = create_grad_scaler(use_amp)
    use_kd = teacher_model is not None

    print("=" * 76)
    print(f"Starting training: {experiment_name}")
    print(f"   Mode: {'KD teacher-380 -> student-224' if use_kd else 'CE student-224'}")
    print(f"   Epochs: {epochs} | LR: {lr}")
    if use_kd:
        print(f"   KD: T={kd_temperature}, alpha={kd_alpha}")
    print("=" * 76)

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
    print("=" * 76)
    print(f"Training complete: {experiment_name}")
    print(f"   Total time  : {total_train_time / 60:.1f} minutes")
    print(f"   Best epoch  : {best_epoch}")
    print(f"   Best val acc: {best_val_acc:.4f}")
    print("=" * 76)
    return history, best_val_acc, best_epoch, best_model_state, total_train_time


def native_experiment_config(**overrides):
    """Return Native-profile metadata for saved checkpoints."""
    config = {
        "track": "conditional_lite0_native_profile",
        "student_img_size": cfg.STUDENT_IMG_SIZE,
        "teacher_kd_img_size": cfg.TEACHER_IMG_SIZE,
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
        "student_weights": cfg.STUDENT_WEIGHTS,
    }
    config.update(overrides)
    return config


def save_checkpoint(model_state, experiment_name, model_name, best_epoch, best_val_acc, filename, config_override=None):
    """Save a Native-profile checkpoint without overwriting core artifacts."""
    checkpoint = {
        "model_state_dict": model_state,
        "track": "conditional_lite0_native_profile",
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
    """Save per-epoch metrics; KD files include total, soft, and hard losses."""
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
            values = [history["epoch"][idx]]
            if is_kd_history:
                values.extend([
                    f"{history['train_loss'][idx]:.6f}",
                    f"{history['train_loss_soft'][idx]:.6f}",
                    f"{history['train_loss_hard'][idx]:.6f}",
                ])
            else:
                values.append(f"{history['train_loss'][idx]:.6f}")
            values.extend([
                f"{history['train_acc'][idx]:.6f}",
                f"{history['val_loss'][idx]:.6f}",
                f"{history['val_acc'][idx]:.6f}",
                f"{history['lr'][idx]:.8f}",
                f"{history['epoch_time'][idx]:.2f}",
            ])
            writer.writerow(values)
    print(f"History saved: {path}")
    return path

# %%
# ============================================================
# 3.2 - Teacher 380x380 Preflight
# ============================================================

teacher_val_loss, teacher_val_acc_380 = validate(
    teacher_model,
    val_teacher_loader,
    nn.CrossEntropyLoss(),
    device,
    cfg.USE_AMP,
)
verify_teacher_frozen(teacher_model)
print("Teacher 380x380 preflight complete")
print(f"   Recorded checkpoint acc: {teacher_ckpt['best_val_acc']:.4f}")
print(f"   Re-evaluated val acc   : {teacher_val_acc_380:.4f}")
print(f"   Re-evaluated val loss  : {teacher_val_loss:.4f}")
assert abs(teacher_val_acc_380 - teacher_ckpt["best_val_acc"]) < 0.02, (
    "Teacher 380x380 preflight differs materially from the checkpoint metric. "
    "Check dataset attachment and preprocessing before training Native KD."
)
del teacher_ckpt
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Native A - Lite0 Baseline

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 76)
print("NATIVE A - EfficientNet-Lite0 Baseline (224x224, ImageNet init)")
print("=" * 76)
native_a_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED).to(device)
native_a_train_loader = make_loader(train_student_dataset, shuffle=True)

native_a_history, native_a_best_acc, native_a_best_epoch, native_a_best_state, native_a_time = run_experiment(
    model=native_a_model,
    train_loader=native_a_train_loader,
    val_loader=val_student_loader,
    device=device,
    experiment_name="Native A - EfficientNet-Lite0 Baseline",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
)
save_checkpoint(
    native_a_best_state,
    "Native A - EfficientNet-Lite0 Baseline",
    cfg.STUDENT_MODEL_NAME,
    native_a_best_epoch,
    native_a_best_acc,
    cfg.NATIVE_BASELINE_CHECKPOINT,
    native_experiment_config(),
)
save_history(native_a_history, cfg.NATIVE_BASELINE_HISTORY)
del native_a_model, native_a_train_loader
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Native B - Lite0 KD From Pretrained Initialization

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 76)
print("NATIVE B - EfficientNet-Lite0 KD (teacher-380 -> student-224)")
print("=" * 76)
native_b_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=cfg.STUDENT_PRETRAINED).to(device)
native_b_train_loader = make_loader(train_kd_dataset, shuffle=True)

native_b_history, native_b_best_acc, native_b_best_epoch, native_b_best_state, native_b_time = run_experiment(
    model=native_b_model,
    train_loader=native_b_train_loader,
    val_loader=val_student_loader,
    device=device,
    experiment_name="Native B - EfficientNet-Lite0 Direct KD",
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    t_max=cfg.T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)
verify_teacher_frozen(teacher_model)
save_checkpoint(
    native_b_best_state,
    "Native B - EfficientNet-Lite0 Direct KD",
    f"{cfg.STUDENT_MODEL_NAME}-KD",
    native_b_best_epoch,
    native_b_best_acc,
    cfg.NATIVE_KD_CHECKPOINT,
    native_experiment_config(
        kd_temperature=cfg.KD_TEMPERATURE,
        kd_alpha=cfg.KD_ALPHA,
        teacher_model=cfg.TEACHER_MODEL_NAME,
    ),
)
save_history(native_b_history, cfg.NATIVE_KD_HISTORY)
del native_b_model, native_b_train_loader
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Native C - Lite0 Two-Stage KD

# %%
seed_everything(cfg.SEED)
print("\n" + "=" * 76)
print("NATIVE C - EfficientNet-Lite0 Two-Stage KD")
print("=" * 76)
native_c_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=False).to(device)
native_c_model.load_state_dict(native_a_best_state)
print(f"Loaded Native A baseline state with val acc {native_a_best_acc:.4f}")
native_c_train_loader = make_loader(train_kd_dataset, shuffle=True)

native_c_history, native_c_best_acc, native_c_best_epoch, native_c_best_state, native_c_time = run_experiment(
    model=native_c_model,
    train_loader=native_c_train_loader,
    val_loader=val_student_loader,
    device=device,
    experiment_name="Native C - EfficientNet-Lite0 Two-Stage KD",
    epochs=cfg.STAGE2_EPOCHS,
    lr=cfg.STAGE2_LR,
    t_max=cfg.STAGE2_T_MAX,
    use_amp=cfg.USE_AMP,
    teacher_model=teacher_model,
    kd_temperature=cfg.KD_TEMPERATURE,
    kd_alpha=cfg.KD_ALPHA,
)
verify_teacher_frozen(teacher_model)
save_checkpoint(
    native_c_best_state,
    "Native C - EfficientNet-Lite0 Two-Stage KD",
    f"{cfg.STUDENT_MODEL_NAME}-KD-TwoStage",
    native_c_best_epoch,
    native_c_best_acc,
    cfg.NATIVE_TWOSTAGE_CHECKPOINT,
    native_experiment_config(
        stage1_epochs=cfg.EPOCHS,
        stage1_lr=cfg.LR,
        stage2_epochs=cfg.STAGE2_EPOCHS,
        stage2_lr=cfg.STAGE2_LR,
        kd_temperature=cfg.KD_TEMPERATURE,
        kd_alpha=cfg.KD_ALPHA,
        teacher_model=cfg.TEACHER_MODEL_NAME,
        stage1_checkpoint=cfg.NATIVE_BASELINE_CHECKPOINT,
    ),
)
save_history(native_c_history, cfg.NATIVE_TWOSTAGE_HISTORY)
del native_c_model, native_c_train_loader
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Native Results Summary and Curves

# %%
print("\n" + "=" * 80)
print("EfficientNet-Lite0 Native Profile Results")
print("=" * 80)
print(f"{'Experiment':<48} {'Best Acc':>10} {'Best Ep':>8} {'Time':>10}")
print("-" * 80)
print(f"{'Native A: Lite0 Baseline':<48} {f'{native_a_best_acc * 100:.2f}%':>10} {native_a_best_epoch:>8} {f'{native_a_time / 60:.1f} min':>10}")
print(f"{'Native B: Lite0 Direct KD':<48} {f'{native_b_best_acc * 100:.2f}%':>10} {native_b_best_epoch:>8} {f'{native_b_time / 60:.1f} min':>10}")
print(f"{'Native C: Lite0 Two-Stage KD':<48} {f'{native_c_best_acc * 100:.2f}%':>10} {native_c_best_epoch:>8} {f'{native_c_time / 60:.1f} min':>10}")
print("-" * 80)
print(f"Direct KD improvement over baseline: {(native_b_best_acc - native_a_best_acc) * 100:+.2f} pp")
print(f"Two-Stage improvement over baseline : {(native_c_best_acc - native_a_best_acc) * 100:+.2f} pp")
print(f"Two-Stage vs Direct KD              : {(native_c_best_acc - native_b_best_acc) * 100:+.2f} pp")
print("=" * 80)

native_experiments = [
    ("Native A: Baseline", native_a_history, "tab:blue"),
    ("Native B: Direct KD", native_b_history, "tab:orange"),
    ("Native C: Two-Stage KD", native_c_history, "tab:green"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for col_idx, (name, history, color) in enumerate(native_experiments):
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
plt.suptitle("EfficientNet-Lite0 Native Profile Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curves_lite0_native.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axis = plt.subplots(figsize=(10, 6))
for name, history, color in native_experiments:
    axis.plot(history["epoch"], history["val_acc"], label=name, color=color, linewidth=2)
axis.set_xlabel("Epoch")
axis.set_ylabel("Validation Accuracy")
axis.set_title("EfficientNet-Lite0 Native Validation Accuracy", fontweight="bold")
axis.legend()
axis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "val_accuracy_comparison_lite0_native.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 8. Verify Native Checkpoints

# %%
for checkpoint_name, experiment_name in [
    (cfg.NATIVE_BASELINE_CHECKPOINT, "Native A - Baseline"),
    (cfg.NATIVE_KD_CHECKPOINT, "Native B - Direct KD"),
    (cfg.NATIVE_TWOSTAGE_CHECKPOINT, "Native C - Two-Stage KD"),
]:
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_name)
    loaded = torch.load(checkpoint_path, map_location="cpu")
    verify_model = create_lite0_model(cfg.NUM_CLASSES, pretrained=False)
    verify_model.load_state_dict(loaded["model_state_dict"])
    print(f"Verified {experiment_name}: {checkpoint_name} | best={loaded['best_val_acc']:.4f}")
    del loaded, verify_model

# %% [markdown]
# ## 9. Native Deployment-Oriented Evaluation

# %%
# ============================================================
# 9.1 - Metrics and Efficiency Utilities
# ============================================================

@torch.no_grad()
def collect_predictions(model, loader, device, use_amp):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with amp_autocast(use_amp):
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
        with amp_autocast(cfg.USE_AMP):
            model(sample)
    for hook in hooks:
        hook.remove()
    return macs * 2


@torch.no_grad()
def measure_inference_profile(model, device, input_size: int, warmup_runs: int, measure_runs: int):
    """Measure batch-size-one latency and CUDA peak allocated memory."""
    sample = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    for _ in range(warmup_runs):
        with amp_autocast(cfg.USE_AMP):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(measure_runs):
        with amp_autocast(cfg.USE_AMP):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_mb = float("nan")
    latency_ms = (time.perf_counter() - start) * 1000 / measure_runs
    return latency_ms, peak_memory_mb


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


def evaluate_native_checkpoint(spec):
    checkpoint = torch.load(spec["path"], map_location="cpu")
    model = create_lite0_model(cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    y_true, y_pred, probabilities = collect_predictions(model, val_student_loader, device, cfg.USE_AMP)
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
    latency_ms, peak_memory_mb = measure_inference_profile(
        model,
        device,
        cfg.STUDENT_IMG_SIZE,
        cfg.INFERENCE_WARMUP_RUNS,
        cfg.INFERENCE_MEASURE_RUNS,
    )
    result = {
        "native_id": spec["native_id"],
        "experiment": spec["experiment"],
        "model": cfg.STUDENT_MODEL_NAME,
        "student_img_size": cfg.STUDENT_IMG_SIZE,
        "validation_accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
        "auc_macro_ovr": roc_auc_macro,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "flops": estimate_flops(model, device, cfg.STUDENT_IMG_SIZE),
        "inference_time_ms": latency_ms,
        "peak_memory_mb": peak_memory_mb,
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
# 9.2 - Evaluate Three Native Variants
# ============================================================

native_checkpoint_specs = [
    {"native_id": "A", "experiment": "Native A: Baseline", "path": os.path.join(cfg.OUTPUT_DIR, cfg.NATIVE_BASELINE_CHECKPOINT)},
    {"native_id": "B", "experiment": "Native B: Direct KD", "path": os.path.join(cfg.OUTPUT_DIR, cfg.NATIVE_KD_CHECKPOINT)},
    {"native_id": "C", "experiment": "Native C: Two-Stage KD", "path": os.path.join(cfg.OUTPUT_DIR, cfg.NATIVE_TWOSTAGE_CHECKPOINT)},
]

native_results = []
native_confusion_matrices = {}
native_roc_curves = {}
for spec in native_checkpoint_specs:
    print(f"Evaluating {spec['experiment']}")
    result, matrix, roc_data = evaluate_native_checkpoint(spec)
    native_results.append(result)
    native_confusion_matrices[spec["native_id"]] = matrix
    native_roc_curves[spec["native_id"]] = roc_data
    print(
        f"   Acc={result['validation_accuracy']:.4f} | F1={result['f1_macro']:.4f} "
        f"| AUC={result['auc_macro_ovr']:.4f} | Params={result['parameters']:,} "
        f"| FLOPs={result['flops'] / 1e9:.3f}G | Latency={result['inference_time_ms']:.3f}ms "
        f"| PeakMem={result['peak_memory_mb']:.1f}MB"
    )

# %%
# ============================================================
# 9.3 - Save Native Comparison CSV and Figures
# ============================================================

comparison_path = os.path.join(cfg.OUTPUT_DIR, cfg.NATIVE_COMPARISON_CSV)
fieldnames = [
    "native_id", "experiment", "model", "student_img_size", "validation_accuracy",
    "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr", "parameters",
    "flops", "inference_time_ms", "peak_memory_mb", "best_epoch", "checkpoint",
]
with open(comparison_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(native_results)
print(f"Native comparison saved: {comparison_path}")

print("\n" + "=" * 140)
print("EfficientNet-Lite0 Native Profile - Deployment-Oriented Follow-Up")
print("=" * 140)
print(f"{'ID':>2} {'Experiment':<28} {'Input':>7} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Params':>12} {'GFLOPs':>10} {'Latency':>10} {'Peak Mem':>10}")
print("-" * 140)
for result in native_results:
    print(
        f"{result['native_id']:>2} {result['experiment']:<28} {result['student_img_size']:>7} "
        f"{result['validation_accuracy'] * 100:>7.2f}% {result['f1_macro']:>8.4f} "
        f"{result['auc_macro_ovr']:>8.4f} {result['parameters']:>12,} "
        f"{result['flops'] / 1e9:>10.3f} {result['inference_time_ms']:>8.3f}ms "
        f"{result['peak_memory_mb']:>8.1f}MB"
    )
print("=" * 140)
print("Note: this Native table is a separate follow-up and must not replace the controlled Exp 5/6/7 table.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, result in zip(axes, native_results):
    matrix = native_confusion_matrices[result["native_id"]]
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(result["experiment"], fontsize=10)
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
fig.colorbar(image, ax=axes.tolist(), shrink=0.72)
plt.suptitle("EfficientNet-Lite0 Native Confusion Matrices", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "confusion_matrices_lite0_native.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 8))
for result in native_results:
    fpr, tpr, plotted_auc = native_roc_curves[result["native_id"]]
    axis.plot(fpr, tpr, linewidth=1.8, label=f"{result['native_id']}: {result['experiment']} (AUC={plotted_auc:.3f})")
axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
axis.set_xlabel("False Positive Rate")
axis.set_ylabel("True Positive Rate")
axis.set_title("EfficientNet-Lite0 Native Macro-Average ROC Curves", fontweight="bold")
axis.legend(fontsize=9)
axis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "roc_curves_lite0_native.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Output Artifacts
#
# - efficientnet_lite0_native_baseline_best.pth
# - efficientnet_lite0_native_kd_scratch_best.pth
# - efficientnet_lite0_native_kd_twostage_best.pth
# - training_history_lite0_native_baseline.csv
# - training_history_lite0_native_direct_kd.csv
# - training_history_lite0_native_kd_twostage.csv
# - training_curves_lite0_native.png
# - val_accuracy_comparison_lite0_native.png
# - lite0_native_comparison_table.csv
# - confusion_matrices_lite0_native.png
# - roc_curves_lite0_native.png

# %%
print("\n" + "=" * 76)
print("Notebook 3B complete - EfficientNet-Lite0 Native follow-up done")
print("=" * 76)
print(f"Native A (Lite0 Baseline)    : {native_a_best_acc:.4f} ({native_a_best_acc * 100:.2f}%)")
print(f"Native B (Lite0 Direct KD)   : {native_b_best_acc:.4f} ({native_b_best_acc * 100:.2f}%)")
print(f"Native C (Lite0 Two-Stage KD): {native_c_best_acc:.4f} ({native_c_best_acc * 100:.2f}%)")
print("Keep this Native result table separate from the controlled seven-experiment matrix.")
print("=" * 76)
