#!/usr/bin/env python3
"""
Kaggle-ready Notebook 4: WasteNet baseline and direct KD experiments.

This script runs four exploratory experiment tracks:
    1. WasteNet-128K baseline CE
    2. WasteNet-128K direct KD from EfficientNet-B4
    3. WasteNet-256K baseline CE
    4. WasteNet-256K direct KD from EfficientNet-B4

Expected Kaggle inputs:
    Dataset:
        /kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized

    EfficientNet-B4 teacher checkpoint:
        /kaggle/input/notebooks/hamzapratama/notebook1-teacher-training/
        efficientnet_b4_teacher_best.pth

The run uses 100 epochs as the controlled exploratory default. The previous
200-epoch follow-up was useful for paper-protocol probing, but it showed
overfitting symptoms and should not be the default for new WasteNet screening.

The direct KD loss follows the existing thesis experiments:
    alpha * KLDiv(student_logits / T, teacher_logits / T) * T^2
    + (1 - alpha) * CrossEntropyLoss
"""

import copy
import csv
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Config:
    EXPERIMENT_NAME = "Notebook 4 - WasteNet Baseline and Direct KD"
    EXPERIMENT_ID = "notebook4-wastenet-baseline-direct-kd"

    SEED = 42
    DATASET_NAME = "TrashNet"
    NUM_CLASSES = 6
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3

    IMG_SIZE = 160
    EPOCHS = 100
    BATCH_SIZE = 16

    OPTIMIZER = "SGD"
    LR = 0.05
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealingLR"
    USE_AMP = True

    KD_TEMPERATURE = 4
    KD_ALPHA = 0.5
    TEACHER_MODEL_NAME = "EfficientNet-B4"
    TEACHER_TIMM_NAME = "efficientnet_b4"

    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    OUTPUT_DIR = "/kaggle/working/wastenet_notebook4"

    TEACHER_CHECKPOINT_CANDIDATES = [
        "/kaggle/input/notebooks/hamzapratama/notebook1-teacher-training/"
        "efficientnet_b4_teacher_best.pth",
        "/kaggle/input/notebook1-teacher-training/efficientnet_b4_teacher_best.pth",
        "/kaggle/input/efficientnet-b4-teacher/efficientnet_b4_teacher_best.pth",
        "/kaggle/working/efficientnet_b4_teacher_best.pth",
    ]

    COMPARISON_FILENAME = "wastenet_baseline_direct_kd_comparison.csv"


cfg = Config()


@dataclass(frozen=True)
class WasteNetVariant:
    variant_id: str
    display_name: str
    channels: tuple
    repeats: tuple
    expand_ratio: float
    classifier_hidden: int
    expected_params: int
    checkpoint_prefix: str


WASTENET_128K = WasteNetVariant(
    variant_id="wastenet_128k",
    display_name="WasteNet-128K",
    channels=(16, 32, 64, 128, 192),
    repeats=(1, 1, 1, 1),
    expand_ratio=2.0,
    classifier_hidden=32,
    expected_params=128_086,
    checkpoint_prefix="wastenet_128k",
)

WASTENET_256K = WasteNetVariant(
    variant_id="wastenet_256k",
    display_name="WasteNet-256K",
    channels=(16, 32, 72, 128, 224),
    repeats=(1, 1, 2, 1),
    expand_ratio=2.5,
    classifier_hidden=0,
    expected_params=256_002,
    checkpoint_prefix="wastenet_256k",
)


EXPERIMENTS = [
    {
        "experiment_id": "wn128_ce",
        "display_name": "WasteNet-128K Baseline CE",
        "variant": WASTENET_128K,
        "training_mode": "baseline_ce",
        "uses_kd": False,
        "seed_offset": 0,
    },
    {
        "experiment_id": "wn128_direct_kd_b4",
        "display_name": "WasteNet-128K Direct KD from EfficientNet-B4",
        "variant": WASTENET_128K,
        "training_mode": "direct_kd_b4",
        "uses_kd": True,
        "seed_offset": 1,
    },
    {
        "experiment_id": "wn256_ce",
        "display_name": "WasteNet-256K Baseline CE",
        "variant": WASTENET_256K,
        "training_mode": "baseline_ce",
        "uses_kd": False,
        "seed_offset": 2,
    },
    {
        "experiment_id": "wn256_direct_kd_b4",
        "display_name": "WasteNet-256K Direct KD from EfficientNet-B4",
        "variant": WASTENET_256K,
        "training_mode": "direct_kd_b4",
        "uses_kd": True,
        "seed_offset": 3,
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AlbumentationsDataset(torch.utils.data.Dataset):
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


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 1.0,
    ):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        layers = []
        if hidden_channels != in_channels:
            layers.append(
                ConvBNAct(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
        layers.extend(
            [
                ConvBNAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                ),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.activation = nn.SiLU(inplace=True)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = y + x
        return self.activation(y)


class WasteNet(nn.Module):
    """
    Compact depthwise-separable CNN used as WasteNet-128K/256K variants.

    The two variants differ only in channel widths, block repeats, expansion
    ratio, and classifier width. Parameter counts are printed and validated at
    runtime to keep the experiment labels auditable.
    """

    def __init__(self, variant: WasteNetVariant, num_classes: int = 6, dropout=0.2):
        super().__init__()
        layers = [ConvBNAct(3, variant.channels[0], kernel_size=3, stride=2)]
        in_channels = variant.channels[0]

        for out_channels, repeat in zip(variant.channels[1:], variant.repeats):
            for block_idx in range(repeat):
                stride = 2 if block_idx == 0 else 1
                layers.append(
                    DepthwiseSeparableBlock(
                        in_channels,
                        out_channels,
                        stride=stride,
                        expand_ratio=variant.expand_ratio,
                    )
                )
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        if variant.classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, variant.classifier_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(variant.classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_wastenet(variant: WasteNetVariant) -> WasteNet:
    return WasteNet(variant=variant, num_classes=cfg.NUM_CLASSES)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def verify_wastenet_variant(variant: WasteNetVariant) -> None:
    model = create_wastenet(variant)
    total_params = count_parameters(model)
    sample = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    output = model(sample)
    assert output.shape == (1, cfg.NUM_CLASSES), f"Unexpected output shape: {output.shape}"
    assert total_params == variant.expected_params, (
        f"{variant.display_name} expected {variant.expected_params:,} params, "
        f"got {total_params:,}"
    )
    print(f"{variant.display_name} parameters: {total_params:,}")
    print(f"{variant.display_name} output shape: {list(output.shape)}")


def make_coarse_dropout():
    hole_min = int(cfg.IMG_SIZE * 0.05)
    hole_max = int(cfg.IMG_SIZE * 0.2)
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(hole_min, hole_max),
            hole_width_range=(hole_min, hole_max),
            fill=0,
            p=0.5,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=1,
            max_height=hole_max,
            max_width=hole_max,
            min_height=hole_min,
            min_width=hole_min,
            fill_value=0,
            p=0.5,
        )


def make_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = A.Compose(
        [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            make_coarse_dropout(),
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


def resolve_teacher_checkpoint_path() -> Path:
    for candidate in cfg.TEACHER_CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return Path(cfg.TEACHER_CHECKPOINT_CANDIDATES[0])


def load_or_create_split(full_dataset):
    checkpoint_path = resolve_teacher_checkpoint_path()
    if checkpoint_path.is_file():
        teacher_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        train_indices = teacher_checkpoint["train_indices"]
        val_indices = teacher_checkpoint["val_indices"]
        print(f"Split source: teacher checkpoint ({checkpoint_path})")
        print(
            "Teacher checkpoint best validation accuracy: "
            f"{teacher_checkpoint.get('best_val_acc', float('nan')):.4f}"
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
        print("Baseline CE can run, but direct KD will require a teacher checkpoint.")

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

    return full_dataset, train_loader, val_loader, train_indices, val_indices


def create_teacher_model():
    return timm.create_model(
        cfg.TEACHER_TIMM_NAME,
        pretrained=False,
        num_classes=cfg.NUM_CLASSES,
    )


def load_teacher_model(device):
    checkpoint_path = resolve_teacher_checkpoint_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Direct KD requires the EfficientNet-B4 teacher checkpoint. "
            f"Attach it at one of: {cfg.TEACHER_CHECKPOINT_CANDIDATES}"
        )

    teacher_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = teacher_checkpoint.get("model_state_dict", teacher_checkpoint)
    teacher_model = create_teacher_model()
    teacher_model.load_state_dict(state_dict)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    teacher_params = count_parameters(teacher_model)
    print(f"Teacher model loaded: {cfg.TEACHER_MODEL_NAME}")
    print(f"Teacher checkpoint: {checkpoint_path}")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Teacher best epoch: {teacher_checkpoint.get('best_epoch', 'n/a')}")
    print(
        "Teacher best validation accuracy: "
        f"{teacher_checkpoint.get('best_val_acc', float('nan')):.4f}"
    )
    print("Teacher mode: eval, frozen")
    del teacher_checkpoint
    return teacher_model


def train_one_epoch_ce(model, loader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        predictions = logits.max(1).indices
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_one_epoch_kd(
    student,
    teacher,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    use_amp,
):
    student.train()
    teacher.eval()
    running_loss_total = 0.0
    running_loss_soft = 0.0
    running_loss_hard = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
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
            loss_hard = criterion(student_logits.float(), labels)
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
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        predictions = logits.max(1).indices
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

    return running_loss / total, correct / total


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
        probabilities = F.softmax(logits.float(), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def compute_metrics(labels, predictions, probabilities):
    accuracy = float((labels == predictions).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    try:
        auc_macro = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc_macro = float("nan")

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auc_macro_ovr": float(auc_macro),
    }


def run_training(
    model,
    teacher_model,
    experiment,
    train_loader,
    val_loader,
    device,
    use_amp,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = GradScaler(enabled=use_amp)
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

        if experiment["uses_kd"]:
            train_loss, train_acc, train_loss_soft, train_loss_hard = train_one_epoch_kd(
                model,
                teacher_model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                use_amp,
            )
        else:
            train_loss, train_acc = train_one_epoch_ce(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                use_amp,
            )
            train_loss_soft = float("nan")
            train_loss_hard = train_loss

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
        history["train_loss_soft"].append(train_loss_soft)
        history["train_loss_hard"].append(train_loss_hard)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        best_marker = " BEST" if is_best else ""
        if experiment["uses_kd"]:
            loss_detail = f"(Soft: {train_loss_soft:.4f}, Hard: {train_loss_hard:.4f})"
        else:
            loss_detail = "(CE)"
        print(
            f"Epoch [{epoch:3d}/{cfg.EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} {loss_detail} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {epoch_time:.1f}s{best_marker}",
            flush=True,
        )

    total_time = time.time() - total_start
    return history, best_val_acc, best_epoch, best_model_state, total_time


def experiment_file_prefix(experiment):
    variant = experiment["variant"]
    return f"{variant.checkpoint_prefix}_{experiment['training_mode']}"


def save_history(history, output_dir: Path, experiment):
    path = output_dir / f"training_history_{experiment_file_prefix(experiment)}.csv"
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
    experiment,
    train_indices,
    val_indices,
    output_dir: Path,
):
    variant = experiment["variant"]
    path = output_dir / f"{experiment_file_prefix(experiment)}_best.pth"
    checkpoint = {
        "model_state_dict": model_state,
        "experiment_id": experiment["experiment_id"],
        "experiment_name": experiment["display_name"],
        "notebook_id": cfg.EXPERIMENT_ID,
        "model_name": variant.display_name,
        "variant": {
            "variant_id": variant.variant_id,
            "channels": variant.channels,
            "repeats": variant.repeats,
            "expand_ratio": variant.expand_ratio,
            "classifier_hidden": variant.classifier_hidden,
            "expected_params": variant.expected_params,
        },
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
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
            "use_amp": cfg.USE_AMP,
            "training_mode": experiment["training_mode"],
            "knowledge_distillation": experiment["uses_kd"],
            "kd_temperature": cfg.KD_TEMPERATURE if experiment["uses_kd"] else None,
            "kd_alpha": cfg.KD_ALPHA if experiment["uses_kd"] else None,
            "teacher_model": cfg.TEACHER_MODEL_NAME if experiment["uses_kd"] else None,
            "teacher_timm_name": cfg.TEACHER_TIMM_NAME if experiment["uses_kd"] else None,
            "teacher_checkpoint_candidates": cfg.TEACHER_CHECKPOINT_CANDIDATES,
        },
        "dataset_name": cfg.DATASET_NAME,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(checkpoint, path)
    return path


def save_curves(history, output_dir: Path, experiment):
    path = output_dir / f"training_curves_{experiment_file_prefix(experiment)}.png"
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    if experiment["uses_kd"]:
        axes[0].plot(history["epoch"], history["train_loss_soft"], label="Train soft loss")
        axes[0].plot(history["epoch"], history["train_loss_hard"], label="Train hard CE loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation CE loss")
    axes[0].set_title(f"{experiment['display_name']} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_acc"], label="Validation accuracy")
    axes[1].set_title(f"{experiment['display_name']} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_predictions(
    full_dataset,
    val_indices,
    labels,
    predictions,
    probabilities,
    output_dir: Path,
    experiment,
):
    path = output_dir / f"predictions_{experiment_file_prefix(experiment)}.csv"
    fieldnames = [
        "image_path",
        "label",
        "prediction",
        "label_name",
        "prediction_name",
    ] + [f"prob_class_{idx}" for idx in range(cfg.NUM_CLASSES)] + [
        "seed",
        "model_id",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, original_index in enumerate(val_indices):
            image_path, _ = full_dataset.samples[original_index]
            label = int(labels[row_idx])
            prediction = int(predictions[row_idx])
            row = {
                "image_path": image_path,
                "label": label,
                "prediction": prediction,
                "label_name": cfg.CLASS_NAMES[label],
                "prediction_name": cfg.CLASS_NAMES[prediction],
                "seed": cfg.SEED,
                "model_id": experiment["experiment_id"],
            }
            for class_idx in range(cfg.NUM_CLASSES):
                row[f"prob_class_{class_idx}"] = f"{probabilities[row_idx][class_idx]:.8f}"
            writer.writerow(row)
    return path


def run_single_experiment(
    experiment,
    teacher_model,
    full_dataset,
    train_loader,
    val_loader,
    train_indices,
    val_indices,
    device,
    output_dir,
):
    variant = experiment["variant"]
    use_amp = cfg.USE_AMP and device.type == "cuda"

    print("\n" + "=" * 80)
    print(experiment["display_name"])
    print("=" * 80)
    print(f"Variant             : {variant.display_name}")
    print(f"Training mode       : {experiment['training_mode']}")
    print(f"Knowledge distill.  : {experiment['uses_kd']}")
    if experiment["uses_kd"]:
        print(f"Teacher             : {cfg.TEACHER_MODEL_NAME}")
        print(f"KD temperature      : {cfg.KD_TEMPERATURE}")
        print(f"KD alpha            : {cfg.KD_ALPHA}")
    print("=" * 80)

    seed_everything(cfg.SEED + experiment["seed_offset"])
    model = create_wastenet(variant).to(device)
    total_params = count_parameters(model)
    assert total_params == variant.expected_params

    if experiment["uses_kd"] and teacher_model is None:
        raise RuntimeError(f"{experiment['display_name']} requires a teacher model.")

    history, best_val_acc, best_epoch, best_model_state, total_time = run_training(
        model,
        teacher_model,
        experiment,
        train_loader,
        val_loader,
        device,
        use_amp,
    )

    model.load_state_dict(best_model_state)
    labels, predictions, probabilities = collect_predictions(
        model,
        val_loader,
        device,
        use_amp,
    )
    metrics = compute_metrics(labels, predictions, probabilities)

    checkpoint_path = save_checkpoint(
        best_model_state,
        history,
        best_epoch,
        best_val_acc,
        experiment,
        train_indices,
        val_indices,
        output_dir,
    )
    history_path = save_history(history, output_dir, experiment)
    curves_path = save_curves(history, output_dir, experiment)
    predictions_path = save_predictions(
        full_dataset,
        val_indices,
        labels,
        predictions,
        probabilities,
        output_dir,
        experiment,
    )

    result = {
        "experiment_id": experiment["experiment_id"],
        "experiment_name": experiment["display_name"],
        "model_name": variant.display_name,
        "training_mode": experiment["training_mode"],
        "uses_kd": experiment["uses_kd"],
        "teacher": cfg.TEACHER_MODEL_NAME if experiment["uses_kd"] else "None",
        "parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "validation_accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "auc_macro_ovr": metrics["auc_macro_ovr"],
        "total_time_minutes": total_time / 60,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "curves_png": str(curves_path),
        "predictions_csv": str(predictions_path),
    }

    print("\nExperiment complete")
    print(f"Best validation accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Best epoch               : {best_epoch}")
    print(f"F1 macro                 : {metrics['f1_macro']:.4f}")
    print(f"AUC macro OVR            : {metrics['auc_macro_ovr']:.4f}")
    print(f"Total time               : {total_time / 60:.1f} minutes")
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"History CSV              : {history_path}")
    print(f"Predictions CSV          : {predictions_path}")
    print(f"Training curves          : {curves_path}")
    return result


def save_comparison_table(results, output_dir: Path):
    path = output_dir / cfg.COMPARISON_FILENAME
    fieldnames = [
        "experiment_id",
        "experiment_name",
        "model_name",
        "training_mode",
        "uses_kd",
        "teacher",
        "parameters",
        "best_epoch",
        "best_val_acc",
        "validation_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_macro_ovr",
        "total_time_minutes",
        "checkpoint",
        "history_csv",
        "curves_png",
        "predictions_csv",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            for key in [
                "best_val_acc",
                "validation_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_macro_ovr",
                "total_time_minutes",
            ]:
                row[key] = f"{row[key]:.6f}"
            writer.writerow(row)
    return path


def print_configuration(device):
    print("=" * 80)
    print(cfg.EXPERIMENT_NAME)
    print("=" * 80)
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Dataset         : {cfg.DATASET_NAME}")
    print(f"Dataset path    : {cfg.DATASET_PATH}")
    print(f"Image size      : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print(f"Epochs          : {cfg.EPOCHS}")
    print(f"Batch size      : {cfg.BATCH_SIZE}")
    print(f"Optimizer       : {cfg.OPTIMIZER}")
    print(f"Learning rate   : {cfg.LR}")
    print(f"Momentum        : {cfg.MOMENTUM}")
    print(f"Weight decay    : {cfg.WEIGHT_DECAY}")
    print(f"Scheduler       : {cfg.SCHEDULER}")
    print(f"KD temperature  : {cfg.KD_TEMPERATURE}")
    print(f"KD alpha        : {cfg.KD_ALPHA}")
    print(f"Teacher         : {cfg.TEACHER_MODEL_NAME} ({cfg.TEACHER_TIMM_NAME})")
    print(f"Output dir      : {cfg.OUTPUT_DIR}")
    print(f"Seed            : {cfg.SEED}")
    print("=" * 80)


def main():
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_configuration(device)

    verify_wastenet_variant(WASTENET_128K)
    verify_wastenet_variant(WASTENET_256K)

    full_dataset, train_loader, val_loader, train_indices, val_indices = prepare_data()

    teacher_model = None
    if any(experiment["uses_kd"] for experiment in EXPERIMENTS):
        teacher_model = load_teacher_model(device)

    results = []
    for experiment in EXPERIMENTS:
        result = run_single_experiment(
            experiment,
            teacher_model,
            full_dataset,
            train_loader,
            val_loader,
            train_indices,
            val_indices,
            device,
            output_dir,
        )
        results.append(result)

    comparison_path = save_comparison_table(results, output_dir)

    print("\n" + "=" * 80)
    print("Notebook 4 complete - WasteNet baseline and direct KD experiments")
    print("=" * 80)
    print(f"Comparison CSV: {comparison_path}")
    print("\nSummary:")
    print(
        f"{'ID':<20} {'Model':<15} {'Mode':<15} {'Acc':>8} "
        f"{'F1':>8} {'AUC':>8} {'Params':>12}"
    )
    print("-" * 94)
    for result in results:
        print(
            f"{result['experiment_id']:<20} "
            f"{result['model_name']:<15} "
            f"{result['training_mode']:<15} "
            f"{result['validation_accuracy'] * 100:>7.2f}% "
            f"{result['f1_macro']:>8.4f} "
            f"{result['auc_macro_ovr']:>8.4f} "
            f"{result['parameters']:>12,}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
