#!/usr/bin/env python3
"""
Kaggle-ready Notebook 5: WasteNet teacher-assistant two-stage KD.

This script runs the two remaining core WasteNet experiments:
    1. WasteNet-128K two-stage KD from Focus-RCNet teacher assistant
    2. WasteNet-256K two-stage KD from Focus-RCNet teacher assistant

Stage 1 is not retrained here. The script loads the best WasteNet CE
checkpoints produced by Notebook 4, then fine-tunes each student with
logits-based KD from Focus-RCNet.

Expected Kaggle inputs:
    Dataset:
        /kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized

    Notebook 4 output:
        wastenet_128k_baseline_ce_best.pth
        wastenet_256k_baseline_ce_best.pth
        wastenet_baseline_direct_kd_comparison.csv

    Primary teacher-assistant checkpoint:
        focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth

Optional fallback teacher-assistant checkpoint:
        focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth

KD loss:
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Config:
    EXPERIMENT_NAME = "Notebook 5 - WasteNet Teacher-Assistant Two-Stage KD"
    EXPERIMENT_ID = "notebook5-wastenet-teacher-assistant-twostage-kd"

    SEED = 42
    DATASET_NAME = "TrashNet"
    NUM_CLASSES = 6
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3

    IMG_SIZE = 160
    STAGE1_EPOCHS = 100
    STAGE2_EPOCHS = 50
    BATCH_SIZE = 16

    OPTIMIZER = "SGD"
    STAGE1_LR = 0.05
    STAGE2_LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealingLR"
    USE_AMP = True

    KD_TEMPERATURE = 4
    KD_ALPHA = 0.5
    TEACHER_ASSISTANT_MODEL_NAME = "Focus-RCNet Direct KD"
    TEACHER_ASSISTANT_FALLBACK_NAME = "Focus-RCNet Two-Stage KD"

    DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
    OUTPUT_DIR = "/kaggle/working/wastenet_notebook5"

    NOTEBOOK4_ROOT_CANDIDATES = [
        "/kaggle/input/notebooks/hamzapratama/notebook4-wastenet-baseline-direct-kd/wastenet_notebook4",
        "/kaggle/input/notebooks/hamzapratama/notebook4-wastenet-baseline-direct-kd",
        "/kaggle/input/notebook4-wastenet-baseline-direct-kd/wastenet_notebook4",
        "/kaggle/input/notebook4-wastenet-baseline-direct-kd",
        "/kaggle/input/wastenet-notebook4/wastenet_notebook4",
        "/kaggle/input/wastenet-notebook4",
        "/kaggle/input/wastenet_notebook4",
        "/kaggle/working/wastenet_notebook4",
    ]

    TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES = [
        (
            "/kaggle/input/notebooks/hamzapratama/focus-rcnet-direct-kd-e200-bs16/"
            "focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth"
        ),
        (
            "/kaggle/input/focus-rcnet-direct-kd-e200-bs16/"
            "focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth"
        ),
        (
            "/kaggle/input/focus-rcnet-direct-kd-e200-bs16-t4-a05/"
            "focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth"
        ),
        "/kaggle/input/focus-rcnet-direct-kd/focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth",
        "/kaggle/working/focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth",
    ]

    FALLBACK_TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES = [
        (
            "/kaggle/input/notebooks/hamzapratama/focus-rcnet-twostage-kd-e200-bs16/"
            "focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth"
        ),
        (
            "/kaggle/input/focus-rcnet-twostage-kd-e200-bs16/"
            "focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth"
        ),
        (
            "/kaggle/input/focus-rcnet-twostage-kd-e200-bs16-t4-a05/"
            "focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth"
        ),
        "/kaggle/input/focus-rcnet-twostage-kd/focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth",
        "/kaggle/working/focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth",
    ]

    NOTEBOOK4_COMPARISON_FILENAME = "wastenet_baseline_direct_kd_comparison.csv"
    NOTEBOOK5_COMPARISON_FILENAME = "wastenet_teacher_assistant_twostage_comparison.csv"
    ALL_CORE_COMPARISON_FILENAME = "wastenet_all_core_experiments_comparison.csv"


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
        "experiment_id": "wn128_ta_twostage_kd",
        "display_name": "WasteNet-128K Teacher-Assistant Two-Stage KD",
        "variant": WASTENET_128K,
        "training_mode": "ta_twostage_kd",
        "stage1_filename": "wastenet_128k_baseline_ce_best.pth",
        "seed_offset": 4,
    },
    {
        "experiment_id": "wn256_ta_twostage_kd",
        "display_name": "WasteNet-256K Teacher-Assistant Two-Stage KD",
        "variant": WASTENET_256K,
        "training_mode": "ta_twostage_kd",
        "stage1_filename": "wastenet_256k_baseline_ce_best.pth",
        "seed_offset": 5,
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_indices(indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy().tolist()
    elif isinstance(indices, np.ndarray):
        indices = indices.tolist()
    return [int(index) for index in indices]


def same_indices(left, right) -> bool:
    return normalize_indices(left) == normalize_indices(right)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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
            layers.append(ConvBNAct(in_channels, hidden_channels, kernel_size=1))
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


class Focus(nn.Module):
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


def resolve_notebook4_file(filename: str, required: bool = True) -> Path:
    for root in cfg.NOTEBOOK4_ROOT_CANDIDATES:
        path = Path(root) / filename
        if path.is_file():
            return path
    if required:
        roots = "\n".join(f"  - {root}" for root in cfg.NOTEBOOK4_ROOT_CANDIDATES)
        raise FileNotFoundError(
            f"Notebook 4 artifact not found: {filename}\n"
            f"Attach Notebook 4 output at one of these roots:\n{roots}"
        )
    return Path(cfg.NOTEBOOK4_ROOT_CANDIDATES[0]) / filename


def resolve_teacher_assistant_checkpoint():
    for candidate in cfg.TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path, cfg.TEACHER_ASSISTANT_MODEL_NAME

    for candidate in cfg.FALLBACK_TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path, cfg.TEACHER_ASSISTANT_FALLBACK_NAME

    candidates = (
        cfg.TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES
        + cfg.FALLBACK_TEACHER_ASSISTANT_CHECKPOINT_CANDIDATES
    )
    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Focus-RCNet teacher-assistant checkpoint not found.\n"
        f"Attach one of these files:\n{candidate_text}"
    )


def validate_split_indices(train_indices, val_indices, full_dataset_size: int) -> None:
    train_indices = normalize_indices(train_indices)
    val_indices = normalize_indices(val_indices)
    all_indices = train_indices + val_indices
    assert len(all_indices) == full_dataset_size
    assert not set(train_indices).intersection(val_indices)
    assert max(all_indices) < full_dataset_size
    assert min(all_indices) >= 0


def load_stage1_checkpoint(experiment):
    checkpoint_path = resolve_notebook4_file(experiment["stage1_filename"])
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Missing model_state_dict in {checkpoint_path}")
    if "train_indices" not in checkpoint or "val_indices" not in checkpoint:
        raise KeyError(
            f"Missing train_indices/val_indices in {checkpoint_path}. "
            "Notebook 5 requires the exact Notebook 4 split."
        )

    variant = experiment["variant"]
    saved_variant = checkpoint.get("variant", {})
    saved_variant_id = saved_variant.get("variant_id")
    if saved_variant_id and saved_variant_id != variant.variant_id:
        raise ValueError(
            f"Checkpoint variant mismatch for {checkpoint_path}: "
            f"expected {variant.variant_id}, found {saved_variant_id}"
        )

    return checkpoint_path, checkpoint


def prepare_data(train_indices, val_indices):
    train_transform, val_transform = make_transforms()
    full_dataset = datasets.ImageFolder(root=cfg.DATASET_PATH, transform=None)
    assert full_dataset.classes == cfg.CLASS_NAMES, (
        f"Expected classes {cfg.CLASS_NAMES}, found {full_dataset.classes}"
    )

    train_indices = normalize_indices(train_indices)
    val_indices = normalize_indices(val_indices)
    validate_split_indices(train_indices, val_indices, len(full_dataset))

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


def verify_architectures():
    for variant in [WASTENET_128K, WASTENET_256K]:
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

    teacher = create_focus_rcnet(cfg.NUM_CLASSES)
    teacher_params = count_parameters(teacher)
    sample = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    output = teacher(sample)
    assert output.shape == (1, cfg.NUM_CLASSES), f"Unexpected teacher output: {output.shape}"
    assert teacher_params == 520_630, f"Unexpected Focus-RCNet params: {teacher_params}"
    print(f"Focus-RCNet teacher-assistant parameters: {teacher_params:,}")
    print(f"Focus-RCNet output shape: {list(output.shape)}")


def load_teacher_assistant_model(device, train_indices, val_indices):
    checkpoint_path, teacher_label = resolve_teacher_assistant_checkpoint()
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "train_indices" in checkpoint and "val_indices" in checkpoint:
        train_match = same_indices(checkpoint["train_indices"], train_indices)
        val_match = same_indices(checkpoint["val_indices"], val_indices)
        if not train_match or not val_match:
            raise ValueError(
                "Teacher-assistant checkpoint split does not match Notebook 4 "
                f"WasteNet split: {checkpoint_path}"
            )

    teacher = create_focus_rcnet(cfg.NUM_CLASSES)
    teacher.load_state_dict(state_dict)
    teacher = teacher.to(device)
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print(f"Teacher assistant loaded: {teacher_label}")
    print(f"Teacher checkpoint      : {checkpoint_path}")
    print(f"Teacher parameters      : {count_parameters(teacher):,}")
    print(f"Teacher best epoch      : {checkpoint.get('best_epoch', 'n/a')}")
    print(f"Teacher best val acc    : {checkpoint.get('best_val_acc', float('nan')):.4f}")
    print("Teacher mode            : eval, frozen")

    metadata = {
        "teacher_assistant_name": teacher_label,
        "teacher_assistant_checkpoint": str(checkpoint_path),
        "teacher_assistant_best_epoch": checkpoint.get("best_epoch", "n/a"),
        "teacher_assistant_best_val_acc": checkpoint.get("best_val_acc", float("nan")),
    }
    del checkpoint
    return teacher, metadata


def load_stage1_model(experiment, device, train_indices, val_indices):
    checkpoint_path, checkpoint = load_stage1_checkpoint(experiment)
    if not same_indices(checkpoint["train_indices"], train_indices):
        raise ValueError(f"Stage 1 train split mismatch: {checkpoint_path}")
    if not same_indices(checkpoint["val_indices"], val_indices):
        raise ValueError(f"Stage 1 validation split mismatch: {checkpoint_path}")

    variant = experiment["variant"]
    model = create_wastenet(variant).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    total_params = count_parameters(model)
    assert total_params == variant.expected_params

    metadata = {
        "stage1_checkpoint": str(checkpoint_path),
        "stage1_best_epoch": checkpoint.get("best_epoch", "n/a"),
        "stage1_best_val_acc": checkpoint.get("best_val_acc", float("nan")),
        "stage1_training_mode": checkpoint.get("config", {}).get("training_mode", "baseline_ce"),
    }

    print("Stage 1 WasteNet checkpoint loaded")
    print(f"Stage 1 checkpoint       : {checkpoint_path}")
    print(f"Stage 1 best epoch       : {metadata['stage1_best_epoch']}")
    print(f"Stage 1 best val accuracy: {metadata['stage1_best_val_acc']:.4f}")
    print(f"Stage 1 parameters       : {total_params:,}")

    del checkpoint
    return model, metadata


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


def run_training(model, teacher_model, train_loader, val_loader, device, use_amp):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.STAGE2_LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.STAGE2_EPOCHS,
    )
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

    for epoch in range(1, cfg.STAGE2_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

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
        print(
            f"Epoch [{epoch:3d}/{cfg.STAGE2_EPOCHS}] "
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


def save_curves(history, output_dir: Path, experiment):
    path = output_dir / f"training_curves_{experiment_file_prefix(experiment)}.png"
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train total loss")
    axes[0].plot(history["epoch"], history["train_loss_soft"], label="Train soft loss")
    axes[0].plot(history["epoch"], history["train_loss_hard"], label="Train hard CE loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation CE loss")
    axes[0].set_title(f"{experiment['display_name']} Loss")
    axes[0].set_xlabel("Stage 2 epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_acc"], label="Validation accuracy")
    axes[1].set_title(f"{experiment['display_name']} Accuracy")
    axes[1].set_xlabel("Stage 2 epoch")
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


def save_confusion_matrix(labels, predictions, output_dir: Path, experiment):
    matrix = confusion_matrix(labels, predictions, labels=list(range(cfg.NUM_CLASSES)))
    csv_path = output_dir / f"confusion_matrix_{experiment_file_prefix(experiment)}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label"] + cfg.CLASS_NAMES)
        for class_idx, row in enumerate(matrix):
            writer.writerow([cfg.CLASS_NAMES[class_idx]] + row.tolist())

    png_path = output_dir / f"confusion_matrix_{experiment_file_prefix(experiment)}.png"
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(cfg.NUM_CLASSES),
        yticks=np.arange(cfg.NUM_CLASSES),
        xticklabels=cfg.CLASS_NAMES,
        yticklabels=cfg.CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{experiment['display_name']} Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.max() else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(
                col_idx,
                row_idx,
                format(matrix[row_idx, col_idx], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return csv_path, png_path


def save_checkpoint(
    model_state,
    history,
    best_epoch,
    best_val_acc,
    experiment,
    stage1_metadata,
    teacher_metadata,
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
        "stage1": stage1_metadata,
        "teacher_assistant": teacher_metadata,
        "config": {
            "img_size": cfg.IMG_SIZE,
            "stage1_epochs": cfg.STAGE1_EPOCHS,
            "stage1_lr": cfg.STAGE1_LR,
            "stage2_epochs": cfg.STAGE2_EPOCHS,
            "stage2_lr": cfg.STAGE2_LR,
            "epochs": cfg.STAGE2_EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "momentum": cfg.MOMENTUM,
            "weight_decay": cfg.WEIGHT_DECAY,
            "optimizer": cfg.OPTIMIZER,
            "scheduler": cfg.SCHEDULER,
            "seed": cfg.SEED,
            "run_seed": cfg.SEED + experiment["seed_offset"],
            "train_split": cfg.TRAIN_SPLIT,
            "val_split": cfg.VAL_SPLIT,
            "use_amp": cfg.USE_AMP,
            "training_mode": experiment["training_mode"],
            "knowledge_distillation": True,
            "kd_temperature": cfg.KD_TEMPERATURE,
            "kd_alpha": cfg.KD_ALPHA,
            "teacher_model": teacher_metadata["teacher_assistant_name"],
            "teacher_checkpoint_path": teacher_metadata["teacher_assistant_checkpoint"],
            "stage1_checkpoint_path": stage1_metadata["stage1_checkpoint"],
        },
        "dataset_name": cfg.DATASET_NAME,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(checkpoint, path)
    return path


def run_single_experiment(
    experiment,
    teacher_model,
    teacher_metadata,
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
    print("Stage 1             : load Notebook 4 CE checkpoint")
    print(f"Stage 2 epochs      : {cfg.STAGE2_EPOCHS}")
    print(f"Stage 2 LR          : {cfg.STAGE2_LR}")
    print(f"Teacher assistant   : {teacher_metadata['teacher_assistant_name']}")
    print(f"KD temperature      : {cfg.KD_TEMPERATURE}")
    print(f"KD alpha            : {cfg.KD_ALPHA}")
    print("=" * 80)

    seed_everything(cfg.SEED + experiment["seed_offset"])
    model, stage1_metadata = load_stage1_model(
        experiment,
        device,
        train_indices,
        val_indices,
    )

    history, best_val_acc, best_epoch, best_model_state, total_time = run_training(
        model,
        teacher_model,
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
        stage1_metadata,
        teacher_metadata,
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
    confusion_csv_path, confusion_png_path = save_confusion_matrix(
        labels,
        predictions,
        output_dir,
        experiment,
    )

    result = {
        "experiment_id": experiment["experiment_id"],
        "experiment_name": experiment["display_name"],
        "model_name": variant.display_name,
        "training_mode": experiment["training_mode"],
        "uses_kd": True,
        "teacher": teacher_metadata["teacher_assistant_name"],
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "validation_accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "auc_macro_ovr": metrics["auc_macro_ovr"],
        "total_time_minutes": total_time / 60,
        "stage1_checkpoint": stage1_metadata["stage1_checkpoint"],
        "stage1_best_epoch": stage1_metadata["stage1_best_epoch"],
        "stage1_best_val_acc": stage1_metadata["stage1_best_val_acc"],
        "teacher_checkpoint": teacher_metadata["teacher_assistant_checkpoint"],
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "curves_png": str(curves_path),
        "predictions_csv": str(predictions_path),
        "confusion_matrix_csv": str(confusion_csv_path),
        "confusion_matrix_png": str(confusion_png_path),
    }

    print("\nExperiment complete")
    print(f"Best validation accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Best stage 2 epoch       : {best_epoch}")
    print(f"Stage 1 best val acc     : {stage1_metadata['stage1_best_val_acc']:.4f}")
    print(f"F1 macro                 : {metrics['f1_macro']:.4f}")
    print(f"AUC macro OVR            : {metrics['auc_macro_ovr']:.4f}")
    print(f"Total time               : {total_time / 60:.1f} minutes")
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"History CSV              : {history_path}")
    print(f"Predictions CSV          : {predictions_path}")
    print(f"Training curves          : {curves_path}")
    print(f"Confusion matrix CSV     : {confusion_csv_path}")
    return result


def ordered_union_fieldnames(rows, preferred):
    fieldnames = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def format_float_fields(row):
    formatted = dict(row)
    for key in [
        "best_val_acc",
        "validation_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_macro_ovr",
        "total_time_minutes",
        "stage1_best_val_acc",
    ]:
        if key in formatted and formatted[key] not in ["", None]:
            try:
                formatted[key] = f"{float(formatted[key]):.6f}"
            except (TypeError, ValueError):
                pass
    return formatted


def save_comparison_table(results, output_dir: Path):
    path = output_dir / cfg.NOTEBOOK5_COMPARISON_FILENAME
    preferred_fields = [
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
        "stage1_checkpoint",
        "stage1_best_epoch",
        "stage1_best_val_acc",
        "teacher_checkpoint",
        "checkpoint",
        "history_csv",
        "curves_png",
        "predictions_csv",
        "confusion_matrix_csv",
        "confusion_matrix_png",
    ]
    fieldnames = ordered_union_fieldnames(results, preferred_fields)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(format_float_fields(result))
    return path


def read_notebook4_comparison():
    try:
        path = resolve_notebook4_file(cfg.NOTEBOOK4_COMPARISON_FILENAME, required=False)
    except FileNotFoundError:
        return [], None

    if not path.is_file():
        print(
            "Notebook 4 comparison CSV not found; "
            "saving Notebook 5 comparison only for all-core table."
        )
        return [], None

    with path.open("r", newline="") as file:
        rows = list(csv.DictReader(file))
    print(f"Notebook 4 comparison loaded: {path}")
    return rows, path


def save_all_core_comparison(results, output_dir: Path):
    notebook4_rows, notebook4_path = read_notebook4_comparison()
    notebook5_rows = [format_float_fields(result) for result in results]
    rows = notebook4_rows + notebook5_rows
    path = output_dir / cfg.ALL_CORE_COMPARISON_FILENAME
    preferred_fields = [
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
    fieldnames = ordered_union_fieldnames(rows, preferred_fields)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path, notebook4_path


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
    print(f"Stage 1 epochs  : {cfg.STAGE1_EPOCHS} (loaded from Notebook 4)")
    print(f"Stage 2 epochs  : {cfg.STAGE2_EPOCHS}")
    print(f"Batch size      : {cfg.BATCH_SIZE}")
    print(f"Optimizer       : {cfg.OPTIMIZER}")
    print(f"Stage 2 LR      : {cfg.STAGE2_LR}")
    print(f"Momentum        : {cfg.MOMENTUM}")
    print(f"Weight decay    : {cfg.WEIGHT_DECAY}")
    print(f"Scheduler       : {cfg.SCHEDULER}")
    print(f"KD temperature  : {cfg.KD_TEMPERATURE}")
    print(f"KD alpha        : {cfg.KD_ALPHA}")
    print(f"Output dir      : {cfg.OUTPUT_DIR}")
    print(f"Seed            : {cfg.SEED}")
    print("=" * 80)


def main():
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_configuration(device)
    verify_architectures()

    stage1_checkpoints = {}
    for experiment in EXPERIMENTS:
        checkpoint_path, checkpoint = load_stage1_checkpoint(experiment)
        stage1_checkpoints[experiment["experiment_id"]] = (checkpoint_path, checkpoint)
        print(
            f"Resolved Stage 1 checkpoint for {experiment['experiment_id']}: "
            f"{checkpoint_path}"
        )

    reference_checkpoint = stage1_checkpoints[EXPERIMENTS[0]["experiment_id"]][1]
    train_indices = normalize_indices(reference_checkpoint["train_indices"])
    val_indices = normalize_indices(reference_checkpoint["val_indices"])

    for experiment in EXPERIMENTS[1:]:
        _, checkpoint = stage1_checkpoints[experiment["experiment_id"]]
        if not same_indices(checkpoint["train_indices"], train_indices):
            raise ValueError("WasteNet Stage 1 train splits do not match.")
        if not same_indices(checkpoint["val_indices"], val_indices):
            raise ValueError("WasteNet Stage 1 validation splits do not match.")
    del stage1_checkpoints

    full_dataset, train_loader, val_loader, train_indices, val_indices = prepare_data(
        train_indices,
        val_indices,
    )
    teacher_model, teacher_metadata = load_teacher_assistant_model(
        device,
        train_indices,
        val_indices,
    )

    results = []
    for experiment in EXPERIMENTS:
        result = run_single_experiment(
            experiment,
            teacher_model,
            teacher_metadata,
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
    all_core_path, notebook4_comparison_path = save_all_core_comparison(results, output_dir)

    print("\n" + "=" * 80)
    print("Notebook 5 complete - WasteNet teacher-assistant two-stage KD")
    print("=" * 80)
    print(f"Notebook 5 comparison CSV : {comparison_path}")
    print(f"All-core comparison CSV   : {all_core_path}")
    if notebook4_comparison_path:
        print(f"Notebook 4 comparison src : {notebook4_comparison_path}")
    print("\nSummary:")
    print(
        f"{'ID':<24} {'Model':<15} {'Mode':<17} {'Acc':>8} "
        f"{'F1':>8} {'AUC':>8} {'Params':>12}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result['experiment_id']:<24} "
            f"{result['model_name']:<15} "
            f"{result['training_mode']:<17} "
            f"{result['validation_accuracy'] * 100:>7.2f}% "
            f"{result['f1_macro']:>8.4f} "
            f"{result['auc_macro_ovr']:>8.4f} "
            f"{result['parameters']:>12,}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
