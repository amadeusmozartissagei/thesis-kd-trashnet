# Experimental Design & Training Protocol

## Research Objective
Penelitian ini terdiri dari dua fase eksperimen:

1. **Paper Reproduction Phase**
   Mereplikasi pendekatan pada paper *Focus-RCNet: A Lightweight Recyclable Waste Classification Algorithm Based on Focus and Knowledge Distillation* dengan penyesuaian resource constraint.

2. **Main Experimental Comparison Phase**
   Melakukan evaluasi komparatif knowledge distillation pada lightweight student models dengan standardized experimental protocol yang fair.

---

# Phase 1 — Paper Reproduction (Focus-RCNet Reproduction)

## Objective
Mereplikasi eksperimen paper Focus-RCNet sefaithful mungkin, dengan penyesuaian karena keterbatasan resource komputasi.

Paper original menggunakan:
- NVIDIA RTX 3090 Ti
- Ubuntu 22.04
- Batch size: 16
- Epoch: 200
- Image size: 380×380
- Optimizer: SGD
- Learning rate: 0.05
- Momentum: 0.9
- Weight decay: 1e-4
- Cosine annealing scheduler

Karena keterbatasan resource, penelitian ini menyesuaikan:
- epoch
- batch size
- hardware environment

Selain itu, konfigurasi lain dibuat semirip mungkin dengan paper.

---

## Compute Environment
```python
PLATFORM = "Kaggle Notebooks"
FRAMEWORK = "PyTorch"

GPU_AVAILABLE = "2x NVIDIA Tesla T4 (16GB VRAM each)"
GPU_USED = "1x NVIDIA Tesla T4"
MULTI_GPU = False

OS = "Linux (Kaggle runtime)"
CUDA = "auto-detected"

USE_AMP = True
```

---

## Training Configuration
```python
EPOCHS = 100
BATCH_SIZE = 8
SEED = 42
```

Notes:
- Paper: 200 epochs → adjusted to 100
- Paper: batch size 16 → adjusted to 8
- Adjustment dilakukan karena keterbatasan Kaggle Tesla T4 environment

---

## Dataset Configuration
```python
DATASET = "TrashNet"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
```

---

## Input Configuration
```python
IMG_SIZE = 380
```

Mengikuti paper.

---

## Optimization
```python
OPTIMIZER = "SGD"

LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
```

---

## Scheduler
```python
SCHEDULER = "CosineAnnealingLR"
T_MAX = EPOCHS
```

---

## Loss Function
Teacher / baseline training:
```python
LOSS = "CrossEntropyLoss"
```

Knowledge distillation:
```python
LOSS = "KD Loss (Hard Label + Soft Label)"
```

---

## Knowledge Distillation Configuration
```python
KD_LOSS = "KLDivLoss"
KD_TEMPERATURE = 4
KD_ALPHA = 0.5
```

Notes:
- Paper mendefinisikan formula `L = α * L_soft + (1-α) * L_hard` tetapi tidak menyebutkan nilai T dan α yang digunakan.
- T=4 dan α=0.5 adalah asumsi reproduksi berdasarkan konvensi umum literatur KD (Hinton et al., 2015).
- Keputusan ini didokumentasikan sebagai **reproduction assumption**, bukan informasi dari paper.

---

## Checkpoint Strategy
```python
CHECKPOINT = "best_val_accuracy"
EARLY_STOPPING = False
```

---

## Weight Initialization
```python
PRETRAINED = True
WEIGHTS = "ImageNet"
```

---

## Model Source Libraries
```python
TEACHER_MODEL = "EfficientNet-B4"
TEACHER_SOURCE = "timm"
TEACHER_TIMM_NAME = "efficientnet_b4"

STUDENT_MODEL = "Focus-RCNet"
STUDENT_SOURCE = "custom (paper reproduction)"
```

Notes:
- Focus-RCNet diimplementasikan manual berdasarkan arsitektur dari paper (tabel konfigurasi stage, Focus module, Sandglass block, SimAM).
- EfficientNet-B4 menggunakan pretrained weights dari `timm`.

---

## Data Augmentation
```python
AUGMENTATION = {
    "HorizontalFlip": 0.5,
    "VerticalFlip": 0.5,
    "RandomBrightnessContrast": 0.5,
    "CoarseDropout": 0.5,
    "Normalize": "ImageNet"
}
```

Validation transform:
```python
VAL_TRANSFORM = {
    "Resize": 380,
    "Normalize": "ImageNet"
}
```

---

## Phase 1 Experiments

### 1. Train Base Model EfficientNet-B4
Teacher model utama.

Purpose:
- menghasilkan teacher model
- baseline high-capacity model

Output:
```text
EfficientNet-B4
```

---

### 2. Train Base Model Focus-RCNet
Focus-RCNet tanpa knowledge distillation.

Purpose:
- lightweight baseline sesuai paper

Output:
```text
Focus-RCNet
```

---

### 3. KD From Scratch (EfficientNet-B4 → Focus-RCNet)
Teacher:
```text
EfficientNet-B4
```

Student:
```text
Focus-RCNet
```

Training:
- hard label dari dataset
- soft label dari teacher

Purpose:
- mereplikasi KD setup paper

Output:
```text
Focus-RCNet-KD
```

---

### 4. Two-Stage KD + Fine-Tuning (EfficientNet-B4 → Focus-RCNet)

**Note: Ini adalah extension penelitian, bukan bagian dari paper original.**

Stage 1:
```text
Train Focus-RCNet normal (menggunakan konfigurasi Phase 1)
```

Stage 2 — Fine-Tuning with KD:
```python
STAGE2_EPOCHS = 50
STAGE2_LR = 0.005
STAGE2_FREEZE_LAYERS = False
STAGE2_SCHEDULER = "CosineAnnealingLR"
STAGE2_T_MAX = STAGE2_EPOCHS
```

Notes:
- Stage 2 menggunakan separuh epoch dari Stage 1 untuk menghindari overfitting.
- Learning rate diturunkan 10x dari Phase 1 LR (0.05 → 0.005) karena model sudah pre-converged.
- Tidak ada layer freezing — seluruh parameter di-fine-tune dengan KD signal.

Purpose:
- membandingkan KD from scratch vs KD fine-tuning

Output:
```text
Focus-RCNet-KD-TwoStage
```

---

# Phase 2 — Main Experimental Comparison

## Objective
Membandingkan lightweight student models secara fair menggunakan standardized experimental protocol.

Semua model menggunakan konfigurasi identik.

---

## Compute Environment
```python
PLATFORM = "Kaggle Notebooks"
FRAMEWORK = "PyTorch"

GPU_AVAILABLE = "2x NVIDIA Tesla T4 (16GB VRAM each)"
GPU_USED = "1x NVIDIA Tesla T4"
MULTI_GPU = False

OS = "Linux (Kaggle runtime)"
CUDA = "auto-detected"

USE_AMP = True
```

---

## Training Configuration
```python
EPOCHS = 100
BATCH_SIZE = 8
SEED = 42
```

---

## Dataset Configuration
```python
DATASET = "TrashNet"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
```

Rules:
- exact same split across all experiments
- same seed
- same train/validation indices

---

## Input Configuration
```python
IMG_SIZE = 224
```

Reason:
- fair comparison
- sesuai native EfficientNet-Lite0
- lebih realistis untuk Kaggle T4

---

## Optimization
```python
OPTIMIZER = "SGD"

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
```

---

## Scheduler
```python
SCHEDULER = "CosineAnnealingLR"
T_MAX = EPOCHS
```

---

## Loss Function
Base training:
```python
LOSS = "CrossEntropyLoss"
```

KD training:
```python
LOSS = "KD Loss (Hard Label + Soft Label)"
```

---

## Knowledge Distillation Configuration
```python
KD_LOSS = "KLDivLoss"
KD_TEMPERATURE = 4
KD_ALPHA = 0.5
```

Notes:
- Menggunakan konfigurasi KD yang sama dengan Phase 1 untuk konsistensi.

---

## Checkpoint Strategy
```python
CHECKPOINT = "best_val_accuracy"
EARLY_STOPPING = False
```

---

## Weight Initialization
```python
PRETRAINED = True
WEIGHTS = "ImageNet"
```

---

## Model Source Libraries
```python
TEACHER_MODEL = "EfficientNet-B4"
TEACHER_SOURCE = "timm"
TEACHER_TIMM_NAME = "efficientnet_b4"

STUDENT_MODEL = "EfficientNet-Lite0"
STUDENT_SOURCE = "timm"
STUDENT_TIMM_NAME = "efficientnet_lite0"
```

Notes:
- Kedua model menggunakan pretrained ImageNet weights dari library `timm`.

---

## Data Augmentation
```python
AUGMENTATION = {
    "HorizontalFlip": 0.5,
    "VerticalFlip": 0.5,
    "RandomBrightnessContrast": 0.5,
    "CoarseDropout": 0.5,
    "Normalize": "ImageNet"
}
```

Validation:
```python
VAL_TRANSFORM = {
    "Resize": 224,
    "Normalize": "ImageNet"
}
```

---

# Phase 2 Experiments

### 5. Train Base Model EfficientNet-Lite0
Purpose:
lightweight baseline kedua.

Output:
```text
EfficientNet-Lite0
```

---

### 6. KD From Scratch (EfficientNet-B4 → EfficientNet-Lite0)
Teacher:
```text
EfficientNet-B4
```

Student:
```text
EfficientNet-Lite0
```

Training:
- hard labels
- soft labels

Output:
```text
EfficientNet-Lite0-KD
```

---

### 7. Two-Stage KD + Fine-Tuning (EfficientNet-B4 → EfficientNet-Lite0)

**Note: Ini adalah extension penelitian, bukan bagian dari paper original.**

Stage 1:
```text
Train EfficientNet-Lite0 normal (menggunakan konfigurasi Phase 2)
```

Stage 2 — Fine-Tuning with KD:
```python
STAGE2_EPOCHS = 50
STAGE2_LR = 0.001
STAGE2_FREEZE_LAYERS = False
STAGE2_SCHEDULER = "CosineAnnealingLR"
STAGE2_T_MAX = STAGE2_EPOCHS
```

Notes:
- Stage 2 menggunakan separuh epoch dari Stage 1.
- Learning rate diturunkan 10x dari Phase 2 LR (0.01 → 0.001).
- Tidak ada layer freezing.

Output:
```text
EfficientNet-Lite0-KD-TwoStage
```

---

# Final Experiment Matrix

| No | Experiment | Purpose |
|----|-----------|---------|
| 1 | EfficientNet-B4 | Teacher baseline |
| 2 | Focus-RCNet | Lightweight baseline |
| 3 | Focus-RCNet KD Scratch | Paper KD reproduction |
| 4 | Focus-RCNet Two-Stage KD | Paper extension |
| 5 | EfficientNet-Lite0 | Proposed lightweight baseline |
| 6 | EfficientNet-Lite0 KD Scratch | Proposed KD |
| 7 | EfficientNet-Lite0 Two-Stage KD | Proposed KD extension |

Total experiments:
```text
7
```

---

# Evaluation Metrics

## Primary Metric
```python
PRIMARY_METRIC = "Accuracy"
```

---

## Reporting Metrics
```python
REPORT_METRICS = [
    "Accuracy",
    "Precision (macro)",
    "Recall (macro)",
    "F1-Score (macro)",
    "Confusion Matrix",
    "ROC Curve",
    "AUC (macro)"
]
```

---

## Model Efficiency Metrics
```python
EFFICIENCY_METRICS = [
    "Total Parameters",
    "FLOPs",
    "Inference Time (ms)"
]
```

Notes:
- Precision, Recall, dan F1-Score menggunakan macro averaging untuk konsistensi pada imbalanced dataset.
- Efficiency metrics diukur pada single NVIDIA Tesla T4 untuk fair comparison.
- ROC/AUC mengikuti paper original.