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
- Learning rate reduced by a factor of 10 every 90 epochs
- Cosine annealing scheduler

Karena keterbatasan resource, penelitian ini menyesuaikan:
- epoch
- batch size
- hardware environment

Selain itu, konfigurasi lain dibuat semirip mungkin dengan paper. Reproduksi ini
tidak diklaim identik karena paper tidak menjelaskan seluruh detail implementasi,
termasuk nilai hyperparameter knowledge distillation dan interaksi antara
step-based learning-rate decay dengan cosine annealing scheduler.

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
DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
OUTPUT_DIR = "/kaggle/working/"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
```

Notes:
- Paper melaporkan 2528 citra TrashNet, termasuk 483 citra plastic.
- Kaggle mirror yang digunakan pada eksperimen ini berisi 2527 citra, termasuk
  482 citra plastic.

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

Notes:
- Paper menyebut cosine annealing dan penurunan learning rate sebesar 10x setiap
  90 epoch.
- Implementasi reproduksi menggunakan `CosineAnnealingLR(T_max=EPOCHS)` tanpa
  step-based decay tambahan.

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
- Paper juga tidak menyebutkan fungsi `L_soft` secara eksplisit maupun faktor
  `T²`.
- T=4 dan α=0.5 adalah asumsi reproduksi berdasarkan konvensi umum literatur KD (Hinton et al., 2015).
- Implementasi reproduksi menggunakan `KLDivLoss(reduction="batchmean")`,
  temperature scaling pada logit teacher dan student, serta faktor `T²`.
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
TEACHER_PRETRAINED = True
TEACHER_WEIGHTS = "ImageNet"

STUDENT_PRETRAINED = False
STUDENT_INITIALIZATION = "Kaiming Normal for Conv2d"
```

Notes:
- EfficientNet-B4 diinisialisasi menggunakan pretrained ImageNet weights pada
  Notebook 1, kemudian checkpoint hasil fine-tuning digunakan sebagai teacher.
- Focus-RCNet adalah custom student model dan dilatih dari random initialization.

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

## Focus-RCNet Architecture
```text
Focus CBS 1x1:  3 → 24
Stage 1:       24 → 48,  4 Sandglass blocks, SimAM
Stage 2:       48 → 96,  3 Sandglass blocks, SimAM
Stage 3:       96 → 192, 2 Sandglass blocks, SimAM
Stage 4:      192 → 384, 2 Sandglass blocks, SimAM
Conv5 1x1:    384 → 512
Classifier:   GAP → Dropout → Linear(512, 6)
```

Notes:
- Block pertama pada setiap stage menggunakan stride 2; block berikutnya
  menggunakan stride 1.
- Hasil verifikasi implementasi untuk klasifikasi enam kelas adalah 520,630
  parameter. Paper melaporkan 525,802 parameter.
- Selisih parameter sebesar 0.98% didokumentasikan sebagai bagian dari
  keterbatasan reproduksi custom architecture.

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

Semua model pada main experimental comparison menggunakan konfigurasi identik
agar Experiment 5/6/7 dapat dibandingkan langsung dengan Experiment 2/3/4.
Perbandingan utama mengisolasi perbedaan arsitektur student model, bukan perbedaan
resolusi input, learning rate, atau pretrained initialization.

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
DATASET_PATH = "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized"
OUTPUT_DIR = "/kaggle/working/"

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
IMG_SIZE = 380
```

Reason:
- identik dengan konfigurasi Focus-RCNet pada Phase 1
- mengisolasi pengaruh arsitektur student model
- memungkinkan perbandingan langsung Experiment 5/6/7 terhadap Experiment 2/3/4

---

## Optimization
```python
OPTIMIZER = "SGD"

LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
```

Notes:
- Learning rate disamakan dengan Phase 1 untuk menjaga controlled comparison.

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
TEACHER_CHECKPOINT = "efficientnet_b4_teacher_best.pth"
STUDENT_PRETRAINED = False
STUDENT_INITIALIZATION = "random initialization from timm"
```

Notes:
- Teacher menggunakan checkpoint EfficientNet-B4 hasil Notebook 1.
- EfficientNet-Lite0 pada main experimental comparison menggunakan
  `pretrained=False`, konsisten dengan Focus-RCNet yang dilatih dari random
  initialization.

---

## Model Source Libraries
```python
TEACHER_MODEL = "EfficientNet-B4"
TEACHER_SOURCE = "timm"
TEACHER_TIMM_NAME = "efficientnet_b4"

STUDENT_MODEL = "EfficientNet-Lite0"
STUDENT_SOURCE = "timm"
STUDENT_TIMM_NAME = "tf_efficientnet_lite0"
```

Notes:
- Script Notebook 3 memprioritaskan registry resmi `tf_efficientnet_lite0` dan
  memiliki runtime resolver untuk kompatibilitas antarversi `timm`.
- Arsitektur EfficientNet-Lite0 dibuat menggunakan `timm` dengan
  `pretrained=False` untuk main experimental comparison.
- Pretrained ImageNet weights untuk EfficientNet-Lite0 hanya dipakai pada
  conditional native-profile follow-up dan tidak dicampur ke tabel utama.

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
    "Resize": 380,
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
STAGE2_LR = 0.005
STAGE2_FREEZE_LAYERS = False
STAGE2_SCHEDULER = "CosineAnnealingLR"
STAGE2_T_MAX = STAGE2_EPOCHS
```

Notes:
- Stage 2 menggunakan separuh epoch dari Stage 1.
- Learning rate diturunkan 10x dari Phase 2 LR (0.05 → 0.005).
- Tidak ada layer freezing.

Output:
```text
EfficientNet-Lite0-KD-TwoStage
```

---

# Conditional Follow-Up — EfficientNet-Lite0 Native Profile

## Execution Rule
Track tambahan ini hanya dijalankan setelah Notebook 3 main experimental
comparison selesai dan hasil Experiment 5/6/7 telah dianalisis. Track ini bersifat
opsional apabila akurasi controlled comparison dinilai kurang memuaskan.

Track ini tidak dimasukkan ke controlled comparison utama dan tidak digunakan
sebagai lawan langsung Experiment 2/3/4 karena menggunakan konfigurasi native
yang berbeda.

## Native Configuration
```python
IMG_SIZE = 224
LR = 0.01
EPOCHS = 100
BATCH_SIZE = 8

STUDENT_PRETRAINED = True
STUDENT_WEIGHTS = "ImageNet"

KD_TEMPERATURE = 4
KD_ALPHA = 0.5

STAGE2_EPOCHS = 50
STAGE2_LR = 0.001
```

## Purpose
- mengevaluasi EfficientNet-Lite0 pada resolusi native dan pretrained ImageNet
- mengukur trade-off accuracy, FLOPs, inference time, dan memory usage untuk deployment
- menyediakan analisis tambahan tanpa mencampurkan hasil ke tabel controlled comparison

## Optional Output Artifacts
```text
efficientnet_lite0_native_baseline_best.pth
efficientnet_lite0_native_kd_scratch_best.pth
efficientnet_lite0_native_kd_twostage_best.pth
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

Total core experiments:
```text
7
```

Notes:
- Conditional EfficientNet-Lite0 Native Profile tidak termasuk ke matriks tujuh
  eksperimen utama.

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

---

# Notebook Execution Plan

Seluruh eksperimen dibagi ke dalam **3 notebook** yang dijalankan secara sequential di Kaggle.

---

## Notebook 1 — Teacher Model Training

### Content
```text
Experiment 1: Train EfficientNet-B4 (Teacher)
```

### Sections
```text
0. Setup (imports, config, seed, device)
1. Dataset & DataLoader (TrashNet, split 70/30, seed 42)
2. Model Definition (EfficientNet-B4 from timm, ImageNet pretrained)
3. Training Loop (100 epochs, SGD, CosineAnnealingLR)
4. Training Results (loss/accuracy curves, best metrics)
5. Save Checkpoint
```

### Configuration
```python
IMG_SIZE = 380
LR = 0.05
EPOCHS = 100
BATCH_SIZE = 8
```

### Input Dependencies
```text
- TrashNet dataset
```

### Output Artifacts
```text
- efficientnet_b4_teacher_best.pth
- training_history_exp1.csv
```

### Estimated Runtime
```text
~2-3 hours
```

---

## Notebook 2 — Phase 1: Focus-RCNet Experiments

### Content
```text
Experiment 2: Train Focus-RCNet (Baseline)
Experiment 3: KD From Scratch (EfficientNet-B4 → Focus-RCNet)
Experiment 4: Two-Stage KD (EfficientNet-B4 → Focus-RCNet)
```

### Sections
```text
0. Setup (imports, config, seed, device)
1. Dataset & DataLoader (TrashNet, split 70/30, seed 42, IMG_SIZE=380)
2. Load Teacher Model (EfficientNet-B4 from checkpoint)
3. Focus-RCNet Model Definition (custom implementation)
4. Experiment 2 — Train Focus-RCNet Baseline
5. Experiment 3 — KD From Scratch
6. Experiment 4 — Two-Stage KD (load Exp 2 checkpoint → fine-tune with KD)
7. Phase 1 Results Summary
8. Save All Checkpoints
```

### Configuration
```python
IMG_SIZE = 380
LR = 0.05
EPOCHS = 100
BATCH_SIZE = 8

# KD Config
KD_TEMPERATURE = 4
KD_ALPHA = 0.5

# Two-Stage (Experiment 4, Stage 2)
STAGE2_EPOCHS = 50
STAGE2_LR = 0.005
```

### Input Dependencies
```text
- TrashNet dataset
- efficientnet_b4_teacher_best.pth (from Notebook 1)
```

### Output Artifacts
```text
- focus_rcnet_baseline_best.pth
- focus_rcnet_kd_scratch_best.pth
- focus_rcnet_kd_twostage_best.pth
- training_history_exp2.csv
- training_history_exp3.csv
- training_history_exp4.csv
```

### Estimated Runtime
```text
~2-3 hours
```

### Actual Results

| Experiment | Best Validation Accuracy | Best Epoch |
|------------|--------------------------|------------|
| Exp 2: Focus-RCNet Baseline | 85.24% | 71 |
| Exp 3: Focus-RCNet KD Scratch | 85.64% | 87 |
| Exp 4: Focus-RCNet Two-Stage KD | 86.30% | 31 |

### Notes
- Experiment 4 Stage 1 tidak perlu di-retrain — langsung load checkpoint dari Experiment 2.
- Teacher model di-load dalam eval mode, tidak di-update selama KD training.
- KD Scratch meningkatkan validation accuracy sebesar 0.40 percentage point
  dibandingkan baseline.
- Two-Stage KD meningkatkan validation accuracy sebesar 1.05 percentage point
  dibandingkan baseline. Eksperimen ini adalah extension penelitian, bukan
  bagian dari paper original.

---

## Notebook 3 — Phase 2: EfficientNet-Lite0 Experiments + Final Evaluation

### Content
```text
Experiment 5: Train EfficientNet-Lite0 (Baseline)
Experiment 6: KD From Scratch (EfficientNet-B4 → EfficientNet-Lite0)
Experiment 7: Two-Stage KD (EfficientNet-B4 → EfficientNet-Lite0)
Final Evaluation & Comparison (All 7 Experiments)
```

### Sections
```text
0. Setup (imports, config, seed, device)
1. Dataset & DataLoader (TrashNet, split 70/30, seed 42, IMG_SIZE=380)
2. Load Teacher Model (EfficientNet-B4 from checkpoint)
3. EfficientNet-Lite0 Model Definition (from timm)
4. Experiment 5 — Train EfficientNet-Lite0 Baseline
5. Experiment 6 — KD From Scratch
6. Experiment 7 — Two-Stage KD (load Exp 5 checkpoint → fine-tune with KD)
7. Phase 2 Results Summary
8. Save All Checkpoints
9. Final Evaluation — Load All 7 Checkpoints
10. Comparison Tables (Accuracy, Precision, Recall, F1, Params, FLOPs)
11. Visualization (Confusion Matrices, ROC Curves, Training Curves)
```

### Configuration
```python
IMG_SIZE = 380
LR = 0.05
EPOCHS = 100
BATCH_SIZE = 8
STUDENT_PRETRAINED = False

# KD Config
KD_TEMPERATURE = 4
KD_ALPHA = 0.5

# Two-Stage (Experiment 7, Stage 2)
STAGE2_EPOCHS = 50
STAGE2_LR = 0.005
```

### Input Dependencies
```text
- TrashNet dataset
- efficientnet_b4_teacher_best.pth (from Notebook 1)
- All Phase 1 checkpoints (from Notebook 2, for final comparison only)
```

### Output Artifacts
```text
- efficientnet_lite0_baseline_best.pth
- efficientnet_lite0_kd_scratch_best.pth
- efficientnet_lite0_kd_twostage_best.pth
- training_history_exp5.csv
- training_history_exp6.csv
- training_history_exp7.csv
- final_comparison_table.csv
- confusion_matrices.png
- roc_curves.png
```

### Estimated Runtime
```text
~2-3 hours
```

### Notes
- Experiment 7 Stage 1 tidak perlu di-retrain — langsung load checkpoint dari Experiment 5.
- Final Evaluation membutuhkan semua checkpoint dari Notebook 1 dan 2 untuk perbandingan lengkap.
- Teacher dan student di Notebook 3 menggunakan IMG_SIZE=380 agar Experiment 5/6/7 dapat dibandingkan langsung dengan Experiment 2/3/4.
- EfficientNet-Lite0 Native Profile hanya dijalankan sebagai follow-up kondisional setelah hasil utama dianalisis dan dilaporkan terpisah.

---

## Execution Flow

```text
Notebook 1                 Notebook 2                    Notebook 3
──────────                 ──────────                    ──────────
Train B4 Teacher    →    Load Teacher                 →  Load Teacher
     │                      │                               │
     │                   Train Focus-RCNet (Exp 2)       Train Lite0 (Exp 5)
     │                      │                               │
     │                   KD Scratch (Exp 3)              KD Scratch (Exp 6)
     │                      │                               │
     │                   Two-Stage KD (Exp 4)            Two-Stage KD (Exp 7)
     │                      │                               │
Save checkpoint          Save checkpoints              Save checkpoints
                                                            │
                                                     Final Evaluation
                                                     (load ALL 7 checkpoints)
```

---

## Checkpoint Naming Convention
```python
CHECKPOINT_FORMAT = "{model_name}_{experiment_type}_best.pth"
```

| Experiment | Checkpoint Filename |
|------------|-------------------|
| 1 | `efficientnet_b4_teacher_best.pth` |
| 2 | `focus_rcnet_baseline_best.pth` |
| 3 | `focus_rcnet_kd_scratch_best.pth` |
| 4 | `focus_rcnet_kd_twostage_best.pth` |
| 5 | `efficientnet_lite0_baseline_best.pth` |
| 6 | `efficientnet_lite0_kd_scratch_best.pth` |
| 7 | `efficientnet_lite0_kd_twostage_best.pth` |