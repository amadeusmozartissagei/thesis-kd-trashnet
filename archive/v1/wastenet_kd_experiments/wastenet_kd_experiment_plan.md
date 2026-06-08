# WasteNet KD Experiment Plan

## Objective

Tujuan eksperimen lanjutan adalah mengevaluasi apakah WasteNet dapat menjadi
student deployment utama dengan parameter kecil, terutama untuk target under
200K parameters agar lebih relevan untuk ESP32-class deployment.

Eksperimen ini melanjutkan hasil sebelumnya:

- EfficientNet-B4 sudah tersedia sebagai high-capacity teacher.
- Focus-RCNet hasil KD/two-stage KD digunakan sebagai teacher assistant.
- KD yang dipakai tetap logits-based vanilla KD:
  `alpha * KLDiv(student_logits / T, teacher_logits / T) * T^2 + (1 - alpha) * CE`.

WasteNet-128K menjadi kandidat deployment utama karena berada di bawah target
200K parameters. WasteNet-256K dipakai sebagai pembanding capacity trade-off,
tetapi tidak boleh diklaim memenuhi target under 200K jika batas tersebut
dipakai secara strict.

## Planned Experiment Matrix

| ID | Model | Training mode | Teacher | Purpose |
| --- | --- | --- | --- | --- |
| WN128-CE | WasteNet-128K | Baseline CE | None | Baseline utama under 200K |
| WN128-DKD | WasteNet-128K | Direct KD | EfficientNet-B4 | Uji direct big-teacher KD ke tiny student |
| WN128-TA2KD | WasteNet-128K | Two-stage / teacher-assistant KD | Focus-RCNet KD | Uji apakah intermediate teacher membantu capacity gap |
| WN256-CE | WasteNet-256K | Baseline CE | None | Baseline pembanding kapasitas lebih besar |
| WN256-DKD | WasteNet-256K | Direct KD | EfficientNet-B4 | Uji direct KD pada WasteNet lebih besar |
| WN256-TA2KD | WasteNet-256K | Two-stage / teacher-assistant KD | Focus-RCNet KD | Uji teacher-assistant KD pada WasteNet lebih besar |

Total eksperimen baru: 6.

Notebook 4 sudah difinalkan sebagai controlled exploratory run dengan 100 epoch
untuk WasteNet. Plan Notebook 5 harus mengikuti konfigurasi final tersebut agar
perbandingan tetap sejajar.

## Notebook Layout

### Notebook 4: WasteNet Baseline + Direct KD

Suggested filename:

```text
notebook4_wastenet_baseline_direct_kd.py
```

Isi:

1. WasteNet-128K baseline CE.
2. WasteNet-128K direct KD from EfficientNet-B4.
3. WasteNet-256K baseline CE.
4. WasteNet-256K direct KD from EfficientNet-B4.

Input dependencies:

- TrashNet dataset.
- EfficientNet-B4 teacher checkpoint.
- Existing split indices, if continuing the controlled comparison protocol.

Expected outputs:

```text
wastenet_128k_baseline_ce_best.pth
wastenet_128k_direct_kd_b4_best.pth
wastenet_256k_baseline_ce_best.pth
wastenet_256k_direct_kd_b4_best.pth
training_history_wastenet_128k_baseline_ce.csv
training_history_wastenet_128k_direct_kd_b4.csv
training_history_wastenet_256k_baseline_ce.csv
training_history_wastenet_256k_direct_kd_b4.csv
predictions_wastenet_128k_baseline_ce.csv
predictions_wastenet_128k_direct_kd_b4.csv
predictions_wastenet_256k_baseline_ce.csv
predictions_wastenet_256k_direct_kd_b4.csv
wastenet_baseline_direct_kd_comparison.csv
```

### Notebook 5: WasteNet Teacher-Assistant Two-Stage KD

Suggested filename:

```text
notebook5_wastenet_teacher_assistant_twostage_kd.py
```

Isi:

1. Load WasteNet-128K CE checkpoint from Notebook 4.
2. Fine-tune WasteNet-128K with KD from Focus-RCNet KD teacher assistant.
3. Load WasteNet-256K CE checkpoint from Notebook 4.
4. Fine-tune WasteNet-256K with KD from Focus-RCNet KD teacher assistant.

Two-stage definition for WasteNet:

```text
Stage 1: train WasteNet with CrossEntropyLoss.
Stage 2: load best CE checkpoint, then fine-tune with KD loss.
```

Teacher-assistant chain:

```text
EfficientNet-B4 -> Focus-RCNet KD -> WasteNet
```

Important interpretation:

- EfficientNet-B4 and Focus-RCNet are training-time teachers only.
- Final deployment model remains WasteNet.
- Focus-RCNet parameter count does not affect deployment cost.

Input dependencies:

- Primary teacher assistant checkpoint:
  `focus_rcnet_direct_kd_e200_bs16_t4_a05_best.pth`.
- Optional teacher-assistant fallback or ablation checkpoint:
  `focus_rcnet_twostage_kd_e200_bs16_t4_a05_best.pth`.
- WasteNet-128K CE checkpoint from Notebook 4:
  `wastenet_128k_baseline_ce_best.pth`.
- WasteNet-256K CE checkpoint from Notebook 4:
  `wastenet_256k_baseline_ce_best.pth`.
- Notebook 4 comparison CSV, if attached:
  `wastenet_baseline_direct_kd_comparison.csv`.
- Same split indices used in Notebook 4. Load `train_indices` and `val_indices`
  from the WasteNet CE checkpoint, then assert that both WasteNet CE
  checkpoints and the Focus-RCNet teacher-assistant checkpoint use the same
  split when those fields are available.

Expected outputs:

```text
wastenet_128k_ta_twostage_kd_best.pth
wastenet_256k_ta_twostage_kd_best.pth
training_history_wastenet_128k_ta_twostage_kd.csv
training_history_wastenet_256k_ta_twostage_kd.csv
predictions_wastenet_128k_ta_twostage_kd.csv
predictions_wastenet_256k_ta_twostage_kd.csv
wastenet_teacher_assistant_twostage_comparison.csv
wastenet_all_core_experiments_comparison.csv
```

## Default Training Configuration

Initial WasteNet runs should stay close to the finalized Notebook 4 controlled
exploratory protocol unless the final validation protocol is started
immediately.

Recommended default:

```python
IMG_SIZE = 160
EPOCHS = 100
BATCH_SIZE = 16
OPTIMIZER = "SGD"
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SCHEDULER = "CosineAnnealingLR"
KD_TEMPERATURE = 4
KD_ALPHA = 0.5
STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 50
STAGE2_LR = 0.005
USE_AMP = True
```

Notes:

- Direct KD trains WasteNet from initialization using KD loss.
- Two-stage KD loads the best WasteNet CE checkpoint from Notebook 4 before KD
  fine-tuning; Stage 1 is not retrained inside Notebook 5.
- Teacher models must be in `eval()` mode with all parameters frozen.
- Save train and validation indices inside every checkpoint.
- Save prediction-level CSV for later statistical testing.
- Notebook 5 should save the teacher-assistant checkpoint path and the Stage 1
  WasteNet checkpoint path inside each output checkpoint.

## Split And Validation Plan

There are two possible tracks.

### Track A: Controlled Continuation

Use the existing stratified 70/30 split with seed 42 to keep results comparable
with the current Focus-RCNet and EfficientNet-Lite0 experiments.

This is useful for quick comparison, but it should not be the final unbiased
claim because the validation split has already been used repeatedly for model
selection.

### Track B: Final Validation Protocol

For final thesis claims, use:

```text
70% train
15% validation
15% independent test
```

Rules:

- Use validation only for checkpoint selection and hyperparameter tuning.
- Use test only once for final reporting.
- Run at least 5 random seeds.
- Save prediction CSV for each model and seed.
- Report mean, standard deviation, and confidence intervals where possible.

Recommended seeds:

```python
SEEDS = [42, 123, 777, 2026, 3407]
```

## Temperature And Alpha Follow-Up

Do not run a large grid before the six core WasteNet experiments are complete.
Start with:

```python
KD_TEMPERATURE = 4
KD_ALPHA = 0.5
```

If direct KD underperforms baseline or teacher-assistant KD, run a small ablation
on the best candidate only:

```python
TEMPERATURES = [2, 4, 6, 8]
ALPHAS = [0.1, 0.3, 0.5]
```

Use validation results for hyperparameter choice. Do not use the independent
test split for choosing temperature or alpha.

Notebook 4 showed direct KD from EfficientNet-B4 underperforming CE for both
WasteNet capacities. Finish Notebook 5 first before any alpha/temperature
ablation, because teacher-assistant KD is the intended capacity-gap follow-up.

## Metrics To Report

Classification metrics:

- Accuracy.
- Precision macro.
- Recall macro.
- F1 macro.
- AUC macro OVR.
- Confusion matrix.

Efficiency metrics:

- Total parameters.
- FLOPs.
- Inference latency.
- Checkpoint size.
- Optional: int8 model size if deployment conversion is tested.
- Optional: peak activation memory estimate for ESP32 relevance.

## Main Comparisons

Primary comparisons:

```text
WasteNet-128K CE vs WasteNet-128K direct KD
WasteNet-128K CE vs WasteNet-128K teacher-assistant two-stage KD
WasteNet-128K direct KD vs WasteNet-128K teacher-assistant two-stage KD
```

Secondary comparisons:

```text
WasteNet-256K CE vs WasteNet-256K direct KD
WasteNet-256K CE vs WasteNet-256K teacher-assistant two-stage KD
WasteNet-128K best vs WasteNet-256K best
```

Interpretation rules:

- If WasteNet-128K wins or is close to WasteNet-256K, emphasize deployment
  efficiency.
- If WasteNet-256K wins clearly, report it as a capacity trade-off, not as the
  under-200K deployment solution.
- If teacher-assistant KD beats direct KD, frame it as evidence that
  intermediate-teacher distillation helps bridge the capacity gap.
- If direct KD beats teacher-assistant KD, keep teacher-assistant KD as a tested
  hypothesis and report the result honestly.

## Statistical Testing Plan

For final test-set predictions:

- Use McNemar test for paired accuracy comparison on the same test images.
- Use bootstrap confidence intervals for accuracy and F1 macro.
- With 5 seeds, also report paired mean differences across seeds.
- If many model pairs are compared, apply a multiple-comparison correction such
  as Holm-Bonferroni.

Minimum artifact for testing:

```text
image_path,label,prediction,prob_class_0,prob_class_1,...,prob_class_5,seed,model_id
```

## Proposed Narrative

The thesis narrative can position the new experiments as follows:

1. EfficientNet-B4 is the high-capacity source teacher.
2. Focus-RCNet KD acts as teacher assistant.
3. WasteNet is the final tiny deployment student.
4. The novelty is not just using KD, but testing whether two-stage /
   teacher-assistant KD is more suitable than direct KD when compressing toward
   under-200K parameters.
