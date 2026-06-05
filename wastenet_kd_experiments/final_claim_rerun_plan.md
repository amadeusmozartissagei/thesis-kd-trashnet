# Final Claim Rerun Plan

## Purpose

Dokumen ini merangkum eksperimen yang perlu di-rerun untuk klaim final tesis
setelah protokol evaluasi diperkuat.

Hasil eksperimen sebelumnya dengan split 70/30 tetap dapat dipakai sebagai
preliminary atau exploratory result. Namun, untuk klaim final, model utama
sebaiknya dilatih dan dievaluasi ulang dengan split independen agar tidak ada
optimistic bias dari penggunaan validation set berulang kali.

## Final Evaluation Protocol

Recommended split:

```text
70% train
15% validation
15% independent test
```

Rules:

- Train set dipakai untuk training.
- Validation set dipakai untuk checkpoint selection dan hyperparameter choice.
- Test set hanya dipakai untuk final reporting.
- Test set tidak boleh dipakai untuk memilih model, epoch, temperature, alpha,
  atau konfigurasi training lain.
- Jalankan minimal 5 random seed untuk eksperimen utama.
- Simpan prediction-level CSV agar statistical test bisa dilakukan.

Recommended seeds:

```python
SEEDS = [42, 123, 777, 2026, 3407]
```

## Experiments That Need Rerun

| ID | Experiment | Teacher | Student / model | Why rerun is needed |
| --- | --- | --- | --- | --- |
| R1 | EfficientNet-B4 teacher | None | EfficientNet-B4 | Split berubah, jadi teacher final harus dilatih ulang tanpa melihat test set. |
| R2 | Focus-RCNet two-stage KD | EfficientNet-B4 | Focus-RCNet | Diperlukan jika Focus-RCNet dipakai sebagai teacher assistant untuk WasteNet. |
| R3 | WasteNet-123K baseline CE | None | WasteNet-123K | Baseline utama untuk klaim under 200K parameters. |
| R4 | WasteNet-123K direct KD | EfficientNet-B4 | WasteNet-123K | Menguji direct KD dari big teacher ke tiny student. |
| R5 | WasteNet-123K two-stage / TA-KD | Focus-RCNet | WasteNet-123K | Menguji teacher-assistant KD untuk target under 200K. |
| R6 | WasteNet-256K baseline CE | None | WasteNet-256K | Baseline pembanding dengan kapasitas lebih besar. |
| R7 | WasteNet-256K direct KD | EfficientNet-B4 | WasteNet-256K | Menguji direct KD pada WasteNet yang lebih besar. |
| R8 | WasteNet-256K two-stage / TA-KD | Focus-RCNet | WasteNet-256K | Menguji teacher-assistant KD pada WasteNet yang lebih besar. |

Total rerun untuk final claim: 8 experiment groups.

Jika semua dijalankan dengan 5 seed, total training run menjadi:

```text
8 experiment groups x 5 seeds = 40 runs
```

## Dependency Chain

```text
R1: EfficientNet-B4 teacher
  -> R2: Focus-RCNet two-stage KD teacher assistant
      -> R5: WasteNet-123K two-stage / TA-KD
      -> R8: WasteNet-256K two-stage / TA-KD

R1: EfficientNet-B4 teacher
  -> R4: WasteNet-123K direct KD
  -> R7: WasteNet-256K direct KD

R3: WasteNet-123K baseline CE
  -> R5: WasteNet-123K two-stage / TA-KD

R6: WasteNet-256K baseline CE
  -> R8: WasteNet-256K two-stage / TA-KD
```

## Notebook Plan

### Notebook A: Final Split And Teacher

Suggested filename:

```text
notebook_final_01_split_teacher_b4.py
```

Contents:

- Create stratified 70/15/15 split.
- Save train, validation, and test indices.
- Train EfficientNet-B4 teacher for each seed.
- Save teacher checkpoints and prediction CSV on validation and test sets.

Outputs:

```text
final_split_indices_seed_{seed}.json
efficientnet_b4_final_teacher_seed_{seed}.pth
predictions_b4_teacher_seed_{seed}.csv
```

### Notebook B: Final Focus-RCNet Teacher Assistant

Suggested filename:

```text
notebook_final_02_focus_rcnet_teacher_assistant.py
```

Contents:

- Train Focus-RCNet CE stage or load the CE stage if produced in the same run.
- Fine-tune Focus-RCNet with KD from EfficientNet-B4.
- Save Focus-RCNet teacher-assistant checkpoints.

Outputs:

```text
focus_rcnet_final_twostage_kd_seed_{seed}.pth
predictions_focus_rcnet_twostage_kd_seed_{seed}.csv
```

### Notebook C: Final WasteNet Baseline And Direct KD

Suggested filename:

```text
notebook_final_03_wastenet_baseline_direct_kd.py
```

Contents:

- WasteNet-123K baseline CE.
- WasteNet-123K direct KD from EfficientNet-B4.
- WasteNet-256K baseline CE.
- WasteNet-256K direct KD from EfficientNet-B4.

Outputs:

```text
wastenet_123k_final_baseline_ce_seed_{seed}.pth
wastenet_123k_final_direct_kd_b4_seed_{seed}.pth
wastenet_256k_final_baseline_ce_seed_{seed}.pth
wastenet_256k_final_direct_kd_b4_seed_{seed}.pth
predictions_wastenet_123k_baseline_ce_seed_{seed}.csv
predictions_wastenet_123k_direct_kd_b4_seed_{seed}.csv
predictions_wastenet_256k_baseline_ce_seed_{seed}.csv
predictions_wastenet_256k_direct_kd_b4_seed_{seed}.csv
```

### Notebook D: Final WasteNet Teacher-Assistant KD

Suggested filename:

```text
notebook_final_04_wastenet_teacher_assistant_kd.py
```

Contents:

- Load WasteNet-123K baseline checkpoint.
- Fine-tune WasteNet-123K with KD from Focus-RCNet teacher assistant.
- Load WasteNet-256K baseline checkpoint.
- Fine-tune WasteNet-256K with KD from Focus-RCNet teacher assistant.

Outputs:

```text
wastenet_123k_final_ta_twostage_kd_seed_{seed}.pth
wastenet_256k_final_ta_twostage_kd_seed_{seed}.pth
predictions_wastenet_123k_ta_twostage_kd_seed_{seed}.csv
predictions_wastenet_256k_ta_twostage_kd_seed_{seed}.csv
```

## KD Configuration

Default KD setup for first final rerun:

```python
KD_TEMPERATURE = 4
KD_ALPHA = 0.5
```

If temperature and alpha tuning is needed, tune only on validation set.

Small follow-up grid:

```python
TEMPERATURES = [2, 4, 6, 8]
ALPHAS = [0.1, 0.3, 0.5]
```

Do not use the independent test set for hyperparameter selection.

## Metrics For Final Claim

Report per seed and aggregate across seeds:

- Accuracy.
- Precision macro.
- Recall macro.
- F1 macro.
- AUC macro OVR.
- Parameters.
- FLOPs.
- Inference latency.
- Checkpoint size.

Aggregate format:

```text
mean
standard deviation
best seed
test-set confidence interval if available
```

## Statistical Testing

Prediction-level CSV is required for each model and seed.

Minimum CSV schema:

```text
image_path,label,prediction,prob_class_0,prob_class_1,prob_class_2,prob_class_3,prob_class_4,prob_class_5,seed,model_id
```

Recommended tests:

- McNemar test for paired accuracy comparison on the same test images.
- Bootstrap confidence interval for accuracy and F1 macro.
- Paired mean difference across 5 seeds.
- Holm-Bonferroni correction if many pairwise comparisons are reported.

## Main Final Comparisons

Primary:

```text
WasteNet-123K baseline CE
vs WasteNet-123K direct KD
vs WasteNet-123K teacher-assistant two-stage KD
```

Secondary:

```text
WasteNet-256K baseline CE
vs WasteNet-256K direct KD
vs WasteNet-256K teacher-assistant two-stage KD
```

Teacher-assistant evidence:

```text
EfficientNet-B4 -> WasteNet
vs EfficientNet-B4 -> Focus-RCNet -> WasteNet
```

Deployment interpretation:

- WasteNet-123K is the main deployment candidate for under 200K parameters.
- WasteNet-256K is a capacity trade-off comparison.
- EfficientNet-B4 and Focus-RCNet are training-time teachers, not deployment
  models.

## What Does Not Need Full Rerun Immediately

These can remain as exploratory or supporting experiments unless they become
part of the final claim:

- EfficientNet-Lite0 controlled experiments.
- EfficientNet-Lite0 native experiments.
- Old Focus-RCNet 70/30 experiments.
- Old alpha ablation runs.
- Old 200 epoch batch size 16 follow-ups.

They can still be cited as preliminary evidence, but final claims should rely
on the 70/15/15 independent-test protocol.

