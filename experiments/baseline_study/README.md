# Baseline Study

This folder archives the experiments completed before the Focus-GhostNet study.
The scripts are ordered by research flow rather than by filename prefixes.

## Research Flow

| Order | Script | Purpose | Executed notebook |
| --- | --- | --- | --- |
| 1 | `kaggle/teacher_efficientnet_b4.py` | Train the shared EfficientNet-B4 teacher | `results/notebooks/teacher_efficientnet_b4.ipynb` |
| 2 | `kaggle/focus_rcnet_controlled.py` | Reproduce Focus-RCNet baseline and KD variants | `results/notebooks/focus_rcnet_controlled.ipynb` |
| 3 | `kaggle/focus_rcnet_kd_alpha_ablation.py` | Test lower KD alpha values for Focus-RCNet | Not downloaded yet |
| 4 | `kaggle/efficientnet_lite0_controlled.py` | Compare EfficientNet-Lite0 under the controlled protocol | `results/notebooks/efficientnet_lite0_controlled.ipynb` |
| 5 | `kaggle/efficientnet_lite0_native.py` | Evaluate Lite0 at its native deployment profile | `results/notebooks/efficientnet_lite0_native.ipynb` |
| 6 | `kaggle/efficientnet_lite0_native_kd_alpha_ablation.py` | Test lower KD alpha values for native Lite0 | `results/notebooks/efficientnet_lite0_native_kd_alpha_ablation.ipynb` |

## Folder Structure

```text
baseline_study/
|-- kaggle/              # Python scripts uploaded and executed on Kaggle
|-- results/
|   |-- notebooks/       # Downloaded executed notebooks
|   `-- runs/            # Extracted CSV and JSON artifacts for future analysis
`-- README.md
```

The Python files retain their original notebook headings because those headings
record the execution order used in the study. Local filenames are semantic so
they remain understandable after additional experiments are added.

## Kaggle Inputs

Some scripts refer to existing Kaggle notebook input slugs such as
`notebook1-teacher-training`. Renaming files locally does not change those
published Kaggle inputs. If an upstream Kaggle notebook is republished under a
new slug, update the corresponding input path before rerunning a dependent
script.
