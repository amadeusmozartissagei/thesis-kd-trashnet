# R1 Final 5-Seed Summary

## Status

R1 EfficientNet-B4 teacher is complete and valid for final reporting.

- Seeds: 42, 123, 777, 2026, 3407
- Run phase: final
- Epochs per seed: 100
- Early stopping: false
- Test evaluation: enabled
- Final judgment: satisfactory; use as final EfficientNet-B4 teacher.

## Aggregate Metrics

| Split | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std | AUC mean | AUC std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 96.77% | 1.05% | 95.89% | 1.48% | 99.74% | 0.14% |
| Test | 94.46% | 0.7% | 93.36% | 0.64% | 99.54% | 0.08% |

## Per-Seed Test Metrics

| Seed | Accuracy | Macro F1 | AUC | Best epoch |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 94.72% | 93.81% | 99.53% | 44 |
| 123 | 94.2% | 93.04% | 99.55% | 100 |
| 777 | 94.2% | 93.31% | 99.45% | 59 |
| 2026 | 93.67% | 92.5% | 99.51% | 22 |
| 3407 | 95.51% | 94.13% | 99.68% | 88 |

## Class-Level Notes

Weakest aggregate test class is trash:

- Precision: 80.91%
- Recall: 89%
- F1: 84.76%

Top confusion pairs on test set across 5 seeds:

- glass -> plastic: 13
- cardboard -> paper: 11
- glass -> metal: 10
- paper -> trash: 9
- paper -> cardboard: 8
- plastic -> trash: 8
- plastic -> glass: 8
- trash -> paper: 7

## Decision

R1 is strong enough to be frozen as the final teacher. Do not tune R1 further. The next step is R2 Focus-RCNet teacher-assistant selection, with R1 checkpoints used as the EfficientNet-B4 teacher source.
