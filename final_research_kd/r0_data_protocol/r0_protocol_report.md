# R0 Final Data Protocol Report

Generated at UTC: `2026-06-12T11:47:37.286006+00:00`

## Dataset

- Dataset: TrashNet
- Dataset directory: `/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized`
- Raw valid images: `2527`
- Excluded duplicate-conflict images: `6`
- Final valid images after exclusion: `2521`
- Classes: `cardboard, glass, metal, paper, plastic, trash`

## Split Protocol

- Train: `0.70`
- Validation: `0.15`
- Independent test: `0.15`
- Seeds: `[42, 123, 777, 2026, 3407]`

## Rule

R0 only creates the dataset inventory, class mapping, duplicate-conflict exclusion list, and split manifests. It does not select the best model setup. Exact duplicate images with conflicting labels are excluded before splitting to prevent label ambiguity and cross-split leakage. Hyperparameter tuning is done later in pilot runs using the validation split only. The independent test split is locked until final reporting.
