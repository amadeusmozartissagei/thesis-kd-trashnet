# R0-K5 Data Protocol Report

Generated at UTC: `2026-06-20T17:03:44.074826+00:00`

## Dataset

- Dataset: TrashNet-K5
- Kaggle source: `kholiqbudiman/trashnet-k5-waste-classification`
- Audited dataset version: `2`
- Dataset directory: `/kaggle/input/datasets/kholiqbudiman/trashnet-k5-waste-classification`
- Raw valid images: `2390`
- Excluded duplicate-conflict images: `6`
- Final valid images after exclusion: `2384`
- Classes: `cardboard, glass, metal, paper, plastic`
- Bundled source split: pooled and replaced by the protocol below

## Split Protocol

- Train: `0.70`
- Validation: `0.15`
- Independent test: `0.15`
- Seeds: `[42, 123, 777, 2026, 3407]`

## Rule

R0-K5 only creates the dataset inventory, class mapping, duplicate-conflict exclusion list, and split manifests. It does not select the best model setup. The dataset's bundled 80/20 assignment is recorded as provenance but pooled before resplitting. Exact duplicate images with conflicting labels are excluded before splitting to prevent label ambiguity and cross-split leakage. Hyperparameter tuning is done later in pilot runs using the validation split only. The independent test split is locked until final reporting.
