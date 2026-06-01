# Experiment Results

Store downloaded Kaggle outputs here. Put executed notebooks in `notebooks/`
and machine-readable artifacts in `runs/<run-id>/`.

## Required Run Artifacts

```text
runs/<run-id>/
|-- config.json
|-- summary.json
|-- history.csv
|-- per_class_metrics.csv
|-- confusion_matrix.csv
|-- model_profile.json
`-- notes.md
```

Minimum contents:

- `config.json`: model variant, distillation method, seed, split, image size,
  epochs, optimizer, scheduler, and loss hyperparameters.
- `summary.json`: best epoch, accuracy, macro-F1, weighted-F1, and validation
  or test loss.
- `history.csv`: loss and metrics for each epoch.
- `per_class_metrics.csv`: precision, recall, F1, and support for each class.
- `confusion_matrix.csv`: final confusion matrix.
- `model_profile.json`: trainable parameters, FLOPs or MACs, model size, CPU
  latency, and peak memory when available.
- `notes.md`: anomalies, failed runs, and any manual observations.

Do not commit model checkpoints or datasets. They can be stored separately if
needed.
