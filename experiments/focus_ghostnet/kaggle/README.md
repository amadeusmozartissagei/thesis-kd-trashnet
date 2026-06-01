# Kaggle Scripts

Place Python files that will be uploaded and run on Kaggle in this folder.

Keep training code in Python scripts rather than only inside notebooks. This
makes each experiment easier to reproduce and compare. A Kaggle notebook may
import the scripts and contain only setup commands, dataset paths, and the
training entry point.

Suggested files for the first implementation:

```text
kaggle/
|-- train.py             # Training and evaluation entry point
|-- models.py            # Focus-RCNet and Focus-GhostNet students
|-- distillation.py      # Vanilla KD and DKD losses
|-- data.py              # Dataset loading and fixed stratified splits
|-- metrics.py           # Accuracy, macro-F1, per-class metrics, and latency
`-- config.py            # Shared experiment settings
```

Each run should export the files listed in
[results/README.md](../results/README.md).
