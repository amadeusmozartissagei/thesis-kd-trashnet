# Focus-GhostNet Experiments

This folder contains experiments for comparing Focus-RCNet with a lightweight
Focus-GhostNet student model. The teacher model remains EfficientNet-B4 so the
comparison stays fair.

## Folder Structure

```text
focus_ghostnet/
|-- kaggle/              # Python scripts uploaded and executed on Kaggle
|-- results/             # Downloaded outputs from Kaggle
|   |-- notebooks/       # Executed Jupyter notebooks
|   `-- runs/            # Metrics and reports grouped by run ID
`-- README.md
```

## Suggested Experiment Order

1. Reproduce Focus-RCNet with vanilla knowledge distillation.
2. Train Focus-GhostNet without knowledge distillation.
3. Train Focus-GhostNet with vanilla knowledge distillation.
4. Train Focus-GhostNet with decoupled knowledge distillation (DKD).
5. Train Focus-GhostNet with DKD and SimAM.

Keep the dataset split, teacher checkpoint, input size, optimizer, and random
seeds fixed unless the experiment explicitly studies one of those variables.

## Run Naming

Use a stable ID for each run:

```text
<student>__<distillation>__seed-<seed>
```

Examples:

```text
focus_rcnet__vanilla-kd__seed-42
focus_ghostnet__no-kd__seed-42
focus_ghostnet__dkd-simam__seed-42
```

See [results/README.md](results/README.md) for the required output files.
