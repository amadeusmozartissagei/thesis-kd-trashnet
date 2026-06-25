---
name: k5-r2-run-workflow
description: How the user runs K5 final notebooks on Kaggle — one seed per run, edit one line
metadata:
  type: feedback
---

The user runs K5 notebooks by uploading to Kaggle and running interactively. The run scheme mirrors K6 exactly: **one seed per Run All**, controlled by a single plain line in the Config cell:

```python
SEED = 42  # Final seeds: 42, 123, 777, 2026, 3407
```

Workflow: edit that number → Run All → output saves to its own `runs/final/<EXP>/seed_<seed>/` (no collision) → repeat for next seed. The user often asks me to edit that one line ("ganti ke seed berikutnya" / "next") — just change the number in the notebook, nothing else.

**Why:** The user wants the simple K6 scheme, not clever indirection.
**How to apply:** Keep the seed as a plain `SEED = <n>` constant with the inline comment. Do NOT add env-var overrides, parallel-run schemes, or extra orchestration — that was tried and the user found it confusing/over-engineered. Keep notebook changes minimal and matched to the K6 notebook of the same number. See [[k5-study-progress]].
