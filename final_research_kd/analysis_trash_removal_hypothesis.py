"""
Proxy analysis (no retraining): does removing the "trash" class look like it
would help, based on existing 6-class model predictions on the test set?

For each R-group (R1 teacher, R3/R6 CE baselines, R5/R8 TA-KD students):
  1. Confusion involving trash: how much trash <-> non-trash bleed exists.
  2. 5-class proxy: drop trash-labeled test samples, re-argmax over the
     remaining 5 prob columns (renormalized), recompute accuracy/F1-macro,
     and compare against the original 6-class metrics on the same samples.

This is a lower-bound proxy only: a real 5-class model would learn different
decision boundaries. It does not replace retraining, only gives early signal.
"""
import glob
import os
import csv
from collections import defaultdict

BASE = "final_research_kd/runs/final"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NON_TRASH = [c for c in CLASSES if c != "trash"]

GROUPS = {
    "R1_EfficientNetB4_teacher": "R1",
    "R3_WasteNet128K_CE": "R3",
    "R6_WasteNet256K_CE": "R6",
    "R5_WasteNet128K_TAKD": "R5",
    "R8_WasteNet256K_TAKD": "R8",
}


def load_predictions(rgroup):
    pattern = os.path.join(BASE, rgroup, "seed_*", f"predictions_*_test.csv")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)
    return rows


def macro_f1(rows, classes, true_key="label", pred_key="prediction"):
    per_class = {}
    for c in classes:
        tp = sum(1 for r in rows if r[true_key] == c and r[pred_key] == c)
        fp = sum(1 for r in rows if r[true_key] != c and r[pred_key] == c)
        fn = sum(1 for r in rows if r[true_key] == c and r[pred_key] != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = (prec, rec, f1)
    macro_p = sum(v[0] for v in per_class.values()) / len(classes)
    macro_r = sum(v[1] for v in per_class.values()) / len(classes)
    macro_f = sum(v[2] for v in per_class.values()) / len(classes)
    return per_class, macro_p, macro_r, macro_f


def accuracy(rows, true_key="label", pred_key="prediction"):
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if r[true_key] == r[pred_key])
    return correct / len(rows)


def confusion_trash_bleed(rows):
    # non-trash predicted as trash (false positives stolen from other classes)
    fp_into_trash = defaultdict(int)
    # trash predicted as other classes (false negatives, trash lost to others)
    fn_from_trash = defaultdict(int)
    for r in rows:
        if r["label"] != "trash" and r["prediction"] == "trash":
            fp_into_trash[r["label"]] += 1
        if r["label"] == "trash" and r["prediction"] != "trash":
            fn_from_trash[r["prediction"]] += 1
    return fp_into_trash, fn_from_trash


def proxy_5class_argmax(row):
    probs = {c: float(row[f"prob_{c}"]) for c in NON_TRASH}
    return max(probs, key=probs.get)


def main():
    print("=" * 80)
    print("PROXY ANALYSIS: effect of removing 'trash' class (no retraining)")
    print("=" * 80)

    for label, rgroup in GROUPS.items():
        rows = load_predictions(rgroup)
        if not rows:
            print(f"\n[{label}] no prediction files found, skipping")
            continue

        n_total = len(rows)
        n_seeds = len(set(r["seed"] for r in rows))

        # --- original 6-class metrics (on these same pooled-seed rows) ---
        acc6 = accuracy(rows)
        per_class6, p6, r6, f16 = macro_f1(rows, CLASSES)

        # --- confusion bleed involving trash ---
        fp_into_trash, fn_from_trash = confusion_trash_bleed(rows)
        n_trash_true = sum(1 for r in rows if r["label"] == "trash")
        n_trash_pred = sum(1 for r in rows if r["prediction"] == "trash")
        total_fp_into_trash = sum(fp_into_trash.values())
        total_fn_from_trash = sum(fn_from_trash.values())

        # --- 5-class proxy: drop trash-labeled samples, re-argmax over 5 ---
        non_trash_rows = [r for r in rows if r["label"] != "trash"]
        for r in non_trash_rows:
            r["proxy5_prediction"] = proxy_5class_argmax(r)
        acc5 = accuracy(non_trash_rows, pred_key="proxy5_prediction")
        per_class5, p5, r5_, f15 = macro_f1(
            non_trash_rows, NON_TRASH, pred_key="proxy5_prediction"
        )

        # also recompute original 6-class metrics restricted to the same
        # non-trash-labeled subset, for an apples-to-apples comparison
        acc6_subset = accuracy(non_trash_rows)
        _, p6s, r6s, f16s = macro_f1(non_trash_rows, NON_TRASH)

        print(f"\n[{label}]  ({rgroup}, pooled {n_seeds} seeds, n={n_total})")
        print(f"  6-class test accuracy : {acc6*100:.2f}%   F1-macro: {f16*100:.2f}%")
        print(f"  trash F1 (6-class)    : {per_class6['trash'][2]*100:.2f}%  "
              f"(prec {per_class6['trash'][0]*100:.2f}%, rec {per_class6['trash'][1]*100:.2f}%)")
        print(f"  trash true count      : {n_trash_true}   trash predicted count: {n_trash_pred}")
        print(f"  non-trash -> trash FP : {total_fp_into_trash} "
              f"({dict(fp_into_trash)})")
        print(f"  trash -> non-trash FN : {total_fn_from_trash} "
              f"({dict(fn_from_trash)})")
        print(f"  --- proxy 5-class (drop trash-labeled samples, re-argmax over 5) ---")
        print(f"  5-class proxy accuracy: {acc5*100:.2f}%   F1-macro: {f15*100:.2f}%")
        print(f"  (same subset, original 6-class argmax: acc {acc6_subset*100:.2f}%, "
              f"F1-macro {f16s*100:.2f}%)")
        delta_acc = (acc5 - acc6_subset) * 100
        delta_f1 = (f15 - f16s) * 100
        print(f"  >> delta from dropping trash column: "
              f"accuracy {delta_acc:+.2f} pts, F1-macro {delta_f1:+.2f} pts")


if __name__ == "__main__":
    main()
