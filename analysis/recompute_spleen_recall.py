"""
recompute_spleen_recall.py

Checks whether Table 2's spleen recall numbers (healthy 0.055 -> 0.44,
high-grade 0.93 -> 0.70) still match what's in the CURRENT
organ_oof_probabilities.csv, now that we know that file's headline accuracy
(~0.500) doesn't match Table 1's cited 0.538 +/- 0.051 for the same config.

This does NOT try to recompute the BASELINE side of Table 2 (0.055 healthy /
0.93 high-grade) from organ_oof_probabilities.csv, since that file only
contains the FINAL (ordinal) config for spleen -- the baseline comparison
now lives in organ_baseline_oof.csv (the file generated for the Wilcoxon
test). Both are read here so the full baseline-vs-ordinal recall comparison
can be recomputed end to end, exactly mirroring what Table 2 reports.

Recall is computed PER FOLD then averaged (matching how Table 1's val_acc
was computed), not pooled across all folds at once -- pooling would silently
weight larger folds more heavily and wouldn't match the existing table's
convention.
"""

import numpy as np
import pandas as pd
from pathlib import Path

FINAL_OOF_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_oof_probabilities.csv")
BASELINE_OOF_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_baseline_oof.csv")

# Class indices, per get_organ_label(): 0=healthy, 1=low-grade, 2=high-grade
HEALTHY, LOW, HIGH = 0, 1, 2


def per_fold_recall(oof_df, organ, target_class):
    """Per-fold recall for one class, then averaged across folds (unweighted
    mean of fold-level recalls, matching Table 1's val_acc convention)."""
    sub = oof_df[oof_df["organ"] == organ].copy()
    prob_cols = sorted([c for c in sub.columns if c.startswith("prob_class")])
    sub["pred_label"] = sub[prob_cols].values.argmax(axis=1)

    fold_recalls = []
    for fold, g in sub.groupby("fold"):
        mask = g["true_label"] == target_class
        n_positive = mask.sum()
        if n_positive == 0:
            continue  # can't compute recall for a fold with zero true positives of this class
        recall = (g.loc[mask, "pred_label"] == target_class).mean()
        fold_recalls.append(recall)

    if not fold_recalls:
        return None, 0
    return float(np.mean(fold_recalls)), len(fold_recalls)


def main():
    final_oof_df = pd.read_csv(FINAL_OOF_PATH)
    baseline_oof_df = pd.read_csv(BASELINE_OOF_PATH) if BASELINE_OOF_PATH.exists() else None

    print("=" * 70)
    print("SPLEEN — recomputed from current OOF exports")
    print("=" * 70)

    for label_name, class_idx in [("healthy", HEALTHY), ("high-grade", HIGH)]:
        final_recall, n_folds_final = per_fold_recall(final_oof_df, "spleen", class_idx)
        print(f"\n{label_name.capitalize()} recall (ordinal/final config):")
        print(f"  Recomputed: {final_recall:.3f}" if final_recall is not None else "  Recomputed: N/A (no positive folds)")
        print(f"  ({n_folds_final}/5 folds had at least one {label_name} example)")

        if baseline_oof_df is not None:
            baseline_recall, n_folds_base = per_fold_recall(baseline_oof_df, "spleen", class_idx)
            print(f"Baseline recall:")
            print(f"  Recomputed: {baseline_recall:.3f}" if baseline_recall is not None else "  Recomputed: N/A")
            print(f"  ({n_folds_base}/5 folds had at least one {label_name} example)")

    print("\n" + "=" * 70)
    print("Table 2 currently cites: healthy 0.055 -> 0.44, high-grade 0.93 -> 0.70")
    print("Compare those against the 'Recomputed' values above.")
    print("If they diverge similarly to Table 1's accuracy mismatch (~4pp+),")
    print("Table 2 needs the same correction/update as Table 1.")
    print("=" * 70)


if __name__ == "__main__":
    main()
