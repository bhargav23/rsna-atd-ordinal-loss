"""
build_baseline_comparison_table.py

Assembles the full loss-function comparison table for spleen and kidney:
baseline CE, our ordinal loss (final config), LDAM, and weighted focal loss.
This is the actual head-to-head comparison Related Work (Sections 2.1, 2.2)
cites other methods for but never ran against our own data until now.

Reads four files:
  organ_oof_probabilities.csv   -- ordinal (final config: spleen a=3.0, kidney a=1.0)
  organ_baseline_oof.csv        -- baseline CE (spleen + kidney)
  organ_ldam_oof.csv            -- LDAM (spleen + kidney)
  organ_weightedfocal_oof.csv   -- weighted focal loss (spleen + kidney)

All four use the identical 5-fold CV protocol and OrganClassifier
architecture, differing only in loss function -- a clean, controlled
comparison.

Metrics computed with the SAME convention as Table 1/2 throughout this
paper: per-fold accuracy/recall, averaged across folds (not pooled).
"""

import numpy as np
import pandas as pd
from pathlib import Path

PATHS = {
    "Ordinal (final)": Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_oof_probabilities.csv"),
    "Baseline CE": Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_baseline_oof.csv"),
    "LDAM": Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_ldam_oof.csv"),
    "Weighted Focal": Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_weightedfocal_oof.csv"),
}

ORGANS = ["spleen", "kidney"]
HEALTHY, LOW, HIGH = 0, 1, 2


def per_fold_metric(oof_df, organ, metric="accuracy", target_class=None):
    sub = oof_df[oof_df["organ"] == organ].copy()
    if sub.empty:
        return None, None
    prob_cols = sorted([c for c in sub.columns if c.startswith("prob_class")])
    sub["pred_label"] = sub[prob_cols].values.argmax(axis=1)

    fold_vals = []
    for fold, g in sub.groupby("fold"):
        if metric == "accuracy":
            fold_vals.append((g["pred_label"] == g["true_label"]).mean())
        elif metric == "recall":
            mask = g["true_label"] == target_class
            if mask.sum() == 0:
                continue
            fold_vals.append((g.loc[mask, "pred_label"] == target_class).mean())

    if not fold_vals:
        return None, None
    return float(np.mean(fold_vals)), float(np.std(fold_vals, ddof=1)) if len(fold_vals) > 1 else 0.0


def main():
    loaded = {}
    for name, path in PATHS.items():
        if not path.exists():
            print(f"WARNING: {name} file not found at {path} -- skipping this config entirely.")
            continue
        loaded[name] = pd.read_csv(path)

    rows = []
    for organ in ORGANS:
        for config_name, df in loaded.items():
            acc_mean, acc_sd = per_fold_metric(df, organ, metric="accuracy")
            healthy_recall, _ = per_fold_metric(df, organ, metric="recall", target_class=HEALTHY)
            high_recall, _ = per_fold_metric(df, organ, metric="recall", target_class=HIGH)
            if acc_mean is None:
                print(f"NOTE: {config_name} has no rows for organ='{organ}' -- skipping this cell.")
                continue
            rows.append({
                "Organ": organ.capitalize(),
                "Config": config_name,
                "val_acc": f"{acc_mean:.3f} ± {acc_sd:.3f}",
                "Healthy recall": f"{healthy_recall:.2f}" if healthy_recall is not None else "N/A",
                "High-grade recall": f"{high_recall:.2f}" if high_recall is not None else "N/A",
            })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    out_path = Path("baseline_comparison_table.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path.resolve()}")

    print("\n" + "=" * 70)
    print("Reference point -- this paper's ordinal loss vs. baseline (already established):")
    print("  Spleen: baseline val_acc 0.479, ordinal 0.500 (healthy 0.055->0.37, high 0.93->0.72)")
    print("  Kidney: baseline val_acc 0.458, ordinal 0.565 (healthy 0.44->0.66, high 0.48->0.64)")
    print("Compare LDAM and Weighted Focal rows above against these to see whether")
    print("standard imbalance-correction baselines match, beat, or underperform")
    print("our ordinal loss on the same data/protocol.")
    print("=" * 70)


if __name__ == "__main__":
    main()
