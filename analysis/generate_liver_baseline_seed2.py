"""
generate_liver_baseline_seed2.py

Robustness check: reruns liver's BASELINE config with a different seed
(123, the second entry in phase1_organ_branch.py's own SEEDS list) to see
whether baseline's 0.100 high-grade recall is a real, reproducible signal
or a single-fold-flip artifact of one run.

Same session requirement as the other generate_*.py scripts: needs
train_organ_cv_with_oof and organ_manifest_cv already defined in this
kernel (run liver_ordinal_check_full.py's setup portion first if you
haven't already this session).

Prints per-class recall directly at the end -- no separate script needed
to read the result.

Output: /kaggle/working/liver_baseline_seed123_oof.csv
"""

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("/kaggle/working/liver_baseline_seed123_oof.csv")
SEED = 123


def per_fold_recall(oof_df, organ, target_class):
    sub = oof_df[oof_df["organ"] == organ].copy()
    prob_cols = sorted([c for c in sub.columns if c.startswith("prob_class")])
    sub["pred_label"] = sub[prob_cols].values.argmax(axis=1)
    fold_recalls = []
    for fold, g in sub.groupby("fold"):
        mask = g["true_label"] == target_class
        if mask.sum() == 0:
            continue
        fold_recalls.append((g.loc[mask, "pred_label"] == target_class).mean())
    return (sum(fold_recalls) / len(fold_recalls)) if fold_recalls else None


def main():
    try:
        train_organ_cv_with_oof  # noqa: F821
    except NameError:
        raise RuntimeError(
            "train_organ_cv_with_oof is not defined in this session. Run the "
            "phase1_organ_branch.py setup cells first (see "
            "liver_ordinal_check_full.py for the fast, no-retrain version), "
            "then re-run this."
        )
    try:
        organ_manifest_cv  # noqa: F821
    except NameError:
        raise RuntimeError(
            "organ_manifest_cv is not defined in this session. Build it via "
            "make_patient_folds(labels_df, organ_manifest) first, then re-run this."
        )

    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists — loading it instead of retraining. "
              f"Delete it first if you want to force a fresh run.")
        liver_baseline_seed2_df = pd.read_csv(OUTPUT_PATH)
    else:
        print(f"\n=== Liver baseline OOF, seed={SEED} (robustness check) ===")
        liver_baseline_seed2_df = train_organ_cv_with_oof(  # noqa: F821
            "liver", organ_manifest_cv, method="baseline", seed=SEED
        )
        liver_baseline_seed2_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved to {OUTPUT_PATH}")

    print("\n" + "=" * 50)
    print(f"Liver baseline, seed={SEED} — per-class recall")
    print("=" * 50)
    for name, idx in [("healthy", 0), ("low-grade", 1), ("high-grade", 2)]:
        r = per_fold_recall(liver_baseline_seed2_df, "liver", idx)
        print(f"  {name:<12} {r:.3f}" if r is not None else f"  {name:<12} N/A")
    print("\nCompare high-grade recall above against seed=42's baseline (0.100) "
          "and ordinal (0.000). If this also lands nonzero, it supports "
          "'baseline retains weak signal, ordinal erases it' as a real effect "
          "rather than single-run noise.")

    return liver_baseline_seed2_df


if __name__ == "__main__":
    main()
