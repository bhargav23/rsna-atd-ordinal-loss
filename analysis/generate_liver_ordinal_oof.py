"""
generate_liver_ordinal_oof.py

Regenerates the missing liver ORDINAL-loss OOF export. Table 2 cites a
liver ordinal config (alpha=0.5, boost=1.5, i.e. train_organ_cv_with_oof's
plain defaults) that was used only during the original hyperparameter
search -- FINAL_ORGAN_CONFIGS for liver settled on baseline CE, so that
ordinal run was never checkpointed to organ_oof_probabilities.csv.

Same session requirement as generate_baseline_oof.py: this calls
train_organ_cv_with_oof() and uses organ_manifest_cv directly as objects
ALREADY DEFINED IN THIS KERNEL from your Phase 1 cells -- not imported.

This is a real training job: 5 epochs x 5 folds, one organ. Needs GPU.

Output: /kaggle/working/liver_ordinal_oof.csv, same schema as
organ_oof_probabilities.csv, for use with recompute_organ_recall.py
(set ORGAN = "liver" and FINAL_OOF_PATH to this file) to check whether
Table 2's liver row (0.00 -> 0.00 high-grade recall either way) still holds.
"""

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("/kaggle/working/liver_ordinal_oof.csv")


def main():
    try:
        train_organ_cv_with_oof  # noqa: F821
    except NameError:
        raise RuntimeError(
            "train_organ_cv_with_oof is not defined in this session. Run your "
            "Phase 1 cells (the content of phase1_organ_branch.py) in this "
            "notebook first via %run, then re-run this."
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
        return pd.read_csv(OUTPUT_PATH)

    print("\n=== Liver ordinal OOF export (alpha=0.5, boost=1.5 — matches Table 2's cited config) ===")
    # No overrides: alpha=0.5, present_boost=1.5 are train_organ_cv_with_oof's
    # own defaults, and match exactly what Table 2 cites for liver's ordinal row.
    liver_ordinal_oof_df = train_organ_cv_with_oof("liver", organ_manifest_cv, method="ordinal")  # noqa: F821
    liver_ordinal_oof_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved liver ordinal OOF to {OUTPUT_PATH}")
    print(liver_ordinal_oof_df.groupby("fold").size())
    return liver_ordinal_oof_df


if __name__ == "__main__":
    main()
