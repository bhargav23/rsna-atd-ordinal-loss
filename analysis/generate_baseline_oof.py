"""
generate_baseline_oof.py

Regenerates the missing baseline-condition (plain cross-entropy) OOF export
for spleen and kidney, needed as the comparison arm for significance_testing.py.
Liver is skipped — its FINAL config already IS baseline CE, so
organ_oof_probabilities.csv already has everything needed for liver.

IMPORTANT: this calls train_organ_cv_with_oof() and uses organ_manifest_cv
directly as OBJECTS ALREADY DEFINED IN THIS KERNEL SESSION — it does not
import them from phase1_organ_branch.py as a module (that import failed
earlier because your pipeline code lives in this notebook as executed cells,
not as a separate importable .py file). Run this in the SAME session where
you already ran your Phase 1 cells, so that train_organ_cv_with_oof,
OrganClassifier, and organ_manifest_cv already exist in memory.

This is a real training job: 5 epochs x 5 folds x 2 organs (spleen, kidney),
the same cost as your original ablation runs. Needs GPU.

Output: /kaggle/working/organ_baseline_oof.csv, in the same schema as
organ_oof_probabilities.csv (organ, fold, true_label, prob_class0..N) so
significance_testing.py can consume it directly via BASELINE_OOF_PATH
(already pointed at this exact path/filename by default).
"""

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("/kaggle/working/organ_baseline_oof.csv")
ORGANS_TO_RUN = ["spleen", "kidney"]  # liver's final config is already baseline


def main():
    # Fail loudly and clearly rather than guessing if the session doesn't
    # already have these defined from your Phase 1 cells.
    missing = [name for name in ("train_organ_cv_with_oof", "organ_manifest_cv")
               if name not in globals() and name not in dir(__builtins__)]
    try:
        train_organ_cv_with_oof  # noqa: F821
    except NameError:
        raise RuntimeError(
            "train_organ_cv_with_oof is not defined in this session. Run your "
            "Phase 1 cells (the ones defining OrganClassifier, "
            "train_organ_cv_with_oof, etc. — i.e. the content of "
            "phase1_organ_branch.py) in this notebook first, then re-run this."
        )
    try:
        organ_manifest_cv  # noqa: F821
    except NameError:
        raise RuntimeError(
            "organ_manifest_cv is not defined in this session. Build it via "
            "make_patient_folds(labels_df, organ_manifest) first (same call "
            "your original Phase 1 run used), then re-run this."
        )

    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists — loading it instead of retraining. "
              f"Delete it first if you want to force a fresh run.")
        return pd.read_csv(OUTPUT_PATH)

    frames = []
    for organ in ORGANS_TO_RUN:
        print(f"\n=== Baseline OOF export: {organ} ===")
        # Same defaults (epochs=5, seed=42) as the FINAL_ORGAN_CONFIGS baseline
        # run for liver, for a fair like-for-like comparison.
        oof_df = train_organ_cv_with_oof(organ, organ_manifest_cv, method="baseline")  # noqa: F821
        frames.append(oof_df)

    baseline_oof_df = pd.concat(frames, ignore_index=True)
    baseline_oof_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved baseline OOF for {ORGANS_TO_RUN} to {OUTPUT_PATH}")
    print(baseline_oof_df.groupby("organ")["fold"].value_counts().sort_index())
    return baseline_oof_df


if __name__ == "__main__":
    main()
