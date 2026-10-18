"""
verify_liver_counts.py

Standalone sanity check for the liver high-grade count discrepancy found in
generate_fold_tables.py's output (19 patients at the organ-crop-branch,
patient-deduplicated level, vs. "27" cited throughout the paper).

This does NOT trust organ_oof_probabilities.csv's own true_label column as
ground truth — it goes back to subset_labels.csv (the actual competition
label file used everywhere else in the pipeline) and cross-checks three
independent counts:

  (A) Total liver_high patients in the FULL labels file (no organ-crop
      restriction at all) — tells you if "27" might refer to the whole
      dataset rather than just the organ-crop subset.
  (B) liver_high count in subset_labels.csv, restricted to exactly the
      patient_ids that appear in the liver rows of
      organ_oof_probabilities.csv — this is the direct, from-source-of-truth
      version of what the fold table is trying to report.
  (C) The count implied by organ_oof_probabilities.csv's own true_label
      column (== 2) for the same patient set — this is what the fold table
      actually reported (19).

If (B) == (C): the OOF export's labels match the source-of-truth label file
correctly, and the fold table's "19" is simply correct — the paper's "27"
needs to be corrected instead.

If (B) != (C): there's a genuine label-decoding or patient-set mismatch
between organ_oof_probabilities.csv and subset_labels.csv, and that bug
should be found before trusting EITHER the "19" or the "27" — go look at
get_organ_label() and how organ_manifest_cv's patient set was built relative
to whatever produced this v3 OOF export.

If (A) == 27 (or close) but (B)/(C) are much lower: "27" was very likely
referring to the FULL dataset's liver_high patients, not the 128-patient
organ-crop subset specifically, and the paper's phrasing needs to
distinguish the two rather than either number being wrong.

USAGE:
  Edit LABELS_PATH and ORGAN_OOF_PATH below to match your actual Kaggle
  paths (same ones used in phase1_organ_branch.py / your v3 OOF dataset),
  then run this in the Kaggle session.
"""

import pandas as pd
from pathlib import Path

# --- EDIT THESE TWO PATHS to match your actual setup ---
LABELS_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-phase0-preprocessed/processed/subset_labels.csv")
ORGAN_OOF_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_oof_probabilities.csv")
# ---------------------------------------------------------


def main():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find subset_labels.csv at {LABELS_PATH}. Edit "
            f"LABELS_PATH at the top of this script to match wherever your "
            f"Phase 0 output (the PHASE0_DIR used in phase1_organ_branch.py) "
            f"actually lives in this Kaggle session."
        )
    if not ORGAN_OOF_PATH.exists():
        raise FileNotFoundError(f"Could not find organ_oof_probabilities.csv at {ORGAN_OOF_PATH}.")

    labels_df = pd.read_csv(LABELS_PATH)
    organ_oof_df = pd.read_csv(ORGAN_OOF_PATH)

    required_label_cols = {"liver_healthy", "liver_low", "liver_high", "patient_id"}
    missing = required_label_cols - set(labels_df.columns)
    if missing:
        raise ValueError(
            f"subset_labels.csv is missing expected columns: {missing}. "
            f"Found columns: {list(labels_df.columns)}"
        )

    # (A) Full-dataset liver_high count, no restriction at all
    full_dataset_liver_high = int(labels_df["liver_high"].sum())
    full_dataset_n_patients = labels_df["patient_id"].nunique()

    # Patient set actually used for the liver organ-crop branch, per the OOF export
    liver_oof = organ_oof_df[organ_oof_df["organ"] == "liver"]
    liver_patients = liver_oof["patient_id"].unique()
    n_liver_patients = len(liver_patients)

    # (B) Ground-truth liver_high count for exactly that patient set,
    #     read directly from subset_labels.csv (not from the OOF file)
    labels_restricted = labels_df[labels_df["patient_id"].isin(liver_patients)].drop_duplicates(subset="patient_id")
    ground_truth_liver_high = int(labels_restricted["liver_high"].sum())
    ground_truth_liver_low = int(labels_restricted["liver_low"].sum())
    ground_truth_liver_healthy = int(labels_restricted["liver_healthy"].sum())

    # (C) What organ_oof_probabilities.csv itself implies (its own true_label column)
    liver_patients_dedup = liver_oof.drop_duplicates(subset="patient_id")
    oof_liver_high = int((liver_patients_dedup["true_label"] == 2).sum())
    oof_liver_low = int((liver_patients_dedup["true_label"] == 1).sum())
    oof_liver_healthy = int((liver_patients_dedup["true_label"] == 0).sum())

    print("=" * 70)
    print("(A) FULL dataset (no organ-crop restriction)")
    print(f"    Total patients in subset_labels.csv: {full_dataset_n_patients}")
    print(f"    liver_high (any patient):            {full_dataset_liver_high}")
    print()
    print("=" * 70)
    print(f"Liver organ-crop branch patient set (from organ_oof_probabilities.csv): {n_liver_patients} patients")
    print()
    print("(B) Ground truth from subset_labels.csv, restricted to those patients")
    print(f"    liver_healthy: {ground_truth_liver_healthy}")
    print(f"    liver_low:     {ground_truth_liver_low}")
    print(f"    liver_high:    {ground_truth_liver_high}")
    print()
    print("(C) As reported by organ_oof_probabilities.csv's own true_label column")
    print(f"    liver_healthy: {oof_liver_healthy}")
    print(f"    liver_low:     {oof_liver_low}")
    print(f"    liver_high:    {oof_liver_high}")
    print("=" * 70)

    if ground_truth_liver_high == oof_liver_high:
        print(
            f"\n(B) == (C): the OOF export's labels match subset_labels.csv exactly. "
            f"The fold table's liver_high count ({oof_liver_high}) is correct — "
            f"the paper's '27' figure needs to be corrected to {oof_liver_high} "
            f"wherever it's cited (Abstract, Results, Discussion)."
        )
    else:
        print(
            f"\n(B) != (C): subset_labels.csv says {ground_truth_liver_high} but the "
            f"OOF export implies {oof_liver_high}. This is a genuine label-decoding "
            f"or patient-set mismatch between organ_oof_probabilities.csv and the "
            f"labels file — investigate get_organ_label() and how the patient set "
            f"used to build this v3 OOF export was chosen before trusting either "
            f"number."
        )

    if abs(full_dataset_liver_high - 27) <= 2 and abs(ground_truth_liver_high - 27) > 5:
        print(
            f"\nNote: the FULL dataset's liver_high count ({full_dataset_liver_high}) "
            f"is close to 27, while the organ-crop-restricted count is not. It's "
            f"worth checking whether '27' in the paper was meant to describe the "
            f"whole dataset's high-grade liver patients rather than specifically "
            f"the 128-patient organ-crop subset — if so, the paper's phrasing needs "
            f"to distinguish the two rather than the number itself being wrong."
        )


if __name__ == "__main__":
    main()
