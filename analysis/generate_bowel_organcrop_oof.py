"""
generate_bowel_organcrop_oof.py

Retrains the bowel ORGAN-CROP baseline classifier under proper 5-fold,
patient-level CV -- the verification the paper's Section 4.4/5.4 claim
("0.0 recall, missing the single positive case in its validation set")
never actually received. That phrasing ("the single positive case") is a
single-split-protocol fingerprint, not a 5-fold CV one, so this number has
never been checked against the same rigor standard applied to liver,
spleen, and kidney elsewhere in this paper.

Good news: `train_organ_cv_with_oof` and `OrganCropDataset` ALREADY support
bowel as an organ (ORGAN_NUM_CLASSES["bowel"] = 2, get_organ_label handles
it via the bowel_injury column) -- this is not new code, just an unused
capability of your existing Phase 1 pipeline being pointed at the organ-crop
branch's own bowel crops, as opposed to the full-volume branch used for the
paper's actual bowel results.

IMPORTANT: bowel injury is almost certainly very rare even within the
128-patient organ-crop cohort (it's a minority-of-a-minority: not every
organ-crop patient has a bowel crop AND has bowel injury). If the total
positive count across all 128 patients is only 1-3, most folds will have
ZERO true positives, and per-fold recall will be mostly undefined (matching
liver's high-grade-class scarcity problem, Section 5.3) -- this script
reports that explicitly rather than silently averaging over folds that
have no positives to recall.

Same session requirement as the other generate_*.py scripts in this
project: needs train_organ_cv_with_oof and organ_manifest_cv already
defined in this kernel (run liver_ordinal_check_full.py's setup portion,
or the base phase1_organ_branch.py cells, first if this is a fresh session).

Output: /kaggle/working/bowel_organcrop_oof.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("/kaggle/working/bowel_organcrop_oof.csv")


def per_fold_recall(oof_df, target_class):
    """Same convention as recompute_organ_recall.py: series-level, grouped
    by fold, recall averaged across folds that had at least one positive."""
    prob_cols = sorted([c for c in oof_df.columns if c.startswith("prob_class")])
    oof_df = oof_df.copy()
    oof_df["pred_label"] = oof_df[prob_cols].values.argmax(axis=1)

    fold_recalls = []
    fold_details = []
    for fold, g in oof_df.groupby("fold"):
        mask = g["true_label"] == target_class
        n_positive = int(mask.sum())
        if n_positive == 0:
            fold_details.append((fold, n_positive, None))
            continue
        recall = (g.loc[mask, "pred_label"] == target_class).mean()
        fold_recalls.append(recall)
        fold_details.append((fold, n_positive, recall))

    overall = float(np.mean(fold_recalls)) if fold_recalls else None
    return overall, fold_details


def main():
    try:
        train_organ_cv_with_oof  # noqa: F821
    except NameError:
        raise RuntimeError(
            "train_organ_cv_with_oof is not defined in this session. Run the "
            "phase1_organ_branch.py setup cells first (see "
            "liver_ordinal_check_full.py for the fast, no-retrain version of "
            "this setup), then re-run this."
        )
    try:
        organ_manifest_cv  # noqa: F821
    except NameError:
        raise RuntimeError(
            "organ_manifest_cv is not defined in this session. Build it via "
            "make_patient_folds(labels_df, organ_manifest) first, then re-run this."
        )

    # Sanity check BEFORE training: how many organ-crop patients even have a
    # bowel crop, and how many of those are injury-positive? If this is very
    # small, the eventual recall number will be noisy/near-undefined almost
    # by construction, which is itself an important finding to report.
    bowel_available = organ_manifest_cv[
        organ_manifest_cv["organs_cropped"].apply(lambda crops: "bowel" in crops)
    ]
    n_bowel_series = len(bowel_available)
    n_bowel_patients = bowel_available["patient_id"].nunique()
    n_bowel_positive = int(bowel_available["bowel_injury"].sum()) if n_bowel_series > 0 else 0
    print(f"Bowel organ-crop availability: {n_bowel_series} series, "
          f"{n_bowel_patients} patients, {n_bowel_positive} bowel-injury-positive")
    if n_bowel_series == 0:
        raise RuntimeError(
            "No series have 'bowel' in organs_cropped -- the organ-crop branch "
            "never had bowel crops available for this patient set, so this "
            "specific claim cannot be produced by retraining; the paper's "
            "text should be corrected to state that bowel organ-cropping was "
            "never attempted/possible, rather than that it failed with 0.0 recall."
        )
    if n_bowel_positive <= 2:
        print(
            f"WARNING: only {n_bowel_positive} bowel-injury-positive series exist in "
            f"the organ-crop cohort. Most folds will likely have zero true positives, "
            f"making per-fold recall undefined for those folds. This alone may explain "
            f"the original single-split claim ('the single positive case') -- with this "
            f"few positives, a single 15% validation split could easily contain exactly "
            f"one, which is a data-availability artifact, not necessarily evidence the "
            f"architecture is unable to learn bowel injury from organ crops."
        )

    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists — loading it instead of retraining. "
              f"Delete it first if you want to force a fresh run.")
        bowel_oof_df = pd.read_csv(OUTPUT_PATH)
    else:
        print("\n=== Bowel organ-crop baseline OOF export (5-fold CV) ===")
        bowel_oof_df = train_organ_cv_with_oof("bowel", organ_manifest_cv, method="baseline")  # noqa: F821
        bowel_oof_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved to {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("Bowel organ-crop baseline — 5-fold CV recall (vs. paper's single-split 0.0 claim)")
    print("=" * 60)
    for name, idx in [("no injury (class 0)", 0), ("injury (class 1)", 1)]:
        overall, fold_details = per_fold_recall(bowel_oof_df, idx)
        print(f"\n{name}:")
        for fold, n_pos, recall in fold_details:
            if recall is None:
                print(f"  fold {fold}: 0 true positives — recall undefined for this fold")
            else:
                print(f"  fold {fold}: {n_pos} true positive(s), recall = {recall:.3f}")
        print(f"  Overall (mean across folds with >=1 positive): "
              f"{overall:.3f}" if overall is not None else "  Overall: undefined (no fold had a positive example)")

    print("\n" + "=" * 60)
    print("Compare 'injury (class 1)' overall recall above against the paper's")
    print("current claim of exactly 0.0 (from the single-split protocol).")
    print("If most/all folds show 'undefined' (zero positives), that itself is")
    print("the honest finding: report as a data-availability limitation, not")
    print("a clean 0.0 recall figure, since 0.0 implies the model saw positive")
    print("examples and missed them, whereas 'undefined' means it may never")
    print("have been meaningfully tested at all in most folds.")
    print("=" * 60)

    return bowel_oof_df


if __name__ == "__main__":
    main()
