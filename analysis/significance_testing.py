"""
significance_testing.py

Wilcoxon signed-rank test comparing the final chosen loss config against
baseline cross-entropy, per organ, PAIRED BY FOLD (n=5 pairs per organ).
Addresses the meta-review's "statistical significance testing" item.

IMPORTANT DATA REQUIREMENT — read before running:
organ_oof_probabilities.csv only contains the FINAL configs actually used
per organ (see FINAL_ORGAN_CONFIGS in phase1_organ_branch.py):
    liver:  baseline CE
    spleen: ordinal, alpha=3.0
    kidney: ordinal, alpha=1.0
It does NOT contain the "other side" of each comparison — baseline for
spleen/kidney, or an ordinal variant for liver. That only exists in whatever
your earlier ablation phase (the pre-committed-candidate-list CV re-evaluation
that produced the 12-17pp correction) produced and saved.

This script supports two ways to supply the missing comparison arm:

  OPTION A (preferred) — point BASELINE_OOF_PATH at a saved OOF-style CSV
  (same schema as organ_oof_probabilities.csv: organ, fold, true_label,
  prob_class0..N) containing the baseline-method run for spleen and kidney.
  If your ablation phase checkpointed OOF exports per candidate config,
  this is that file.

  OPTION B (fallback) — if you don't have that file but still have the
  per-fold accuracy numbers (e.g. printed in your ablation run's notebook
  output, or in a results CSV with one row per fold), hand-enter them into
  MANUAL_FOLD_ACCURACIES below. Each organ needs two length-5 lists: the
  final config's per-fold val accuracy, and baseline's per-fold val accuracy,
  in the SAME fold order (fold 0..4) so pairing is correct — an unpaired or
  misordered comparison invalidates the whole test.

Leave a list as None to skip that organ (e.g. liver has no second condition
unless you also ran an ordinal variant on it during ablation).

STATISTICAL CAVEAT (do not omit this from the paper): with only 5 folds,
an exact two-sided Wilcoxon signed-rank test has a minimum achievable
p-value of 1/16 = 0.0625 (all 5 differences same sign, none tied) — it is
IMPOSSIBLE to reach the conventional p<0.05 threshold with n=5, no matter
how large or consistent the true effect is. Report this test as descriptive
evidence of directional consistency across folds, not as a claim of
significance at a specific alpha level. This is worth stating explicitly in
the paper's Methods/Results to preempt a reviewer flagging it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

FINAL_OOF_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_oof_probabilities.csv")
BASELINE_OOF_PATH = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3/organ_baseline_oof.csv")  # EDIT if your ablation OOF lives elsewhere

# OPTION B fallback: fill in only if BASELINE_OOF_PATH doesn't exist / isn't applicable.
# Each value must be a length-5 list in fold order (fold 0, 1, 2, 3, 4), or None to skip.
MANUAL_FOLD_ACCURACIES = {
    "liver":  {"final": None, "baseline": None},
    "spleen": {"final": None, "baseline": None},
    "kidney": {"final": None, "baseline": None},
}

ORGANS_WITH_COMPARISON = ["spleen", "kidney"]  # liver's final IS baseline; no comparison by default


def per_fold_accuracy(oof_df, organ):
    """Computes val accuracy per fold for one organ from an OOF-style DataFrame."""
    sub = oof_df[oof_df["organ"] == organ].copy()
    prob_cols = sorted([c for c in sub.columns if c.startswith("prob_class")])
    sub["correct"] = (sub[prob_cols].values.argmax(axis=1) == sub["true_label"]).astype(float)
    acc_by_fold = sub.groupby("fold")["correct"].mean().sort_index()
    return acc_by_fold  # pandas Series indexed by fold


def run_wilcoxon(final_accs, baseline_accs, organ):
    final_accs = np.asarray(final_accs, dtype=float)
    baseline_accs = np.asarray(baseline_accs, dtype=float)
    if len(final_accs) != len(baseline_accs):
        raise ValueError(
            f"[{organ}] final has {len(final_accs)} folds but baseline has "
            f"{len(baseline_accs)} — these must be paired 1:1 by fold."
        )
    diffs = final_accs - baseline_accs
    n_nonzero = int(np.sum(diffs != 0))

    print(f"\n--- {organ.capitalize()} ---")
    print(f"  Final (per fold):    {np.round(final_accs, 4).tolist()}")
    print(f"  Baseline (per fold): {np.round(baseline_accs, 4).tolist()}")
    print(f"  Diffs (final-base):  {np.round(diffs, 4).tolist()}")
    print(f"  Mean diff: {diffs.mean():+.4f}  |  Folds improved: {int((diffs > 0).sum())}/{len(diffs)}")

    if n_nonzero == 0:
        print("  All fold-level differences are zero — Wilcoxon test undefined (no variation).")
        return None

    try:
        stat, p = wilcoxon(final_accs, baseline_accs, alternative="two-sided", mode="exact")
    except ValueError as e:
        print(f"  Exact Wilcoxon failed ({e}); falling back to asymptotic approximation "
              f"(unreliable at n={len(final_accs)}, reported for completeness only).")
        stat, p = wilcoxon(final_accs, baseline_accs, alternative="two-sided", mode="approx")

    print(f"  Wilcoxon signed-rank: W={stat:.3f}, p={p:.4f}  (n={len(final_accs)} paired folds)")
    print(f"  NOTE: minimum achievable exact p-value at n={len(final_accs)} folds is "
          f"{1 / (2 ** (len(final_accs) - 1)):.4f} — treat this as directional-consistency "
          f"evidence, not a significance claim at p<0.05.")
    return {"organ": organ, "n_folds": len(final_accs), "W": stat, "p": p,
            "mean_diff": diffs.mean(), "folds_improved": int((diffs > 0).sum())}


def main():
    final_oof_df = pd.read_csv(FINAL_OOF_PATH)

    baseline_oof_df = None
    if BASELINE_OOF_PATH.exists():
        baseline_oof_df = pd.read_csv(BASELINE_OOF_PATH)
        print(f"Loaded baseline comparison OOF from {BASELINE_OOF_PATH}")
    else:
        print(f"No baseline OOF file found at {BASELINE_OOF_PATH} — will use "
              f"MANUAL_FOLD_ACCURACIES where filled in, and skip organs with neither.")

    results = []
    for organ in ORGANS_WITH_COMPARISON:
        final_accs = None
        baseline_accs = None

        # Prefer OOF-derived accuracy (recomputed directly, not trusting any
        # pre-aggregated summary) whenever the relevant file is available.
        final_series = per_fold_accuracy(final_oof_df, organ)
        if len(final_series) > 0:
            final_accs = final_series.values

        if baseline_oof_df is not None and organ in baseline_oof_df["organ"].unique():
            baseline_series = per_fold_accuracy(baseline_oof_df, organ)
            baseline_accs = baseline_series.reindex(final_series.index).values

        # Fall back to manual entry for whichever arm is still missing
        manual = MANUAL_FOLD_ACCURACIES.get(organ, {})
        if final_accs is None and manual.get("final") is not None:
            final_accs = np.asarray(manual["final"])
        if baseline_accs is None and manual.get("baseline") is not None:
            baseline_accs = np.asarray(manual["baseline"])

        if final_accs is None or baseline_accs is None:
            print(f"\n--- {organ.capitalize()} ---")
            print(
                f"  SKIPPED: missing {'final' if final_accs is None else 'baseline'} "
                f"per-fold accuracy. Provide it via BASELINE_OOF_PATH or "
                f"MANUAL_FOLD_ACCURACIES['{organ}']."
            )
            continue

        result = run_wilcoxon(final_accs, baseline_accs, organ)
        if result:
            results.append(result)

    if results:
        results_df = pd.DataFrame(results)
        out_path = Path("significance_test_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path.resolve()}")
        print(results_df.to_string(index=False))
    else:
        print(
            "\nNo comparisons were run — every organ was missing at least one "
            "arm's per-fold accuracy. Fill in BASELINE_OOF_PATH or "
            "MANUAL_FOLD_ACCURACIES and re-run."
        )


if __name__ == "__main__":
    main()
