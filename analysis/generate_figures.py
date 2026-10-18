"""
Figure generation for the RSNA ATD paper.

Produces the three figures requested in the meta-review:
  Figure 1 — Pipeline architecture diagram (matplotlib-drawn, no external tool needed)
  Figure 2 — Calibration reliability curves, before/after fold-honest calibration
  Figure 3 — Per-target confusion matrices, aggregated across CV folds

INPUTS REQUIRED (from earlier phases):
  organ_oof_probabilities.csv   — columns: patient_id, series_id, organ, fold,
                                  true_label, prob_class0..prob_class2
  fv_oof_probabilities.csv      — columns: patient_id, series_id, fold,
                                  bowel_true, extrav_true, bowel_prob, extrav_prob
                                  (calibrated columns are recomputed here so the
                                   figure matches the paper's fold-honest protocol)

All figures are saved as both PNG (300 dpi, for review) and PDF (vector, for
camera-ready submission).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.linear_model import LogisticRegression

# ============================================================
# Config
# ============================================================
DATA_DIR = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3")  # actual dataset path
FIG_DIR = Path("/kaggle/working/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORGAN_OOF_PATH = DATA_DIR / "organ_oof_probabilities.csv"
FV_OOF_PATH = DATA_DIR / "fv_oof_probabilities_v3.csv"  # note the _v3 suffix

N_CALIBRATION_BINS = 10

# Consistent styling across all figures
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL_ORGAN = "#E8834A"     # organ-crop branch
COL_FULLVOL = "#4A9D7F"   # full-volume branch
COL_SHARED = "#4A7DB5"    # shared backbone
COL_POST = "#8A8A8A"      # calibration / stacking


def save_fig(fig, name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(f"  saved {png.name} + {pdf.name}")
    plt.close(fig)


# ============================================================
# FIGURE 1 — Pipeline architecture diagram
# ============================================================
def make_figure1_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, label, color, dashed=False, fontsize=8.2, text_color="black"):
        style = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45",
            linewidth=1.3, edgecolor="#333333", facecolor=color, alpha=0.88,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(style)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, linespacing=1.45)

    def arrow(x1, y1, x2, y2, style="-|>", lw=1.2, color="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                      mutation_scale=11, linewidth=lw, color=color,
                                      shrinkA=1, shrinkB=1))

    # --- Input ---
    box(1.5, 26, 12, 10, "Input\nCT volume\n(axial, variable depth)", "#E9E9E9")

    # --- Stage 1: segmentation (dashed = inference-time dependency) ---
    box(17, 38, 15, 9.5, "TotalSegmentator\norgan masks\n(206/1803 series)",
        COL_POST, dashed=True)

    # --- Stage 2a: organ-crop branch ---
    box(17, 20.5, 15, 12.5, "Organ-specific\nbounding-box crop\n256×256×32\n(liver, spleen, kidney)", COL_ORGAN)
    box(36.5, 24.5, 14, 8.5, "EfficientNet-B0\n+ slice attention\npooling", COL_SHARED)
    box(54.5, 24.5, 14, 8.5, "3-class head\nhealthy / low /\nhigh (per organ)", COL_ORGAN)

    # --- Stage 2b: full-volume branch ---
    box(17, 5, 15, 11.5, "Full uncropped\nvolume\n256×256×64", COL_FULLVOL)
    box(36.5, 6.5, 14, 8.5, "EfficientNet-B0\n+ slice attention\npooling", COL_SHARED)
    box(54.5, 6.5, 14, 8.5, "2-output head\nbowel /\nextravasation", COL_FULLVOL)

    # --- Stage 3/4: calibration + selective stacking (full-volume branch only) ---
    box(72.5, 6.5, 12.5, 8.5, "Fold-honest\nPlatt\ncalibration", COL_POST, dashed=True)
    box(72.5, 17.5, 12.5, 7.5, "LightGBM\nstacker\n(+ metadata)", COL_POST, dashed=True)

    # --- Output ---
    box(88.5, 20, 10, 16,
        "Final\n13-target\nprobability\nvector", "#E9E9E9")

    # --- Arrows: input fan-out ---
    arrow(13.5, 33, 17, 42)        # to segmentation
    arrow(13.5, 31, 17, 27)        # to organ crop
    arrow(13.5, 29, 17, 11)        # to full volume
    arrow(24.5, 38, 24.5, 33)      # segmentation -> organ crop

    # --- Organ-crop path ---
    arrow(32, 27, 36.5, 28.75)
    arrow(50.5, 28.75, 54.5, 28.75)
    # Direct pass-through (no stacking) — the key ablation finding. Routed ABOVE the
    # stacker box so it is unambiguous that this path bypasses the meta-learner.
    ax.plot([68.5, 78.75, 78.75, 88.5], [28.75, 28.75, 32.5, 32.5],
            color="#B5453A", lw=2.0, solid_joinstyle="round", zorder=4)
    arrow(86.0, 32.5, 88.5, 32.5, lw=2.0, color="#B5453A")
    ax.text(78.0, 34.2, "direct pass-through", ha="center", fontsize=7.6,
            color="#B5453A", style="italic")
    ax.text(78.0, 30.0, "(n=128: stacking hurts)", ha="center", fontsize=7.0,
            color="#B5453A")

    # --- Full-volume path ---
    arrow(32, 10.75, 36.5, 10.75)
    arrow(50.5, 10.75, 54.5, 10.75)
    arrow(68.5, 10.75, 72.5, 10.75)
    arrow(78.75, 15, 78.75, 17.5)
    arrow(85, 21.25, 88.5, 24)
    ax.text(79.5, 3.0, "n=1,028: stacking helps", ha="center", fontsize=7.0, color="#2E6B54")

    # --- Legend ---
    handles = [
        mpatches.Patch(facecolor=COL_SHARED, edgecolor="#333", label="Shared backbone"),
        mpatches.Patch(facecolor=COL_ORGAN, edgecolor="#333", label="Organ-crop branch (baseline CE for liver; ordinal loss for spleen/kidney)"),
        mpatches.Patch(facecolor=COL_FULLVOL, edgecolor="#333", label="Full-volume branch (focal loss)"),
        mpatches.Patch(facecolor=COL_POST, edgecolor="#333", linestyle="--",
                        label="Post-hoc / inference-time stage"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=4, frameon=False, fontsize=8)

    ax.text(50, 53.5, "Multi-branch pipeline with selective stacking",
            ha="center", fontsize=10.5, weight="bold")

    save_fig(fig, "figure1_architecture")


# ============================================================
# FIGURE 2 — Calibration reliability curves (before / after)
# ============================================================
def fold_honest_calibrate(df, prob_col, true_col):
    """Matches the paper's protocol: calibrator for each fold is fit only on
    the other folds' out-of-fold predictions."""
    calibrated = np.zeros(len(df))
    for fold in df["fold"].unique():
        tr, te = df["fold"] != fold, df["fold"] == fold
        clf = LogisticRegression()
        clf.fit(df.loc[tr, prob_col].values.reshape(-1, 1), df.loc[tr, true_col].values)
        calibrated[te.values] = clf.predict_proba(df.loc[te, prob_col].values.reshape(-1, 1))[:, 1]
    return calibrated


def reliability_curve(y_true, y_prob, n_bins=N_CALIBRATION_BINS):
    """Equal-width bins; returns per-bin mean predicted prob, observed rate, and count."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            mean_pred.append(y_prob[m].mean())
            obs_rate.append(y_true[m].mean())
            counts.append(int(m.sum()))
    return np.array(mean_pred), np.array(obs_rate), np.array(counts)


def expected_calibration_error(y_true, y_prob, n_bins=N_CALIBRATION_BINS):
    mean_pred, obs_rate, counts = reliability_curve(y_true, y_prob, n_bins)
    return float(np.sum(counts / counts.sum() * np.abs(obs_rate - mean_pred)))


def add_calibrated_columns(fv_oof_df):
    """Computes fold-honest calibrated columns ONCE and returns a dataframe
    with them attached, so every downstream figure uses the same calibrated
    probabilities rather than each figure silently falling back to raw ones."""
    fv = fv_oof_df.copy()
    fv["bowel_prob_cal"] = fold_honest_calibrate(fv, "bowel_prob", "bowel_true")
    fv["extrav_prob_cal"] = fold_honest_calibrate(fv, "extrav_prob", "extrav_true")
    return fv


def make_figure2_calibration(fv_oof_df):
    fv = fv_oof_df  # already has *_cal columns from add_calibrated_columns

    panels = [
        ("Bowel injury",        "bowel_true",  "bowel_prob",     "Uncalibrated"),
        ("Active extravasation","extrav_true", "extrav_prob",    "Uncalibrated"),
        ("Bowel injury",        "bowel_true",  "bowel_prob_cal", "Calibrated"),
        ("Active extravasation","extrav_true", "extrav_prob_cal","Calibrated"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0), sharex=True, sharey=True)
    for ax, (target, tcol, pcol, state) in zip(axes.ravel(), panels):
        y_true = fv[tcol].values
        y_prob = fv[pcol].values
        mean_pred, obs_rate, counts = reliability_curve(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob)
        ll = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])

        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Perfect calibration")
        # marker area scales with bin count so sparsely-populated bins read as less certain
        ax.scatter(mean_pred, obs_rate, s=18 + 90 * counts / counts.max(),
                   color=COL_FULLVOL if "Bowel" in target else "#B5453A",
                   edgecolor="#333333", linewidth=0.6, zorder=3, alpha=0.9)
        ax.plot(mean_pred, obs_rate, color="#555555", lw=0.9, zorder=2)

        ax.set_title(f"{target} — {state}\nECE = {ece:.3f}   log loss = {ll:.3f}", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes[1, :]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed positive fraction")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, frameon=False)

    fig.suptitle("Reliability diagrams: full-volume branch, before and after fold-honest calibration",
                 fontsize=10.5, y=0.98)
    fig.text(0.5, 0.005,
             "Marker area ∝ number of predictions in bin. Bins pooled across all 5 CV folds.",
             ha="center", fontsize=7.5, color="#555555")
    save_fig(fig, "figure2_calibration")

    # Report the numbers so they can be quoted in the caption / Results text
    print("\n  ECE summary:")
    for target, tcol, pcol_raw, pcol_cal in [
        ("bowel", "bowel_true", "bowel_prob", "bowel_prob_cal"),
        ("extravasation", "extrav_true", "extrav_prob", "extrav_prob_cal"),
    ]:
        e_raw = expected_calibration_error(fv[tcol].values, fv[pcol_raw].values)
        e_cal = expected_calibration_error(fv[tcol].values, fv[pcol_cal].values)
        print(f"    {target}: ECE {e_raw:.4f} -> {e_cal:.4f}")


# ============================================================
# FIGURE 3 — Per-target confusion matrices (CV-aggregated)
# ============================================================
def series_level_fold_averaged_confusion(df, fold_col, true_col, pred_col, n_classes):
    """
    Computes a confusion matrix using the SAME convention as Table 2 /
    recompute_organ_recall.py's actual per_fold_recall(): series-level (no
    patient deduplication), row-normalize a confusion matrix PER FOLD, then
    average the normalized matrices across folds (unweighted mean over
    folds that had at least one true example of that row's class).

    This deliberately does NOT pool raw counts across all folds and
    normalize once globally -- pooling-then-normalizing and
    per-fold-normalize-then-average are NOT the same thing when fold sizes
    or per-fold class balance vary (a mean-of-ratios vs. ratio-of-sums
    distinction), and that difference -- not a series-vs-patient unit
    mismatch -- was the actual cause of Figure 3's liver panel (6%)
    disagreeing with Table 2 (10%) for the same underlying result.

    Returns (pct_matrix, count_matrix): pct_matrix is the fold-averaged
    recall-style percentages to display (matches Table 2 exactly);
    count_matrix is the pooled raw series-level counts across all folds,
    shown alongside for context only -- NOT used to compute the percentages.
    """
    fold_matrices = []  # list of (n_classes, n_classes) row-normalized matrices, one per fold
    for fold, g in df.groupby(fold_col):
        cm = confusion_matrix(g[true_col].values, g[pred_col].values, labels=range(n_classes))
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sums > 0, cm / np.clip(row_sums, 1, None), np.nan)
        fold_matrices.append(cm_norm)

    stacked = np.stack(fold_matrices, axis=0)  # (n_folds, n_classes, n_classes)
    pct_matrix = np.nanmean(stacked, axis=0)   # average across folds, ignoring folds with no examples of a class

    count_matrix = confusion_matrix(df[true_col].values, df[pred_col].values, labels=range(n_classes))
    return pct_matrix, count_matrix


def make_figure3_confusion(organ_oof_df, fv_oof_df):
    organ_specs = [
        ("Liver", "liver", ["Healthy", "Low", "High"]),
        ("Spleen", "spleen", ["Healthy", "Low", "High"]),
        ("Kidney", "kidney", ["Healthy", "Low", "High"]),
    ]
    binary_specs = [
        ("Bowel injury", "bowel_true", "bowel_prob_cal", ["No injury", "Injury"]),
        ("Active extravasation", "extrav_true", "extrav_prob_cal", ["No extrav.", "Extrav."]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.5))
    letters = "ABCDE"

    # --- Three-class organ matrices (series-level, per-fold-averaged -- matches Table 2 exactly) ---
    # matches Table 2's convention exactly, so numbers agree across the paper) ---
    for i, (title, organ, labels) in enumerate(organ_specs):
        sub = organ_oof_df[organ_oof_df["organ"] == organ].copy()
        prob_cols = sorted([c for c in sub.columns if c.startswith("prob_class")])
        sub["pred_label"] = sub[prob_cols].values.argmax(axis=1)

        cm_norm, cm_counts = series_level_fold_averaged_confusion(
            sub, fold_col="fold", true_col="true_label", pred_col="pred_label", n_classes=len(labels)
        )

        ax = axes[i]
        ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
        for r in range(len(labels)):
            for c in range(len(labels)):
                pct_display = "N/A" if np.isnan(cm_norm[r, c]) else f"{cm_norm[r, c]*100:.0f}%"
                ax.text(c, r, f"{pct_display}\n(n={cm_counts[r, c]})",
                        ha="center", va="center", fontsize=7.2,
                        color="white" if (not np.isnan(cm_norm[r, c]) and cm_norm[r, c] > 0.55) else "#222222")
        macro_recall = np.nanmean(np.diag(cm_norm))
        ax.set_title(f"({letters[i]}) {title}\nmacro recall = {macro_recall:.2f}", fontsize=9)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Predicted", fontsize=8)
        if i == 0:
            ax.set_ylabel("True", fontsize=8)

    # --- Two binary matrices (series-level, per-fold-averaged, same method
    # as above, thresholded at 0.5 on CALIBRATED probabilities) ---
    for j, (title, tcol, pcol, labels) in enumerate(binary_specs):
        ax = axes[3 + j]
        sub = fv_oof_df.copy()
        sub["pred_label"] = (sub[pcol].values >= 0.5).astype(int)

        cm_norm, cm_counts = series_level_fold_averaged_confusion(
            sub, fold_col="fold", true_col=tcol, pred_col="pred_label", n_classes=2
        )

        ax.imshow(np.nan_to_num(cm_norm), cmap="Greens", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                pct_display = "N/A" if np.isnan(cm_norm[r, c]) else f"{cm_norm[r, c]*100:.0f}%"
                ax.text(c, r, f"{pct_display}\n(n={cm_counts[r, c]})",
                        ha="center", va="center", fontsize=7.2,
                        color="white" if (not np.isnan(cm_norm[r, c]) and cm_norm[r, c] > 0.55) else "#222222")
        recall_pos = cm_norm[1, 1]
        recall_display = "N/A" if np.isnan(recall_pos) else f"{recall_pos:.2f}"
        ax.set_title(f"({letters[3+j]}) {title}\npositive recall = {recall_display}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Predicted", fontsize=8)

    fig.suptitle("Row-normalised confusion matrices, aggregated over 5-fold cross-validation",
                 fontsize=10.5, y=1.04)
    fig.text(0.5, -0.10,
             "(A–C) organ-crop branch, three-level ordinal severity. (D–E) full-volume branch, "
             "fold-honest calibrated probabilities, thresholded at 0.5. Percentages are "
             "series-level, per-fold-averaged recall (matching Table 2's actual computation "
             "method); counts in parentheses are pooled series-level totals shown for context.",
             ha="center", fontsize=7.8, color="#555555")
    save_fig(fig, "figure3_confusion_matrices")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Figure 1: architecture diagram")
    make_figure1_architecture()

    missing = [p for p in (ORGAN_OOF_PATH, FV_OOF_PATH) if not p.exists()]
    if missing:
        print("\nSkipping Figures 2 and 3 — missing input files:")
        for p in missing:
            print(f"  {p}")
        print("Run the Phase 1b/1c OOF export steps first, or point DATA_DIR at the "
              "dataset where those CSVs are stored.")
    else:
        organ_oof_df = pd.read_csv(ORGAN_OOF_PATH)
        fv_oof_df_raw = pd.read_csv(FV_OOF_PATH)
        fv_oof_df = add_calibrated_columns(fv_oof_df_raw)
        print(f"\nLoaded organ OOF: {len(organ_oof_df)} rows "
              f"({organ_oof_df['organ'].nunique()} organs)")
        print(f"Loaded full-volume OOF: {len(fv_oof_df)} rows")

        print("\nFigure 2: calibration reliability curves")
        make_figure2_calibration(fv_oof_df)

        print("\nFigure 3: confusion matrices")
        make_figure3_confusion(organ_oof_df, fv_oof_df)

    print(f"\nAll figures written to {FIG_DIR}")
