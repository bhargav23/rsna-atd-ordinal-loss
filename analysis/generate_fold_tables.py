"""
generate_fold_tables.py  (v3 — reads the already-saved OOF CSVs)

Produces the fold-level class distribution tables requested by meta-review
item #9.

PRIMARY DATA SOURCE (v3 change): rather than requiring organ_manifest_cv /
fv_manifest_cv to exist live in the current kernel session — which fails on
a fresh session, since nothing in that session built them — this version
reads the OOF export CSVs your Phase 1/1c code already writes to disk:

  organ_oof_probabilities.csv   (from phase1_organ_branch.run_organ_oof_export)
    columns: patient_id, series_id, organ, fold, true_label, prob_class0..N
    'organ' distinguishes liver/spleen/kidney (all three concatenated in one
    file); 'true_label' is already the decoded 0/1/2 class, no need to
    re-derive it from one-hot columns.

  fv_oof_probabilities.csv      (from phase1c_fullvolume_branch.run_fullvolume_branch)
    columns: patient_id, series_id, fold, bowel_true, extrav_true,
    bowel_prob, extrav_prob (+ calibrated variants once calibration has run)

Both are checkpointed to OUTPUT_DIR (/kaggle/working by default in your
pipeline) and reloaded on any re-run rather than retraining, so if Phase 1/1c
have been run at all in this notebook's persistent /kaggle/working, these
files should already be sitting there.

FALLBACK: if the CSVs aren't found, the script falls back to looking for
`organ_manifest_cv` / `fv_manifest_cv` already defined in the current
session's namespace (the shape produced by
phase1_organ_branch.make_patient_folds / phase1c_fullvolume_branch.make_fv_patient_folds),
in case you're running this mid-session rather than after a full save/reload.

Outputs (written to ./fold_tables/):
  - organ_crop_{organ}_counts.csv / _proportions.csv   for liver, spleen, kidney
  - full_volume_{target}_counts.csv / _proportions.csv for bowel, extravasation
  - fold_distribution_tables.md   — both tables formatted for direct paper inclusion
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/input/datasets/<username>/rsna-atd-fv-oof-v3")
ORGAN_OOF_FILENAME = "organ_oof_probabilities.csv"
FV_OOF_FILENAME = "fv_oof_probabilities_v3.csv"
OUT_DIR = Path("fold_tables")
OUT_DIR.mkdir(exist_ok=True)

ORGAN_TARGETS = ["liver", "spleen", "kidney"]
ORGAN_CLASS_NAMES = {0: "healthy", 1: "low-grade", 2: "high-grade"}
FV_TARGETS = {"bowel": "bowel_true", "extravasation": "extrav_true"}
FV_CLASS_NAMES = {0: "healthy", 1: "injury"}


def get_organ_label(row, organ):
    """Inlined from phase1_organ_branch.py — only used by the manifest_cv
    fallback path, since the OOF CSVs already carry a decoded true_label."""
    if organ == "bowel":
        return int(row["bowel_injury"])
    healthy, low, high = row[f"{organ}_healthy"], row[f"{organ}_low"], row[f"{organ}_high"]
    return int(np.argmax([healthy, low, high]))


def _counts_and_proportions(df, label_col, class_names):
    df = df.copy()
    df["label_name"] = df[label_col].map(class_names)
    counts = (
        df.groupby(["fold", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(class_names.values()), fill_value=0)
        .sort_index()
    )
    counts.loc["Total"] = counts.sum()
    proportions = counts.div(counts.sum(axis=1), axis=0).round(3)
    return counts, proportions


def format_markdown_table(counts, proportions, title):
    lines = [f"**{title}**\n"]
    header = "| Fold | " + " | ".join(counts.columns) + " |"
    sep = "|---" * (len(counts.columns) + 1) + "|"
    lines += [header, sep]
    for idx in counts.index:
        row_counts = counts.loc[idx]
        if idx == "Total":
            cells = [str(int(v)) for v in row_counts]
        else:
            row_props = proportions.loc[idx]
            cells = [f"{int(c)} ({p:.1%})" for c, p in zip(row_counts, row_props)]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Primary path: read from saved OOF CSVs
# ============================================================
def load_from_oof_csvs(output_dir=OUTPUT_DIR):
    organ_path = output_dir / ORGAN_OOF_FILENAME
    fv_path = output_dir / FV_OOF_FILENAME
    if not organ_path.exists() or not fv_path.exists():
        return None, None
    organ_oof_df = pd.read_csv(organ_path)
    fv_oof_df = pd.read_csv(fv_path)
    return organ_oof_df, fv_oof_df


def build_organ_fold_table_from_oof(organ_oof_df, organ, dedupe_by_patient=False):
    sub = organ_oof_df[organ_oof_df["organ"] == organ]
    if sub.empty:
        raise ValueError(
            f"No rows with organ == '{organ}' in {ORGAN_OOF_FILENAME}. "
            f"Found organs: {organ_oof_df['organ'].unique().tolist()}"
        )
    if dedupe_by_patient:
        # Safe to dedupe on patient_id alone: fold and true_label are both
        # patient-level attributes broadcast across a patient's series rows,
        # so any one row per patient carries the same (fold, true_label).
        sub = sub.drop_duplicates(subset="patient_id")
    return _counts_and_proportions(sub, "true_label", ORGAN_CLASS_NAMES)


def build_fv_fold_table_from_oof(fv_oof_df, label_col, dedupe_by_patient=False):
    df = fv_oof_df.drop_duplicates(subset="patient_id") if dedupe_by_patient else fv_oof_df
    return _counts_and_proportions(df, label_col, FV_CLASS_NAMES)


# ============================================================
# Fallback path: build from live manifest_cv objects in the session
# ============================================================
def build_organ_fold_table_from_manifest(organ_manifest_cv, organ):
    sub = organ_manifest_cv[
        organ_manifest_cv["organs_cropped"].apply(lambda crops: organ in crops)
    ].copy()
    sub["label"] = sub.apply(lambda row: get_organ_label(row, organ), axis=1)
    return _counts_and_proportions(sub, "label", ORGAN_CLASS_NAMES)


def build_fv_fold_table_from_manifest(fv_manifest_cv, label_col):
    return _counts_and_proportions(fv_manifest_cv, label_col, FV_CLASS_NAMES)


def main():
    organ_oof_df, fv_oof_df = load_from_oof_csvs()
    using_oof = organ_oof_df is not None

    if not using_oof:
        print(
            f"Could not find {ORGAN_OOF_FILENAME} / {FV_OOF_FILENAME} "
            f"under {OUTPUT_DIR}. Falling back to organ_manifest_cv / fv_manifest_cv "
            f"if they're defined in this session..."
        )
        try:
            organ_manifest_cv  # noqa: F821
            fv_manifest_cv  # noqa: F821
        except NameError:
            raise RuntimeError(
                "Neither the OOF CSVs nor organ_manifest_cv/fv_manifest_cv are "
                "available. Either: (1) point OUTPUT_DIR at wherever "
                "organ_oof_probabilities.csv / fv_oof_probabilities_v3.csv actually "
                "live (check the output of your Phase 1/1c runs), or (2) rebuild "
                "the manifests via phase1_organ_branch.make_patient_folds(...) / "
                "phase1c_fullvolume_branch.make_fv_patient_folds(...) in this "
                "session first."
            )

    md_sections = [
        "## Supplementary Table: Fold-Level Class Distribution\n",
        "(Addresses meta-review item #9 — verifies stratified 5-fold CV produced "
        "balanced per-fold class representation for each target. Two units are "
        "reported: SERIES-level, matching what each branch's dataset class "
        "actually iterates over during training/validation, and PATIENT-level "
        "(deduplicated), matching the headline patient counts cited elsewhere "
        "in the paper, e.g. 206 organ-crop series / 128 patients, 27 high-grade "
        "liver examples, 1,028 full-volume patients.)\n",
        "### Organ-crop branch (liver / spleen / kidney) — series-level\n",
    ]

    organ_tables_patient = {}
    for organ in ORGAN_TARGETS:
        if using_oof:
            counts, props = build_organ_fold_table_from_oof(organ_oof_df, organ)
            counts_p, props_p = build_organ_fold_table_from_oof(organ_oof_df, organ, dedupe_by_patient=True)
        else:
            counts, props = build_organ_fold_table_from_manifest(organ_manifest_cv, organ)  # noqa: F821
            counts_p, props_p = counts, props  # manifest fallback path: dedup not implemented, series-level only
        organ_tables_patient[organ] = (counts_p, props_p)
        counts.to_csv(OUT_DIR / f"organ_crop_{organ}_counts_series.csv")
        props.to_csv(OUT_DIR / f"organ_crop_{organ}_proportions_series.csv")
        counts_p.to_csv(OUT_DIR / f"organ_crop_{organ}_counts_patient.csv")
        props_p.to_csv(OUT_DIR / f"organ_crop_{organ}_proportions_patient.csv")
        md_sections.append(format_markdown_table(counts, props, organ.capitalize()))

    md_sections.append("### Organ-crop branch — patient-level (deduplicated)\n")
    for organ in ORGAN_TARGETS:
        counts_p, props_p = organ_tables_patient[organ]
        md_sections.append(format_markdown_table(counts_p, props_p, f"{organ.capitalize()} (patients)"))

    md_sections.append("### Full-volume branch (bowel / extravasation) — series-level\n")
    fv_tables_patient = {}
    for target, label_col in FV_TARGETS.items():
        if using_oof:
            counts, props = build_fv_fold_table_from_oof(fv_oof_df, label_col)
            counts_p, props_p = build_fv_fold_table_from_oof(fv_oof_df, label_col, dedupe_by_patient=True)
        else:
            manifest_label_col = "bowel_injury" if target == "bowel" else "extravasation_injury"
            counts, props = build_fv_fold_table_from_manifest(fv_manifest_cv, manifest_label_col)  # noqa: F821
            counts_p, props_p = counts, props
        fv_tables_patient[target] = (counts_p, props_p)
        counts.to_csv(OUT_DIR / f"full_volume_{target}_counts_series.csv")
        props.to_csv(OUT_DIR / f"full_volume_{target}_proportions_series.csv")
        counts_p.to_csv(OUT_DIR / f"full_volume_{target}_counts_patient.csv")
        props_p.to_csv(OUT_DIR / f"full_volume_{target}_proportions_patient.csv")
        md_sections.append(format_markdown_table(counts, props, target.capitalize()))

    md_sections.append("### Full-volume branch — patient-level (deduplicated)\n")
    for target in FV_TARGETS:
        counts_p, props_p = fv_tables_patient[target]
        md_sections.append(format_markdown_table(counts_p, props_p, f"{target.capitalize()} (patients)"))

    md = "\n".join(md_sections)
    (OUT_DIR / "fold_distribution_tables.md").write_text(md)
    print(md)
    print(f"\nWrote CSVs and markdown table to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
