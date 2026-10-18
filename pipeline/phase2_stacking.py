"""
Phase 2 — Stacking and final pipeline assembly

Combines:
  - Organ-crop branch OOF probabilities (liver/spleen/kidney, from Phase 1b)
  - Full-volume branch calibrated OOF probabilities (bowel/extravasation, from Phase 1c)
  - Series metadata (aortic_hu, incomplete_organ)

CONFIRMED FINDING (stacking ablation): a single LightGBM meta-learner trained
across all 5 targets UNDERPERFORMS a hybrid architecture that:
  - passes through the organ-crop branch's own calibrated CV probabilities
    directly for liver/spleen/kidney (only 128 patients available — too few
    for a meta-learner to improve on, and it actively overfits at this scale)
  - reserves LightGBM stacking only for bowel/extravasation, where 1,028
    patients are available (abundant enough for the meta-learner to add value)

Full stack:   overall weighted log loss = 0.8387
Hybrid:       overall weighted log loss = 0.7775   <-- final architecture
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

OUTPUT_DIR = Path("/kaggle/working")


# ============================================================
# Competition weighted log-loss metric
# ============================================================
def competition_weighted_log_loss(y_true_df, y_pred_df):
    """
    Implements the RSNA ATD competition's sample-weighted log loss:
      healthy=1x, low-grade=2x, high-grade=4x (solid organs)
      bowel: healthy=1x, injury=2x
      extravasation: healthy=1x, injury=6x
      any_injury (derived, 1 - P(all healthy)): weight 6x
    Final score = unweighted mean across the 6 group losses.
    """
    weights = {'bowel_healthy': 1, 'bowel_injury': 2,
               'extravasation_healthy': 1, 'extravasation_injury': 6,
               'kidney_healthy': 1, 'kidney_low': 2, 'kidney_high': 4,
               'liver_healthy': 1, 'liver_low': 2, 'liver_high': 4,
               'spleen_healthy': 1, 'spleen_low': 2, 'spleen_high': 4}
    organ_groups = {
        'bowel': ['bowel_healthy', 'bowel_injury'],
        'extravasation': ['extravasation_healthy', 'extravasation_injury'],
        'kidney': ['kidney_healthy', 'kidney_low', 'kidney_high'],
        'liver': ['liver_healthy', 'liver_low', 'liver_high'],
        'spleen': ['spleen_healthy', 'spleen_low', 'spleen_high'],
    }

    losses = []
    for organ, cols in organ_groups.items():
        y_true_group = y_true_df[cols].values
        y_pred_group = np.clip(y_pred_df[cols].values, 1e-7, 1 - 1e-7)
        y_pred_group = y_pred_group / y_pred_group.sum(axis=1, keepdims=True)
        sample_weights = sum(y_true_group[:, i] * weights[col] for i, col in enumerate(cols))
        per_sample_loss = -np.sum(y_true_group * np.log(y_pred_group), axis=1)
        losses.append(np.sum(per_sample_loss * sample_weights) / np.sum(sample_weights))

    any_true = 1 - (y_true_df['bowel_healthy'] * y_true_df['extravasation_healthy'] *
                    y_true_df['kidney_healthy'] * y_true_df['liver_healthy'] * y_true_df['spleen_healthy'])
    any_pred = np.clip(1 - (y_pred_df['bowel_healthy'] * y_pred_df['extravasation_healthy'] *
                            y_pred_df['kidney_healthy'] * y_pred_df['liver_healthy'] * y_pred_df['spleen_healthy']),
                        1e-7, 1 - 1e-7)
    losses.append(-np.mean(any_true * np.log(any_pred) + (1 - any_true) * np.log(1 - any_pred)))

    return np.mean(losses), dict(zip(list(organ_groups.keys()) + ['any_injury'], losses))


# ============================================================
# Feature matrix assembly
# ============================================================
def build_stacking_feature_matrix(organ_oof_df, fv_oof_df, series_meta_path):
    series_meta = pd.read_csv(series_meta_path)[['patient_id', 'series_id', 'aortic_hu', 'incomplete_organ']]

    organ_wide = organ_oof_df.pivot_table(
        index='patient_id', columns='organ',
        values=[c for c in organ_oof_df.columns if c.startswith('prob_class')],
        aggfunc='mean'
    )
    organ_wide.columns = [f"{organ}_{col}" for col, organ in organ_wide.columns]
    organ_wide = organ_wide.reset_index()

    fv_patient_probs = fv_oof_df.groupby('patient_id')[['bowel_prob_calibrated', 'extrav_prob_calibrated']].mean().reset_index()

    meta_patient = series_meta.groupby('patient_id').agg(
        aortic_hu=('aortic_hu', 'mean'), incomplete_organ=('incomplete_organ', 'max')
    ).reset_index()

    stack_features = organ_wide.merge(fv_patient_probs, on='patient_id', how='outer')
    stack_features = stack_features.merge(meta_patient, on='patient_id', how='left')

    # Diagnostic: confirms the backfill fix actually closed the organ/full-volume overlap gap
    organ_patients_set = set(organ_oof_df['patient_id'].unique())
    fv_patients_set = set(fv_patient_probs['patient_id'].unique())
    overlap = organ_patients_set & fv_patients_set
    print(f"Organ-branch patients: {len(organ_patients_set)}")
    print(f"Full-volume-branch patients: {len(fv_patients_set)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/len(organ_patients_set):.1%} of organ patients)")

    return stack_features


# ============================================================
# Full LightGBM stack (all 5 targets) — CONFIRMED WORSE than hybrid, kept
# here for the ablation comparison
# ============================================================
def run_full_stack(stack_features, labels_df, seed=42):
    import lightgbm as lgb

    stack_full = stack_features.merge(labels_df, on='patient_id', how='inner')
    feature_cols = [c for c in stack_features.columns if c != 'patient_id']

    stacker_oof_preds = {organ: np.zeros((len(stack_full), n)) for organ, n in
                         [('liver', 3), ('spleen', 3), ('kidney', 3), ('bowel', 2), ('extravasation', 2)]}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(stack_full, stack_full['any_injury'])):
        for organ, class_cols in [('liver', ['liver_healthy', 'liver_low', 'liver_high']),
                                   ('spleen', ['spleen_healthy', 'spleen_low', 'spleen_high']),
                                   ('kidney', ['kidney_healthy', 'kidney_low', 'kidney_high']),
                                   ('bowel', ['bowel_healthy', 'bowel_injury']),
                                   ('extravasation', ['extravasation_healthy', 'extravasation_injury'])]:
            y_train = stack_full.iloc[train_idx][class_cols].values.argmax(axis=1)
            X_train = stack_full.iloc[train_idx][feature_cols]
            X_val = stack_full.iloc[val_idx][feature_cols]
            clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1)
            clf.fit(X_train, y_train)
            stacker_oof_preds[organ][val_idx] = clf.predict_proba(X_val)

    final_preds = _assemble_preds(stacker_oof_preds)
    score, breakdown = competition_weighted_log_loss(stack_full, final_preds)
    return score, breakdown, stack_full, stacker_oof_preds


def _assemble_preds(stacker_oof_preds):
    return pd.DataFrame({
        'liver_healthy': stacker_oof_preds['liver'][:, 0], 'liver_low': stacker_oof_preds['liver'][:, 1], 'liver_high': stacker_oof_preds['liver'][:, 2],
        'spleen_healthy': stacker_oof_preds['spleen'][:, 0], 'spleen_low': stacker_oof_preds['spleen'][:, 1], 'spleen_high': stacker_oof_preds['spleen'][:, 2],
        'kidney_healthy': stacker_oof_preds['kidney'][:, 0], 'kidney_low': stacker_oof_preds['kidney'][:, 1], 'kidney_high': stacker_oof_preds['kidney'][:, 2],
        'bowel_healthy': stacker_oof_preds['bowel'][:, 0], 'bowel_injury': stacker_oof_preds['bowel'][:, 1],
        'extravasation_healthy': stacker_oof_preds['extravasation'][:, 0], 'extravasation_injury': stacker_oof_preds['extravasation'][:, 1],
    })


# ============================================================
# FINAL ARCHITECTURE — hybrid: direct organ pass-through + stacked bowel/extrav
# ============================================================
def run_hybrid_pipeline(stack_features, organ_oof_df, fv_stacked_preds, labels_df):
    """
    stack_features: patient-level feature matrix (from build_stacking_feature_matrix)
    organ_oof_df:   organ-branch OOF probabilities (used directly, NOT re-stacked)
    fv_stacked_preds: LightGBM-stacked bowel/extravasation predictions (2 columns each)
    """
    stack_full = stack_features.merge(labels_df, on='patient_id', how='inner')

    organ_direct = organ_oof_df.pivot_table(
        index='patient_id', columns='organ',
        values=[c for c in organ_oof_df.columns if c.startswith('prob_class')],
        aggfunc='mean'
    )
    organ_direct.columns = [f"{organ}_{col}" for col, organ in organ_direct.columns]
    organ_direct = organ_direct.reset_index()

    direct_preds = pd.DataFrame({'patient_id': stack_full['patient_id']})
    direct_preds = direct_preds.merge(organ_direct, on='patient_id', how='left')

    # Fill NaNs (patients with no organ-crop data) with population base rates
    for organ, cols in [('kidney', ['healthy', 'low', 'high']),
                         ('liver', ['healthy', 'low', 'high']),
                         ('spleen', ['healthy', 'low', 'high'])]:
        base_rates = labels_df[[f'{organ}_{c}' for c in cols]].mean().values
        for i, c in enumerate(cols):
            direct_preds[f'{organ}_prob_class{i}'] = direct_preds[f'{organ}_prob_class{i}'].fillna(base_rates[i])

    final_preds_hybrid = pd.DataFrame({
        'liver_healthy': direct_preds['liver_prob_class0'], 'liver_low': direct_preds['liver_prob_class1'], 'liver_high': direct_preds['liver_prob_class2'],
        'spleen_healthy': direct_preds['spleen_prob_class0'], 'spleen_low': direct_preds['spleen_prob_class1'], 'spleen_high': direct_preds['spleen_prob_class2'],
        'kidney_healthy': direct_preds['kidney_prob_class0'], 'kidney_low': direct_preds['kidney_prob_class1'], 'kidney_high': direct_preds['kidney_prob_class2'],
        'bowel_healthy': fv_stacked_preds['bowel_healthy'], 'bowel_injury': fv_stacked_preds['bowel_injury'],
        'extravasation_healthy': fv_stacked_preds['extravasation_healthy'], 'extravasation_injury': fv_stacked_preds['extravasation_injury'],
    })

    score, breakdown = competition_weighted_log_loss(stack_full, final_preds_hybrid)
    return score, breakdown


if __name__ == "__main__":
    organ_oof_df = pd.read_csv(OUTPUT_DIR / "organ_oof_probabilities.csv")
    fv_oof_df = pd.read_csv(OUTPUT_DIR / "fv_oof_probabilities.csv")
    labels_df = pd.read_csv(OUTPUT_DIR / "subset_labels.csv")
    SERIES_META_PATH = Path("/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection/train_series_meta.csv")

    stack_features = build_stacking_feature_matrix(organ_oof_df, fv_oof_df, SERIES_META_PATH)

    # Ablation: full stack vs. hybrid
    full_score, full_breakdown, stack_full, stacker_oof_preds = run_full_stack(stack_features, labels_df)
    print(f"\nFull LightGBM stack: {full_score:.4f}")
    print(full_breakdown)

    fv_stacked_preds = _assemble_preds(stacker_oof_preds)[
        ['bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'extravasation_injury']]
    hybrid_score, hybrid_breakdown = run_hybrid_pipeline(stack_features, organ_oof_df, fv_stacked_preds, labels_df)
    print(f"\nHybrid (final architecture): {hybrid_score:.4f}")
    print(hybrid_breakdown)
