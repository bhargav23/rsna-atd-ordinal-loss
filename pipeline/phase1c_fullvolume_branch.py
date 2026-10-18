"""
Phase 1c — Full-volume branch (bowel injury, active extravasation)

Motivation: segmentation-guided organ cropping completely fails for bowel
(diffuse structure spanning nearly the whole abdomen — a "bowel crop" is
barely distinguishable from the uncropped volume) and is inapplicable in
principle to extravasation (no corresponding segmentation mask exists).
An organ-crop bowel classifier achieved 0.0 recall, missing its single
validation positive case entirely. This branch resolves that by training
on the full, uncropped 64-slice abdominal volume instead.

Includes:
  - Multi-label classifier (shared backbone + attention pooling, 2-output head)
  - Multi-label focal loss with per-fold inverse-frequency pos_weight
  - Patient-level 5-fold CV with out-of-fold probability export
  - Fold-honest post-hoc logistic (Platt) calibration — raw probabilities are
    well-ranked (ROC-AUC 0.665 bowel, 0.572 extravasation) but poorly
    calibrated due to aggressive pos_weight; calibration recovers most of
    the value (log loss: bowel 0.763->0.216, extravasation 0.941->0.443, and
    weighted-by-competition-metric: bowel 0.335, extravasation 0.688)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression

from phase1_organ_branch import (
    TARGET_SIZE, N_FOLDS, device, set_seed, SliceAttentionPool, OUTPUT_DIR
)

import timm

FULL_VOLUME_N_SLICES = 64


# ============================================================
# Build the full-volume training population
# ============================================================
def build_full_volume_manifest(manifest_df, labels_df, organ_manifest, negative_cap_ratio=4, seed=42):
    """
    Subsamples the full-volume population to a tractable size, while FORCE-
    INCLUDING every organ-crop patient. This force-inclusion is essential:
    Phase 0 deliberately skipped saving full_volume for any series that
    already had all 4 organ crops (a disk-space optimization), which made
    the organ-crop and full-volume patient populations almost completely
    disjoint (confirmed: 2/128 overlap before the fix, 128/128 after).
    Requires the Phase 0 backfill step to have been run first (see
    phase0_preprocessing.run_full_volume_backfill) so has_full_volume is
    True for organ-crop patients too.
    """
    full_vol_manifest = manifest_df.merge(labels_df, on='patient_id', how='inner', validate="many_to_one")
    full_vol_manifest = full_vol_manifest[full_vol_manifest['has_full_volume'] == True].reset_index(drop=True)

    np.random.seed(seed)
    patient_level = full_vol_manifest.drop_duplicates(subset='patient_id').copy()
    positives = patient_level[(patient_level['bowel_injury'] == 1) | (patient_level['extravasation_injury'] == 1)]
    negatives = patient_level[(patient_level['bowel_injury'] == 0) & (patient_level['extravasation_injury'] == 0)]

    organ_crop_patients = set(organ_manifest['patient_id'].unique())
    forced_include = patient_level[patient_level['patient_id'].isin(organ_crop_patients)]

    n_negative_cap = min(len(negatives), len(positives) * negative_cap_ratio)
    negatives_sampled = negatives.sample(n=n_negative_cap, random_state=seed)

    subset_patients = pd.concat([positives, negatives_sampled, forced_include])['patient_id'].unique()
    full_vol_manifest_subset = full_vol_manifest[full_vol_manifest['patient_id'].isin(subset_patients)].reset_index(drop=True)

    print(f"Full-volume training set: {len(full_vol_manifest_subset)} series, "
          f"{full_vol_manifest_subset['patient_id'].nunique()} patients")
    print(f"Organ-crop patients included: "
          f"{len(organ_crop_patients & set(full_vol_manifest_subset['patient_id']))} / {len(organ_crop_patients)}")

    return full_vol_manifest_subset


def make_fv_patient_folds(full_vol_manifest, n_folds=N_FOLDS, seed=42):
    patient_labels = full_vol_manifest.drop_duplicates(subset='patient_id').reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignment = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(patient_labels['patient_id'], patient_labels['bowel_injury'])):
        for pid in patient_labels.iloc[test_idx]['patient_id']:
            fold_assignment[pid] = fold_idx
    fv_manifest_cv = full_vol_manifest.copy()
    fv_manifest_cv['fold'] = fv_manifest_cv['patient_id'].map(fold_assignment)
    return fv_manifest_cv


# ============================================================
# Dataset, model, loss
# ============================================================
class FullVolumeDataset(Dataset):
    def __init__(self, manifest_df, n_slices=FULL_VOLUME_N_SLICES, img_size=TARGET_SIZE):
        self.df = manifest_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with np.load(row['output_path']) as data:
            vol = data['full_volume'].astype(np.float32) / 255.0
        bowel_label = int(row['bowel_injury'])
        extrav_label = int(row['extravasation_injury'])
        # Competition sample weights: bowel injury=2x, extravasation injury=6x
        bowel_weight = 2.0 if bowel_label == 1 else 1.0
        extrav_weight = 6.0 if extrav_label == 1 else 1.0
        vol = torch.from_numpy(vol).unsqueeze(1)
        labels = torch.tensor([bowel_label, extrav_label], dtype=torch.float32)
        weights = torch.tensor([bowel_weight, extrav_weight], dtype=torch.float32)
        return vol, labels, weights


def make_fv_loader(manifest_df, batch_size=2, num_workers=0, shuffle=True):
    ds = FullVolumeDataset(manifest_df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class FullVolumeClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", n_outputs=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=1)
        feat_dim = self.backbone.num_features
        self.pool = SliceAttentionPool(feat_dim)
        self.head = nn.Linear(feat_dim, n_outputs)  # [bowel_logit, extravasation_logit]

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x).view(B, S, -1)
        return self.head(self.pool(feats))


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, pos_weights, gamma=2.0):
        super().__init__()
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, logits, targets, sample_weight=None):
        device = logits.device
        pos_weight = self.pos_weights.to(device)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt) ** self.gamma
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight.unsqueeze(0).expand_as(logits))
        loss = focal_factor * bce
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def compute_pos_weights(manifest_df):
    n = len(manifest_df)
    bowel_pos = manifest_df['bowel_injury'].sum()
    extrav_pos = manifest_df['extravasation_injury'].sum()
    return [(n - bowel_pos) / max(bowel_pos, 1), (n - extrav_pos) / max(extrav_pos, 1)]


# ============================================================
# CV training with out-of-fold probability export
# ============================================================
def train_fv_cv_with_oof(manifest_cv, epochs=5, seed=42, gamma=2.0, n_folds=N_FOLDS, batch_size=2):
    fold_results, oof_records = [], []
    for fold in range(n_folds):
        set_seed(seed)
        train_df = manifest_cv[manifest_cv['fold'] != fold]
        val_df = manifest_cv[manifest_cv['fold'] == fold]
        if len(val_df) == 0 or len(train_df) == 0:
            continue

        train_loader = make_fv_loader(train_df, batch_size=batch_size, shuffle=True)
        val_loader = make_fv_loader(val_df, batch_size=batch_size, shuffle=False)
        pos_weights = compute_pos_weights(train_df)
        print(f"[fold {fold}] train={len(train_df)}, val={len(val_df)}, pos_weights={pos_weights}")

        model = FullVolumeClassifier().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = MultiLabelFocalLoss(pos_weights, gamma=gamma)

        val_patient_ids_ordered = val_df['patient_id'].values
        val_series_ids_ordered = val_df['series_id'].values

        best_val_loss, best_preds = float('inf'), None
        for epoch in range(epochs):
            model.train()
            for vol, labels, weights in tqdm(train_loader, desc=f"fold{fold} epoch{epoch+1}"):
                vol, labels, weights = vol.to(device), labels.to(device), weights.to(device)
                optimizer.zero_grad()
                logits = model(vol)
                loss = criterion(logits, labels, sample_weight=weights)
                loss.backward(); optimizer.step()

            model.eval()
            val_losses, val_probs, val_labels = [], [], []
            with torch.no_grad():
                for vol, labels, weights in val_loader:
                    vol, labels, weights = vol.to(device), labels.to(device), weights.to(device)
                    logits = model(vol)
                    loss = criterion(logits, labels, sample_weight=weights)
                    val_losses.append(loss.item())
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())

            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = (np.concatenate(val_labels), np.concatenate(val_probs))

        val_labels_arr, val_probs_arr = best_preds
        val_preds_arr = (val_probs_arr >= 0.5).astype(int)

        for i in range(len(val_labels_arr)):
            oof_records.append({
                "patient_id": val_patient_ids_ordered[i], "series_id": val_series_ids_ordered[i], "fold": fold,
                "bowel_true": int(val_labels_arr[i, 0]), "extrav_true": int(val_labels_arr[i, 1]),
                "bowel_prob": float(val_probs_arr[i, 0]), "extrav_prob": float(val_probs_arr[i, 1]),
            })
        fold_results.append({
            "fold": fold, "val_loss": best_val_loss,
            "bowel_recall": recall_score(val_labels_arr[:, 0], val_preds_arr[:, 0], zero_division=0),
            "extrav_recall": recall_score(val_labels_arr[:, 1], val_preds_arr[:, 1], zero_division=0),
        })
    return pd.DataFrame(fold_results), pd.DataFrame(oof_records)


# ============================================================
# Fold-honest post-hoc calibration
# ============================================================
def fold_honest_calibrate(df, prob_col, true_col):
    """Fits a logistic (Platt) calibrator on 4 folds, applies to the 5th —
    avoids a second layer of adaptive overfitting on top of the model's own
    fold-honest OOF predictions."""
    calibrated = np.zeros(len(df))
    for fold in df['fold'].unique():
        train_mask = df['fold'] != fold
        test_mask = df['fold'] == fold
        X_train = df.loc[train_mask, prob_col].values.reshape(-1, 1)
        y_train = df.loc[train_mask, true_col].values
        X_test = df.loc[test_mask, prob_col].values.reshape(-1, 1)
        calibrator = LogisticRegression()
        calibrator.fit(X_train, y_train)
        calibrated[test_mask] = calibrator.predict_proba(X_test)[:, 1]
    return calibrated


def get_sample_weights(y_true, target_type):
    """Competition's own weighting scheme, for weighted-log-loss evaluation."""
    if target_type == 'bowel':
        return np.where(y_true == 1, 2, 1)
    elif target_type == 'extrav':
        return np.where(y_true == 1, 6, 1)
    return np.ones_like(y_true)


# ============================================================
# Run
# ============================================================
def run_fullvolume_branch(manifest_df, labels_df, organ_manifest, output_dir=OUTPUT_DIR):
    full_vol_manifest_subset = build_full_volume_manifest(manifest_df, labels_df, organ_manifest)
    fv_manifest_cv = make_fv_patient_folds(full_vol_manifest_subset)

    fv_results_path = output_dir / "fv_cv_results.csv"
    fv_oof_path = output_dir / "fv_oof_probabilities.csv"
    if fv_results_path.exists() and fv_oof_path.exists():
        fv_results_df = pd.read_csv(fv_results_path)
        fv_oof_df = pd.read_csv(fv_oof_path)
    else:
        fv_results_df, fv_oof_df = train_fv_cv_with_oof(fv_manifest_cv, epochs=5, seed=42)
        fv_results_df.to_csv(fv_results_path, index=False)
        fv_oof_df.to_csv(fv_oof_path, index=False)

    print(fv_results_df.to_string(index=False))

    fv_oof_df['bowel_prob_calibrated'] = fold_honest_calibrate(fv_oof_df, 'bowel_prob', 'bowel_true')
    fv_oof_df['extrav_prob_calibrated'] = fold_honest_calibrate(fv_oof_df, 'extrav_prob', 'extrav_true')

    # Unweighted calibrated log loss (diagnostic)
    print(f"\nBowel log loss calibrated: {log_loss(fv_oof_df['bowel_true'], fv_oof_df['bowel_prob_calibrated']):.4f}")
    print(f"Extravasation log loss calibrated: {log_loss(fv_oof_df['extrav_true'], fv_oof_df['extrav_prob_calibrated']):.4f}")

    # Competition-weighted calibrated log loss (the number that matters for the paper)
    bowel_w = get_sample_weights(fv_oof_df['bowel_true'], 'bowel')
    extrav_w = get_sample_weights(fv_oof_df['extrav_true'], 'extrav')
    bowel_loss_w = log_loss(fv_oof_df['bowel_true'], fv_oof_df['bowel_prob_calibrated'], sample_weight=bowel_w)
    extrav_loss_w = log_loss(fv_oof_df['extrav_true'], fv_oof_df['extrav_prob_calibrated'], sample_weight=extrav_w)
    print(f"Weighted Bowel Log Loss (Calibrated): {bowel_loss_w:.4f}")
    print(f"Weighted Extravasation Log Loss (Calibrated): {extrav_loss_w:.4f}")

    # Discrimination diagnostics (ranking quality independent of threshold/calibration)
    for label_name, true_col, prob_col in [("bowel", "bowel_true", "bowel_prob"),
                                              ("extravasation", "extrav_true", "extrav_prob")]:
        y_true, y_prob = fv_oof_df[true_col].values, fv_oof_df[prob_col].values
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        print(f"{label_name}: ROC-AUC={auc:.3f}, PR-AUC={ap:.3f} (baseline={y_true.mean():.3f})")

    fv_oof_df.to_csv(fv_oof_path, index=False)
    return fv_oof_df, fv_results_df


if __name__ == "__main__":
    from phase1_organ_branch import load_manifest_and_labels
    manifest_df, labels_df, organ_manifest = load_manifest_and_labels()
    fv_oof_df, fv_results_df = run_fullvolume_branch(manifest_df, labels_df, organ_manifest)
