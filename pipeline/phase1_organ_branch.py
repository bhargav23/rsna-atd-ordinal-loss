"""
Phase 1a/1b — Organ-crop branch (liver, spleen, kidney)

- 2.5D EfficientNet-B0 + attention-pooling classifier per organ
- Baseline: standard sample-weighted cross-entropy
- Ordinal Present-Label Upweighting Loss (novel contribution): effective-number
  class balancing + ordinal expected-rank distance penalty + present-label boost
- Patient-level 5-fold stratified cross-validation (corrects an earlier,
  adaptively-overfit single-split evaluation — see confirmed CV results below)
- Out-of-fold probability export for downstream stacking (Phase 2)

Confirmed final configs from CV-based hyperparameter search:
  liver:  baseline CE (ordinal loss showed no benefit; class-2 recall = 0.0
          at every alpha/boost setting tested, including boost up to 4.0 —
          attributed to sample scarcity, ~27 high-grade examples total)
  spleen: ordinal, alpha=3.0, present_boost=1.0
  kidney: ordinal, alpha=1.0, present_boost=1.0
(present_boost=1.0 effectively disables the present-label boost term; the
 ordinal-distance penalty alone drove the observed improvement for both organs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# ============================================================
# Config
# ============================================================
PHASE0_DIR = Path("/kaggle/input/datasets/<username>/rsna-atd-phase0-preprocessed/processed")
MANIFEST_PATH = PHASE0_DIR / "manifest.csv"
LABELS_PATH = PHASE0_DIR / "subset_labels.csv"

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 256
N_SLICES_PER_ORGAN = 32
ORGAN_NUM_CLASSES = {"liver": 3, "spleen": 3, "kidney": 3, "bowel": 2}
SEEDS = [42, 123, 2024, 7, 555]
N_FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data loading (with backfill path patch applied — see Phase 0)
# ============================================================
def load_manifest_and_labels(phase0_dir=PHASE0_DIR, backfill_dir: Path = None):
    manifest_df = pd.read_csv(phase0_dir / "manifest.csv")
    manifest_df['output_path'] = manifest_df['output_path'].apply(lambda p: str(phase0_dir / Path(p).name))
    manifest_df['organs_cropped'] = manifest_df['organs_cropped'].apply(literal_eval)
    manifest_df['resize_factors'] = manifest_df['resize_factors'].apply(literal_eval)

    if backfill_dir is not None:
        backfill_log = pd.read_csv(backfill_dir / "backfill_log.csv")
        backfill_success = backfill_log[backfill_log['status'] == 'success'].copy()
        backfill_success['new_path'] = backfill_success['new_path'].apply(
            lambda p: str(backfill_dir / "backfilled" / Path(p).name))
        backfill_map = dict(zip(backfill_success['output_path'], backfill_success['new_path']))

        manifest_df['output_path'] = manifest_df['output_path'].apply(
            lambda p: backfill_map.get(str(p), str(p)))
        manifest_df['has_full_volume'] = (
            manifest_df['has_full_volume'] | manifest_df['output_path'].isin(backfill_map.values()))

    labels_df = pd.read_csv(phase0_dir / "subset_labels.csv")
    if labels_df['patient_id'].duplicated().any():
        raise ValueError("Duplicate patient labels found")

    manifest_labeled = manifest_df.merge(labels_df, on='patient_id', how='inner', validate="many_to_one")
    organ_manifest = manifest_labeled[manifest_labeled['organs_cropped'].apply(len) > 0].reset_index(drop=True)
    return manifest_df, labels_df, organ_manifest


def get_organ_label(row, organ):
    if organ == "bowel":
        return int(row['bowel_injury'])
    healthy, low, high = row[f'{organ}_healthy'], row[f'{organ}_low'], row[f'{organ}_high']
    return int(np.argmax([healthy, low, high]))


class OrganCropDataset(Dataset):
    def __init__(self, manifest_df, organ, n_slices=N_SLICES_PER_ORGAN, img_size=TARGET_SIZE):
        self.df = manifest_df[manifest_df['organs_cropped'].apply(lambda x: organ in x)].reset_index(drop=True)
        self.organ = organ

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with np.load(row['output_path']) as data:
            vol = data[self.organ].astype(np.float32) / 255.0
        label = get_organ_label(row, self.organ)
        weight = self._get_sample_weight(row)
        vol = torch.from_numpy(vol).unsqueeze(1)
        return vol, label, weight

    def _get_sample_weight(self, row):
        """Competition-style severity sample weights (1/2/4)."""
        if self.organ == "bowel":
            return 2.0 if row['bowel_injury'] == 1 else 1.0
        healthy, low, high = row[f'{self.organ}_healthy'], row[f'{self.organ}_low'], row[f'{self.organ}_high']
        if high == 1: return 4.0
        if low == 1: return 2.0
        return 1.0


def make_cv_loader(manifest_df, organ, fold, mode, batch_size=4):
    df_split = manifest_df[manifest_df['fold'] != fold] if mode == 'train' else manifest_df[manifest_df['fold'] == fold]
    ds = OrganCropDataset(df_split, organ)
    # num_workers=0: avoids a DataLoader multiprocessing deadlock observed on
    # Kaggle immediately after a `pip install` in the same session.
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=0, pin_memory=True)


def make_patient_folds(labels_df, organ_manifest, n_folds=N_FOLDS, seed=42):
    """Patient-level (not series-level) stratified folds — prevents a patient's
    multiple scans from crossing a fold boundary."""
    patients_with_crops = organ_manifest['patient_id'].unique()
    patient_labels = labels_df[labels_df['patient_id'].isin(patients_with_crops)].drop_duplicates(
        subset='patient_id').reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignment = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(patient_labels['patient_id'], patient_labels['any_injury'])):
        for pid in patient_labels.iloc[test_idx]['patient_id']:
            fold_assignment[pid] = fold_idx
    organ_manifest_cv = organ_manifest.copy()
    organ_manifest_cv['fold'] = organ_manifest_cv['patient_id'].map(fold_assignment)
    if organ_manifest_cv['fold'].isna().any():
        raise ValueError("Some rows have no fold assignment")
    return organ_manifest_cv


# ============================================================
# Model: shared 2D EfficientNet-B0 backbone + attention pooling over slices
# ============================================================
# !pip install -q timm
import timm


class SliceAttentionPool(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(feat_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, feats):  # feats: (B, n_slices, feat_dim)
        weights = torch.softmax(self.attn(feats), dim=1)
        return (feats * weights).sum(dim=1)


class OrganClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name="efficientnet_b0"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=1)
        feat_dim = self.backbone.num_features
        self.pool = SliceAttentionPool(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):  # x: (B, n_slices, 1, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x).view(B, S, -1)
        return self.head(self.pool(feats))


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================
# Ordinal Present-Label Upweighting Loss (novel contribution)
# ============================================================
class OrdinalPresentLabelUpweightingLoss(nn.Module):
    """
    Generalizes present-label loss upweighting (binary presence/absence in
    prior work) to ordinal severity grading. Combines:
      (a) Effective-number class balancing (Cui et al., 2019)
      (b) Ordinal expected-rank distance penalty (squared)
      (c) Present-label boost (multiplicative, applied when target > 0)

    CONFIRMED FINDING: in our CV results, the present-label boost term
    (component c) contributed nothing at any tested magnitude (1.5 up to 4.0)
    — the observed improvement over baseline is attributable to the ordinal
    distance penalty (b) alone. Best configs use present_boost=1.0 (disabled).
    """
    def __init__(self, class_counts, beta=0.999, ordinal_alpha=0.5, present_boost=1.5, focal_gamma=1.0):
        super().__init__()
        class_counts = np.array(class_counts, dtype=np.float32)
        if np.any(class_counts == 0):
            raise ValueError(f"Zero-count class in training split: {class_counts.tolist()}")
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        self.ordinal_alpha = ordinal_alpha
        self.present_boost = present_boost
        self.focal_gamma = focal_gamma
        self.num_classes = len(class_counts)

    def forward(self, logits, targets, sample_weight=None):
        device = logits.device
        class_weights = self.class_weights.to(device)

        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)

        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - target_probs) ** self.focal_gamma
        ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        cls_w = class_weights[targets]
        base_loss = focal_factor * cls_w * ce

        class_ranks = torch.arange(self.num_classes, device=device).float()
        expected_rank = (probs * class_ranks.unsqueeze(0)).sum(dim=1)
        ordinal_penalty = (expected_rank - targets.float()) ** 2

        present_mask = (targets > 0).float()
        present_weight = 1.0 + (self.present_boost - 1.0) * present_mask

        loss = present_weight * (base_loss + self.ordinal_alpha * ordinal_penalty)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# NOTE (reproducibility): confirmed experiments below do NOT pass the dataset's
# external per-example severity sample_weight into the ordinal loss — results
# reflect class-balancing + ordinal penalty + present-label boost only.
ORDINAL_USE_SAMPLE_WEIGHT = False


def get_class_counts(manifest_df, organ):
    labels = manifest_df.apply(lambda row: get_organ_label(row, organ), axis=1)
    return [int((labels == c).sum()) for c in range(ORGAN_NUM_CLASSES[organ])]


# ============================================================
# CV training with out-of-fold probability export
# ============================================================
def train_organ_cv_with_oof(organ, manifest_cv, method="baseline", epochs=5, seed=42,
                              ordinal_alpha=0.5, present_boost=1.5, focal_gamma=1.0, n_folds=N_FOLDS):
    oof_records = []
    for fold in range(n_folds):
        set_seed(seed)
        train_loader = make_cv_loader(manifest_cv, organ, fold, 'train')
        val_loader = make_cv_loader(manifest_cv, organ, fold, 'val')
        if len(val_loader.dataset) == 0 or len(train_loader.dataset) == 0:
            print(f"[{organ}] fold {fold}: empty split, skipping"); continue

        val_patient_ids = val_loader.dataset.df['patient_id'].values
        val_series_ids = val_loader.dataset.df['series_id'].values

        model = OrganClassifier(num_classes=ORGAN_NUM_CLASSES[organ]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        if method == "baseline":
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            class_counts = get_class_counts(train_loader.dataset.df, organ)
            criterion = OrdinalPresentLabelUpweightingLoss(
                class_counts, ordinal_alpha=ordinal_alpha, present_boost=present_boost, focal_gamma=focal_gamma)

        best_val_loss, best_probs, best_labels = float('inf'), None, None
        for epoch in range(epochs):
            model.train()
            for vol, label, weight in train_loader:
                vol, label, weight = vol.to(device), label.to(device), weight.to(device).float()
                optimizer.zero_grad()
                logits = model(vol)
                if method == "baseline":
                    loss = (criterion(logits, label) * weight).mean()
                else:
                    sw = weight if ORDINAL_USE_SAMPLE_WEIGHT else None
                    loss = criterion(logits, label, sample_weight=sw)
                loss.backward(); optimizer.step()

            model.eval()
            val_losses, val_probs_list, val_labels_list = [], [], []
            with torch.no_grad():
                for vol, label, weight in val_loader:
                    vol, label, weight = vol.to(device), label.to(device), weight.to(device).float()
                    logits = model(vol)
                    if method == "baseline":
                        loss = (criterion(logits, label) * weight).mean()
                    else:
                        sw = weight if ORDINAL_USE_SAMPLE_WEIGHT else None
                        loss = criterion(logits, label, sample_weight=sw)
                    val_losses.append(loss.item())
                    val_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                    val_labels_list.append(label.cpu().numpy())

            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probs = np.concatenate(val_probs_list)
                best_labels = np.concatenate(val_labels_list)

        for i in range(len(best_labels)):
            record = {"patient_id": val_patient_ids[i], "series_id": val_series_ids[i],
                      "organ": organ, "fold": fold, "true_label": int(best_labels[i])}
            for c in range(ORGAN_NUM_CLASSES[organ]):
                record[f"prob_class{c}"] = float(best_probs[i, c])
            oof_records.append(record)
        print(f"[{organ}] fold {fold} done, best_val_loss={best_val_loss:.4f}")

    return pd.DataFrame(oof_records)


# ============================================================
# Run: final confirmed configs, checkpointed
# ============================================================
FINAL_ORGAN_CONFIGS = {
    "liver":  {"method": "baseline"},
    "spleen": {"method": "ordinal", "ordinal_alpha": 3.0, "present_boost": 1.0},
    "kidney": {"method": "ordinal", "ordinal_alpha": 1.0, "present_boost": 1.0},
}


def run_organ_oof_export(organ_manifest_cv, output_dir=OUTPUT_DIR):
    organ_oof_path = output_dir / "organ_oof_probabilities.csv"
    if organ_oof_path.exists():
        return pd.read_csv(organ_oof_path)

    frames = []
    for organ, cfg in FINAL_ORGAN_CONFIGS.items():
        print(f"\n=== OOF export: {organ}, {cfg} ===")
        frames.append(train_organ_cv_with_oof(organ, organ_manifest_cv, **cfg))
    organ_oof_df = pd.concat(frames, ignore_index=True)
    organ_oof_df.to_csv(organ_oof_path, index=False)
    return organ_oof_df


if __name__ == "__main__":
    manifest_df, labels_df, organ_manifest = load_manifest_and_labels()
    organ_manifest_cv = make_patient_folds(labels_df, organ_manifest)
    organ_oof_df = run_organ_oof_export(organ_manifest_cv)
    print(organ_oof_df.groupby('organ').size())
