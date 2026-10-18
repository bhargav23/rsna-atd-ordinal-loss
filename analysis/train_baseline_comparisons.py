"""
train_baseline_comparisons.py

Runs the LDAM and weighted-focal-loss baselines cited in Related Work
(Sections 2.1, 2.2) but never actually trained -- the last fully open item
from the meta-review. Uses the exact same 5-fold, patient-level CV protocol,
OrganClassifier architecture, and OOF export schema as train_organ_cv_with_oof
in phase1_organ_branch.py, so outputs plug directly into
recompute_organ_recall.py / significance_testing.py without modification.

Run for spleen and kidney -- the two organs where the paper's own ordinal
loss showed a genuine improvement over baseline CE (Section 4.2), which is
exactly where a reader will ask "how does this compare to standard
imbalance-correction baselines?" Liver is skipped: its ordinal loss already
showed a null/negative result (Section 5.3), so a further baseline
comparison there is lower-priority and can be added later if reviewers ask.

DRW (deferred re-weighting) schedule for LDAM: trains with unweighted CE for
the first 60% of epochs, then switches to inverse-frequency class weights
for the remainder -- adapted from the original DRW schedule (~80% deferred)
to fit within our 5-epoch budget (3 unweighted epochs, 2 weighted epochs),
since epochs=5 is too short for a literal 80/20 split to have any effect in
the weighted phase.

Same session requirement as the other generate_*.py scripts: needs
OrganClassifier, make_cv_loader, set_seed, ORGAN_NUM_CLASSES,
get_class_counts, device, and organ_manifest_cv already defined in this
kernel (run phase1_organ_branch.py's setup cells first, e.g. via
liver_ordinal_check_full.py's setup portion).

Output: /kaggle/working/organ_ldam_oof.csv, /kaggle/working/organ_weightedfocal_oof.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path


# ============================================================
# Loss implementations (merged from baseline_losses.py)
# ============================================================


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin loss (Cao et al., 2019, cited as [3]).

    Per-class margin: Delta_j = C / n_j^(1/4), scaled so max margin = max_margin.
    Applied by subtracting the true class's margin from its own logit before
    scaling by `scale` and computing cross-entropy -- this pushes minority
    classes' decision boundaries further from the majority class than a flat
    softmax would, without any reference to ordinal structure (unlike our
    own ordinal-distance penalty).

    Supports Deferred Re-Weighting (DRW): pass `weight` (per-class tensor)
    to switch on inverse-frequency reweighting after an initial unweighted
    training phase -- call `set_drw_weight()` externally once the deferred
    phase begins (see train_baseline_comparisons.py's epoch loop).
    """

    def __init__(self, class_counts, max_margin=0.5, scale=30.0):
        super().__init__()
        class_counts = np.array(class_counts, dtype=np.float32)
        if np.any(class_counts == 0):
            raise ValueError(f"Zero-count class in training split: {class_counts.tolist()}")
        margins = 1.0 / np.power(class_counts, 0.25)
        margins = margins * (max_margin / np.max(margins))
        self.margins = torch.tensor(margins, dtype=torch.float32)
        self.scale = scale
        self.weight = None  # set via set_drw_weight() for the deferred re-weighting phase

    def set_drw_weight(self, class_counts):
        """Call once, when entering the deferred re-weighting phase."""
        class_counts = np.array(class_counts, dtype=np.float32)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)
        self.weight = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets, sample_weight=None):
        device = logits.device
        margins = self.margins.to(device)
        batch_margin = margins[targets]

        # Subtract the true class's margin from its own logit only, then scale.
        true_class_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        logits_adj = logits.clone()
        logits_adj.scatter_(1, targets.unsqueeze(1), (true_class_logit - batch_margin).unsqueeze(1))
        logits_adj = logits_adj * self.scale

        class_weight = self.weight.to(device) if self.weight is not None else None
        loss = F.cross_entropy(logits_adj, targets, weight=class_weight, reduction="none")
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


class WeightedFocalLoss(nn.Module):
    """
    Standard multi-class focal loss (Lin et al., 2017, cited as [2]) with
    inverse-frequency per-class weighting -- a common, simple imbalance
    baseline distinct from our own effective-number-weighted ordinal loss.
    No ordinal-distance term: treats classes as flat categories.
    """

    def __init__(self, class_counts, gamma=2.0):
        super().__init__()
        class_counts = np.array(class_counts, dtype=np.float32)
        if np.any(class_counts == 0):
            raise ValueError(f"Zero-count class in training split: {class_counts.tolist()}")
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, logits, targets, sample_weight=None):
        device = logits.device
        class_weights = self.class_weights.to(device)

        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - target_probs) ** self.gamma
        ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        cls_w = class_weights[targets]

        loss = focal_factor * cls_w * ce
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

ORGANS_TO_RUN = ["spleen", "kidney"]
EPOCHS = 5
DRW_START_EPOCH = 3  # 0-indexed: epochs 0,1,2 unweighted; epochs 3,4 use DRW weights
N_FOLDS = 5

LDAM_OUTPUT_PATH = Path("/kaggle/working/organ_ldam_oof.csv")
WFL_OUTPUT_PATH = Path("/kaggle/working/organ_weightedfocal_oof.csv")


def train_organ_cv_with_baseline_loss(organ, manifest_cv, loss_name, epochs=EPOCHS, seed=42, n_folds=N_FOLDS):
    """
    Mirrors train_organ_cv_with_oof's structure exactly (best-val-loss
    checkpoint selection per fold, same OOF record schema) but swaps in
    LDAM or weighted focal loss instead of baseline CE / the ordinal loss.
    """
    oof_records = []
    for fold in range(n_folds):
        set_seed(seed)  # noqa: F821
        train_loader = make_cv_loader(manifest_cv, organ, fold, "train")  # noqa: F821
        val_loader = make_cv_loader(manifest_cv, organ, fold, "val")  # noqa: F821
        if len(val_loader.dataset) == 0 or len(train_loader.dataset) == 0:
            print(f"[{organ}/{loss_name}] fold {fold}: empty split, skipping")
            continue

        val_patient_ids = val_loader.dataset.df["patient_id"].values
        val_series_ids = val_loader.dataset.df["series_id"].values

        model = OrganClassifier(num_classes=ORGAN_NUM_CLASSES[organ]).to(device)  # noqa: F821
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        class_counts = get_class_counts(train_loader.dataset.df, organ)  # noqa: F821
        if loss_name == "ldam":
            criterion = LDAMLoss(class_counts, max_margin=0.5, scale=30.0)
        elif loss_name == "weighted_focal":
            criterion = WeightedFocalLoss(class_counts, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

        best_val_loss, best_probs, best_labels = float("inf"), None, None
        for epoch in range(epochs):
            if loss_name == "ldam" and epoch == DRW_START_EPOCH:
                print(f"[{organ}/ldam] fold {fold}: entering DRW phase at epoch {epoch}")
                criterion.set_drw_weight(class_counts)

            model.train()
            for vol, label, weight in train_loader:
                vol, label, weight = vol.to(device), label.to(device), weight.to(device).float()  # noqa: F821
                optimizer.zero_grad()
                logits = model(vol)
                loss = criterion(logits, label, sample_weight=weight)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses, val_probs_list, val_labels_list = [], [], []
            with torch.no_grad():
                for vol, label, weight in val_loader:
                    vol, label, weight = vol.to(device), label.to(device), weight.to(device).float()  # noqa: F821
                    logits = model(vol)
                    loss = criterion(logits, label, sample_weight=weight)
                    val_losses.append(loss.item())
                    val_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                    val_labels_list.append(label.cpu().numpy())

            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probs = np.concatenate(val_probs_list)
                best_labels = np.concatenate(val_labels_list)

        for i in range(len(best_labels)):
            record = {
                "patient_id": val_patient_ids[i], "series_id": val_series_ids[i],
                "organ": organ, "fold": fold, "true_label": int(best_labels[i]),
            }
            for c in range(ORGAN_NUM_CLASSES[organ]):  # noqa: F821
                record[f"prob_class{c}"] = float(best_probs[i, c])
            oof_records.append(record)
        print(f"[{organ}/{loss_name}] fold {fold} done, best_val_loss={best_val_loss:.4f}")

    return pd.DataFrame(oof_records)


def main():
    try:
        OrganClassifier  # noqa: F821
    except NameError:
        raise RuntimeError(
            "OrganClassifier is not defined in this session. Run the "
            "phase1_organ_branch.py setup cells first, then re-run this."
        )
    try:
        organ_manifest_cv  # noqa: F821
    except NameError:
        raise RuntimeError(
            "organ_manifest_cv is not defined in this session. Build it via "
            "make_patient_folds(labels_df, organ_manifest) first, then re-run this."
        )

    for loss_name, output_path in [("ldam", LDAM_OUTPUT_PATH), ("weighted_focal", WFL_OUTPUT_PATH)]:
        if output_path.exists():
            print(f"{output_path} already exists — skipping {loss_name} (delete file to force re-run).")
            continue
        frames = []
        for organ in ORGANS_TO_RUN:
            print(f"\n=== {loss_name} baseline: {organ} ===")
            frames.append(train_organ_cv_with_baseline_loss(organ, organ_manifest_cv, loss_name))
        result_df = pd.concat(frames, ignore_index=True)
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved {loss_name} results to {output_path}")


if __name__ == "__main__":
    main()
