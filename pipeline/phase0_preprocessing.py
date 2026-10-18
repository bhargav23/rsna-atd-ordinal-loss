"""
Phase 0 — RSNA Abdominal Trauma Detection: DICOM preprocessing pipeline

Produces:
  - Segmentation-guided organ crops (liver, spleen, kidney, bowel) as .npz files
  - Full-volume representations for series without organ crops
  - manifest.csv tracking every processed series

Run as a sequence of cells in a Kaggle notebook (GPU not required for this phase).
Designed for Kaggle Save Version mode: resumable via manifest/failed_log checkpointing.
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-rle python-gdcm nibabel

import pydicom
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import cv2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CELL 2 — Config & paths
# ============================================================
COMP_DIR = Path("/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection")
TRAIN_CSV        = COMP_DIR / "train_2024.csv"
SERIES_META_CSV  = COMP_DIR / "train_series_meta.csv"
IMG_LEVEL_CSV    = COMP_DIR / "image_level_labels_2024.csv"
DEMO_CSV         = COMP_DIR / "train_demographics_2024.csv"
TRAIN_IMAGES_DIR = COMP_DIR / "train_images"
SEG_DIR          = COMP_DIR / "segmentations"

OUTPUT_DIR = Path("/kaggle/working/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE        = 256
N_SLICES_PER_ORGAN  = 32
N_PATIENTS_SUBSET  = 1200


# ============================================================
# CELL 0 — Resume from previous Save Version (run FIRST, always)
# ============================================================
def resume_from_previous_version(prev_output_dir: Path, output_dir: Path):
    """Copy forward any previously-processed files so a fresh Kaggle session
    can resume without repeating expensive DICOM decompression."""
    if prev_output_dir.exists():
        prev_files = list(prev_output_dir.glob("**/*"))
        print(f"Found previous output dir with {len(prev_files)} files. Copying forward...")
        copied = 0
        for f in prev_files:
            if f.is_file():
                dest = output_dir / f.name
                if not dest.exists():
                    import shutil
                    shutil.copy(f, dest)
                    copied += 1
        print(f"Copied {copied} new files into {output_dir}")
    else:
        print("No previous output found — starting fresh.")


# ============================================================
# CELL 4 — Segmentation-prioritized stratified patient subset
# ============================================================
def build_stratified_subset(train_csv, series_meta_csv, seg_dir, n_patients_subset=N_PATIENTS_SUBSET, seed=42):
    train_df = pd.read_csv(train_csv)
    series_df = pd.read_csv(series_meta_csv)

    train_df['patient_id'] = train_df['patient_id'].astype(int)
    series_df['patient_id'] = series_df['patient_id'].astype(int)
    series_df['series_id']  = series_df['series_id'].astype(int)

    injury_cols = ['bowel_injury', 'extravasation_injury',
                   'kidney_low', 'kidney_high',
                   'liver_low', 'liver_high',
                   'spleen_low', 'spleen_high']
    if 'any_injury' not in train_df.columns:
        train_df['any_injury'] = (train_df[injury_cols].sum(axis=1) > 0).astype(int)

    seg_files = list(seg_dir.glob("*.nii*"))
    seg_series_ids_int = set(int(f.stem.split('.')[0]) for f in seg_files)
    print(f"Segmentation files found: {len(seg_files)}")

    series_df['has_seg'] = series_df['series_id'].isin(seg_series_ids_int)
    patients_with_seg = set(series_df[series_df['has_seg']]['patient_id'])
    train_df['has_seg'] = train_df['patient_id'].isin(patients_with_seg)

    # Prioritize segmented + positive patients (highest value for training)
    positive_seg   = train_df[(train_df['any_injury'] == 1) & (train_df['has_seg'])]
    positive_noseg = train_df[(train_df['any_injury'] == 1) & (~train_df['has_seg'])]
    healthy_df     = train_df[train_df['any_injury'] == 0]

    n_seg_positive = len(positive_seg)
    remaining = max(0, n_patients_subset - n_seg_positive)
    n_noseg = min(len(positive_noseg), remaining // 2)
    n_healthy = min(len(healthy_df), remaining - n_noseg)

    subset_df = pd.concat([
        positive_seg,
        positive_noseg.sample(n=n_noseg, random_state=seed) if n_noseg > 0 else positive_noseg.iloc[0:0],
        healthy_df.sample(n=n_healthy, random_state=seed) if n_healthy > 0 else healthy_df.iloc[0:0],
    ]).reset_index(drop=True)

    print(f"Final subset: {len(subset_df)} patients "
          f"(seg+positive={n_seg_positive}, noseg+positive={n_noseg}, healthy={n_healthy})")

    subset_patients = set(subset_df['patient_id'])
    series_subset = series_df[series_df['patient_id'].isin(subset_patients)].reset_index(drop=True)

    overlap = set(series_subset['series_id']) & seg_series_ids_int
    print(f"Series in subset with available segmentation: {len(overlap)} / {len(series_subset)}")

    return subset_df, series_subset


# ============================================================
# CELL 5 — DICOM loading + windowing utilities
# ============================================================
def load_dicom_series(patient_id, series_id, images_dir=TRAIN_IMAGES_DIR):
    series_path = images_dir / str(patient_id) / str(series_id)
    dcm_files = sorted(series_path.glob("*.dcm"), key=lambda p: int(p.stem))
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found at {series_path}")
    slices, positions = [], []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        slices.append(ds)
        try:
            positions.append(float(ds.ImagePositionPatient[2]))
        except Exception:
            positions.append(int(ds.InstanceNumber))
    order = np.argsort(positions)
    return [slices[i] for i in order]


def apply_windowing(pixel_array, window_center, window_width, slope=1.0, intercept=0.0):
    img = pixel_array.astype(np.float32) * slope + intercept
    lower, upper = window_center - window_width / 2, window_center + window_width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower + 1e-6)
    return (img * 255).astype(np.uint8)


WC_SOFT, WW_SOFT = 50, 400   # soft-tissue window (liver/spleen/kidney/bowel)
WC_BLOOD, WW_BLOOD = 90, 350  # blood/contrast window (reserved for extravasation work)


def dicom_to_volume(slices, window_center=WC_SOFT, window_width=WW_SOFT):
    if len(slices) == 0:
        raise ValueError("Empty slice list passed to dicom_to_volume")
    vol = []
    for ds in slices:
        arr = ds.pixel_array
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        vol.append(apply_windowing(arr, window_center, window_width, slope, intercept))
    return np.stack(vol, axis=0)


def resample_depth(volume, target_depth):
    src_depth = volume.shape[0]
    if src_depth == target_depth:
        return volume
    idx = np.linspace(0, src_depth - 1, target_depth)
    idx = np.round(idx).astype(int)
    return volume[idx]


# ============================================================
# CELL 6/6b — Affine-based coordinate transform (DICOM <-> NIfTI)
# ============================================================
# NOTE: NIfTI segmentation masks and DICOM pixel arrays live in DIFFERENT
# coordinate systems (confirmed empirically: seg orientation LAS, DICOM LPS,
# and slice ordering directions may not match). Never assume index-to-index
# correspondence — always transform through real-world (mm) coordinates.

def seg_index_to_dicom_world(i, j, k, seg_affine):
    """Seg voxel index -> DICOM-LPS world coordinates (mm)."""
    world = seg_affine @ np.array([i, j, k, 1.0])
    Xw, Yw, Zw = world[0], world[1], world[2]
    # seg orientation is LAS; DICOM patient space is LPS -> flip A to P
    return Xw, -Yw, Zw


def dicom_world_to_index(world_x, world_y, world_z, dicom_slices, pixel_spacing):
    """DICOM-LPS world coordinates -> (slice_idx, row, col) in the sorted DICOM volume."""
    first_ipp = np.array(dicom_slices[0].ImagePositionPatient, dtype=float)
    col = (world_x - first_ipp[0]) / pixel_spacing[1]
    row = (world_y - first_ipp[1]) / pixel_spacing[0]
    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in dicom_slices])
    slice_idx = int(np.argmin(np.abs(z_positions - world_z)))
    return slice_idx, row, col


def seg_bbox_to_dicom_bbox(mins, maxs, seg_affine, dicom_slices, pixel_spacing):
    """Transform an axis-aligned seg-space bbox into an axis-aligned DICOM-space
    bbox by mapping all 8 corners through world space and taking min/max."""
    corners_seg = [
        (mins[0], mins[1], mins[2]), (mins[0], mins[1], maxs[2]),
        (mins[0], maxs[1], mins[2]), (mins[0], maxs[1], maxs[2]),
        (maxs[0], mins[1], mins[2]), (maxs[0], mins[1], maxs[2]),
        (maxs[0], maxs[1], mins[2]), (maxs[0], maxs[1], maxs[2]),
    ]
    dicom_coords = []
    for (i, j, k) in corners_seg:
        wx, wy, wz = seg_index_to_dicom_world(i, j, k, seg_affine)
        s_idx, r, c = dicom_world_to_index(wx, wy, wz, dicom_slices, pixel_spacing)
        dicom_coords.append((s_idx, r, c))
    dicom_coords = np.array(dicom_coords)
    return dicom_coords.min(axis=0), dicom_coords.max(axis=0)


# ============================================================
# CELL 7 — Verified organ labels + crop function
# ============================================================
# Verified empirically against actual RSNA-provided segmentation masks
# (custom 5-class scheme, NOT generic TotalSegmentator numbering):
ORGAN_LABELS = {
    "liver":  [1],
    "spleen": [2],
    "kidney": [3, 4],
    "bowel":  [5],
}

# Organ-specific margins: small organs need a bigger relative margin so
# resizing to TARGET_SIZE doesn't blow up a tiny raw crop into blurry noise.
ORGAN_MARGINS = {
    "liver":  10,
    "spleen": 25,
    "kidney": 20,
    "bowel":  10,
}


def get_organ_bbox(seg_array, label_ids):
    mask = np.isin(seg_array, label_ids)
    if not mask.any():
        return None
    coords = np.array(np.where(mask))
    return coords.min(axis=1), coords.max(axis=1)


def crop_and_resize_organ(volume, seg_array, seg_affine, dicom_slices, pixel_spacing,
                           organ_name, target_size=TARGET_SIZE, n_slices=N_SLICES_PER_ORGAN):
    mask = np.isin(seg_array, ORGAN_LABELS[organ_name])
    if not mask.any():
        return None
    coords = np.array(np.where(mask))
    mins, maxs = coords.min(axis=1), coords.max(axis=1)

    d_mins, d_maxs = seg_bbox_to_dicom_bbox(mins, maxs, seg_affine, dicom_slices, pixel_spacing)
    margin = ORGAN_MARGINS.get(organ_name, 10)

    z0, z1 = int(min(d_mins[0], d_maxs[0])) - margin, int(max(d_mins[0], d_maxs[0])) + margin
    r0, r1 = int(min(d_mins[1], d_maxs[1])) - margin, int(max(d_mins[1], d_maxs[1])) + margin
    c0, c1 = int(min(d_mins[2], d_maxs[2])) - margin, int(max(d_mins[2], d_maxs[2])) + margin

    z0, r0, c0 = max(z0, 0), max(r0, 0), max(c0, 0)
    z1 = min(z1, volume.shape[0]); r1 = min(r1, volume.shape[1]); c1 = min(c1, volume.shape[2])

    if z1 <= z0 or r1 <= r0 or c1 <= c0:
        return None

    cropped = volume[z0:z1, r0:r1, c0:c1]
    if cropped.size == 0:
        return None

    raw_r, raw_c = cropped.shape[1], cropped.shape[2]
    resize_factor = max(target_size / raw_r, target_size / raw_c)

    resized = np.stack([cv2.resize(sl, (target_size, target_size)) for sl in cropped])
    return resample_depth(resized, n_slices), resize_factor


# ============================================================
# CELL 8 — Main preprocessing loop (checkpointed, time-budgeted, resumable)
# ============================================================
def run_preprocessing(subset_df, series_subset, output_dir=OUTPUT_DIR,
                       save_every=50, time_budget_seconds=8 * 3600):
    manifest_path = output_dir / "manifest.csv"
    failed_path = output_dir / "failed_log.csv"
    start_time = time.time()

    if manifest_path.exists():
        prev_manifest = pd.read_csv(manifest_path)
        manifest = prev_manifest.to_dict('records')
        already_done = set(zip(prev_manifest['patient_id'].astype(str), prev_manifest['series_id'].astype(str)))
        print(f"Resuming: {len(already_done)} series already processed.")
    else:
        manifest, already_done = [], set()

    if failed_path.exists():
        prev_failed = pd.read_csv(failed_path)
        failed_log = prev_failed.to_dict('records')
        already_failed = set(zip(prev_failed['patient_id'].astype(str), prev_failed['series_id'].astype(str)))
    else:
        failed_log, already_failed = [], set()

    skip_set = already_done | already_failed
    new_count = 0
    stopped_early = False

    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        if time.time() - start_time > time_budget_seconds:
            print(f"\nTime budget reached — stopping gracefully to save progress.")
            stopped_early = True
            break

        patient_id = str(int(row['patient_id']))
        patient_series = series_subset[series_subset['patient_id'] == int(row['patient_id'])]

        for _, srow in patient_series.iterrows():
            series_id = str(int(srow['series_id']))
            if (patient_id, series_id) in skip_set:
                continue

            try:
                dicom_slices = load_dicom_series(patient_id, series_id)
                vol_soft_local = dicom_to_volume(dicom_slices, WC_SOFT, WW_SOFT)
                pixel_spacing_local = [float(x) for x in dicom_slices[0].PixelSpacing]

                seg_path = SEG_DIR / f"{series_id}.nii"  # NOTE: .nii, not .nii.gz
                organ_crops, resize_factors = {}, {}
                if seg_path.exists():
                    seg_img_local = nib.load(seg_path)
                    seg_arr_local = seg_img_local.get_fdata()
                    for organ in ORGAN_LABELS:
                        result = crop_and_resize_organ(
                            vol_soft_local, seg_arr_local, seg_img_local.affine,
                            dicom_slices, pixel_spacing_local, organ
                        )
                        if result is not None:
                            crop, resize_factor = result
                            organ_crops[organ] = crop
                            resize_factors[organ] = resize_factor

                # Skip full_volume if all 4 organ crops already obtained (disk-space
                # optimization). NOTE: this later required a backfill step — see below.
                save_full_volume = len(organ_crops) < len(ORGAN_LABELS)
                npz_payload = dict(organ_crops)
                if save_full_volume:
                    full_resized = np.stack([cv2.resize(sl, (TARGET_SIZE, TARGET_SIZE)) for sl in vol_soft_local])
                    full_resampled = resample_depth(full_resized, 64)
                    npz_payload["full_volume"] = full_resampled

                out_path = output_dir / f"{patient_id}_{series_id}.npz"
                np.savez_compressed(out_path, **npz_payload)

                manifest.append({
                    "patient_id": patient_id, "series_id": series_id,
                    "has_segmentation": seg_path.exists(),
                    "organs_cropped": list(organ_crops.keys()),
                    "resize_factors": resize_factors,
                    "has_full_volume": save_full_volume,
                    "output_path": str(out_path),
                })
                new_count += 1
            except Exception as e:
                failed_log.append({"patient_id": patient_id, "series_id": series_id, "error": str(e)})
                new_count += 1

            if new_count > 0 and new_count % save_every == 0:
                pd.DataFrame(manifest).to_csv(manifest_path, index=False)
                pd.DataFrame(failed_log).to_csv(failed_path, index=False)

        if stopped_early:
            break

    manifest_df = pd.DataFrame(manifest)
    failed_df = pd.DataFrame(failed_log)
    manifest_df.to_csv(manifest_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    print(f"\nTotal processed this run: {new_count}")
    print(f"Total in manifest: {len(manifest_df)} | Total failed: {len(failed_df)}")
    return manifest_df, failed_df


# ============================================================
# BACKFILL — regenerate full_volume for series that were skipped
# because they already had all 4 organ crops (fixes the stacking
# overlap gap discovered in Phase 2: organ-crop and full-volume
# patient populations were almost completely disjoint by construction).
# ============================================================
def run_full_volume_backfill(manifest_df, output_dir: Path, log_path: Path):
    """Writes to a NEW directory (never back into a read-only mounted input dataset)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    organ_crop_series = manifest_df[manifest_df['organs_cropped'].apply(len) == 4].reset_index(drop=True)
    print(f"Series needing full_volume backfill: {len(organ_crop_series)}")

    if log_path.exists():
        backfill_log = pd.read_csv(log_path)
        done_paths = set(backfill_log['output_path'])
    else:
        backfill_log = pd.DataFrame(columns=['output_path', 'new_path', 'status'])
        done_paths = set()

    records = backfill_log.to_dict('records')

    for _, row in tqdm(organ_crop_series.iterrows(), total=len(organ_crop_series)):
        if row['output_path'] in done_paths:
            continue

        patient_id, series_id = str(row['patient_id']), str(row['series_id'])
        new_path = output_dir / Path(row['output_path']).name

        try:
            dicom_slices = load_dicom_series(patient_id, series_id)
            vol_soft = dicom_to_volume(dicom_slices, WC_SOFT, WW_SOFT)
            full_resized = np.stack([cv2.resize(sl, (TARGET_SIZE, TARGET_SIZE)) for sl in vol_soft])
            full_resampled = resample_depth(full_resized, 64)

            existing = np.load(row['output_path'])
            payload = {k: existing[k] for k in existing.files}
            payload['full_volume'] = full_resampled

            np.savez_compressed(new_path, **payload)
            records.append({'output_path': row['output_path'], 'new_path': str(new_path), 'status': 'success'})
        except Exception as e:
            print(f"Failed backfill for {patient_id}/{series_id}: {e}")
            records.append({'output_path': row['output_path'], 'new_path': None, 'status': f'failed: {e}'})

        pd.DataFrame(records).to_csv(log_path, index=False)

    backfill_log = pd.DataFrame(records)
    print(f"\nTotal: {len(backfill_log)}")
    print(backfill_log['status'].apply(lambda s: 'success' if s == 'success' else 'failed').value_counts())
    return backfill_log


# ============================================================
# Sanity check plot — run after a small dry-run before scaling up
# ============================================================
def sanity_check_plot(manifest_df):
    seg_rows = manifest_df[manifest_df['organs_cropped'].apply(len) > 0]
    if len(seg_rows) == 0:
        print("No organ crops present yet.")
        return
    sample = seg_rows.iloc[0]
    data = np.load(sample['output_path'])
    print("Patient:", sample['patient_id'], "Series:", sample['series_id'])
    print("Keys:", list(data.keys()))

    fig, axes = plt.subplots(1, len(data.keys()), figsize=(5 * len(data.keys()), 5))
    if len(data.keys()) == 1:
        axes = [axes]
    for ax, key in zip(axes, data.keys()):
        mid_slice = data[key][data[key].shape[0] // 2]
        ax.imshow(mid_slice, cmap='gray')
        ax.set_title(key)
    plt.show()


if __name__ == "__main__":
    # Example run order (Kaggle Save Version mode):
    # resume_from_previous_version(Path("/kaggle/input/<prev-version>"), OUTPUT_DIR)
    subset_df, series_subset = build_stratified_subset(TRAIN_CSV, SERIES_META_CSV, SEG_DIR)
    manifest_df, failed_df = run_preprocessing(subset_df, series_subset)
    sanity_check_plot(manifest_df)
