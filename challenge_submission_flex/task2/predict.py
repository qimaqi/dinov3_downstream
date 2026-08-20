#!/usr/bin/env python3
"""FOMO26 Task 2: Meningioma Segmentation (Binary).

Segments meningiomas in brain MRI using nnU-Net with FlexiCT backbone
(Primus_Onescale, linear decoder, frozen backbone).
Output: NIfTI binary mask (0=background, 1=meningioma).

Supports fold ensembling: place multiple fold checkpoints and they are
automatically averaged during prediction.

Container layout:
    /app/predict.py
    /app/dinov3_downstream/flexi_ct/
    /app/dinov3_downstream/downstream/segmentation/nnUNet/
    /app/weights/
        2D_final_model.pth          <- pretrained FlexiCT 2D backbone
        nnunet_model/               <- nnU-Net trained model folder
            plans.json
            dataset.json
            dataset_fingerprint.json
            fold_0/checkpoint_final.pth
            fold_1/checkpoint_final.pth
            ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# ─── Paths inside the container ───────────────────────────────────────────────
APP_DIR = Path("/app")
NNUNET_SOURCE = APP_DIR / "dinov3_downstream" / "downstream" / "segmentation" / "nnUNet"
REPO_ROOT = APP_DIR / "dinov3_downstream"

sys.path.insert(0, str(NNUNET_SOURCE))
sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/flexict_mplconfig"

# ============================================================================
# MODEL WEIGHT PATHS — edit these for your setup
# ============================================================================

# Pretrained FlexiCT backbone
FLEXICT_2D_CHECKPOINT_PATH = APP_DIR / "weights" / "2D_final_model.pth"

# nnU-Net model folder (contains plans.json, dataset.json, fold_X/)
MODEL_FOLDER = APP_DIR / "weights" / "nnunet_model"

# Which folds to ensemble — "all" uses all available folds in the model folder
# Or specify a tuple like (0, 1, 2) for specific folds
USE_FOLDS = "all"

# ============================================================================

os.environ["FLEXICT_2D_CHECKPOINT"] = str(FLEXICT_2D_CHECKPOINT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 2: Meningioma Segmentation")
    parser.add_argument("--flair", type=str, required=True, help="Path to T2 FLAIR image")
    parser.add_argument("--dwi", type=str, help="Path to DWI image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")
    parser.add_argument("--output", type=str, required=True, help="Path to save segmentation NIfTI file")
    return parser.parse_args()


def prepare_nnunet_input(args: argparse.Namespace, input_dir: Path) -> nib.Nifti1Image:
    """Copy and rename input modalities to nnU-Net format (_0000, _0001, ...).

    The channel ordering must match what was used during training preprocessing.
    """
    modalities = [
        ("flair", args.flair),
        ("dwi", args.dwi),
        ("t2s", args.t2s),
        ("swi", args.swi),
    ]

    ref_img = None
    channel_idx = 0
    for name, path in modalities:
        if path is None:
            continue
        img = nib.load(path)
        if ref_img is None:
            ref_img = img
        dest = input_dir / f"case_0000_{channel_idx:04d}.nii.gz"
        nib.save(img, str(dest))
        channel_idx += 1

    if ref_img is None:
        raise ValueError("At least --flair must be provided.")

    return ref_img


def get_available_folds(model_folder: Path) -> tuple:
    """Detect which fold checkpoints are available."""
    folds = []
    for fold_dir in sorted(model_folder.glob("fold_*")):
        if (fold_dir / "checkpoint_final.pth").exists():
            fold_num = int(fold_dir.name.split("_")[1])
            folds.append(fold_num)
    return tuple(folds) if folds else (0,)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Prepare inputs in nnU-Net format
        ref_img = prepare_nnunet_input(args, input_dir)

        # Determine which folds to use
        if USE_FOLDS == "all":
            folds = get_available_folds(MODEL_FOLDER)
        else:
            folds = USE_FOLDS

        print(f"Using folds: {folds} (ensemble of {len(folds)} model(s))")

        # Run nnU-Net prediction with FlexiCT trainer
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(MODEL_FOLDER),
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )
        predictor.predict_from_files(
            [[str(f) for f in sorted(input_dir.glob("*.nii.gz"))]],
            str(output_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        # Find and copy output
        pred_files = list(output_dir.glob("*.nii.gz"))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if pred_files:
            shutil.copy2(str(pred_files[0]), str(output_path))
        else:
            # Fallback: create empty mask with same geometry as input
            print("WARNING: No prediction file found, creating empty mask.")
            mask = np.zeros(ref_img.shape[:3], dtype=np.uint8)
            out_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
            nib.save(out_img, str(output_path))

    print(f"Segmentation saved to {args.output}")


if __name__ == "__main__":
    main()
