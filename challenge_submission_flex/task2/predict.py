#!/usr/bin/env python3
"""FOMO26 Task 2 submission for FLAIR-only meningioma segmentation."""
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

_THIS_DIR = Path(__file__).resolve().parent
_CONTAINER_ROOT = Path("/app")

if (_THIS_DIR / "nnUNet").is_dir():
    APP_ROOT = _THIS_DIR
    WEIGHTS_DIR = _THIS_DIR / "weights"
elif (_CONTAINER_ROOT / "nnUNet").is_dir():
    APP_ROOT = _CONTAINER_ROOT
    WEIGHTS_DIR = _CONTAINER_ROOT / "weights"
else:
    raise RuntimeError("Could not resolve task2 package root.")

NNUNET_ROOT = APP_ROOT / "nnUNet"
if str(NNUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_ROOT))

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/meddinov3_mplconfig"
os.environ.setdefault("nnUNet_compile", "false")

DINOV3_PRETRAINED_CHECKPOINT = WEIGHTS_DIR / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
MODEL_FOLDER = WEIGHTS_DIR / "nnunet_model"
CHECKPOINT_NAME = "checkpoint_best.pth"
USE_FOLDS = (0,)

os.environ["DINOV3_2D_CHECKPOINT"] = str(DINOV3_PRETRAINED_CHECKPOINT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLAIR-only meningioma segmentation")
    parser.add_argument("--flair", required=True, help="Path to the FLAIR NIfTI volume")
    parser.add_argument("--output", required=True, help="Path to the output segmentation NIfTI file")
    return parser.parse_args()


def _assert_required_files() -> None:
    required = [
        DINOV3_PRETRAINED_CHECKPOINT,
        MODEL_FOLDER / "dataset.json",
        MODEL_FOLDER / "plans.json",
        MODEL_FOLDER / "fold_0" / CHECKPOINT_NAME,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required task2 assets:\n" + "\n".join(missing))


def _prepare_case(flair_path: Path, input_dir: Path) -> nib.Nifti1Image:
    image = nib.load(str(flair_path))
    nib.save(image, str(input_dir / "case_0000_0000.nii.gz"))
    return image


def _copy_prediction(output_dir: Path, destination: Path, reference: nib.Nifti1Image) -> None:
    prediction_files = sorted(output_dir.glob("*.nii.gz"))
    destination.parent.mkdir(parents=True, exist_ok=True)

    if prediction_files:
        shutil.copy2(prediction_files[0], destination)
        return

    mask = np.zeros(reference.shape[:3], dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, reference.affine, reference.header), str(destination))


def main() -> None:
    args = parse_args()
    _assert_required_files()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix="task2_predict_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_dir = tmpdir_path / "input"
        output_dir = tmpdir_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        reference = _prepare_case(Path(args.flair), input_dir)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(MODEL_FOLDER),
            use_folds=USE_FOLDS,
            checkpoint_name=CHECKPOINT_NAME,
        )
        predictor.predict_from_files(
            [[str(input_dir / "case_0000_0000.nii.gz")]],
            [str(output_dir / "case_0000")],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        _copy_prediction(output_dir, Path(args.output), reference)


if __name__ == "__main__":
    main()
