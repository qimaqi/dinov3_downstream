#!/usr/bin/env python3
"""FOMO26 Task 3: Brain Age Estimation (Regression).

Predicts brain age from T1-weighted MRI using the DINOv3 2D slice encoder with
the 3D regression head trained in `downstream/3d_regression/
dinov3_fintune_reg_from_slices.py`.

Supports model ensembling by averaging predictions from the selected top folds.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_CONTAINER_ROOT = Path("/app")

if (_THIS_DIR / "../../flexi_ct").resolve().is_dir():
    REPO_ROOT = (_THIS_DIR / "../..").resolve()
    WEIGHTS_DIR = _THIS_DIR / "weights"
elif _CONTAINER_ROOT.is_dir() and (_CONTAINER_ROOT / "flexi_ct").is_dir():
    REPO_ROOT = _CONTAINER_ROOT
    WEIGHTS_DIR = _CONTAINER_ROOT / "weights"
else:
    raise RuntimeError(
        "Cannot resolve project root. Run from within the repo or inside the container."
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "3d_regression"
    / "REGR002_FOMO26_BrainAge"
)

_selected_folds_env = os.environ.get("TASK3_SELECTED_FOLDS")
if _selected_folds_env:
    SELECTED_FOLDS = tuple(
        fold.strip() for fold in _selected_folds_env.split(",") if fold.strip()
    )
else:
    SELECTED_FOLDS = ("fold_4", "fold_1", "fold_0")


def _resolve_backbone_checkpoint() -> Path:
    env_path = os.environ.get("DINOV3_2D_CHECKPOINT")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    candidates = [
        WEIGHTS_DIR / "2D_final_model.pth",
        REPO_ROOT
        / "ckpts"
        / "pretrain_fomo_10k_pretrained_dinov3_dino_base_g8_e400_p16_mri_normalize"
        / "2D_final_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not resolve the DINOv3 2D backbone checkpoint. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nSet DINOV3_2D_CHECKPOINT to a valid checkpoint path if needed."
    )


BACKBONE_CHECKPOINT_PATH = _resolve_backbone_checkpoint()
os.environ["DINOV3_2D_CHECKPOINT"] = str(BACKBONE_CHECKPOINT_PATH)

_TRAIN_SCRIPT = (
    REPO_ROOT / "downstream" / "3d_regression" / "dinov3_fintune_reg_from_slices.py"
)


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TRAIN_MODULE = _load_module(_TRAIN_SCRIPT, "dinov3_fintune_reg_from_slices")
DINOv3SliceVolumeRegressor = _TRAIN_MODULE.DINOv3SliceVolumeRegressor
center_crop_or_pad_volume_chwd = _TRAIN_MODULE.cls_mod.center_crop_or_pad_volume_chwd
normalize_volume_chwd = _TRAIN_MODULE.cls_mod.normalize_volume_chwd
respace_xy_volume_chwd = _TRAIN_MODULE.cls_mod.respace_xy_volume_chwd
resize_volume_chwd = _TRAIN_MODULE.cls_mod.resize_volume_chwd
_parse_hw_resize = _TRAIN_MODULE.cls_mod._parse_hw_resize
_parse_spacing_xy = _TRAIN_MODULE.cls_mod._parse_spacing_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 3: Brain Age Estimation (Regression)"
    )
    parser.add_argument("--t1", type=str, required=True, help="Path to T1-weighted image")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output .txt file"
    )
    return parser.parse_args()


def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    import nibabel as nib

    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    zooms = nii.header.get_zooms()
    spacing_xy = None
    if len(zooms) >= 2:
        spacing_xy = (float(zooms[0]), float(zooms[1]))
    return data, spacing_xy


def preprocess_volume(
    volume: torch.Tensor,
    spacing_xy: tuple[float, float] | None,
    saved_args: dict,
) -> torch.Tensor:
    resize_hw = _parse_hw_resize(saved_args.get("resize_hw"))
    target_spacing_xy = _parse_spacing_xy(saved_args.get("target_spacing_xy", "1.0,1.0"))
    if target_spacing_xy is None:
        target_spacing_xy = (1.0, 1.0)
    pad_hw = _parse_hw_resize(saved_args.get("pad_hw", "256,256"))
    if pad_hw is None:
        pad_hw = (256, 256)

    mri_normalization = saved_args.get("mri_normalization", "robust_zscore")
    mri_low_percentile = float(saved_args.get("mri_low_percentile", 0.5))
    mri_high_percentile = float(saved_args.get("mri_high_percentile", 99.5))

    channel = respace_xy_volume_chwd(volume.clone(), spacing_xy, target_spacing_xy)
    channel = center_crop_or_pad_volume_chwd(channel, pad_hw)
    channel = resize_volume_chwd(channel, resize_hw)
    channel = normalize_volume_chwd(
        channel,
        method=mri_normalization,
        low_percentile=mri_low_percentile,
        high_percentile=mri_high_percentile,
    )
    return channel


def build_model(ckpt: dict, device: torch.device):
    saved_args = ckpt["args"]
    num_outputs = int(saved_args.get("num_outputs", 1))

    model = DINOv3SliceVolumeRegressor(
        dinov3_checkpoint=str(BACKBONE_CHECKPOINT_PATH),
        num_outputs=num_outputs,
        slice_pool=saved_args["slice_pool"],
        modality_pool=saved_args["modality_pool"],
        slice_axis=saved_args["slice_axis"],
        slice_size=saved_args["slice_size"],
        patch_size=saved_args["patch_size"],
        max_slices=saved_args["max_slices"],
        encoder_tuning="frozen",
        lora_r=saved_args.get("lora_r", 16),
        lora_alpha=saved_args.get("lora_alpha", 16),
        lora_targets=[
            s.strip()
            for s in saved_args.get("lora_targets", "qkv,proj").split(",")
            if s.strip()
        ],
        lora_dropout=saved_args.get("lora_dropout", 0.0),
        transformer_depth=saved_args.get("transformer_depth", 2),
        transformer_heads=saved_args.get("transformer_heads", 8),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved_args


def _candidate_model_roots() -> list[Path]:
    env_root = os.environ.get("TASK3_MODEL_ROOT")
    roots = []
    if env_root:
        roots.append(Path(env_root))
    roots.extend([WEIGHTS_DIR, DEFAULT_RESULTS_DIR])

    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def get_model_paths() -> list[Path]:
    roots = _candidate_model_roots()
    attempted = []
    for root in roots:
        model_paths = [root / fold / "best.pt" for fold in SELECTED_FOLDS]
        attempted.extend(model_paths)
        if all(path.exists() for path in model_paths):
            print(f"Using model root: {root}")
            return model_paths

    raise FileNotFoundError(
        "Could not find all selected Task 3 ensemble checkpoints.\n"
        "Expected these files:\n"
        + "\n".join(f"  - {p}" for p in attempted)
    )


def predict_single_model(
    model_path: Path,
    volume: torch.Tensor,
    device: torch.device,
) -> float:
    ckpt = torch.load(str(model_path), map_location="cpu")
    model, saved_args = build_model(ckpt, device)

    vol = volume.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = model(volume=vol, slice_batch_size=int(saved_args.get("slice_batch_size", 32)))
    return float(pred[0].item())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = get_model_paths()
    print(f"Backbone checkpoint: {BACKBONE_CHECKPOINT_PATH}")
    print(f"Ensemble size: {len(model_paths)} model(s)")

    t1_data, spacing_xy = load_nifti(args.t1)
    volume = torch.from_numpy(t1_data[np.newaxis, ...])

    predictions = []
    for i, model_path in enumerate(model_paths):
        ckpt = torch.load(str(model_path), map_location="cpu")
        processed_volume = preprocess_volume(volume, spacing_xy, ckpt["args"])
        pred = predict_single_model(model_path, processed_volume, device)
        predictions.append(pred)
        print(f"  Model [{i}] age: {pred:.4f}")

    ensemble_prediction = sum(predictions) / len(predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{ensemble_prediction:.2f}\n")

    print(
        f"\nFinal ensemble predicted brain age: {ensemble_prediction:.2f} "
        f"(from {len(model_paths)} model(s))"
    )


if __name__ == "__main__":
    main()
