#!/usr/bin/env python3
"""FOMO26 Task 6/7: extract frozen DINOv3 patch_cls embeddings."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_CONTAINER_ROOT = Path("/app")

if (_THIS_DIR / "../../flexi_ct").resolve().is_dir():
    REPO_ROOT = (_THIS_DIR / "../..").resolve()
    WEIGHTS_DIR = _THIS_DIR / "weights"
elif (_CONTAINER_ROOT / "dinov3_downstream" / "flexi_ct").is_dir():
    REPO_ROOT = _CONTAINER_ROOT / "dinov3_downstream"
    WEIGHTS_DIR = _CONTAINER_ROOT / "weights"
else:
    raise RuntimeError("Cannot resolve project root. Run from within the repo or inside the container.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKBONE_CHECKPOINT_PATH = Path(
    os.environ.get(
        "DINOV3_2D_CHECKPOINT",
        WEIGHTS_DIR / "2D_final_model.pth",
    )
)
os.environ["DINOV3_2D_CHECKPOINT"] = str(BACKBONE_CHECKPOINT_PATH)

TRAIN_SCRIPT = REPO_ROOT / "downstream" / "3d_classify" / "dinov3_finetune_cls_from_slices.py"


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TRAIN_MODULE = _load_module(TRAIN_SCRIPT, "dinov3_finetune_cls_from_slices")
FlexiCTSliceVolumeClassifier = _TRAIN_MODULE.FlexiCTSliceVolumeClassifier
_parse_hw_resize = _TRAIN_MODULE._parse_hw_resize
_parse_spacing_xy = _TRAIN_MODULE._parse_spacing_xy
normalize_volume_chwd = _TRAIN_MODULE.normalize_volume_chwd
center_crop_or_pad_volume_chwd = _TRAIN_MODULE.center_crop_or_pad_volume_chwd
respace_xy_volume_chwd = _TRAIN_MODULE.respace_xy_volume_chwd
resize_volume_chwd = _TRAIN_MODULE.resize_volume_chwd

DEFAULT_SLICE_POOL = os.environ.get("TASK6_SLICE_POOL", "patch_cls")
DEFAULT_MODALITY_POOL = os.environ.get("TASK6_MODALITY_POOL", "each")
DEFAULT_TARGET_SPACING_XY = os.environ.get("TASK6_TARGET_SPACING_XY", "1.0,1.0")
DEFAULT_PAD_HW = os.environ.get("TASK6_PAD_HW", "256,256")
DEFAULT_RESIZE_HW = os.environ.get("TASK6_RESIZE_HW", "none")
DEFAULT_SLICE_BATCH_SIZE = int(os.environ.get("TASK6_SLICE_BATCH_SIZE", "32"))
DEFAULT_PATCH_SIZE = int(os.environ.get("TASK6_PATCH_SIZE", "16"))
DEFAULT_TRANSFORMER_DEPTH = int(os.environ.get("TASK6_TRANSFORMER_DEPTH", "2"))
DEFAULT_TRANSFORMER_HEADS = int(os.environ.get("TASK6_TRANSFORMER_HEADS", "8"))
DEFAULT_MRI_NORMALIZATION = os.environ.get("TASK6_MRI_NORMALIZATION", "robust_zscore")
DEFAULT_MRI_LOW_PERCENTILE = float(os.environ.get("TASK6_MRI_LOW_PERCENTILE", "0.5"))
DEFAULT_MRI_HIGH_PERCENTILE = float(os.environ.get("TASK6_MRI_HIGH_PERCENTILE", "99.5"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen DINOv3 patch_cls embedding")
    parser.add_argument("--input", type=str, help="Path to a single 3D MR NIfTI file")
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, help="Path to DWI image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image")
    parser.add_argument("--swi", type=str, help="Path to SWI image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .npy file")
    return parser.parse_args()


def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    zooms = nii.header.get_zooms()
    spacing_xy = None
    if len(zooms) >= 2:
        spacing_xy = (float(zooms[0]), float(zooms[1]))
    return data, spacing_xy


def collect_modalities(args: argparse.Namespace) -> tuple[torch.Tensor, list[tuple[float, float] | None]]:
    if args.input:
        arr, spacing_xy = load_nifti(args.input)
        return torch.from_numpy(arr[None, ...]), [spacing_xy]

    modality_paths = [
        ("flair", args.flair),
        ("adc", args.adc),
        ("dwi", args.dwi),
        ("t2s", args.t2s),
        ("swi", args.swi),
    ]

    arrays = []
    spacings_xy = []
    reference_shape = None
    for name, path in modality_paths:
        if path is None:
            continue
        arr, spacing_xy = load_nifti(path)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(f"Shape mismatch: expected {reference_shape}, got {arr.shape} for modality '{name}'")
        arrays.append(arr)
        spacings_xy.append(spacing_xy)

    if not arrays:
        raise ValueError("Provide either --input or at least one modality path.")

    return torch.from_numpy(np.stack(arrays, axis=0)), spacings_xy


def preprocess_volume(volume: torch.Tensor, spacings_xy: list[tuple[float, float] | None]) -> torch.Tensor:
    resize_hw = _parse_hw_resize(DEFAULT_RESIZE_HW)
    target_spacing_xy = _parse_spacing_xy(DEFAULT_TARGET_SPACING_XY)
    pad_hw = _parse_hw_resize(DEFAULT_PAD_HW)

    processed_channels = []
    for channel_idx in range(volume.shape[0]):
        channel = volume[channel_idx : channel_idx + 1].clone()
        channel = respace_xy_volume_chwd(channel, spacings_xy[channel_idx], target_spacing_xy)
        channel = center_crop_or_pad_volume_chwd(channel, pad_hw)
        channel = resize_volume_chwd(channel, resize_hw)
        channel = normalize_volume_chwd(
            channel,
            method=DEFAULT_MRI_NORMALIZATION,
            low_percentile=DEFAULT_MRI_LOW_PERCENTILE,
            high_percentile=DEFAULT_MRI_HIGH_PERCENTILE,
        )
        processed_channels.append(channel)
    return torch.cat(processed_channels, dim=0)


def build_model(device: torch.device) -> torch.nn.Module:
    if not BACKBONE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing DINOv3 backbone checkpoint: {BACKBONE_CHECKPOINT_PATH}")

    model = FlexiCTSliceVolumeClassifier(
        dinov3_checkpoint=str(BACKBONE_CHECKPOINT_PATH),
        num_classes=2,
        slice_pool=DEFAULT_SLICE_POOL,
        modality_pool=DEFAULT_MODALITY_POOL,
        slice_axis=-1,
        slice_size=None,
        patch_size=DEFAULT_PATCH_SIZE,
        max_slices=None,
        encoder_tuning="frozen",
        lora_r=16,
        lora_alpha=16,
        lora_targets=["qkv", "proj"],
        lora_dropout=0.0,
        transformer_depth=DEFAULT_TRANSFORMER_DEPTH,
        transformer_heads=DEFAULT_TRANSFORMER_HEADS,
        device=device,
    ).to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    volume, spacings_xy = collect_modalities(args)
    volume = preprocess_volume(volume, spacings_xy).unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model(device)
    with torch.no_grad():
        embedding = model.forward_features(
            volume=volume,
            slice_batch_size=DEFAULT_SLICE_BATCH_SIZE,
        )

    embedding_np = embedding[0].detach().cpu().numpy().astype(np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embedding_np)
    print(f"Embedding shape: {embedding_np.shape}, saved to {args.output}")


if __name__ == "__main__":
    main()
