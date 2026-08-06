#!/usr/bin/env python3
"""Extract FlexiCT 2D slice features from a NIfTI volume and save PCA figures.

This script mirrors the relevant logic from inference_demo.ipynb, but:
- normalizes the full volume first using volume-wise z-score by default
- samples multiple slices across the volume
- supports even sampling or GT-guided sampling from nnUNet labels
- saves a 3-panel figure per slice: input image, PCA 0-2 RGB, PCA 3-5 RGB
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from flexi_ct import Flexi_CT_2D

DEFAULT_CHECKPOINT = (
    "/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/"
    "flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri/"
    "2D_final_model.pth"
)

DEFAULT_INPUT_NII = (
    "/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/"
    "Dataset069_FOMO26_Meningioma_FLAIR/imagesTr/sub-01_0000.nii.gz"
)
DEFAULT_LABEL_ROOT = (
    "/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/"
    "Dataset069_FOMO26_Meningioma_FLAIR/labelsTr"
)
DEFAULT_OUTPUT_DIR = "outputs/nii_gz_to_slice_feautre_sub-01_fomo10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input_nii", default=DEFAULT_INPUT_NII)
    parser.add_argument("--label_root", default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slice_size", type=int, default=320, help="Square slice size, divisible by patch size.")
    parser.add_argument("--patch_size", type=int, default=8, help="Runtime patch size for the 2D backbone.")
    parser.add_argument("--slice_axis", type=int, default=0, choices=[0, 1, 2], help="Axis in the loaded NumPy volume.")
    parser.add_argument("--num_slices", type=int, default=10, help="Number of slices to sample.")
    parser.add_argument("--slice_method", default="even", choices=["even", "gt"], help="How to choose slices.")
    parser.add_argument("--gt_context", type=int, default=4, help="For gt mode, also include this many slices before and after positive GT slices.")
    parser.add_argument(
        "--norm_mode",
        default="zscore",
        choices=["zscore", "hu"],
        help="Default zscore is volume-wise. HU mode follows the notebook CT demo.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def hu_normalize(x: np.ndarray, lo: float = -1000.0, hi: float = 1000.0) -> np.ndarray:
    x = np.clip(x.astype(np.float32), lo, hi)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def volume_zscore_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    values = x[np.isfinite(x) & (x != 0)]
    if values.size == 0:
        values = x[np.isfinite(x)]
    if values.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mean = float(values.mean())
    std = max(float(values.std()), eps)
    return np.nan_to_num((x - mean) / std).astype(np.float32)


def normalize_volume(volume: np.ndarray, norm_mode: str) -> np.ndarray:
    if norm_mode == "hu":
        return hu_normalize(volume)
    return volume_zscore_normalize(volume)


def center_crop_or_resize_2d(x: np.ndarray, target_size: int) -> np.ndarray:
    h, w = x.shape
    if h >= target_size and w >= target_size:
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        return x[top : top + target_size, left : left + target_size].astype(np.float32, copy=False)

    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    x = x[top : top + crop_size, left : left + crop_size].astype(np.float32, copy=False)

    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    x_t = F.interpolate(x_t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x_t.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)


def sampled_slice_indices(n_slices: int, count: int) -> list[int]:
    if n_slices <= 0:
        raise ValueError("Volume has no slices on the selected axis.")
    count = max(1, min(count, n_slices))
    return np.linspace(0, n_slices - 1, num=count, dtype=int).tolist()


def evenly_sample_candidates(candidates: list[int], count: int) -> list[int]:
    if not candidates:
        raise ValueError("No candidate slices available for sampling.")
    count = max(1, min(count, len(candidates)))
    picked = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    return [candidates[idx] for idx in picked.tolist()]


def label_path_for_image(input_nii: str | Path, label_root: str | Path) -> Path:
    input_name = Path(input_nii).name
    if not input_name.endswith('.nii.gz'):
        raise ValueError(f"Expected .nii.gz input, got {input_name}")
    label_name = input_name.replace('_0000.nii.gz', '.nii.gz')
    label_path = Path(label_root) / label_name
    if not label_path.exists():
        raise FileNotFoundError(f"Could not find GT label for {input_name} at {label_path}")
    return label_path


def gt_slice_candidates(label_volume: np.ndarray, axis: int, context: int) -> list[int]:
    positive = []
    for idx in range(label_volume.shape[axis]):
        label_slice = np.take(label_volume, indices=idx, axis=axis)
        if np.any(label_slice >= 1):
            positive.append(idx)
    if not positive:
        raise ValueError("No slices with GT segmentation >= 1 were found.")

    expanded = set()
    axis_len = label_volume.shape[axis]
    for idx in positive:
        start = max(0, idx - max(0, context))
        end = min(axis_len - 1, idx + max(0, context))
        for j in range(start, end + 1):
            expanded.add(j)
    return sorted(expanded)


def select_slice_indices(args: argparse.Namespace, volume: np.ndarray) -> list[int]:
    axis_len = volume.shape[args.slice_axis]
    if args.slice_method == 'even':
        return sampled_slice_indices(axis_len, args.num_slices)

    label_path = label_path_for_image(args.input_nii, args.label_root)
    label_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.float32)
    if label_volume.shape != volume.shape:
        raise ValueError(
            f"Label volume shape {label_volume.shape} does not match image volume shape {volume.shape}."
        )
    candidates = gt_slice_candidates(label_volume, axis=args.slice_axis, context=args.gt_context)
    return evenly_sample_candidates(candidates, args.num_slices)


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    slice_np = np.take(volume, indices=index, axis=axis)
    if slice_np.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {slice_np.shape}.")
    return slice_np.astype(np.float32)


def patch_pca_rgbs(patch_tokens: torch.Tensor, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = patch_tokens[0].float().cpu().numpy()
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(tokens, full_matrices=False)

    rgb_012 = tokens @ vt[:3].T
    rgb_345 = tokens @ vt[3:6].T

    rgb_012 = (rgb_012 - rgb_012.min(axis=0)) / (rgb_012.max(axis=0) - rgb_012.min(axis=0) + 1e-8)
    rgb_345 = (rgb_345 - rgb_345.min(axis=0)) / (rgb_345.max(axis=0) - rgb_345.min(axis=0) + 1e-8)

    return rgb_012.reshape(grid_size, grid_size, 3), rgb_345.reshape(grid_size, grid_size, 3)


def main() -> None:
    args = parse_args()
    if args.slice_size % args.patch_size != 0:
        raise ValueError("slice_size must be divisible by patch_size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = Flexi_CT_2D(checkpoint_path=args.checkpoint, device=device)
    model.eval()
    model.backbone.patch_embed_2D.set_patch_size(args.patch_size)

    volume = sitk.GetArrayFromImage(sitk.ReadImage(args.input_nii)).astype(np.float32)
    volume = normalize_volume(volume, args.norm_mode)

    slice_indices = select_slice_indices(args, volume)
    grid_size = args.slice_size // args.patch_size
    stem = Path(args.input_nii).name.replace(".nii.gz", "")

    saved = []
    for slice_index in slice_indices:
        slice_np = extract_slice(volume, axis=args.slice_axis, index=slice_index)
        slice_np = center_crop_or_resize_2d(slice_np, args.slice_size)

        x = torch.from_numpy(slice_np).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)

        rgb_012, rgb_345 = patch_pca_rgbs(out["patch_tokens"], grid_size=grid_size)

        feature_path = output_dir / f"{stem}_slice{slice_index:03d}_features.pt"
        figure_path = output_dir / f"{stem}_slice{slice_index:03d}_pca.png"

        torch.save(
            {
                "input_nii": args.input_nii,
                "checkpoint": args.checkpoint,
                "slice_axis": args.slice_axis,
                "slice_index": slice_index,
                "slice_size": args.slice_size,
                "patch_size": args.patch_size,
                "norm_mode": args.norm_mode,
                "num_slices": args.num_slices,
                "slice_method": args.slice_method,
                "gt_context": args.gt_context,
                "input_slice": torch.from_numpy(slice_np).unsqueeze(0),
                "cls_token": out["cls_token"].detach().cpu(),
                "patch_tokens": out["patch_tokens"].detach().cpu(),
            },
            feature_path,
        )

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(slice_np, cmap="gray")
        ax[0].set_title(f"Input slice {slice_index}")
        ax[0].axis("off")
        ax[1].imshow(rgb_012)
        ax[1].set_title(f"PCA 0-2 ({grid_size}x{grid_size})")
        ax[1].axis("off")
        ax[2].imshow(rgb_345)
        ax[2].set_title(f"PCA 3-5 ({grid_size}x{grid_size})")
        ax[2].axis("off")
        plt.tight_layout()
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved.append((slice_index, feature_path, figure_path, tuple(out["cls_token"].shape), tuple(out["patch_tokens"].shape)))

    for slice_index, feature_path, figure_path, cls_shape, patch_shape in saved:
        print(f"slice {slice_index:03d}")
        print(f"  features: {feature_path}")
        print(f"  figure:   {figure_path}")
        print(f"  cls:      {cls_shape}")
        print(f"  patches:  {patch_shape}")


if __name__ == "__main__":
    main()
