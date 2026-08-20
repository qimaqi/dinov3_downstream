#!/usr/bin/env python3
"""FOMO26 Task 3: Brain Age Estimation (Regression).

Predicts brain age from T1-weighted MRI using FlexiCT 2D slice encoder.
Output: A text file with predicted age in years.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

# ─── Setup paths ───────────────────────────────────────────────────────────────
APP_DIR = Path("/app")
sys.path.insert(0, str(APP_DIR / "dinov3_downstream"))

from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

# Import the training module to get model class
TRAIN_SCRIPT = APP_DIR / "dinov3_downstream" / "downstream" / "3d_regression" / "fomo_finetune_reg_from_slices.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("fomo_reg_train", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_MODULE = _load_train_module()
FlexiCTSliceVolumeRegressor = TRAIN_MODULE.FlexiCTSliceVolumeRegressor
resize_volume_chwd = TRAIN_MODULE.resize_volume_chwd

# ─── Model checkpoint path ────────────────────────────────────────────────────
MODEL_PATH = APP_DIR / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 3: Brain Age Estimation")
    parser.add_argument("--t1", type=str, required=True, help="Path to T1-weighted image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt file")
    return parser.parse_args()


def load_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data


def build_model(ckpt: dict, device: torch.device):
    saved_args = ckpt["args"]
    num_outputs = int(saved_args.get("num_outputs", 1))

    model = FlexiCTSliceVolumeRegressor(
        checkpoint=saved_args.get("checkpoint"),
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
        lora_targets=[s.strip() for s in saved_args.get("lora_targets", "qkv,proj").split(",") if s.strip()],
        lora_dropout=saved_args.get("lora_dropout", 0.0),
        transformer_depth=saved_args.get("transformer_depth", 2),
        transformer_heads=saved_args.get("transformer_heads", 8),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved_args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
    model, saved_args = build_model(ckpt, device)

    # Load T1 image
    t1_data = load_nifti(args.t1)
    volume = torch.from_numpy(t1_data[np.newaxis, ...])  # [1, H, W, D]

    # Resize if needed
    resize_hw_raw = saved_args.get("resize_hw")
    resize_hw = None
    if resize_hw_raw not in (None, "none", "null"):
        parts = tuple(int(x) for x in str(resize_hw_raw).split(",") if x.strip())
        if len(parts) == 2:
            resize_hw = parts
    volume = resize_volume_chwd(volume, resize_hw)
    volume = volume.unsqueeze(0).to(device=device, dtype=torch.float32)

    # Inference
    with torch.no_grad():
        prediction = model(volume=volume, slice_batch_size=int(saved_args.get("slice_batch_size", 32)))

    predicted_age = float(prediction[0].item())

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{predicted_age:.2f}\n")

    print(f"Predicted brain age: {predicted_age:.2f}")


if __name__ == "__main__":
    main()
