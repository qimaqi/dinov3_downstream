#!/usr/bin/env python3
"""FOMO26 Tasks 6 & 7: Linear Probing and Bias & Fairness (Embedding Extraction).

Produces a fixed-length 1D embedding from an arbitrary MR image using
the FlexiCT 2D encoder with cross-slice pooling (frozen backbone).
Output: A NumPy .npy file with the embedding vector.
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

# We use the classifier/regressor model architecture, but extract the
# volume-level embedding (before the classification/regression head).
# Use whichever model file is available (cls or reg).
TRAIN_SCRIPT_CLS = APP_DIR / "dinov3_downstream" / "downstream" / "3d_classify" / "fomo_finetune_cls_from_slices.py"
TRAIN_SCRIPT_REG = APP_DIR / "dinov3_downstream" / "downstream" / "3d_regression" / "fomo_finetune_reg_from_slices.py"


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Try classifier first, fallback to regressor
if TRAIN_SCRIPT_CLS.exists():
    TRAIN_MODULE = _load_module(TRAIN_SCRIPT_CLS, "fomo_cls_train")
    ModelClass = TRAIN_MODULE.FlexiCTSliceVolumeClassifier
    MODEL_TYPE = "classifier"
elif TRAIN_SCRIPT_REG.exists():
    TRAIN_MODULE = _load_module(TRAIN_SCRIPT_REG, "fomo_reg_train")
    ModelClass = TRAIN_MODULE.FlexiCTSliceVolumeRegressor
    MODEL_TYPE = "regressor"
else:
    raise ImportError("No training module found for embedding extraction.")

resize_volume_chwd = TRAIN_MODULE.resize_volume_chwd

# ─── Model checkpoint path ────────────────────────────────────────────────────
MODEL_PATH = APP_DIR / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Tasks 6/7: Embedding Extraction")
    parser.add_argument("--input", type=str, required=True, help="Path to input MR image (NIfTI)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .npy file")
    return parser.parse_args()


def load_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data


def build_model(ckpt: dict, device: torch.device):
    saved_args = ckpt["args"]

    if MODEL_TYPE == "classifier":
        num_classes = int(saved_args.get("num_classes", 2))
        model = ModelClass(
            checkpoint=saved_args.get("checkpoint"),
            num_classes=num_classes,
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
    else:
        num_outputs = int(saved_args.get("num_outputs", 1))
        model = ModelClass(
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

    # Load input image
    image_data = load_nifti(args.input)
    volume = torch.from_numpy(image_data[np.newaxis, ...])  # [1, H, W, D]

    # Resize if needed
    resize_hw_raw = saved_args.get("resize_hw")
    resize_hw = None
    if resize_hw_raw not in (None, "none", "null"):
        parts = tuple(int(x) for x in str(resize_hw_raw).split(",") if x.strip())
        if len(parts) == 2:
            resize_hw = parts
    volume = resize_volume_chwd(volume, resize_hw)
    volume = volume.unsqueeze(0).to(device=device, dtype=torch.float32)

    # Extract embedding (volume token before the head)
    with torch.no_grad():
        embedding = model.forward_features(
            volume=volume,
            slice_batch_size=int(saved_args.get("slice_batch_size", 32)),
        )

    # Convert to numpy and save
    embedding_np = embedding[0].cpu().numpy().astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embedding_np)

    print(f"Embedding shape: {embedding_np.shape}, saved to {args.output}")


if __name__ == "__main__":
    main()
