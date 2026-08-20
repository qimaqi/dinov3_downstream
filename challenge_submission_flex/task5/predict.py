#!/usr/bin/env python3
"""FOMO26 Task 5: Polymicrogyria Classification (Binary).

Classifies the presence of polymicrogyria in brain MRI from T1-weighted images.
Uses FlexiCT 2D slice encoder + cross-slice pooling (cls) + classification head.

Supports model ensembling: provide multiple checkpoint paths to average predictions.

Container layout:
    /app/predict.py                         <- this script
    /app/dinov3_downstream/flexi_ct/        <- FlexiCT encoder package
    /app/dinov3_downstream/downstream/3d_classify/  <- model architecture
    /app/weights/2D_final_model.pth         <- pretrained FlexiCT 2D backbone
    /app/weights/fold_X/best.pt             <- trained classifier checkpoint(s)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ─── Paths inside the container ───────────────────────────────────────────────
APP_DIR = Path("/app")

# ============================================================================
# MODEL WEIGHT PATHS — edit these for your setup
# ============================================================================

# 1) List of trained classifier checkpoints for ensembling.
#    If len >= 2, predictions are averaged across all models.
#    Each checkpoint format: {"model": state_dict, "args": training_args, "task": ...}
MODEL_PATHS = [
    APP_DIR / "weights" / "fold_0" / "best.pt",
    APP_DIR / "weights" / "fold_1" / "best.pt",
    APP_DIR / "weights" / "fold_2" / "best.pt",
    APP_DIR / "weights" / "fold_3" / "best.pt",
    APP_DIR / "weights" / "fold_4" / "best.pt",
]

# 2) Pretrained FlexiCT 2D backbone checkpoint
FLEXICT_2D_CHECKPOINT_PATH = APP_DIR / "weights" / "2D_final_model.pth"

# ============================================================================

# Set env var so flexi_ct.checkpoints can resolve the backbone
os.environ["FLEXICT_2D_CHECKPOINT"] = str(FLEXICT_2D_CHECKPOINT_PATH)

# Add dinov3_downstream to path for imports
sys.path.insert(0, str(APP_DIR / "dinov3_downstream"))

from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

# Import the training module to get the model class (FlexiCTSliceVolumeClassifier)
TRAIN_SCRIPT = (
    APP_DIR / "dinov3_downstream" / "downstream" / "3d_classify"
    / "fomo_finetune_cls_from_slices.py"
)


def _load_module(script_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the training module (contains FlexiCTSliceVolumeClassifier + resize_volume_chwd)
TRAIN_MODULE = _load_module(TRAIN_SCRIPT, "fomo_finetune_cls_from_slices")
FlexiCTSliceVolumeClassifier = TRAIN_MODULE.FlexiCTSliceVolumeClassifier
resize_volume_chwd = TRAIN_MODULE.resize_volume_chwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 5: Polymicrogyria Classification (Binary)"
    )
    parser.add_argument("--t1", type=str, required=True, help="Path to T1-weighted image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt file")
    return parser.parse_args()


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI image and return as float32 numpy array."""
    import nibabel as nib

    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data


def build_model(ckpt: dict, device: torch.device) -> tuple:
    """Reconstruct FlexiCTSliceVolumeClassifier from saved checkpoint."""
    saved_args = ckpt["args"]
    task = ckpt.get("task", "CLS003_FOMO26_Polymicrogyria")
    num_classes = 2 if task == "CLS003_FOMO26_Polymicrogyria" else int(saved_args.get("num_classes", 2))

    model = FlexiCTSliceVolumeClassifier(
        checkpoint=str(FLEXICT_2D_CHECKPOINT_PATH),
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


def get_available_model_paths() -> list[Path]:
    """Filter MODEL_PATHS to only those that exist on disk."""
    available = [p for p in MODEL_PATHS if p.exists()]
    if not available:
        raise FileNotFoundError(
            f"No model checkpoints found. Expected at least one of:\n"
            + "\n".join(f"  - {p}" for p in MODEL_PATHS)
        )
    return available


def predict_single_model(
    model_path: Path,
    volume: torch.Tensor,
    device: torch.device,
) -> float:
    """Run inference with a single model and return positive class probability."""
    ckpt = torch.load(str(model_path), map_location="cpu")
    model, saved_args = build_model(ckpt, device)

    # Apply resize if the model was trained with --resize_hw
    resize_hw_raw = saved_args.get("resize_hw")
    resize_hw = None
    if resize_hw_raw not in (None, "none", "null"):
        parts = tuple(int(x) for x in str(resize_hw_raw).split(",") if x.strip())
        if len(parts) == 2:
            resize_hw = parts
    vol = resize_volume_chwd(volume.clone(), resize_hw)

    # Add batch dimension: [1, C, H, W, D]
    vol = vol.unsqueeze(0).to(device=device, dtype=torch.float32)

    # Inference
    with torch.no_grad():
        logits = model(
            volume=vol,
            slice_batch_size=int(saved_args.get("slice_batch_size", 32)),
        )
        probs = torch.softmax(logits, dim=-1)[0]

    return float(probs[1].item())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Discover available model checkpoints ──────────────────────────────────
    model_paths = get_available_model_paths()
    n_models = len(model_paths)
    print(f"Ensemble size: {n_models} model(s)")
    for i, p in enumerate(model_paths):
        print(f"  [{i}] {p}")

    # ── Load T1 image ─────────────────────────────────────────────────────────
    t1_data = load_nifti(args.t1)
    volume = torch.from_numpy(t1_data[np.newaxis, ...])  # [1, H, W, D]

    # ── Run inference (ensemble if multiple models) ───────────────────────────
    probabilities = []
    for i, model_path in enumerate(model_paths):
        prob = predict_single_model(model_path, volume, device)
        probabilities.append(prob)
        print(f"  Model [{i}] probability: {prob:.4f}")

    # Average probabilities across all models
    ensemble_probability = sum(probabilities) / len(probabilities)

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{ensemble_probability:.3f}")

    print(f"\nFinal ensemble probability: {ensemble_probability:.3f} "
          f"(from {n_models} model(s))")


if __name__ == "__main__":
    main()
