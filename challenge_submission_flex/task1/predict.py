#!/usr/bin/env python3
"""FOMO26 Task 1: Infarct Detection (Binary Classification).

Predicts the probability of infarct presence from multi-modal brain MRI.
Uses FlexiCT 2D slice encoder + cross-slice pooling (patch_cls) + classification head.

Supports model ensembling: provide multiple checkpoint paths to average predictions.

Modality handling:
    When trained with --modality_pool each, the model processes each modality
    channel independently (separate forward passes) and averages logits across
    modalities before computing probabilities.

Repository layout (local development):
    dinov3_downstream/                      <- repo root
    ├── flexi_ct/                           <- FlexiCT encoder package
    ├── downstream/3d_classify/             <- model architecture
    └── challenge_submission_flex/task1/
        ├── predict.py                      <- this script
        └── weights/                        <- trained checkpoints

Container layout (Apptainer):
    /app/predict.py
    /app/flexi_ct/
    /app/downstream/3d_classify/
    /app/weights/
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ─── Resolve repo root vs container root ──────────────────────────────────────
# Local: predict.py is at <repo_root>/challenge_submission_flex/task1/predict.py
# Container: predict.py is at /app/predict.py with sources copied alongside
_THIS_DIR = Path(__file__).resolve().parent
_CONTAINER_ROOT = Path("/app")

if (_THIS_DIR / "../../flexi_ct").resolve().is_dir():
    # Running locally from within the repo
    REPO_ROOT = (_THIS_DIR / "../..").resolve()
    WEIGHTS_DIR = _THIS_DIR / "weights"
elif _CONTAINER_ROOT.is_dir() and (_CONTAINER_ROOT / "flexi_ct").is_dir():
    # Running inside the Apptainer container
    REPO_ROOT = _CONTAINER_ROOT
    WEIGHTS_DIR = _CONTAINER_ROOT / "weights"
else:
    raise RuntimeError(
        "Cannot resolve project root. Run from within the repo or inside the container."
    )

# Add repo root to sys.path for flexi_ct imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# MODEL WEIGHT PATHS
# ============================================================================
MODEL_PATHS = [
    WEIGHTS_DIR / "fold_0" / "best.pt",
    WEIGHTS_DIR / "fold_1" / "best.pt",
    WEIGHTS_DIR / "fold_2" / "best.pt",
    WEIGHTS_DIR / "fold_3" / "best.pt",
    WEIGHTS_DIR / "fold_4" / "best.pt",
]

FLEXICT_2D_CHECKPOINT_PATH = WEIGHTS_DIR / "2D_final_model.pth"

# Set env var so flexi_ct.checkpoints can resolve the backbone
os.environ["FLEXICT_2D_CHECKPOINT"] = str(FLEXICT_2D_CHECKPOINT_PATH)

# ─── Imports from repo ────────────────────────────────────────────────────────
from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

# downstream/3d_classify starts with a digit, so use importlib
_TRAIN_SCRIPT = REPO_ROOT / "downstream" / "3d_classify" / "fomo_finetune_cls_from_slices.py"


def _load_module(script_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAIN_MODULE = _load_module(_TRAIN_SCRIPT, "fomo_finetune_cls_from_slices")
FlexiCTSliceVolumeClassifier = _TRAIN_MODULE.FlexiCTSliceVolumeClassifier
resize_volume_chwd = _TRAIN_MODULE.resize_volume_chwd


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 1: Infarct Detection (Binary Classification)"
    )
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, help="Path to DWI image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt file")
    return parser.parse_args()


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI image and return as float32 numpy array."""
    import nibabel as nib

    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data


def collect_modalities(args: argparse.Namespace) -> torch.Tensor:
    """Stack available modalities into a single [C, H, W, D] tensor."""
    modality_paths = [
        ("flair", args.flair),
        ("adc", args.adc),
        ("dwi", args.dwi),
        ("t2s", args.t2s),
        ("swi", args.swi),
    ]

    arrays = []
    reference_shape = None
    for name, path in modality_paths:
        if path is None:
            continue
        arr = load_nifti(path)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch: expected {reference_shape}, "
                f"got {arr.shape} for modality '{name}'"
            )
        arrays.append(arr)

    if not arrays:
        raise ValueError("At least one modality path must be provided.")

    # Stack to [C, H, W, D]
    return torch.from_numpy(np.stack(arrays, axis=0))


def build_model(ckpt: dict, device: torch.device) -> tuple:
    """Reconstruct FlexiCTSliceVolumeClassifier from saved checkpoint."""
    saved_args = ckpt["args"]
    task = ckpt.get("task", "CLS002_FOMO26_Infarct")
    num_classes = 2 if task == "CLS002_FOMO26_Infarct" else int(saved_args.get("num_classes", 2))

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
    """Run inference with a single model and return positive class probability.

    Handles modality_pool="each" by running the model forward for each modality
    independently and averaging logits across modalities before softmax.
    """
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

    modality_pool = saved_args.get("modality_pool", "mean")
    slice_batch_size = int(saved_args.get("slice_batch_size", 32))

    # Inference
    with torch.no_grad():
        if modality_pool == "each":
            # Per-modality forward: the model's modality_views() splits
            # [1, C, H, W, D] into [C, 1, H, W, D] so each modality is
            # processed independently, yielding logits of shape [C, num_classes].
            # We average logits across modalities before softmax.
            num_modalities = vol.shape[1]
            logits_each = model(volume=vol, slice_batch_size=slice_batch_size)
            # logits_each: [num_modalities, num_classes]
            logits = logits_each.view(1, num_modalities, -1).mean(dim=1)  # [1, num_classes]
        else:
            logits = model(volume=vol, slice_batch_size=slice_batch_size)

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

    # ── Load and preprocess input volumes ─────────────────────────────────────
    volume = collect_modalities(args)  # [C, H, W, D]

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
