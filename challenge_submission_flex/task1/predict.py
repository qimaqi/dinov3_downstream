#!/usr/bin/env python3
"""FOMO26 Task 1: Infarct Detection (Binary Classification).

Predicts the probability of infarct presence from multi-modal brain MRI.
Uses FlexiCT 2D slice encoder + cross-slice pooling (patch_cls) + classification head.

Supports model ensembling: averages predictions from the top 3 selected folds.

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
import json
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
BACKBONE_KIND = os.environ.get("TASK1_BACKBONE_KIND", "flex").strip().lower()
if BACKBONE_KIND not in {"flex", "dinov3"}:
    raise ValueError(
        f"Unsupported TASK1_BACKBONE_KIND='{BACKBONE_KIND}'. Expected 'flex' or 'dinov3'."
    )

if BACKBONE_KIND == "dinov3":
    DEFAULT_RESULTS_DIR = (
        REPO_ROOT
        / "results"
        / "3d_classify"
        / "dinov3_slice_cls_each_robust_zscore"
        / "CLS002_FOMO26_Infarct"
    )
else:
    DEFAULT_RESULTS_DIR = (
        REPO_ROOT
        / "results"
        / "3d_classify"
        / "fomo_slice_cls_each_robust_zscore"
        / "CLS002_FOMO26_Infarct"
    )
_selected_folds_env = os.environ.get("TASK1_SELECTED_FOLDS")
if _selected_folds_env:
    SELECTED_FOLDS = tuple(
        fold.strip() for fold in _selected_folds_env.split(",") if fold.strip()
    )
else:
    SELECTED_FOLDS = ("fold_0", "fold_2", "fold_4")


def _resolve_backbone_checkpoint() -> Path:
    env_var = "DINOV3_2D_CHECKPOINT" if BACKBONE_KIND == "dinov3" else "FLEXICT_2D_CHECKPOINT"
    env_path = os.environ.get(env_var)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    if BACKBONE_KIND == "dinov3":
        candidates = [
            WEIGHTS_DIR / "2D_final_model.pth",
            REPO_ROOT
            / "ckpts"
            / "pretrain_fomo_10k_pretrained_dinov3_dino_base_g8_e400_p16_mri_normalize"
            / "2D_final_model.pth",
        ]
    else:
        candidates = [
            WEIGHTS_DIR / "2D_final_model_fomo100k_gram.pth",
            WEIGHTS_DIR / "2D_final_model.pth",
            REPO_ROOT
            / "ckpts"
            / "pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram"
            / "2D_final_model_fomo100k_gram.pth",
        ]
    for path in candidates:
        if path.exists():
            return path

    model_name = "DINOv3" if BACKBONE_KIND == "dinov3" else "FlexiCT"
    raise FileNotFoundError(
        f"Could not resolve the {model_name} 2D backbone checkpoint. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + (
            f"\n  - ${env_var}"
            if env_path
            else f"\nSet {env_var} to a valid checkpoint path if needed."
        )
    )


BACKBONE_CHECKPOINT_PATH = _resolve_backbone_checkpoint()

# Keep the relevant env var set so the downstream loader resolves consistently.
if BACKBONE_KIND == "dinov3":
    os.environ["DINOV3_2D_CHECKPOINT"] = str(BACKBONE_CHECKPOINT_PATH)
else:
    os.environ["FLEXICT_2D_CHECKPOINT"] = str(BACKBONE_CHECKPOINT_PATH)

# ─── Imports from repo ────────────────────────────────────────────────────────
from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

# downstream/3d_classify starts with a digit, so use importlib
if BACKBONE_KIND == "dinov3":
    _TRAIN_SCRIPT = REPO_ROOT / "downstream" / "3d_classify" / "dinov3_finetune_cls_from_slices.py"
    _TRAIN_MODULE_NAME = "dinov3_finetune_cls_from_slices"
else:
    _TRAIN_SCRIPT = REPO_ROOT / "downstream" / "3d_classify" / "fomo_finetune_cls_from_slices.py"
    _TRAIN_MODULE_NAME = "fomo_finetune_cls_from_slices"


def _load_module(script_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Register the module before execution so decorators like @dataclass
    # can resolve __module__ through sys.modules during import.
    sys.modules[module_name] = module
    # Register in sys.modules BEFORE exec so @dataclass can resolve the module
    # sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TRAIN_MODULE = _load_module(_TRAIN_SCRIPT, _TRAIN_MODULE_NAME)
FlexiCTSliceVolumeClassifier = _TRAIN_MODULE.FlexiCTSliceVolumeClassifier
_parse_hw_resize = _TRAIN_MODULE._parse_hw_resize
_parse_spacing_xy = _TRAIN_MODULE._parse_spacing_xy
normalize_volume_chwd = _TRAIN_MODULE.normalize_volume_chwd
pad_volume_chwd = _TRAIN_MODULE.pad_volume_chwd
respace_xy_volume_chwd = _TRAIN_MODULE.respace_xy_volume_chwd
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


def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Load a NIfTI image and return data plus in-plane spacing (X, Y)."""
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


def collect_modalities(args: argparse.Namespace) -> tuple[torch.Tensor, list[tuple[float, float] | None]]:
    """Stack available modalities into a single [C, H, W, D] tensor."""
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
            raise ValueError(
                f"Shape mismatch: expected {reference_shape}, "
                f"got {arr.shape} for modality '{name}'"
            )
        arrays.append(arr)
        spacings_xy.append(spacing_xy)

    if not arrays:
        raise ValueError("At least one modality path must be provided.")

    # Stack to [C, H, W, D]
    return torch.from_numpy(np.stack(arrays, axis=0)), spacings_xy


def preprocess_volume(
    volume: torch.Tensor,
    spacings_xy: list[tuple[float, float] | None],
    saved_args: dict,
) -> torch.Tensor:
    """Match training-time volume preprocessing as closely as possible."""
    resize_hw = _parse_hw_resize(saved_args.get("resize_hw"))
    target_spacing_xy = _parse_spacing_xy(saved_args.get("target_spacing_xy"))
    pad_hw = _parse_hw_resize(saved_args.get("pad_hw"))
    mri_normalization = saved_args.get("mri_normalization", "robust_zscore")
    mri_low_percentile = float(saved_args.get("mri_low_percentile", 0.5))
    mri_high_percentile = float(saved_args.get("mri_high_percentile", 99.5))

    processed_channels = []
    for channel_idx in range(volume.shape[0]):
        channel = volume[channel_idx : channel_idx + 1].clone()
        channel = respace_xy_volume_chwd(channel, spacings_xy[channel_idx], target_spacing_xy)
        channel = pad_volume_chwd(channel, pad_hw)
        channel = resize_volume_chwd(channel, resize_hw)
        channel = normalize_volume_chwd(
            channel,
            method=mri_normalization,
            low_percentile=mri_low_percentile,
            high_percentile=mri_high_percentile,
        )
        processed_channels.append(channel)
    return torch.cat(processed_channels, dim=0)


def build_model(ckpt: dict, device: torch.device) -> tuple:
    """Reconstruct the saved slice classifier from a checkpoint."""
    saved_args = ckpt["args"]
    task = ckpt.get("task", "CLS002_FOMO26_Infarct")
    num_classes = 2 if task == "CLS002_FOMO26_Infarct" else int(saved_args.get("num_classes", 2))

    if BACKBONE_KIND == "dinov3":
        model = FlexiCTSliceVolumeClassifier(
            dinov3_checkpoint=str(BACKBONE_CHECKPOINT_PATH),
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
    else:
        model = FlexiCTSliceVolumeClassifier(
            checkpoint=str(BACKBONE_CHECKPOINT_PATH),
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


def _candidate_model_roots() -> list[Path]:
    env_root = os.environ.get("TASK1_MODEL_ROOT")
    roots = []
    if env_root:
        roots.append(Path(env_root))
    roots.extend(
        [
            WEIGHTS_DIR,
            DEFAULT_RESULTS_DIR,
        ]
    )
    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def get_model_paths() -> list[Path]:
    """Resolve the selected ensemble checkpoints from known local/container roots."""
    roots = _candidate_model_roots()
    attempted = []
    for root in roots:
        model_paths = [root / fold / "best.pt" for fold in SELECTED_FOLDS]
        attempted.extend(model_paths)
        if all(path.exists() for path in model_paths):
            metrics_path = root / SELECTED_FOLDS[0] / "metrics.json"
            if metrics_path.exists():
                print(f"Using model root: {root}")
            return model_paths

    raise FileNotFoundError(
        "Could not find all selected Task 1 ensemble checkpoints.\n"
        "Expected these files:\n"
        + "\n".join(f"  - {p}" for p in attempted)
    )


def print_selected_folds_summary(model_paths: list[Path]) -> None:
    print("Selected ensemble folds (ranked by lowest validation loss):")
    for path in model_paths:
        metrics_path = path.with_name("metrics.json")
        summary = ""
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            best_val_loss = metrics.get("best_val_loss")
            test_metrics = metrics.get("test", {})
            summary = (
                f" | best_val_loss={best_val_loss:.6f}"
                f", test_auc={test_metrics.get('auc', float('nan')):.3f}"
                f", test_bal_acc={test_metrics.get('bal_acc', float('nan')):.3f}"
            )
        print(f"  - {path.parent.name}: {path}{summary}")


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

    # Add batch dimension: [1, C, H, W, D]
    vol = volume.unsqueeze(0).to(device=device, dtype=torch.float32)

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

    # ── Discover ensemble checkpoints ─────────────────────────────────────────
    model_paths = get_model_paths()
    n_models = len(model_paths)
    print(f"Backbone kind: {BACKBONE_KIND}")
    print(f"Backbone checkpoint: {BACKBONE_CHECKPOINT_PATH}")
    print(f"Ensemble size: {n_models} model(s)")
    print_selected_folds_summary(model_paths)

    # ── Load and preprocess input volumes ─────────────────────────────────────
    volume, spacings_xy = collect_modalities(args)  # [C, H, W, D]

    # ── Run inference (ensemble if multiple models) ───────────────────────────
    probabilities = []
    for i, model_path in enumerate(model_paths):
        ckpt = torch.load(str(model_path), map_location="cpu")
        processed_volume = preprocess_volume(volume, spacings_xy, ckpt["args"])
        prob = predict_single_model(model_path, processed_volume, device)
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
