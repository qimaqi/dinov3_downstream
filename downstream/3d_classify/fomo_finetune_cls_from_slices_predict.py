#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).resolve().with_name("fomo_finetune_cls_from_slices.py")


def _load_train_module():
    spec = importlib.util.spec_from_file_location("fomo_finetune_cls_from_slices_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load training module from {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_MODULE = _load_train_module()
FlexiCTSliceVolumeClassifier = TRAIN_MODULE.FlexiCTSliceVolumeClassifier
resize_volume_chwd = TRAIN_MODULE.resize_volume_chwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict FOMO infarct probability from a trained slice classifier.")
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, help="Path to DWI image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path used to derive the output <subject_id>.txt file")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU")
    return parser.parse_args()


def _load_nifti(path: str) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for prediction. Please install nibabel in the active environment.") from exc

    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data


def _collect_modalities(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    modality_paths = [
        ("flair", args.flair),
        ("adc", args.adc),
        ("dwi", args.dwi),
        ("t2s", args.t2s),
        ("swi", args.swi),
    ]
    arrays = []
    used_names = []
    reference_shape = None
    for name, path in modality_paths:
        if path is None:
            continue
        arr = _load_nifti(path)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(f"All modalities must share the same shape. Expected {reference_shape}, got {arr.shape} for {path}")
        arrays.append(arr)
        used_names.append(name)

    if not arrays:
        raise ValueError("At least one modality path must be provided.")

    volume = np.stack(arrays, axis=0)
    return torch.from_numpy(volume), ",".join(used_names)


def _build_model(ckpt: dict, device: torch.device):
    saved_args = ckpt["args"]
    task = ckpt.get("task", "CLS002_FOMO26_Infarct")
    num_classes = 2 if task == "CLS002_FOMO26_Infarct" else int(saved_args.get("num_classes", 2))
    model = FlexiCTSliceVolumeClassifier(
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
        lora_targets=[item.strip() for item in saved_args.get("lora_targets", "qkv,proj").split(",") if item.strip()],
        lora_dropout=saved_args.get("lora_dropout", 0.0),
        transformer_depth=saved_args.get("transformer_depth", 2),
        transformer_heads=saved_args.get("transformer_heads", 8),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved_args


def predict_probability(args: argparse.Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.model, map_location="cpu")
    model, saved_args = _build_model(ckpt, device)

    volume, used_modalities = _collect_modalities(args)
    resize_hw_raw = saved_args.get("resize_hw")
    resize_hw = None
    if resize_hw_raw not in (None, "none", "null"):
        parts = tuple(int(item) for item in str(resize_hw_raw).split(",") if item.strip())
        if len(parts) != 2:
            raise ValueError(f"Expected resize_hw to be H,W, got {resize_hw_raw}")
        resize_hw = parts
    volume = resize_volume_chwd(volume, resize_hw)
    volume = volume.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(volume=volume, slice_batch_size=int(saved_args.get("slice_batch_size", 32)))
        probs = torch.softmax(logits, dim=-1)[0]

    if probs.numel() < 2:
        raise ValueError("Expected a binary classifier with at least 2 output logits.")

    print(f"Using modalities: {used_modalities}")
    return float(probs[1].item())


def main() -> int:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    probability = predict_probability(args)
    subject_id = Path(args.output).stem
    output_file = Path(args.output).parent / f"{subject_id}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
