"""Fine-tune FOMO 3D classification with a FlexiCT 2D slice encoder.

This is a draft interface for using the 2D Flexi_CT backbone on 3D FOMO
classification tasks. Each volume is converted into a sequence of 2D slices,
Flexi_CT_2D produces one CLS token per slice, and a configurable pooling module
merges the slice tokens into one volume-level token for classification.

Supported tasks:
  - CLS002_FOMO26_Infarct
  - CLS003_FOMO26_Polymicrogyria

The default keeps the FlexiCT encoder frozen and trains only the cross-slice
pooler plus linear classifier. Pass --lora_encoder to train LoRA adapters in
the 2D backbone, or --unfreeze_encoder for full fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DINOV3_ROOT = Path("/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT")
if str(DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV3_ROOT))

from dinov3.models.vision_transformer import vit_base  # noqa: E402

DEFAULT_PROCESSED_ROOT = (
    "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/"
    "FOMO_Challenge/processed_data"
)
DEFAULT_RAW_ROOT = (
    "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks"
)
DEFAULT_OUTPUT_DIR = str(ROOT / "results" / "3d_classify" / "fomo_slice_cls")
DEFAULT_TASKS = ["CLS002_FOMO26_Infarct", "CLS003_FOMO26_Polymicrogyria"]
DEFAULT_SPLIT = "split_80_10_10"
DEFAULT_TEST_SPLIT = "TEST_80_10_10"
SEED = 42

RAW_TASK_CONFIG = {
    "CLS002_FOMO26_Infarct": {
        "image_root": Path(DEFAULT_RAW_ROOT) / "Task_1" / "Task_1" / "preprocessed",
        "label_root": Path(DEFAULT_RAW_ROOT) / "Task_1" / "Task_1" / "labels",
        "modalities": ("flair.nii.gz", "adc.nii.gz", "dwi_b1000.nii.gz"),
        "label_file": "label.txt",
    },
    "CLS003_FOMO26_Polymicrogyria": {
        "image_root": Path(DEFAULT_RAW_ROOT) / "Task_5" / "preprocessed",
        "label_root": Path(DEFAULT_RAW_ROOT) / "Task_5" / "labels",
        "modalities": ("t1.nii.gz",),
        "label_file": "labels.txt",
    },
}


def resolve_dinov3_checkpoint(explicit_checkpoint: str | None = None) -> str:
    if explicit_checkpoint:
        return explicit_checkpoint

    env_checkpoint = os.environ.get("DINOV3_2D_CHECKPOINT")
    if env_checkpoint:
        return env_checkpoint

    raise ValueError(
        "DINOv3 checkpoint path is required. Pass --dinov3_checkpoint or set "
        "DINOV3_2D_CHECKPOINT."
    )


class DINOv3SliceEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, patch_size: int, device: torch.device):
        super().__init__()
        self.backbone = vit_base(
            in_chans=3,
            patch_size=patch_size,
            qkv_bias=True,
            n_storage_tokens=4,
            mask_k_bias=False,
            layerscale_init=1.0e-05,
        )
        chkpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = chkpt.get("teacher", chkpt.get("model", chkpt))
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading DINOv3 checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys while loading DINOv3 checkpoint: {missing}")
        self.to(device)

    @staticmethod
    def prepare_input_channels(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError(f"DINOv3 slice encoder expects 1 or 3 channels, got {x.shape[1]}.")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.prepare_input_channels(x)
        out = self.backbone(x, is_training=True)
        return {
            "cls_token": out["x_norm_clstoken"],
            "patch_tokens": out["x_norm_patchtokens"],
        }


@dataclass(frozen=True)
class SplitFiles:
    train: list[str]
    val: list[str]
    test: list[str]


def _set_patch_size(model: torch.nn.Module, patch_size: int) -> None:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        current = getattr(model.patch_embed, "patch_size", None)
        if current is not None:
            current_tuple = current if isinstance(current, tuple) else (current, current)
            if tuple(current_tuple) != (patch_size, patch_size):
                raise ValueError(
                    f"DINOv3 checkpoint was built for patch_size={current_tuple[0]}, "
                    f"but requested patch_size={patch_size}."
                )
        return
    for module in model.modules():
        if hasattr(module, "set_patch_size"):
            module.set_patch_size(patch_size)
    if hasattr(model, "patch_size"):
        model.patch_size = patch_size


def _apply_lora_to_backbone(
    encoder: Flexi_CT_2D,
    r: int,
    alpha: int,
    target_modules: list[str],
    dropout: float,
) -> None:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "--lora_encoder requires peft. Install it in the active environment first."
        ) from exc

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
    )
    encoder.backbone = get_peft_model(encoder.backbone, config)
    if hasattr(encoder.backbone, "print_trainable_parameters"):
        encoder.backbone.print_trainable_parameters()


def load_task_metadata(task_dir: Path) -> dict:
    with open(task_dir / "dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_files(
    task_dir: Path,
    split_name: str,
    test_split_name: str,
    fold: int,
) -> SplitFiles:
    with open(task_dir / f"{split_name}.json", "r", encoding="utf-8") as f:
        folds = json.load(f)
    if not isinstance(folds, list):
        raise ValueError(f"Expected {split_name}.json to contain a list of folds.")
    if fold < 0 or fold >= len(folds):
        raise ValueError(f"Fold {fold} is out of range for {split_name}.json with {len(folds)} folds.")

    fold_data = folds[fold]
    with open(task_dir / f"{test_split_name}.json", "r", encoding="utf-8") as f:
        test_files = json.load(f)
    return SplitFiles(
        train=list(fold_data["train"]),
        val=list(fold_data["val"]),
        test=list(test_files),
    )


def _as_image_label(sample: object, path: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(sample, (list, tuple)) or len(sample) < 2:
        raise ValueError(f"Expected {path} to contain [image_tensor, label_tensor].")
    image = torch.as_tensor(sample[0], dtype=torch.float32)
    label = torch.as_tensor(sample[1]).view(-1)[0].long()
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [C,H,W,D] in {path}, got shape {tuple(image.shape)}.")
    return image, label


def _parse_hw_resize(value: str | None) -> tuple[int, int] | None:
    if value is None or value.lower() in {"none", "null"}:
        return None
    parts = tuple(int(item) for item in value.split(",") if item.strip())
    if len(parts) != 2:
        raise ValueError(f"Expected H,W resize tuple, got: {value}")
    return parts


def resize_volume_chwd(image: torch.Tensor, target_hw: tuple[int, int] | None) -> torch.Tensor:
    if target_hw is None:
        return image
    if tuple(image.shape[1:3]) == target_hw:
        return image
    image_dhw = image.permute(0, 3, 1, 2).unsqueeze(0)
    image_dhw = F.interpolate(image_dhw, size=(image.shape[3], target_hw[0], target_hw[1]), mode="trilinear")
    return image_dhw.squeeze(0).permute(0, 2, 3, 1).contiguous()


def _metadata_path_for_sample(path: str) -> Path:
    return Path(path).with_suffix(".pkl")


def _load_spacing_metadata(path: str) -> tuple[float, float] | None:
    meta_path = _metadata_path_for_sample(path)
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    spacing = meta.get("original_spacing") or meta.get("new_spacing")
    if spacing is None or len(spacing) < 2:
        return None
    return float(spacing[0]), float(spacing[1])


def _parse_spacing_xy(value: str | None) -> tuple[float, float] | None:
    if value is None or value.lower() in {"none", "null"}:
        return None
    parts = tuple(float(item) for item in value.split(",") if item.strip())
    if len(parts) != 2:
        raise ValueError(f"Expected X,Y spacing tuple, got: {value}")
    return parts


def respace_xy_volume_chwd(
    image: torch.Tensor,
    source_spacing_xy: tuple[float, float] | None,
    target_spacing_xy: tuple[float, float] | None,
) -> torch.Tensor:
    if source_spacing_xy is None or target_spacing_xy is None:
        return image
    src_x, src_y = source_spacing_xy
    tgt_x, tgt_y = target_spacing_xy
    target_h = max(1, int(round(image.shape[1] * src_y / tgt_y)))
    target_w = max(1, int(round(image.shape[2] * src_x / tgt_x)))
    if (target_h, target_w) == tuple(image.shape[1:3]):
        return image
    image_dhw = image.permute(0, 3, 1, 2).unsqueeze(0)
    image_dhw = F.interpolate(image_dhw, size=(image.shape[3], target_h, target_w), mode="trilinear", align_corners=False)
    return image_dhw.squeeze(0).permute(0, 2, 3, 1).contiguous()


def pad_volume_chwd(image: torch.Tensor, target_hw: tuple[int, int] | None) -> torch.Tensor:
    if target_hw is None:
        return image
    target_h, target_w = target_hw
    height, width = image.shape[1:3]
    if height > target_h or width > target_w:
        raise ValueError(
            f"pad_hw={target_hw} is smaller than image shape {(height, width)} after respacing. "
            "Increase pad_hw or use resize_hw."
        )
    pad_h = target_h - height
    pad_w = target_w - width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h == 0 and pad_w == 0:
        return image
    image_dhw = image.permute(0, 3, 1, 2)
    image_dhw = F.pad(image_dhw, (left, right, top, bottom))
    return image_dhw.permute(0, 2, 3, 1).contiguous()


def center_crop_or_pad_volume_chwd(image: torch.Tensor, target_hw: tuple[int, int] | None) -> torch.Tensor:
    if target_hw is None:
        return image
    target_h, target_w = target_hw
    height, width = image.shape[1:3]
    start_h = max((height - target_h) // 2, 0)
    start_w = max((width - target_w) // 2, 0)
    end_h = start_h + min(height, target_h)
    end_w = start_w + min(width, target_w)
    cropped = image[:, start_h:end_h, start_w:end_w, :]
    crop_h, crop_w = cropped.shape[1:3]
    if crop_h == target_h and crop_w == target_w:
        return cropped
    pad_h = target_h - crop_h
    pad_w = target_w - crop_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    cropped_dhw = cropped.permute(0, 3, 1, 2)
    cropped_dhw = F.pad(cropped_dhw, (left, right, top, bottom))
    return cropped_dhw.permute(0, 2, 3, 1).contiguous()


def _import_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required for raw FOMO .nii.gz loading.") from exc
    return sitk


def _resolve_raw_sample_paths(task: str, split_path: str, raw_root: str | None) -> tuple[list[Path], Path]:
    config = RAW_TASK_CONFIG[task]
    split_file = Path(split_path)
    subject = split_file.parents[1].name
    session = split_file.parents[0].name
    if raw_root is None:
        image_root = config["image_root"]
        label_root = config["label_root"]
    else:
        base_root = Path(raw_root)
        image_root = base_root / config["image_root"].relative_to(DEFAULT_RAW_ROOT)
        label_root = base_root / config["label_root"].relative_to(DEFAULT_RAW_ROOT)
    image_paths = [image_root / subject / session / modality for modality in config["modalities"]]
    label_path = label_root / subject / session / config["label_file"]
    return image_paths, label_path


def _load_raw_volume_chwd(image_paths: list[Path], target_spacing_xy: tuple[float, float] | None) -> torch.Tensor:
    sitk = _import_sitk()
    channels: list[torch.Tensor] = []
    for image_path in image_paths:
        image = sitk.ReadImage(str(image_path))
        oriented = sitk.DICOMOrient(image, "RAS")
        original_spacing = tuple(float(v) for v in oriented.GetSpacing())
        if target_spacing_xy is not None:
            target_spacing = (float(target_spacing_xy[0]), float(target_spacing_xy[1]), original_spacing[2])
            original_size = oriented.GetSize()
            new_size = [
                max(1, int(round(size * spacing / target)))
                for size, spacing, target in zip(original_size, original_spacing, target_spacing)
            ]
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(oriented.GetDirection())
            resampler.SetOutputOrigin(oriented.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0.0)
            resampler.SetInterpolator(sitk.sitkLinear)
            oriented = resampler.Execute(oriented)
        array_zyx = sitk.GetArrayFromImage(oriented).astype(np.float32, copy=False)
        channels.append(torch.from_numpy(np.transpose(array_zyx, (1, 2, 0))).contiguous())
    return torch.stack(channels, dim=0)


def _load_raw_label(label_path: Path) -> torch.Tensor:
    return torch.tensor(int(label_path.read_text().strip()), dtype=torch.long)


def save_debug_slice_image(batch: dict[str, object], output_path: Path) -> None:
    if output_path.exists():
        return
    image = batch["image"]
    if not isinstance(image, torch.Tensor):
        return
    volume = image[0]
    channel = volume[0]
    mid_slice = channel[:, :, channel.shape[-1] // 2].detach().cpu().float().numpy()
    finite_mask = np.isfinite(mid_slice)
    if not finite_mask.any():
        mid_slice = np.zeros_like(mid_slice, dtype=np.uint8)
    else:
        values = mid_slice[finite_mask]
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            mid_slice = np.zeros_like(mid_slice, dtype=np.uint8)
        else:
            mid_slice = np.clip((mid_slice - lo) / (hi - lo), 0.0, 1.0)
            mid_slice = (mid_slice * 255.0).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save debug validation slice images.") from exc
    Image.fromarray(mid_slice, mode="L").save(output_path)


def normalize_volume_chwd(
    image: torch.Tensor,
    method: str,
    low_percentile: float,
    high_percentile: float,
) -> torch.Tensor:
    if method == "none":
        return torch.nan_to_num(image).float()

    max_quantile_values = 2_000_000
    out_channels = []
    for channel_idx in range(image.shape[0]):
        channel = image[channel_idx]
        values = channel[torch.isfinite(channel) & (channel != 0)]
        if values.numel() < 16:
            out_channels.append(torch.nan_to_num(channel).float())
            continue
        quantile_values = values
        if values.numel() > max_quantile_values:
            step = max(1, values.numel() // max_quantile_values)
            quantile_values = values[::step]
        lo = torch.quantile(quantile_values, low_percentile / 100.0)
        hi = torch.quantile(quantile_values, high_percentile / 100.0)
        clipped = channel.clamp(min=lo.item(), max=hi.item())
        if method == "robust_minmax":
            denom = max(float((hi - lo).item()), 1e-6)
            normalized = (clipped - lo) / denom
        elif method == "robust_zscore":
            clipped_values = clipped[torch.isfinite(clipped) & (clipped != 0)]
            mean = clipped_values.mean()
            std = clipped_values.std(unbiased=False).clamp_min(1e-6)
            normalized = (clipped - mean) / std
        else:
            raise ValueError(f"Unknown mri_normalization: {method}")
        out_channels.append(torch.nan_to_num(normalized).float())
    return torch.stack(out_channels, dim=0)


class FOMOClsRegDataset(Dataset):
    """Classification dataset reader that loads raw FOMO NIfTI volumes on demand."""

    def __init__(
        self,
        task: str,
        files: Iterable[str],
        raw_root: str | None = None,
        resize_hw: tuple[int, int] | None = None,
        target_spacing_xy: tuple[float, float] | None = None,
        pad_hw: tuple[int, int] | None = None,
        mri_normalization: str = "robust_zscore",
        mri_low_percentile: float = 0.5,
        mri_high_percentile: float = 99.5,
    ):
        self.task = task
        self.files = list(files)
        self.raw_root = raw_root
        self.resize_hw = resize_hw
        self.target_spacing_xy = target_spacing_xy
        self.pad_hw = pad_hw
        self.mri_normalization = mri_normalization
        self.mri_low_percentile = mri_low_percentile
        self.mri_high_percentile = mri_high_percentile

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.files[index]
        image_paths, label_path = _resolve_raw_sample_paths(self.task, path, self.raw_root)
        image = _load_raw_volume_chwd(image_paths, self.target_spacing_xy)
        label = _load_raw_label(label_path)
        image = center_crop_or_pad_volume_chwd(image, self.pad_hw)
        image = resize_volume_chwd(image, self.resize_hw)
        image = normalize_volume_chwd(
            image,
            method=self.mri_normalization,
            low_percentile=self.mri_low_percentile,
            high_percentile=self.mri_high_percentile,
        )
        return {
            "file_path": str(image_paths[0]),
            "image": image,
            "label": label,
        }


class MeanSlicePool(nn.Module):
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        return (tokens * mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1) / denom


class MaxSlicePool(nn.Module):
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), torch.finfo(tokens.dtype).min)
        return tokens.max(dim=1).values


class AttentionSlicePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class TransformerSlicePool(nn.Module):
    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8, max_slices: int = 512):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, n_slices, dim = tokens.shape
        if n_slices + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"Transformer pooler supports at most {self.pos_embed.shape[1] - 1} slices.")
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed[:, : n_slices + 1]
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = torch.cat(
                [torch.ones(batch, 1, dtype=torch.bool, device=mask.device), mask],
                dim=1,
            )
            key_padding_mask = ~key_padding_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x[:, 0])


def make_slice_pooler(name: str, dim: int, transformer_depth: int, transformer_heads: int, max_slices: int) -> nn.Module:
    if name == "mean":
        return MeanSlicePool()
    if name == "max":
        return MaxSlicePool()
    if name == "attention":
        return AttentionSlicePool(dim)
    if name == "transformer":
        return TransformerSlicePool(dim, depth=transformer_depth, num_heads=transformer_heads, max_slices=max_slices)
    raise ValueError(f"Unknown slice_pool: {name}")


def _cache_file_name(
    source_path: str,
    slice_axis: int,
    max_slices: int | None,
    slice_size: int | None,
    patch_size: int,
    modality_pool: str,
    resize_hw: tuple[int, int] | None,
    slice_id: int | None = None,
) -> str:
    resize_tag = "orig" if resize_hw is None else f"{resize_hw[0]}x{resize_hw[1]}"
    slice_size_tag = "none" if slice_size is None else str(slice_size)
    signature = (
        f"{Path(source_path).stem}|axis={slice_axis}|max={max_slices}|slice={slice_size_tag}|"
        f"patch={patch_size}|modality={modality_pool}|resize={resize_tag}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:12]
    if slice_id is None:
        return f"{Path(source_path).stem}.flexict_cache.{digest}.pt"
    return f"slice_{slice_id:04d}.flexict_cache.{digest}.pt"


def _cache_subject_dir_name(source_path: str) -> str:
    path = Path(source_path)
    for parent in path.parents:
        if parent.name.startswith(("sub-", "sub_")):
            return parent.name
    if path.parent.name:
        return path.parent.name
    return path.stem


def _cache_subject_root(cache_root: Path, source_path: str) -> Path:
    return cache_root / _cache_subject_dir_name(source_path)


def _resolve_cache_root(args: argparse.Namespace, task: str) -> Path:
    if args.cache_path is not None:
        return Path(args.cache_path) / task
    return Path(args.processed_root) / task


def _load_cache_tensor(cache: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    value = cache[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Cache entry {key} must be a tensor.")
    return value.to(device)


def _load_cached_features(cache_file: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cache = torch.load(cache_file, map_location="cpu")
    return {
        "cls_tokens": _load_cache_tensor(cache, "cls_tokens", device),
        "patch_tokens": _load_cache_tensor(cache, "patch_tokens", device),
        "register_tokens": _load_cache_tensor(cache, "register_tokens", device),
    }


class FlexiCTSliceVolumeClassifier(nn.Module):
    """2D FlexiCT encoder + cross-slice pooling + classification head."""

    def __init__(
        self,
        dinov3_checkpoint: str,
        num_classes: int,
        slice_pool: str,
        modality_pool: str,
        slice_axis: int,
        slice_size: int | None,
        patch_size: int,
        max_slices: int | None,
        encoder_tuning: str,
        lora_r: int,
        lora_alpha: int,
        lora_targets: list[str],
        lora_dropout: float,
        transformer_depth: int,
        transformer_heads: int,
        device: torch.device,
    ):
        super().__init__()
        if encoder_tuning not in {"frozen", "lora", "full"}:
            raise ValueError(f"Unknown encoder_tuning: {encoder_tuning}")

        self.encoder = DINOv3SliceEncoder(
            checkpoint_path=dinov3_checkpoint,
            patch_size=patch_size,
            device=device,
        )
        self.encoder.eval()
        _set_patch_size(self.encoder.backbone, patch_size)
        if encoder_tuning == "lora":
            _apply_lora_to_backbone(
                self.encoder,
                r=lora_r,
                alpha=lora_alpha,
                target_modules=lora_targets,
                dropout=lora_dropout,
            )
        self.modality_pool = modality_pool
        self.slice_axis = slice_axis
        self.slice_size = slice_size
        self.max_slices = max_slices
        self.encoder_tuning = encoder_tuning
        self.freeze_encoder = encoder_tuning == "frozen"
        self.slice_pool_name = slice_pool

        probe_size = slice_size or 256
        probe = torch.zeros(1, 1, probe_size, probe_size, device=device)
        with torch.no_grad():
            probe_out = self.encoder(probe)
            self.token_dim = int(probe_out["cls_token"].shape[-1])
            self.patch_dim = int(probe_out["patch_tokens"].shape[-1])
        self.num_register_tokens = int(getattr(self.encoder.backbone, "n_storage_tokens", 0))

        pool_input_dim = self.token_dim
        if slice_pool == "patch":
            pool_input_dim = self.patch_dim
        elif slice_pool == "patch_cls":
            pool_input_dim = self.patch_dim + self.token_dim

        if slice_pool in {"patch", "cls", "patch_cls"}:
            self.slice_pool = MeanSlicePool()
        else:
            self.slice_pool = make_slice_pooler(
                slice_pool,
                pool_input_dim,
                transformer_depth=transformer_depth,
                transformer_heads=transformer_heads,
                max_slices=max_slices or 512,
            )
        self.head = nn.Linear(pool_input_dim, num_classes)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif encoder_tuning == "full":
            for param in self.encoder.parameters():
                param.requires_grad = True

    def _slice_selection_indices(self, n_slices: int, device: torch.device) -> torch.Tensor:
        if self.max_slices is not None and n_slices > self.max_slices:
            return torch.linspace(0, n_slices - 1, self.max_slices, device=device).round().long()
        return torch.arange(n_slices, device=device)

    def selected_slice_ids(self, volume: torch.Tensor) -> list[int]:
        slices = self._volume_to_slices(volume)
        indices = self._slice_selection_indices(slices.shape[1], device=slices.device)
        return indices.detach().cpu().tolist()

    def modality_views(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W,D], got {tuple(volume.shape)}")
        if self.modality_pool == "each":
            batch, channels, height, width, depth = volume.shape
            return volume.permute(0, 1, 2, 3, 4).reshape(batch * channels, 1, height, width, depth)
        if self.modality_pool == "mean":
            return volume.mean(dim=1, keepdim=True)
        if self.modality_pool == "first":
            return volume[:, :1]
        raise ValueError(f"Unknown modality_pool: {self.modality_pool}")

    def _volume_to_slices(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W,D], got {tuple(volume.shape)}")
        if self.slice_axis not in (2, 3, 4, -1):
            raise ValueError("--slice_axis is relative to [B,C,H,W,D] and must be 2, 3, 4, or -1.")
        axis = 4 if self.slice_axis == -1 else self.slice_axis

        if axis != 4:
            volume = volume.movedim(axis, 4)
        volume = self.modality_views(volume)

        return volume.permute(0, 4, 1, 2, 3).contiguous()

    def _encode_slice_batch(self, slices: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(slices)
        result = {
            "cls_tokens": out["cls_token"],
            "patch_tokens": out["patch_tokens"],
        }
        if self.num_register_tokens > 0:
            backbone_in = self.encoder.prepare_input_channels(slices)
            backbone_out = self.encoder.backbone(backbone_in, is_training=True)
            result["register_tokens"] = backbone_out["x_storage_tokens"]
        else:
            result["register_tokens"] = torch.empty(
                slices.shape[0],
                0,
                self.token_dim,
                device=slices.device,
                dtype=out["cls_token"].dtype,
            )
        return result

    def _stack_encoded_slices(self, encoded_batches: list[dict[str, torch.Tensor]], batch: int, n_slices: int) -> dict[str, torch.Tensor]:
        cls_tokens = torch.cat([item["cls_tokens"] for item in encoded_batches], dim=0).view(batch, n_slices, -1)
        patch_tokens = torch.cat([item["patch_tokens"] for item in encoded_batches], dim=0)
        register_tokens = torch.cat([item["register_tokens"] for item in encoded_batches], dim=0)

        spatial_dim = int(math.isqrt(patch_tokens.shape[1]))
        if spatial_dim * spatial_dim != patch_tokens.shape[1]:
            raise ValueError(f"Patch token count is not square: {patch_tokens.shape[1]}")

        patch_grid = patch_tokens.view(batch, n_slices, spatial_dim, spatial_dim, self.patch_dim)
        patch_grid = patch_grid.permute(0, 4, 2, 3, 1).contiguous()
        register_grid = register_tokens.view(batch, n_slices, self.num_register_tokens, self.token_dim)
        register_grid = register_grid.permute(0, 3, 2, 1).contiguous()
        return {
            "cls_tokens": cls_tokens,
            "patch_tokens": patch_grid,
            "register_tokens": register_grid,
        }

    def _compose_slice_features(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        cls_tokens = encoded["cls_tokens"]
        patch_mean = encoded["patch_tokens"].mean(dim=(2, 3)).transpose(1, 2).contiguous()
        if self.slice_pool_name == "patch":
            return patch_mean
        if self.slice_pool_name == "cls":
            return cls_tokens
        if self.slice_pool_name == "patch_cls":
            return torch.cat([patch_mean, cls_tokens], dim=-1)
        return cls_tokens

    def encode_slice_tokens(self, volume: torch.Tensor, slice_batch_size: int) -> dict[str, torch.Tensor]:
        slices = self._volume_to_slices(volume)
        idx = self._slice_selection_indices(slices.shape[1], device=slices.device)
        slices = slices.index_select(1, idx)

        batch, n_slices, channels, height, width = slices.shape
        flat = slices.view(batch * n_slices, channels, height, width)
        if self.slice_size is not None:
            flat = F.interpolate(flat, size=(self.slice_size, self.slice_size), mode="bilinear", align_corners=False)
        flat = self._normalize_slices(flat)

        encoded_batches = []
        if self.freeze_encoder:
            with torch.no_grad():
                for start in range(0, flat.shape[0], slice_batch_size):
                    encoded_batches.append(self._encode_slice_batch(flat[start : start + slice_batch_size]))
        else:
            for start in range(0, flat.shape[0], slice_batch_size):
                encoded_batches.append(self._encode_slice_batch(flat[start : start + slice_batch_size]))
        return self._stack_encoded_slices(encoded_batches, batch=batch, n_slices=n_slices)

    @staticmethod
    def _normalize_slices(slices: torch.Tensor) -> torch.Tensor:
        mean = slices.mean(dim=(-2, -1), keepdim=True)
        std = slices.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (slices - mean) / std

    def forward_features(
        self,
        volume: torch.Tensor | None = None,
        slice_batch_size: int = 32,
        cached_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if cached_features is None:
            if volume is None:
                raise ValueError("Either volume or cached_features must be provided.")
            cached_features = self.encode_slice_tokens(volume, slice_batch_size=slice_batch_size)
        slice_tokens = self._compose_slice_features(cached_features)
        return self.slice_pool(slice_tokens)

    def forward(
        self,
        volume: torch.Tensor | None = None,
        slice_batch_size: int = 32,
        cached_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        volume_token = self.forward_features(
            volume=volume,
            slice_batch_size=slice_batch_size,
            cached_features=cached_features,
        )
        return self.head(volume_token)


def _repeat_labels_for_modalities(labels: torch.Tensor, num_modalities: int) -> torch.Tensor:
    return labels.repeat_interleave(num_modalities)


def _pool_logits_across_modalities(logits: torch.Tensor, batch_size: int, num_modalities: int) -> torch.Tensor:
    return logits.view(batch_size, num_modalities, -1).mean(dim=1)


def _reshape_logits_by_modality(
    logits: torch.Tensor,
    batch_size: int,
    num_modalities: int,
) -> torch.Tensor:
    return logits.view(batch_size, num_modalities, -1)


def _prepare_cached_batch(cached_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ("cls_tokens", "patch_tokens", "register_tokens")
    return {key: torch.cat([item[key] for item in cached_list], dim=0) for key in keys}


def _load_cached_slices(cache_files: list[Path], device: torch.device) -> dict[str, torch.Tensor]:
    slice_caches = [_load_cached_features(path, device) for path in cache_files]
    cls_tokens = torch.cat([item["cls_tokens"] for item in slice_caches], dim=1)
    patch_tokens = torch.cat([item["patch_tokens"] for item in slice_caches], dim=-1)
    register_tokens = torch.cat([item["register_tokens"] for item in slice_caches], dim=-1)
    return {
        "cls_tokens": cls_tokens,
        "patch_tokens": patch_tokens,
        "register_tokens": register_tokens,
    }


def _cache_slice_files_for_sample(
    cache_root: Path,
    source_path: str,
    slice_ids: list[int],
    slice_axis: int,
    max_slices: int | None,
    slice_size: int,
    patch_size: int,
    modality_pool: str,
    resize_hw: tuple[int, int] | None,
) -> list[Path]:
    subject_root = _cache_subject_root(cache_root, source_path)
    return [
        subject_root
        / _cache_file_name(
            source_path=source_path,
            slice_axis=slice_axis,
            max_slices=max_slices,
            slice_size=slice_size,
            patch_size=patch_size,
            modality_pool=modality_pool,
            resize_hw=resize_hw,
            slice_id=slice_id,
        )
        for slice_id in slice_ids
    ]


def maybe_load_or_create_cached_features(
    args: argparse.Namespace,
    task: str,
    batch: dict[str, object],
    model: FlexiCTSliceVolumeClassifier,
    device: torch.device,
    slice_batch_size: int,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
    file_paths = batch["file_path"]
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    from_cache_root = Path(args.from_cache_path) / task if args.from_cache_path is not None else None
    cache_root = _resolve_cache_root(args, task)
    cache_root.mkdir(parents=True, exist_ok=True)
    resize_hw = _parse_hw_resize(args.resize_hw)
    images = batch["image"].to(device)
    active_cache_root = from_cache_root if from_cache_root is not None else cache_root
    cache_files = []
    modality_counts = []
    for idx, path in enumerate(file_paths):
        sample = images[idx : idx + 1]
        sample_views = model.modality_views(sample)
        modality_counts.append(sample_views.shape[0])
        sample_cache_files = []
        for modality_idx in range(sample_views.shape[0]):
            modality_path = f"{path}::modality_{modality_idx}"
            sample_cache_files.append(
                _cache_slice_files_for_sample(
                    cache_root=active_cache_root,
                    source_path=modality_path,
                    slice_ids=model.selected_slice_ids(sample_views[modality_idx : modality_idx + 1]),
                    slice_axis=args.slice_axis,
                    max_slices=args.max_slices,
                    slice_size=args.slice_size,
                    patch_size=args.patch_size,
                    modality_pool=args.modality_pool,
                    resize_hw=resize_hw,
                )
            )
        cache_files.append(sample_cache_files)

    if from_cache_root is not None:
        missing = [
            str(path)
            for sample_modalities in cache_files
            for sample_paths in sample_modalities
            for path in sample_paths
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing cached features in --from_cache_path: {missing[0]}")
        loaded = []
        for sample_modalities in cache_files:
            loaded.extend(_load_cached_slices(sample_paths, device) for sample_paths in sample_modalities)
        return None, _prepare_cached_batch(loaded)

    if not args.cache:
        return images, None

    uncached_indices = [
        idx
        for idx, sample_modalities in enumerate(cache_files)
        if any(not path.exists() for sample_paths in sample_modalities for path in sample_paths)
    ]
    if uncached_indices:
        uncached_views = []
        for sample_idx in uncached_indices:
            uncached_views.append(model.modality_views(images[sample_idx : sample_idx + 1]))
        flat_uncached = torch.cat(uncached_views, dim=0)
        encoded = model.encode_slice_tokens(flat_uncached, slice_batch_size=slice_batch_size)
        offset = 0
        for sample_idx in uncached_indices:
            sample_modalities = cache_files[sample_idx]
            for sample_paths in sample_modalities:
                for slice_offset, cache_file in enumerate(sample_paths):
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    sample_cache = {
                        "cls_tokens": encoded["cls_tokens"][offset : offset + 1, slice_offset : slice_offset + 1].detach().cpu(),
                        "patch_tokens": encoded["patch_tokens"][offset : offset + 1, :, :, :, slice_offset : slice_offset + 1].detach().cpu(),
                        "register_tokens": encoded["register_tokens"][offset : offset + 1, :, :, slice_offset : slice_offset + 1].detach().cpu(),
                    }
                    torch.save(sample_cache, cache_file)
                offset += 1

    cached = []
    for sample_modalities in cache_files:
        cached.extend(_load_cached_slices(sample_paths, device) for sample_paths in sample_modalities)
    return None, _prepare_cached_batch(cached)


def _auc(proba: np.ndarray, labels: np.ndarray) -> float:
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(labels, proba[:, 1]))
        return float(roc_auc_score(labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def evaluate(
    args: argparse.Namespace,
    task: str,
    model: FlexiCTSliceVolumeClassifier,
    loader: DataLoader,
    device: torch.device,
    slice_batch_size: int,
    debug_image_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    probs = []
    labels = []
    total_loss = 0.0
    n_seen = 0
    loss_fn = nn.CrossEntropyLoss()
    fused_correct = 0
    per_modality_correct: list[int] | None = None
    debug_saved = debug_image_path is None
    with torch.no_grad():
        for batch in loader:
            if not debug_saved and debug_image_path is not None:
                save_debug_slice_image(batch, debug_image_path)
                debug_saved = True
            x, cached_features = maybe_load_or_create_cached_features(
                args=args,
                task=task,
                batch=batch,
                model=model,
                device=device,
                slice_batch_size=slice_batch_size,
            )
            y = batch["label"].view(-1).to(device)
            if args.modality_pool == "each":
                num_modalities = int(batch["image"].shape[1])
                logits_each = model(x, slice_batch_size=slice_batch_size, cached_features=cached_features)
                logits_each = _reshape_logits_by_modality(
                    logits_each,
                    batch_size=y.numel(),
                    num_modalities=num_modalities,
                )
                logits = logits_each.mean(dim=1)
                pred_each = logits_each.argmax(dim=-1)
                if per_modality_correct is None:
                    per_modality_correct = [0 for _ in range(num_modalities)]
                for modality_idx in range(num_modalities):
                    per_modality_correct[modality_idx] += int(
                        (pred_each[:, modality_idx] == y).sum().item()
                    )
            else:
                logits = model(x, slice_batch_size=slice_batch_size, cached_features=cached_features)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * y.numel()
            n_seen += y.numel()
            fused_correct += int((logits.argmax(dim=-1) == y).sum().item())
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            labels.append(y.cpu().numpy())
    proba = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = proba.argmax(axis=1)
    metrics = {
        "loss": total_loss / max(1, n_seen),
        "auc": _auc(proba, y_true),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy_fused": fused_correct / max(1, n_seen),
    }
    if per_modality_correct is not None:
        for modality_idx, correct in enumerate(per_modality_correct):
            metrics[f"accuracy_modality_{modality_idx}"] = correct / max(1, n_seen)
    return metrics


def run_one_task(args: argparse.Namespace, task: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    task_dir = Path(args.processed_root) / task
    metadata = load_task_metadata(task_dir)
    n_classes = int(metadata["metadata"]["n_classes"])
    split_files = load_split_files(task_dir, args.train_split, args.test_split, args.fold)
    resize_hw = _parse_hw_resize(args.resize_hw)
    target_spacing_xy = _parse_spacing_xy(args.target_spacing_xy)
    if target_spacing_xy is None:
        target_spacing_xy = (1.0, 1.0)
    pad_hw = _parse_hw_resize(args.pad_hw)
    if pad_hw is None:
        pad_hw = (256, 256)

    train_ds = FOMOClsRegDataset(
        task,
        split_files.train,
        raw_root=args.raw_root,
        resize_hw=resize_hw,
        target_spacing_xy=target_spacing_xy,
        pad_hw=pad_hw,
        mri_normalization=args.mri_normalization,
        mri_low_percentile=args.mri_low_percentile,
        mri_high_percentile=args.mri_high_percentile,
    )
    val_ds = FOMOClsRegDataset(
        task,
        split_files.val,
        raw_root=args.raw_root,
        resize_hw=resize_hw,
        target_spacing_xy=target_spacing_xy,
        pad_hw=pad_hw,
        mri_normalization=args.mri_normalization,
        mri_low_percentile=args.mri_low_percentile,
        mri_high_percentile=args.mri_high_percentile,
    )
    test_ds = FOMOClsRegDataset(
        task,
        split_files.test,
        raw_root=args.raw_root,
        resize_hw=resize_hw,
        target_spacing_xy=target_spacing_xy,
        pad_hw=pad_hw,
        mri_normalization=args.mri_normalization,
        mri_low_percentile=args.mri_low_percentile,
        mri_high_percentile=args.mri_high_percentile,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    final_eval_loader = DataLoader(
        ConcatDataset([val_ds, test_ds]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    output_dir = Path(args.output_dir) / task / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = FlexiCTSliceVolumeClassifier(
        dinov3_checkpoint=args.dinov3_checkpoint,
        num_classes=n_classes,
        slice_pool=args.slice_pool,
        modality_pool=args.modality_pool,
        slice_axis=args.slice_axis,
        slice_size=args.slice_size,
        patch_size=args.patch_size,
        max_slices=args.max_slices,
        encoder_tuning=args.encoder_tuning,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        lora_dropout=args.lora_dropout,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = math.inf
    best_state = None
    best_epoch = 0

    print(
        f"\nTask={task} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={n_classes} device={device} encoder_tuning={args.encoder_tuning}"
    )
    print(
        f"slice_pool={args.slice_pool} modality_pool={args.modality_pool} "
        f"spacing_xy={target_spacing_xy} pad_hw={pad_hw} resize_hw={resize_hw} "
        f"slice_size={args.slice_size} max_slices={args.max_slices} "
        f"mri_norm={args.mri_normalization}[{args.mri_low_percentile},{args.mri_high_percentile}]"
    )

    for epoch in range(args.epochs):
        model.train()
        if args.encoder_tuning == "frozen":
            model.encoder.eval()
        train_loss = 0.0
        n_seen = 0
        progress = tqdm(train_loader, desc=f"{task} epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            x, cached_features = maybe_load_or_create_cached_features(
                args=args,
                task=task,
                batch=batch,
                model=model,
                device=device,
                slice_batch_size=args.slice_batch_size,
            )
            y = batch["label"].view(-1).to(device)
            if args.modality_pool == "each":
                num_modalities = int(batch["image"].shape[1])
                logits = model(x, slice_batch_size=args.slice_batch_size, cached_features=cached_features)
                loss = loss_fn(logits, _repeat_labels_for_modalities(y, num_modalities))
            else:
                logits = model(x, slice_batch_size=args.slice_batch_size, cached_features=cached_features)
                loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * y.numel()
            n_seen += y.numel()
            progress.set_postfix(loss=train_loss / max(1, n_seen))

        val_metrics = evaluate(
            args,
            task,
            model,
            val_loader,
            device=device,
            slice_batch_size=args.slice_batch_size,
            debug_image_path=output_dir / "val_debug_slice.png",
        )
        val_message = (
            f"epoch={epoch + 1:03d} train_loss={train_loss / max(1, n_seen):.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_bal_acc={val_metrics['bal_acc']:.4f} val_acc_fused={val_metrics['accuracy_fused']:.4f}"
        )
        if args.modality_pool == "each":
            per_modality_parts = []
            modality_idx = 0
            while f"accuracy_modality_{modality_idx}" in val_metrics:
                per_modality_parts.append(
                    f"val_acc_m{modality_idx}={val_metrics[f'accuracy_modality_{modality_idx}']:.4f}"
                )
                modality_idx += 1
            if per_modality_parts:
                val_message += " " + " ".join(per_modality_parts)
        print(val_message)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(args, task, model, test_loader, device=device, slice_batch_size=args.slice_batch_size)
    final_metrics = evaluate(
        args,
        task,
        model,
        final_eval_loader,
        device=device,
        slice_batch_size=args.slice_batch_size,
    )
    test_message = (
        f"[final val+test] {task} loss={final_metrics['loss']:.4f} auc={final_metrics['auc']:.4f} "
        f"bal_acc={final_metrics['bal_acc']:.4f} acc_fused={final_metrics['accuracy_fused']:.4f}"
    )
    if args.modality_pool == "each":
        per_modality_parts = []
        modality_idx = 0
        while f"accuracy_modality_{modality_idx}" in final_metrics:
            per_modality_parts.append(
                f"acc_m{modality_idx}={final_metrics[f'accuracy_modality_{modality_idx}']:.4f}"
            )
            modality_idx += 1
        if per_modality_parts:
            test_message += " " + " ".join(per_modality_parts)
    print(test_message)

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "task": task,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "final_metrics": final_metrics,
        },
        output_dir / "best.pt",
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test": test_metrics,
                "final_val_test": final_metrics,
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", choices=[*DEFAULT_TASKS, "all"])
    parser.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--train_split", default=DEFAULT_SPLIT)
    parser.add_argument("--test_split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dinov3_checkpoint",
        default=None,
        help="Path to the pretrained DINOv3 2D checkpoint. If omitted, uses DINOV3_2D_CHECKPOINT.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--slice_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--resize_hw", default=None, help="Resize only H,W as 'H,W'; Z is never resized.")
    parser.add_argument("--target_spacing_xy", default=None, help="Optional X,Y spacing in mm as 'X,Y' before padding/resizing.")
    parser.add_argument("--pad_hw", default=None, help="Optional zero-pad target H,W as 'H,W' after respacing.")
    parser.add_argument("--slice_size", type=int, default=None)
    parser.add_argument("--slice_axis", type=int, default=-1, help="Slice axis in [B,C,H,W,D]; default -1 means D.")
    parser.add_argument("--max_slices", type=int, default=None, help="Keep all slices by default; optionally subsample to this many slices.")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--modality_pool", default="mean", choices=["mean", "first", "each"])
    parser.add_argument("--mri_normalization", choices=["robust_zscore", "robust_minmax", "none"], default="robust_zscore")
    parser.add_argument("--mri_low_percentile", type=float, default=0.5)
    parser.add_argument("--mri_high_percentile", type=float, default=99.5)
    parser.add_argument("--slice_pool", default="attention", choices=["mean", "max", "attention", "transformer", "patch", "cls", "patch_cls"])
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--cache", action="store_true", help="Cache backbone cls/patch/register features per volume.")
    parser.add_argument("--cache_path", default=None, help="Where to save caches. Defaults to <processed_root>/<task>.")
    parser.add_argument("--from_cache_path", default=None, help="Load cached features from this root instead of running the backbone.")
    parser.add_argument("--unfreeze_encoder", action="store_true", help="Train the full FlexiCT 2D encoder.")
    parser.add_argument("--lora_encoder", action="store_true", help="Train LoRA adapters on the FlexiCT 2D backbone.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_targets", default="qkv,proj")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.dinov3_checkpoint = resolve_dinov3_checkpoint(args.dinov3_checkpoint)

    if args.unfreeze_encoder and args.lora_encoder:
        parser.error("Use only one of --unfreeze_encoder or --lora_encoder.")
    args.encoder_tuning = "full" if args.unfreeze_encoder else "lora" if args.lora_encoder else "frozen"

    if args.from_cache_path is not None and args.encoder_tuning != "frozen":
        parser.error("--from_cache_path requires a frozen encoder because cached features bypass backbone training.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_one_task(args, task)


if __name__ == "__main__":
    main()
