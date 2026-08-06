"""Fine-tune FOMO 3D regression with a FlexiCT 2D slice encoder.

This is a draft interface for using the 2D Flexi_CT backbone on 3D FOMO
regression tasks. Each volume is converted into a sequence of 2D slices,
Flexi_CT_2D produces one CLS token per slice, and a configurable pooling module
merges the slice tokens into one volume-level token for regression.

Supported tasks:
  - REGR002_FOMO26_BrainAge

The default keeps the FlexiCT encoder frozen and trains only the cross-slice
pooler plus linear regression head. Pass --lora_encoder to train LoRA adapters in
the 2D backbone, or --unfreeze_encoder for full fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

DEFAULT_PROCESSED_ROOT = (
    "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/"
    "FOMO_Challenge/processed_data"
)
DEFAULT_OUTPUT_DIR = str(ROOT / "results" / "3d_regression" / "fomo_slice_reg")
DEFAULT_TASKS = ["REGR002_FOMO26_BrainAge"]
DEFAULT_SPLIT = "split_80_10_10"
DEFAULT_TEST_SPLIT = "TEST_80_10_10"
SEED = 42


@dataclass(frozen=True)
class SplitFiles:
    train: list[str]
    val: list[str]
    test: list[str]


@dataclass
class StageTimer:
    data: float = 0.0
    backbone: float = 0.0
    slice_pool: float = 0.0
    optimization: float = 0.0
    steps: int = 0

    def add(
        self,
        *,
        data: float = 0.0,
        backbone: float = 0.0,
        slice_pool: float = 0.0,
        optimization: float = 0.0,
    ) -> None:
        self.data += data
        self.backbone += backbone
        self.slice_pool += slice_pool
        self.optimization += optimization
        self.steps += 1

    def summary(self) -> str:
        denom = max(1, self.steps)
        return (
            f"data={self.data / denom:.4f}s "
            f"backbone={self.backbone / denom:.4f}s "
            f"slice_pool={self.slice_pool / denom:.4f}s "
            f"optimization={self.optimization / denom:.4f}s"
        )


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now(device: torch.device) -> float:
    _sync_device(device)
    return time.perf_counter()


def _set_patch_size(model: torch.nn.Module, patch_size: int) -> None:
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
    if isinstance(sample, dict):
        image = sample.get("image")
        label = sample.get("label")
    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
        image, label = sample[0], sample[1]
    else:
        raise TypeError(f"Unsupported sample type from {path}: {type(sample)!r}")
    if image is None or label is None:
        raise ValueError(f"Sample from {path} is missing image/label.")
    image = torch.as_tensor(image)
    label = torch.as_tensor(label)
    return image, label


class FOMOClsRegDataset(Dataset):
    def __init__(self, files: list[str], resize_hw: tuple[int, int] | None = None):
        self.files = files
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.files[index]
        sample = torch.load(path, map_location="cpu")
        image, label = _as_image_label(sample, path)
        image = image.float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected 4D [C,H,W,D] image tensor from {path}, got {tuple(image.shape)}")
        if self.resize_hw is not None:
            image = image.permute(3, 0, 1, 2)
            image = F.interpolate(image, size=self.resize_hw, mode="bilinear", align_corners=False)
            image = image.permute(1, 2, 3, 0).contiguous()
        return {
            "image": image,
            "label": label.float(),
            "file_path": path,
        }


def _parse_hw_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"", "none", "native"}:
        return None
    for sep in ("x", ","):
        if sep in value:
            h_str, w_str = value.split(sep, 1)
            return int(h_str), int(w_str)
    size = int(value)
    return size, size


def _cache_subject_root(cache_root: Path, source_path: str) -> Path:
    source = Path(source_path)
    stem = source.stem
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{stem}_{digest}"


def _cache_file_name(
    source_path: str,
    slice_axis: int,
    max_slices: int | None,
    slice_size: int,
    patch_size: int,
    modality_pool: str,
    resize_hw: tuple[int, int] | None,
    slice_id: int,
) -> str:
    resize_tag = "native" if resize_hw is None else f"{resize_hw[0]}x{resize_hw[1]}"
    max_slices_tag = "all" if max_slices is None else str(max_slices)
    digest_input = json.dumps(
        {
            "source_path": str(source_path),
            "slice_axis": slice_axis,
            "max_slices": max_slices,
            "slice_size": slice_size,
            "patch_size": patch_size,
            "modality_pool": modality_pool,
            "resize_hw": resize_hw,
            "slice_id": slice_id,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    return (
        f"slice_{slice_id:04d}_axis{slice_axis}_max{max_slices_tag}_size{slice_size}_"
        f"patch{patch_size}_{modality_pool}_{resize_tag}_{digest}.pt"
    )


def _resolve_cache_root(args: argparse.Namespace, task: str) -> Path:
    if args.cache_path is not None:
        return Path(args.cache_path) / task
    return Path(args.processed_root) / task / "cache"


def _load_cache_tensor(cache: dict, key: str, device: torch.device) -> torch.Tensor:
    tensor = cache.get(key)
    if tensor is None:
        raise KeyError(f"Cached feature file is missing key '{key}'.")
    return tensor.to(device)


def _load_cached_features(cache_file: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cache = torch.load(cache_file, map_location="cpu")
    return {
        "cls_tokens": _load_cache_tensor(cache, "cls_tokens", device),
        "patch_tokens": _load_cache_tensor(cache, "patch_tokens", device),
        "register_tokens": _load_cache_tensor(cache, "register_tokens", device),
    }


class MeanSlicePool(nn.Module):
    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        return slice_tokens.mean(dim=1)


class TransformerSlicePool(nn.Module):
    def __init__(self, token_dim: int, depth: int, heads: int, max_slices: int):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos = nn.Parameter(torch.zeros(1, max_slices + 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        batch, n_slices, _ = slice_tokens.shape
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, slice_tokens], dim=1)
        x = x + self.pos[:, : n_slices + 1]
        x = self.encoder(x)
        return x[:, 0]


class AttentionSlicePool(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(slice_tokens).squeeze(-1), dim=1)
        return torch.einsum("bs,bsd->bd", weights, slice_tokens)


def make_slice_pooler(name: str, token_dim: int, transformer_depth: int, transformer_heads: int, max_slices: int) -> nn.Module:
    name = name.lower()
    if name == "mean":
        return MeanSlicePool()
    if name == "transformer":
        return TransformerSlicePool(token_dim, transformer_depth, transformer_heads, max_slices)
    if name == "attention":
        return AttentionSlicePool(token_dim)
    raise ValueError(f"Unsupported --slice_pool '{name}'.")


class FlexiCTSliceVolumeRegressor(nn.Module):
    """2D FlexiCT encoder + cross-slice pooling + regression head."""

    def __init__(
        self,
        checkpoint: str | None,
        num_outputs: int,
        slice_pool: str,
        modality_pool: str,
        slice_axis: int,
        slice_size: int,
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

        self.encoder = Flexi_CT_2D(checkpoint_path=resolve_flexict_checkpoint("2d", checkpoint), device=device)
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

        probe = torch.zeros(1, 1, slice_size, slice_size, device=device)
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
        self.head = nn.Linear(pool_input_dim, num_outputs)

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

    def _volume_to_slices(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W,D], got {tuple(volume.shape)}")
        if self.slice_axis not in (2, 3, 4, -1):
            raise ValueError("--slice_axis is relative to [B,C,H,W,D] and must be 2, 3, 4, or -1.")
        axis = 4 if self.slice_axis == -1 else self.slice_axis

        if axis != 4:
            volume = volume.movedim(axis, 4)

        if self.modality_pool == "mean":
            volume = volume.mean(dim=1, keepdim=True)
        elif self.modality_pool == "first":
            volume = volume[:, :1]
        elif self.modality_pool == "each":
            batch, channels, height, width, depth = volume.shape
            volume = volume.reshape(batch, 1, height, width, channels * depth)
        else:
            raise ValueError(f"Unknown modality_pool: {self.modality_pool}")

        return volume.permute(0, 4, 1, 2, 3).contiguous()

    def _encode_slice_batch(self, slices: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(slices)
        result = {
            "cls_tokens": out["cls_token"],
            "patch_tokens": out["patch_tokens"],
        }
        if self.num_register_tokens > 0:
            backbone_out = self.encoder.backbone(slices, is_training=True)
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

    def forward_with_profile(
        self,
        volume: torch.Tensor | None = None,
        slice_batch_size: int = 32,
        cached_features: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = self.head.weight.device
        timings = {"backbone": 0.0, "slice_pool": 0.0}

        if cached_features is None:
            if volume is None:
                raise ValueError("Either volume or cached_features must be provided.")
            start = _now(device)
            cached_features = self.encode_slice_tokens(volume, slice_batch_size=slice_batch_size)
            timings["backbone"] = _now(device) - start

        start = _now(device)
        slice_tokens = self._compose_slice_features(cached_features)
        volume_token = self.slice_pool(slice_tokens)
        pred = self.head(volume_token)
        timings["slice_pool"] = _now(device) - start
        return pred, timings

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
    model: FlexiCTSliceVolumeRegressor,
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
    cache_files = [
        _cache_slice_files_for_sample(
            cache_root=active_cache_root,
            source_path=path,
            slice_ids=model.selected_slice_ids(images[idx : idx + 1]),
            slice_axis=args.slice_axis,
            max_slices=args.max_slices,
            slice_size=args.slice_size,
            patch_size=args.patch_size,
            modality_pool=args.modality_pool,
            resize_hw=resize_hw,
        )
        for idx, path in enumerate(file_paths)
    ]

    if from_cache_root is not None:
        missing = [str(path) for sample_paths in cache_files for path in sample_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing cached features in --from_cache_path: {missing[0]}")
        return None, _prepare_cached_batch([_load_cached_slices(sample_paths, device) for sample_paths in cache_files])

    if not args.cache:
        return images, None

    uncached_indices = [idx for idx, sample_paths in enumerate(cache_files) if any(not path.exists() for path in sample_paths)]
    if uncached_indices:
        encoded = model.encode_slice_tokens(images[uncached_indices], slice_batch_size=slice_batch_size)
        for offset, sample_idx in enumerate(uncached_indices):
            sample_paths = cache_files[sample_idx]
            for slice_offset, cache_file in enumerate(sample_paths):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                sample_cache = {
                    "cls_tokens": encoded["cls_tokens"][offset : offset + 1, slice_offset : slice_offset + 1].detach().cpu(),
                    "patch_tokens": encoded["patch_tokens"][offset : offset + 1, :, :, :, slice_offset : slice_offset + 1].detach().cpu(),
                    "register_tokens": encoded["register_tokens"][offset : offset + 1, :, :, slice_offset : slice_offset + 1].detach().cpu(),
                }
                torch.save(sample_cache, cache_file)

    cached = [_load_cached_slices(sample_paths, device) for sample_paths in cache_files]
    return None, _prepare_cached_batch(cached)


def regression_metrics(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(labels.shape)
    mse = float(mean_squared_error(labels, pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, pred)),
        "r2": float(r2_score(labels, pred)) if len(labels) > 1 else float("nan"),
    }


def evaluate(
    args: argparse.Namespace,
    task: str,
    model: FlexiCTSliceVolumeRegressor,
    loader: DataLoader,
    device: torch.device,
    slice_batch_size: int,
) -> dict[str, float]:
    model.eval()
    preds = []
    labels = []
    total_loss = 0.0
    n_seen = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            x, cached_features = maybe_load_or_create_cached_features(
                args=args,
                task=task,
                batch=batch,
                model=model,
                device=device,
                slice_batch_size=slice_batch_size,
            )
            y = batch["label"].to(device).float()
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(x, slice_batch_size=slice_batch_size, cached_features=cached_features)
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * y.shape[0]
            n_seen += y.shape[0]
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    metrics = regression_metrics(y_pred, y_true)
    metrics["loss"] = total_loss / max(1, n_seen)
    return metrics


def run_one_task(args: argparse.Namespace, task: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    task_dir = Path(args.processed_root) / task
    metadata = load_task_metadata(task_dir)
    n_outputs = int(metadata["metadata"].get("n_classes", 1))
    split_files = load_split_files(task_dir, args.train_split, args.test_split, args.fold)
    resize_hw = _parse_hw_resize(args.resize_hw)
    output_dir = Path(args.output_dir) / task / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = output_dir / "last.pt"

    train_ds = FOMOClsRegDataset(split_files.train, resize_hw=resize_hw)
    val_ds = FOMOClsRegDataset(split_files.val, resize_hw=resize_hw)
    test_ds = FOMOClsRegDataset(split_files.test, resize_hw=resize_hw)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FlexiCTSliceVolumeRegressor(
        checkpoint=args.checkpoint,
        num_outputs=n_outputs,
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
    loss_fn = nn.MSELoss()
    best_val_loss = math.inf
    best_state = None
    start_epoch = 0

    if args.resume and last_ckpt_path.exists():
        checkpoint = torch.load(last_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        best_state_raw = checkpoint.get("best_state")
        if best_state_raw is not None:
            best_state = {k: v.detach().cpu() for k, v in best_state_raw.items()}
        print(f"[resume] Loaded {last_ckpt_path} and continuing from epoch {start_epoch + 1}/{args.epochs}")

    print(
        f"\nTask={task} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"outputs={n_outputs} device={device} encoder_tuning={args.encoder_tuning}"
    )
    print(
        f"slice_pool={args.slice_pool} modality_pool={args.modality_pool} "
        f"resize_hw={resize_hw} slice_size={args.slice_size} max_slices={args.max_slices}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.encoder_tuning == "frozen":
            model.encoder.eval()
        train_loss = 0.0
        n_seen = 0
        stage_timer = StageTimer()
        progress = tqdm(train_loader, desc=f"{task} epoch {epoch + 1}/{args.epochs}", leave=False)
        data_end = _now(device)
        for step, batch in enumerate(progress, start=1):
            data_ready = _now(device)
            data_time = data_ready - data_end
            x, cached_features = maybe_load_or_create_cached_features(
                args=args,
                task=task,
                batch=batch,
                model=model,
                device=device,
                slice_batch_size=args.slice_batch_size,
            )
            data_loaded = _now(device)
            data_time += data_loaded - data_ready
            y = batch["label"].to(device).float()
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred, profile = model.forward_with_profile(
                x,
                slice_batch_size=args.slice_batch_size,
                cached_features=cached_features,
            )
            loss = loss_fn(pred, y)

            opt_start = _now(device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimization_time = _now(device) - opt_start

            train_loss += float(loss.item()) * y.shape[0]
            n_seen += y.shape[0]
            stage_timer.add(
                data=data_time,
                backbone=profile["backbone"],
                slice_pool=profile["slice_pool"],
                optimization=optimization_time,
            )
            progress.set_postfix(loss=train_loss / max(1, n_seen), data=f"{data_time:.3f}s", bb=f"{profile['backbone']:.3f}s", pool=f"{profile['slice_pool']:.3f}s", opt=f"{optimization_time:.3f}s")
            if args.profile_print_freq > 0 and step % args.profile_print_freq == 0:
                print(
                    f"[profile][{task}][epoch {epoch + 1:03d} step {step:04d}] "
                    f"data={data_time:.4f}s backbone={profile['backbone']:.4f}s "
                    f"slice_pool={profile['slice_pool']:.4f}s optimization={optimization_time:.4f}s"
                )
            data_end = _now(device)

        val_metrics = evaluate(args, task, model, val_loader, device=device, slice_batch_size=args.slice_batch_size)
        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss / max(1, n_seen):.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_r2={val_metrics['r2']:.4f}"
        )
        print(f"[profile][{task}][epoch {epoch + 1:03d} avg] {stage_timer.summary()}")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_state": best_state,
                "args": vars(args),
                "task": task,
            },
            last_ckpt_path,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(args, task, model, test_loader, device=device, slice_batch_size=args.slice_batch_size)
    print(
        f"[test] {task} loss={test_metrics['loss']:.4f} mae={test_metrics['mae']:.4f} "
        f"rmse={test_metrics['rmse']:.4f} r2={test_metrics['r2']:.4f}"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "task": task,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
        },
        output_dir / "best.pt",
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train_split", default=DEFAULT_SPLIT)
    parser.add_argument("--test_split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--slice_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--slice_pool", default="cls", choices=["cls", "patch", "patch_cls", "mean", "attention", "transformer"])
    parser.add_argument("--modality_pool", default="mean", choices=["mean", "first", "each"])
    parser.add_argument("--slice_axis", type=int, default=-1)
    parser.add_argument("--slice_size", type=int, default=224)
    parser.add_argument("--resize_hw", default=None, help="Optional HxW resize before slicing (e.g. 256x256).")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache backbone cls/patch/register features per volume.")
    parser.add_argument("--cache_path", default=None, help="Write cached features under this root (defaults to processed_root/<task>/cache).")
    parser.add_argument("--from_cache_path", default=None, help="Load cached features from this root instead of running the backbone.")
    parser.add_argument("--profile_print_freq", type=int, default=1, help="Print training profile every N steps. Set 0 to disable per-step profile lines.")
    parser.add_argument("--resume", action="store_true", help="Resume training from output_dir/<task>/fold_<fold>/last.pt if it exists.")
    parser.add_argument("--lora_encoder", action="store_true", help="Train LoRA adapters on the FlexiCT 2D backbone.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_targets", default="qkv")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--unfreeze_encoder", action="store_true", help="Train all FlexiCT 2D encoder parameters.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.lora_encoder and args.unfreeze_encoder:
        raise ValueError("Use either --lora_encoder or --unfreeze_encoder, not both.")
    args.encoder_tuning = "frozen"
    if args.unfreeze_encoder:
        args.encoder_tuning = "full"
    elif args.lora_encoder:
        args.encoder_tuning = "lora"
    if args.from_cache_path is not None and args.encoder_tuning != "frozen":
        raise ValueError("--from_cache_path requires a frozen encoder because cached features bypass backbone training.")
    if args.cache and args.encoder_tuning != "frozen":
        raise ValueError("--cache requires a frozen encoder because cached features bypass backbone training.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_args(args)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tasks = args.task if isinstance(args.task, list) else [args.task]
    for task in tasks:
        run_one_task(args, task)


if __name__ == "__main__":
    main()
