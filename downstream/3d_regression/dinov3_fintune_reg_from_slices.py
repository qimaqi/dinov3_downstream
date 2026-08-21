#!/usr/bin/env python3
"""Fine-tune FOMO 3D regression with a DINOv3 2D slice encoder.

This mirrors the DINOv3 slice-classification pipeline, but replaces the final
classification head with a scalar regression head and reads Task 3 raw NIfTI
data directly from the challenge dataset layout.

Supported tasks:
  - REGR002_FOMO26_BrainAge

The default keeps the DINOv3 encoder frozen and trains only the cross-slice
pooler plus linear regression head. Pass --lora_encoder to train LoRA adapters
in the 2D backbone, or --unfreeze_encoder for full fine-tuning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_MOD_PATH = ROOT / "downstream" / "3d_classify" / "dinov3_finetune_cls_from_slices.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("dinov3_finetune_cls_from_slices", TRAIN_MOD_PATH)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError(f"Cannot load shared DINOv3 slice module from {TRAIN_MOD_PATH}")
cls_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules["dinov3_finetune_cls_from_slices"] = cls_mod
TRAIN_SPEC.loader.exec_module(cls_mod)

DEFAULT_RAW_ROOT = (
    "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/"
    "FOMO_Tasks/Task_3/Task_3"
)
DEFAULT_OUTPUT_DIR = str(ROOT / "results" / "3d_regression" / "dinov3_slice_reg")
DEFAULT_SPLIT_ROOT = str(ROOT / "results" / "3d_regression" / "_splits")
DEFAULT_TASKS = ["REGR002_FOMO26_BrainAge"]
DEFAULT_SPLIT = "split_5fold_cv"
DEFAULT_TEST_SPLIT = "test_5fold_cv"
SEED = 42

RAW_TASK_CONFIG = {
    "REGR002_FOMO26_BrainAge": {
        "image_root": Path(DEFAULT_RAW_ROOT) / "preprocessed",
        "label_root": Path(DEFAULT_RAW_ROOT) / "labels",
        "modalities": ("t1w.nii.gz",),
        "label_file": "labels.txt",
        "num_outputs": 1,
    }
}


@dataclass(frozen=True)
class SplitFiles:
    train: list[str]
    val: list[str]
    test: list[str]


def resolve_dinov3_checkpoint(explicit_checkpoint: str | None = None) -> str:
    return cls_mod.resolve_dinov3_checkpoint(explicit_checkpoint)


def _discover_cases(task: str, raw_root: str | None = None) -> list[str]:
    config = RAW_TASK_CONFIG[task]
    base_root = Path(raw_root) if raw_root is not None else Path(DEFAULT_RAW_ROOT)
    image_root = base_root / config["image_root"].relative_to(DEFAULT_RAW_ROOT)
    label_root = base_root / config["label_root"].relative_to(DEFAULT_RAW_ROOT)
    label_cases = sorted(
        f"{label_path.parent.parent.name}/{label_path.parent.name}"
        for label_path in label_root.rglob(config["label_file"])
    )
    filtered = []
    for case in label_cases:
        subject, session = case.split("/")
        image_paths = [image_root / subject / session / modality for modality in config["modalities"]]
        if all(path.exists() for path in image_paths):
            filtered.append(case)
    return filtered


def _make_5fold_splits(cases: list[str], seed: int) -> tuple[list[dict[str, list[str]]], list[dict[str, list[str]]]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(cases), dtype=object)
    rng.shuffle(shuffled)
    shared_test_cases = np.array_split(shuffled, 5)[0].tolist()
    train_val_pool = shuffled[len(shared_test_cases) :]
    fold_arrays = [fold.tolist() for fold in np.array_split(train_val_pool, 5)]

    train_val_folds = []
    test_folds = []
    for fold_idx in range(5):
        val_cases = fold_arrays[fold_idx]
        train_cases = []
        for idx, fold_cases in enumerate(fold_arrays):
            if idx != fold_idx:
                train_cases.extend(fold_cases)
        train_val_folds.append({"train": sorted(train_cases), "val": sorted(val_cases)})
        test_folds.append({"test": sorted(shared_test_cases)})
    return train_val_folds, test_folds


def ensure_split_files(task: str, split_root: str, split_name: str, test_split_name: str, raw_root: str | None, seed: int) -> tuple[Path, Path]:
    task_split_root = Path(split_root) / task
    task_split_root.mkdir(parents=True, exist_ok=True)
    split_path = task_split_root / f"{split_name}.json"
    test_split_path = task_split_root / f"{test_split_name}.json"
    if split_path.exists() and test_split_path.exists():
        return split_path, test_split_path

    cases = _discover_cases(task, raw_root=raw_root)
    if len(cases) != 494:
        print(f"[warn] discovered {len(cases)} cases for {task}; expected about 494.")
    train_val_folds, test_folds = _make_5fold_splits(cases, seed=seed)
    split_path.write_text(json.dumps(train_val_folds, indent=2), encoding="utf-8")
    test_split_path.write_text(json.dumps(test_folds, indent=2), encoding="utf-8")
    return split_path, test_split_path


def load_split_files(
    split_root: str,
    task: str,
    split_name: str,
    test_split_name: str,
    fold: int,
    raw_root: str | None,
    seed: int,
) -> SplitFiles:
    split_path, test_split_path = ensure_split_files(task, split_root, split_name, test_split_name, raw_root, seed)
    with open(split_path, "r", encoding="utf-8") as f:
        folds = json.load(f)
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_folds = json.load(f)

    if not isinstance(folds, list) or not isinstance(test_folds, list):
        raise ValueError("Expected split JSON files to contain a list of folds.")
    if fold < 0 or fold >= len(folds) or fold >= len(test_folds):
        raise ValueError(f"Fold {fold} is out of range for 5-fold split files.")

    fold_data = folds[fold]
    test_data = test_folds[fold]
    return SplitFiles(
        train=list(fold_data["train"]),
        val=list(fold_data["val"]),
        test=list(test_data["test"]),
    )


def _resolve_raw_sample_paths(task: str, case_id: str, raw_root: str | None) -> tuple[list[Path], Path]:
    config = RAW_TASK_CONFIG[task]
    subject, session = case_id.split("/")
    base_root = Path(raw_root) if raw_root is not None else Path(DEFAULT_RAW_ROOT)
    image_root = base_root / config["image_root"].relative_to(DEFAULT_RAW_ROOT)
    label_root = base_root / config["label_root"].relative_to(DEFAULT_RAW_ROOT)
    image_paths = [image_root / subject / session / modality for modality in config["modalities"]]
    label_path = label_root / subject / session / config["label_file"]
    return image_paths, label_path


def _load_raw_label(label_path: Path) -> torch.Tensor:
    return torch.tensor(float(label_path.read_text().strip()), dtype=torch.float32)


class FOMORegressionDataset(Dataset):
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
        case_id = self.files[index]
        image_paths, label_path = _resolve_raw_sample_paths(self.task, case_id, self.raw_root)
        image = cls_mod._load_raw_volume_chwd(image_paths, self.target_spacing_xy)
        label = _load_raw_label(label_path)
        image = cls_mod.center_crop_or_pad_volume_chwd(image, self.pad_hw)
        image = cls_mod.resize_volume_chwd(image, self.resize_hw)
        image = cls_mod.normalize_volume_chwd(
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


class DINOv3SliceVolumeRegressor(nn.Module):
    """DINOv3 encoder + cross-slice pooling + scalar regression head."""

    def __init__(
        self,
        dinov3_checkpoint: str,
        num_outputs: int,
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

        self.encoder = cls_mod.DINOv3SliceEncoder(
            checkpoint_path=dinov3_checkpoint,
            patch_size=patch_size,
            device=device,
        )
        self.encoder.eval()
        cls_mod._set_patch_size(self.encoder.backbone, patch_size)
        if encoder_tuning == "lora":
            cls_mod._apply_lora_to_backbone(
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
            self.slice_pool = cls_mod.MeanSlicePool()
        else:
            self.slice_pool = cls_mod.make_slice_pooler(
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

    def _num_views(self, volume: torch.Tensor) -> int:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W,D], got {tuple(volume.shape)}")
        if self.modality_pool == "first":
            return 1
        if self.modality_pool in {"mean", "each"}:
            return int(volume.shape[1])
        raise ValueError(f"Unknown modality_pool: {self.modality_pool}")

    def modality_views(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W,D], got {tuple(volume.shape)}")
        if self.modality_pool in {"mean", "each"}:
            batch, channels, height, width, depth = volume.shape
            return volume.permute(0, 1, 2, 3, 4).reshape(batch * channels, 1, height, width, depth)
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
        return {
            "cls_tokens": out["cls_token"],
            "patch_tokens": out["patch_tokens"],
            "register_tokens": torch.empty(
                slices.shape[0],
                0,
                self.token_dim,
                device=slices.device,
                dtype=out["cls_token"].dtype,
            ),
        }

    def _stack_encoded_slices(self, encoded_batches: list[dict[str, torch.Tensor]], batch: int, n_slices: int) -> dict[str, torch.Tensor]:
        cls_tokens = torch.cat([item["cls_tokens"] for item in encoded_batches], dim=0).view(batch, n_slices, -1)
        patch_tokens = torch.cat([item["patch_tokens"] for item in encoded_batches], dim=0)
        spatial_dim = int(math.isqrt(patch_tokens.shape[1]))
        if spatial_dim * spatial_dim != patch_tokens.shape[1]:
            raise ValueError(f"Patch token count is not square: {patch_tokens.shape[1]}")
        patch_grid = patch_tokens.view(batch, n_slices, spatial_dim, spatial_dim, self.patch_dim)
        patch_grid = patch_grid.permute(0, 4, 2, 3, 1).contiguous()
        register_grid = torch.empty(batch, self.token_dim, 0, n_slices, device=cls_tokens.device, dtype=cls_tokens.dtype)
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
        num_views = self._num_views(volume)
        slices = self._volume_to_slices(volume)
        idx = self._slice_selection_indices(slices.shape[1], device=slices.device)
        slices = slices.index_select(1, idx)
        batch, n_slices, channels, height, width = slices.shape
        flat = slices.view(batch * n_slices, channels, height, width)
        if self.slice_size is not None:
            flat = nn.functional.interpolate(flat, size=(self.slice_size, self.slice_size), mode="bilinear", align_corners=False)

        encoded_batches = []
        if self.freeze_encoder:
            with torch.no_grad():
                for start in range(0, flat.shape[0], slice_batch_size):
                    encoded_batches.append(self._encode_slice_batch(flat[start : start + slice_batch_size]))
        else:
            for start in range(0, flat.shape[0], slice_batch_size):
                encoded_batches.append(self._encode_slice_batch(flat[start : start + slice_batch_size]))
        encoded = self._stack_encoded_slices(encoded_batches, batch=batch, n_slices=n_slices)
        encoded["num_views"] = torch.tensor(num_views, device=flat.device, dtype=torch.int64)
        return encoded

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
        volume_tokens = self.slice_pool(slice_tokens)
        num_views = int(cached_features.get("num_views", torch.tensor(1)).item())
        batch_size = volume_tokens.shape[0] // max(1, num_views)
        volume_tokens = volume_tokens.view(batch_size, num_views, -1)
        if self.modality_pool == "mean":
            return volume_tokens.mean(dim=1)
        if self.modality_pool == "each":
            return volume_tokens
        if self.modality_pool == "first":
            return volume_tokens[:, 0, :]
        raise ValueError(f"Unknown modality_pool: {self.modality_pool}")

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
        if self.modality_pool == "each":
            return self.head(volume_token).mean(dim=1)
        return self.head(volume_token)

    def forward_each(
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
        if self.modality_pool != "each":
            pred = self.head(volume_token)
            return pred.unsqueeze(1)
        return self.head(volume_token)


def _prepare_cached_batch(cached_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ("cls_tokens", "patch_tokens", "register_tokens")
    batch = {key: torch.cat([item[key] for item in cached_list], dim=0) for key in keys}
    if cached_list and "num_views" in cached_list[0]:
        batch["num_views"] = torch.stack([item["num_views"] for item in cached_list]).flatten()
    return batch


def _load_cached_slices(cache_files: list[Path], device: torch.device) -> dict[str, torch.Tensor]:
    slice_caches = [cls_mod._load_cached_features(path, device) for path in cache_files]
    cls_tokens = torch.cat([item["cls_tokens"] for item in slice_caches], dim=1)
    patch_tokens = torch.cat([item["patch_tokens"] for item in slice_caches], dim=-1)
    register_tokens = torch.cat([item["register_tokens"] for item in slice_caches], dim=-1)
    return {
        "cls_tokens": cls_tokens,
        "patch_tokens": patch_tokens,
        "register_tokens": register_tokens,
    }


def maybe_load_or_create_cached_features(
    args: argparse.Namespace,
    task: str,
    batch: dict[str, object],
    model: DINOv3SliceVolumeRegressor,
    device: torch.device,
    slice_batch_size: int,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
    file_paths = batch["file_path"]
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    from_cache_root = Path(args.from_cache_path) / task if args.from_cache_path is not None else None
    cache_root = cls_mod._resolve_cache_root(args, task)
    cache_root.mkdir(parents=True, exist_ok=True)
    resize_hw = cls_mod._parse_hw_resize(args.resize_hw)
    images = batch["image"].to(device)
    active_cache_root = from_cache_root if from_cache_root is not None else cache_root
    cache_files = [
        cls_mod._cache_slice_files_for_sample(
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
        cached = _prepare_cached_batch([_load_cached_slices(sample_paths, device) for sample_paths in cache_files])
        cached["num_views"] = torch.full((len(cache_files),), int(images.shape[1] if args.modality_pool in {"mean", "each"} else 1), device=device, dtype=torch.int64)
        return None, cached

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
    cached_batch = _prepare_cached_batch(cached)
    cached_batch["num_views"] = torch.full((len(cache_files),), int(images.shape[1] if args.modality_pool in {"mean", "each"} else 1), device=device, dtype=torch.int64)
    return None, cached_batch


def regression_metrics(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(labels.shape)
    mse = float(mean_squared_error(labels, pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, pred)),
        "r2": float(r2_score(labels, pred)) if len(labels) > 1 else float("nan"),
    }


def _reshape_predictions_by_modality(
    pred: torch.Tensor,
    batch_size: int,
    num_modalities: int,
) -> torch.Tensor:
    return pred.view(batch_size, num_modalities, -1)


def evaluate(
    args: argparse.Namespace,
    task: str,
    model: DINOv3SliceVolumeRegressor,
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
    per_modality_losses: list[float] | None = None
    per_modality_abs_errors: list[float] | None = None
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
            if args.modality_pool == "each":
                num_modalities = int(batch["image"].shape[1])
                pred_each = model.forward_each(
                    x,
                    slice_batch_size=slice_batch_size,
                    cached_features=cached_features,
                )
                pred = pred_each.mean(dim=1)
                if per_modality_losses is None:
                    per_modality_losses = [0.0 for _ in range(num_modalities)]
                    per_modality_abs_errors = [0.0 for _ in range(num_modalities)]
                for modality_idx in range(num_modalities):
                    modality_pred = pred_each[:, modality_idx, :]
                    per_modality_losses[modality_idx] += float(loss_fn(modality_pred, y).item()) * y.shape[0]
                    per_modality_abs_errors[modality_idx] += float((modality_pred - y).abs().sum().item())
            else:
                pred = model(x, slice_batch_size=slice_batch_size, cached_features=cached_features)
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * y.shape[0]
            n_seen += y.shape[0]
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    metrics = regression_metrics(y_pred, y_true)
    metrics["loss"] = total_loss / max(1, n_seen)
    if per_modality_losses is not None and per_modality_abs_errors is not None:
        for modality_idx, modality_loss in enumerate(per_modality_losses):
            metrics[f"loss_modality_{modality_idx}"] = modality_loss / max(1, n_seen)
            metrics[f"mae_modality_{modality_idx}"] = per_modality_abs_errors[modality_idx] / max(1, n_seen)
    return metrics


def run_one_task(args: argparse.Namespace, task: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split_files = load_split_files(
        split_root=args.split_root,
        task=task,
        split_name=args.train_split,
        test_split_name=args.test_split,
        fold=args.fold,
        raw_root=args.raw_root,
        seed=args.seed,
    )
    resize_hw = cls_mod._parse_hw_resize(args.resize_hw)
    target_spacing_xy = cls_mod._parse_spacing_xy(args.target_spacing_xy)
    if target_spacing_xy is None:
        target_spacing_xy = (1.0, 1.0)
    pad_hw = cls_mod._parse_hw_resize(args.pad_hw)
    if pad_hw is None:
        pad_hw = (256, 256)

    train_ds = FOMORegressionDataset(
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
    val_ds = FOMORegressionDataset(
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
    test_ds = FOMORegressionDataset(
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

    output_dir = Path(args.output_dir) / task / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DINOv3SliceVolumeRegressor(
        dinov3_checkpoint=args.dinov3_checkpoint,
        num_outputs=RAW_TASK_CONFIG[task]["num_outputs"],
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

    print(
        f"\nTask={task} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"outputs=1 device={device} encoder_tuning={args.encoder_tuning}"
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
        train_loss_fused = 0.0
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
            y = batch["label"].to(device).float()
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if args.modality_pool == "each":
                num_modalities = int(batch["image"].shape[1])
                pred_each = model.forward_each(
                    x,
                    slice_batch_size=args.slice_batch_size,
                    cached_features=cached_features,
                )
                y_each = y.unsqueeze(1).expand(-1, num_modalities, -1)
                loss = loss_fn(pred_each, y_each)
                pred = pred_each.mean(dim=1)
                fused_loss = loss_fn(pred, y)
            else:
                pred = model(x, slice_batch_size=args.slice_batch_size, cached_features=cached_features)
                loss = loss_fn(pred, y)
                fused_loss = loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * y.shape[0]
            train_loss_fused += float(fused_loss.item()) * y.shape[0]
            n_seen += y.shape[0]
            if args.modality_pool == "each":
                progress.set_postfix(loss=train_loss / max(1, n_seen), fused=train_loss_fused / max(1, n_seen))
            else:
                progress.set_postfix(loss=train_loss / max(1, n_seen))

        val_metrics = evaluate(args, task, model, val_loader, device=device, slice_batch_size=args.slice_batch_size)
        val_message = (
            f"epoch={epoch + 1:03d} train_loss={train_loss / max(1, n_seen):.4f} "
            f"train_loss_fused={train_loss_fused / max(1, n_seen):.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_r2={val_metrics['r2']:.4f}"
        )
        if args.modality_pool == "each":
            modality_idx = 0
            per_modality_parts = []
            while f"loss_modality_{modality_idx}" in val_metrics:
                per_modality_parts.append(
                    f"val_loss_m{modality_idx}={val_metrics[f'loss_modality_{modality_idx}']:.4f}"
                )
                modality_idx += 1
            if per_modality_parts:
                val_message += " " + " ".join(per_modality_parts)
        print(val_message)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(args, task, model, test_loader, device=device, slice_batch_size=args.slice_batch_size)
    test_message = (
        f"[test] {task} loss={test_metrics['loss']:.4f} mae={test_metrics['mae']:.4f} "
        f"rmse={test_metrics['rmse']:.4f} r2={test_metrics['r2']:.4f}"
    )
    if args.modality_pool == "each":
        modality_idx = 0
        per_modality_parts = []
        while f"loss_modality_{modality_idx}" in test_metrics:
            per_modality_parts.append(
                f"loss_m{modality_idx}={test_metrics[f'loss_modality_{modality_idx}']:.4f}"
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
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split_root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dinov3_checkpoint", default=None)
    parser.add_argument("--train_split", default=DEFAULT_SPLIT)
    parser.add_argument("--test_split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--slice_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize_hw", default=None, help="Optional H,W resize before slicing, formatted as 'H,W'.")
    parser.add_argument("--target_spacing_xy", default="1.0,1.0", help="Optional X,Y spacing in mm before padding/resizing.")
    parser.add_argument("--pad_hw", default="256,256", help="Optional zero-pad target H,W after respacing.")
    parser.add_argument("--slice_size", type=int, default=None)
    parser.add_argument("--slice_axis", type=int, default=-1)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--modality_pool", default="mean", choices=["mean", "first", "each"])
    parser.add_argument("--mri_normalization", choices=["robust_zscore", "robust_minmax", "none"], default="robust_zscore")
    parser.add_argument("--mri_low_percentile", type=float, default=0.5)
    parser.add_argument("--mri_high_percentile", type=float, default=99.5)
    parser.add_argument("--slice_pool", default="patch_cls", choices=["mean", "max", "attention", "transformer", "patch", "cls", "patch_cls"])
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache backbone cls/patch/register features per volume.")
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--from_cache_path", default=None)
    parser.add_argument("--unfreeze_encoder", action="store_true", help="Train the full DINOv3 2D encoder.")
    parser.add_argument("--lora_encoder", action="store_true", help="Train LoRA adapters on the DINOv3 2D backbone.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_targets", default="qkv,proj")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.dinov3_checkpoint = resolve_dinov3_checkpoint(args.dinov3_checkpoint)
    if args.unfreeze_encoder and args.lora_encoder:
        raise ValueError("Use only one of --unfreeze_encoder or --lora_encoder.")
    args.encoder_tuning = "full" if args.unfreeze_encoder else "lora" if args.lora_encoder else "frozen"
    if args.from_cache_path is not None and args.encoder_tuning != "frozen":
        raise ValueError("--from_cache_path requires a frozen encoder because cached features bypass backbone training.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tasks = args.task if isinstance(args.task, list) else [args.task]
    for task in tasks:
        run_one_task(args, task)


if __name__ == "__main__":
    main()
