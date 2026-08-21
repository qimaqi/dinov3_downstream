#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_MOD_PATH = ROOT / "downstream" / "3d_classify" / "dinov3_finetune_cls_from_slices.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("dinov3_finetune_cls_from_slices", TRAIN_MOD_PATH)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError(f"Cannot load training module from {TRAIN_MOD_PATH}")
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules["dinov3_finetune_cls_from_slices"] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


TASK_MODALITIES = {
    "CLS002_FOMO26_Infarct": ("flair", "adc", "dwi"),
}


class ModalitySubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, keep_indices: tuple[int, ...]):
        self.base_dataset = base_dataset
        self.keep_indices = keep_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.base_dataset[index]
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise ValueError("Expected sample['image'] to be a torch.Tensor.")
        subset = image[list(self.keep_indices), ...].contiguous()
        out = dict(sample)
        out["image"] = subset
        return out


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 modality ablations from saved Task 1 checkpoints.")
    parser.add_argument("--results_root", required=True, help="Root result directory that contains <task>/fold_*/best.pt.")
    parser.add_argument("--task", default="CLS002_FOMO26_Infarct", choices=sorted(TASK_MODALITIES))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--slice_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser


def ablation_specs(task: str) -> list[dict[str, object]]:
    modalities = TASK_MODALITIES[task]
    specs: list[dict[str, object]] = [
        {"name": "all_modalities", "keep_indices": tuple(range(len(modalities))), "keep_names": list(modalities)}
    ]
    for idx, name in enumerate(modalities):
        specs.append({"name": f"{name}_only", "keep_indices": (idx,), "keep_names": [name]})
    for drop_idx, drop_name in enumerate(modalities):
        keep_indices = tuple(i for i in range(len(modalities)) if i != drop_idx)
        keep_names = [modalities[i] for i in keep_indices]
        specs.append({"name": f"drop_{drop_name}", "keep_indices": keep_indices, "keep_names": keep_names})
    return specs


def build_eval_args(saved_args: dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    args_dict = deepcopy(saved_args)
    args_dict["epochs"] = 0
    args_dict["batch_size"] = cli_args.batch_size
    args_dict["slice_batch_size"] = cli_args.slice_batch_size
    args_dict["num_workers"] = cli_args.num_workers
    args_dict["cpu"] = cli_args.cpu
    args_dict["cache"] = False
    args_dict["cache_path"] = None
    args_dict["from_cache_path"] = None
    args_dict["unfreeze_encoder"] = False
    args_dict["lora_encoder"] = False
    args_dict["encoder_tuning"] = "frozen"
    return argparse.Namespace(**args_dict)


def dataset_bundle(args: argparse.Namespace, task: str, fold: int):
    task_dir = Path(args.processed_root) / task
    split_files = train_mod.load_split_files(task_dir, args.train_split, args.test_split, fold)
    resize_hw = train_mod._parse_hw_resize(args.resize_hw)
    target_spacing_xy = train_mod._parse_spacing_xy(args.target_spacing_xy)
    if target_spacing_xy is None:
        target_spacing_xy = (1.0, 1.0)
    pad_hw = train_mod._parse_hw_resize(args.pad_hw)
    if pad_hw is None:
        pad_hw = (256, 256)

    common = dict(
        task=task,
        raw_root=args.raw_root,
        resize_hw=resize_hw,
        target_spacing_xy=target_spacing_xy,
        pad_hw=pad_hw,
        mri_normalization=args.mri_normalization,
        mri_low_percentile=args.mri_low_percentile,
        mri_high_percentile=args.mri_high_percentile,
    )
    val_ds = train_mod.FOMOClsRegDataset(files=split_files.val, **common)
    test_ds = train_mod.FOMOClsRegDataset(files=split_files.test, **common)
    final_eval_ds = ConcatDataset([val_ds, test_ds])
    return val_ds, test_ds, final_eval_ds


def load_model(saved_args: dict, checkpoint_path: Path, task: str, device: torch.device):
    task_dir = Path(saved_args["processed_root"]) / task
    metadata = train_mod.load_task_metadata(task_dir)
    n_classes = int(metadata["metadata"]["n_classes"])
    model = train_mod.FlexiCTSliceVolumeClassifier(
        dinov3_checkpoint=saved_args["dinov3_checkpoint"],
        num_classes=n_classes,
        slice_pool=saved_args["slice_pool"],
        modality_pool=saved_args["modality_pool"],
        slice_axis=saved_args["slice_axis"],
        slice_size=saved_args["slice_size"],
        patch_size=saved_args["patch_size"],
        max_slices=saved_args["max_slices"],
        encoder_tuning="frozen",
        lora_r=saved_args["lora_r"],
        lora_alpha=saved_args["lora_alpha"],
        lora_targets=[item.strip() for item in saved_args["lora_targets"].split(",") if item.strip()],
        lora_dropout=saved_args["lora_dropout"],
        transformer_depth=saved_args["transformer_depth"],
        transformer_heads=saved_args["transformer_heads"],
        device=device,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    return model, payload


def summarize(rows: list[dict[str, object]], split_name: str) -> dict[str, dict[str, float | int]]:
    by_ablation: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ablation.setdefault(str(row["ablation"]), []).append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for ablation, items in sorted(by_ablation.items()):
        metrics = [item[split_name] for item in items]
        summary[ablation] = {
            "num_folds": len(items),
            "mean_bal_acc": float(np.mean([m["bal_acc"] for m in metrics])),
            "mean_auc": float(np.mean([m["auc"] for m in metrics])),
            "mean_accuracy_fused": float(np.mean([m["accuracy_fused"] for m in metrics])),
            "min_bal_acc": float(np.min([m["bal_acc"] for m in metrics])),
            "max_bal_acc": float(np.max([m["bal_acc"] for m in metrics])),
        }
    return summary


def main() -> None:
    parser = make_parser()
    cli_args = parser.parse_args()

    results_root = Path(cli_args.results_root).resolve()
    task_dir = results_root / cli_args.task
    ablation_dir = task_dir / "modality_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(task_dir.glob("fold_*/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {task_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not cli_args.cpu else "cpu")
    rows: list[dict[str, object]] = []

    for checkpoint_path in checkpoint_paths:
        fold_name = checkpoint_path.parent.name
        fold = int(fold_name.split("_")[-1])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = payload["args"]
        eval_args = build_eval_args(saved_args, cli_args)
        model, _ = load_model(saved_args, checkpoint_path, cli_args.task, device)
        _, test_ds, final_eval_ds = dataset_bundle(eval_args, cli_args.task, fold)

        print(f"Evaluating {fold_name} on {device} ...")
        for spec in ablation_specs(cli_args.task):
            keep_indices = tuple(spec["keep_indices"])
            keep_names = list(spec["keep_names"])
            test_subset = ModalitySubsetDataset(test_ds, keep_indices)
            final_subset = ModalitySubsetDataset(final_eval_ds, keep_indices)
            test_loader = DataLoader(test_subset, batch_size=eval_args.batch_size, shuffle=False, num_workers=eval_args.num_workers)
            final_loader = DataLoader(final_subset, batch_size=eval_args.batch_size, shuffle=False, num_workers=eval_args.num_workers)

            test_metrics = train_mod.evaluate(
                eval_args,
                cli_args.task,
                model,
                test_loader,
                device=device,
                slice_batch_size=eval_args.slice_batch_size,
            )
            final_metrics = train_mod.evaluate(
                eval_args,
                cli_args.task,
                model,
                final_loader,
                device=device,
                slice_batch_size=eval_args.slice_batch_size,
            )
            rows.append(
                {
                    "fold": fold_name,
                    "ablation": spec["name"],
                    "modalities": keep_names,
                    "num_modalities": len(keep_indices),
                    "test": test_metrics,
                    "final_val_test": final_metrics,
                }
            )

    summary = {
        "task": cli_args.task,
        "results_root": str(results_root),
        "evaluated_on": str(device),
        "ablations": ablation_specs(cli_args.task),
        "fold_results": rows,
        "summary_test": summarize(rows, "test"),
        "summary_final_val_test": summarize(rows, "final_val_test"),
    }

    with open(ablation_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(ablation_dir / "ablation_summary.csv", "w", encoding="utf-8") as f:
        f.write("split,ablation,num_folds,mean_bal_acc,mean_auc,mean_accuracy_fused,min_bal_acc,max_bal_acc\n")
        for split_name, key in (("test", "summary_test"), ("final_val_test", "summary_final_val_test")):
            for ablation, metrics in summary[key].items():
                f.write(
                    f"{split_name},{ablation},{metrics['num_folds']},{metrics['mean_bal_acc']:.6f},"
                    f"{metrics['mean_auc']:.6f},{metrics['mean_accuracy_fused']:.6f},"
                    f"{metrics['min_bal_acc']:.6f},{metrics['max_bal_acc']:.6f}\n"
                )

    print(f"Saved ablation results to {ablation_dir}")


if __name__ == "__main__":
    main()
