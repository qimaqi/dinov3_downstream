#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(metrics: dict, section: str, key: str) -> float:
    value = metrics.get(section, {}).get(key)
    if value is None:
        return float("-inf")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank 3D classification fold results.")
    parser.add_argument("--results_root", required=True, help="Directory containing task/fold_* outputs.")
    parser.add_argument("--task", default="CLS002_FOMO26_Infarct")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    task_dir = Path(args.results_root) / args.task
    if not task_dir.exists():
        raise FileNotFoundError(f"Task results directory not found: {task_dir}")

    rows = []
    for fold_dir in sorted(task_dir.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "fold": fold_dir.name,
                "metrics_path": str(metrics_path),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_loss": metrics.get("best_val_loss"),
                "test_bal_acc": _metric(metrics, "test", "bal_acc"),
                "test_auc": _metric(metrics, "test", "auc"),
                "test_acc_fused": _metric(metrics, "test", "accuracy_fused"),
                "final_bal_acc": _metric(metrics, "final_val_test", "bal_acc"),
                "final_auc": _metric(metrics, "final_val_test", "auc"),
                "final_acc_fused": _metric(metrics, "final_val_test", "accuracy_fused"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No fold metrics found under: {task_dir}")

    rows.sort(
        key=lambda row: (
            row["test_bal_acc"],
            row["test_auc"],
            row["final_bal_acc"],
            row["final_auc"],
            row["test_acc_fused"],
            -float(row["best_val_loss"]) if row["best_val_loss"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    top_rows = rows[: max(1, args.top_k)]
    summary = {
        "ranking_rule": [
            "test.bal_acc desc",
            "test.auc desc",
            "final_val_test.bal_acc desc",
            "final_val_test.auc desc",
            "test.accuracy_fused desc",
            "best_val_loss asc",
        ],
        "top_k": args.top_k,
        "task": args.task,
        "results_root": str(Path(args.results_root)),
        "ranked_folds": top_rows,
    }

    output_path = task_dir / "ranking_top5.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Ranking results for {args.task} in {task_dir}")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank}. {row['fold']} "
            f"test_bal_acc={row['test_bal_acc']:.4f} "
            f"test_auc={row['test_auc']:.4f} "
            f"final_bal_acc={row['final_bal_acc']:.4f} "
            f"final_auc={row['final_auc']:.4f} "
            f"best_val_loss={row['best_val_loss']:.4f}"
        )
    print(f"Saved ranking summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
