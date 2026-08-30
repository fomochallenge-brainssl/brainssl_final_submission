"""Aggregate metrics across folds and tasks for one model experiment.

Handles two evaluation modes:
  - finetuning:      <base>/<task>/<fold>/metrics.tsv  (one file per fold)
  - linear_probing:  <base>/<task>_linear_probing/metrics.tsv  (all folds in one file)

Usage
-----
    python scripts/gather_results.py \\
        --results_dir outputs/results \\
        --model densenet121 \\
        --exp_name my_exp

Prints a per-task x per-mode summary (mean ± std across folds for each metric)
and saves aggregated tables to:
    <results_dir>/<model>/<exp_name>/all_metrics.tsv
    <results_dir>/<model>/<exp_name>/summary.txt
"""

import os
import re
import sys
from pathlib import Path
from omegaconf import DictConfig

import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd

# Ensure the repo root is on the path when calling from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import rootutils


def parse_metric_value(value) -> float:
    """Parse metric values saved as floats or ``tensor(...)`` strings."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"tensor\(([-+eE0-9.]+)\)", text)
    if match:
        return float(match.group(1))
    return float(text)


def parse_metric_value(value) -> float:
    """Parse metric values saved as floats or ``tensor(...)`` strings."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"tensor\(([-+eE0-9.]+)\)", text)
    if match:
        return float(match.group(1))
    return float(text)


def gather(results_dir: str, model: str, exp_name: str) -> pd.DataFrame:
    base = Path(results_dir) / model / exp_name
    if not base.exists():
        sys.exit(f"[gather_results] Directory not found: {base}")

    frames = []
    for metrics_file in sorted(base.glob("**/metrics.tsv")):
        parts = metrics_file.relative_to(base).parts
        if len(parts) == 3:
            # <task>/<fold>/metrics.tsv — finetuning
            task = parts[0]
            mode = "finetuning"
        elif len(parts) == 2:
            # <task>_linear_probing/metrics.tsv — linear probing
            task = parts[0].replace("_linear_probing", "")
            mode = "linear_probing"
        else:
            continue

        df = pd.read_csv(metrics_file, sep="\t")
        df["task"] = task
        df["mode"] = mode
        frames.append(df)

    if not frames:
        sys.exit(f"[gather_results] No metrics.tsv files found under {base}")

    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics["value"] = all_metrics["value"].map(parse_metric_value)
    return all_metrics


def summarise(all_metrics: pd.DataFrame) -> str:
    lines = ["\n=== Per-task summary (mean ± std across folds) ===\n"]
    for (task, mode), grp in all_metrics.groupby(["task", "mode"]):
        lines.append(f"  {task}  [{mode}]")
        for metric, mgrp in grp.groupby("metric"):
            vals = mgrp["value"]
            lines.append(
                f"    {metric}: {vals.mean():.2f} ± {vals.std():.2f}  (n={len(vals)})"
            )
    return "\n".join(lines)

rootutils.setup_root(__file__, indicator=".env")

@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig):
    results_dir = cfg.task.probing.results_dir
    model = cfg.eval.model_name
    exp_name = cfg.exp_name

    all_metrics = gather(results_dir, model, exp_name)
    summary = summarise(all_metrics)
    print(summary)

    out_path = Path(results_dir) / model / exp_name
    all_metrics.to_csv(out_path / "all_metrics.tsv", sep="\t", index=False)
    (out_path / "summary.txt").write_text(summary)
    print(f"[gather_results] Saved aggregated metrics to {out_path}")


if __name__ == "__main__":
    main()
