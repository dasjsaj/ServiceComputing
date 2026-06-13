"""Plot aligned MAPPO and SLG-SAGE long-run metrics from logged CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = []
        for row in csv.DictReader(handle):
            parsed.append({key: _float(value) for key, value in row.items()})
    if not parsed:
        raise ValueError(f"empty curve CSV: {path}")
    return parsed


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _series(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = [(row["step"], row.get(key, math.nan)) for row in rows if math.isfinite(row.get(key, math.nan))]
    if not values:
        return np.array([]), np.array([])
    return np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values])


def _rolling(rows: list[dict[str, float]], key: str, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = _series(rows, key)
    if not len(ys):
        return xs, ys
    kernel = np.ones(min(window, len(ys)), dtype=np.float64)
    smooth = np.convolve(ys, kernel / kernel.size, mode="valid")
    return xs[kernel.size - 1 :], smooth


def _compare_panels(
    path: Path,
    base_rows: list[dict[str, float]],
    sage_rows: list[dict[str, float]],
    panels: list[tuple[str, str, str]],
    title: str,
    smooth: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, (key, label, direction) in zip(axes.flat, panels):
        extractor = _rolling if smooth else _series
        for rows, name, color in [(base_rows, "MAPPO", "#2878B5"), (sage_rows, "SLG-SAGE", "#C44E52")]:
            xs, ys = extractor(rows, key)
            if len(xs):
                axis.plot(xs, ys, label=name, color=color, linewidth=2.0, marker=None if smooth else "o")
        axis.set_title(f"{label} ({direction})")
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _semantic_panels(path: Path, rows: list[dict[str, float]]) -> None:
    panels = [
        ("semantic_prior_loss", "Semantic prior loss"),
        ("semantic_guidance_loss", "Guidance loss"),
        ("semantic_aux_loss", "Auxiliary loss"),
        ("lambda_guide", "Guide coefficient"),
        ("semantic_logit_scale", "Residual logit scale"),
        ("teacher_policy_top1_match_rate_active", "Teacher top-1 match (active)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, panels):
        xs, ys = _rolling(rows, key)
        axis.plot(xs, ys, color="#C44E52", linewidth=2.0)
        axis.set_title(label)
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
    fig.suptitle("SLG-SAGE Semantic Guidance Training (10-record rolling mean)", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def generate_figures(base_run: Path, sage_run: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_eval = _rows(base_run / "eval_curve.csv")
    sage_eval = _rows(sage_run / "eval_curve.csv")
    base_train = _rows(base_run / "train_curve.csv")
    sage_train = _rows(sage_run / "train_curve.csv")
    outputs = [
        output_dir / "eval_performance_comparison.png",
        output_dir / "train_optimization_comparison.png",
        output_dir / "routing_behavior_comparison.png",
        output_dir / "semantic_guidance_training.png",
    ]
    _compare_panels(
        outputs[0],
        base_eval,
        sage_eval,
        [
            ("stochastic_eval_return", "Stochastic return", "higher is better"),
            ("stochastic_completion_ratio", "Completion ratio", "higher is better"),
            ("stochastic_mean_service_delay", "Service delay", "lower is better"),
            ("stochastic_deadline_violation_rate", "Deadline violation", "lower is better"),
            ("stochastic_mean_queue_length", "Queue length", "lower is better"),
            ("stochastic_mean_energy_cost", "Energy cost", "lower is better"),
        ],
        "Evaluation Metrics: MAPPO vs SLG-SAGE Joint Guidance",
    )
    _compare_panels(
        outputs[1],
        base_train,
        sage_train,
        [
            ("train_return", "Training return", "higher is better"),
            ("policy_loss", "Policy loss", "diagnostic"),
            ("value_loss", "Value loss", "diagnostic"),
            ("entropy_loss", "Entropy", "diagnostic"),
            ("grad_norm", "Gradient norm", "diagnostic"),
            ("total_loss", "Total loss", "diagnostic"),
        ],
        "Training Optimization Metrics (10-record rolling mean)",
        smooth=True,
    )
    _compare_panels(
        outputs[2],
        base_eval,
        sage_eval,
        [
            ("stochastic_usv_local_compute_preference_mean", "USV local compute rate", "behavior"),
            ("stochastic_usv_forward_uav_preference_mean", "USV to UAV rate", "behavior"),
            ("stochastic_usv_forward_shore_preference_mean", "USV to shore rate", "behavior"),
            ("stochastic_auv_upload_usv_ratio_mean", "AUV upload to USV rate", "behavior"),
            ("stochastic_uav_forward_shore_preference_mean", "UAV to shore rate", "behavior"),
            ("stochastic_weighted_backlog_cost", "Weighted backlog cost", "lower is better"),
        ],
        "Evaluation Routing Behavior: MAPPO vs SLG-SAGE Joint Guidance",
    )
    _semantic_panels(outputs[3], sage_train)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run", type=Path, required=True)
    parser.add_argument("--sage_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_figures(args.base_run, args.sage_run, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
