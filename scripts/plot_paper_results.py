"""Plot ServiceComputing paper experiment curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOTS = {
    "return_vs_steps_by_algorithm.png": ("stochastic_eval_return", "eval_return", "Return"),
    "completion_ratio_vs_steps_by_algorithm.png": ("stochastic_completion_ratio", "completion_ratio", "Completion"),
    "deadline_violation_vs_steps_by_algorithm.png": (
        "stochastic_deadline_violation_rate",
        "deadline_violation_rate",
        "Deadline violation",
    ),
    "energy_cost_vs_steps_by_algorithm.png": ("stochastic_mean_energy_cost", "mean_energy_cost", "Energy cost"),
    "queue_length_vs_steps_by_algorithm.png": ("stochastic_mean_queue_length", "mean_queue_length", "Queue length"),
}


def _meta(eval_curve: Path, root: Path) -> tuple[str, str, str, str]:
    rel = eval_curve.relative_to(root)
    parts = rel.parts
    if len(parts) >= 6:
        return parts[0], parts[1], parts[2], parts[3].replace("seed_", "")
    return "unknown", "unknown", parts[-4] if len(parts) >= 4 else "unknown", "unknown"


def _series(df: pd.DataFrame, primary: str, fallback: str, smooth_window: int) -> pd.Series | None:
    key = primary if primary in df.columns else fallback
    if key not in df.columns:
        return None
    values = df[key].astype(float)
    return values.rolling(window=max(1, smooth_window), min_periods=1).mean()


def plot_curves(root: Path, smooth_window: int) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curves = []
    for eval_curve in root.rglob("eval_curve.csv"):
        try:
            df = pd.read_csv(eval_curve)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "step" not in df.columns:
            continue
        difficulty, scale, algo, seed = _meta(eval_curve, root)
        curves.append((difficulty, scale, algo, seed, df))

    for filename, (primary, fallback, ylabel) in PLOTS.items():
        plt.figure(figsize=(9, 5))
        plotted = False
        for difficulty, scale, algo, seed, df in curves:
            y = _series(df, primary, fallback, smooth_window)
            if y is None:
                continue
            label = f"{difficulty}/{scale}/{algo}/s{seed}"
            plt.plot(df["step"], y, label=label, linewidth=1.4, alpha=0.85)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Environment steps")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=180)
        plt.close()

    _plot_scale_sensitivity(root, figure_dir)


def _plot_scale_sensitivity(root: Path, figure_dir: Path) -> None:
    summary_path = root / "paper_main_results.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty or "scale" not in df.columns:
        return
    scale_order = ["small", "medium", "large", "xlarge"]
    for metric, filename, ylabel in [
        ("final_return", "scale_sensitivity_return.png", "Final return"),
        ("completion_ratio", "scale_sensitivity_completion.png", "Completion"),
        ("mean_service_delay", "scale_sensitivity_delay.png", "Delay"),
        ("mean_energy_cost", "scale_sensitivity_energy.png", "Energy"),
    ]:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        for method, group in df[df["status"] == "completed"].groupby("method"):
            means = group.groupby("scale")[metric].mean()
            xs = [scale for scale in scale_order if scale in means.index]
            if not xs:
                continue
            plt.plot(xs, [means[x] for x in xs], marker="o", label=method)
        plt.xlabel("Scale")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_paper")
    parser.add_argument("--smooth_window", type=int, default=10)
    args = parser.parse_args()
    plot_curves(Path(args.root), args.smooth_window)
    print(f"Wrote figures under {Path(args.root) / 'figures'}")


if __name__ == "__main__":
    main()
