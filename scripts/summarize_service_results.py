"""Summarize service offloading experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _meta(eval_curve: Path, root: Path) -> dict[str, str]:
    try:
        rel = eval_curve.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = eval_curve.parts
    if len(parts) >= 6:
        return {
            "difficulty": parts[0],
            "scale": parts[1],
            "method": parts[2],
            "seed": parts[3].replace("seed_", ""),
            "run_name": parts[4],
        }
    return {
        "difficulty": "unknown",
        "scale": "unknown",
        "method": parts[-4] if len(parts) >= 4 else "unknown",
        "seed": parts[-3].replace("seed_", "") if len(parts) >= 3 else "unknown",
        "run_name": parts[-2] if len(parts) >= 2 else "unknown",
    }


def _value(tail, *keys: str) -> float:
    for key in keys:
        if key in tail:
            value = float(tail.get(key, np.nan))
            if np.isfinite(value):
                return value
    return float("nan")


def summarize(root: Path, tail_points: int = 5) -> pd.DataFrame:
    rows = []
    for eval_curve in root.rglob("eval_curve.csv"):
        try:
            df = pd.read_csv(eval_curve)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        meta = _meta(eval_curve, root)
        if "status" in df.columns and str(df.iloc[-1].get("status")) == "adapter_failure":
            rows.append({**meta, "status": "adapter_failure", "reason": str(df.iloc[-1].get("reason", ""))})
            continue
        tail = df.tail(min(tail_points, len(df))).mean(numeric_only=True)
        rows.append(
            {
                **meta,
                "status": "completed",
                "final_return": _value(tail, "stochastic_eval_return", "eval_return"),
                "completion_ratio": _value(tail, "stochastic_completion_ratio", "completion_ratio"),
                "mean_service_delay": _value(tail, "stochastic_mean_service_delay", "mean_service_delay"),
                "deadline_violation_rate": _value(
                    tail, "stochastic_deadline_violation_rate", "deadline_violation_rate"
                ),
                "mean_energy_cost": _value(tail, "stochastic_mean_energy_cost", "mean_energy_cost"),
                "offload_success_rate": _value(tail, "stochastic_offload_success_rate", "offload_success_rate"),
                "mean_queue_length": _value(tail, "stochastic_mean_queue_length", "mean_queue_length"),
                "semantic_changed_argmax_rate": _value(tail, "semantic_changed_argmax_rate"),
                "teacher_policy_top1_match_rate": _value(tail, "teacher_policy_top1_match_rate"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_paper")
    parser.add_argument("--tail_points", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.root)
    out = summarize(root, args.tail_points)
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "paper_main_results.csv", index=False)
    if not out.empty and {"difficulty", "scale", "method"}.issubset(out.columns):
        completed = out[out["status"] == "completed"].copy()
        numeric_cols = completed.select_dtypes(include=[np.number]).columns
        if not completed.empty:
            summary = completed.groupby(["difficulty", "scale", "method"])[numeric_cols].agg(["mean", "std"])
            summary.to_csv(root / "paper_algorithm_ranking.csv")
    print(f"Wrote {root / 'paper_main_results.csv'}")


if __name__ == "__main__":
    main()
