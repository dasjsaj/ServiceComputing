"""Compare analyzed service-computing training runs.

Each run directory should already contain ``mappo_convergence_summary.json``
from ``analyze_service_mappo_convergence``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEY_METRICS = [
    "eval_return_auc",
    "completion_ratio_auc",
    "mean_delay_auc",
    "queue_length_auc",
    "N90_eval_return",
    "last_20pct_eval_return_mean",
    "last_20pct_completion_mean",
    "last_20pct_delay_mean",
    "last_20pct_queue_mean",
    "value_loss_max",
    "value_loss_last_mean",
    "entropy_last_mean",
    "gradient_norm_max",
]


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "mappo_convergence_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing analyzer summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], output_path: Path, baseline_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Service MAPPO vs SLG-SAGE Run Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        "| method | passed | return AUC | tail return | tail completion | tail delay | tail queue | value loss max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {passed} | {eval_return_auc:.6f} | {last_20pct_eval_return_mean:.6f} | "
            "{last_20pct_completion_mean:.6f} | {last_20pct_delay_mean:.6f} | "
            "{last_20pct_queue_mean:.6f} | {value_loss_max:.6f} |".format(**row)
        )
    if len(rows) > 1:
        best_return = max(rows, key=lambda r: r["last_20pct_eval_return_mean"])
        best_delay = min(rows, key=lambda r: r["last_20pct_delay_mean"])
        lines.extend(
            [
                "",
                "## Automatic Reading",
                "",
                f"- Best tail return: `{best_return['method']}`.",
                f"- Best tail delay: `{best_delay['method']}`.",
                "- Interpret completion together with delay and queue length; a higher return with equal completion usually means the policy is completing similar work with lower service cost or delay.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR. First run is the baseline.")
    parser.add_argument("--output_dir", default="artifacts/service_comparisons")
    args = parser.parse_args()

    parsed = [parse_run_arg(v) for v in args.run]
    baseline_name = parsed[0][0]
    baseline_summary = load_summary(parsed[0][1])
    baseline_metrics = baseline_summary["metrics"]
    rows = []
    for name, run_dir in parsed:
        summary = load_summary(run_dir)
        metrics = summary["metrics"]
        passed = bool(summary.get("passed", summary.get("decision", {}).get("passed", False)))
        row = {
            "method": name,
            "run_dir": str(run_dir),
            "passed": passed,
        }
        for key in KEY_METRICS:
            row[key] = float(metrics.get(key, float("nan")))
            row[f"delta_vs_{baseline_name}_{key}"] = row[key] - float(baseline_metrics.get(key, float("nan")))
        rows.append(row)

    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "service_run_comparison.csv")
    write_report(rows, output_dir / "service_run_comparison_report.md", baseline_name)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
