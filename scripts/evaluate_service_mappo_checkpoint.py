"""Evaluate a MAPPO checkpoint under matched deterministic/stochastic protocols.

This script does not train or modify policies.  It keeps random, greedy, and
MAPPO rollouts on identical seed windows so execution-mode effects are not
mistaken for learning gains.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from ServiceComputing.algorithms.service_mappo_di import ServiceMAPPOTrainer
from ServiceComputing.scripts.common import load_json, write_json
from ServiceComputing.scripts.evaluate_greedy_service_policy import rollout as greedy_rollout
from ServiceComputing.scripts.evaluate_random_service_policy import rollout as random_rollout
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics

OUTPUT_FIELDS = [
    "method",
    "seed_offset",
    "eval_episodes",
    "eval_return",
    "completion_ratio",
    "mean_service_delay",
    "deadline_violation_rate",
    "mean_energy_cost",
    "total_consumed_energy",
    "completed_task_energy",
    "timeout_task_energy",
    "dropped_task_energy",
    "inflight_task_energy",
    "energy_per_generated_task",
    "energy_per_completed_task",
    "energy_per_successful_completion",
    "energy_accounting_balance_error",
    "energy_compute_auv",
    "energy_compute_usv",
    "energy_compute_uav",
    "energy_compute_shore",
    "energy_transfer_auv_usv",
    "energy_transfer_usv_uav",
    "energy_transfer_usv_shore",
    "energy_transfer_uav_shore",
    "offload_success_rate",
    "mean_queue_length",
]


def _metric(row: dict, key: str) -> float:
    return float(row.get(key, float("nan")))


def _baseline_metrics(config: dict, rollout_fn, seed_offset: int, episodes: int) -> dict[str, float]:
    env = make_service_env(config)
    base_seed = int(config.get("seed", 0)) + seed_offset
    rows = [rollout_fn(env, base_seed + index) for index in range(episodes)]
    summary = mean_metrics(rows)
    summary["eval_return"] = float(np.mean([row["episode_return"] for row in rows]))
    return summary


def contribution_rows(rows: list[dict], policy_prefix: str = "MAPPO") -> list[dict[str, float | int | str]]:
    """Compare a learned policy's execution modes against each baseline per seed window."""
    by_window = {(str(row["method"]), int(row["seed_offset"])): row for row in rows}
    comparisons: list[dict[str, float | int | str]] = []
    offsets = sorted({int(row["seed_offset"]) for row in rows})
    for offset in offsets:
        for policy in (f"{policy_prefix}-deterministic", f"{policy_prefix}-stochastic"):
            for baseline in ("Random", "Greedy"):
                learned = by_window[(policy, offset)]
                base = by_window[(baseline, offset)]
                comparisons.append(
                    {
                        "seed_offset": offset,
                        "comparison": f"{policy}_vs_{baseline}",
                        "eval_return_gain": round(_metric(learned, "eval_return") - _metric(base, "eval_return"), 12),
                        "completion_ratio_gain": round(
                            _metric(learned, "completion_ratio") - _metric(base, "completion_ratio"), 12
                        ),
                        "mean_service_delay_reduction": round(
                            _metric(base, "mean_service_delay") - _metric(learned, "mean_service_delay"), 12
                        ),
                        "deadline_violation_reduction": round(
                            _metric(base, "deadline_violation_rate") - _metric(learned, "deadline_violation_rate"), 12
                        ),
                    }
                )
    return comparisons


def _mean_std(rows: list[dict], method: str, key: str) -> tuple[float, float]:
    vals = [_metric(row, key) for row in rows if row["method"] == method]
    return float(mean(vals)), float(pstdev(vals) if len(vals) > 1 else 0.0)


def write_evaluation_outputs(run_dir: Path, rows: list[dict], checkpoint_label: str, policy_prefix: str = "MAPPO") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "checkpoint_policy_mode_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    comparisons = contribution_rows(rows, policy_prefix=policy_prefix)
    contribution_path = run_dir / "checkpoint_policy_mode_contribution.csv"
    with contribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()) if comparisons else [])
        if comparisons:
            writer.writeheader()
            writer.writerows(comparisons)

    methods = ["Random", "Greedy", f"{policy_prefix}-deterministic", f"{policy_prefix}-stochastic"]
    summary = {
        method: {
            key: {"mean": _mean_std(rows, method, key)[0], "std": _mean_std(rows, method, key)[1]}
            for key in ["eval_return", "completion_ratio", "mean_service_delay", "deadline_violation_rate"]
        }
        for method in methods
        if any(row["method"] == method for row in rows)
    }
    write_json(
        {"checkpoint": checkpoint_label, "evaluation_rows": rows, "comparisons": comparisons, "summary": summary},
        run_dir / "checkpoint_policy_mode_summary.json",
    )

    report = [
        f"# {policy_prefix} Checkpoint Policy-Mode Evaluation",
        "",
        f"Checkpoint: `{checkpoint_label}`",
        "",
        f"This evaluation uses matched rollout seed windows for all methods. {policy_prefix} is evaluated both with "
        "deterministic argmax execution and stochastic sampling from its learned policy.",
        "",
        "## Aggregate Results",
        "",
        "| Method | Eval return (mean +/- std) | Completion (mean +/- std) | Delay (mean +/- std) | SLA violation (mean +/- std) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        if method not in summary:
            continue
        item = summary[method]
        report.append(
            f"| {method} | {item['eval_return']['mean']:.4f} +/- {item['eval_return']['std']:.4f} | "
            f"{item['completion_ratio']['mean']:.4f} +/- {item['completion_ratio']['std']:.4f} | "
            f"{item['mean_service_delay']['mean']:.4f} +/- {item['mean_service_delay']['std']:.4f} | "
            f"{item['deadline_violation_rate']['mean']:.4f} +/- {item['deadline_violation_rate']['std']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            f"A stronger {policy_prefix} result here demonstrates checkpoint performance under the stated execution "
            "mode. It does not by itself establish stable deterministic convergence across training.",
            f"A gap between deterministic and stochastic {policy_prefix} indicates an execution/coordination issue "
            "that should be settled before introducing semantic losses.",
        ]
    )
    (run_dir / "checkpoint_policy_mode_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"rows": rows, "comparisons": comparisons, "summary": summary}


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    given = Path(checkpoint)
    if given.is_absolute():
        return given
    if len(given.parts) == 1:
        return run_dir / "checkpoints" / given
    return run_dir / given


def evaluate_checkpoint(
    run_dir: Path, checkpoint: Path, seed_offsets: list[int], eval_episodes: int | None, trainer_type: str = "mappo"
) -> dict:
    config = load_json(run_dir / "config.json")
    if eval_episodes is not None:
        config.setdefault("mappo", {})["eval_episodes"] = int(eval_episodes)
    episodes = int(config.get("mappo", {}).get("eval_episodes", 5))
    if trainer_type == "mappo":
        trainer_class = ServiceMAPPOTrainer
        policy_prefix = "MAPPO"
    elif trainer_type == "slg_sage":
        from ServiceComputing.algorithms.slg_sage_mappo_di import SLGSAGEMAPPOTrainer

        trainer_class = SLGSAGEMAPPOTrainer
        policy_prefix = "SLG-SAGE"
    else:
        raise ValueError(f"unsupported trainer_type: {trainer_type}")
    trainer = trainer_class(config, run_dir / "_checkpoint_eval")
    trainer.load_checkpoint(checkpoint)
    rows: list[dict] = []
    for seed_offset in seed_offsets:
        method_metrics = {
            "Random": _baseline_metrics(config, random_rollout, seed_offset, episodes),
            "Greedy": _baseline_metrics(config, greedy_rollout, seed_offset, episodes),
            f"{policy_prefix}-deterministic": trainer.evaluate(seed_offset=seed_offset, deterministic=True),
            f"{policy_prefix}-stochastic": trainer.evaluate(seed_offset=seed_offset, deterministic=False),
        }
        for method, metrics in method_metrics.items():
            rows.append(
                {
                    "method": method,
                    "seed_offset": seed_offset,
                    "eval_episodes": episodes,
                    "eval_return": float(metrics.get("eval_return", metrics.get("episode_return", float("nan")))),
                    "completion_ratio": _metric(metrics, "completion_ratio"),
                    "mean_service_delay": _metric(metrics, "mean_service_delay"),
                    "deadline_violation_rate": _metric(metrics, "deadline_violation_rate"),
                    "mean_energy_cost": _metric(metrics, "mean_energy_cost"),
                    "total_consumed_energy": _metric(metrics, "total_consumed_energy"),
                    "completed_task_energy": _metric(metrics, "completed_task_energy"),
                    "timeout_task_energy": _metric(metrics, "timeout_task_energy"),
                    "dropped_task_energy": _metric(metrics, "dropped_task_energy"),
                    "inflight_task_energy": _metric(metrics, "inflight_task_energy"),
                    "energy_per_generated_task": _metric(metrics, "energy_per_generated_task"),
                    "energy_per_completed_task": _metric(metrics, "energy_per_completed_task"),
                    "energy_per_successful_completion": _metric(metrics, "energy_per_successful_completion"),
                    "energy_accounting_balance_error": _metric(metrics, "energy_accounting_balance_error"),
                    "energy_compute_auv": _metric(metrics, "energy_compute_auv"),
                    "energy_compute_usv": _metric(metrics, "energy_compute_usv"),
                    "energy_compute_uav": _metric(metrics, "energy_compute_uav"),
                    "energy_compute_shore": _metric(metrics, "energy_compute_shore"),
                    "energy_transfer_auv_usv": _metric(metrics, "energy_transfer_auv_usv"),
                    "energy_transfer_usv_uav": _metric(metrics, "energy_transfer_usv_uav"),
                    "energy_transfer_usv_shore": _metric(metrics, "energy_transfer_usv_shore"),
                    "energy_transfer_uav_shore": _metric(metrics, "energy_transfer_uav_shore"),
                    "offload_success_rate": _metric(metrics, "offload_success_rate"),
                    "mean_queue_length": _metric(metrics, "mean_queue_length"),
                }
            )
    return write_evaluation_outputs(run_dir, rows, str(checkpoint), policy_prefix=policy_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--seed_offsets", default="10000,20000,30000")
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--trainer_type", choices=["mappo", "slg_sage"], default="mappo")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    offsets = [int(token.strip()) for token in args.seed_offsets.split(",") if token.strip()]
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    result = evaluate_checkpoint(run_dir, checkpoint, offsets, args.eval_episodes, trainer_type=args.trainer_type)
    print(json.dumps({"checkpoint": str(checkpoint), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
