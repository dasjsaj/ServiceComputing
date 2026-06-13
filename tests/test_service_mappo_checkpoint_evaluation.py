from __future__ import annotations

import csv
from pathlib import Path

from ServiceComputing.scripts.evaluate_service_mappo_checkpoint import (
    contribution_rows,
    write_evaluation_outputs,
)


def _row(method: str, seed_offset: int, episode_return: float, completion: float, delay: float) -> dict[str, float | str]:
    return {
        "method": method,
        "seed_offset": seed_offset,
        "eval_return": episode_return,
        "completion_ratio": completion,
        "mean_service_delay": delay,
        "deadline_violation_rate": 0.01,
        "mean_energy_cost": 0.04,
        "total_consumed_energy": 1.2,
        "completed_task_energy": 0.8,
        "timeout_task_energy": 0.1,
        "dropped_task_energy": 0.0,
        "inflight_task_energy": 0.3,
        "energy_per_generated_task": 0.012,
        "energy_per_completed_task": 0.010,
        "energy_per_successful_completion": 0.015,
        "mean_queue_length": 0.5,
    }


def test_contribution_rows_compare_each_policy_mode_against_same_window_baselines() -> None:
    rows = [
        _row("Random", 10000, 1.0, 0.70, 2.5),
        _row("Greedy", 10000, 3.0, 0.78, 2.1),
        _row("MAPPO-deterministic", 10000, 8.0, 0.90, 1.5),
        _row("MAPPO-stochastic", 10000, 9.0, 0.92, 1.4),
    ]

    comparisons = contribution_rows(rows)

    assert comparisons[0]["comparison"] == "MAPPO-deterministic_vs_Random"
    assert comparisons[0]["eval_return_gain"] == 7.0
    assert comparisons[0]["completion_ratio_gain"] == 0.20
    assert comparisons[0]["mean_service_delay_reduction"] == 1.0
    assert comparisons[-1]["comparison"] == "MAPPO-stochastic_vs_Greedy"


def test_write_evaluation_outputs_preserves_policy_modes_and_honest_interpretation() -> None:
    output_dir = Path("artifacts/test_service_mappo_checkpoint_eval")
    rows = [
        _row("Random", 10000, 1.0, 0.70, 2.5),
        _row("Greedy", 10000, 3.0, 0.78, 2.1),
        _row("MAPPO-deterministic", 10000, 8.0, 0.90, 1.5),
        _row("MAPPO-stochastic", 10000, 9.0, 0.92, 1.4),
    ]

    write_evaluation_outputs(output_dir, rows, checkpoint_label="checkpoint_best.pt")

    with (output_dir / "checkpoint_policy_mode_evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    report = (output_dir / "checkpoint_policy_mode_report.md").read_text(encoding="utf-8")

    assert {row["method"] for row in saved} == {
        "Random",
        "Greedy",
        "MAPPO-deterministic",
        "MAPPO-stochastic",
    }
    assert "checkpoint_best.pt" in report
    assert "deterministic" in report.lower()
    assert "stochastic" in report.lower()
    assert "does not by itself establish stable deterministic convergence" in report


def test_contribution_rows_support_semantic_policy_prefix_without_calling_it_mappo() -> None:
    rows = [
        _row("Random", 10000, 1.0, 0.70, 2.5),
        _row("Greedy", 10000, 3.0, 0.78, 2.1),
        _row("SLG-SAGE-deterministic", 10000, 8.5, 0.91, 1.45),
        _row("SLG-SAGE-stochastic", 10000, 9.5, 0.93, 1.35),
    ]

    comparisons = contribution_rows(rows, policy_prefix="SLG-SAGE")

    assert comparisons[0]["comparison"] == "SLG-SAGE-deterministic_vs_Random"
    assert comparisons[-1]["comparison"] == "SLG-SAGE-stochastic_vs_Greedy"


def test_write_evaluation_outputs_labels_semantic_checkpoint_methods() -> None:
    output_dir = Path("artifacts/test_service_slg_sage_checkpoint_eval")
    rows = [
        _row("Random", 10000, 1.0, 0.70, 2.5),
        _row("Greedy", 10000, 3.0, 0.78, 2.1),
        _row("SLG-SAGE-deterministic", 10000, 8.5, 0.91, 1.45),
        _row("SLG-SAGE-stochastic", 10000, 9.5, 0.93, 1.35),
    ]

    write_evaluation_outputs(output_dir, rows, checkpoint_label="checkpoint_best_stochastic.pt", policy_prefix="SLG-SAGE")

    report = (output_dir / "checkpoint_policy_mode_report.md").read_text(encoding="utf-8")

    assert "SLG-SAGE-stochastic" in report


def test_checkpoint_evaluation_csv_preserves_lifecycle_energy_fields() -> None:
    output_dir = Path("artifacts/test_service_energy_checkpoint_eval")
    rows = [
        _row("Random", 10000, 1.0, 0.70, 2.5),
        _row("Greedy", 10000, 3.0, 0.78, 2.1),
        _row("MAPPO-deterministic", 10000, 8.0, 0.90, 1.5),
        _row("MAPPO-stochastic", 10000, 9.0, 0.92, 1.4),
    ]

    write_evaluation_outputs(output_dir, rows, checkpoint_label="checkpoint_best_stochastic.pt")

    with (output_dir / "checkpoint_policy_mode_evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    for key in [
        "total_consumed_energy",
        "completed_task_energy",
        "timeout_task_energy",
        "inflight_task_energy",
        "energy_per_generated_task",
        "energy_per_successful_completion",
    ]:
        assert key in saved[0]
