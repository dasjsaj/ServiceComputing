import csv
from pathlib import Path
from uuid import uuid4


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(tmp_path: Path, sage: bool) -> Path:
    run_dir = tmp_path / ("sage" if sage else "base")
    run_dir.mkdir(parents=True)
    eval_rows = []
    train_rows = []
    for index in range(12):
        row = {
            "step": float(index * 100),
            "stochastic_eval_return": float(index),
            "stochastic_completion_ratio": 0.5 + index * 0.01,
            "stochastic_mean_service_delay": 2.0 - index * 0.01,
            "stochastic_deadline_violation_rate": 0.1,
            "stochastic_mean_queue_length": 1.0,
            "stochastic_mean_energy_cost": 0.05,
            "stochastic_usv_local_compute_preference_mean": 0.5,
            "stochastic_usv_forward_uav_preference_mean": 0.25,
            "stochastic_usv_forward_shore_preference_mean": 0.25,
            "stochastic_auv_upload_usv_ratio_mean": 0.8,
            "stochastic_uav_forward_shore_preference_mean": 0.5,
            "stochastic_weighted_backlog_cost": 0.3,
        }
        if sage:
            row["stochastic_semantic_active_fraction"] = 0.2
        eval_rows.append(row)
        train_row = {
            "step": float(index * 100),
            "train_return": float(index),
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy_loss": 0.8,
            "total_loss": 0.3,
            "grad_norm": 0.4,
        }
        if sage:
            train_row.update(
                {
                    "semantic_prior_loss": 0.2,
                    "semantic_guidance_loss": 0.1,
                    "semantic_aux_loss": 0.05,
                    "lambda_prior": 0.08,
                    "lambda_guide": 0.03,
                    "semantic_logit_scale": 0.2,
                    "semantic_active_fraction": 0.2,
                    "semantic_changed_argmax_rate_active": 0.1,
                    "teacher_policy_top1_match_rate_active": 0.6,
                }
            )
        train_rows.append(train_row)
    _write_csv(run_dir / "eval_curve.csv", eval_rows)
    _write_csv(run_dir / "train_curve.csv", train_rows)
    return run_dir


def test_generate_figures_writes_comparison_panels():
    from ServiceComputing.scripts.plot_slg_sage_training_comparison import generate_figures

    root = Path("artifacts") / "test_plot_outputs" / uuid4().hex
    outputs = generate_figures(_run_dir(root, False), _run_dir(root, True), root / "figures")

    assert {path.name for path in outputs} == {
        "eval_performance_comparison.png",
        "routing_behavior_comparison.png",
        "semantic_guidance_training.png",
        "train_optimization_comparison.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
