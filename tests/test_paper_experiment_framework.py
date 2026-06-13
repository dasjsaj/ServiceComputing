from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ServiceComputing.scripts.paper_experiment_utils import apply_scale, load_config_with_scale, SCALE_PRESETS
from ServiceComputing.scripts.train_service_algo import run_algorithm


def test_paper_scale_configs_match_declared_agent_counts():
    root = Path("ServiceComputing/configs")
    for difficulty in ["medium", "hard"]:
        for scale in ["small", "medium", "large"]:
            cfg = json.loads((root / f"service_paper_{difficulty}_{scale}.json").read_text(encoding="utf-8"))
            expected = SCALE_PRESETS[scale]
            assert cfg["paper"]["difficulty"] == difficulty
            assert cfg["paper"]["scale"] == scale
            for key, value in expected.items():
                assert cfg["env"][key] == value
            assert cfg["mappo"]["report_stochastic_eval"] is True


def test_apply_scale_updates_only_declared_agent_counts():
    cfg = load_config_with_scale(Path("ServiceComputing/configs/service_paper_medium_medium.json"))
    large = apply_scale(cfg, "large")

    assert large["env"]["n_auv"] == 6
    assert large["env"]["n_usv"] == 3
    assert large["env"]["n_uav"] == 3
    assert large["env"]["reward_mode"] == cfg["env"]["reward_mode"]
    assert large["env"]["action_mode"] == cfg["env"]["action_mode"]


def test_continuous_algorithm_runs_with_simple_action_mode():
    output_root = Path("artifacts/service_tests/paper_framework/maddpg")
    summary = run_algorithm(
        algo="maddpg",
        config_path=Path("ServiceComputing/configs/service_paper_medium_small.json"),
        seed=1,
        total_env_steps=256,
        run_name="maddpg_smoke",
        output_root=output_root,
        difficulty="medium",
        scale="small",
        eval_episodes=1,
    )

    run_dir = output_root / "medium" / "small" / "maddpg" / "seed_1" / "maddpg_smoke"
    assert summary["status"] == "completed"
    assert summary["env"]["action_mode"] == "simple"
    assert (run_dir / "checkpoints" / "checkpoint_best_stochastic.pt").exists()
    assert (run_dir / "summary.json").exists()
    df = pd.read_csv(run_dir / "eval_curve.csv")
    assert "stochastic_eval_return" in df.columns


def test_qmix_paper_entrypoint_runs_short_training(tmp_path=None):
    output_root = Path("artifacts/service_tests/paper_framework/qmix")
    summary = run_algorithm(
        algo="qmix",
        config_path=Path("ServiceComputing/configs/service_paper_medium_small.json"),
        seed=1,
        total_env_steps=256,
        run_name="qmix_train_smoke",
        output_root=output_root,
        difficulty="medium",
        scale="small",
        eval_episodes=1,
    )

    run_dir = output_root / "medium" / "small" / "qmix" / "seed_1" / "qmix_train_smoke"
    assert summary["status"] == "completed"
    assert (run_dir / "checkpoints" / "checkpoint_best_stochastic.pt").exists()
    df = pd.read_csv(run_dir / "eval_curve.csv")
    assert "stochastic_eval_return" in df.columns


def test_random_paper_entrypoint_writes_eval_curve_and_summary():
    output_root = Path("artifacts/service_tests/paper_framework/random")
    summary = run_algorithm(
        algo="random",
        config_path=Path("ServiceComputing/configs/service_paper_medium_small.json"),
        seed=1,
        total_env_steps=0,
        run_name="random_smoke",
        output_root=output_root,
        difficulty="medium",
        scale="small",
        eval_episodes=2,
    )

    run_dir = output_root / "medium" / "small" / "random" / "seed_1" / "random_smoke"
    assert summary["status"] == "completed"
    assert (run_dir / "eval_curve.csv").exists()
    assert (run_dir / "summary.json").exists()
    df = pd.read_csv(run_dir / "eval_curve.csv")
    assert len(df) == 2
    assert "completion_ratio" in df.columns
