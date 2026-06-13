"""Unified ServiceComputing paper experiment training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ServiceComputing.algorithms.service_mappo_di import ServiceMAPPOTrainer
from ServiceComputing.algorithms.service_continuous_baselines import ServiceContinuousMARLTrainer
from ServiceComputing.algorithms.service_policy_baselines import ServiceOnPolicyMARLTrainer
from ServiceComputing.algorithms.service_value_baselines import ServiceValueMARLTrainer
from ServiceComputing.algorithms.slg_sage_mappo_di import SLGSAGEMAPPOTrainer
from ServiceComputing.scripts.evaluate_greedy_service_policy import rollout as greedy_rollout
from ServiceComputing.scripts.evaluate_random_service_policy import rollout as random_rollout
from ServiceComputing.scripts.paper_experiment_utils import (
    HEURISTIC_ALGOS,
    KNOWN_ALGOS,
    SUPPORTED_TRAINERS,
    UNSUPPORTED_REASON,
    apply_run_overrides,
    copy_best_stochastic_checkpoint,
    ensure_continuous_defaults,
    ensure_mappo_defaults,
    ensure_sage_defaults,
    infer_difficulty,
    infer_scale,
    load_config_with_scale,
    paper_run_dir,
    write_json,
    write_rows,
)
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics


def _heuristic_rollouts(config: dict[str, Any], algo: str, eval_episodes: int) -> list[dict[str, Any]]:
    env = make_service_env(config)
    rollout_fn = random_rollout if algo == "random" else greedy_rollout
    base_seed = int(config.get("seed", 0))
    rows = []
    for index in range(eval_episodes):
        metrics = rollout_fn(env, base_seed + index)
        row = {
            "step": 0,
            "episode": index,
            "eval_return": float(metrics.get("episode_return", np.nan)),
            **{key: value for key, value in metrics.items() if key != "episode_return"},
        }
        rows.append(row)
    return rows


def _write_heuristic_run(config: dict[str, Any], run_dir: Path, algo: str) -> dict[str, Any]:
    eval_episodes = int(config.get("mappo", {}).get("eval_episodes", 20))
    rows = _heuristic_rollouts(config, algo, eval_episodes)
    summary_metrics = mean_metrics(rows)
    summary_metrics["eval_return"] = float(np.mean([row["eval_return"] for row in rows]))
    summary = {
        "algo": algo,
        "status": "completed",
        "total_env_steps": 0,
        "episodes": eval_episodes,
        "last_stochastic_eval": summary_metrics,
        "best_stochastic_eval": summary_metrics,
        "best_stochastic_eval_return": summary_metrics["eval_return"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config, run_dir / "config.json")
    write_rows(run_dir / "eval_curve.csv", rows)
    write_rows(run_dir / "train_curve.csv", [])
    write_json(summary, run_dir / "summary.json")
    return summary


def _write_adapter_failure(config: dict[str, Any], run_dir: Path, algo: str) -> dict[str, Any]:
    summary = {
        "algo": algo,
        "status": "adapter_failure",
        "reason": UNSUPPORTED_REASON,
        "best_stochastic_eval_return": None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config, run_dir / "config.json")
    write_json(summary, run_dir / "summary.json")
    write_rows(
        run_dir / "eval_curve.csv",
        [{"step": 0, "algo": algo, "status": "adapter_failure", "reason": UNSUPPORTED_REASON}],
    )
    write_rows(run_dir / "train_curve.csv", [])
    return summary


def run_algorithm(
    *,
    algo: str,
    config_path: Path,
    seed: int,
    total_env_steps: int | None,
    run_name: str | None,
    output_root: Path,
    difficulty: str | None = None,
    scale: str | None = None,
    eval_episodes: int | None = None,
) -> dict[str, Any]:
    algo = algo.lower()
    if algo not in KNOWN_ALGOS:
        raise ValueError(f"unknown algorithm: {algo}")
    cfg = load_config_with_scale(config_path, scale)
    difficulty_name = infer_difficulty(cfg, difficulty)
    scale_name = infer_scale(cfg, scale)
    name = run_name or f"{algo}_{difficulty_name}_{scale_name}_seed{seed}"
    cfg = apply_run_overrides(
        cfg,
        seed=seed,
        total_env_steps=total_env_steps,
        eval_episodes=eval_episodes,
        run_name=name,
        output_root=output_root,
        difficulty=difficulty_name,
        scale=scale_name,
        algo=algo,
    )
    run_dir = paper_run_dir(output_root, difficulty_name, scale_name, algo, seed, name)

    summary_path = run_dir / "summary.json"
    best_checkpoint = run_dir / "checkpoints" / "checkpoint_best_stochastic.pt"
    if summary_path.exists():
        try:
            prior_summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior_summary = {}
        if (
            prior_summary.get("status") == "completed"
            and (algo in HEURISTIC_ALGOS or best_checkpoint.exists() or (run_dir / "checkpoints" / "best_stochastic.pt").exists())
        ):
            prior_summary["status"] = "completed"
            prior_summary["skipped_existing_completed_run"] = True
            return prior_summary

    if algo in HEURISTIC_ALGOS:
        return _write_heuristic_run(cfg, run_dir, algo)
    if algo not in SUPPORTED_TRAINERS:
        return _write_adapter_failure(cfg, run_dir, algo)

    if algo == "mappo":
        cfg = ensure_mappo_defaults(cfg)
        trainer = ServiceMAPPOTrainer(cfg, run_dir)
    elif algo == "slg_sage":
        cfg = ensure_sage_defaults(cfg)
        trainer = SLGSAGEMAPPOTrainer(cfg, run_dir)
    elif algo in {"coma", "happo"}:
        cfg = ensure_mappo_defaults(cfg)
        trainer = ServiceOnPolicyMARLTrainer(cfg, run_dir, algo=algo)
    elif algo in {"maddpg", "masac", "matd3"}:
        cfg = ensure_continuous_defaults(cfg)
        trainer = ServiceContinuousMARLTrainer(cfg, run_dir, algo=algo)
    else:
        cfg = ensure_mappo_defaults(cfg)
        trainer = ServiceValueMARLTrainer(cfg, run_dir, algo=algo)
    summary = trainer.train()
    copy_best_stochastic_checkpoint(run_dir)
    summary["algo"] = algo
    summary["status"] = "completed"
    write_json(summary, run_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_root", default="artifacts/service_paper")
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--policy_mode", choices=["stochastic"], default="stochastic")
    args = parser.parse_args()

    summary = run_algorithm(
        algo=args.algo,
        config_path=Path(args.config),
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        run_name=args.run_name,
        output_root=Path(args.output_root),
        difficulty=args.difficulty,
        scale=args.scale,
        eval_episodes=args.eval_episodes,
    )
    print(summary)


if __name__ == "__main__":
    main()
