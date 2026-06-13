"""Stochastic-only best checkpoint evaluation for ServiceComputing paper runs."""

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
from ServiceComputing.scripts.paper_experiment_utils import write_rows
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics


def _path_meta(config_path: Path, root: Path) -> dict[str, str]:
    rel = config_path.relative_to(root)
    parts = rel.parts
    return {
        "difficulty": parts[0] if len(parts) > 0 else "unknown",
        "scale": parts[1] if len(parts) > 1 else "unknown",
        "algo": parts[2] if len(parts) > 2 else "unknown",
        "seed": parts[3].replace("seed_", "") if len(parts) > 3 else "unknown",
        "run_name": parts[4] if len(parts) > 4 else "unknown",
    }


def _eval_heuristic(config: dict[str, Any], algo: str, episodes: int) -> dict[str, float]:
    env = make_service_env(config)
    rollout_fn = random_rollout if algo == "random" else greedy_rollout
    seed = int(config.get("seed", 0)) + 50000
    rows = [rollout_fn(env, seed + index) for index in range(episodes)]
    summary = mean_metrics(rows)
    summary["eval_return"] = float(np.mean([row["episode_return"] for row in rows]))
    return summary


def _eval_trainer(config: dict[str, Any], run_dir: Path, algo: str, episodes: int) -> dict[str, float]:
    checkpoint = run_dir / "checkpoints" / "checkpoint_best_stochastic.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "checkpoints" / "best_stochastic.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing stochastic checkpoint: {run_dir}")
    config.setdefault("mappo", {})["eval_episodes"] = int(episodes)
    if algo == "mappo":
        trainer = ServiceMAPPOTrainer(config, run_dir / "_best_eval")
    elif algo == "slg_sage":
        trainer = SLGSAGEMAPPOTrainer(config, run_dir / "_best_eval")
    elif algo in {"madqn", "qmix", "wqmix", "qtran"}:
        trainer = ServiceValueMARLTrainer(config, run_dir / "_best_eval", algo=algo)
    elif algo in {"coma", "happo"}:
        trainer = ServiceOnPolicyMARLTrainer(config, run_dir / "_best_eval", algo=algo)
    elif algo in {"maddpg", "masac", "matd3"}:
        config.setdefault("env", {})["action_mode"] = "simple"
        trainer = ServiceContinuousMARLTrainer(config, run_dir / "_best_eval", algo=algo)
    else:
        raise ValueError(f"unsupported checkpoint eval algo: {algo}")
    trainer.load_checkpoint(checkpoint)
    return trainer.evaluate(seed_offset=50000, deterministic=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_paper")
    parser.add_argument("--policy_mode", choices=["stochastic"], default="stochastic")
    parser.add_argument("--eval_episodes", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    # for config_path in sorted(root.rglob("config.json")):
    #     run_dir = config_path.parent
    #     meta = _path_meta(config_path, root)
    #     config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
    #     summary_path = run_dir / "summary.json"
    #     if summary_path.exists():
    #         summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    #         if summary.get("status") == "adapter_failure":
    #             rows.append({**meta, "status": "adapter_failure", "reason": summary.get("reason", "")})
    #             continue
    #     try:
    #         if meta["algo"] in {"random", "greedy"}:
    #             metrics = _eval_heuristic(config, meta["algo"], args.eval_episodes)
    #         else:
    #             metrics = _eval_trainer(config, run_dir, meta["algo"], args.eval_episodes)
    #         rows.append({**meta, "status": "completed", **metrics})
    #     except Exception as exc:  # noqa: BLE001
    #         rows.append({**meta, "status": "eval_failure", "reason": repr(exc)})
    # write_rows(root / "paper_best_checkpoint_eval.csv", rows)
    # print(f"Wrote {root / 'paper_best_checkpoint_eval.csv'}")
    config_paths = sorted(root.rglob("config.json"))
    total = len(config_paths)

    for idx, config_path in enumerate(config_paths, start=1):
        run_dir = config_path.parent
        meta = _path_meta(config_path, root)

        print(
            f"[{idx}/{total}] evaluating "
            f"difficulty={meta['difficulty']}, "
            f"scale={meta['scale']}, "
            f"algo={meta['algo']}, "
            f"seed={meta['seed']}, "
            f"run={meta['run_name']}",
            flush=True
        )

        config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "summary.json"

        if summary_path.exists():
            summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "adapter_failure":
                row = {**meta, "status": "adapter_failure", "reason": summary.get("reason", "")}
                rows.append(row)
                print(row, flush=True)
                write_rows(root / "paper_best_checkpoint_eval_progress.csv", rows)
                continue

        try:
            if meta["algo"] in {"random", "greedy"}:
                metrics = _eval_heuristic(config, meta["algo"], args.eval_episodes)
            else:
                metrics = _eval_trainer(config, run_dir, meta["algo"], args.eval_episodes)

            row = {**meta, "status": "completed", **metrics}
            rows.append(row)
            print(row, flush=True)

        except Exception as exc:
            row = {**meta, "status": "eval_failure", "reason": repr(exc)}
            rows.append(row)
            print(row, flush=True)

        write_rows(root / "paper_best_checkpoint_eval_progress.csv", rows)


if __name__ == "__main__":
    main()
