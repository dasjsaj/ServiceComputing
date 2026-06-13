"""Run ServiceComputing paper experiment suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from ServiceComputing.scripts.paper_experiment_utils import parse_csv_list, parse_int_list, write_json, write_rows
from ServiceComputing.scripts.train_service_algo import run_algorithm

DEFAULT_CORE = ["random", "greedy", "mappo", "slg_sage", "qmix", "coma", "happo"]
DEFAULT_FULL = [
    "random",
    "greedy",
    "mappo",
    "slg_sage",
    "coma",
    "qmix",
    "wqmix",
    "madqn",
    "qtran",
    "happo",
    "maddpg",
    "masac",
    "matd3",
]


def default_config(difficulty: str, scale: str) -> str:
    return f"ServiceComputing/configs/service_paper_{difficulty}_{scale}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="core")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scales", default=None)
    parser.add_argument("--algos", default=None)
    parser.add_argument("--seeds", default="1,7,42")
    parser.add_argument("--total_env_steps", type=int, default=100000)
    parser.add_argument("--output_root", default="artifacts/service_paper")
    parser.add_argument("--policy_mode", choices=["stochastic"], default="stochastic")
    parser.add_argument("--eval_episodes", type=int, default=None)
    args = parser.parse_args()

    scales = parse_csv_list(args.scales, [args.scale or "medium"])
    if args.algos:
        algos = parse_csv_list(args.algos, [])
    elif args.suite == "full":
        algos = DEFAULT_FULL
    else:
        algos = DEFAULT_CORE
    seeds = parse_int_list(args.seeds, [1, 7, 42])

    rows = []
    for scale in scales:
        config = Path(default_config(args.difficulty, scale))
        for algo in algos:
            for seed in seeds:
                run_name = f"{algo}_{args.difficulty}_{scale}_seed{seed}_{args.total_env_steps // 1000}k"
                try:
                    summary = run_algorithm(
                        algo=algo,
                        config_path=config,
                        seed=seed,
                        total_env_steps=args.total_env_steps,
                        run_name=run_name,
                        output_root=Path(args.output_root),
                        difficulty=args.difficulty,
                        scale=scale,
                        eval_episodes=args.eval_episodes,
                    )
                    status = str(summary.get("status", "completed"))
                    best = summary.get("best_stochastic_eval_return")
                    reason = summary.get("reason", "")
                except Exception as exc:  # noqa: BLE001 - experiment runner must keep suite moving.
                    status = "run_failure"
                    best = None
                    reason = repr(exc)
                    summary = {"status": status, "reason": reason}
                row = {
                    "suite": args.suite,
                    "difficulty": args.difficulty,
                    "scale": scale,
                    "algo": algo,
                    "seed": seed,
                    "status": status,
                    "best_stochastic_eval_return": best,
                    "reason": reason,
                }
                rows.append(row)
                print(row, flush=True)

    root = Path(args.output_root)
    write_rows(root / f"paper_{args.suite}_run_status.csv", rows)
    write_json({"rows": rows}, root / f"paper_{args.suite}_run_status.json")


if __name__ == "__main__":
    main()
