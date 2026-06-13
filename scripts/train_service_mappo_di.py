"""Train DI-engine MAVAC MAPPO baseline on service offloading."""

from __future__ import annotations

import argparse
from pathlib import Path

from ServiceComputing.algorithms.service_mappo_di import ServiceMAPPOTrainer
from ServiceComputing.scripts.common import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ServiceComputing/configs/service_mappo_smoke.json")
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.total_env_steps is not None:
        cfg.setdefault("mappo", {})["total_env_steps"] = int(args.total_env_steps)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    method = "MAPPO-DI"
    seed = int(cfg.get("seed", 0))
    base = Path(cfg.get("output_dir", "artifacts/service_mappo_di"))
    run_name = cfg.get("run_name", "service_mappo")
    run_dir = base / method / f"seed_{seed}" / run_name
    trainer = ServiceMAPPOTrainer(cfg, run_dir)
    if args.resume_checkpoint is not None:
        trainer.load_checkpoint(Path(args.resume_checkpoint))
    summary = trainer.train()
    print("summary_path", run_dir / "summary.json")
    print("last_eval", summary.get("last_eval", {}))


if __name__ == "__main__":
    main()
