"""Smoke trainer for SLG-SAGE-MAPPO service offloading experiments."""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch

from ServiceComputing.algorithms.slg_sage_mappo import SLGSAGEMAPPO
from ServiceComputing.scripts.common import load_json, run_dir, write_json
from ServiceComputing.service_offloading import CrossDomainServiceOffloadingEnv
from ServiceComputing.service_offloading.metrics import mean_metrics


def write_row(path, row):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ServiceComputing/configs/service_slg_sage_mappo_smoke.json")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.total_steps is not None:
        cfg["total_steps"] = int(args.total_steps)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.eval_interval is not None:
        cfg["eval_interval"] = int(args.eval_interval)
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    method = cfg.get("method", "slg_sage_mappo_full")
    out_dir = run_dir(cfg, method, seed)
    try:
        for stale in ["train_curve.csv", "eval_curve.csv", "summary.json"]:
            stale_path = out_dir / stale
            if stale_path.exists():
                stale_path.unlink()
    except PermissionError:
        out_dir = out_dir.parent / f"{out_dir.name}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "config.json")
    write_json({"seed": seed}, out_dir / "seed.json")
    env = CrossDomainServiceOffloadingEnv(cfg)
    env.reset(seed=seed)
    agent = SLGSAGEMAPPO(env, cfg)
    total_steps = int(cfg.get("total_steps", 5000))
    eval_interval = int(cfg.get("eval_interval", 1000))
    episode = 0
    env_steps = 0
    start = time.perf_counter()
    train_curve = out_dir / "train_curve.csv"
    eval_curve = out_dir / "eval_curve.csv"
    while env_steps < total_steps:
        batch, metrics = agent.collect_episode(seed=seed * 100000 + episode, train=True)
        losses = agent.update(batch)
        env_steps += len(metrics)
        episode += 1
        row = {
            "step": env_steps,
            "episode": episode,
            "train_return": float(np.sum([m["reward"] for m in metrics])),
            **mean_metrics(metrics),
            **losses,
        }
        write_row(train_curve, row)
        if env_steps == len(metrics) or env_steps % eval_interval < len(metrics) or env_steps >= total_steps:
            eval_metrics = agent.evaluate(episodes=int(cfg.get("eval_episodes", 3)), seed=seed)
            eval_row = {
                "step": env_steps,
                "episode": episode,
                "eval_return": float(eval_metrics.get("reward", 0.0) * cfg.get("env", {}).get("episode_length", 100)),
                **eval_metrics,
                "semantic_prior_loss": losses.get("semantic_prior_loss", 0.0),
                "semantic_aux_loss": losses.get("semantic_aux_loss", 0.0),
                "semantic_prior_coef": losses.get("semantic_prior_coef", 0.0),
            }
            write_row(eval_curve, eval_row)
            print(
                f"step={env_steps} episode={episode} train_return={row['train_return']:.3f} "
                f"completion={row['completion_ratio']:.3f} eval_completion={eval_metrics.get('completion_ratio', 0.0):.3f} "
                f"sem_coef={losses.get('semantic_prior_coef', 0.0):.4f}",
                flush=True,
            )
    summary = {
        "method": method,
        "seed": seed,
        "total_steps": env_steps,
        "episodes": episode,
        "runtime_sec": time.perf_counter() - start,
        "output_dir": str(out_dir),
    }
    write_json(summary, out_dir / "summary.json")
    print(f"Saved service run to {out_dir}")


if __name__ == "__main__":
    main()
