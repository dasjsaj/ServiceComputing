"""Run random continuous actions in the service offloading environment."""

from __future__ import annotations

import argparse
import numpy as np

from ServiceComputing.scripts.common import load_json
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics


def rollout(env, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    episode_return = 0.0
    done = False
    while not done:
        if getattr(env, "action_mode", "simple") == "discrete_route":
            actions = {aid: int(rng.integers(env.action_dim)) for aid in obs}
        else:
            actions = {aid: rng.random(env.action_dim, dtype=np.float32) for aid in obs}
        obs, rewards, done, _, info = env.step(actions)
        episode_return += float(np.mean(list(rewards.values())))
        rows.append(info["metrics"])
    out = mean_metrics(rows)
    out["episode_return"] = episode_return
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ServiceComputing/configs/service_mappo_smoke.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()
    cfg = load_json(args.config)
    env = make_service_env(cfg)
    base_seed = int(cfg.get("seed", 0)) + args.seed_offset
    rows = [rollout(env, base_seed + i) for i in range(args.episodes)]
    summary = mean_metrics(rows)
    summary["episode_return"] = float(np.mean([r["episode_return"] for r in rows]))
    for key in [
        "episode_return",
        "completion_ratio",
        "mean_service_delay",
        "deadline_violation_rate",
        "mean_energy_cost",
        "offload_success_rate",
        "mean_queue_length",
    ]:
        print(key, f"{summary[key]:.4f}")


if __name__ == "__main__":
    main()
