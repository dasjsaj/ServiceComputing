"""Inspect reset/step contract of CrossDomainServiceOffloadingEnv."""

from __future__ import annotations

import argparse
import numpy as np

from ServiceComputing.scripts.common import load_json
from ServiceComputing.service_offloading import make_service_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ServiceComputing/configs/service_mappo_smoke.json")
    parser.add_argument("--rollout_steps", type=int, default=50)
    args = parser.parse_args()
    cfg = load_json(args.config)
    env = make_service_env(cfg)
    obs, info = env.reset(seed=int(cfg.get("seed", 0)))
    rng = np.random.default_rng(0)
    rows = []
    reward_values = []
    done_seen = False
    truncated_seen = False
    current_obs = obs
    for _ in range(args.rollout_steps):
        if getattr(env, "action_mode", "simple") == "discrete_route":
            actions = {aid: int(rng.integers(env.action_dim)) for aid in current_obs}
        else:
            actions = {aid: rng.random(env.action_dim, dtype=np.float32) for aid in current_obs}
        current_obs, rewards, done, truncated, step_info = env.step(actions)
        rows.append(step_info["metrics"])
        reward_values.extend(float(v) for v in rewards.values())
        done_seen = done_seen or done
        truncated_seen = truncated_seen or truncated
        if done:
            break
    first = next(iter(obs.values()))
    print("agent_count", len(obs))
    print("obs_shape", first["obs"].shape)
    print("raw_shape", first["raw"].shape)
    print("semantic_shape", first["semantic"].shape)
    print("global_obs_shape", env.get_global_state().shape)
    print("action_dim", env.action_dim)
    print("action_mode", env.action_mode)
    print("obs_mode", env.obs_mode)
    print("reward_mode", env.reward_mode)
    print("difficulty", env.difficulty)
    print("env_model", getattr(env, "env_model", "legacy"))
    print("reward_finite", bool(np.isfinite(reward_values).all()))
    print("reward_min", f"{min(reward_values):.4f}" if reward_values else "nan")
    print("reward_max", f"{max(reward_values):.4f}" if reward_values else "nan")
    print("done_seen", done_seen, "truncated_seen", truncated_seen)
    print("info_keys", sorted(rows[-1].keys() if rows else []))
    if rows:
        print("generated_tasks", f"{np.mean([r.get('generated_tasks', 0.0) for r in rows]):.4f}")
        print("completed_tasks", f"{np.mean([r.get('completed_tasks', 0.0) for r in rows]):.4f}")
        print("timeout_tasks", f"{np.mean([r.get('timeout_tasks', 0.0) for r in rows]):.4f}")
        print("has_nan_inf", bool(any(not np.isfinite(float(v)) for r in rows for v in r.values())))


if __name__ == "__main__":
    main()
