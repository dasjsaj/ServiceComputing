"""Evaluate a simple delay-greedy service offloading policy."""

from __future__ import annotations

import argparse

import numpy as np

from ServiceComputing.scripts.common import load_json
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics


def dual_hop_greedy_actions(env) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    for aid in env.agent_ids:
        node = env.nodes[aid]
        action = np.zeros(env.action_dim, dtype=np.float32)
        if node.role == "auv":
            task = env.queues[aid][0] if env.queues[aid] else None
            if task is None:
                actions[aid] = action
                continue
            local_delay = task.remaining_cycles / max(0.05, node.cpu_capacity * env.compute_service_scale)
            usv_ids = env._role_ids("usv")
            if usv_ids:
                usv = min(usv_ids, key=lambda target: len(env.queues[target]))
                rate, _, prop = env._link_features(node, env.nodes[usv], "auv_usv")
                relay_delay = (
                    task.data_size / max(0.02, rate)
                    + prop
                    + len(env.queues[usv]) * env.slot_duration
                    + task.remaining_cycles / max(0.05, env.nodes[usv].cpu_capacity * env.compute_service_scale)
                )
            else:
                relay_delay = float("inf")
            if local_delay <= relay_delay:
                action[0] = 1.0
            else:
                action[1] = 1.0
                if env.action_dim > 3:
                    action[3] = 1.0
            actions[aid] = action
        elif node.role == "usv":
            task = env.queues[aid][0] if env.queues[aid] else None
            if task is None:
                actions[aid] = action
                continue
            local_delay = task.remaining_cycles / max(0.05, node.cpu_capacity * env.compute_service_scale)
            uav_ids = env._role_ids("uav")
            uav_delay = float("inf")
            if uav_ids:
                uav = min(uav_ids, key=lambda target: len(env.queues[target]))
                rate, _, prop = env._link_features(node, env.nodes[uav], "usv_uav")
                uav_delay = (
                    task.data_size / max(0.02, rate)
                    + prop
                    + len(env.queues[uav]) * env.slot_duration
                    + task.remaining_cycles / max(0.05, env.nodes[uav].cpu_capacity * env.compute_service_scale)
                )
            rate, _, prop = env._link_features(node, env.shore, "usv_shore")
            shore_delay = (
                task.data_size / max(0.02, rate)
                + prop
                + len(env.shore_queue) * env.slot_duration
                + task.remaining_cycles / max(0.05, env.shore.cpu_capacity * env.compute_service_scale)
            )
            choice = int(np.argmin([local_delay, uav_delay, shore_delay]))
            action[choice] = 1.0
            if env.action_dim > 3:
                action[3] = 1.0
            actions[aid] = action
        else:
            task = env.queues[aid][0] if env.queues[aid] else None
            if task is None:
                actions[aid] = action
                continue
            local_delay = task.remaining_cycles / max(0.05, node.cpu_capacity * env.compute_service_scale)
            rate, _, prop = env._link_features(node, env.shore, "uav_shore")
            shore_delay = (
                task.data_size / max(0.02, rate)
                + prop
                + len(env.shore_queue) * env.slot_duration
                + task.remaining_cycles / max(0.05, env.shore.cpu_capacity * env.compute_service_scale)
            )
            action[0 if local_delay <= shore_delay else 1] = 1.0
            if env.action_dim > 3:
                action[2] = 1.0
                action[3] = 1.0
            actions[aid] = action
    if getattr(env, "action_mode", "simple") == "discrete_route":
        return {aid: int(np.argmax(action[:3])) for aid, action in actions.items()}
    return actions


def greedy_actions(env) -> dict[str, np.ndarray]:
    if getattr(env, "env_model", "legacy") == "dual_hop_queue":
        return dual_hop_greedy_actions(env)
    actions: dict[str, np.ndarray] = {}
    for aid in env.agent_ids:
        node = env.nodes[aid]
        action = np.zeros(env.action_dim, dtype=np.float32)
        if node.role != "auv":
            action[0] = 1.0
            action[1] = 1.0
            if env.action_dim > 2:
                action[2] = 0.5
            actions[aid] = action
            continue
        task = env.pending_tasks.get(aid)
        if task is None:
            action[:3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            actions[aid] = action
            continue
        local_delay = env.local_execution_delay_scale * task.task_cpu_cycles / max(0.05, node.cpu_capacity * (1.0 - node.cpu_load))
        usv = env._nearest_role(node, "usv")
        uav = env._nearest_role(node, "uav")
        def estimate(dst, link_type: str) -> float:
            if dst is None:
                return 1e6
            rate, _, prop = env._link_features(node, dst, link_type)
            tx_delay = env.transmission_delay_scale * task.task_data_size / max(0.05, rate * 1.12) + prop
            queue_delay = 0.35 * dst.queue_length / env.max_queue
            exec_delay = env.edge_execution_delay_scale * task.task_cpu_cycles / max(0.05, dst.cpu_capacity * (1.0 - dst.cpu_load))
            return float(tx_delay + queue_delay + exec_delay)

        usv_delay = estimate(usv, "auv_usv")
        uav_delay = estimate(uav, "usv_uav")
        best = int(np.argmin([local_delay, usv_delay, uav_delay]))
        action[best] = 1.0
        action[3] = 0.8
        actions[aid] = action
    return actions


def rollout(env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    rows = []
    episode_return = 0.0
    done = False
    while not done:
        actions = greedy_actions(env)
        obs, rewards, done, _, info = env.step(actions)
        episode_return += float(np.mean(list(rewards.values())))
        rows.append(info["metrics"])
    out = mean_metrics(rows)
    out["episode_return"] = episode_return
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ServiceComputing/configs/service_mappo_smoke.json")
    parser.add_argument("--episodes", type=int, default=20)
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
