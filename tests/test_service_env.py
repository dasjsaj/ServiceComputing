import numpy as np
import pytest

from ServiceComputing.service_offloading import CrossDomainServiceOffloadingEnv
from ServiceComputing.service_offloading.scenario import ServiceTask


def test_service_env_reset_step_contract():
    env = CrossDomainServiceOffloadingEnv({"env": {"episode_length": 5}})
    obs, info = env.reset(seed=0)
    assert len(obs) == 8
    first = next(iter(obs.values()))
    assert first["obs"].shape == (env.obs_dim,)
    assert first["semantic"].shape == (env.semantic_dim,)
    actions = {aid: np.ones(env.action_dim, dtype=np.float32) * 0.5 for aid in obs}
    next_obs, rewards, done, truncated, step_info = env.step(actions)
    assert len(next_obs) == len(obs)
    assert all(np.isfinite(v) for v in rewards.values())
    assert not truncated
    for key in [
        "completion_ratio",
        "mean_service_delay",
        "deadline_violation_rate",
        "mean_energy_cost",
        "offload_success_rate",
        "semantic_match_rate",
    ]:
        assert key in step_info["metrics"]


def test_service_env_minimal_simple_stable_contract():
    env = CrossDomainServiceOffloadingEnv(
        {
            "env": {
                "episode_length": 5,
                "action_mode": "simple",
                "obs_mode": "minimal",
                "reward_mode": "stable_v1",
                "difficulty": "easy",
                "use_mobility_control": False,
            }
        }
    )
    obs, _ = env.reset(seed=0)
    first = next(iter(obs.values()))
    assert env.action_dim == 4
    assert env.obs_dim == 17
    assert first["obs"].shape == (17,)
    assert first["semantic"].shape == (0,)
    assert env.get_global_state().shape == (env.global_state_dim,)
    actions = {aid: np.ones(env.action_dim, dtype=np.float32) * 0.5 for aid in obs}
    _, rewards, _, truncated, step_info = env.step(actions)
    assert not truncated
    assert all(np.isfinite(v) for v in rewards.values())
    for key in [
        "reward_completion",
        "reward_delay",
        "reward_deadline",
        "reward_energy",
        "reward_invalid_action",
        "reward_queue_overflow",
        "reward_total_mean",
    ]:
        assert key in step_info["metrics"]


def test_offload_relevant_profile_preserves_compute_heavy_tasks_and_edge_advantage():
    env = CrossDomainServiceOffloadingEnv(
        {
            "env": {
                "difficulty": "easy",
                "task_profile_mode": "full",
                "task_data_scale": 0.5,
                "task_cpu_scale": 1.2,
                "auv_cpu_scale": 0.6,
                "edge_cpu_scale": 1.3,
                "action_mode": "simple",
                "obs_mode": "minimal",
                "reward_mode": "stable_v1",
                "use_mobility_control": False,
            }
        }
    )
    env.reset(seed=0)
    task = ServiceTask(
        task_id=1,
        source_agent_id="auv_0",
        task_type=3,
        task_data_size=0.30,
        task_cpu_cycles=0.75,
        task_deadline=1.0,
        task_priority=1.1,
    )
    task = env._calibrate_task(task)
    auv = env.nodes["auv_0"]
    uav = env.nodes["uav_0"]
    uav.position = auv.position.copy()

    local_delay = 0.35 * task.task_cpu_cycles / (auv.cpu_capacity * (1.0 - auv.cpu_load))
    offload_delay, _, _ = env._offload_delay(auv, uav, task.task_data_size, task.task_cpu_cycles, "usv_uav", 0.8)

    assert task.task_type == 3
    assert task.task_data_size == 0.15
    assert task.task_cpu_cycles == pytest.approx(0.9)
    assert auv.cpu_capacity < 0.35
    assert uav.cpu_capacity > 0.9
    assert offload_delay < local_delay


def test_configurable_delay_model_reduces_edge_latency_without_changing_defaults():
    common = {
        "difficulty": "easy",
        "action_mode": "simple",
        "obs_mode": "minimal",
        "reward_mode": "stable_v1",
        "use_mobility_control": False,
    }
    default_env = CrossDomainServiceOffloadingEnv({"env": common})
    calibrated_env = CrossDomainServiceOffloadingEnv(
        {
            "env": {
                **common,
                "transmission_delay_scale": 0.18,
                "edge_execution_delay_scale": 0.20,
                "local_execution_delay_scale": 0.50,
            }
        }
    )
    default_env.reset(seed=0)
    calibrated_env.reset(seed=0)
    for env in [default_env, calibrated_env]:
        env.nodes["uav_0"].position = env.nodes["auv_0"].position.copy()
    data, cycles = 0.25, 0.85
    default_offload, _, _ = default_env._offload_delay(
        default_env.nodes["auv_0"], default_env.nodes["uav_0"], data, cycles, "usv_uav", 0.8
    )
    calibrated_offload, _, _ = calibrated_env._offload_delay(
        calibrated_env.nodes["auv_0"], calibrated_env.nodes["uav_0"], data, cycles, "usv_uav", 0.8
    )
    calibrated_local = (
        calibrated_env.local_execution_delay_scale
        * cycles
        / calibrated_env.nodes["auv_0"].cpu_capacity
    )

    assert default_env.transmission_delay_scale == 0.45
    assert calibrated_offload < default_offload
    assert calibrated_offload < calibrated_local


def test_task_type_intensity_scales_create_route_tradeoff_without_changing_defaults():
    env = CrossDomainServiceOffloadingEnv(
        {
            "env": {
                "difficulty": "easy",
                "task_profile_mode": "full",
                "task_type_data_scales": [0.8, 1.0, 0.65, 0.7],
                "task_type_cpu_scales": [0.7, 1.0, 1.45, 1.55],
                "auv_cpu_scale": 1.2,
                "edge_cpu_scale": 1.3,
                "transmission_delay_scale": 0.08,
                "edge_execution_delay_scale": 0.18,
                "local_execution_delay_scale": 0.25,
                "action_mode": "simple",
                "obs_mode": "minimal",
                "reward_mode": "stable_v1",
                "use_mobility_control": False,
            }
        }
    )
    default_env = CrossDomainServiceOffloadingEnv({"env": {"difficulty": "easy", "task_profile_mode": "full"}})
    env.reset(seed=0)
    default_env.reset(seed=0)
    tracking = ServiceTask(1, "auv_0", 0, 0.20, 0.25, 0.90, 1.2)
    replanning = ServiceTask(2, "auv_0", 3, 0.50, 0.85, 1.05, 1.1)
    tracking = env._calibrate_task(tracking)
    replanning = env._calibrate_task(replanning)
    default_replanning = default_env._calibrate_task(ServiceTask(2, "auv_0", 3, 0.50, 0.85, 1.05, 1.1))
    source = env.nodes["auv_0"]
    uav = env.nodes["uav_0"]

    tracking_local = env.local_execution_delay_scale * tracking.task_cpu_cycles / source.cpu_capacity
    tracking_edge = env._offload_delay(source, uav, tracking.task_data_size, tracking.task_cpu_cycles, "usv_uav", 0.8)[0]
    replanning_local = env.local_execution_delay_scale * replanning.task_cpu_cycles / source.cpu_capacity
    replanning_edge = env._offload_delay(source, uav, replanning.task_data_size, replanning.task_cpu_cycles, "usv_uav", 0.8)[0]

    assert replanning.task_cpu_cycles > default_replanning.task_cpu_cycles
    assert tracking_local < tracking_edge
    assert replanning_edge < replanning_local
