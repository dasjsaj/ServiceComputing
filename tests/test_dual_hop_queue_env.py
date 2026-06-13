import numpy as np

from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.scripts.evaluate_greedy_service_policy import greedy_actions


def make_env(**overrides):
    env_cfg = {
        "env_model": "dual_hop_queue",
        "n_auv": 1,
        "n_usv": 1,
        "n_uav": 1,
        "episode_length": 6,
        "task_arrival_rate": 0.0,
        "initial_tasks_per_auv": 1,
        "action_mode": "simple",
        "obs_mode": "minimal",
        "reward_mode": "stable_v1",
        "use_semantic_reward": False,
    }
    env_cfg.update(overrides)
    return make_service_env({"env": env_cfg})


def test_factory_selects_queue_aware_two_hop_environment():
    env = make_env()
    obs, _ = env.reset(seed=0)

    assert env.env_model == "dual_hop_queue"
    assert env.agent_ids == ["auv_0", "usv_0", "uav_0"]
    assert env.action_dim == 4
    assert all(row["obs"].shape == (env.obs_dim,) for row in obs.values())
    assert env.get_global_state().shape == (env.global_state_dim,)


def test_auv_offload_must_arrive_at_usv_before_uav_or_shore():
    env = make_env()
    env.reset(seed=1)
    actions = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }

    _, _, _, _, info = env.step(actions)

    assert len(env.queues["auv_0"]) == 0
    assert len(env.queues["usv_0"]) == 1
    assert len(env.queues["uav_0"]) == 0
    assert len(env.shore_queue) == 0
    assert info["metrics"]["auv_to_usv_tasks"] == 1.0
    assert info["metrics"]["usv_to_uav_tasks"] == 0.0


def test_usv_can_forward_a_task_and_uav_can_complete_it_on_later_steps():
    env = make_env(task_cpu_scale=0.1)
    env.reset(seed=2)
    upload = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    env.step(upload)
    forward = {
        "auv_0": np.zeros(4, dtype=np.float32),
        "usv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    _, _, _, _, forward_info = env.step(forward)
    compute = {
        "auv_0": np.zeros(4, dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    _, rewards, _, _, finish_info = env.step(compute)

    assert forward_info["metrics"]["usv_to_uav_tasks"] >= 1.0
    assert finish_info["metrics"]["completed_tasks"] >= 1.0
    assert all(np.isfinite(value) for value in rewards.values())
    for key in [
        "auv_queue_mean",
        "usv_queue_mean",
        "uav_queue_mean",
        "shore_queue_length",
        "completion_ratio",
        "deadline_violation_rate",
        "reward_backlog",
    ]:
        assert key in finish_info["metrics"]


def test_compute_scaling_can_make_edge_route_necessary_and_tracks_cumulative_offload():
    env = make_env(auv_cpu_scale=0.15, usv_cpu_scale=1.2, uav_cpu_scale=1.5, task_cpu_scale=2.0)
    env.reset(seed=3)
    assert env.nodes["auv_0"].cpu_capacity < env.nodes["usv_0"].cpu_capacity * 0.2
    local = {
        "auv_0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    _, _, _, _, local_info = env.step(local)

    assert local_info["metrics"]["completed_tasks"] == 0.0

    env.reset(seed=3)
    upload = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    _, _, _, _, upload_info = env.step(upload)

    assert upload_info["metrics"]["offload_success_rate"] == 1.0
    assert upload_info["metrics"]["first_hop_offloaded_tasks"] == 1.0


def test_greedy_forwarding_accounts_for_slow_downstream_compute():
    env = make_env(
        usv_cpu_scale=1.5,
        uav_cpu_scale=0.001,
        shore_cpu_scale=0.001,
        task_cpu_scale=2.0,
        task_data_scale=0.05,
        deadline_scale=20.0,
    )
    env.reset(seed=4)
    upload = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    env.step(upload)

    actions = greedy_actions(env)

    assert actions["usv_0"][0] == 1.0
    assert actions["usv_0"][1] == 0.0


def test_source_backlog_cost_rewards_progress_to_usv_queue():
    env = make_env(auv_cpu_scale=0.01, task_cpu_scale=2.0, deadline_scale=20.0)
    env.reset(seed=5)
    _, _, _, _, local_info = env.step(
        {
            "auv_0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "usv_0": np.zeros(4, dtype=np.float32),
            "uav_0": np.zeros(4, dtype=np.float32),
        }
    )
    env.reset(seed=5)
    _, _, _, _, upload_info = env.step(
        {
            "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "usv_0": np.zeros(4, dtype=np.float32),
            "uav_0": np.zeros(4, dtype=np.float32),
        }
    )

    assert upload_info["metrics"]["weighted_backlog_cost"] < local_info["metrics"]["weighted_backlog_cost"]
    assert upload_info["metrics"]["reward_backlog"] > local_info["metrics"]["reward_backlog"]


def test_downstream_relay_does_not_receive_free_backlog_progress_reward():
    common = {
        "auv_cpu_scale": 0.01,
        "usv_cpu_scale": 0.0,
        "uav_cpu_scale": 0.0,
        "task_cpu_scale": 2.0,
        "task_data_scale": 0.05,
        "deadline_scale": 20.0,
    }
    hold_env = make_env(**common)
    hold_env.reset(seed=6)
    upload = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    hold_env.step(upload)
    _, _, _, _, hold_info = hold_env.step(
        {
            "auv_0": np.zeros(4, dtype=np.float32),
            "usv_0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "uav_0": np.zeros(4, dtype=np.float32),
        }
    )

    relay_env = make_env(**common)
    relay_env.reset(seed=6)
    relay_env.step(upload)
    _, _, _, _, relay_info = relay_env.step(
        {
            "auv_0": np.zeros(4, dtype=np.float32),
            "usv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "uav_0": np.zeros(4, dtype=np.float32),
        }
    )

    assert relay_info["metrics"]["usv_to_uav_tasks"] == 1.0
    assert relay_info["metrics"]["reward_backlog"] <= hold_info["metrics"]["reward_backlog"]


def test_edge_compute_budget_can_complete_multiple_small_queued_tasks_in_one_slot():
    env = make_env(n_auv=2, usv_cpu_scale=10.0, task_cpu_scale=0.05, deadline_scale=20.0)
    env.reset(seed=7)
    upload = {
        "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "auv_1": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "usv_0": np.zeros(4, dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }
    env.step(upload)
    compute = {
        "auv_0": np.zeros(4, dtype=np.float32),
        "auv_1": np.zeros(4, dtype=np.float32),
        "usv_0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "uav_0": np.zeros(4, dtype=np.float32),
    }

    _, _, _, _, info = env.step(compute)

    assert info["metrics"]["completed_tasks"] >= 2.0
    assert len(env.queues["usv_0"]) == 0


def test_usv_observation_exposes_comparable_route_delay_estimates():
    env = make_env(
        usv_cpu_scale=0.01,
        uav_cpu_scale=10.0,
        shore_cpu_scale=0.01,
        task_cpu_scale=1.5,
        task_data_scale=0.05,
        deadline_scale=20.0,
    )
    env.reset(seed=8)
    env.step(
        {
            "auv_0": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "usv_0": np.zeros(4, dtype=np.float32),
            "uav_0": np.zeros(4, dtype=np.float32),
        }
    )

    usv_obs = env._obs_dict()["usv_0"]
    route_estimates = usv_obs["route_delay_estimates"]

    assert route_estimates.shape == (3,)
    assert np.allclose(usv_obs["obs"][-3:], route_estimates)
    assert route_estimates[1] < route_estimates[0]
    assert route_estimates[1] < route_estimates[2]


def test_discrete_route_actions_preserve_two_hop_execution_order():
    env = make_env(action_mode="discrete_route", task_cpu_scale=0.1)
    env.reset(seed=9)
    env.step({"auv_0": 1, "usv_0": 0, "uav_0": 0})
    assert env.action_mode == "discrete_route"
    assert len(env.queues["usv_0"]) == 1

    _, _, _, _, relay_info = env.step({"auv_0": 0, "usv_0": 1, "uav_0": 0})
    assert relay_info["metrics"]["usv_to_uav_tasks"] == 1.0
    assert len(env.queues["uav_0"]) == 1

    _, _, _, _, finish_info = env.step({"auv_0": 0, "usv_0": 0, "uav_0": 0})
    assert finish_info["metrics"]["completed_tasks"] >= 1.0


def test_greedy_policy_supports_discrete_route_action_contract():
    env = make_env(action_mode="discrete_route", auv_cpu_scale=0.01, task_cpu_scale=1.5)
    env.reset(seed=10)

    actions = greedy_actions(env)

    assert all(isinstance(action, int) for action in actions.values())
    assert all(0 <= action < env.action_dim for action in actions.values())
    _, rewards, _, _, _ = env.step(actions)
    assert all(np.isfinite(value) for value in rewards.values())


def test_discrete_route_includes_idle_action_that_does_not_implicitly_offload():
    env = make_env(action_mode="discrete_route", auv_cpu_scale=0.01, task_cpu_scale=1.5)
    env.reset(seed=11)

    env.step({"auv_0": 3, "usv_0": 3, "uav_0": 3})

    assert env.action_dim == 4
    assert len(env.queues["auv_0"]) == 1
    assert len(env.queues["usv_0"]) == 0


def test_auv_route_estimates_align_with_its_two_usv_upload_actions():
    env = make_env(action_mode="discrete_route", n_usv=2, n_uav=0, task_cpu_scale=0.5)
    obs, _ = env.reset(seed=12)

    routes = obs["auv_0"]["route_delay_estimates"]

    assert routes[1] < 1.0
    assert routes[2] < 1.0
    env.step({"auv_0": 2, "usv_0": 3, "usv_1": 3})
    assert len(env.queues["usv_1"]) == 1


def test_role_action_mask_is_available_without_semantic_side_channel():
    env = make_env(action_mode="discrete_route", n_usv=1, use_semantic_side_channel=False)
    obs, _ = env.reset(seed=13)

    assert np.array_equal(obs["auv_0"]["action_mask"], np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
    assert np.array_equal(obs["auv_0"]["semantic_action_mask"], obs["auv_0"]["action_mask"])
    assert np.array_equal(obs["usv_0"]["action_mask"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert np.array_equal(obs["uav_0"]["action_mask"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_usv_forward_uav_action_is_masked_without_any_uav():
    env = make_env(action_mode="discrete_route", n_usv=1, n_uav=0, use_semantic_side_channel=False)
    env.reset(seed=14)
    env.step({"auv_0": 1, "usv_0": 3})

    usv_mask = env._obs_dict()["usv_0"]["action_mask"]

    assert np.array_equal(usv_mask, np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))


def test_step_reports_agent_level_route_progress_and_completion():
    env = make_env(action_mode="discrete_route", task_cpu_scale=0.05)
    env.reset(seed=15)

    _, _, _, _, upload_info = env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})
    _, _, _, _, compute_info = env.step({"auv_0": 3, "usv_0": 0, "uav_0": 3})

    assert upload_info["per_agent"]["auv_0"]["moved_task"] == 1.0
    assert upload_info["per_agent"]["auv_0"]["route_progress"] == 1.0
    assert upload_info["per_agent"]["usv_0"]["moved_task"] == 0.0
    assert compute_info["per_agent"]["usv_0"]["completed_task"] >= 1.0
    assert compute_info["per_agent"]["usv_0"]["energy_used"] > 0.0


def test_route_progress_reward_nonzero_when_enabled_and_task_moves():
    env = make_env(
        action_mode="discrete_route",
        use_route_progress_reward=True,
        reward_auv_to_usv=0.08,
    )
    env.reset(seed=16)

    _, _, _, _, info = env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})

    assert info["metrics"]["auv_to_usv_tasks"] == 1.0
    assert info["metrics"]["route_progress_per_step"] == 1.0
    assert info["metrics"]["reward_route_progress"] > 0.0


def test_route_progress_reward_is_disabled_by_default():
    env = make_env(action_mode="discrete_route")
    env.reset(seed=17)

    _, _, _, _, info = env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})

    assert info["metrics"]["reward_route_progress"] == 0.0


def test_energy_aware_teacher_adds_configured_action_energy_cost():
    common = {
        "action_mode": "discrete_route",
        "n_usv": 2,
        "use_semantic_side_channel": True,
        "use_task_aware_semantic_teacher": True,
    }
    base_env = make_env(**common, semantic_energy_cost_weight=0.0)
    energy_env = make_env(**common, semantic_energy_cost_weight=0.08)
    base_obs, _ = base_env.reset(seed=18)
    energy_obs, _ = energy_env.reset(seed=18)

    base_row = base_obs["auv_0"]
    energy_row = energy_obs["auv_0"]
    energy_cost = energy_row["semantic_energy_cost"]

    assert energy_cost.shape == (energy_env.action_dim,)
    assert energy_cost[3] == 0.0
    assert np.any(energy_cost[:3] > 0.0)
    assert np.allclose(
        energy_row["semantic_route_costs"][:3] - base_row["semantic_route_costs"][:3],
        0.08 * energy_cost[:3],
    )


def test_energy_accounting_includes_transferred_energy_while_task_is_inflight():
    env = make_env(action_mode="discrete_route")
    env.reset(seed=19)

    _, _, _, _, info = env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})
    metrics = info["metrics"]

    assert metrics["total_consumed_energy"] > 0.0
    assert metrics["completed_task_energy"] == 0.0
    assert metrics["timeout_task_energy"] == 0.0
    assert metrics["inflight_task_energy"] == metrics["total_consumed_energy"]
    assert metrics["energy_transfer_auv_usv"] == metrics["total_consumed_energy"]


def test_energy_accounting_preserves_energy_spent_on_timed_out_task():
    env = make_env(action_mode="discrete_route")
    env.reset(seed=1)
    env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})
    active_queue = next(queue for queue in env.queues.values() if queue)
    active_queue[0].deadline = 0.0

    _, _, _, _, info = env.step({"auv_0": 3, "usv_0": 3, "uav_0": 3})
    metrics = info["metrics"]

    assert metrics["timeout_tasks"] == 1.0
    assert metrics["timeout_task_energy"] > 0.0
    assert metrics["inflight_task_energy"] == 0.0
    assert metrics["total_consumed_energy"] == metrics["timeout_task_energy"]


def test_downstream_aware_semantic_side_channel_exposes_completion_estimates():
    env = make_env(
        action_mode="discrete_route",
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
    )
    obs, _ = env.reset(seed=20)
    row = obs["auv_0"]

    assert row["semantic_downstream_delay"].shape == (env.action_dim,)
    assert row["semantic_downstream_energy"].shape == (env.action_dim,)
    assert row["semantic_downstream_queue"].shape == (env.action_dim,)
    assert np.all(row["semantic_downstream_delay"] >= 0.0)
    assert np.all(row["semantic_downstream_delay"] <= 1.0)
    assert env.semantic_dim == 29
    assert row["semantic"].shape == (env.semantic_dim,)


def test_downstream_aware_teacher_changes_route_costs_with_completion_estimates():
    common = {
        "action_mode": "discrete_route",
        "use_semantic_side_channel": True,
        "use_task_aware_semantic_teacher": True,
        "semantic_downstream_delay_weight": 0.4,
        "semantic_downstream_energy_weight": 0.2,
        "semantic_downstream_queue_weight": 0.2,
    }
    base_env = make_env(**common, use_downstream_aware_semantic_teacher=False)
    downstream_env = make_env(**common, use_downstream_aware_semantic_teacher=True)
    base_obs, _ = base_env.reset(seed=21)
    downstream_obs, _ = downstream_env.reset(seed=21)

    base_costs = base_obs["auv_0"]["semantic_route_costs"]
    downstream_row = downstream_obs["auv_0"]

    assert not np.allclose(downstream_row["semantic_downstream_delay"][:3], 0.0)
    assert not np.allclose(downstream_row["semantic_route_costs"][:3], base_costs[:3])


def test_downstream_coordination_teacher_prefers_idle_uav_when_shore_is_congested():
    env = make_env(
        action_mode="discrete_route",
        n_auv=4,
        n_usv=2,
        n_uav=2,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=1.7,
        task_data_scale=0.38,
        usv_cpu_scale=1.0,
        uav_cpu_scale=1.55,
        shore_cpu_scale=1.75,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_downstream_coordination_teacher=True,
        semantic_downstream_parallelism_weight=0.35,
    )
    env.reset(seed=22)
    for _ in range(2):
        env.step(
            {
                "auv_0": 1,
                "auv_1": 2,
                "auv_2": 1,
                "auv_3": 2,
                "usv_0": 3,
                "usv_1": 3,
                "uav_0": 3,
                "uav_1": 3,
            }
        )
    for _ in range(12):
        env.shore_queue.append(env.queues["usv_0"][0])

    row = env._obs_dict()["usv_0"]

    assert row["semantic_downstream_parallelism_bonus"][1] > 0.0
    assert row["semantic_route_costs"][1] < row["semantic_route_costs"][2]


def test_downstream_coordination_teacher_does_not_use_uav_when_shore_is_faster_and_idle():
    env = make_env(
        action_mode="discrete_route",
        n_auv=4,
        n_usv=2,
        n_uav=2,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=1.7,
        task_data_scale=0.38,
        usv_cpu_scale=1.0,
        uav_cpu_scale=1.55,
        shore_cpu_scale=1.75,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_downstream_coordination_teacher=True,
        semantic_downstream_parallelism_weight=0.35,
    )
    env.reset(seed=22)
    for _ in range(2):
        env.step(
            {
                "auv_0": 1,
                "auv_1": 2,
                "auv_2": 1,
                "auv_3": 2,
                "usv_0": 3,
                "usv_1": 3,
                "uav_0": 3,
                "uav_1": 3,
            }
        )

    row = env._obs_dict()["usv_0"]

    assert row["semantic_downstream_delay"][2] < row["semantic_downstream_delay"][1]
    assert row["semantic_downstream_parallelism_bonus"][1] == 0.0
    assert row["semantic_route_costs"][2] < row["semantic_route_costs"][1]


def test_downstream_coordination_teacher_avoids_unnecessary_uav_to_shore_extra_hop():
    env = make_env(
        action_mode="discrete_route",
        n_auv=1,
        n_usv=1,
        n_uav=1,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=1.7,
        task_data_scale=0.38,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_downstream_coordination_teacher=True,
        semantic_downstream_parallelism_weight=0.35,
        semantic_extra_hop_penalty_weight=0.30,
    )
    env.reset(seed=30)
    env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})
    env.step({"auv_0": 3, "usv_0": 1, "uav_0": 3})

    row = env._obs_dict()["uav_0"]

    assert len(env.queues["uav_0"]) == 1
    assert env.queues["uav_0"][0].hops >= 2
    assert row["semantic_extra_hop_penalty"][1] > 0.0
    assert row["semantic_route_costs"][0] < row["semantic_route_costs"][1]


def test_marginal_completion_teacher_values_uav_local_compute_after_two_hop_arrival():
    env = make_env(
        action_mode="discrete_route",
        n_auv=1,
        n_usv=1,
        n_uav=1,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=0.8,
        task_data_scale=0.3,
        uav_cpu_scale=2.0,
        shore_cpu_scale=1.0,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_downstream_coordination_teacher=True,
        use_marginal_completion_teacher=True,
        semantic_marginal_completion_weight=0.35,
    )
    env.reset(seed=31)
    env.step({"auv_0": 1, "usv_0": 3, "uav_0": 3})
    env.step({"auv_0": 3, "usv_0": 1, "uav_0": 3})

    row = env._obs_dict()["uav_0"]

    assert len(env.queues["uav_0"]) == 1
    assert row["semantic_marginal_completion_value"][0] > row["semantic_marginal_completion_value"][1]
    assert row["semantic_route_costs"][0] < row["semantic_route_costs"][1]


def test_marginal_completion_teacher_prefers_shore_when_usv_local_cannot_finish():
    env = make_env(
        action_mode="discrete_route",
        n_auv=1,
        n_usv=1,
        n_uav=0,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=1.8,
        task_data_scale=0.2,
        usv_cpu_scale=0.05,
        shore_cpu_scale=2.0,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_marginal_completion_teacher=True,
        semantic_marginal_completion_weight=0.35,
    )
    env.reset(seed=32)
    env.step({"auv_0": 1, "usv_0": 3})

    row = env._obs_dict()["usv_0"]

    assert row["semantic_marginal_completion_value"][2] > row["semantic_marginal_completion_value"][0]
    assert row["semantic_deadline_risk"][2] <= row["semantic_deadline_risk"][0]
    assert row["semantic_route_costs"][2] < row["semantic_route_costs"][0]


def test_marginal_completion_teacher_gives_clear_bonus_to_terminal_local_compute():
    env = make_env(
        action_mode="discrete_route",
        n_auv=1,
        n_usv=1,
        n_uav=0,
        initial_tasks_per_auv=1,
        task_arrival_rate=0.0,
        task_cpu_scale=0.5,
        task_data_scale=0.3,
        usv_cpu_scale=2.0,
        shore_cpu_scale=2.0,
        use_semantic_side_channel=True,
        use_task_aware_semantic_teacher=True,
        use_downstream_aware_semantic_teacher=True,
        use_marginal_completion_teacher=True,
        semantic_marginal_completion_weight=0.25,
        semantic_terminal_compute_bonus_weight=0.20,
    )
    env.reset(seed=35)
    env.step({"auv_0": 1, "usv_0": 3})

    row = env._obs_dict()["usv_0"]

    assert row["semantic_terminal_compute_bonus"][0] > 0.0
    assert row["semantic_marginal_completion_value"][0] >= row["semantic_marginal_completion_value"][2] + 0.12
    assert row["semantic_route_costs"][0] <= row["semantic_route_costs"][2] - 0.10
