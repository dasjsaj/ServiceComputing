from __future__ import annotations

import numpy as np
import torch

from ServiceComputing.service_offloading import make_service_env


def make_semantic_env(**overrides):
    env_config = {
        "env_model": "dual_hop_queue",
        "action_mode": "discrete_route",
        "n_auv": 1,
        "n_usv": 2,
        "n_uav": 1,
        "episode_length": 5,
        "initial_tasks_per_auv": 1,
        "task_arrival_rate": 0.0,
        "use_semantic_side_channel": True,
    }
    env_config.update(overrides)
    return make_service_env(
        {
            "env": env_config
        }
    )


def test_semantic_side_channel_is_action_aligned_without_changing_baseline_obs():
    env = make_semantic_env()
    obs, _ = env.reset(seed=0)
    row = obs["auv_0"]

    assert env.obs_dim == 21
    assert env.semantic_dim > 0
    assert row["obs"].shape == (21,)
    assert row["semantic"].shape == (env.semantic_dim,)
    assert row["semantic_route_costs"].shape == (env.action_dim,)
    assert row["semantic_action_mask"].shape == (env.action_dim,)
    assert np.isfinite(row["semantic"]).all()
    assert row["semantic_action_mask"][1] == 1.0
    assert row["semantic_action_mask"][2] == 1.0
    assert row["semantic_route_costs"][3] > row["semantic_route_costs"][1]


def test_semantic_guidance_residual_starts_at_zero_and_teacher_masks_invalid_actions():
    from ServiceComputing.models.service_semantic_guidance import (
        DiscreteSemanticGuidance,
        semantic_teacher_distribution,
    )

    module = DiscreteSemanticGuidance(semantic_dim=17, action_dim=4, hidden_dim=16, zero_init_output=True)
    semantic = torch.rand(3, 17)
    output = module(semantic, torch.tensor([0, 1, 2]))
    teacher = semantic_teacher_distribution(
        torch.tensor([[0.4, 0.1, 0.2, 0.9]], dtype=torch.float32),
        torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        temperature=0.5,
    )

    assert torch.allclose(output["residual_logits"], torch.zeros((3, 4)))
    assert teacher.shape == (1, 4)
    assert teacher[0, 2] < 1e-7
    assert torch.allclose(teacher.sum(dim=-1), torch.ones(1))


def test_semantic_guidance_auxiliary_heads_receive_gradients():
    from ServiceComputing.models.service_semantic_guidance import DiscreteSemanticGuidance

    module = DiscreteSemanticGuidance(semantic_dim=17, action_dim=4, hidden_dim=16)
    output = module(torch.rand(5, 17), torch.tensor([0, 1, 2, 3, 0]))
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(output["completion_logit"], torch.ones(5))
        + torch.nn.functional.binary_cross_entropy_with_logits(output["deadline_logit"], torch.zeros(5))
        + torch.nn.functional.smooth_l1_loss(output["delay_pred"], torch.rand(5))
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.encoder.parameters())


def test_semantic_teacher_can_prefer_capable_usv_local_compute_over_unnecessary_forwarding():
    from ServiceComputing.models.service_semantic_guidance import semantic_teacher_distribution

    medium = {
        "difficulty": "medium_load",
        "task_data_scale": 0.38,
        "task_cpu_scale": 1.7,
        "usv_cpu_scale": 1.0,
        "uav_cpu_scale": 1.55,
        "shore_cpu_scale": 1.75,
    }

    def teacher_top_action(env):
        obs, _ = env.reset(seed=0)
        actions = {aid: 3 for aid in env.agent_ids}
        actions["auv_0"] = 1
        obs, _, _, _, _ = env.step(actions)
        usv_row = obs["usv_0"]
        teacher = semantic_teacher_distribution(
            torch.tensor(usv_row["semantic_route_costs"][None, :]),
            torch.tensor(usv_row["semantic_action_mask"][None, :]),
            temperature=0.5,
        )
        return usv_row, torch.argmax(teacher, dim=-1).item()

    base_row, base_top = teacher_top_action(make_semantic_env(**medium))
    usv_row, calibrated_top = teacher_top_action(
        make_semantic_env(**medium, semantic_usv_forward_coordination_penalty=0.5)
    )

    assert base_row["semantic"][7] == 1.0
    assert base_top == 2
    assert usv_row["semantic"][7] == 1.0
    assert calibrated_top == 0


def test_task_aware_teacher_costs_change_with_service_task_type():
    env = make_semantic_env(use_task_aware_semantic_teacher=True)
    env.reset(seed=18)
    env.queues["auv_0"][0].task_type = 0  # tracking_update: latency sensitive.
    tracking = env._obs_dict()["auv_0"]
    env.queues["auv_0"][0].task_type = 2  # sonar_recognition: compute intensive.
    sonar = env._obs_dict()["auv_0"]

    assert not np.allclose(tracking["semantic_route_costs"], sonar["semantic_route_costs"])
    assert not np.allclose(tracking["semantic_task_compatibility"], sonar["semantic_task_compatibility"])
    assert tracking["semantic_convergence_bonus"].shape == (env.action_dim,)
