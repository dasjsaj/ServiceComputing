import numpy as np
import pytest
import torch

from ServiceComputing.algorithms.slg_sage_mappo import SLGSAGEActorCritic
from ServiceComputing.algorithms.slg_sage_mappo import SLGSAGEMAPPO
from ServiceComputing.service_offloading import CrossDomainServiceOffloadingEnv
from ServiceComputing.service_offloading.semantic import semantic_features, semantic_prior_action


def test_minimal_obs_can_expose_semantic_side_channel_without_changing_policy_obs():
    env = CrossDomainServiceOffloadingEnv(
        {
            "env": {
                "n_auv": 4,
                "n_usv": 2,
                "n_uav": 2,
                "episode_length": 20,
                "difficulty": "easy",
                "action_mode": "simple",
                "obs_mode": "minimal",
                "reward_mode": "stable_v1",
                "use_semantic_side_channel": True,
            }
        }
    )
    obs, _ = env.reset(seed=0)
    first = obs[env.agent_ids[0]]

    assert env.obs_dim == 17
    assert env.action_dim == 4
    assert env.semantic_dim == 10
    assert first["obs"].shape == (17,)
    assert first["semantic"].shape == (10,)
    assert first["semantic_prior"].shape == (4,)
    assert np.isfinite(first["semantic"]).all()
    assert np.isfinite(first["semantic_prior"]).all()


def test_slg_actor_uses_full_policy_obs_and_separate_semantic_side_channel():
    model = SLGSAGEActorCritic(obs_dim=17, semantic_dim=10, action_dim=4, n_agents=8, hidden_dim=32)

    assert model.raw_encoder[0].in_features == 17

    obs = torch.rand(16, 17)
    semantic = torch.rand(16, 10)
    mean, std, prior, aux = model.policy(obs, semantic)

    assert mean.shape == (16, 4)
    assert std.shape == (16, 4)
    assert prior.shape == (16, 4)
    assert aux["success_logit"].shape == (16,)


def test_slg_actor_route_mean_is_simplex_for_simple_action_space():
    model = SLGSAGEActorCritic(obs_dim=17, semantic_dim=10, action_dim=4, n_agents=8, hidden_dim=32)
    mean, _, _, _ = model.policy(torch.rand(16, 17), torch.rand(16, 10))

    assert torch.allclose(mean[:, :3].sum(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.all(mean[:, :3] >= 0.0)
    assert torch.all(mean[:, :3] <= 1.0)


def test_slg_actor_accepts_configurable_initial_log_std():
    model = SLGSAGEActorCritic(
        obs_dim=17,
        semantic_dim=10,
        action_dim=4,
        n_agents=8,
        hidden_dim=32,
        initial_log_std=-1.4,
    )

    assert torch.allclose(model.log_std, torch.full((4,), -1.4))


def test_slg_update_logs_positive_entropy_and_grad_norm():
    cfg = {
        "seed": 0,
        "env": {
            "n_auv": 2,
            "n_usv": 1,
            "n_uav": 1,
            "episode_length": 5,
            "difficulty": "easy",
            "action_mode": "simple",
            "obs_mode": "minimal",
            "reward_mode": "stable_v1",
            "use_semantic_side_channel": True,
        },
        "algo": {
            "hidden_dim": 16,
            "update_epochs": 1,
            "minibatch_size": 64,
            "learning_rate": 3e-4,
        },
    }
    env = CrossDomainServiceOffloadingEnv(cfg)
    env.reset(seed=0)
    agent = SLGSAGEMAPPO(env, cfg)

    batch, _ = agent.collect_episode(seed=0, train=True)
    losses = agent.update(batch)

    assert "grad_norm" in losses
    assert losses["grad_norm"] >= 0.0
    assert losses["entropy_loss"] > 0.0


def test_auv_semantic_prior_uses_primary_route_and_raises_tx_power_under_pressure():
    raw = {
        "task_deadline": 0.45,
        "task_data_size": 0.75,
        "task_cpu_cycles": 0.75,
        "link_reliability": 0.35,
        "neighbor_queue_pressure": 0.85,
        "queue_length": 0.85,
        "remaining_energy": 0.8,
        "cpu_load": 0.65,
        "role_id": 0,
    }
    raw["semantic"] = semantic_features(raw)

    prior = semantic_prior_action(raw)

    assert prior[:3].sum() == pytest.approx(1.0, abs=1e-5)
    assert prior[:3].max() >= 0.70
    assert prior[3] >= 0.75


def test_auv_semantic_prior_prefers_local_when_offload_links_are_slow_and_queues_high():
    raw = {
        "task_deadline": 0.9,
        "task_data_size": 0.35,
        "task_cpu_cycles": 0.25,
        "link_reliability": 0.4,
        "link_rate": np.array([0.12, 0.10, 0.8], dtype=np.float32),
        "neighbor_queue_pressure": 0.9,
        "queue_length": 0.9,
        "remaining_energy": 0.9,
        "cpu_capacity": 0.8,
        "cpu_load": 0.1,
        "role_id": 0,
    }
    raw["semantic"] = semantic_features(raw)

    prior = semantic_prior_action(raw)

    assert prior[0] >= 0.70
    assert prior[0] == prior[:3].max()
