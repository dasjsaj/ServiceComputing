from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from ServiceComputing.algorithms.slg_sage_mappo_di import SLGSAGEMAPPOTrainer


RUN_DIR = Path("artifacts/service_tests/test_slg_sage_mappo_di")


def config() -> dict:
    return {
        "seed": 0,
        "env": {
            "env_model": "dual_hop_queue",
            "action_mode": "discrete_route",
            "n_auv": 1,
            "n_usv": 1,
            "n_uav": 1,
            "episode_length": 4,
            "initial_tasks_per_auv": 1,
            "task_arrival_rate": 0.0,
            "use_semantic_side_channel": True,
            "use_semantic_reward": False,
        },
        "mappo": {
            "hidden_dim": 32,
            "eval_episodes": 1,
            "rollout_steps": 4,
            "minibatch_steps": 2,
            "update_epochs": 1,
        },
        "semantic": {
            "hidden_dim": 16,
            "semantic_logit_scale_max": 0.3,
            "semantic_residual_warmup_steps": 1,
            "lambda_prior_0": 0.08,
            "lambda_guide_0": 0.03,
            "lambda_aux": 0.03,
            "prior_decay_steps": 100,
            "guide_decay_steps": 100,
        },
    }


def test_zero_initialized_semantic_residual_preserves_base_logits_then_can_change_them():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "guided_logits")
    obs, _ = trainer.env.reset(seed=1)
    obs_arr = trainer._obs_array(obs)
    global_arr = trainer._global_array()
    semantic = trainer._semantic_array(obs)
    base, final, guidance = trainer._guided_forward(obs_arr, global_arr, semantic)

    assert torch.allclose(final, base)
    trainer.semantic_step = 1
    with torch.no_grad():
        trainer.guidance.residual_head.bias.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    _, shifted, _ = trainer._guided_forward(obs_arr, global_arr, semantic)
    assert not torch.allclose(shifted, base)
    assert guidance["residual_logits"].shape[-1] == trainer.env.action_dim


def test_semantic_residual_only_changes_logits_for_agents_with_pending_tasks():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "task_active_residual")
    obs, _ = trainer.env.reset(seed=1)
    obs_arr = trainer._obs_array(obs)
    global_arr = trainer._global_array()
    semantic = trainer._semantic_array(obs)
    trainer.semantic_step = 1
    with torch.no_grad():
        trainer.guidance.residual_head.bias.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]))

    base, final, _ = trainer._guided_forward(obs_arr, global_arr, semantic)
    active = semantic[:, 7] > 0.5

    assert active.tolist() == [True, False, False]
    assert not torch.allclose(final[active], base[active])
    assert torch.allclose(final[~active], base[~active])


def test_semantic_loss_ignores_empty_queue_agents():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "task_active_loss")
    semantic = torch.zeros((1, 3, trainer.env.semantic_dim), dtype=torch.float32)
    semantic[0, 0, 7] = 1.0
    per_agent_loss = torch.tensor([[2.0, 100.0, 200.0]], dtype=torch.float32)

    loss = trainer._active_semantic_mean(per_agent_loss, semantic)

    assert torch.isclose(loss, torch.tensor(2.0))


def test_semantic_policy_guidance_can_fade_out_after_early_training_phase():
    cfg = config()
    cfg["semantic"]["semantic_residual_decay_start_steps"] = 2
    cfg["semantic"]["semantic_residual_decay_steps"] = 2
    trainer = SLGSAGEMAPPOTrainer(cfg, RUN_DIR / "late_fade")

    trainer.semantic_step = 1
    early_prior, early_guide = trainer.semantic_coefficients()
    assert trainer.semantic_logit_scale() > 0.0
    assert early_prior > 0.0
    assert early_guide > 0.0

    trainer.semantic_step = 4
    late_prior, late_guide = trainer.semantic_coefficients()
    assert late_prior > 0.0
    assert trainer.semantic_logit_scale() == 0.0
    assert late_guide == 0.0


def test_semantic_logit_scale_respects_configured_minimum_after_decay():
    cfg = config()
    cfg["semantic"]["semantic_residual_decay_start_steps"] = 2
    cfg["semantic"]["semantic_residual_decay_steps"] = 2
    cfg["semantic"]["semantic_min_logit_scale"] = 0.05
    trainer = SLGSAGEMAPPOTrainer(cfg, RUN_DIR / "late_floor")

    trainer.semantic_step = 10

    assert trainer.semantic_logit_scale() >= 0.05


def test_semantic_guidance_loss_can_be_restricted_to_residual_branch_for_ablation():
    cfg = config()
    cfg["semantic"]["guide_updates_base_actor"] = False
    trainer = SLGSAGEMAPPOTrainer(cfg, RUN_DIR / "residual_only_guidance")
    obs, _ = trainer.env.reset(seed=1)
    obs_t = torch.tensor(trainer._obs_array(obs)[None, ...], dtype=torch.float32)
    global_t = torch.tensor(trainer._global_array()[None, ...], dtype=torch.float32)
    semantic_t = torch.tensor(trainer._semantic_array(obs)[None, ...], dtype=torch.float32)
    actions = torch.zeros((1, len(trainer.env.agent_ids)), dtype=torch.long)
    trainer.semantic_step = 1
    base, _, _, guidance = trainer._guided_batch(obs_t, global_t, semantic_t, actions)
    teacher = torch.full_like(base, 0.25)

    loss = trainer._semantic_guidance_kl(base, guidance["residual_logits"], semantic_t, teacher)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert any(parameter.grad is not None for parameter in trainer.guidance.residual_head.parameters())


def test_semantic_guidance_loss_can_train_base_actor_when_enabled():
    cfg = config()
    cfg["semantic"]["guide_updates_base_actor"] = True
    trainer = SLGSAGEMAPPOTrainer(cfg, RUN_DIR / "joint_actor_guidance")
    obs, _ = trainer.env.reset(seed=1)
    obs_t = torch.tensor(trainer._obs_array(obs)[None, ...], dtype=torch.float32)
    global_t = torch.tensor(trainer._global_array()[None, ...], dtype=torch.float32)
    semantic_t = torch.tensor(trainer._semantic_array(obs)[None, ...], dtype=torch.float32)
    actions = torch.zeros((1, len(trainer.env.agent_ids)), dtype=torch.long)
    trainer.semantic_step = 1
    base, _, _, guidance = trainer._guided_batch(obs_t, global_t, semantic_t, actions)
    teacher = torch.full_like(base, 0.25)

    loss = trainer._semantic_guidance_kl(base, guidance["residual_logits"], semantic_t, teacher)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert any(parameter.grad is not None for parameter in trainer.model.parameters())
    assert any(parameter.grad is not None for parameter in trainer.guidance.residual_head.parameters())


def test_semantic_training_update_logs_finite_guidance_losses():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "semantic_losses")

    batch = trainer.collect()
    losses = trainer.update(batch)

    for key in [
        "semantic_prior_loss",
        "semantic_guidance_loss",
        "semantic_aux_loss",
        "lambda_prior",
        "lambda_guide",
        "semantic_logit_scale",
        "semantic_changed_argmax_rate",
        "semantic_policy_factor",
        "semantic_min_logit_scale",
        "semantic_residual_active",
        "deterministic_distill_loss",
    ]:
        assert key in losses
        assert np.isfinite(losses[key])


def test_semantic_evaluation_keeps_training_environment_isolated():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "eval_isolation")
    step_before = trainer.env.step_count

    metrics = trainer.evaluate(deterministic=False)

    assert trainer.env.step_count == step_before
    assert np.isfinite(metrics["eval_return"])
    assert "semantic_changed_argmax_rate" in metrics


def test_semantic_residual_cannot_make_masked_action_executable():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "semantic_execution_mask")
    obs, _ = trainer.env.reset(seed=16)
    trainer.semantic_step = 1
    with torch.no_grad():
        trainer.guidance.residual_head.bias.copy_(torch.tensor([0.0, 0.0, 1000.0, 0.0]))

    masks = trainer._action_mask_array(obs)
    base, final, _ = trainer._guided_forward(
        trainer._obs_array(obs), trainer._global_array(), trainer._semantic_array(obs), masks
    )

    assert base[0, 2] < -1e8
    assert final[0, 2] < -1e8
    assert torch.argmax(final[0]).item() != 2


def test_semantic_aux_completion_label_uses_per_agent_outcome():
    trainer = SLGSAGEMAPPOTrainer(config(), RUN_DIR / "agent_level_completion_label")
    info = {
        "metrics": {
            "reward_completion": 1.0,
            "reward_deadline": 0.0,
            "mean_service_delay": 0.4,
        },
        "per_agent": {
            "auv_0": {"completed_task": 0.0},
            "usv_0": {"completed_task": 1.0},
            "uav_0": {"completed_task": 0.0},
        },
    }

    completion, deadline, delay = trainer._auxiliary_label_arrays(info)

    assert np.array_equal(completion, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(deadline, np.zeros(3, dtype=np.float32))
    assert np.array_equal(delay, np.full(3, 0.4 / trainer.env.route_delay_normalizer, dtype=np.float32))


def test_existing_long_run_pair_has_shared_backbone_and_dense_route_reward():
    root = Path("ServiceComputing/configs")
    with (root / "service_mappo_dual_hop_medium_discrete_policy_modes.json").open(encoding="utf-8") as f:
        base = json.load(f)
    with (root / "service_slg_sage_dual_hop_short.json").open(encoding="utf-8") as f:
        sage = json.load(f)

    assert base["mappo"] == sage["mappo"]
    for key in [
        "use_route_progress_reward",
        "reward_auv_to_usv",
        "reward_usv_to_uav",
        "reward_usv_to_shore",
        "reward_uav_to_shore",
    ]:
        assert base["env"][key] == sage["env"][key]
    assert base["env"]["use_route_progress_reward"] is True
    assert sage["semantic"]["semantic_min_logit_scale"] == 0.05
    assert sage["semantic"]["lambda_deterministic_distill"] == 0.02
