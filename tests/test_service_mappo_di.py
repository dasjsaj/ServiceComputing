from pathlib import Path
import csv

import numpy as np
import torch

from ServiceComputing.algorithms.service_mappo_di import ServiceMAPPOTrainer
from ServiceComputing.service_offloading import DualHopQueueServiceOffloadingEnv


TEST_RUN_DIR = Path("artifacts/service_tests/test_service_mappo_di")


def make_trainer() -> ServiceMAPPOTrainer:
    return ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {
                "episode_length": 4,
                "action_mode": "simple",
                "obs_mode": "minimal",
                "reward_mode": "stable_v1",
                "difficulty": "easy",
                "use_mobility_control": False,
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR,
    )


def test_action_metrics_report_edge_actions_by_role():
    trainer = make_trainer()
    actions = {}
    for aid in trainer.env.agent_ids:
        role = trainer.env.nodes[aid].role
        if role == "auv":
            actions[aid] = np.array([0.2, 0.3, 0.5, 0.4], dtype=np.float32)
        elif role == "usv":
            actions[aid] = np.array([0.8, 0.7, 0.6, 0.0], dtype=np.float32)
        else:
            actions[aid] = np.array([0.9, 0.55, 0.0, 0.0], dtype=np.float32)

    metrics = trainer._action_metrics(actions)

    assert metrics["usv_accept_ratio_mean"] == np.float32(0.8)
    assert metrics["usv_cpu_allocation_ratio_mean"] == np.float32(0.7)
    assert metrics["usv_relay_ratio_mean"] == np.float32(0.6)
    assert metrics["uav_accept_ratio_mean"] == np.float32(0.9)
    assert metrics["uav_cpu_allocation_ratio_mean"] == np.float32(0.55)


def test_save_checkpoint_records_model_and_eval_metrics():
    trainer = make_trainer()
    target = trainer._save_checkpoint(1024, {"eval_return": 1.25, "completion_ratio": 0.4})

    assert target.exists()
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["step"] == 1024
    assert payload["eval_metrics"]["eval_return"] == 1.25
    assert "model_state_dict" in payload


def test_trainer_selects_dual_hop_queue_environment_from_config():
    trainer = ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {
                "env_model": "dual_hop_queue",
                "n_auv": 1,
                "n_usv": 1,
                "n_uav": 1,
                "episode_length": 4,
                "initial_tasks_per_auv": 1,
                "task_arrival_rate": 0.0,
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR / "dual_hop_factory",
    )

    assert isinstance(trainer.env, DualHopQueueServiceOffloadingEnv)
    assert trainer.env.global_state_dim == trainer.env.obs_dim * 3 + trainer.env.global_extra_dim


def test_dual_hop_action_metrics_use_routing_semantics():
    trainer = ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {
                "env_model": "dual_hop_queue",
                "n_auv": 1,
                "n_usv": 1,
                "n_uav": 1,
                "episode_length": 4,
                "task_arrival_rate": 0.0,
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR / "dual_hop_action_metrics",
    )
    actions = {
        "auv_0": np.array([0.25, 0.75, 0.4, 0.9], dtype=np.float32),
        "usv_0": np.array([0.2, 0.6, 0.2, 0.8], dtype=np.float32),
        "uav_0": np.array([0.3, 0.7, 0.8, 0.9], dtype=np.float32),
    }

    metrics = trainer._action_metrics(actions)

    assert metrics["auv_upload_usv_ratio_mean"] == np.float32(0.75)
    assert metrics["usv_forward_uav_preference_mean"] == np.float32(0.6)
    assert metrics["usv_forward_shore_preference_mean"] == np.float32(0.2)
    assert metrics["uav_forward_shore_preference_mean"] == np.float32(0.7)


def test_discrete_route_trainer_uses_categorical_mappo_and_updates():
    trainer = ServiceMAPPOTrainer(
        {
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
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1, "rollout_steps": 4, "minibatch_steps": 2, "update_epochs": 1},
        },
        TEST_RUN_DIR / "dual_hop_discrete_route",
    )

    assert trainer.model.action_space == "discrete"
    batch = trainer.collect()
    assert batch.actions.shape == (4, 3)
    assert batch.action_masks.shape == (4, 3, trainer.env.action_dim)
    assert np.array_equal(batch.action_masks[0, 0], np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
    losses = trainer.update(batch)
    assert np.isfinite(losses["total_loss"])
    stochastic_eval = trainer.evaluate(deterministic=False)
    assert np.isfinite(stochastic_eval["eval_return"])


def test_learning_rate_anneals_only_after_a_positive_best_policy_degrades():
    trainer = ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {"episode_length": 4},
            "mappo": {
                "hidden_dim": 32,
                "learning_rate": 1e-4,
                "anneal_after_positive_eval_drop": True,
                "anneal_factor": 0.2,
                "anneal_min_learning_rate": 1e-5,
            },
        },
        TEST_RUN_DIR / "anneal_after_positive_eval",
    )
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.best_eval_return = -1.0
    trainer._maybe_anneal_learning_rate(-2.0)
    assert trainer.optimizer.param_groups[0]["lr"] == initial_lr

    trainer.best_eval_return = 2.0
    trainer._maybe_anneal_learning_rate(1.0)
    assert trainer.optimizer.param_groups[0]["lr"] == initial_lr * 0.2


def test_load_checkpoint_resumes_step_and_best_validation_state():
    source = make_trainer()
    target = source._save_checkpoint(2048, {"eval_return": 3.5, "completion_ratio": 0.75})
    restored = make_trainer()

    restored.load_checkpoint(target)

    assert restored.initial_env_steps == 2048
    assert restored.best_eval_return == 3.5
    assert restored.best_eval_metrics["completion_ratio"] == 0.75


def test_training_can_log_stochastic_evaluation_and_select_its_best_checkpoint():
    run_dir = TEST_RUN_DIR / "stochastic_eval_logging"
    trainer = ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {
                "env_model": "dual_hop_queue",
                "action_mode": "discrete_route",
                "n_auv": 1,
                "n_usv": 1,
                "n_uav": 1,
                "episode_length": 3,
                "initial_tasks_per_auv": 1,
                "task_arrival_rate": 0.0,
            },
            "mappo": {
                "hidden_dim": 32,
                "eval_episodes": 1,
                "rollout_steps": 2,
                "minibatch_steps": 2,
                "update_epochs": 1,
                "total_env_steps": 2,
                "eval_freq": 2,
                "report_stochastic_eval": True,
            },
        },
        run_dir,
    )

    summary = trainer.train()

    with (run_dir / "eval_curve.csv").open("r", encoding="utf-8", newline="") as handle:
        eval_row = list(csv.DictReader(handle))[-1]
    assert "stochastic_eval_return" in eval_row
    assert "best_stochastic_eval_return" in summary
    assert (run_dir / "checkpoints" / "checkpoint_best_stochastic.pt").exists()


def test_evaluation_does_not_reset_or_advance_the_training_environment():
    trainer = ServiceMAPPOTrainer(
        {
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
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR / "eval_env_isolation",
    )
    training_step_before_eval = trainer.env.step_count

    trainer.evaluate()

    assert trainer.env.step_count == training_step_before_eval


def test_discrete_route_logits_mask_invalid_actions_for_execution():
    trainer = ServiceMAPPOTrainer(
        {
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
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR / "execution_mask",
    )
    obs, _ = trainer.env.reset(seed=15)
    masks = trainer._action_mask_array(obs)
    logits, _, _ = trainer._forward(trainer._obs_array(obs), trainer._global_array(), masks)

    assert logits[0, 2] < -1e8
    assert torch.argmax(logits[1]).item() == 3
    assert torch.argmax(logits[2]).item() == 3


def test_discrete_action_metrics_do_not_count_idle_as_routing():
    trainer = ServiceMAPPOTrainer(
        {
            "seed": 0,
            "env": {
                "env_model": "dual_hop_queue",
                "action_mode": "discrete_route",
                "n_auv": 1,
                "n_usv": 1,
                "n_uav": 1,
                "episode_length": 4,
            },
            "mappo": {"hidden_dim": 32, "eval_episodes": 1},
        },
        TEST_RUN_DIR / "discrete_action_metrics",
    )

    metrics = trainer._action_metrics({"auv_0": 3, "usv_0": 3, "uav_0": 3})

    assert metrics["auv_upload_usv_ratio_mean"] == 0.0
    assert metrics["uav_forward_shore_preference_mean"] == 0.0
