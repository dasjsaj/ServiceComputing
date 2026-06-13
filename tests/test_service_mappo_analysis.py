from ServiceComputing.scripts import analyze_service_mappo_convergence as analysis


def test_random_convergence_baseline_uses_same_seed_offset_as_policy_evaluation(monkeypatch):
    seen_seeds = []
    monkeypatch.setattr(analysis, "make_service_env", lambda config: object())
    monkeypatch.setattr(
        analysis,
        "random_rollout",
        lambda env, seed: seen_seeds.append(seed) or {"episode_return": 0.0},
    )
    monkeypatch.setattr(analysis, "mean_metrics", lambda rows: {})

    analysis.random_baseline({"seed": 3}, episodes=2, seed_offset=10000)

    assert seen_seeds == [10003, 10004]


def test_stochastic_policy_mode_maps_logged_metrics_to_primary_analysis_columns():
    eval_rows = [
        {
            "step": 100.0,
            "eval_return": 1.0,
            "completion_ratio": 0.2,
            "stochastic_eval_return": 7.0,
            "stochastic_completion_ratio": 0.8,
            "stochastic_mean_service_delay": 1.4,
            "stochastic_deadline_violation_rate": 0.02,
            "stochastic_mean_queue_length": 0.4,
        }
    ]

    rows = analysis.evaluation_mode_rows(eval_rows, "stochastic")

    assert rows[0]["eval_return"] == 7.0
    assert rows[0]["completion_ratio"] == 0.8
    assert rows[0]["mean_service_delay"] == 1.4
