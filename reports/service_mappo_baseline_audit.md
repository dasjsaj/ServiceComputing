# Service Offloading MAPPO Baseline Audit

This audit pauses all semantic-loss / SAGE work and checks whether the
CrossDomainServiceOffloadingEnv can support a learnable native MAPPO baseline.

## Environment Contract

- Package: `ServiceComputing`, independent from `Tracking/AUV6DOF`.
- Environment: `CrossDomainServiceOffloadingEnv`.
- Default MAPPO audit mode:
  - `difficulty = easy`
  - `action_mode = simple`
  - `obs_mode = minimal`
  - `reward_mode = stable_v1`
  - `use_mobility_control = false`
  - `use_semantic_reward = false`
- Agent count: 8 (`4 AUV + 2 USV + 2 UAV`).
- Local observation shape: 17.
- Global observation shape: 141.
- Continuous action shape: 4.
- Reward range under random smoke rollout: finite, approximately `[-0.245, 0.728]`.

## Key Calibration

The first easy environment was too forgiving: random, greedy, and MAPPO had
similar completion ratios, and deadline violation was always zero. To make
service decisions meaningful while keeping the task learnable, the environment
now requires AUV jobs to choose a primary execution route:

```text
primary_route_ratio = max(local_compute_ratio, offload_to_usv_ratio, offload_to_uav_ratio)
completed iff primary_route_ratio >= min_primary_route_ratio
```

Default `min_primary_route_ratio` is `0.52` for easy mode.

This prevents a random policy from completing tasks simply by spreading small
fractions across all routes.

## Baselines

Command:

```powershell
python -m ServiceComputing.scripts.evaluate_random_service_policy --config ServiceComputing/configs/service_mappo_smoke.json --episodes 20
```

Random policy:

| metric | value |
|---|---:|
| episode_return | -0.8430 |
| completion_ratio | 0.2457 |
| mean_service_delay | 0.1639 |
| deadline_violation_rate | 0.0000 |
| mean_energy_cost | 0.0316 |
| offload_success_rate | 1.0000 |
| mean_queue_length | 1.6723 |

Command:

```powershell
python -m ServiceComputing.scripts.evaluate_greedy_service_policy --config ServiceComputing/configs/service_mappo_smoke.json --episodes 20
```

Delay-greedy policy:

| metric | value |
|---|---:|
| episode_return | 6.6647 |
| completion_ratio | 0.4180 |
| mean_service_delay | 0.0249 |
| deadline_violation_rate | 0.0000 |
| mean_energy_cost | 0.0377 |
| offload_success_rate | 1.0000 |
| mean_queue_length | 0.0000 |

The calibrated environment now has a meaningful random-greedy gap.

## MAPPO Smoke / Debug

The trainer uses DI-engine's `MAVAC` continuous multi-agent actor-critic model
with a lightweight MAPPO rollout/update harness. DI-engine core files were not
modified.

Command:

```powershell
python -m ServiceComputing.scripts.train_service_mappo_di --config ServiceComputing/configs/service_mappo_smoke.json --total_env_steps 10000 --run_name service_mappo_10k_calibrated_debug
```

Artifacts:

```text
artifacts/service_mappo_di/MAPPO-DI/seed_0/service_mappo_10k_calibrated_debug/
```

Evaluation trend:

| step | eval_return | completion_ratio | mean_service_delay | deadline_violation_rate |
|---:|---:|---:|---:|---:|
| 128 | -19.1391 | 0.0000 | 0.4615 | 0.0000 |
| 2048 | -19.4282 | 0.0000 | 0.4665 | 0.0000 |
| 4096 | 4.9549 | 0.4068 | 0.0701 | 0.0000 |
| 6016 | 4.9552 | 0.4068 | 0.0701 | 0.0000 |
| 8064 | 4.9552 | 0.4068 | 0.0701 | 0.0000 |
| 10112 | 4.9558 | 0.4068 | 0.0701 | 0.0000 |

MAPPO improves from a failed deterministic policy to a usable policy within
roughly 4k environment steps. It clearly exceeds random on return, completion,
delay, and queue length, but it remains below greedy on return and delay.

## Diagnosis

- Reset/step contract: OK.
- Observation/action shapes: fixed and MAPPO-compatible.
- Reward: finite and stable; no explosion observed.
- Done/truncated logic: episode termination works; truncated remains false.
- Service metrics: completion, delay, deadline, energy, offload, queue, and reward components are logged.
- MAPPO learnability: initial evidence is positive.
- Deadline signal: currently too easy because deadline violation stays at 0. This should be tightened after confirming longer MAPPO stability.
- Completion metric: now action-sensitive after primary-route calibration.
- Greedy gap: present, useful for checking whether MAPPO has room to improve.

## Recommendation

Do not add semantic loss yet. First run a longer debug train, for example
`50k-100k` environment steps, and confirm:

1. MAPPO stays above random.
2. Eval return continues toward greedy.
3. Value loss remains bounded.
4. Entropy does not collapse too early.
5. A slightly harder deadline setting introduces nonzero but learnable SLA violation.

Suggested next command:

```powershell
python -m ServiceComputing.scripts.train_service_mappo_di --config ServiceComputing/configs/service_mappo_debug.json --total_env_steps 50000 --run_name service_mappo_50k_debug
```

## Long-Test Update

The 100k single-seed MAPPO long test on easy mode passed the convergence gate.

Command:

```powershell
python -m ServiceComputing.scripts.train_service_mappo_di --config ServiceComputing/configs/service_mappo_debug.json --total_env_steps 100000 --run_name service_mappo_100k_debug_seed0
python -m ServiceComputing.scripts.analyze_service_mappo_convergence --run_dir artifacts/service_mappo_di/MAPPO-DI/seed_0/service_mappo_100k_debug_seed0
```

Key results:

| metric | value |
|---|---:|
| convergence passed | True |
| eval_return_auc | 6.6211 |
| N90_eval_return | 10240 |
| last_20pct_eval_return_mean | 9.7678 |
| last_20pct_completion_mean | 0.4136 |
| last_20pct_delay_mean | 0.0677 |
| value_loss_max | 4.3091 |
| entropy_last_mean | 5.9036 |

This confirms that the environment is learnable for MAPPO under the current
simple/minimal/stable-v1 setup.

## SLA Pressure Calibration

Because easy mode keeps deadline violation almost always at zero, a configurable
`deadline_scale` was added. Two SLA pressure settings were checked:

| config | random deadline violation | greedy deadline violation | MAPPO diagnostic |
|---|---:|---:|---|
| `deadline_scale=0.75` | 0.0163 | 0.0000 | baseline gap preserved |
| `deadline_scale=0.70` | 0.0377 | 0.0000 | 20k MAPPO passed |

The `deadline_scale=0.70` 20k run passed convergence gates, but `value_loss_max`
rose to `9.4711`, near the current threshold of `10`. This is a useful pressure
setting for the next baseline validation, but it should be monitored carefully
before adding SLG-SAGE.

The recommended 100k SLA-pressure baseline was also run:

```powershell
python -m ServiceComputing.scripts.train_service_mappo_di --config ServiceComputing/configs/service_mappo_sla070_debug.json --total_env_steps 100000 --run_name service_mappo_100k_sla070_seed0
python -m ServiceComputing.scripts.analyze_service_mappo_convergence --run_dir artifacts/service_mappo_di/MAPPO-DI/seed_0/service_mappo_100k_sla070_seed0
```

Key results:

| metric | value |
|---|---:|
| convergence passed | True |
| eval_return_auc | 6.5552 |
| N90_eval_return | 10240 |
| last_20pct_eval_return_mean | 9.7679 |
| last_20pct_completion_mean | 0.4136 |
| last_20pct_delay_mean | 0.0677 |
| value_loss_max | 9.4711 |
| entropy_last_mean | 5.7906 |

This confirms that `deadline_scale=0.70` is still learnable for MAPPO. However,
the learned policy drives evaluation deadline violation back to zero, so SLA is
now useful mainly as an early-training/random-policy pressure signal. For a
future paper setting, the environment still needs a richer SLA stressor such as
moderate packet loss, queue bursts, or harder task arrivals.

SLG-SAGE can be re-enabled next on `service_mappo_sla070_debug.json`, but its
claim should be tested against this stable MAPPO baseline using curve metrics
and not only final return.
