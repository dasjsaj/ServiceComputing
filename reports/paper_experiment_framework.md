# ServiceComputing Paper Experiment Framework

This framework records paper results with stochastic policy evaluation only.
All runs save `train_curve.csv`, `eval_curve.csv`, `summary.json`,
`config.json`, and `checkpoints/checkpoint_best_stochastic.pt` when the
algorithm has trainable parameters.

## Runnable Algorithms

Discrete route setting:

- `random`
- `greedy`
- `mappo`
- `slg_sage`
- `madqn`
- `qmix`
- `wqmix`
- `qtran`
- `coma`
- `happo`

Continuous simple-action setting:

- `maddpg`
- `masac`
- `matd3`

The continuous-control baselines automatically switch the environment to
`action_mode=simple`. Their results should be reported as continuous-control
comparisons, not as identical discrete-route action-space comparisons.

## Scale Configs

- `small`: 2 AUV, 1 USV, 1 UAV
- `medium`: 4 AUV, 2 USV, 2 UAV
- `large`: 6 AUV, 3 USV, 3 UAV

Configs live under:

```text
ServiceComputing/configs/service_paper_<difficulty>_<scale>.json
```

## Commands

补跑 medium scale 的 MAPPO 与 SLG-SAGE：

```powershell
python -m ServiceComputing.scripts.run_paper_experiments `
  --suite core `
  --difficulty medium `
  --scale medium `
  --algos mappo,slg_sage `
  --seeds 7,42 `
  --total_env_steps 100000 `
  --policy_mode stochastic
```

规模扩展：

```powershell
python -m ServiceComputing.scripts.run_paper_experiments `
  --suite scale `
  --difficulty medium `
  --scales small,medium,large `
  --algos random,greedy,mappo,slg_sage,qmix,coma,happo `
  --seeds 1,7,42 `
  --total_env_steps 100000 `
  --policy_mode stochastic
```

高难度鲁棒性：

```powershell
python -m ServiceComputing.scripts.run_paper_experiments `
  --suite hard `
  --difficulty hard `
  --scales small,medium,large `
  --algos random,greedy,mappo,slg_sage,qmix,coma,happo `
  --seeds 1,7,42 `
  --total_env_steps 100000 `
  --policy_mode stochastic
```

完整算法补充：

```powershell
python -m ServiceComputing.scripts.run_paper_experiments `
  --suite full `
  --difficulty medium `
  --scale medium `
  --algos wqmix,madqn,qtran,maddpg,masac,matd3 `
  --seeds 1,7,42 `
  --total_env_steps 100000 `
  --policy_mode stochastic
```

Best stochastic checkpoint 复测：

```powershell
python -m ServiceComputing.scripts.evaluate_best_checkpoints `
  --root artifacts/service_paper `
  --policy_mode stochastic `
  --eval_episodes 50
```

汇总和画图：

```powershell
python -m ServiceComputing.scripts.summarize_service_results --root artifacts/service_paper
python -m ServiceComputing.scripts.plot_paper_results --root artifacts/service_paper --smooth_window 10
```
