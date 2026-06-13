# Service-Semantic Reinforcement Learning for Service-Chain Scheduling in Air--Sea--Underwater Edge Networks

This repository contains the codebase for the paper:

> **Service-Semantic Reinforcement Learning for Service-Chain Scheduling in Air--Sea--Underwater Edge Networks**

The project studies multi-agent service-chain scheduling in a cross-domain edge computing system where underwater AUVs generate computation tasks, USVs act as acoustic-to-radio gateways and surface edge nodes, UAVs provide aerial edge computing, and a shore-side server offers high-capacity edge/cloud execution.

The main method is **SLG-SAGE-MAPPO**: **Semantic-Loss-Guided Service-Aware Graph/Guidance-Enhanced MAPPO**. The key idea is to keep the environment reward unchanged and inject service semantics into policy learning through a masked task-aware teacher, semantic residual logits, and auxiliary service-outcome prediction losses.

---

## Highlights

- **Air--sea--underwater edge service-chain environment**
  - AUV task generation.
  - AUV-to-USV acoustic first-hop access.
  - USV local processing, USV-to-UAV forwarding, and USV-to-shore forwarding.
  - UAV local processing and UAV-to-shore forwarding.
  - Queueing, task deadlines, service delay, energy accounting, and completion metrics.

- **Role-aware discrete route action space**
  - Fixed four-action abstraction for AUVs, USVs, and UAVs.
  - Role-dependent action semantics and role-feasibility masks.
  - Same policy/action dimension across small, medium, and large system scales.

- **SLG-SAGE-MAPPO**
  - MAPPO backbone with centralized training and decentralized execution.
  - Service-semantic side channel.
  - Task-aware masked semantic teacher.
  - Zero-initialized residual route logits added to MAPPO base logits.
  - Semantic prior/guidance losses and auxiliary completion/deadline/delay losses.

- **Baselines and experiment tools**
  - Random and Greedy policies.
  - MAPPO.
  - COMA, QMIX, WQMIX, MADQN, QTRAN.
  - MADDPG, MASAC, MATD3 continuous-control references.
  - Paper-level experiment runner, best-checkpoint evaluator, result summarizer, and plotting scripts.

---

## Repository Structure

```text
ServiceComputing/
├── algorithms/
│   ├── service_mappo_di.py              # MAPPO baseline trainer
│   ├── slg_sage_mappo_di.py             # SLG-SAGE-MAPPO trainer
│   ├── service_value_baselines.py        # QMIX, WQMIX, MADQN, QTRAN adapters
│   ├── service_policy_baselines.py       # COMA / on-policy MARL adapters
│   ├── service_continuous_baselines.py   # MADDPG, MASAC, MATD3 references
│   └── slg_sage_mappo.py                # lightweight continuous SLG-SAGE prototype
├── configs/
│   ├── service_paper_medium_small.json
│   ├── service_paper_medium_medium.json
│   ├── service_paper_medium_large.json
│   ├── service_paper_hard_small.json
│   ├── service_paper_hard_medium.json
│   ├── service_paper_hard_large.json
│   └── service_slg_sage_*.json
├── models/
│   └── service_semantic_guidance.py      # semantic teacher / guidance modules
├── scripts/
│   ├── inspect_service_env.py            # environment sanity check
│   ├── evaluate_random_service_policy.py
│   ├── evaluate_greedy_service_policy.py
│   ├── train_service_algo.py             # unified single-algorithm entrypoint
│   ├── run_paper_experiments.py          # batch paper experiment runner
│   ├── evaluate_best_checkpoints.py      # best stochastic checkpoint evaluation
│   ├── summarize_service_results.py      # result summarization
│   └── plot_paper_results.py             # paper figures
├── service_offloading/
│   ├── queue_env.py                      # dual-hop queue-aware service environment
│   ├── env.py                            # legacy/simple service offloading environment
│   ├── scenario.py                       # node/task generation
│   ├── semantic.py                       # semantic feature helpers
│   └── metrics.py                        # metric aggregation helpers
├── tests/
└── README.md
```

---

## System Model

The default simulator is a **dual-hop queue-aware service offloading environment**:

```text
AUV task source
    ↓ acoustic access
USV gateway / surface edge
    ↓ radio forwarding
UAV aerial edge or shore-side edge server
```

Each task has:

- task type,
- data size,
- remaining CPU cycles,
- deadline,
- priority,
- source AUV,
- elapsed time,
- accumulated energy,
- hop count.

The environment tracks:

- task completion ratio,
- average service delay,
- deadline violation rate,
- queue length,
- offloading success,
- route progress,
- energy breakdown,
- semantic-policy diagnostics when enabled.

---

## Action Space

The main experiments use a role-aware discrete route action space:

```text
A = {0, 1, 2, 3}
```

The meaning of each action depends on the agent role.

| Role | Action 0 | Action 1 | Action 2 | Action 3 |
|---|---|---|---|---|
| AUV | local compute | offload to primary candidate USV | offload to secondary candidate USV | idle |
| USV | local compute | forward to selected UAV candidate | forward to shore | idle |
| UAV | local compute | forward to shore | invalid | idle |

Invalid actions are masked before the categorical distribution is built. If an agent has no head-of-line task, only the idle action is feasible.

The fixed four-action abstraction keeps the action dimension independent of the number of AUV/USV/UAV nodes. When more service nodes exist, candidate-node selection is handled by the environment's route candidate logic rather than by expanding the policy output dimension.

---

## Service Semantics

In this project, "semantics" does **not** mean semantic communication, semantic compression, or natural-language content similarity.

Service semantics are structured decision signals extracted from the service-chain state:

- agent role,
- task type,
- task urgency,
- data intensity,
- computation intensity,
- local queue pressure,
- downstream queue pressure,
- route delay estimates,
- route energy estimates,
- deadline risk,
- marginal completion value,
- terminal-compute preference.

SLG-SAGE-MAPPO uses these semantics to construct a task-aware masked teacher distribution and a residual-logit guidance branch. The environment reward is not replaced by a semantic reward in the main setting.

---

## Main Method: SLG-SAGE-MAPPO

SLG-SAGE-MAPPO keeps MAPPO as the backbone policy learner:

```text
base_logits = MAPPO_actor(obs)
semantic_residual = SemanticGuidance(semantic_features)
final_logits = base_logits + semantic_scale(t) * semantic_residual
```

The final logits are masked by the role-feasibility action mask and sampled stochastically during evaluation.

The training objective combines:

- PPO clipped policy loss,
- centralized value loss,
- entropy regularization,
- semantic prior/guidance loss,
- deterministic-distillation guidance loss,
- auxiliary completion/deadline/delay prediction losses.

The semantic branch is zero-initialized and warmed up, so early training starts close to vanilla MAPPO. Semantic coefficients decay over training to avoid permanently hard-coding the teacher.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

If you are developing from the original monorepo, run commands from the directory that contains `ServiceComputing/`.

### 2. Create a Python environment

Python 3.10 or later is recommended.

```bash
conda create -n service-semantic-rl python=3.10 -y
conda activate service-semantic-rl
```

### 3. Install dependencies

Minimal dependencies:

```bash
pip install numpy pandas matplotlib torch
```

For testing:

```bash
pip install pytest
```

If you use the DI-engine-compatible components in your local setup, install the corresponding DI-engine dependencies according to your platform.

---

## Quick Start

Run all commands from the parent directory that contains `ServiceComputing/`.

### 1. Inspect the environment

```bash
python -m ServiceComputing.scripts.inspect_service_env \
  --config ServiceComputing/configs/service_paper_medium_medium.json \
  --rollout_steps 100
```

This checks observation shape, global state shape, action dimension, reward finiteness, done logic, and metric keys.

### 2. Evaluate random policy

```bash
python -m ServiceComputing.scripts.evaluate_random_service_policy \
  --config ServiceComputing/configs/service_paper_medium_medium.json \
  --episodes 20
```

### 3. Evaluate greedy policy

```bash
python -m ServiceComputing.scripts.evaluate_greedy_service_policy \
  --config ServiceComputing/configs/service_paper_medium_medium.json \
  --episodes 20
```

### 4. Train MAPPO

```bash
python -m ServiceComputing.scripts.train_service_algo \
  --algo mappo \
  --config ServiceComputing/configs/service_paper_medium_medium.json \
  --seed 1 \
  --total_env_steps 100000 \
  --run_name mappo_medium_medium_seed1_100k \
  --policy_mode stochastic
```

### 5. Train SLG-SAGE-MAPPO

```bash
python -m ServiceComputing.scripts.train_service_algo \
  --algo slg_sage \
  --config ServiceComputing/configs/service_slg_sage_dual_hop_short.json \
  --seed 1 \
  --total_env_steps 100000 \
  --run_name slg_sage_medium_medium_seed1_100k \
  --policy_mode stochastic
```

Outputs are written under:

```text
artifacts/service_paper/<difficulty>/<scale>/<algo>/seed_<seed>/<run_name>/
```

Typical files:

```text
train_curve.csv
eval_curve.csv
summary.json
config.json
checkpoints/
```

---

## Paper-Style Experiments

The paper experiments evaluate multiple algorithms, system scales, and seeds.

### Medium difficulty, medium scale

```bash
python -m ServiceComputing.scripts.run_paper_experiments \
  --suite core \
  --difficulty medium \
  --scale medium \
  --algos random,greedy,mappo,slg_sage,qmix,coma \
  --seeds 1,7,42 \
  --total_env_steps 100000 \
  --policy_mode stochastic
```

### Scale sensitivity

```bash
python -m ServiceComputing.scripts.run_paper_experiments \
  --suite scale \
  --difficulty medium \
  --scales small,medium,large \
  --algos random,greedy,mappo,slg_sage,qmix,coma,wqmix,madqn,qtran \
  --seeds 1,7,42 \
  --total_env_steps 100000 \
  --policy_mode stochastic
```

### Full baseline set

```bash
python -m ServiceComputing.scripts.run_paper_experiments \
  --suite full \
  --difficulty medium \
  --scale medium \
  --algos random,greedy,mappo,slg_sage,coma,qmix,wqmix,madqn,qtran,maddpg,masac,matd3 \
  --seeds 1,7,42 \
  --total_env_steps 100000 \
  --policy_mode stochastic
```

The current paper workflow reports **stochastic policy evaluation**.

---

## Best-Checkpoint Evaluation

After training, evaluate the best stochastic checkpoints:

```bash
python -m ServiceComputing.scripts.evaluate_best_checkpoints \
  --root artifacts/service_paper \
  --policy_mode stochastic \
  --eval_episodes 50
```

Then summarize:

```bash
python -m ServiceComputing.scripts.summarize_service_results \
  --root artifacts/service_paper
```

Plot paper figures:

```bash
python -m ServiceComputing.scripts.plot_paper_results \
  --root artifacts/service_paper \
  --smooth_window 10
```

---

## Configurations

The main paper configuration files are:

| File | Purpose |
|---|---|
| `service_paper_medium_small.json` | medium difficulty, small scale |
| `service_paper_medium_medium.json` | medium difficulty, medium scale |
| `service_paper_medium_large.json` | medium difficulty, large scale |
| `service_paper_hard_small.json` | hard difficulty, small scale |
| `service_paper_hard_medium.json` | hard difficulty, medium scale |
| `service_paper_hard_large.json` | hard difficulty, large scale |
| `service_slg_sage_dual_hop_short.json` | SLG-SAGE-MAPPO semantic-guidance configuration |

Scale definitions:

| Scale | AUVs | USVs | UAVs |
|---|---:|---:|---:|
| small | 2 | 1 | 1 |
| medium | 4 | 2 | 2 |
| large | 6 | 3 | 3 |

---

## Metrics

The main reported metrics are:

- average evaluation return,
- task completion ratio,
- average service delay,
- deadline violation rate,
- average queue length,
- key mobile transmission energy.

The key mobile transmission energy is defined as:

```text
AUV-to-USV acoustic access energy + UAV-to-shore forwarding energy
```

USV-side forwarding and computation energy are excluded from this primary metric because USVs are modeled as provisioned surface service units. The simulator still records additional energy components for diagnostics.

---

## Testing

Run all tests:

```bash
python -m pytest ServiceComputing/tests -q
```

Run a targeted environment smoke check:

```bash
python -m ServiceComputing.scripts.inspect_service_env \
  --config ServiceComputing/configs/service_paper_medium_medium.json \
  --rollout_steps 100
```

---

## Notes on Reproducibility

- Each training run is controlled by a seed.
- Paper results are intended to use seeds `1`, `7`, and `42`.
- Checkpoints and raw experiment artifacts can be large and are not recommended for normal Git commits.
- Use `.gitignore` to exclude `artifacts/`, `checkpoints/`, `*.pt`, `*.pth`, and log files when publishing the code.

Suggested `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.pytest_cache/
artifacts/
runs/
outputs/
checkpoints/
*.pt
*.pth
*.pkl
*.log
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{service_semantic_rl_2026,
  title   = {Service-Semantic Reinforcement Learning for Service-Chain Scheduling in Air--Sea--Underwater Edge Networks},
  author  = {Shengchao Zhu, Guangjie Han, Chuan Lin, and Yuan Liu},
  journal = {Under Review IEEE TON},
  year    = {2026}
}
```

Please replace the placeholder author and venue fields with the final publication information.

---

## License

Please add a license before public release. If you are unsure, common choices are:

- MIT License for permissive academic/code reuse,
- Apache-2.0 for permissive reuse with explicit patent terms,
- GPL-style licenses if derivative-code openness is required.

---

## Acknowledgement

This project builds a cross-domain service-computing simulator and MARL training pipeline inspired by edge offloading and multi-agent reinforcement learning research. The experiment workflow is compatible with DI-engine-style MAPPO components and supports paper-level stochastic checkpoint evaluation.

