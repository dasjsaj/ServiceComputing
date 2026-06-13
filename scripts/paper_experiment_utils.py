"""Utilities for ServiceComputing paper-scale experiment runners."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ServiceComputing.scripts.common import load_json

SCALE_PRESETS: dict[str, dict[str, int]] = {
    "small": {"n_auv": 2, "n_usv": 1, "n_uav": 1},
    "medium": {"n_auv": 4, "n_usv": 2, "n_uav": 2},
    "large": {"n_auv": 6, "n_usv": 3, "n_uav": 3},
    "xlarge": {"n_auv": 8, "n_usv": 4, "n_uav": 4},
}

SUPPORTED_TRAINERS = {
    "mappo",
    "slg_sage",
    "madqn",
    "qmix",
    "wqmix",
    "qtran",
    "coma",
    "happo",
    "maddpg",
    "masac",
    "matd3",
}
HEURISTIC_ALGOS = {"random", "greedy"}
KNOWN_ALGOS = {
    "random",
    "greedy",
    "mappo",
    "slg_sage",
    "coma",
    "qmix",
    "wqmix",
    "madqn",
    "qtran",
    "happo",
    "maddpg",
    "masac",
    "matd3",
}

UNSUPPORTED_REASON = (
    "ServiceComputing adapter is not implemented yet. This runner records the "
    "failure explicitly instead of reusing another algorithm or fabricating results."
)


def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    return [int(token) for token in parse_csv_list(value, [str(item) for item in default])]


def infer_difficulty(config: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    return str(config.get("paper", {}).get("difficulty", config.get("env", {}).get("difficulty", "medium")))


def infer_scale(config: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    paper_scale = config.get("paper", {}).get("scale")
    if paper_scale:
        return str(paper_scale)
    env = config.get("env", {})
    triple = (int(env.get("n_auv", 4)), int(env.get("n_usv", 2)), int(env.get("n_uav", 2)))
    for name, values in SCALE_PRESETS.items():
        if triple == (values["n_auv"], values["n_usv"], values["n_uav"]):
            return name
    return "custom"


def apply_scale(config: dict[str, Any], scale: str) -> dict[str, Any]:
    if scale not in SCALE_PRESETS:
        return config
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("env", {}).update(SCALE_PRESETS[scale])
    cfg.setdefault("paper", {})["scale"] = scale
    return cfg


def apply_run_overrides(
    config: dict[str, Any],
    *,
    seed: int,
    total_env_steps: int | None,
    eval_episodes: int | None,
    run_name: str,
    output_root: Path,
    difficulty: str,
    scale: str,
    algo: str,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg["seed"] = int(seed)
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(output_root / difficulty / scale / algo)
    cfg.setdefault("paper", {}).update({"difficulty": difficulty, "scale": scale, "algo": algo})
    if total_env_steps is not None:
        cfg.setdefault("mappo", {})["total_env_steps"] = int(total_env_steps)
    if eval_episodes is not None:
        cfg.setdefault("mappo", {})["eval_episodes"] = int(eval_episodes)
    cfg.setdefault("mappo", {})["report_stochastic_eval"] = True
    return cfg


def ensure_sage_defaults(config: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    env = cfg.setdefault("env", {})
    env["use_semantic_side_channel"] = True
    env.setdefault("use_task_aware_semantic_teacher", True)
    env.setdefault("use_downstream_aware_semantic_teacher", True)
    env.setdefault("use_downstream_coordination_teacher", True)
    env.setdefault("use_marginal_completion_teacher", True)
    env.setdefault("semantic_task_compatibility_weight", 0.15)
    env.setdefault("semantic_convergence_bonus_weight", 0.10)
    env.setdefault("semantic_downstream_delay_weight", 0.25)
    env.setdefault("semantic_downstream_energy_weight", 0.08)
    env.setdefault("semantic_downstream_queue_weight", 0.08)
    env.setdefault("semantic_downstream_parallelism_weight", 0.35)
    env.setdefault("semantic_extra_hop_penalty_weight", 0.30)
    env.setdefault("semantic_marginal_completion_weight", 0.25)
    env.setdefault("semantic_deadline_risk_weight", 0.10)
    env.setdefault("semantic_terminal_compute_bonus_weight", 0.20)
    cfg.setdefault("semantic", {}).update(
        {
            "hidden_dim": int(cfg.get("semantic", {}).get("hidden_dim", 64)),
            "zero_init_semantic_output": bool(cfg.get("semantic", {}).get("zero_init_semantic_output", True)),
            "semantic_logit_scale_max": float(cfg.get("semantic", {}).get("semantic_logit_scale_max", 0.3)),
            "semantic_min_logit_scale": float(cfg.get("semantic", {}).get("semantic_min_logit_scale", 0.05)),
            "semantic_residual_warmup_steps": int(
                cfg.get("semantic", {}).get("semantic_residual_warmup_steps", 5000)
            ),
            "semantic_residual_decay_start_steps": cfg.get("semantic", {}).get(
                "semantic_residual_decay_start_steps", None
            ),
            "lambda_prior_0": float(cfg.get("semantic", {}).get("lambda_prior_0", 0.08)),
            "lambda_guide_0": float(cfg.get("semantic", {}).get("lambda_guide_0", 0.03)),
            "lambda_aux": float(cfg.get("semantic", {}).get("lambda_aux", 0.03)),
            "lambda_deterministic_distill": float(
                cfg.get("semantic", {}).get("lambda_deterministic_distill", 0.02)
            ),
            "lambda_completion_aux": float(cfg.get("semantic", {}).get("lambda_completion_aux", 1.0)),
            "lambda_deadline_aux": float(cfg.get("semantic", {}).get("lambda_deadline_aux", 0.3)),
            "lambda_delay_aux": float(cfg.get("semantic", {}).get("lambda_delay_aux", 1.0)),
            "prior_decay_steps": int(cfg.get("semantic", {}).get("prior_decay_steps", 40000)),
            "guide_decay_steps": int(cfg.get("semantic", {}).get("guide_decay_steps", 25000)),
            "teacher_temperature": float(cfg.get("semantic", {}).get("teacher_temperature", 0.5)),
        }
    )
    return cfg


def ensure_mappo_defaults(config: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("env", {})["use_semantic_side_channel"] = False
    cfg.setdefault("mappo", {})["report_stochastic_eval"] = True
    return cfg


def ensure_continuous_defaults(config: dict[str, Any]) -> dict[str, Any]:
    cfg = ensure_mappo_defaults(config)
    cfg.setdefault("env", {})["action_mode"] = "simple"
    cfg.setdefault("paper", {})["continuous_control_comparison"] = True
    cfg.setdefault("continuous_baseline", {}).update(
        {
            "batch_size": int(cfg.get("continuous_baseline", {}).get("batch_size", 128)),
            "buffer_size": int(cfg.get("continuous_baseline", {}).get("buffer_size", 100000)),
            "updates_per_collect": int(cfg.get("continuous_baseline", {}).get("updates_per_collect", 1)),
            "target_tau": float(cfg.get("continuous_baseline", {}).get("target_tau", 0.01)),
            "exploration_noise": float(cfg.get("continuous_baseline", {}).get("exploration_noise", 0.12)),
            "policy_noise": float(cfg.get("continuous_baseline", {}).get("policy_noise", 0.08)),
            "policy_noise_clip": float(cfg.get("continuous_baseline", {}).get("policy_noise_clip", 0.20)),
            "policy_delay": int(cfg.get("continuous_baseline", {}).get("policy_delay", 2)),
            "sac_alpha": float(cfg.get("continuous_baseline", {}).get("sac_alpha", 0.05)),
        }
    )
    return cfg


def paper_run_dir(output_root: Path, difficulty: str, scale: str, algo: str, seed: int, run_name: str) -> Path:
    return output_root / difficulty / scale / algo / f"seed_{seed}" / run_name


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_best_stochastic_checkpoint(run_dir: Path) -> None:
    source = run_dir / "checkpoints" / "checkpoint_best_stochastic.pt"
    if source.exists():
        shutil.copy2(source, run_dir / "checkpoints" / "best_stochastic.pt")


def load_config_with_scale(config_path: Path, scale: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    if scale is not None:
        cfg = apply_scale(cfg, scale)
    return cfg


def tail_mean_numeric(csv_path: Path, tail: int = 5) -> dict[str, float]:
    import pandas as pd

    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    values = df.tail(min(tail, len(df))).mean(numeric_only=True)
    return {key: float(value) for key, value in values.items() if np.isfinite(value)}
