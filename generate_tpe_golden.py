#!/usr/bin/env python3
"""Generate golden-output fixtures for JS TPE parity tests.

Usage:
  python generate_tpe_golden.py
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any, Dict, List, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

ROOT = os.path.abspath(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(ROOT, "fixtures", "golden-tpe")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "tpe_golden.json")


@dataclass
class TrialRecord:
    number: int
    params: Dict[str, Any]
    state: str
    value: Optional[float]
    values: Optional[List[float]]
    intermediate_values: List[List[float]]
    constraint: Optional[List[float]]


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def objective_single(params: Dict[str, Any]) -> float:
    cat_map = {"a": 0.0, "b": 1.0, "c": 2.0}
    x = float(params["x"])
    y = float(params["y"])
    mode = params["mode"]
    log_u = float(params["log_u"])
    log_i = float(params["log_i"])
    cat_score = cat_map.get(mode, 0.0)
    return (
        (x - 0.3) ** 2
        + (y - 5.0) ** 2
        + (math.log(log_u) ** 2)
        + (math.log(log_i) ** 2)
        + cat_score * 0.1
    )


def objective_single_numeric(params: Dict[str, Any]) -> float:
    x = float(params["x"])
    y = float(params["y"])
    log_u = float(params["log_u"])
    return (x - 0.3) ** 2 + (y - 5.0) ** 2 + (math.log(log_u) ** 2)


def objective_multi(params: Dict[str, Any]) -> List[float]:
    group = params.get("group", "a")
    common = float(params.get("common", 0.0))
    if group == "a":
        x = float(params.get("a1", 0.0))
        y = common
    else:
        x = float(params.get("b1", 0.0))
        y = common
    return [x * x + (y - 2.0) ** 2, (x - 1.5) ** 2 + (y + 0.25) ** 2]


def constraint_func(params: Dict[str, Any]) -> List[float]:
    x = float(params["x"])
    y = float(params["y"])
    return [max(0.0, x - 1.5), max(0.0, 0.5 - y)]


def suggest_core(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params["x"] = trial.suggest_float("x", -5.0, 5.0)
    params["y"] = trial.suggest_int("y", 1, 9, step=2)
    params["mode"] = trial.suggest_categorical("mode", ["a", "b", "c"])
    params["log_u"] = trial.suggest_float("log_u", 1e-3, 1e2, log=True)
    params["log_i"] = trial.suggest_int("log_i", 1, 64, log=True)
    return params


def enqueued_core_params() -> List[Dict[str, Any]]:
    return [
        {
            "x": -4.25,
            "y": 1,
            "mode": "c",
            "log_u": 0.0015,
            "log_i": 2,
        },
        {
            "x": 4.9,
            "y": 9,
            "mode": "a",
            "log_u": 12.5,
            "log_i": 16,
        },
        {
            "x": 0.0,
            "y": 5,
            "mode": "b",
            "log_u": 0.1,
            "log_i": 4,
        },
    ]


def suggest_numeric(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params["x"] = trial.suggest_float("x", -5.0, 5.0)
    params["y"] = trial.suggest_float("y", -2.0, 8.0)
    params["log_u"] = trial.suggest_float("log_u", 1e-3, 1e2, log=True)
    return params


def suggest_group(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params["group"] = trial.suggest_categorical("group", ["a", "b"])
    params["common"] = trial.suggest_float("common", -1.0, 1.0)
    if params["group"] == "a":
        params["a1"] = trial.suggest_float("a1", -2.0, 2.0)
    else:
        params["b1"] = trial.suggest_float("b1", 0.0, 4.0)
    return params


def evaluate_with_pruning(trial: optuna.trial.Trial, params: Dict[str, Any]) -> tuple[str, Optional[float], list[list[float]]]:
    base = objective_single(params)
    intermediates: list[list[float]] = []
    for step in [1, 2, 3]:
        intermediate = base + step * 0.1
        trial.report(intermediate, step)
        intermediates.append([float(step), float(intermediate)])
        if step == 2 and intermediate > 1.25:
            return "pruned", None, intermediates
    return "complete", base, intermediates


def gamma_custom(n_trials: int) -> int:
    return min(32, math.ceil(0.2 * n_trials))


def weights_custom(n_observations: int) -> List[float]:
    return [float((i + 1) ** 2) for i in range(n_observations)]


def run_single_objective_numeric_with_sampler(
    seed: int,
    n_trials: int,
    direction: str,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> List[TrialRecord]:
    sampler_args: Dict[str, Any] = {
        "seed": seed,
        "n_startup_trials": 10,
        "n_ei_candidates": 24,
    }
    if sampler_kwargs is not None:
        sampler_args.update(sampler_kwargs)

    sampler = TPESampler(**sampler_args)
    study = optuna.create_study(sampler=sampler, direction=direction)
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_numeric(trial)
        value = objective_single_numeric(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_single_objective(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_core(trial)
        value = objective_single(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_single_objective_enqueued_trials(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for params in enqueued_core_params():
        study.enqueue_trial(params)

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_core(trial)
        value = objective_single(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_single_objective_maximize_numeric(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="maximize",
    )


def run_single_objective_prior_weight(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="minimize",
        sampler_kwargs={"prior_weight": 0.2},
    )


def run_single_objective_magic_clip_endpoints(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="minimize",
        sampler_kwargs={"consider_magic_clip": False, "consider_endpoints": True},
    )


def run_single_objective_gamma_custom(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="minimize",
        sampler_kwargs={"gamma": gamma_custom},
    )


def run_single_objective_weights_custom(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="minimize",
        sampler_kwargs={"weights": weights_custom},
    )


def run_single_objective_n_ei_candidates_custom(seed: int, n_trials: int) -> List[TrialRecord]:
    return run_single_objective_numeric_with_sampler(
        seed=seed,
        n_trials=n_trials,
        direction="minimize",
        sampler_kwargs={"n_ei_candidates": 64},
    )


def run_single_objective_high_startup(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=30,
        n_ei_candidates=24,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_core(trial)
        value = objective_single(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_single_objective_multivariate(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_core(trial)
        value = objective_single(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def objective_single_group(params: Dict[str, Any]) -> float:
    values = objective_multi(params)
    return values[0] + 0.5 * values[1]


def run_single_objective_group(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_group(trial)
        value = objective_single_group(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_single_objective_dynamic_independent(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=30,
        n_ei_candidates=24,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_group(trial)
        value = objective_single_group(params)
        study.tell(trial, value)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_multi_objective(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_group(trial)
        values = objective_multi(params)
        study.tell(trial, values=values)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=None,
                values=[float(values[0]), float(values[1])],
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_multi_objective_dynamic_independent(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=30,
        n_ei_candidates=24,
    )
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_group(trial)
        values = objective_multi(params)
        study.tell(trial, values=values)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=None,
                values=[float(values[0]), float(values[1])],
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_multi_objective_mixed_directions(seed: int, n_trials: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize"])
    records: List[TrialRecord] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_group(trial)
        values = objective_multi(params)
        study.tell(trial, values=values)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=None,
                values=[float(values[0]), float(values[1])],
                intermediate_values=[],
                constraint=None,
            )
        )

    return records


def run_constant_liar_delayed_single(seed: int, n_trials: int, tell_lag: int) -> List[TrialRecord]:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        constant_liar=True,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []
    pending: List[Dict[str, Any]] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_numeric(trial)
        value = objective_single_numeric(params)

        records.append(
            TrialRecord(
                number=trial.number,
                params=sanitize_params(params),
                state="complete",
                value=float(value),
                values=None,
                intermediate_values=[],
                constraint=None,
            )
        )
        pending.append({"trial": trial, "value": value})

        if len(pending) > tell_lag:
            oldest = pending.pop(0)
            study.tell(oldest["trial"], oldest["value"])

    while pending:
        oldest = pending.pop(0)
        study.tell(oldest["trial"], oldest["value"])

    return records


def run_constraints_pruning_constant_liar(seed: int, n_trials: int, tell_lag: int) -> List[TrialRecord]:
    def constraints_func(frozen: optuna.trial.FrozenTrial) -> List[float]:
        return constraint_func(frozen.params)

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        n_ei_candidates=24,
        constant_liar=True,
        constraints_func=constraints_func,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    records: List[TrialRecord] = []
    pending: List[Dict[str, Any]] = []

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_core(trial)
        state, value, intermediate_values = evaluate_with_pruning(trial, params)

        constraint = constraint_func(params) if state == "complete" else None
        record = TrialRecord(
            number=trial.number,
            params=sanitize_params(params),
            state=state,
            value=float(value) if value is not None else None,
            values=None,
            intermediate_values=intermediate_values,
            constraint=constraint,
        )
        records.append(record)
        pending.append({"trial": trial, "state": state, "value": value})

        if len(pending) > tell_lag:
            oldest = pending.pop(0)
            if oldest["state"] == "pruned":
                study.tell(oldest["trial"], state=TrialState.PRUNED)
            else:
                study.tell(oldest["trial"], oldest["value"])

    while pending:
        oldest = pending.pop(0)
        if oldest["state"] == "pruned":
            study.tell(oldest["trial"], state=TrialState.PRUNED)
        else:
            study.tell(oldest["trial"], oldest["value"])

    return records


def main() -> None:
    scenarios = []
    seeds = [0, 1, 2, 3, 4]
    n_trials = 200
    extended_n_trials = 120
    enqueued_n_trials = 8

    scenarios.append(
        {
            "name": "core_single_objective",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [record.__dict__ for record in run_single_objective(seed, n_trials)],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_enqueued_trials",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_enqueued_trials(seed, enqueued_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_maximize_numeric",
            "tellLag": 0,
            "objectiveDirections": ["maximize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_maximize_numeric(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_prior_weight",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_prior_weight(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_magic_clip_endpoints",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_magic_clip_endpoints(
                            seed, extended_n_trials
                        )
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_gamma_custom",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_gamma_custom(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_weights_custom",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_weights_custom(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_n_ei_candidates_custom",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_n_ei_candidates_custom(
                            seed, extended_n_trials
                        )
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_high_startup",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_high_startup(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_multivariate",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_multivariate(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_group",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_group(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "single_objective_dynamic_independent",
            "tellLag": 0,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_single_objective_dynamic_independent(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "multi_objective_group",
            "tellLag": 0,
            "objectiveDirections": ["minimize", "minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [record.__dict__ for record in run_multi_objective(seed, n_trials)],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "multi_objective_dynamic_independent",
            "tellLag": 0,
            "objectiveDirections": ["minimize", "minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_multi_objective_dynamic_independent(seed, n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "multi_objective_mixed_directions",
            "tellLag": 0,
            "objectiveDirections": ["minimize", "maximize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_multi_objective_mixed_directions(seed, extended_n_trials)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "constant_liar_delayed_single",
            "tellLag": 2,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_constant_liar_delayed_single(seed, extended_n_trials, 2)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    scenarios.append(
        {
            "name": "constraints_pruning_constant_liar",
            "tellLag": 1,
            "objectiveDirections": ["minimize"],
            "runs": [
                {
                    "seed": seed,
                    "trials": [
                        record.__dict__
                        for record in run_constraints_pruning_constant_liar(seed, n_trials, 1)
                    ],
                }
                for seed in seeds
            ],
        }
    )

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "optuna_version": getattr(optuna, "__version__", "unknown"),
        },
        "scenarios": scenarios,
    }

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote golden fixture to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
