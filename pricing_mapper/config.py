from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pricing_mapper.domain import (
    DOMAIN_CATEGORICAL_NAMES,
    DOMAIN_CONTINUOUS_NAMES,
    DOMAIN_INTEGER_NAMES,
)


@dataclass
class MapperConfig:
    seed: int = 42
    budget: int = 260
    init_n: int = 95
    batch_size: int = 20
    pool_size: int = 14000

    output_dir: str = "outputs"
    run_id: str | None = None
    output_csv: str = "comp_car_quotes_advanced.csv"
    output_metadata_json: str = "run_metadata.json"
    state_path: str = "run_state.json"
    engine_path: str = "pricing_engine.pkl"

    resume: bool = False
    checkpoint_every_batches: int = 1
    refit_every_batches: int = 1
    cv_subsample_max: int = 1200

    use_monotone_if_available: bool = True
    distance_backend: str = "brute"
    acquisition_mix: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10)
    breakpoint_vars: list[str] = field(
        default_factory=lambda: [
            "driver_age",
            "postcode_risk",
            "vehicle_value",
            "excess",
            "vehicle_year",
            "annual_km",
        ]
    )

    rf_n_models: int = 20
    rf_n_estimators: int = 600
    rf_n_jobs: int = -1

    early_stop_patience_batches: int = 0
    early_stop_min_batches: int = 4
    early_stop_min_rel_improvement: float = 0.005

    staged_mapping_enabled: bool = False
    staged_stage1_fraction: float = 0.40
    staged_focus_jitter_per_anchor: int = 12

    segment_focus_enabled: bool = False
    segment_constraints: dict[str, Any] = field(default_factory=dict)
    segment_target_weight: float = 0.35
    segment_sigma_weight: float = 0.20
    segment_min_candidates: int = 400
    segment_pool_max_tries: int = 4

    quote_provider: str | None = None
    domain_overrides: dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG = MapperConfig()


def load_config(path: str | None) -> MapperConfig:
    if path is None:
        cfg = MapperConfig()
        validate_config(cfg)
        return cfg

    raw = json.loads(Path(path).read_text())
    cfg_data = asdict(DEFAULT_CONFIG)
    cfg_data.update(raw)

    if "acquisition_mix" in cfg_data:
        cfg_data["acquisition_mix"] = tuple(cfg_data["acquisition_mix"])

    cfg = MapperConfig(**cfg_data)
    validate_config(cfg)
    return cfg


def dump_config(cfg: MapperConfig) -> dict[str, Any]:
    return asdict(cfg)


def validate_domain_overrides(overrides: dict[str, Any]) -> None:
    if not isinstance(overrides, dict):
        raise ValueError("domain_overrides must be a dictionary.")

    for bucket in ("continuous", "integers", "categorical"):
        if bucket in overrides and not isinstance(overrides[bucket], dict):
            raise ValueError(f"domain_overrides.{bucket} must be a dictionary.")

    for name, cfg in overrides.get("continuous", {}).items():
        if name not in DOMAIN_CONTINUOUS_NAMES:
            raise ValueError(f"Unknown continuous override '{name}'.")
        if not isinstance(cfg, dict) or "low" not in cfg or "high" not in cfg:
            raise ValueError(f"continuous override '{name}' must provide low/high.")
        if float(cfg["low"]) >= float(cfg["high"]):
            raise ValueError(f"continuous override '{name}' has low >= high.")

    for name, cfg in overrides.get("integers", {}).items():
        if name not in DOMAIN_INTEGER_NAMES:
            raise ValueError(f"Unknown integers override '{name}'.")
        if not isinstance(cfg, dict) or "low" not in cfg or "high" not in cfg:
            raise ValueError(f"integers override '{name}' must provide low/high.")
        if int(cfg["low"]) >= int(cfg["high"]):
            raise ValueError(f"integers override '{name}' has low >= high.")

    for name, levels in overrides.get("categorical", {}).items():
        if name not in DOMAIN_CATEGORICAL_NAMES:
            raise ValueError(f"Unknown categorical override '{name}'.")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(
                f"categorical override '{name}' must be a list with at least 2 levels."
            )
        if len(set(levels)) != len(levels):
            raise ValueError(f"categorical override '{name}' contains duplicate levels.")


def validate_segment_constraints(constraints: dict[str, Any]) -> None:
    if not isinstance(constraints, dict):
        raise ValueError("segment_constraints must be a dictionary.")

    known_vars = set(DOMAIN_CONTINUOUS_NAMES) | set(DOMAIN_INTEGER_NAMES) | set(
        DOMAIN_CATEGORICAL_NAMES
    )
    numeric_vars = set(DOMAIN_CONTINUOUS_NAMES) | set(DOMAIN_INTEGER_NAMES)
    categorical_vars = set(DOMAIN_CATEGORICAL_NAMES)

    for name, raw_rule in constraints.items():
        if name not in known_vars:
            raise ValueError(f"Unknown segment constraint variable '{name}'.")

        if isinstance(raw_rule, dict):
            allowed = {"min", "max", "eq", "in"}
            unknown = set(raw_rule) - allowed
            if unknown:
                raise ValueError(
                    f"segment_constraints['{name}'] has unknown keys: {sorted(unknown)}"
                )
            if not raw_rule:
                raise ValueError(f"segment_constraints['{name}'] cannot be empty.")
        else:
            raw_rule = {"eq": raw_rule}

        if name in numeric_vars:
            if "in" in raw_rule:
                raise ValueError(
                    f"segment_constraints['{name}'] cannot use 'in' for numeric variables."
                )
            if "min" in raw_rule and "max" in raw_rule:
                if float(raw_rule["min"]) > float(raw_rule["max"]):
                    raise ValueError(
                        f"segment_constraints['{name}'] has min greater than max."
                    )

        if name in categorical_vars:
            if "min" in raw_rule or "max" in raw_rule:
                raise ValueError(
                    f"segment_constraints['{name}'] cannot use min/max for categorical variables."
                )
            if "in" in raw_rule:
                if not isinstance(raw_rule["in"], list) or len(raw_rule["in"]) == 0:
                    raise ValueError(
                        f"segment_constraints['{name}'].in must be a non-empty list."
                    )


def validate_config(cfg: MapperConfig) -> None:
    if cfg.budget <= 0:
        raise ValueError("budget must be > 0")
    if cfg.init_n <= 0:
        raise ValueError("init_n must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.pool_size <= 0:
        raise ValueError("pool_size must be > 0")
    if cfg.refit_every_batches <= 0:
        raise ValueError("refit_every_batches must be > 0")
    if cfg.cv_subsample_max <= 0:
        raise ValueError("cv_subsample_max must be > 0")
    if cfg.checkpoint_every_batches < 0:
        raise ValueError("checkpoint_every_batches must be >= 0")
    if cfg.rf_n_models <= 0:
        raise ValueError("rf_n_models must be > 0")
    if cfg.rf_n_estimators <= 0:
        raise ValueError("rf_n_estimators must be > 0")
    if cfg.rf_n_jobs == 0:
        raise ValueError("rf_n_jobs must be -1 or a positive integer")
    if cfg.early_stop_patience_batches < 0:
        raise ValueError("early_stop_patience_batches must be >= 0")
    if cfg.early_stop_min_batches < 1:
        raise ValueError("early_stop_min_batches must be >= 1")
    if cfg.early_stop_min_rel_improvement < 0:
        raise ValueError("early_stop_min_rel_improvement must be >= 0")
    if not (0 < cfg.staged_stage1_fraction < 1):
        raise ValueError("staged_stage1_fraction must be between 0 and 1")
    if cfg.staged_focus_jitter_per_anchor <= 0:
        raise ValueError("staged_focus_jitter_per_anchor must be > 0")
    if cfg.segment_target_weight < 0:
        raise ValueError("segment_target_weight must be >= 0")
    if cfg.segment_sigma_weight < 0:
        raise ValueError("segment_sigma_weight must be >= 0")
    if cfg.segment_min_candidates < 1:
        raise ValueError("segment_min_candidates must be >= 1")
    if cfg.segment_pool_max_tries < 1:
        raise ValueError("segment_pool_max_tries must be >= 1")

    if cfg.distance_backend not in {"brute", "knn"}:
        raise ValueError("distance_backend must be one of: brute, knn")

    if len(cfg.acquisition_mix) != 4:
        raise ValueError("acquisition_mix must contain exactly 4 values")
    if any(v < 0 for v in cfg.acquisition_mix):
        raise ValueError("acquisition_mix values must be non-negative")

    mix_sum = float(sum(cfg.acquisition_mix))
    if abs(mix_sum - 1.0) > 1e-6:
        raise ValueError(f"acquisition_mix must sum to 1.0, got {mix_sum:.6f}")

    if cfg.run_id is not None and not cfg.run_id.strip():
        raise ValueError("run_id cannot be empty if provided")
    if not cfg.engine_path:
        raise ValueError("engine_path cannot be empty")

    validate_domain_overrides(cfg.domain_overrides)
    validate_segment_constraints(cfg.segment_constraints)
