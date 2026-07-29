from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from pricing_mapper.domain import (
    CATEGORICAL_DEFAULT_LEVELS,
    DOMAIN_CATEGORICAL_NAMES,
    DOMAIN_CONTINUOUS_NAMES,
    DOMAIN_INTEGER_NAMES,
    DomainSpec,
    build_comp_car_domain,
    validate_domain_overrides,
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
    distance_backend: str = "knn"
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

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must contain an object at the top level.")

    known_fields = {item.name for item in fields(MapperConfig)}
    unknown_fields = sorted(set(raw) - known_fields)
    if unknown_fields:
        raise ValueError(f"Unknown config fields: {unknown_fields}")

    cfg_data = asdict(DEFAULT_CONFIG)
    cfg_data.update(raw)

    if isinstance(cfg_data.get("acquisition_mix"), (list, tuple)):
        cfg_data["acquisition_mix"] = tuple(cfg_data["acquisition_mix"])

    cfg = MapperConfig(**cfg_data)
    validate_config(cfg)
    return cfg


def dump_config(cfg: MapperConfig) -> dict[str, Any]:
    return asdict(cfg)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _positive_integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer.")
    integer = int(value)
    if integer < minimum:
        comparator = "> 0" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{label} must be {comparator}")
    return integer


def _validate_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain null bytes")
    return value


def validate_segment_constraints(
    constraints: dict[str, Any],
    domain: DomainSpec | None = None,
) -> None:
    if not isinstance(constraints, dict):
        raise ValueError("segment_constraints must be a dictionary.")

    domain = domain or build_comp_car_domain()
    known_vars = (
        set(DOMAIN_CONTINUOUS_NAMES) | set(DOMAIN_INTEGER_NAMES) | set(DOMAIN_CATEGORICAL_NAMES)
    )
    numeric_vars = set(DOMAIN_CONTINUOUS_NAMES) | set(DOMAIN_INTEGER_NAMES)
    categorical_vars = set(DOMAIN_CATEGORICAL_NAMES)
    numeric_bounds = {var.name: (float(var.low), float(var.high)) for var in domain.continuous}
    numeric_bounds.update({var.name: (float(var.low), float(var.high)) for var in domain.integers})
    integer_vars = {var.name for var in domain.integers}
    categorical_levels = {var.name: list(var.levels) for var in domain.categorical}

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
            rule = raw_rule
        else:
            rule = {"eq": raw_rule}

        if name in numeric_vars:
            if "in" in rule:
                raise ValueError(
                    f"segment_constraints['{name}'] cannot use 'in' for numeric variables."
                )
            numeric_rule: dict[str, float] = {}
            for operator in ("min", "max", "eq"):
                if operator in rule:
                    numeric_rule[operator] = _finite_number(
                        rule[operator],
                        f"segment_constraints['{name}'].{operator}",
                    )
            if "min" in rule and "max" in rule and numeric_rule["min"] > numeric_rule["max"]:
                raise ValueError(f"segment_constraints['{name}'] has min greater than max.")
            low, high = numeric_bounds[name]
            effective_low = max(low, numeric_rule.get("min", low))
            effective_high = min(high, numeric_rule.get("max", high))
            if effective_low > effective_high:
                raise ValueError(f"segment_constraints['{name}'] does not intersect the domain.")
            if "eq" in numeric_rule:
                eq = numeric_rule["eq"]
                if not effective_low <= eq <= effective_high:
                    raise ValueError(
                        f"segment_constraints['{name}'].eq conflicts with its range "
                        "or the domain."
                    )
                if name in integer_vars and not eq.is_integer():
                    raise ValueError(f"segment_constraints['{name}'].eq must be an integer.")
            elif name in integer_vars and math.ceil(effective_low) > math.floor(effective_high):
                raise ValueError(f"segment_constraints['{name}'] contains no valid integer values.")

        if name in categorical_vars:
            if "min" in rule or "max" in rule:
                raise ValueError(
                    f"segment_constraints['{name}'] cannot use min/max for categorical variables."
                )
            if "in" in rule:
                if not isinstance(rule["in"], list) or len(rule["in"]) == 0:
                    raise ValueError(f"segment_constraints['{name}'].in must be a non-empty list.")
                scalar_types = (str, int, float, bool, type(None))
                if any(not isinstance(level, scalar_types) for level in rule["in"]):
                    raise ValueError(
                        f"segment_constraints['{name}'].in values must be JSON scalars."
                    )
                if len(set(rule["in"])) != len(rule["in"]):
                    raise ValueError(f"segment_constraints['{name}'].in contains duplicate values.")
                unknown_levels = [
                    level for level in rule["in"] if level not in categorical_levels[name]
                ]
                if unknown_levels:
                    raise ValueError(
                        f"segment_constraints['{name}'].in contains values outside "
                        f"the domain: {unknown_levels!r}"
                    )
            if "eq" in rule and rule["eq"] not in categorical_levels[name]:
                raise ValueError(f"segment_constraints['{name}'].eq is outside the domain.")
            if "eq" in rule and "in" in rule and rule["eq"] not in rule["in"]:
                raise ValueError(
                    f"segment_constraints['{name}'].eq conflicts with its 'in' values."
                )

    def constrained_numeric_range(name: str) -> tuple[float, float]:
        low, high = numeric_bounds[name]
        raw_rule = constraints.get(name)
        if raw_rule is None:
            return low, high
        rule = raw_rule if isinstance(raw_rule, dict) else {"eq": raw_rule}
        if "eq" in rule:
            value = float(rule["eq"])
            return value, value
        return (
            max(low, float(rule.get("min", low))),
            min(high, float(rule.get("max", high))),
        )

    _, maximum_driver_age = constrained_numeric_range("driver_age")
    minimum_years_licensed, _ = constrained_numeric_range("years_licensed")
    if math.ceil(minimum_years_licensed) > max(0, int(maximum_driver_age) - 16):
        raise ValueError(
            "segment_constraints are infeasible: years_licensed cannot be "
            "satisfied within the constrained driver_age range."
        )


def validate_config(cfg: MapperConfig) -> None:
    _positive_integer(cfg.seed, "seed", minimum=0)
    for name in (
        "budget",
        "init_n",
        "batch_size",
        "pool_size",
        "refit_every_batches",
        "cv_subsample_max",
        "rf_n_models",
        "rf_n_estimators",
        "early_stop_min_batches",
        "staged_focus_jitter_per_anchor",
        "segment_min_candidates",
        "segment_pool_max_tries",
    ):
        _positive_integer(getattr(cfg, name), name)

    for name in ("checkpoint_every_batches", "early_stop_patience_batches"):
        _positive_integer(getattr(cfg, name), name, minimum=0)

    if (
        isinstance(cfg.rf_n_jobs, bool)
        or not isinstance(cfg.rf_n_jobs, Integral)
        or (cfg.rf_n_jobs != -1 and cfg.rf_n_jobs < 1)
    ):
        raise ValueError("rf_n_jobs must be -1 or a positive integer")

    early_stop_min_rel_improvement = _finite_number(
        cfg.early_stop_min_rel_improvement,
        "early_stop_min_rel_improvement",
    )
    if early_stop_min_rel_improvement < 0:
        raise ValueError("early_stop_min_rel_improvement must be >= 0")

    staged_stage1_fraction = _finite_number(
        cfg.staged_stage1_fraction,
        "staged_stage1_fraction",
    )
    if not 0 < staged_stage1_fraction < 1:
        raise ValueError("staged_stage1_fraction must be between 0 and 1")

    for name in ("segment_target_weight", "segment_sigma_weight"):
        if _finite_number(getattr(cfg, name), name) < 0:
            raise ValueError(f"{name} must be >= 0")

    for name in (
        "resume",
        "use_monotone_if_available",
        "staged_mapping_enabled",
        "segment_focus_enabled",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be a boolean")

    if not isinstance(cfg.distance_backend, str) or cfg.distance_backend not in {
        "brute",
        "knn",
    }:
        raise ValueError("distance_backend must be one of: brute, knn")

    if not isinstance(cfg.acquisition_mix, (list, tuple)) or len(cfg.acquisition_mix) != 4:
        raise ValueError("acquisition_mix must contain exactly 4 values")
    acquisition_mix = [
        _finite_number(value, f"acquisition_mix[{index}]")
        for index, value in enumerate(cfg.acquisition_mix)
    ]
    if any(value < 0 for value in acquisition_mix):
        raise ValueError("acquisition_mix values must be non-negative")

    mix_sum = float(sum(acquisition_mix))
    if abs(mix_sum - 1.0) > 1e-6:
        raise ValueError(f"acquisition_mix must sum to 1.0, got {mix_sum:.6f}")

    numeric_vars = set(DOMAIN_CONTINUOUS_NAMES) | set(DOMAIN_INTEGER_NAMES)
    if not isinstance(cfg.breakpoint_vars, list) or any(
        not isinstance(name, str) for name in cfg.breakpoint_vars
    ):
        raise ValueError("breakpoint_vars must be a list of variable names")
    unknown_breakpoint_vars = sorted(set(cfg.breakpoint_vars) - numeric_vars)
    if unknown_breakpoint_vars:
        raise ValueError(f"Unknown breakpoint_vars: {unknown_breakpoint_vars}")
    if len(set(cfg.breakpoint_vars)) != len(cfg.breakpoint_vars):
        raise ValueError("breakpoint_vars cannot contain duplicates")
    if acquisition_mix[3] > 0 and not cfg.breakpoint_vars:
        raise ValueError("breakpoint_vars cannot be empty when breakpoint acquisition is enabled")

    _validate_nonempty_string(cfg.output_dir, "output_dir")
    for name in (
        "output_csv",
        "output_metadata_json",
        "state_path",
        "engine_path",
    ):
        value = _validate_nonempty_string(getattr(cfg, name), name)
        if Path(value).name in {"", ".", ".."}:
            raise ValueError(f"{name} must identify a file")

    artifact_names = [
        Path(getattr(cfg, name)).name
        for name in (
            "output_csv",
            "output_metadata_json",
            "state_path",
            "engine_path",
        )
    ]
    if len(set(artifact_names)) != len(artifact_names):
        raise ValueError("Artifact filenames must be distinct")

    if cfg.run_id is not None:
        run_id = _validate_nonempty_string(cfg.run_id, "run_id")
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single safe path component")

    if cfg.quote_provider is not None:
        provider = _validate_nonempty_string(cfg.quote_provider, "quote_provider")
        module_name, separator, function_name = provider.partition(":")
        if (
            not separator
            or provider.count(":") != 1
            or not module_name.strip()
            or not function_name.strip()
        ):
            raise ValueError("quote_provider must use 'module:function' format")

    validate_domain_overrides(cfg.domain_overrides)
    normalized_provider = (
        cfg.quote_provider.strip() if isinstance(cfg.quote_provider, str) else None
    )
    if normalized_provider in {None, "pricing_mapper.quote:mock_comp_car_quote"}:
        for name, levels in cfg.domain_overrides.get("categorical", {}).items():
            unsupported = [
                level for level in levels if level not in CATEGORICAL_DEFAULT_LEVELS[name]
            ]
            if unsupported:
                raise ValueError(
                    f"categorical override '{name}' contains values unsupported by "
                    f"the default quote provider: {unsupported!r}"
                )
    domain = build_comp_car_domain(cfg.domain_overrides)
    validate_segment_constraints(cfg.segment_constraints, domain)
    if cfg.segment_focus_enabled and not cfg.segment_constraints:
        raise ValueError("segment_constraints cannot be empty when segment_focus_enabled is true")
