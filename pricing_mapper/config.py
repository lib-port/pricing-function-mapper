from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
        if not isinstance(cfg, dict) or "low" not in cfg or "high" not in cfg:
            raise ValueError(f"continuous override '{name}' must provide low/high.")
        if float(cfg["low"]) >= float(cfg["high"]):
            raise ValueError(f"continuous override '{name}' has low >= high.")

    for name, cfg in overrides.get("integers", {}).items():
        if not isinstance(cfg, dict) or "low" not in cfg or "high" not in cfg:
            raise ValueError(f"integers override '{name}' must provide low/high.")
        if int(cfg["low"]) >= int(cfg["high"]):
            raise ValueError(f"integers override '{name}' has low >= high.")

    for name, levels in overrides.get("categorical", {}).items():
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(
                f"categorical override '{name}' must be a list with at least 2 levels."
            )
        if len(set(levels)) != len(levels):
            raise ValueError(f"categorical override '{name}' contains duplicate levels.")


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

    validate_domain_overrides(cfg.domain_overrides)
