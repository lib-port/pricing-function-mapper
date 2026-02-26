from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pricing_mapper.config import MapperConfig


def default_run_id(seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_seed{seed}"


def resolve_run_paths(cfg: MapperConfig) -> MapperConfig:
    run_id = cfg.run_id or default_run_id(cfg.seed)
    run_dir = Path(cfg.output_dir) / run_id

    output_csv = str(run_dir / Path(cfg.output_csv).name)
    output_metadata_json = str(run_dir / Path(cfg.output_metadata_json).name)
    state_path = str(run_dir / Path(cfg.state_path).name)

    return replace(
        cfg,
        run_id=run_id,
        output_csv=output_csv,
        output_metadata_json=output_metadata_json,
        state_path=state_path,
    )


def ensure_parent_dirs(cfg: MapperConfig) -> None:
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_metadata_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.state_path).parent.mkdir(parents=True, exist_ok=True)
