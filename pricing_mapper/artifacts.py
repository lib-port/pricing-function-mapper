from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pricing_mapper.config import MapperConfig


def default_run_id(seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{stamp}_{uuid4().hex[:8]}_seed{seed}"


def resolve_run_paths(cfg: MapperConfig) -> MapperConfig:
    run_id = cfg.run_id or default_run_id(cfg.seed)
    run_dir = Path(cfg.output_dir) / run_id

    output_csv = str(run_dir / Path(cfg.output_csv).name)
    output_metadata_json = str(run_dir / Path(cfg.output_metadata_json).name)
    state_path = str(run_dir / Path(cfg.state_path).name)
    engine_path = str(run_dir / Path(cfg.engine_path).name)

    return replace(
        cfg,
        run_id=run_id,
        output_csv=output_csv,
        output_metadata_json=output_metadata_json,
        state_path=state_path,
        engine_path=engine_path,
    )


def ensure_parent_dirs(cfg: MapperConfig) -> None:
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_metadata_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.engine_path).parent.mkdir(parents=True, exist_ok=True)


def existing_artifacts(cfg: MapperConfig) -> list[Path]:
    """Return existing paths that a non-resume run would overwrite."""
    paths = (
        Path(cfg.output_csv),
        Path(cfg.output_metadata_json),
        Path(cfg.state_path),
        Path(cfg.engine_path),
    )
    return [path for path in paths if path.exists()]
