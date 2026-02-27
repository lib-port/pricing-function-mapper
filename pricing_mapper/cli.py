from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.api import serve_api
from pricing_mapper.artifacts import ensure_parent_dirs, resolve_run_paths
from pricing_mapper.benchmark import run_benchmark
from pricing_mapper.config import MapperConfig, dump_config, load_config, validate_config
from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.encoding import encode_features
from pricing_mapper.engine import PricingEngine, load_row_json, load_rows_csv, write_rows_csv
from pricing_mapper.quote import load_quote_fn

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active pricing function mapper")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    p.add_argument("--budget", type=int)
    p.add_argument("--init-n", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--pool-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--run-id", type=str)
    p.add_argument("--output-csv", type=str)
    p.add_argument("--output-metadata-json", type=str)
    p.add_argument("--state-path", type=str)
    p.add_argument("--engine-path", type=str)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--distance-backend", choices=["brute", "knn"])
    p.add_argument("--refit-every-batches", type=int)
    p.add_argument("--checkpoint-every-batches", type=int)
    p.add_argument("--cv-subsample-max", type=int)
    p.add_argument("--rf-n-models", type=int)
    p.add_argument("--rf-n-estimators", type=int)
    p.add_argument("--rf-n-jobs", type=int)
    p.add_argument("--early-stop-patience-batches", type=int)
    p.add_argument("--early-stop-min-batches", type=int)
    p.add_argument("--early-stop-min-rel-improvement", type=float)
    p.add_argument("--staged-mapping-enabled", action="store_true")
    p.add_argument("--staged-stage1-fraction", type=float)
    p.add_argument("--staged-focus-jitter-per-anchor", type=int)
    p.add_argument("--segment-focus-enabled", action="store_true")
    p.add_argument("--segment-target-weight", type=float)
    p.add_argument("--segment-sigma-weight", type=float)
    p.add_argument("--segment-min-candidates", type=int)
    p.add_argument("--segment-pool-max-tries", type=int)
    p.add_argument(
        "--segment-constraints",
        type=str,
        help="JSON object for segment constraints",
    )
    p.add_argument("--quote-provider", type=str)
    p.add_argument("--disable-monotone", action="store_true")
    p.add_argument("--price-row", type=str, help="JSON row string for single premium prediction")
    p.add_argument("--price-row-json", type=str, help="Path to JSON row file for prediction")
    p.add_argument("--price-input-csv", type=str, help="CSV path of rows to score")
    p.add_argument("--price-output-csv", type=str, help="Output CSV path for scored batch")
    p.add_argument("--serve-api", action="store_true", help="Serve pricing engine API")
    p.add_argument("--host", type=str, default="127.0.0.1", help="API host for --serve-api")
    p.add_argument("--port", type=int, default=8000, help="API port for --serve-api")

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved settings",
    )
    p.add_argument("--benchmark", action="store_true", help="Run benchmark presets")
    p.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark_results.json",
        help="Benchmark JSON output path",
    )

    return p.parse_args()


def apply_cli_overrides(cfg: MapperConfig, args: argparse.Namespace) -> MapperConfig:
    updates: dict[str, Any] = {}
    mapping = {
        "budget": "budget",
        "init_n": "init_n",
        "batch_size": "batch_size",
        "pool_size": "pool_size",
        "seed": "seed",
        "output_dir": "output_dir",
        "run_id": "run_id",
        "output_csv": "output_csv",
        "output_metadata_json": "output_metadata_json",
        "state_path": "state_path",
        "engine_path": "engine_path",
        "distance_backend": "distance_backend",
        "refit_every_batches": "refit_every_batches",
        "checkpoint_every_batches": "checkpoint_every_batches",
        "cv_subsample_max": "cv_subsample_max",
        "rf_n_models": "rf_n_models",
        "rf_n_estimators": "rf_n_estimators",
        "rf_n_jobs": "rf_n_jobs",
        "early_stop_patience_batches": "early_stop_patience_batches",
        "early_stop_min_batches": "early_stop_min_batches",
        "early_stop_min_rel_improvement": "early_stop_min_rel_improvement",
        "staged_stage1_fraction": "staged_stage1_fraction",
        "staged_focus_jitter_per_anchor": "staged_focus_jitter_per_anchor",
        "segment_target_weight": "segment_target_weight",
        "segment_sigma_weight": "segment_sigma_weight",
        "segment_min_candidates": "segment_min_candidates",
        "segment_pool_max_tries": "segment_pool_max_tries",
        "quote_provider": "quote_provider",
    }

    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[cfg_name] = value

    if args.resume:
        updates["resume"] = True

    if args.disable_monotone:
        updates["use_monotone_if_available"] = False
    if args.staged_mapping_enabled:
        updates["staged_mapping_enabled"] = True
    if args.segment_focus_enabled:
        updates["segment_focus_enabled"] = True
    if args.segment_constraints is not None:
        parsed = json.loads(args.segment_constraints)
        if not isinstance(parsed, dict):
            raise ValueError("--segment-constraints must be a JSON object")
        updates["segment_constraints"] = parsed

    data = dump_config(cfg)
    data.update(updates)
    out = MapperConfig(**data)
    validate_config(out)
    return out


def _run_single(cfg: MapperConfig, logger: logging.Logger) -> int:
    cfg = resolve_run_paths(cfg)
    ensure_parent_dirs(cfg)

    quote_fn = load_quote_fn(cfg.quote_provider)
    domain = build_comp_car_domain(cfg.domain_overrides)
    mapper = ActiveQuoteMapper(domain=domain, quote_fn=quote_fn, cfg=cfg, logger=logger)
    df, stats = mapper.run()

    x_train, cols = encode_features(domain, df.drop(columns=["premium"]).to_dict(orient="records"))
    mu_rf, _ = mapper.rf.predict_mean_std(x_train)
    mae_rf = float(np.mean(np.abs(mu_rf - df["premium"].to_numpy(dtype=float))))

    output_csv = Path(cfg.output_csv)
    df.to_csv(output_csv, index=False)

    metadata: dict[str, Any] = {
        "stats": stats.__dict__,
        "mae_rf": mae_rf,
        "monotone": mapper.use_monotone,
        "config": dump_config(cfg),
        "features": cols,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "artifacts": {
            "run_id": cfg.run_id,
            "output_csv": str(output_csv),
            "output_metadata_json": cfg.output_metadata_json,
            "state_path": cfg.state_path,
            "engine_path": cfg.engine_path,
        },
    }

    if cfg.segment_constraints:
        feature_rows = df.drop(columns=["premium"]).to_dict(orient="records")
        seg_mask = np.asarray(
            [mapper.row_in_segment(row) for row in feature_rows],
            dtype=bool,
        )
        seg_count = int(seg_mask.sum())
        metadata["segment"] = {
            "enabled": bool(cfg.segment_focus_enabled),
            "constraints": cfg.segment_constraints,
            "count": seg_count,
            "fraction": float(seg_count / max(1, len(df))),
        }
        if seg_count > 0:
            y_true = df["premium"].to_numpy(dtype=float)
            metadata["segment"]["mae_rf"] = float(
                np.mean(np.abs(mu_rf[seg_mask] - y_true[seg_mask]))
            )

    if mapper.use_monotone and mapper.hgb is not None:
        mu_m = mapper.hgb.predict(x_train)
        metadata["mae_monotone"] = float(
            np.mean(np.abs(mu_m - df["premium"].to_numpy(dtype=float)))
        )
        if cfg.segment_constraints and "segment" in metadata:
            seg_mask = np.asarray(
                [mapper.row_in_segment(row) for row in feature_rows],
                dtype=bool,
            )
            if int(seg_mask.sum()) > 0:
                y_true = df["premium"].to_numpy(dtype=float)
                metadata["segment"]["mae_monotone"] = float(
                    np.mean(np.abs(mu_m[seg_mask] - y_true[seg_mask]))
                )

    output_meta = Path(cfg.output_metadata_json)
    output_meta.write_text(json.dumps(metadata, indent=2))

    engine = PricingEngine.from_mapper(
        domain=domain,
        rf=mapper.rf,
        hgb=mapper.hgb,
        use_monotone=mapper.use_monotone,
        cfg=cfg,
    )
    engine.save(cfg.engine_path)

    logger.info("Saved quotes to %s", output_csv)
    logger.info("Saved metadata to %s", output_meta)
    logger.info("Saved pricing engine to %s", cfg.engine_path)
    return 0


def _run_pricing_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    engine_path = args.engine_path
    if not engine_path:
        raise ValueError("engine_path is required for pricing mode")
    engine = PricingEngine.load(engine_path)

    if args.serve_api:
        serve_api(engine_path=engine_path, host=args.host, port=args.port)
        return 0

    has_single = bool(args.price_row or args.price_row_json)
    has_batch = bool(args.price_input_csv)
    if has_single and has_batch:
        raise ValueError("Use either single-row or batch pricing mode, not both")
    if not has_single and not has_batch:
        raise ValueError(
            "Pricing mode requires --price-row, --price-row-json, or --price-input-csv"
        )

    if args.price_row_json:
        row = load_row_json(args.price_row_json)
        premium = engine.predict_row(row)
        payload = {"premium": round(float(premium), 2)}
        print(json.dumps(payload, indent=2))
        return 0

    if args.price_row:
        row = json.loads(args.price_row)
        if not isinstance(row, dict):
            raise ValueError("--price-row must be a JSON object")
        premium = engine.predict_row(row)
        payload = {"premium": round(float(premium), 2)}
        print(json.dumps(payload, indent=2))
        return 0

    rows = load_rows_csv(args.price_input_csv)
    scored = engine.predict_rows_with_inputs(rows)
    out_csv = args.price_output_csv or str(
        Path(args.price_input_csv).with_name("priced_output.csv")
    )
    write_rows_csv(out_csv, scored)
    logger.info("Scored %d rows", len(scored))
    logger.info("Saved scored output to %s", out_csv)
    return 0


def run_cli() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("pricing_mapper")

    pricing_mode = bool(
        args.price_row
        or args.price_row_json
        or args.price_input_csv
        or args.serve_api
    )
    if pricing_mode:
        return _run_pricing_mode(args, logger)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.dry_run:
        resolved = resolve_run_paths(cfg)
        payload = dump_config(resolved)
        logger.info("Dry-run successful. Resolved config follows.")
        print(json.dumps(payload, indent=2))
        return 0

    if args.benchmark:
        resolved = resolve_run_paths(cfg)
        payload = run_benchmark(resolved, args.benchmark_output)
        results = payload.get("results")
        n_results = len(results) if isinstance(results, list) else 0
        logger.info("Benchmark completed with %d presets", n_results)
        logger.info("Saved benchmark output to %s", args.benchmark_output)
        return 0

    return _run_single(cfg, logger)


if __name__ == "__main__":
    raise SystemExit(run_cli())
