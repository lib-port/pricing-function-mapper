from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.artifacts import ensure_parent_dirs, resolve_run_paths
from pricing_mapper.config import MapperConfig, dump_config
from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.encoding import encode_features
from pricing_mapper.quote import load_quote_fn

BENCHMARK_PRESETS: list[dict[str, str | int]] = [
    {
        "name": "baseline_brute",
        "distance_backend": "brute",
        "rf_n_models": 6,
        "rf_n_estimators": 120,
        "refit_every_batches": 1,
    },
    {
        "name": "knn_distance",
        "distance_backend": "knn",
        "rf_n_models": 6,
        "rf_n_estimators": 120,
        "refit_every_batches": 1,
    },
    {
        "name": "lean_rf",
        "distance_backend": "knn",
        "rf_n_models": 4,
        "rf_n_estimators": 80,
        "refit_every_batches": 2,
    },
]


def run_benchmark(cfg: MapperConfig, output_json: str) -> dict[str, object]:
    quote_fn = load_quote_fn(cfg.quote_provider)
    rows: list[dict[str, object]] = []

    for preset in BENCHMARK_PRESETS:
        run_cfg = replace(
            cfg,
            distance_backend=str(preset["distance_backend"]),
            rf_n_models=int(preset["rf_n_models"]),
            rf_n_estimators=int(preset["rf_n_estimators"]),
            refit_every_batches=int(preset["refit_every_batches"]),
        )
        run_cfg = resolve_run_paths(run_cfg)
        ensure_parent_dirs(run_cfg)

        t0 = perf_counter()
        domain = build_comp_car_domain(run_cfg.domain_overrides)
        mapper = ActiveQuoteMapper(domain=domain, quote_fn=quote_fn, cfg=run_cfg)
        df, _ = mapper.run()
        elapsed = perf_counter() - t0

        x_train, _ = encode_features(domain, df.drop(columns=["premium"]).to_dict(orient="records"))
        mu_rf, _ = mapper.rf.predict_mean_std(x_train)
        mae_rf = float(np.mean(np.abs(mu_rf - df["premium"].to_numpy(dtype=float))))

        row = {
            "name": str(preset["name"]),
            "elapsed_seconds": elapsed,
            "samples": len(df),
            "mae_rf": mae_rf,
            "distance_backend": run_cfg.distance_backend,
            "rf_n_models": run_cfg.rf_n_models,
            "rf_n_estimators": run_cfg.rf_n_estimators,
            "refit_every_batches": run_cfg.refit_every_batches,
        }
        rows.append(row)

    payload: dict[str, object] = {
        "base_config": dump_config(cfg),
        "presets": BENCHMARK_PRESETS,
        "results": rows,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    csv_path = out.with_suffix(".csv")
    if rows:
        headers = list(rows[0].keys())
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(row[h]) for h in headers))
        csv_path.write_text("\n".join(lines) + "\n")

    return payload
