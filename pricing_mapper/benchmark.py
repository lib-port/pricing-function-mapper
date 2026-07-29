from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.config import MapperConfig, dump_config, validate_config
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
    """Run isolated benchmark presets and write JSON and CSV summaries."""
    validate_config(cfg)
    if not output_json or not output_json.strip():
        raise ValueError("benchmark output path cannot be empty")

    quote_fn = load_quote_fn(cfg.quote_provider)
    rows: list[dict[str, object]] = []
    out = Path(output_json)
    if out.suffix.lower() != ".json":
        raise ValueError("benchmark output path must end in .json")

    for preset in BENCHMARK_PRESETS:
        run_cfg = replace(
            cfg,
            distance_backend=str(preset["distance_backend"]),
            rf_n_models=int(preset["rf_n_models"]),
            rf_n_estimators=int(preset["rf_n_estimators"]),
            refit_every_batches=int(preset["refit_every_batches"]),
            resume=False,
            checkpoint_every_batches=0,
        )

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

    out.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = out.with_suffix(out.suffix + ".tmp")
    json_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_tmp.replace(out)

    csv_path = out.with_suffix(".csv")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as csv_file:
        headers = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    csv_tmp.replace(csv_path)

    return payload
