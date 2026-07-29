from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pricing_mapper.config import MapperConfig, dump_config, validate_config
from pricing_mapper.domain import DomainSpec, canonicalize_comp_car_input
from pricing_mapper.encoding import encode_features
from pricing_mapper.models import BootstrappedRF, MonotoneHGBWrapper

ENGINE_SCHEMA_VERSION = 1


class PricingEngine:
    def __init__(
        self,
        domain: DomainSpec,
        rf: BootstrappedRF,
        hgb: MonotoneHGBWrapper | None,
        use_monotone: bool,
        cfg: MapperConfig,
    ) -> None:
        self.domain = domain
        self.rf = rf
        self.hgb = hgb
        self.use_monotone = bool(use_monotone and hgb is not None)
        self.cfg = cfg
        _, cols = encode_features(domain, [])
        self.feature_columns = cols

    @classmethod
    def from_mapper(
        cls,
        domain: DomainSpec,
        rf: BootstrappedRF,
        hgb: MonotoneHGBWrapper | None,
        use_monotone: bool,
        cfg: MapperConfig,
    ) -> PricingEngine:
        return cls(
            domain=domain,
            rf=rf,
            hgb=hgb,
            use_monotone=use_monotone,
            cfg=cfg,
        )

    def predict_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.empty((0,), dtype=float)
        canon = [canonicalize_comp_car_input(row, self.domain) for row in rows]
        x_eval, _ = encode_features(self.domain, canon)
        if self.use_monotone and self.hgb is not None:
            preds = self.hgb.predict(x_eval)
        else:
            preds, _ = self.rf.predict_mean_std(x_eval)
        predictions = np.asarray(preds, dtype=float)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Pricing engine produced a non-finite prediction.")
        return np.maximum(predictions, 0.0)

    def predict_row(self, row: dict[str, Any]) -> float:
        preds = self.predict_rows([row])
        return float(preds[0])

    def save(self, path: str | Path) -> None:
        if not self.rf.fitted or not self.rf.models:
            raise ValueError("Cannot save an engine with an unfitted RF model.")
        if self.use_monotone and (self.hgb is None or not self.hgb.fitted):
            raise ValueError("Cannot save an engine with an unfitted monotone model.")

        payload = {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "domain": self.domain,
            "rf": self.rf,
            "hgb": self.hgb,
            "use_monotone": self.use_monotone,
            "cfg": asdict(self.cfg),
            "feature_columns": list(self.feature_columns),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> PricingEngine:
        target = Path(path)
        try:
            payload = pickle.loads(target.read_bytes())
        except Exception as exc:
            raise ValueError(f"Failed to load engine from {target}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Invalid engine payload format.")
        version = payload.get("schema_version", -1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("Engine schema_version must be an integer.")
        if version != ENGINE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported engine schema version: {version}")

        cfg_raw = payload.get("cfg")
        if not isinstance(cfg_raw, dict):
            raise ValueError("Engine payload missing config.")
        cfg_raw = dict(cfg_raw)

        try:
            if "acquisition_mix" in cfg_raw:
                cfg_raw["acquisition_mix"] = tuple(cfg_raw["acquisition_mix"])
            cfg = MapperConfig(**cfg_raw)
            validate_config(cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Engine payload contains an invalid config: {exc}") from exc

        domain = payload.get("domain")
        rf = payload.get("rf")
        hgb = payload.get("hgb")
        if not isinstance(domain, DomainSpec):
            raise ValueError("Engine payload contains an invalid domain.")
        if not isinstance(rf, BootstrappedRF) or not rf.fitted or not rf.models:
            raise ValueError("Engine payload contains an unfitted or invalid RF model.")
        if (
            len(rf.models) != rf.n_models
            or rf.n_models != cfg.rf_n_models
            or rf.n_estimators != cfg.rf_n_estimators
            or rf.n_jobs != cfg.rf_n_jobs
        ):
            raise ValueError("Engine RF model metadata does not match its config.")
        if hgb is not None and not isinstance(hgb, MonotoneHGBWrapper):
            raise ValueError("Engine payload contains an invalid monotone model.")
        raw_use_monotone = payload.get("use_monotone", False)
        if not isinstance(raw_use_monotone, bool):
            raise ValueError("Engine payload has an invalid use_monotone flag.")
        use_monotone = raw_use_monotone
        if use_monotone and (hgb is None or not hgb.fitted):
            raise ValueError("Engine payload selects an unfitted monotone model.")

        engine = cls(
            domain=domain,
            rf=rf,
            hgb=hgb,
            use_monotone=use_monotone,
            cfg=cfg,
        )
        saved_columns = payload.get("feature_columns")
        if not isinstance(saved_columns, list) or saved_columns != engine.feature_columns:
            raise ValueError("Engine feature metadata does not match its domain.")
        return engine

    def model_info(self) -> dict[str, Any]:
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "use_monotone": self.use_monotone,
            "rf_n_models": self.cfg.rf_n_models,
            "rf_n_estimators": self.cfg.rf_n_estimators,
            "input_columns": [var.name for var in self.domain.continuous]
            + [var.name for var in self.domain.integers]
            + [var.name for var in self.domain.categorical],
            "domain": asdict(self.domain),
            "feature_columns": list(self.feature_columns),
            "config": dump_config(self.cfg),
        }

    def predict_rows_with_inputs(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        canon = [canonicalize_comp_car_input(row, self.domain) for row in rows]
        preds = self.predict_rows(canon)
        out: list[dict[str, Any]] = []
        for row, pred in zip(canon, preds, strict=True):
            item = dict(row)
            item["premium"] = float(np.round(pred, 2))
            out.append(item)
        return out


def load_rows_csv(path: str | Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    if "premium" in df.columns:
        df = df.drop(columns=["premium"])
    return df.to_dict(orient="records")


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["premium"])
    tmp = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(target)


def load_row_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON row input must be an object.")
    return raw
