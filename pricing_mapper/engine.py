from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pricing_mapper.config import MapperConfig, dump_config
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
    ):
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
        canon = [canonicalize_comp_car_input(row, self.domain) for row in rows]
        x_eval, _ = encode_features(self.domain, canon)
        if self.use_monotone and self.hgb is not None:
            preds = self.hgb.predict(x_eval)
        else:
            preds, _ = self.rf.predict_mean_std(x_eval)
        return np.asarray(preds, dtype=float)

    def predict_row(self, row: dict[str, Any]) -> float:
        preds = self.predict_rows([row])
        return float(preds[0])

    def save(self, path: str | Path) -> None:
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
        version = int(payload.get("schema_version", -1))
        if version != ENGINE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported engine schema version: {version}")

        cfg_raw = payload.get("cfg")
        if not isinstance(cfg_raw, dict):
            raise ValueError("Engine payload missing config.")
        if "acquisition_mix" in cfg_raw:
            cfg_raw["acquisition_mix"] = tuple(cfg_raw["acquisition_mix"])
        cfg = MapperConfig(**cfg_raw)
        engine = cls(
            domain=payload["domain"],
            rf=payload["rf"],
            hgb=payload.get("hgb"),
            use_monotone=bool(payload.get("use_monotone", False)),
            cfg=cfg,
        )
        return engine

    def model_info(self) -> dict[str, Any]:
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "use_monotone": self.use_monotone,
            "rf_n_models": self.cfg.rf_n_models,
            "rf_n_estimators": self.cfg.rf_n_estimators,
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
    pd.DataFrame(rows).to_csv(target, index=False)


def load_row_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("JSON row input must be an object.")
    return raw
