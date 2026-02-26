import unittest

import numpy as np

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.config import MapperConfig
from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.quote import mock_comp_car_quote
from pricing_mapper.utils import binary_search_breakpoint, jitter_around


class UtilsAndMapperTests(unittest.TestCase):
    def test_jitter_around_size(self) -> None:
        domain = build_comp_car_domain()
        rng = np.random.default_rng(0)
        base = domain.sample_lhs(1, rng)[0]
        rows = jitter_around(base, domain, rng, n=7)
        self.assertEqual(len(rows), 7)

    def test_binary_search_breakpoint_returns_points(self) -> None:
        base = {
            "driver_age": 30,
            "years_licensed": 10,
            "vehicle_year": 2020,
            "vehicle_value": 50000,
            "annual_km": 10000,
            "claims_5y": 0,
            "convictions_5y": 0,
            "postcode_risk": 0.2,
            "theft_risk": 0.1,
            "excess": 500,
            "usage": "private",
            "parking": "garage",
            "hire_car": "none",
            "windscreen": "no",
            "rating": "market",
        }

        def predict_fn(rows):
            vals = np.asarray([r["driver_age"] for r in rows], dtype=float)
            return vals * 10.0

        pts = binary_search_breakpoint(
            base,
            "driver_age",
            17,
            90,
            predict_fn,
            max_queries=4,
            threshold=5.0,
        )
        self.assertEqual(len(pts), 4)

    def test_small_integration_run(self) -> None:
        cfg = MapperConfig(
            budget=30,
            init_n=15,
            batch_size=5,
            pool_size=1000,
            seed=123,
            output_csv="/tmp/out.csv",
            output_metadata_json="/tmp/meta.json",
            state_path="/tmp/state.json",
            checkpoint_every_batches=0,
            refit_every_batches=1,
        )
        mapper = ActiveQuoteMapper(
            domain=build_comp_car_domain(),
            quote_fn=mock_comp_car_quote,
            cfg=cfg,
        )
        df, stats = mapper.run()
        self.assertEqual(len(df), 30)
        self.assertEqual(stats.samples, 30)


if __name__ == "__main__":
    unittest.main()
