import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path

from pricing_mapper.cli import _run_pricing_mode, _run_single
from pricing_mapper.config import MapperConfig
from pricing_mapper.engine import PricingEngine


class EngineCliTests(unittest.TestCase):
    def test_engine_export_and_load_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MapperConfig(
                budget=12,
                init_n=6,
                batch_size=3,
                pool_size=120,
                output_dir=tmpdir,
                run_id="engine_case",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=20,
                checkpoint_every_batches=0,
            )
            logger = logging.getLogger("test_engine")
            _run_single(cfg, logger)

            engine_path = Path(tmpdir) / "engine_case" / "pricing_engine.pkl"
            self.assertTrue(engine_path.exists())
            engine = PricingEngine.load(engine_path)

            row = {
                "driver_age": 30,
                "years_licensed": 10,
                "vehicle_year": 2020,
                "vehicle_value": 30000,
                "annual_km": 12000,
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
            premium = engine.predict_row(row)
            self.assertGreater(premium, 0.0)

    def test_pricing_mode_batch_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MapperConfig(
                budget=10,
                init_n=5,
                batch_size=5,
                pool_size=100,
                output_dir=tmpdir,
                run_id="price_batch_case",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=20,
                checkpoint_every_batches=0,
            )
            logger = logging.getLogger("test_engine")
            _run_single(cfg, logger)

            engine_path = Path(tmpdir) / "price_batch_case" / "pricing_engine.pkl"
            in_csv = Path(tmpdir) / "rows.csv"
            in_csv.write_text(
                "driver_age,years_licensed,vehicle_year,vehicle_value,annual_km,claims_5y,convictions_5y,postcode_risk,theft_risk,excess,usage,parking,hire_car,windscreen,rating\n"
                "35,15,2021,40000,12000,0,0,0.2,0.1,600,private,garage,none,no,market\n"
            )
            out_csv = Path(tmpdir) / "priced.csv"

            args = argparse.Namespace(
                engine_path=str(engine_path),
                serve_api=False,
                host="127.0.0.1",
                port=8000,
                price_row=None,
                price_row_json=None,
                price_input_csv=str(in_csv),
                price_output_csv=str(out_csv),
            )
            code = _run_pricing_mode(args, logger)
            self.assertEqual(code, 0)
            self.assertTrue(out_csv.exists())
            lines = out_csv.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("premium", lines[0])

    def test_pricing_mode_single_row_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MapperConfig(
                budget=10,
                init_n=5,
                batch_size=5,
                pool_size=100,
                output_dir=tmpdir,
                run_id="price_single_case",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=20,
                checkpoint_every_batches=0,
            )
            logger = logging.getLogger("test_engine")
            _run_single(cfg, logger)

            engine_path = Path(tmpdir) / "price_single_case" / "pricing_engine.pkl"
            row_path = Path(tmpdir) / "row.json"
            row_path.write_text(
                json.dumps(
                    {
                        "driver_age": 40,
                        "years_licensed": 20,
                        "vehicle_year": 2022,
                        "vehicle_value": 35000,
                        "annual_km": 10000,
                        "claims_5y": 0,
                        "convictions_5y": 0,
                        "postcode_risk": 0.2,
                        "theft_risk": 0.2,
                        "excess": 700,
                        "usage": "private",
                        "parking": "garage",
                        "hire_car": "none",
                        "windscreen": "no",
                        "rating": "market",
                    }
                )
            )
            args = argparse.Namespace(
                engine_path=str(engine_path),
                serve_api=False,
                host="127.0.0.1",
                port=8000,
                price_row=None,
                price_row_json=str(row_path),
                price_input_csv=None,
                price_output_csv=None,
            )
            code = _run_pricing_mode(args, logger)
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
