import json
import logging
import tempfile
import unittest
from pathlib import Path

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.benchmark import BENCHMARK_PRESETS, run_benchmark
from pricing_mapper.cli import _run_single
from pricing_mapper.config import MapperConfig, validate_config
from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.quote import mock_comp_car_quote


class ConfigResumeCliTests(unittest.TestCase):
    def test_validate_config_acquisition_mix_sum(self) -> None:
        cfg = MapperConfig(acquisition_mix=(0.5, 0.2, 0.2, 0.2))
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_validate_config_rf_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            validate_config(MapperConfig(rf_n_models=0))
        with self.assertRaises(ValueError):
            validate_config(MapperConfig(rf_n_estimators=0))
        with self.assertRaises(ValueError):
            validate_config(MapperConfig(rf_n_jobs=0))

    def test_validate_config_unknown_domain_override_key(self) -> None:
        cfg = MapperConfig(
            domain_overrides={"continuous": {"unknown_feature": {"low": 0, "high": 1}}}
        )
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_state_corruption_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad_state.json"
            bad.write_text("{not-json")

            mapper = ActiveQuoteMapper(
                domain=build_comp_car_domain(),
                quote_fn=mock_comp_car_quote,
                cfg=MapperConfig(
                    budget=10,
                    init_n=5,
                    batch_size=5,
                    pool_size=100,
                    use_monotone_if_available=False,
                    rf_n_models=2,
                    rf_n_estimators=20,
                    checkpoint_every_batches=0,
                ),
            )
            with self.assertRaises(ValueError):
                mapper.load_state(bad)

    def test_resume_and_metadata_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = MapperConfig(
                budget=12,
                init_n=6,
                batch_size=3,
                pool_size=150,
                output_dir=tmpdir,
                run_id="resume_case",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=30,
                checkpoint_every_batches=1,
            )

            logger = logging.getLogger("test_cli")
            _run_single(base_cfg, logger)

            resumed = MapperConfig(
                budget=16,
                init_n=6,
                batch_size=3,
                pool_size=150,
                output_dir=tmpdir,
                run_id="resume_case",
                resume=True,
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=30,
                checkpoint_every_batches=1,
            )
            _run_single(resumed, logger)

            run_dir = Path(tmpdir) / "resume_case"
            csv_path = run_dir / "comp_car_quotes_advanced.csv"
            meta_path = run_dir / "run_metadata.json"
            engine_path = run_dir / "pricing_engine.pkl"

            self.assertTrue(csv_path.exists())
            self.assertTrue(meta_path.exists())
            self.assertTrue(engine_path.exists())

            with meta_path.open() as f:
                meta = json.load(f)

            self.assertIn("stats", meta)
            self.assertIn("mae_rf", meta)
            self.assertIn("artifacts", meta)
            self.assertIn("profile_seconds", meta["stats"])
            self.assertEqual(meta["artifacts"]["run_id"], "resume_case")
            self.assertEqual(meta["artifacts"]["engine_path"], str(engine_path))

            lines = csv_path.read_text().strip().splitlines()
            self.assertEqual(len(lines) - 1, 16)

    def test_resume_rejects_incompatible_state_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = MapperConfig(
                budget=10,
                init_n=5,
                batch_size=5,
                pool_size=120,
                output_dir=tmpdir,
                run_id="resume_mismatch_case",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=30,
                checkpoint_every_batches=1,
            )
            logger = logging.getLogger("test_cli")
            _run_single(base_cfg, logger)

            resumed = MapperConfig(
                budget=12,
                init_n=5,
                batch_size=5,
                pool_size=120,
                output_dir=tmpdir,
                run_id="resume_mismatch_case",
                resume=True,
                quote_provider="pricing_mapper.quote:mock_comp_car_quote",
                use_monotone_if_available=False,
                rf_n_models=2,
                rf_n_estimators=30,
                checkpoint_every_batches=1,
            )
            with self.assertRaises(ValueError):
                _run_single(resumed, logger)

    def test_benchmark_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MapperConfig(
                budget=10,
                init_n=5,
                batch_size=5,
                pool_size=120,
                output_dir=tmpdir,
                run_id="bench_case",
                use_monotone_if_available=False,
                rf_n_jobs=1,
                checkpoint_every_batches=0,
            )
            out = str(Path(tmpdir) / "bench.json")
            payload = run_benchmark(cfg, out)
            self.assertEqual(len(payload["results"]), len(BENCHMARK_PRESETS))
            self.assertTrue(Path(out).exists())
            self.assertTrue(Path(out).with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
