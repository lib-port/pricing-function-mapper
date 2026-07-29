import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pricing_mapper.active_mapper import ActiveQuoteMapper
from pricing_mapper.artifacts import default_run_id
from pricing_mapper.cli import _run_pricing_mode
from pricing_mapper.config import MapperConfig, load_config, validate_config
from pricing_mapper.domain import build_comp_car_domain, canonicalize_comp_car_input
from pricing_mapper.engine import PricingEngine
from pricing_mapper.quote import mock_comp_car_quote
from pricing_mapper.utils import allocate_integer_counts, propose_segment_pool


def valid_row() -> dict[str, object]:
    return {
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


def small_config(**overrides: object) -> MapperConfig:
    values: dict[str, object] = {
        "budget": 4,
        "init_n": 2,
        "batch_size": 2,
        "pool_size": 40,
        "use_monotone_if_available": False,
        "rf_n_models": 1,
        "rf_n_estimators": 5,
        "rf_n_jobs": 1,
        "checkpoint_every_batches": 0,
    }
    values.update(overrides)
    return MapperConfig(**values)


class RegressionTests(unittest.TestCase):
    def test_canonicalization_rejects_malformed_rows(self) -> None:
        row = valid_row()
        del row["rating"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            canonicalize_comp_car_input(row)

        row = valid_row()
        row["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            canonicalize_comp_car_input(row)

        row = valid_row()
        row["parking"] = "carport"
        with self.assertRaisesRegex(ValueError, "parking"):
            canonicalize_comp_car_input(row)

        row = valid_row()
        row["vehicle_value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            canonicalize_comp_car_input(row)

        row = valid_row()
        row["vehicle_value"] = 10**10000
        with self.assertRaisesRegex(ValueError, "numeric"):
            canonicalize_comp_car_input(row)

    def test_domain_override_validation_is_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown domain_overrides"):
            build_comp_car_domain({"typo": {}})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            build_comp_car_domain({"integers": {"claims_5y": {"low": 0.5, "high": 3}}})
        with self.assertRaisesRegex(ValueError, "infeasible"):
            build_comp_car_domain(
                {
                    "continuous": {"driver_age": {"low": 17, "high": 30}},
                    "integers": {"years_licensed": {"low": 5, "high": 10}},
                }
            )

        domain = build_comp_car_domain({"categorical": {"usage": [1, 2]}})
        sampled = domain.sample_lhs(2, np.random.default_rng(1))
        json.dumps(sampled)

    def test_config_rejects_invalid_numeric_and_targeting_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "rf_n_jobs"):
            validate_config(MapperConfig(rf_n_jobs=-2))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_config(MapperConfig(segment_target_weight=float("nan")))
        with self.assertRaisesRegex(ValueError, "Unknown breakpoint_vars"):
            validate_config(MapperConfig(breakpoint_vars=["unknown"]))
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            validate_config(MapperConfig(segment_constraints={"claims_5y": {"min": 100}}))
        with self.assertRaisesRegex(ValueError, "no valid integer"):
            validate_config(
                MapperConfig(segment_constraints={"claims_5y": {"min": 0.2, "max": 0.8}})
            )
        with self.assertRaisesRegex(ValueError, "infeasible"):
            validate_config(
                MapperConfig(
                    segment_constraints={
                        "driver_age": {"max": 17},
                        "years_licensed": {"min": 2},
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_config(MapperConfig(segment_focus_enabled=True))
        with self.assertRaisesRegex(ValueError, "unsupported by the default"):
            validate_config(
                MapperConfig(domain_overrides={"categorical": {"usage": ["private", "fleet"]}})
            )
        with self.assertRaisesRegex(ValueError, "filenames must be distinct"):
            validate_config(MapperConfig(output_csv="same.csv", engine_path="same.csv"))
        with self.assertRaisesRegex(ValueError, "must identify a file"):
            validate_config(MapperConfig(state_path=".."))

    def test_acquisition_allocation_never_overallocates(self) -> None:
        counts = allocate_integer_counts(2, (0.34, 0.33, 0.33, 0.0))
        self.assertEqual(sum(counts), 2)
        self.assertTrue(all(count >= 0 for count in counts))

    def test_tiny_cv_subsample_disables_cv_cleanly(self) -> None:
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            mock_comp_car_quote,
            small_config(cv_subsample_max=1),
        )
        residuals = mapper._cv_residuals(
            np.zeros((60, len(mapper.encoder.cols))),
            np.ones(60),
        )
        np.testing.assert_array_equal(residuals, np.zeros(60))

    def test_segment_pool_supports_exact_continuous_constraints(self) -> None:
        cfg = small_config(
            segment_focus_enabled=True,
            segment_constraints={
                "driver_age": {"eq": 21.5},
                "claims_5y": {"max": 1},
                "usage": {"in": ["private", "business"]},
            },
            segment_min_candidates=10,
            segment_pool_max_tries=1,
            pool_size=20,
        )
        validate_config(cfg)
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            mock_comp_car_quote,
            cfg,
        )
        pool = mapper._build_candidate_pool(np.empty((0, len(mapper.encoder.cols))))
        segment_count = sum(mapper.row_in_segment(row) for row in pool)
        self.assertGreaterEqual(segment_count, 10)

    def test_segment_pool_enforces_driver_licence_cross_constraint(self) -> None:
        domain = build_comp_car_domain()
        rows = propose_segment_pool(
            domain,
            n=20,
            rng=np.random.default_rng(8),
            constraints={"years_licensed": {"eq": 70}},
        )
        self.assertTrue(all(row["years_licensed"] == 70 for row in rows))
        self.assertTrue(all(row["driver_age"] >= 86 for row in rows))

    def test_load_config_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"budegt": 10}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown config fields"):
                load_config(str(path))

    def test_mapper_always_fits_all_final_samples(self) -> None:
        cfg = small_config(budget=3, init_n=3)
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            mock_comp_car_quote,
            cfg,
        )
        frame, _ = mapper.run()

        self.assertTrue(mapper.rf.fitted)
        self.assertEqual(mapper._last_fit_n, len(frame))
        predictions, _ = mapper.rf.predict_mean_std(mapper.encoder.encode(mapper.x_rows))
        self.assertEqual(len(predictions), len(frame))

    def test_mapper_rejects_invalid_quote_values(self) -> None:
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            lambda _: float("nan"),
            small_config(),
        )
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            mapper.query(valid_row())

    def test_mapper_fails_cleanly_when_no_unique_candidate_exists(self) -> None:
        row = canonicalize_comp_car_input(valid_row())
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            mock_comp_car_quote,
            small_config(budget=2, init_n=1, batch_size=1, pool_size=1),
        )
        with (
            patch(
                "pricing_mapper.active_mapper.propose_pool",
                return_value=[row],
            ),
            self.assertRaisesRegex(RuntimeError, "Unable to propose"),
        ):
            mapper.run()

    def test_state_validation_rejects_mismatched_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            cfg = small_config(state_path=str(state_path))
            mapper = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                cfg,
            )
            mapper.add_samples([valid_row()])
            mapper.save_state(state_path)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["y_vals"] = []
            state_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                cfg,
            )
            with self.assertRaisesRegex(ValueError, "lengths do not match"):
                restored.load_state(state_path)

    def test_resume_requires_an_existing_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = small_config(
                resume=True,
                state_path=str(Path(tmpdir) / "missing.json"),
            )
            mapper = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                cfg,
            )
            with self.assertRaisesRegex(FileNotFoundError, "Cannot resume"):
                mapper.run()

    def test_resume_matches_an_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            common = {
                "budget": 8,
                "init_n": 4,
                "batch_size": 2,
                "pool_size": 80,
                "seed": 91,
                "refit_every_batches": 2,
                "use_monotone_if_available": False,
                "rf_n_models": 1,
                "rf_n_estimators": 8,
                "rf_n_jobs": 1,
            }
            full = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                MapperConfig(**common, checkpoint_every_batches=0),
            )
            full.run()

            state_path = str(Path(tmpdir) / "state.json")
            partial = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                MapperConfig(
                    **{**common, "budget": 6},
                    checkpoint_every_batches=1,
                    state_path=state_path,
                ),
            )
            partial.run()
            saved_state = json.loads(Path(state_path).read_text(encoding="utf-8"))
            self.assertEqual(saved_state["schema_version"], 3)
            self.assertEqual(saved_state["last_fit_n"], 4)
            self.assertEqual(saved_state["batch_count"], 1)

            resumed = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                MapperConfig(
                    **common,
                    checkpoint_every_batches=1,
                    state_path=state_path,
                    resume=True,
                ),
            )
            resumed.run()

            self.assertEqual(resumed.x_rows, full.x_rows)
            self.assertEqual(resumed.y_vals, full.y_vals)

    def test_resume_does_not_restart_an_early_stopped_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            cfg = small_config(
                budget=40,
                init_n=10,
                batch_size=5,
                pool_size=80,
                state_path=state_path,
                checkpoint_every_batches=1,
                early_stop_patience_batches=1,
                early_stop_min_batches=1,
                early_stop_min_rel_improvement=1.0,
            )
            initial = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                cfg,
            )
            initial_frame, initial_stats = initial.run()
            self.assertTrue(initial_stats.early_stopped)

            resumed = ActiveQuoteMapper(
                build_comp_car_domain(),
                mock_comp_car_quote,
                small_config(
                    budget=40,
                    init_n=10,
                    batch_size=5,
                    pool_size=80,
                    state_path=state_path,
                    checkpoint_every_batches=1,
                    early_stop_patience_batches=1,
                    early_stop_min_batches=1,
                    early_stop_min_rel_improvement=1.0,
                    resume=True,
                ),
            )
            resumed_frame, resumed_stats = resumed.run()
            self.assertTrue(resumed_stats.early_stopped)
            self.assertEqual(len(resumed_frame), len(initial_frame))

    def test_engine_handles_an_empty_batch(self) -> None:
        mapper = ActiveQuoteMapper(
            build_comp_car_domain(),
            mock_comp_car_quote,
            small_config(budget=2, init_n=2),
        )
        mapper.run()
        engine = PricingEngine.from_mapper(
            mapper.domain,
            mapper.rf,
            mapper.hgb,
            mapper.use_monotone,
            mapper.cfg,
        )
        self.assertEqual(engine.predict_rows([]).shape, (0,))
        self.assertEqual(engine.predict_rows_with_inputs([]), [])

    def test_pricing_mode_rejects_conflicts_before_loading_engine(self) -> None:
        args = argparse.Namespace(
            engine_path="missing.pkl",
            serve_api=False,
            host="127.0.0.1",
            port=8000,
            price_row="{}",
            price_row_json="row.json",
            price_input_csv=None,
            price_output_csv=None,
        )
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            _run_pricing_mode(
                args,
                logging.getLogger("test"),
            )

    def test_pricing_mode_rejects_an_empty_selected_argument(self) -> None:
        args = argparse.Namespace(
            engine_path="missing.pkl",
            serve_api=False,
            host="127.0.0.1",
            port=8000,
            price_row="",
            price_row_json=None,
            price_input_csv=None,
            price_output_csv=None,
        )
        with self.assertRaisesRegex(ValueError, "--price-row cannot be empty"):
            _run_pricing_mode(args, logging.getLogger("test"))

    def test_generated_run_ids_do_not_collide(self) -> None:
        self.assertNotEqual(default_run_id(42), default_run_id(42))


if __name__ == "__main__":
    unittest.main()
