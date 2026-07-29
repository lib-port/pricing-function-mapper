from __future__ import annotations

import numpy as np
import pytest

from pricing_mapper.acquisition import AcquisitionStrategy, build_context, normalize01
from pricing_mapper.config import AcquisitionConfig, EvaluationConfig, ModelConfig
from pricing_mapper.domain import DomainSpec
from pricing_mapper.evaluation import (
    EarlyStopTracker,
    conformal_bounds,
    conformal_radius,
    interval_report,
    point_metrics,
    regression_report,
)
from pricing_mapper.models import (
    AcquisitionCommittee,
    finite_nonnegative_targets,
    predict_estimator,
    refit_selected,
    select_model,
)
from pricing_mapper.provider import reference_car_quote


def test_metrics_and_conformal_math() -> None:
    actual = np.asarray([100.0, 200.0, 300.0, 400.0])
    predicted = np.asarray([110.0, 190.0, 330.0, 360.0])
    metrics = point_metrics(actual, predicted)
    assert metrics["mae"] == 22.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(675.0))
    assert metrics["wape"] == 0.09
    assert metrics["max_absolute_error"] == 40.0

    radius = conformal_radius(actual, predicted, coverage=0.9)
    assert radius == 40.0
    lower, upper = conformal_bounds(predicted, radius)
    intervals = interval_report(actual, lower, upper)
    assert intervals["coverage"] == 1.0
    assert intervals["mean_width"] == 80.0


def test_empty_and_invalid_metric_inputs_fail_cleanly() -> None:
    with pytest.raises(ValueError, match="at least one"):
        point_metrics([], [])
    with pytest.raises(ValueError, match="aligned"):
        point_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        finite_nonnegative_targets([-1.0])
    assert normalize01(np.asarray([5.0, 5.0])).tolist() == [0.0, 0.0]


def test_risk_slice_report() -> None:
    rows = [
        {
            "driver_age": 20.0,
            "claims_5y": 1,
            "postcode_risk": 0.8,
            "vehicle_value": 120_000.0,
            "usage": "business",
            "parking": "street",
        },
        {
            "driver_age": 65.0,
            "claims_5y": 0,
            "postcode_risk": 0.2,
            "vehicle_value": 20_000.0,
            "usage": "private",
            "parking": "garage",
        },
    ]
    report = regression_report(
        rows,
        [100.0, 200.0],
        [110.0, 190.0],
        evaluation=EvaluationConfig(evaluation_budget=9, bootstrap_iterations=0),
        seed=1,
    )
    assert report["risk_slices"]["young_driver_under_25"]["count"] == 1
    assert report["risk_slices"]["senior_driver_60_plus"]["count"] == 1
    assert report["bootstrap_intervals"]["mae"] == {"lower": 10.0, "upper": 10.0}


def test_acquisition_components_and_greedy_selection_are_deterministic() -> None:
    domain = DomainSpec.default()
    rng = np.random.default_rng(8)
    training = domain.sample_lhs(6, rng)
    candidates = domain.sample_lhs(15, rng)
    context = build_context(
        candidate_rows=candidates,
        training_rows=training,
        predictive_std=np.linspace(0.0, 1.0, len(candidates)),
        training_residuals=np.linspace(-10.0, 10.0, len(training)),
        domain=domain,
    )
    strategy = AcquisitionStrategy(AcquisitionConfig())
    first = strategy.select(context, 5)
    second = strategy.select(context, 5)
    assert first == second
    assert len(first) == len(set(first)) == 5
    _, components = strategy.combined_scores(context)
    assert set(components) == {"uncertainty", "residual", "breakpoint", "diversity"}


def test_committee_and_candidate_selection_use_heldout_validation() -> None:
    domain = DomainSpec.default()
    rng = np.random.default_rng(11)
    mapping_rows = domain.sample_lhs(18, rng)
    validation_rows = domain.sample_lhs(8, rng)
    mapping_targets = np.asarray(
        [
            reference_car_quote(__import__("pricing_mapper").CarQuoteInput.from_mapping(row))
            for row in mapping_rows
        ]
    )
    validation_targets = np.asarray(
        [
            reference_car_quote(__import__("pricing_mapper").CarQuoteInput.from_mapping(row))
            for row in validation_rows
        ]
    )
    model_config = ModelConfig(
        search_iterations=1,
        hgb_max_iter=20,
        extra_trees_estimators=20,
        committee_size=2,
        committee_estimators=10,
        latency_repetitions=5,
        max_p95_latency_ms=1_000.0,
    )
    committee = AcquisitionCommittee(model_config, domain, seed=2).fit(
        mapping_rows,
        mapping_targets,
    )
    mean, standard_deviation = committee.predict(validation_rows)
    assert mean.shape == standard_deviation.shape == (8,)
    assert np.all(standard_deviation >= 0)

    selection = select_model(
        mapping_rows=mapping_rows,
        mapping_targets=mapping_targets,
        validation_rows=validation_rows,
        validation_targets=validation_targets,
        config=model_config,
        domain=domain,
        seed=4,
    )
    assert selection.selected.eligible
    assert selection.report()["selection_rule"].startswith("lowest validation MAE")
    assert "mae_delta_constrained_minus_unconstrained" in selection.monotonic_tradeoff

    fitted = refit_selected(
        selection,
        rows=[*mapping_rows, *validation_rows],
        targets=np.concatenate((mapping_targets, validation_targets)),
        config=model_config,
        domain=domain,
        seed=5,
    )
    predictions = predict_estimator(
        fitted,
        selection.selected.spec.kind,
        __import__("pricing_mapper.encoding", fromlist=["FeatureEncoder"]).FeatureEncoder(domain),
        validation_rows,
    )
    assert predictions.shape == (8,)


def test_early_stopping_requires_validation_confidence_stagnation() -> None:
    from pricing_mapper.config import EarlyStoppingConfig

    config = EarlyStoppingConfig(
        patience_batches=2,
        minimum_batches=2,
        minimum_relative_improvement=0.01,
    )
    tracker = EarlyStopTracker().update(
        mae=10.0,
        lower_bound=9.0,
        upper_bound=11.0,
        config=config,
    )
    tracker = tracker.update(
        mae=9.9,
        lower_bound=9.0,
        upper_bound=10.8,
        config=config,
    )
    assert not tracker.stopped
    tracker = tracker.update(
        mae=9.8,
        lower_bound=8.9,
        upper_bound=10.7,
        config=config,
    )
    assert tracker.stopped
