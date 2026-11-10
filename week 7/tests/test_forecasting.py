"""Week 5 forecasting validation. Run from project root: python -m pytest tests/test_forecasting.py -v.

The dataset is SYNTHETIC. Tests deliberately do not claim forecast flags detect the
observed low-coverage events unless their detection rate actually supports it.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.forecasting import (
    DATA_PATH, load_aido_data, trailing_four_baseline,
    prepare_lagged_features, expanding_window_forecasts, model_metrics,
)


@pytest.fixture(scope="module")
def aido():
    return load_aido_data()


@pytest.fixture(scope="module")
def forecasts(aido):
    return expanding_window_forecasts(aido)


@pytest.fixture(scope="module")
def scored(forecasts):
    return model_metrics(forecasts)


def test_dataset_exists():
    assert DATA_PATH.is_file()


def test_dataset_nonempty(aido):
    assert len(aido) > 0


def test_only_aido(aido):
    assert aido.platform.eq("Aido Rover").all()


def test_ten_deployments(aido):
    assert aido.deployment_id.nunique() == 10


def test_120_rows(aido):
    assert len(aido) == 120


@pytest.mark.parametrize("col", ["deployment_id", "date", "patrol_coverage_rate", "incident_detection_rate", "uptime_pct"])
def test_required_column_is_present_and_not_null(aido, col):
    assert col in aido.columns and aido[col].notna().all()


def test_dates_are_datetime(aido):
    assert pd.api.types.is_datetime64_any_dtype(aido.date)


def test_unique_deployment_dates(aido):
    assert not aido.duplicated(["deployment_id", "date"]).any()


def test_sorted_by_deployment_date(aido):
    assert_frame_equal(aido, aido.sort_values(["deployment_id", "date"]).reset_index(drop=True))


def test_duplicate_rows_rejected(tmp_path, aido):
    f = tmp_path / "dupes.csv"
    pd.concat([aido, aido.iloc[[0]]], ignore_index=True).to_csv(f, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_aido_data(f)


def test_missing_target_rejected(tmp_path, aido):
    f = tmp_path / "missing.csv"
    copy = aido.copy()
    copy.loc[0, "patrol_coverage_rate"] = np.nan
    copy.to_csv(f, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_aido_data(f)


def test_no_aido_rejected(tmp_path, aido):
    f = tmp_path / "empty.csv"
    copy = aido.copy()
    copy["platform"] = "Fari"
    copy.to_csv(f, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_aido_data(f)


def test_baseline_first_four_are_nan(aido):
    out = trailing_four_baseline(aido)
    assert out.groupby("deployment_id").head(4).baseline_pred.isna().all()


def test_baseline_prediction_count(aido):
    assert trailing_four_baseline(aido).baseline_pred.notna().sum() == 80


@pytest.mark.parametrize("deployment_index", range(10))
def test_baseline_matches_manual_previous_four(aido, deployment_index):
    robot = sorted(aido.deployment_id.unique())[deployment_index]
    sub = trailing_four_baseline(aido).query("deployment_id == @robot").reset_index(drop=True)
    assert sub.loc[4, "baseline_pred"] == pytest.approx(sub.loc[:3, "patrol_coverage_rate"].mean())
    assert sub.loc[11, "baseline_pred"] == pytest.approx(sub.loc[7:10, "patrol_coverage_rate"].mean())


def test_baseline_ignores_current_target(aido):
    changed = aido.copy()
    robot = changed.deployment_id.iloc[0]
    idx = changed.index[(changed.deployment_id == robot)][5]
    expected = trailing_four_baseline(aido).loc[idx, "baseline_pred"]
    changed.loc[idx, "patrol_coverage_rate"] = 0.0
    actual = trailing_four_baseline(changed).loc[idx, "baseline_pred"]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("source", ["patrol_coverage_rate", "incident_detection_rate", "uptime_pct"])
def test_lag_first_month_null(aido, source):
    out = prepare_lagged_features(aido)
    assert out.groupby("deployment_id").head(1)[f"lag1_{source}"].isna().all()


@pytest.mark.parametrize("source", ["patrol_coverage_rate", "incident_detection_rate", "uptime_pct"])
def test_lag_equals_previous_value(aido, source):
    out = prepare_lagged_features(aido)
    for _, sub in out.groupby("deployment_id"):
        np.testing.assert_allclose(sub[f"lag1_{source}"].iloc[1:], sub[source].iloc[:-1])


def test_day_of_quarter_valid(aido):
    assert prepare_lagged_features(aido).day_of_quarter.between(1, 92).all()


def test_oos_80_forecasts(forecasts):
    assert len(forecasts) == 80


def test_oos_unique_deployment_dates(forecasts):
    assert not forecasts.duplicated(["deployment_id", "date"]).any()


def test_no_lookahead_all_windows(forecasts):
    # Critical: all reported most-recent training dates must strictly precede forecast dates.
    assert (forecasts.train_latest_date < forecasts.date).all()


def test_expanding_window_training_rows_increase(forecasts):
    counts = forecasts.groupby("date").ridge_training_rows.first().sort_index()
    assert counts.is_monotonic_increasing and (counts.diff().dropna() > 0).all()


def test_same_train_size_at_same_date(forecasts):
    assert forecasts.groupby("date").ridge_training_rows.nunique().eq(1).all()


def test_forecast_numeric_finite(forecasts):
    for col in ["actual", "baseline_pred", "holt_pred", "ridge_pred"]:
        assert np.isfinite(forecasts[col].to_numpy(dtype=float)).all()


def test_baseline_matches_source(forecasts, aido):
    baseline = trailing_four_baseline(aido)[["deployment_id", "date", "baseline_pred"]]
    merged = forecasts.merge(baseline, on=["deployment_id", "date"], suffixes=("_forecast", "_source"))
    np.testing.assert_allclose(merged.baseline_pred_forecast, merged.baseline_pred_source)


def test_actual_matches_source(forecasts, aido):
    actual = aido[["deployment_id", "date", "patrol_coverage_rate"]]
    merged = forecasts.merge(actual, on=["deployment_id", "date"])
    np.testing.assert_allclose(merged.actual, merged.patrol_coverage_rate)


def test_flag_is_exactly_ridge_below_90(forecasts):
    assert (forecasts.intervention_flag == (forecasts.ridge_pred < 90)).all()


def test_flag_dtype_boolean(forecasts):
    assert pd.api.types.is_bool_dtype(forecasts.intervention_flag)


def test_low_event_detection_is_measured_not_assumed(forecasts):
    actual_low = forecasts.actual < 90
    flagged = forecasts.intervention_flag
    true_positives = int((actual_low & flagged).sum())
    assert 0 <= true_positives <= int(actual_low.sum())
    # A zero detection count is a genuine limitation, not a passing accuracy claim.


def test_metrics_three_models(scored):
    assert len(scored) == 3


def test_metrics_columns(scored):
    assert {"model", "MAE", "RMSE"}.issubset(scored.columns)


def test_mae_matches_manual(forecasts, scored):
    for _, row in scored.iterrows():
        col = {"Layer 1 - Baseline": "baseline_pred", "Layer 2 - Holt": "holt_pred", "Layer 3 - Ridge": "ridge_pred"}[row["model"]]
        assert row.MAE == pytest.approx(np.mean(np.abs(forecasts[col] - forecasts.actual)))


def test_rmse_matches_manual(forecasts, scored):
    for _, row in scored.iterrows():
        col = {"Layer 1 - Baseline": "baseline_pred", "Layer 2 - Holt": "holt_pred", "Layer 3 - Ridge": "ridge_pred"}[row["model"]]
        assert row.RMSE == pytest.approx(np.sqrt(np.mean((forecasts[col] - forecasts.actual) ** 2)))


def test_rmse_at_least_mae(scored):
    assert (scored.RMSE >= scored.MAE).all()


def test_metrics_nonnegative(scored):
    assert (scored[["MAE", "RMSE"]].to_numpy() >= 0).all()


def test_metrics_zero_for_perfect_predictions():
    fake = pd.DataFrame({"actual": [92., 95.], "baseline_pred": [92., 95.], "holt_pred": [92., 95.], "ridge_pred": [92., 95.]})
    result = model_metrics(fake)
    assert (result[["MAE", "RMSE"]] == 0).all().all()


def test_repeated_metric_calculation_is_identical(forecasts):
    assert_frame_equal(model_metrics(forecasts), model_metrics(forecasts))


def test_full_pipeline_reproducible(aido, forecasts):
    rerun = expanding_window_forecasts(aido)
    assert_frame_equal(forecasts, rerun)


def test_future_observations_do_not_change_earlier_forecasts(aido, forecasts):
    # Change targets at the final date only; earlier predictions must remain invariant.
    perturbed = aido.copy()
    last_date = perturbed.date.max()
    perturbed.loc[perturbed.date.eq(last_date), "patrol_coverage_rate"] = 10.0
    rerun = expanding_window_forecasts(perturbed)
    prior = forecasts[forecasts.date < last_date].reset_index(drop=True)
    altered_prior = rerun[rerun.date < last_date].reset_index(drop=True)
    assert_frame_equal(prior, altered_prior)


# Supplemental Week 5 validation: coefficient changes and operational detection.
from src.forecasting import ridge_coefficient_stability, intervention_diagnostics

def test_coefficient_audit_uses_only_past_dates(aido):
    audit=ridge_coefficient_stability(aido)
    assert len(audit)>0 and (audit.training_latest_date<audit.prediction_date).all()

def test_coefficient_audit_has_four_features(aido):
    audit=ridge_coefficient_stability(aido)
    assert audit.feature.nunique()==4

def test_coefficient_change_formula(aido):
    audit=ridge_coefficient_stability(aido)
    defined=audit.loc[audit.relative_change_pct.notna()]
    np.testing.assert_allclose(defined.relative_change_pct,100*defined.absolute_change/defined.prior_coefficient.abs())

def test_coefficient_stability_breaches_are_reported_accurately(aido):
    # Validate the audit's identification of threshold violations; report status
    # separately rather than masking the real model's noncompliance.
    audit=ridge_coefficient_stability(aido)
    comparable=audit.relative_change_pct.notna()
    assert comparable.any()
    assert (audit.loc[comparable,'within_20pct'].astype(bool).to_numpy() ==
            audit.loc[comparable,'relative_change_pct'].le(20).to_numpy()).all()
    assert int((audit.relative_change_pct>20).sum())==len(audit[audit.within_20pct==False])


def test_alert_confusion_matrix_exhaustive(forecasts):
    d=intervention_diagnostics(forecasts)
    assert d['TP']+d['FP']+d['FN']+d['TN']==len(forecasts)

def test_alert_recall_matches_manual(forecasts):
    d=intervention_diagnostics(forecasts)
    count=int((forecasts.actual<90).sum())
    assert d['recall']==pytest.approx(d['TP']/count) if count else d['recall'] is None

def test_alert_precision_undefined_without_positive_predictions(forecasts):
    d=intervention_diagnostics(forecasts)
    if d['TP']+d['FP']==0:
        assert d['precision'] is None

def test_alert_diagnostics_detect_perfect_case():
    fake=pd.DataFrame({'actual':[80.,95.,80.,99.], 'ridge_pred':[80.,95.,95.,80.]})
    d=intervention_diagnostics(fake)
    assert (d['TP'],d['FP'],d['FN'],d['TN'])==(1,1,1,1)
    assert d['precision']==d['recall']==0.5
