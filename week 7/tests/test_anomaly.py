"""Week 3: automated checks for the synthetic robot anomaly pipeline.

From the week3_step2 project root, run:
    python -m pytest -q tests/test_anomaly.py

All performance assertions refer ONLY to our Week 2 SYNTHETIC labeled dataset.
The labels are used for evaluation, not as model prediction features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    PLATFORM_METRICS, REQUIRED_COLUMNS,
    load_data, load_kpi_config, calculate_kpis, detect_anomalies,
    contribution_analysis, build_anomaly_report, evaluate_against_labels,
    run_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "W02_Robot_Deployment_Dataset.csv"
CONFIG_FILE = ROOT / "config" / "W03_KPI_Framework.xlsx"


@pytest.fixture(scope="module")
def data():
    return load_data(DATA_FILE)


@pytest.fixture(scope="module")
def config():
    return load_kpi_config(CONFIG_FILE)


@pytest.fixture(scope="module")
def calculated(data, config):
    return calculate_kpis(data, config)


@pytest.fixture(scope="module")
def detected(calculated):
    readings, _ = calculated
    return detect_anomalies(readings)


@pytest.fixture(scope="module")
def stats(detected, data):
    _, predictions = detected
    return evaluate_against_labels(predictions, data)


def reject_csv(tmp_path, raw, error):
    """Save modified input temporarily; assert the loader catches the issue."""
    path = tmp_path / "invalid.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_data(path)


# ---------------- 1. Dataset ingestion and quality checks ----------------

def test_01_dataset_has_360_rows(data):
    assert len(data) == 360


def test_02_dataset_has_all_three_platforms(data):
    assert set(data["platform"]) == set(PLATFORM_METRICS)


def test_03_each_platform_has_120_rows(data):
    assert (data.groupby("platform").size() == 120).all()


def test_04_each_platform_has_ten_deployments(data):
    assert (data.groupby("platform")["deployment_id"].nunique() == 10).all()


def test_05_dataset_has_twelve_months(data):
    assert data["date"].dt.to_period("M").nunique() == 12


def test_06_all_rows_labeled_synthetic(data):
    assert data["data_label"].eq("SYNTHETIC").all()


def test_07_required_columns_exist(data):
    assert REQUIRED_COLUMNS.issubset(data.columns)


def test_08_deployment_ids_never_missing(data):
    assert data["deployment_id"].notna().all()


def test_09_deployment_months_are_unique(data):
    assert not data.duplicated(["deployment_id", "date"]).any()


def test_10_dates_are_datetime(data):
    assert pd.api.types.is_datetime64_any_dtype(data["date"])


def test_11_labels_are_binary_and_planted_count_is_38(data):
    assert set(data["anomaly_flag"]) == {0, 1}
    assert int(data["anomaly_flag"].sum()) == 38


@pytest.mark.parametrize("missing", ["deployment_id", "platform", "anomaly_flag"])
def test_12_missing_required_column_is_rejected(tmp_path, missing):
    raw = pd.read_csv(DATA_FILE).drop(columns=[missing])
    reject_csv(tmp_path, raw, "Missing input columns")


def test_13_blank_deployment_id_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    raw.loc[0, "deployment_id"] = None
    reject_csv(tmp_path, raw, "deployment_id contains missing")


def test_14_non_synthetic_label_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    raw.loc[0, "data_label"] = "REAL"
    reject_csv(tmp_path, raw, "SYNTHETIC")


def test_15_unknown_platform_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    raw.loc[0, "platform"] = "UNKNOWN"
    reject_csv(tmp_path, raw, "Unrecognized")


def test_16_duplicate_deployment_month_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    reject_csv(tmp_path, raw, "Duplicate")


def test_17_nonbinary_anomaly_flag_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    raw.loc[0, "anomaly_flag"] = 4
    reject_csv(tmp_path, raw, "anomaly_flag")


def test_18_missing_applicable_sensor_value_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_FILE)
    index = raw.index[raw["platform"].eq("Aido Rover")][0]
    raw.loc[index, "patrol_coverage_rate"] = np.nan
    reject_csv(tmp_path, raw, "Missing applicable values")


def test_19_fari_derived_minutes_is_multiplication(data):
    fari = data.loc[data["platform"].eq("Fari")]
    expected = fari["avg_session_duration_min"] * fari["daily_engagement_sessions"]
    np.testing.assert_allclose(fari["estimated_daily_engagement_min"], expected)


# ---------------- 2. KPI config and computations ----------------

def test_20_config_defines_fifteen_kpis(config):
    assert len(config) == 15
    assert config["kpi_id"].nunique() == 15


def test_21_config_has_five_kpis_per_platform(config):
    assert (config.groupby("platform").size() == 5).all()


def test_22_all_z_cutoffs_positive(config):
    assert (config["z_cutoff"] > 0).all()


def test_23_all_thresholds_parameterized_in_config(config):
    assert config["alert_threshold"].notna().all()
    assert config["alert_operator"].isin(["<", ">"]).all()


def test_24_has_1800_deployment_kpi_readings(calculated):
    readings, _ = calculated
    assert len(readings) == 360 * 5


def test_25_has_180_monthly_platform_kpis(calculated):
    _, monthly = calculated
    assert len(monthly) == 3 * 5 * 12


@pytest.mark.parametrize("platform,kpi_id,source", [
    ("Aido Rover", "AR01", "patrol_coverage_rate"),
    ("Sentinel Prime AI", "SP03", "battery_consumption_per_cycle"),
    ("Fari", "FA02", "daily_engagement_sessions"),
])
def test_26_three_manual_kpi_matches(data, calculated, platform, kpi_id, source):
    """The KPI output must equal the relevant input reading exactly."""
    readings, _ = calculated
    row = data.loc[data["platform"].eq(platform)].iloc[0]
    actual = readings.loc[
        readings["platform"].eq(platform)
        & readings["deployment_id"].eq(row["deployment_id"])
        & readings["date"].eq(row["date"])
        & readings["kpi_id"].eq(kpi_id), "value",
    ].iloc[0]
    assert actual == pytest.approx(row[source])


def test_27_monthly_mean_matches_manual_average(data, calculated):
    _, monthly = calculated
    date = pd.Timestamp("2025-06-01")
    manual = data.loc[data["platform"].eq("Aido Rover")
                      & data["date"].eq(date), "patrol_coverage_rate"].mean()
    actual = monthly.loc[monthly["platform"].eq("Aido Rover")
                         & monthly["kpi_id"].eq("AR01")
                         & monthly["date"].eq(date), "mean_value"].iloc[0]
    assert actual == pytest.approx(manual)


def test_28_mom_change_matches_manual_difference(calculated):
    _, monthly = calculated
    s = monthly.loc[monthly["platform"].eq("Aido Rover")
                    & monthly["kpi_id"].eq("AR01")].sort_values("date")
    assert s.iloc[1]["mom_change"] == pytest.approx(
        s.iloc[1]["mean_value"] - s.iloc[0]["mean_value"]
    )


def test_29_first_month_mom_change_is_nan(calculated):
    _, monthly = calculated
    first = monthly.sort_values("date").groupby(["platform", "kpi_id"]).head(1)
    assert first["mom_change"].isna().all()


def test_30_rolling_four_mean_matches_manual(calculated):
    _, monthly = calculated
    s = monthly.loc[monthly["platform"].eq("Aido Rover")
                    & monthly["kpi_id"].eq("AR01")].sort_values("date")
    assert s.iloc[3]["rolling4_mean"] == pytest.approx(s.iloc[:4]["mean_value"].mean())


def test_31_no_duplicate_deployment_kpi_months(calculated):
    readings, _ = calculated
    assert not readings.duplicated(["deployment_id", "date", "kpi_id"]).any()


# ---------------- 3. Detection accuracy, parameter changes, leakage ----------------

def test_32_one_prediction_per_original_record(detected, data):
    _, predictions = detected
    assert len(predictions) == len(data)
    assert not predictions.duplicated(["deployment_id", "date"]).any()


def test_33_all_planted_anomalies_detected(stats):
    assert stats["true_positive"] == 38
    assert stats["false_negative"] == 0
    assert stats["recall"] == pytest.approx(1.0)


def test_34_false_positive_rate_below_five_percent(stats):
    assert stats["false_positive_rate"] < 0.05


def test_35_confusion_matrix_reconciles(stats):
    assert sum(stats[k] for k in (
        "true_positive", "false_positive", "false_negative", "true_negative"
    )) == stats["rows"] == 360


def test_36_precision_matches_manual_formula(stats):
    expected = stats["true_positive"] / (stats["true_positive"] + stats["false_positive"])
    assert stats["precision"] == pytest.approx(expected)


def test_37_predictions_are_binary(detected):
    _, predictions = detected
    assert predictions["pred_anomaly"].isin([0, 1]).all()


def test_38_zscore_threshold_1_5_increases_kpi_flags(calculated, detected):
    readings, _ = calculated
    scored_default, _ = detected
    scored_lower, _ = detect_anomalies(readings, z_cutoff_override=1.5)
    assert scored_lower["z_flag"].sum() > scored_default["z_flag"].sum()


def test_39_lower_zscore_threshold_increases_deployment_alerts(calculated, detected):
    readings, _ = calculated
    _, default_predictions = detected
    _, changed_predictions = detect_anomalies(readings, z_cutoff_override=1.5)
    assert changed_predictions["pred_anomaly"].sum() > default_predictions["pred_anomaly"].sum()


@pytest.mark.parametrize("bad", [0, -1.0])
def test_40_nonpositive_zscore_override_rejected(calculated, bad):
    readings, _ = calculated
    with pytest.raises(ValueError, match="positive"):
        detect_anomalies(readings, z_cutoff_override=bad)


def test_41_changing_excel_threshold_changes_flags(calculated):
    readings, _ = calculated
    changed = readings.copy()
    mask = changed["kpi_id"].eq("AR01")
    changed.loc[mask, "alert_threshold"] = 100.0
    changed_scored, _ = detect_anomalies(changed)
    assert changed_scored.loc[mask, "threshold_flag"].all()


def test_42_renewal_outcome_is_not_used_for_anomaly_prediction(data, config):
    """Week 6 renewal outcomes must not leak into Week 3 detection."""
    original_readings, _ = calculate_kpis(data, config)
    _, original_preds = detect_anomalies(original_readings)
    modified = data.copy()
    if "renewal_status" in modified:
        modified["renewal_status"] = "CHANGED_FOR_TEST"
    changed_readings, _ = calculate_kpis(modified, config)
    _, changed_preds = detect_anomalies(changed_readings)
    assert original_preds["pred_anomaly"].tolist() == changed_preds["pred_anomaly"].tolist()


def test_43_ground_truth_labels_not_used_as_prediction_features(data, config):
    original_readings, _ = calculate_kpis(data, config)
    _, original_preds = detect_anomalies(original_readings)
    modified = data.copy()
    modified["anomaly_flag"] = 1 - modified["anomaly_flag"]
    changed_readings, _ = calculate_kpis(modified, config)
    _, changed_preds = detect_anomalies(changed_readings)
    assert original_preds["pred_anomaly"].tolist() == changed_preds["pred_anomaly"].tolist()


# ---------------- 4. Contribution analysis and anomaly report ----------------

def test_44_contribution_is_sorted_highest_first(detected):
    scored, _ = detected
    result = contribution_analysis(scored, "Aido Rover", "2025-06-01", "AR01")
    assert result["allocated_deviation"].is_monotonic_decreasing


def test_45_contribution_sum_equals_mean_deviation(detected):
    scored, _ = detected
    result = contribution_analysis(scored, "Aido Rover", "2025-06-01", "AR01")
    assert result["signed_contribution"].sum() == pytest.approx(
        result["total_signed_deviation"].iloc[0]
    )
    assert result["allocated_deviation"].sum() == pytest.approx(
        abs(result["total_signed_deviation"].iloc[0])
    )


def test_46_contribution_shares_total_100_percent(detected):
    scored, _ = detected
    result = contribution_analysis(scored, "Aido Rover", "2025-06-01", "AR01")
    assert result["share_pct"].sum() == pytest.approx(100.0)
    assert (result["share_pct"] >= 0).all()


def test_47_aido_005_is_top_june_patrol_contributor(detected):
    scored, _ = detected
    result = contribution_analysis(scored, "Aido Rover", "2025-06-01", "AR01")
    assert result.iloc[0]["deployment_id"] == "AIDO-005"


@pytest.mark.parametrize("platform,date,kpi_id", [
    ("Aido Rover", "2025-06-01", "AR01"),
    ("Fari", "2025-12-01", "FA01"),
    ("Sentinel Prime AI", "2025-07-01", "SP01"),
])
def test_48_contributors_reconcile_across_platforms(detected, platform, date, kpi_id):
    scored, _ = detected
    result = contribution_analysis(scored, platform, date, kpi_id)
    assert len(result) == 10
    assert result["allocated_deviation"].sum() == pytest.approx(
        abs(result["total_signed_deviation"].iloc[0]), abs=1e-8
    )


def test_49_invalid_contribution_kpi_is_rejected(detected):
    scored, _ = detected
    with pytest.raises(ValueError, match="Unknown platform"):
        contribution_analysis(scored, "Aido Rover", "2025-06-01", "NOT_A_KPI")


def test_50_invalid_contribution_month_is_rejected(detected):
    scored, _ = detected
    with pytest.raises(ValueError, match="Unknown date"):
        contribution_analysis(scored, "Aido Rover", "2035-01-01", "AR01")


def test_51_report_has_required_business_fields(detected):
    scored, _ = detected
    report = build_anomaly_report(scored)
    assert {"platform", "date", "kpi_name", "deviation_magnitude",
            "primary_contributing_subsegment", "severity"}.issubset(report.columns)
    assert report["primary_contributing_subsegment"].notna().all()
    assert report["severity"].isin(["HIGH", "MEDIUM"]).all()


def test_52_report_entries_only_refer_to_flagged_deployments(detected):
    scored, predictions = detected
    report = build_anomaly_report(scored)
    flagged = predictions.loc[predictions["pred_anomaly"].eq(1),
                              ["platform", "deployment_id", "date"]].drop_duplicates()
    mapped = report.merge(flagged, on=["platform", "deployment_id", "date"], how="left",
                          indicator=True)
    assert mapped["_merge"].eq("both").all()


def test_53_report_contains_40_distinct_deployment_months(detected):
    scored, predictions = detected
    report = build_anomaly_report(scored)
    distinct = report[["platform", "deployment_id", "date"]].drop_duplicates()
    assert len(distinct) == int(predictions["pred_anomaly"].sum()) == 40


def test_54_no_report_negatives_or_null_severity(detected):
    scored, _ = detected
    report = build_anomaly_report(scored)
    assert (report["deviation_magnitude"] >= 0).all()
    assert report["severity"].notna().all()


# ---------------- 5. Entire pipeline, files and repeatability ----------------

def test_55_full_pipeline_exports_four_tables(tmp_path):
    out = tmp_path / "test_outputs"
    result = run_pipeline(DATA_FILE, CONFIG_FILE, out)
    expected = ["monthly_kpis.csv", "scored_kpis.csv",
                "deployment_predictions.csv", "anomaly_report.csv"]
    assert all((out / file).exists() for file in expected)
    assert result["true_positive"] == 38
    assert pd.read_csv(out / "monthly_kpis.csv").shape[0] == 180
    assert pd.read_csv(out / "deployment_predictions.csv").shape[0] == 360


def test_56_pipeline_is_reproducible(tmp_path):
    a = tmp_path / "first"
    b = tmp_path / "second"
    metrics_a = run_pipeline(DATA_FILE, CONFIG_FILE, a)
    metrics_b = run_pipeline(DATA_FILE, CONFIG_FILE, b)
    assert metrics_a == metrics_b
    for name in ["monthly_kpis.csv", "scored_kpis.csv",
                 "deployment_predictions.csv", "anomaly_report.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(a / name), pd.read_csv(b / name))
