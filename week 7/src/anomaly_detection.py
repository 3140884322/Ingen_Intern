"""Week 3: Synthetic robot KPI & anomaly detection pipeline.

Uses Week 2's SYNTHETIC observations and editable W03 KPI thresholds.
Labels (anomaly_flag) are NEVER used to make predictions; they are for evaluation.
This is a retrospective educational demonstration, not a deployed monitoring system.

Quick start, from the project folder:
    python -m src.anomaly_detection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Data contains deployment-month observations, not raw incident/session counts.
PLATFORM_METRICS = {
    "Aido Rover": {
        "AR01": "patrol_coverage_rate", "AR02": "incident_detection_rate",
        "AR03": "battery_consumption_per_cycle", "AR04": "uptime_pct",
        "AR05": "shift_completion_rate",
    },
    "Sentinel Prime AI": {
        "SP01": "patrol_coverage_rate", "SP02": "incident_detection_rate",
        "SP03": "battery_consumption_per_cycle", "SP04": "uptime_pct",
        "SP05": "shift_completion_rate",
    },
    "Fari": {
        "FA01": "avg_session_duration_min", "FA02": "daily_engagement_sessions",
        "FA03": "health_alert_response_time_min", "FA04": "caregiver_app_interactions",
        "FA05": "estimated_daily_engagement_min",
    },
}

REQUIRED_COLUMNS = {"data_label", "deployment_id", "platform", "date", "anomaly_flag"}
REQUIRED_COLUMNS.update(
    f for metrics in PLATFORM_METRICS.values() for f in metrics.values()
    if f != "estimated_daily_engagement_min"
)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load Week 2 CSV and validate the structure before analysis."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if not df["data_label"].eq("SYNTHETIC").all():
        raise ValueError("Only clearly labeled SYNTHETIC data are supported")
    if df["deployment_id"].isna().any():
        raise ValueError("deployment_id contains missing values")
    if df["platform"].isna().any() or not set(df["platform"]).issubset(PLATFORM_METRICS):
        raise ValueError("Unrecognized or missing platform")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any():
        raise ValueError("date contains missing values")
    if df.duplicated(["deployment_id", "date"]).any():
        raise ValueError("Duplicate (deployment_id, date) pairs")
    if not df["anomaly_flag"].isin([0, 1]).all():
        raise ValueError("anomaly_flag must be either 0 or 1")
    # Fari renewal_status is the Week 6 outcome and must not be a detection feature.
    df["estimated_daily_engagement_min"] = (
        pd.to_numeric(df["avg_session_duration_min"], errors="coerce")
        * pd.to_numeric(df["daily_engagement_sessions"], errors="coerce")
    )
    for platform, mapping in PLATFORM_METRICS.items():
        mask = df["platform"].eq(platform)
        for col in mapping.values():
            if df.loc[mask, col].isna().any():
                raise ValueError(f"Missing applicable values: {platform} / {col}")
    return df.sort_values(["platform", "deployment_id", "date"]).reset_index(drop=True)


def load_kpi_config(xlsx_path: str | Path) -> pd.DataFrame:
    """Read the same editable configuration displayed in W03_KPI_Framework.xlsx.

    Excel is *read only*. Edit thresholds in Threshold_Config to experiment.
    pandas.read_excel requires openpyxl (listed in requirements.txt).
    """
    catalog = pd.read_excel(xlsx_path, sheet_name="KPI_Framework", header=3)
    thresholds = pd.read_excel(xlsx_path, sheet_name="Threshold_Config", header=3)
    catalog = catalog.dropna(subset=["KPI ID"])
    thresholds = thresholds.dropna(subset=["KPI ID"])
    # The final note under the threshold table is not a KPI row.
    thresholds = thresholds[thresholds["KPI ID"].astype(str).str.match(r"^(AR|SP|FA)\d{2}$")]
    needed = [
        "KPI ID", "Alert operator", "Alert threshold", "Z-score |z| cutoff",
        "Rolling baseline (months)", "Tail share (each side)", "Tukey IQR multiplier",
    ]
    if any(col not in thresholds for col in needed):
        raise ValueError("Threshold_Config sheet has unexpected/missing column headers")
    merged = catalog[["KPI ID", "Platform", "KPI / metric name", "Better when"]].merge(
        thresholds[needed], on="KPI ID", validate="one_to_one"
    )
    expected = {k for m in PLATFORM_METRICS.values() for k in m}
    if set(merged["KPI ID"]) != expected or len(merged) != 15:
        raise ValueError("Expected exactly 15 KPIs in KPI_Framework / Threshold_Config")
    if not merged["Alert operator"].isin(["<", ">"]).all():
        raise ValueError("Alert operator must be '<' or '>'")
    for col in ["Alert threshold", "Z-score |z| cutoff", "Rolling baseline (months)",
                "Tail share (each side)", "Tukey IQR multiplier"]:
        merged[col] = pd.to_numeric(merged[col], errors="raise")
    if (merged["Z-score |z| cutoff"] <= 0).any():
        raise ValueError("Z-score cutoff must be positive")
    if (merged["Rolling baseline (months)"] < 3).any():
        raise ValueError("Rolling history window must contain at least 3 months")
    if (~merged["Tail share (each side)"].between(0.001, 0.49)).any():
        raise ValueError("Tail share must be between 0 and 0.5")
    return merged.rename(columns={
        "KPI ID": "kpi_id", "Platform": "platform", "KPI / metric name": "kpi_name",
        "Better when": "better_when", "Alert operator": "alert_operator",
        "Alert threshold": "alert_threshold", "Z-score |z| cutoff": "z_cutoff",
        "Rolling baseline (months)": "baseline_months",
        "Tail share (each side)": "tail_share", "Tukey IQR multiplier": "iqr_multiplier",
    }).reset_index(drop=True)


def calculate_kpis(df: pd.DataFrame, config: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deployment-level KPI readings and monthly platform-level means.

    Values are pre-aggregated synthetic readings, so equal-weight means are used;
    raw numerator/denominator data are not present to make weighted rates.
    """
    parts = []
    for platform, metrics in PLATFORM_METRICS.items():
        platform_df = df.loc[df["platform"].eq(platform)]
        for kpi_id, source in metrics.items():
            chunk = platform_df[["platform", "deployment_id", "date", source]].copy()
            chunk = chunk.rename(columns={source: "value"})
            chunk["kpi_id"] = kpi_id
            parts.append(chunk)
    readings = pd.concat(parts, ignore_index=True)
    readings = readings.merge(config, on=["platform", "kpi_id"], validate="many_to_one")
    readings = readings.sort_values(["platform", "kpi_id", "deployment_id", "date"])

    groups = readings.groupby(["platform", "kpi_id", "deployment_id"])["value"]
    readings["previous_value"] = groups.shift(1)
    readings["mom_change"] = readings["value"] - readings["previous_value"]
    readings["mom_change_pct"] = 100 * readings["mom_change"] / readings["previous_value"].replace(0, np.nan)
    readings["rolling4_mean"] = groups.transform(lambda x: x.rolling(4, min_periods=1).mean())

    monthly = readings.groupby(["platform", "kpi_id", "kpi_name", "date"], as_index=False).agg(
        mean_value=("value", "mean"), deployment_count=("deployment_id", "nunique")
    )
    monthly = monthly.sort_values(["platform", "kpi_id", "date"])
    g = monthly.groupby(["platform", "kpi_id"])["mean_value"]
    monthly["mom_change"] = g.diff()
    monthly["rolling4_mean"] = g.transform(lambda x: x.rolling(4, min_periods=1).mean())
    monthly["rolling4_change"] = monthly["rolling4_mean"] - g.transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    return readings.reset_index(drop=True), monthly.reset_index(drop=True)


def detect_anomalies(readings: pd.DataFrame, *, z_cutoff_override: float | None = None,
                     fari_duration_drop: float = 0.08,
                     fari_sessions_drop: float = 0.12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect unusual records without accessing their planted anomaly labels.

    Operational alerts: from workbook. Other diagnostics: rolling prior z-score,
    baseline 10% tails and true Tukey IQR fences (not the same algorithm).
    First three months supply a cold-start reference, since 12 months is not enough
    for 12 strictly *prior* months of observations for any individual deployment.
    """
    k = readings.copy().sort_values(["platform", "kpi_id", "deployment_id", "date"])
    if z_cutoff_override is not None and z_cutoff_override <= 0:
        raise ValueError("z_cutoff_override must be positive")
    if z_cutoff_override is not None:
        k["z_cutoff"] = z_cutoff_override

    # Pre-period historical baseline and standard deviation; never uses current row.
    gg = k.groupby(["platform", "kpi_id", "deployment_id"], sort=False)
    k["history_mean"] = np.nan
    k["history_std"] = np.nan
    for _, ix in gg.groups.items():
        idx = list(ix)
        vals = k.loc[idx, "value"]
        window = int(k.loc[idx[0], "baseline_months"])
        previous = vals.shift(1)
        k.loc[idx, "history_mean"] = previous.rolling(window, min_periods=3).mean().values
        k.loc[idx, "history_std"] = previous.rolling(window, min_periods=3).std(ddof=1).values

    # All early-period baselines are the same for a given platform and KPI.
    # Using the first three months creates a stable reference for retrospective
    # tests but does NOT make this a live/no-look-ahead forecasting evaluator.
    k["reference_mean"] = np.nan
    k["reference_std"] = np.nan
    k["q_low"] = np.nan
    k["q_high"] = np.nan
    k["iqr_lower"] = np.nan
    k["iqr_upper"] = np.nan
    for (_, _), ix in k.groupby(["platform", "kpi_id"]).groups.items():
        idx = list(ix)
        s = k.loc[idx]
        early_months = np.sort(s["date"].unique())[:3]
        early = s[s["date"].isin(early_months)]["value"]
        reference = float(early.median())
        stdev = max(float(early.std(ddof=1)), 1e-6)
        tail = float(s["tail_share"].iloc[0])
        q1, q3 = early.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        mult = float(s["iqr_multiplier"].iloc[0])
        k.loc[idx, "reference_mean"] = reference
        k.loc[idx, "reference_std"] = stdev
        k.loc[idx, "q_low"] = float(early.quantile(tail))
        k.loc[idx, "q_high"] = float(early.quantile(1-tail))
        k.loc[idx, "iqr_lower"] = q1 - mult * iqr
        k.loc[idx, "iqr_upper"] = q3 + mult * iqr

    # Cold start: use pooled baseline for z-score when fewer than 3 prior months.
    # After 3 months, historical means become deployment-specific. A pooled
    # standard-deviation floor prevents hypersensitivity to short sequences.
    mean_used = k["history_mean"].fillna(k["reference_mean"])
    std_used = pd.concat([k["history_std"], k["reference_std"]], axis=1).max(axis=1)
    k["z_score"] = (k["value"] - mean_used) / std_used
    k["z_flag"] = k["z_score"].abs() > k["z_cutoff"]
    k["tail_flag"] = k["value"].lt(k["q_low"]) | k["value"].gt(k["q_high"])
    k["iqr_flag"] = k["value"].lt(k["iqr_lower"]) | k["value"].gt(k["iqr_upper"])
    k["threshold_flag"] = np.where(k["alert_operator"].eq("<"),
                                   k["value"].lt(k["alert_threshold"]),
                                   k["value"].gt(k["alert_threshold"]))
    # For operations, only deviations in the *unhealthy* direction count as alerts.
    k["adverse_z_flag"] = k["z_flag"] & np.where(k["better_when"].eq("Higher"),
                                                  k["z_score"].lt(0), k["z_score"].gt(0))
    k["adverse_tail_flag"] = k["tail_flag"] & np.where(k["better_when"].eq("Higher"),
                                                        k["value"].lt(k["q_low"]),
                                                        k["value"].gt(k["q_high"]))

    # Combine KPI-level signals into one deployment-month decision. FA05 is
    # calculated from FA01×FA02, so don't treat it as independent evidence.
    k["independent_kpi"] = ~k["kpi_id"].eq("FA05")
    k["confirmed_statistical"] = k["adverse_z_flag"] & k["adverse_tail_flag"] & k["independent_kpi"]
    grouped = k.groupby(["platform", "deployment_id", "date"], as_index=False).agg(
        any_threshold=("threshold_flag", "any"),
        strong_kpi_count=("confirmed_statistical", "sum"),
    )

    # Fari trend: compare the month with the SAME deployment's first 3 months.
    # This uses only engagement measures; never the renewal_status outcome.
    f = k.loc[k["platform"].eq("Fari") & k["kpi_id"].isin(["FA01", "FA02"]),
              ["deployment_id", "date", "kpi_id", "value"]].copy()
    f = f.sort_values(["deployment_id", "kpi_id", "date"])
    baseline = f.groupby(["deployment_id", "kpi_id"])["value"].transform(
        lambda x: x.iloc[:3].mean()
    )
    f["decline"] = (baseline - f["value"]) / baseline
    f = f.pivot(index=["deployment_id", "date"], columns="kpi_id", values="decline").reset_index()
    f["fari_trend_flag"] = f["FA01"].ge(fari_duration_drop) & f["FA02"].ge(fari_sessions_drop)
    # A three-month onboarding baseline; earliest operational trend alert = month 4.
    f["month_num"] = f.groupby("deployment_id").cumcount() + 1
    f["fari_trend_flag"] &= f["month_num"].ge(4)
    grouped = grouped.merge(f[["deployment_id", "date", "fari_trend_flag"]],
                            on=["deployment_id", "date"], how="left")
    grouped["fari_trend_flag"] = grouped["fari_trend_flag"].map(
        lambda x: False if pd.isna(x) else bool(x)
    ).astype(bool)
    grouped["pred_anomaly"] = (
        grouped["any_threshold"] | grouped["strong_kpi_count"].ge(2) | grouped["fari_trend_flag"]
    ).astype(int)
    k = k.merge(grouped, on=["platform", "deployment_id", "date"], validate="many_to_one")
    return k, grouped


def contribution_analysis(readings: pd.DataFrame, platform: str, date,
                          kpi_id: str) -> pd.DataFrame:
    """Rank deployment contributions to platform-month KPI deviation.

    Signed contributions sum exactly to the observed platform mean minus the
    first-three-month reference. Nonnegative 'allocated_deviation' distributes
    the ABSOLUTE net deviation across same-direction contributors. This is
    descriptive attribution, NOT evidence of root cause or causality.
    """
    sub = readings.loc[
        readings["platform"].eq(platform) & readings["kpi_id"].eq(kpi_id)
    ].sort_values(["deployment_id", "date"]).copy()
    if sub.empty:
        raise ValueError(f"Unknown platform / KPI: {platform} / {kpi_id}")
    month = sub.loc[sub["date"].eq(pd.Timestamp(date)), ["deployment_id", "value"]].copy()
    if month.empty:
        raise ValueError("Unknown date")
    reference = sub.groupby("deployment_id")["value"].transform(lambda x: x.iloc[:3].mean())
    sub["ref"] = reference
    month = month.merge(sub[["deployment_id", "date", "ref"]].drop_duplicates("deployment_id"),
                        on="deployment_id", validate="one_to_one")
    n = len(month)
    month["signed_contribution"] = (month["value"] - month["ref"]) / n
    total_deviation = float(month["signed_contribution"].sum())
    adverse = month["signed_contribution"] * (1 if total_deviation >= 0 else -1)
    adverse = adverse.clip(lower=0)
    weight = adverse / adverse.sum() if adverse.sum() > 0 else np.zeros(n)
    month["allocated_deviation"] = abs(total_deviation) * weight
    month["share_pct"] = 100 * weight
    month["total_signed_deviation"] = total_deviation
    month["platform"] = platform
    month["date"] = pd.Timestamp(date)
    month["kpi_id"] = kpi_id
    return month.sort_values(["allocated_deviation", "deployment_id"], ascending=[False, True]).reset_index(drop=True)


def build_anomaly_report(kpi_readings: pd.DataFrame) -> pd.DataFrame:
    """Build a structured report of flagged deployment-month observations."""
    k = kpi_readings.copy()
    # Show the most informative flagged KPIs (plus trend-only Fari drivers).
    k["triggered_kpi"] = k["threshold_flag"] | k["adverse_z_flag"] | (
        k["fari_trend_flag"] & k["kpi_id"].isin(["FA01", "FA02"])
    )
    filtered = k.loc[k["pred_anomaly"].eq(1) & k["triggered_kpi"]].copy()
    # Rare case: a deployment was confirmed by 2 statistical signals but the
    # selection above missed both; include its most deviant KPI instead.
    filtered["severity"] = np.where(filtered["threshold_flag"] & filtered["z_flag"],
                                    "HIGH", "MEDIUM")
    filtered["deviation_magnitude"] = (filtered["value"] - filtered["reference_mean"]).abs().round(3)
    # Contribution lookup only per unique flagged platform/date/KPI.
    raw_long = kpi_readings
    rank = {}
    for p, d, metric in filtered[["platform", "date", "kpi_id"]].drop_duplicates().itertuples(index=False, name=None):
        attribution = contribution_analysis(raw_long, p, d, metric)
        rank[(p, d, metric)] = str(attribution.iloc[0]["deployment_id"])
    filtered["primary_contributing_subsegment"] = [
        rank[(r.platform, r.date, r.kpi_id)] for r in filtered.itertuples()
    ]
    cols = ["platform", "deployment_id", "date", "kpi_id", "kpi_name",
            "value", "deviation_magnitude", "primary_contributing_subsegment",
            "severity", "threshold_flag", "z_flag", "tail_flag", "iqr_flag", "fari_trend_flag"]
    return filtered[cols].sort_values(["date", "platform", "deployment_id", "kpi_id"]).reset_index(drop=True)


def evaluate_against_labels(predictions: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Evaluate *after* generating predictions; labels are not model inputs."""
    merged = predictions[["deployment_id", "date", "pred_anomaly"]].merge(
        data[["deployment_id", "date", "anomaly_flag"]], on=["deployment_id", "date"],
        validate="one_to_one"
    )
    if len(merged) != len(data):
        raise ValueError("Predictions missing input records")
    actual, predicted = merged["anomaly_flag"].eq(1), merged["pred_anomaly"].eq(1)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    return {"rows": len(merged), "planted_anomalies": int(actual.sum()),
            "predicted_anomalies": int(predicted.sum()),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def run_pipeline(data_path: str | Path, config_path: str | Path,
                 out_dir: str | Path, *, z_cutoff_override: float | None = None) -> dict:
    """Compute KPIs, detect anomalies, write outputs, and evaluate labels."""
    data = load_data(data_path)
    config = load_kpi_config(config_path)
    kpis, monthly = calculate_kpis(data, config)
    scored, predictions = detect_anomalies(kpis, z_cutoff_override=z_cutoff_override)
    report = build_anomaly_report(scored)
    results = evaluate_against_labels(predictions, data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / "monthly_kpis.csv", index=False)
    scored.to_csv(out / "scored_kpis.csv", index=False)
    predictions.to_csv(out / "deployment_predictions.csv", index=False)
    report.to_csv(out / "anomaly_report.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 3 robot anomaly detection (SYNTHETIC data)")
    parser.add_argument("--data", default="data/W02_Robot_Deployment_Dataset.csv")
    parser.add_argument("--config", default="config/W03_KPI_Framework.xlsx")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--z-cutoff", type=float, default=None)
    args = parser.parse_args()
    stats = run_pipeline(args.data, args.config, args.out, z_cutoff_override=args.z_cutoff)
    print("SYNTHETIC robot anomaly pipeline")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Outputs saved to: {args.out}")


if __name__ == "__main__":
    main()
