"""Week 5: Aido Rover forecasting (Step 1: data preparation and baseline)."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "W02_Robot_Deployment_Dataset.csv"


def load_aido_data(path=DATA_PATH):
    """Load monthly synthetic Aido Rover deployment records in chronological order."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.loc[df["platform"].eq("Aido Rover")].copy()
    required = ["deployment_id", "date", "patrol_coverage_rate", "incident_detection_rate", "uptime_pct"]
    if df.empty or df[required].isna().any().any():
        raise ValueError("Aido Rover data is empty or required fields have missing values")
    if df.duplicated(["deployment_id", "date"]).any():
        raise ValueError("Duplicate deployment_id/date records")
    return df.sort_values(["deployment_id", "date"]).reset_index(drop=True)


def trailing_four_baseline(df):
    """Predict each monthly observation using that deployment's prior 4 months ONLY."""
    df = df.sort_values(["deployment_id", "date"]).copy()
    df["baseline_pred"] = df.groupby("deployment_id")["patrol_coverage_rate"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=4).mean()
    )
    return df


if __name__ == "__main__":
    data = load_aido_data()
    results = trailing_four_baseline(data)
    valid = results.dropna(subset=["baseline_pred"])
    print(f"Aido Rover: {len(data)} rows, {data.deployment_id.nunique()} deployments")
    print(f"Baseline: {len(valid)} one-step-ahead forecasts; no future data used")
    print(valid[["deployment_id", "date", "patrol_coverage_rate", "baseline_pred"]].head(8).to_string(index=False))


# --- Week 5 Step 2: expanding-window out-of-sample models ---
def prepare_lagged_features(df):
    """Use only previous-period values for predictors (no current target leakage)."""
    out = df.sort_values(["deployment_id", "date"]).copy()
    for column in ["patrol_coverage_rate", "incident_detection_rate", "uptime_pct"]:
        out[f"lag1_{column}"] = out.groupby("deployment_id")[column].shift(1)
    out["day_of_quarter"] = (out["date"] - out["date"].dt.to_period("Q").dt.start_time).dt.days + 1
    return out


def expanding_window_forecasts(df):
    """Predict each Aido Rover's months 5-12 using expanding historical windows.

    Model 2: Holt's linear-trend exponential smoothing per deployment.
    Model 3: pooled Ridge regression with lagged information only.
    At each date t, every fit uses strictly earlier target dates.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    data = prepare_lagged_features(df)
    data = trailing_four_baseline(data)
    feature_cols = ["lag1_patrol_coverage_rate", "lag1_incident_detection_rate",
                    "lag1_uptime_pct", "day_of_quarter"]
    all_dates = sorted(data["date"].unique())
    predictions = []
    for prediction_date in all_dates:
        evaluation_rows = data.loc[data["date"].eq(prediction_date) & data["baseline_pred"].notna()].copy()
        if evaluation_rows.empty:
            continue
        train = data.loc[data["date"].lt(prediction_date)].dropna(subset=feature_cols + ["patrol_coverage_rate"])
        if train.empty:
            continue
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge.fit(train[feature_cols], train["patrol_coverage_rate"])
        for _, row in evaluation_rows.iterrows():
            history = data.loc[(data["deployment_id"] == row["deployment_id"]) &
                               (data["date"] < prediction_date), "patrol_coverage_rate"].to_numpy()
            if len(history) < 4:
                continue
            model2 = ExponentialSmoothing(history, trend="add", seasonal=None,
                                          initialization_method="estimated").fit(optimized=True)
            holt_pred = float(model2.forecast(1)[0])
            ridge_pred = float(ridge.predict(row[feature_cols].to_frame().T)[0])
            predictions.append({
                "deployment_id": row["deployment_id"], "date": prediction_date,
                "actual": float(row["patrol_coverage_rate"]),
                "baseline_pred": float(row["baseline_pred"]),
                "holt_pred": holt_pred, "ridge_pred": ridge_pred,
                "train_latest_date": train["date"].max(),
                "ridge_training_rows": len(train),
                "intervention_flag": ridge_pred < 90.0,
            })
    return pd.DataFrame(predictions)


def model_metrics(forecasts):
    """Out-of-sample MAE and RMSE of all three models, on identical observations."""
    import numpy as np
    rows = []
    for label, column in [("Layer 1 - Baseline", "baseline_pred"),
                          ("Layer 2 - Holt", "holt_pred"),
                          ("Layer 3 - Ridge", "ridge_pred")]:
        errors = forecasts[column].to_numpy() - forecasts["actual"].to_numpy()
        rows.append({"model": label, "MAE": float(np.mean(np.abs(errors))),
                     "RMSE": float(np.sqrt(np.mean(errors**2)))})
    return pd.DataFrame(rows)


def run_step2():
    df = load_aido_data()
    forecasts = expanding_window_forecasts(df)
    print(f"Aido Rover: {len(df)} rows / {df.deployment_id.nunique()} deployments")
    print(f"Expanding-window OOS: {len(forecasts)} forecasts")
    assert (forecasts["train_latest_date"] < forecasts["date"]).all(), "Future-date training data detected"
    print("Date check: all training dates precede prediction dates")
    scores = model_metrics(forecasts)
    print("\nModel comparison:\n" + scores.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    baseline_mae = scores.iloc[0]["MAE"]
    ridge_mae = scores.iloc[2]["MAE"]
    print(f"\nRidge MAE improvement vs baseline: {100*(baseline_mae-ridge_mae)/baseline_mae:.2f}%")
    print(f"Deployments forecast below 90% (all OOS periods): {forecasts.intervention_flag.sum()}")
    out_path = DATA_PATH.parent / "W05_Forecast_Results.csv"
    forecasts.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("\n--- Step 2: Holt and Ridge OOS forecasts ---")
    run_step2()


def ridge_coefficient_stability(df, alpha=10.0):
    """Audit Ridge standardized coefficients at consecutive OOS prediction dates.

    20% is relative to preceding coefficient magnitude; near-zero prior values
    are not meaningfully comparable and are explicitly recorded as undefined.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    data = prepare_lagged_features(df)
    features = ["lag1_patrol_coverage_rate", "lag1_incident_detection_rate",
                "lag1_uptime_pct", "day_of_quarter"]
    dates = sorted(data.date.unique())[4:]
    rows=[]
    for date in dates:
        train=data.loc[data.date.lt(date)].dropna(subset=features+["patrol_coverage_rate"])
        scaler=StandardScaler().fit(train[features])
        model=Ridge(alpha=alpha).fit(scaler.transform(train[features]),train.patrol_coverage_rate)
        for name, coef in zip(features,model.coef_):
            rows.append({"prediction_date":date,"feature":name,"coefficient":float(coef),"training_latest_date":train.date.max(),"n_training":len(train)})
    out=pd.DataFrame(rows).sort_values(["feature","prediction_date"])
    out['prior_coefficient']=out.groupby('feature').coefficient.shift(1)
    out['absolute_change']=abs(out.coefficient-out.prior_coefficient)
    out['relative_change_pct']=np.where(out.prior_coefficient.abs()>1e-8,100*out.absolute_change/out.prior_coefficient.abs(),np.nan)
    out['within_20pct']=out.relative_change_pct.le(20).where(out.relative_change_pct.notna())
    return out.sort_values(['prediction_date','feature']).reset_index(drop=True)

def intervention_diagnostics(forecasts, threshold=90.0):
    """Realized event-based 2x2 classification metrics, no optimistic substitutions."""
    actual=forecasts.actual.lt(threshold)
    predicted=forecasts.ridge_pred.lt(threshold)
    tp=int((actual & predicted).sum()); fp=int((~actual & predicted).sum())
    fn=int((actual & ~predicted).sum()); tn=int((~actual & ~predicted).sum())
    return {'threshold':threshold,'TP':tp,'FP':fp,'FN':fn,'TN':tn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'specificity':tn/(tn+fp) if tn+fp else None}
