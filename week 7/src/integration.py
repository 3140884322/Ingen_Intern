"""Pure, UI-independent integration layer for the four-tab Week 7 dashboard.

Every view is driven by the original Week 3 / 5 / 6 analytical modules.
All operational and contract data is synthetic. Dashboard tests import this
module without requiring a Streamlit session.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import anomaly_detection as anomaly
from src import forecasting as forecast
from src import churn_model as churn
from src.contract_events import synthetic_contract_retention

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'W02_Robot_Deployment_Dataset.csv'
CONFIG = ROOT / 'config' / 'W03_KPI_Framework.xlsx'
CONTRACTS = ROOT / 'data' / 'W06_Fari_Contract_Metadata_SYNTHETIC_ASSUMPTIONS.csv'
EVENTS = ROOT / 'data' / 'W06_Synthetic_Contract_Events_SCENARIO.csv'
FORECAST_OOS = ROOT / 'data' / 'W05_Forecast_Results.csv'
RISK = ROOT / 'outputs' / 'W06_Revenue_Risk_All_Deployments.csv'


def anomaly_bundle(raw=None, config=None):
    """Run the exact Week 3 anomaly pipeline, no duplicated alert calculation."""
    raw = anomaly.load_data(DATA) if raw is None else raw
    config = anomaly.load_kpi_config(CONFIG) if config is None else config
    readings, monthly = anomaly.calculate_kpis(raw, config)
    scored, predictions = anomaly.detect_anomalies(readings)
    alerts = anomaly.build_anomaly_report(scored)
    return {'raw':raw, 'config':config, 'readings':readings, 'monthly':monthly,
            'scored':scored,'predictions':predictions,'alerts':alerts}


def alert_counts(predictions):
    """Counts of unique flagged platform/deployment/month records."""
    if predictions.empty:
        return pd.DataFrame(columns=['platform','anomaly_count'])
    required={'platform','deployment_id','date','pred_anomaly'}
    if not required.issubset(predictions): raise ValueError('Invalid anomaly predictions')
    p=predictions.loc[predictions.pred_anomaly.eq(1)]
    return (p.groupby('platform',as_index=False).size().rename(columns={'size':'anomaly_count'})
            if not p.empty else pd.DataFrame(columns=['platform','anomaly_count']))


def anomaly_dashboard_flags(bundle):
    """Same event identity as the original anomaly module output."""
    p=bundle['predictions']
    return p[['platform','deployment_id','date','pred_anomaly']].sort_values(
        ['platform','deployment_id','date']).reset_index(drop=True)


def anomaly_contributions(bundle, platform, date, kpi_id):
    return anomaly.contribution_analysis(bundle['scored'],platform,pd.Timestamp(date),kpi_id)


def forecast_bundle(*, include_backtest=False):
    """Use Week 5 OOS historical results; current next-month model fitted through as-of date."""
    aido=forecast.load_aido_data(DATA)
    oos=pd.read_csv(FORECAST_OOS,parse_dates=['date','train_latest_date'])
    if include_backtest:
        oos=forecast.expanding_window_forecasts(aido)
    if not (oos.train_latest_date < oos.date).all():
        raise ValueError('Forecast training includes future target dates')
    return {'aido':aido, 'oos':oos, 'metrics':forecast.model_metrics(oos),
            'next':next_month_forecast(aido,oos)}


def next_month_forecast(aido, oos, threshold=90.0):
    """Fit pooled Ridge using historical lag rows through as-of; forecast next monthly point.

    Approximate 95% empirical forecast interval based on historical OOS
    actual-minus-Ridge residual quantiles (2.5%, 97.5%), clipped to [0,100].
    This is NOT a formally calibrated conditional interval.
    """
    if aido.empty or 'patrol_coverage_rate' not in aido: raise ValueError('Missing Aido observations')
    lagged=forecast.prepare_lagged_features(aido)
    features=['lag1_patrol_coverage_rate','lag1_incident_detection_rate','lag1_uptime_pct','day_of_quarter']
    train=lagged.dropna(subset=features+['patrol_coverage_rate'])
    if train.empty: raise ValueError('Insufficient historical data')
    cutoff=aido.date.max()
    if train.date.max()>cutoff:raise ValueError('Future train records')
    fitted=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    fitted.fit(train[features],train.patrol_coverage_rate)
    rows=[]
    for _,g in aido.groupby('deployment_id'):
        latest=g.sort_values('date').iloc[-1]
        target=pd.Timestamp(latest.date)+pd.DateOffset(months=1)
        quarter=target.to_period('Q').start_time
        x=pd.DataFrame([{
            'lag1_patrol_coverage_rate':latest.patrol_coverage_rate,
            'lag1_incident_detection_rate':latest.incident_detection_rate,
            'lag1_uptime_pct':latest.uptime_pct,
            'day_of_quarter':int((target-quarter).days+1)}])
        prediction=float(fitted.predict(x)[0])
        rows.append({'deployment_id':latest.deployment_id,'last_observed_date':latest.date,
                     'forecast_date':target,'patrol_coverage_forecast':prediction,
                     'service_threshold':threshold,'intervention_flag':prediction<threshold,
                     'train_latest_date':train.date.max()})
    out=pd.DataFrame(rows)
    if not (out.train_latest_date < out.forecast_date).all():
        raise ValueError('Forecast as-of leak')
    errors=oos.ridge_pred-oos.actual
    residuals=(-errors).to_numpy()
    lower_offset,upper_offset=np.quantile(residuals,[.025,.975])
    out['illustrative_lower']=np.minimum(out.patrol_coverage_forecast,np.maximum(0,out.patrol_coverage_forecast+lower_offset))
    out['illustrative_upper']=np.maximum(out.patrol_coverage_forecast,np.minimum(100,out.patrol_coverage_forecast+upper_offset))
    return out.sort_values('deployment_id').reset_index(drop=True)


def churn_bundle(as_of='2025-09-01', k=10):
    """Reuse Week 6 validated model outputs; risk scores are exploratory IN-SAMPLE.

    The original synthetic contract metadata and separate simulated retention
    scenario are intentionally kept distinct from observed deployment activity.
    """
    obs=pd.read_csv(DATA)
    contracts=pd.read_csv(CONTRACTS)
    events=pd.read_csv(EVENTS)
    lifecycle=churn.lifecycle_as_of(obs,contracts,as_of)
    revenue=pd.read_csv(RISK)
    ranked=churn.select_top_k(revenue,k)
    cohort=churn.contract_cohort_summary(contracts)
    retention=synthetic_contract_retention(events)
    return {'lifecycle':lifecycle,'revenue':revenue,'top_k':ranked,
            'cohorts':cohort,'retention':retention,
            'renewal_observability':churn.retention_observability(contracts)}


def renewal_trend(raw):
    """Observed annual/static labels have no timestamped dates; do not invent a time trend."""
    subset=raw.loc[raw.platform.eq('Fari')]
    unique=subset[['deployment_id','renewal_status']].drop_duplicates('deployment_id')
    if unique.empty: return {'renewals':0,'total':0,'rate':None,'trend_available':False}
    renewals=int(unique.renewal_status.astype(str).str.lower().eq('renewal').sum())
    return {'renewals':renewals,'total':len(unique),'rate':renewals/len(unique),
            'trend_available':False}


def executive_summary(anomalies, forecasts, churn_data):
    counts=alert_counts(anomalies['predictions'])
    n=forecasts['next']
    l=churn_data['lifecycle']
    return {'anomaly_counts':counts,'latest_forecast_date':n.forecast_date.max(),
            'patrol_forecast_mean':float(n.patrol_coverage_forecast.mean()),
            'forecast_below_90_count':int(n.intervention_flag.sum()),
            'fari_at_risk_count':int(l.lifecycle_stage.eq('At-Risk').sum()),
            'fari_churned_count':int(l.lifecycle_stage.eq('Churned').sum()),
            'top_k_count':len(churn_data['top_k'])}


def quality_checks(bundle):
    """Per-platform KPIs must ignore not-applicable platform columns."""
    out=[]
    for platform,mapping in anomaly.PLATFORM_METRICS.items():
        sub=bundle['raw'].loc[bundle['raw'].platform.eq(platform)]
        for metric,col in mapping.items():
            if col=='estimated_daily_engagement_min':
                vals=pd.to_numeric(sub.avg_session_duration_min,errors='coerce')*pd.to_numeric(sub.daily_engagement_sessions,errors='coerce')
            else: vals=pd.to_numeric(sub[col],errors='coerce')
            out.append({'platform':platform,'kpi_id':metric,'missing_pct':float(vals.isna().mean()*100) if len(vals) else 100.0})
    return pd.DataFrame(out)
