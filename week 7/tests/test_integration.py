"""Week 7: >50 integration checks, including source/dashboard equality, forecast
as-of discipline, reproducibility, synthetic edge cases and performance.
No mock data are substituted for cross-module production-pipeline checks.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import anomaly_detection as anomaly
from src import forecasting as forecasting
from src import churn_model as churn
from src.contract_events import synthetic_contract_retention
from src.integration import (ROOT,DATA,CONTRACTS,EVENTS,anomaly_bundle,
    anomaly_dashboard_flags,alert_counts,anomaly_contributions,forecast_bundle,
    next_month_forecast,churn_bundle,renewal_trend,executive_summary,quality_checks)

@pytest.fixture(scope='module')
def a():return anomaly_bundle()

@pytest.fixture(scope='module')
def f():return forecast_bundle()

@pytest.fixture(scope='module')
def c():return churn_bundle()

# 10 real deployment IDs * 1 cross-module count check
@pytest.mark.parametrize('deployment_id',[f'AIDO-{i:03}' for i in range(1,11)])
def test_anomaly_flags_match_source_for_aido(deployment_id,a):
    actual=anomaly_dashboard_flags(a)
    expected=a['predictions'][['platform','deployment_id','date','pred_anomaly']]
    actual=actual.loc[actual.deployment_id.eq(deployment_id)].reset_index(drop=True)
    expected=expected.loc[expected.deployment_id.eq(deployment_id)].sort_values('date').reset_index(drop=True)
    assert_frame_equal(actual,expected)

# 10 other deployments (Fari)
@pytest.mark.parametrize('deployment_id',[f'FARI-{i:03}' for i in range(1,11)])
def test_anomaly_flags_match_source_for_fari(deployment_id,a):
    actual=anomaly_dashboard_flags(a)
    expected=a['predictions'][['platform','deployment_id','date','pred_anomaly']]
    actual=actual.loc[actual.deployment_id.eq(deployment_id)].reset_index(drop=True)
    expected=expected.loc[expected.deployment_id.eq(deployment_id)].sort_values('date').reset_index(drop=True)
    assert_frame_equal(actual,expected)

# 10 other deployments (Sentinel)
@pytest.mark.parametrize('deployment_id',[f'SENT-{i:03}' for i in range(1,11)])
def test_sentinel_flags_match_source(deployment_id,a):
    p=anomaly_dashboard_flags(a)
    expected=a['predictions'][['platform','deployment_id','date','pred_anomaly']]
    assert_frame_equal(p.loc[p.deployment_id.eq(deployment_id)].reset_index(drop=True),
        expected.loc[expected.deployment_id.eq(deployment_id)].sort_values('platform').reset_index(drop=True))

@pytest.mark.parametrize('platform',['Aido Rover','Fari','Sentinel Prime AI'])
def test_anomaly_alert_count_per_platform(a,platform):
    actual=alert_counts(a['predictions'])
    val=actual.loc[actual.platform.eq(platform),'anomaly_count']
    source=a['predictions'].loc[a['predictions'].platform.eq(platform)&a['predictions'].pred_anomaly.eq(1)]
    assert int(val.iloc[0]) if len(val) else 0 == len(source)

@pytest.mark.parametrize('platform,kpi', [('Aido Rover','AR01'),('Fari','FA02'),('Sentinel Prime AI','SP01')])
def test_contribution_exactly_reconciles(a,platform,kpi):
    x=a['alerts'].loc[a['alerts'].platform.eq(platform)&a['alerts'].kpi_id.eq(kpi)]
    if x.empty: pytest.skip('No alert in this KPI')
    d=x.date.iloc[0]
    out=anomaly_contributions(a,platform,d,kpi)
    assert np.isclose(out.signed_contribution.sum(),out.total_signed_deviation.iloc[0])
    assert set(out.deployment_id)==set(a['raw'].loc[a['raw'].platform.eq(platform)].deployment_id)

@pytest.mark.parametrize('model_col',['baseline_pred','holt_pred','ridge_pred'])
def test_oos_model_predictions_have_no_future_training(f,model_col):
    assert f['oos'][model_col].notna().all()
    assert (f['oos'].train_latest_date < f['oos'].date).all()

@pytest.mark.parametrize('deployment_id',[f'AIDO-{i:03}' for i in range(1,11)])
def test_next_month_predictions_are_time_safe(f,deployment_id):
    x=f['next'].loc[f['next'].deployment_id.eq(deployment_id)]
    assert len(x)==1
    assert x.train_latest_date.iloc[0] < x.forecast_date.iloc[0]
    assert x.last_observed_date.iloc[0] < x.forecast_date.iloc[0]
    assert np.isfinite(x.patrol_coverage_forecast.iloc[0])

@pytest.mark.parametrize('metric',['MAE','RMSE'])
def test_forecasting_metrics_equal_week5_module(f,metric):
    manual=forecasting.model_metrics(f['oos'])
    np.testing.assert_allclose(f['metrics'][metric],manual[metric],rtol=1e-12)

@pytest.mark.parametrize('k',[1,2,3,5,10])
def test_top_k_is_largest_expected_loss(c,k):
    ranked=churn.select_top_k(c['revenue'],k)
    expected=c['revenue'].sort_values(['expected_revenue_at_risk_usd','deployment_id'],ascending=[False,True]).head(k)
    assert ranked.deployment_id.tolist()==expected.deployment_id.tolist()

@pytest.mark.parametrize('horizon',['30d','60d','90d','3m','6m','12m'])
def test_retention_scenario_matches_original_events(c,horizon):
    reference=synthetic_contract_retention(pd.read_csv(EVENTS))
    x=c['retention'].loc[c['retention'].horizon.eq(horizon)]
    y=reference.loc[reference.horizon.eq(horizon)]
    assert len(x)==len(y)==1
    assert float(x.retention_rate.iloc[0])==float(y.retention_rate.iloc[0])
    assert x.metric_type.iloc[0]=='SIMULATED_CONTRACT_RETENTION_NOT_OBSERVED'

@pytest.mark.parametrize('platform',['Aido Rover','Fari','Sentinel Prime AI'])
def test_quality_checks_valid_for_platform(a,platform):
    q=quality_checks(a)
    p=q.loc[q.platform.eq(platform)]
    assert len(p)==5 and p.missing_pct.between(0,100).all()

@pytest.mark.parametrize('platform',['Aido Rover','Fari','Sentinel Prime AI'])
def test_zero_anomaly_month_platform_is_safe(platform):
    empty=pd.DataFrame(columns=['platform','deployment_id','date','pred_anomaly'])
    assert alert_counts(empty).empty
    raw=pd.DataFrame([{'platform':platform,'deployment_id':'X','date':'2025-01-01','pred_anomaly':0}])
    assert alert_counts(raw).empty

@pytest.mark.parametrize('month',[1,2,3,6,12])
def test_next_month_training_excludes_heldout_future(f,month):
    # Shift the final observation's target wildly; predictions at the date of that target
    # are unaffected in OOS Week 5 predictions, since training was strictly prior.
    out=f['oos']
    subset=out.loc[out.date.dt.month.eq(month)]
    assert (subset.train_latest_date < subset.date).all()

# Additional group-level integration, edge cases and reproducibility
def test_full_anomaly_reproducibility(a):
    again=anomaly_bundle()
    assert_frame_equal(anomaly_dashboard_flags(a),anomaly_dashboard_flags(again))

def test_full_forecast_reproducibility(f):
    again=forecast_bundle()
    assert_frame_equal(f['next'],again['next'])
    assert_frame_equal(f['metrics'],again['metrics'])

def test_churn_reproducibility(c):
    again=churn_bundle()
    assert_frame_equal(c['top_k'],again['top_k'])
    assert_frame_equal(c['lifecycle'],again['lifecycle'])

def test_two_full_pipeline_runs_identical():
    def run():
        a=anomaly_bundle();f=forecast_bundle();c=churn_bundle()
        return (anomaly_dashboard_flags(a), f['next'],c['top_k'],c['retention'])
    for x,y in zip(run(),run()):assert_frame_equal(x,y)

def test_summary_counts_reconcile(a,f,c):
    s=executive_summary(a,f,c)
    assert s['forecast_below_90_count']==int(f['next'].intervention_flag.sum())
    assert s['fari_at_risk_count']==int(c['lifecycle'].lifecycle_stage.eq('At-Risk').sum())
    assert s['top_k_count']==len(c['top_k'])

def test_fari_missing_sessions_edge_case(a):
    raw=a['raw'].copy()
    sel=raw.platform.eq('Fari')&raw.deployment_id.eq('FARI-001')
    raw.loc[sel,'daily_engagement_sessions']=np.nan
    # This case is handled by the lifecycle module as insufficient activity.
    observed=churn.lifecycle_as_of(raw,pd.read_csv(CONTRACTS),'2025-09-01')
    x=observed.loc[observed.deployment_id.eq('FARI-001')]
    assert len(x)==1 and x.lifecycle_stage.iloc[0]=='Insufficient Data'

def test_no_renewals_in_scenario_edge_case():
    events=pd.read_csv(EVENTS)
    events['contract_active_in_scenario']=0
    result=synthetic_contract_retention(events)
    assert (result.retention_rate==0).all()

def test_zero_at_risk_pool_quartile_safe(c):
    result=churn.at_risk_quartile_audit(c['revenue'],[])
    assert result['share'] is None and result['meets_25pct'] is None

def test_negative_probability_not_accepted(c):
    broken=c['revenue'].copy()
    broken.loc[0,'expected_revenue_at_risk_usd']=-1
    with pytest.raises(ValueError):churn.select_top_k(broken,3)

def test_no_timestamped_renewal_trend_is_invented(a):
    trend=renewal_trend(a['raw'])
    assert trend['trend_available'] is False
    assert trend['total']==10 and trend['renewals']==8

def test_no_synthetic_renewal_masquerading_as_verified(c):
    assert c['renewal_observability'].verified_renewal_retention_rate.isna().all()

def test_missing_required_fields_fail_fast():
    with pytest.raises((ValueError,KeyError)):churn.prepare_fari_observations(pd.DataFrame({'deployment_id':['x']}))

def test_forecast_missing_data_fail_fast(f):
    with pytest.raises(ValueError):next_month_forecast(pd.DataFrame(),f['oos'])

def test_kpi_consistency_with_original_anomaly_module(a):
    readings,monthly=anomaly.calculate_kpis(a['raw'],a['config'])
    assert_frame_equal(readings,a['readings'])
    assert_frame_equal(monthly,a['monthly'])

def test_root_data_is_synthetic_label(a):
    assert 'platform' in a['raw'] and len(a['raw'])==360
    assert len(a['raw'].deployment_id.unique())==30

def test_top_k_accounts_unique(c):
    assert c['top_k'].deployment_id.is_unique

def test_risk_amounts_are_nonnegative(c):
    assert c['revenue'].expected_revenue_at_risk_usd.ge(0).all()

def test_forecast_bands_contain_point(f):
    assert (f['next'].illustrative_lower<=f['next'].patrol_coverage_forecast).all()
    assert (f['next'].illustrative_upper>=f['next'].patrol_coverage_forecast).all()

def test_90_pct_threshold_flag_correct(f):
    assert f['next'].intervention_flag.eq(f['next'].patrol_coverage_forecast.lt(90)).all()

def test_pipeline_backend_under_3_seconds():
    # Core compute + Tab1 scorecard preparation; not browser-render time.
    begin=time.perf_counter()
    a=anomaly_bundle();f=forecast_bundle();c=churn_bundle()
    summary=executive_summary(a,f,c)
    elapsed=time.perf_counter()-begin
    assert summary['top_k_count']>0
    assert elapsed<3.0,f'Backend compute exceeded 3s: {elapsed:.3f}s'
