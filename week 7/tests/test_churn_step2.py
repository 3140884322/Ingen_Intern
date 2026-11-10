import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.churn_model import (prepare_fari_observations, prepare_contracts,
                             contract_cohort_summary,lifecycle_as_of,activity_continuity_proxy)

@pytest.fixture
def obs():
    return pd.read_csv(ROOT/'data/W02_Robot_Deployment_Dataset.csv')

@pytest.fixture
def contracts():
    return pd.read_csv(ROOT/'data/W06_Fari_Contract_Metadata_SYNTHETIC_ASSUMPTIONS.csv')

@pytest.fixture
def small_contracts(contracts):
    return contracts[contracts.deployment_id.eq('FARI-001')].copy()

@pytest.fixture
def make_observations(obs):
    def build(engagement, dates=None, statuses=None):
        base=obs[obs.deployment_id.eq('FARI-001')].iloc[:len(engagement)].copy()
        base['daily_engagement_sessions']=engagement
        if dates is not None:base['date']=dates
        if statuses is not None:base['renewal_status']=statuses
        return base
    return build

# 10 input, provenance and cohort assertions

def test_only_fari(obs):assert prepare_fari_observations(obs).platform.eq('Fari').all()
def test_120_rows(obs):assert len(prepare_fari_observations(obs))==120
def test_10_ids(obs):assert prepare_fari_observations(obs).deployment_id.nunique()==10
def test_dates_converted(obs):assert pd.api.types.is_datetime64_any_dtype(prepare_fari_observations(obs).date)
def test_non_mutating_input(obs):
    x=obs.copy(deep=True);prepare_fari_observations(x);pd.testing.assert_frame_equal(x,obs)
def test_sorted_rows(obs):
    x=prepare_fari_observations(obs)
    assert x[['deployment_id','date']].equals(x[['deployment_id','date']].sort_values(['deployment_id','date']).reset_index(drop=True))
def test_missing_obs_column(obs):
    with pytest.raises(ValueError,match='Missing observation'):prepare_fari_observations(obs.drop(columns=['date']))
def test_duplicate_obs(obs):
    with pytest.raises(ValueError,match='Duplicate'):prepare_fari_observations(pd.concat([obs,obs.loc[obs.platform.eq("Fari")].iloc[[0]]]))
def test_missing_deployment_id(obs):
    x=obs.copy();x.loc[x.platform.eq('Fari').idxmax(),'deployment_id']=None
    with pytest.raises(ValueError,match='Null'):prepare_fari_observations(x)
def test_negative_engagement(obs):
    x=obs.copy();x.loc[x.platform.eq('Fari').idxmax(),'daily_engagement_sessions']=-1
    with pytest.raises(ValueError,match='Negative'):prepare_fari_observations(x)
def test_contract_unique(contracts):assert prepare_contracts(contracts).deployment_id.is_unique
def test_contract_source(contracts):assert contracts.metadata_source.eq('W06_ADDED_SYNTHETIC_ASSUMPTION').all()
def test_cohort_counts(contracts):assert contract_cohort_summary(contracts).deployment_count.tolist()==[10]
def test_single_assumed_cohort(contracts):assert contract_cohort_summary(contracts).assumed_start_cohort.tolist()==['2025Q1']
def test_cohort_monthly_sum(contracts):
    assert contract_cohort_summary(contracts).assumed_monthly_contract_value_usd.iloc[0]==contracts.assumed_monthly_contract_value_usd.sum()
def test_bad_quarter(contracts):
    c=contracts.copy();c.loc[0,'assumed_contract_start_quarter']='2025Q2'
    with pytest.raises(ValueError,match='quarter'):prepare_contracts(c)
def test_duplicate_contract(contracts):
    with pytest.raises(ValueError,match='unique'):prepare_contracts(pd.concat([contracts,contracts.iloc[[0]]]))
def test_negative_contract_value(contracts):
    c=contracts.copy();c.loc[0,'assumed_monthly_contract_value_usd']=-10
    with pytest.raises(ValueError,match='Negative'):prepare_contracts(c)
def test_bad_contract_dates(contracts):
    c=contracts.copy();c.loc[0,'assumed_contract_end_date']='2024-01-01'
    with pytest.raises(ValueError,match='precedes'):prepare_contracts(c)
def test_missing_contract_column(contracts):
    with pytest.raises(ValueError,match='Missing metadata'):prepare_contracts(contracts.drop(columns=['metadata_source']))

# Lifecycle behavior and temporal-leakage tests

def test_default_active(make_observations,small_contracts):
    assert lifecycle_as_of(make_observations([5,5,5]),small_contracts,'2025-03-31').lifecycle_stage.iloc[0]=='Active'
def test_one_decline_not_at_risk(make_observations,small_contracts):
    assert lifecycle_as_of(make_observations([5,4]),small_contracts,'2025-02-28').lifecycle_stage.iloc[0]=='Active'
def test_two_declines_at_risk(make_observations,small_contracts):
    x=lifecycle_as_of(make_observations([5,4,3]),small_contracts,'2025-03-31')
    assert x.lifecycle_stage.iloc[0]=='At-Risk' and bool(x.two_consecutive_declines_observed.iloc[0])
def test_low_two_periods_churned(make_observations,small_contracts):
    assert lifecycle_as_of(make_observations([0.5,0.4]),small_contracts,'2025-02-28').lifecycle_stage.iloc[0]=='Churned'
def test_one_low_not_churned(make_observations,small_contracts):
    assert lifecycle_as_of(make_observations([3,0.5]),small_contracts,'2025-02-28').lifecycle_stage.iloc[0]=='Active'
def test_churn_precedes_risk(make_observations,small_contracts):
    assert lifecycle_as_of(make_observations([3,0.8,0.5]),small_contracts,'2025-03-31').lifecycle_stage.iloc[0]=='Churned'
def test_future_engagement_not_used(make_observations,small_contracts):
    x=make_observations([5,5,4,3]);assert lifecycle_as_of(x,small_contracts,'2025-02-28').lifecycle_stage.iloc[0]=='Active'
def test_endyear_nonrenewal_not_used_early(make_observations,small_contracts):
    x=make_observations([5,5,5],statuses=['Non-Renewal']*3)
    assert lifecycle_as_of(x,small_contracts,'2025-03-31').lifecycle_stage.iloc[0]=='Active'
def test_matured_nonrenewal(make_observations,small_contracts):
    x=make_observations([5,5,5],statuses=['Non-Renewal']*3)
    assert lifecycle_as_of(x,small_contracts,'2026-01-01').lifecycle_stage.iloc[0]=='Churned'
def test_no_matured_renewal_option(make_observations,small_contracts):
    x=make_observations([5,5],statuses=['Non-Renewal']*2)
    assert lifecycle_as_of(x,small_contracts,'2026-01-01',use_matured_renewal=False).lifecycle_stage.iloc[0]=='Active'
def test_missing_period_not_consecutive(make_observations,small_contracts):
    x=make_observations([5,4,3],dates=['2025-01-01','2025-03-01','2025-04-01'])
    assert lifecycle_as_of(x,small_contracts,'2025-04-30').lifecycle_stage.iloc[0]=='Active'
def test_missing_values_insufficient(make_observations,small_contracts):
    x=make_observations([np.nan,np.nan]);assert lifecycle_as_of(x,small_contracts,'2025-02-28').lifecycle_stage.iloc[0]=='Insufficient Data'
def test_before_any_observation(make_observations,small_contracts):
    x=make_observations([5,4]);assert lifecycle_as_of(x,small_contracts,'2024-12-01').lifecycle_stage.iloc[0]=='Insufficient Data'
def test_invalid_threshold(make_observations,small_contracts):
    with pytest.raises(ValueError,match='Invalid'):lifecycle_as_of(make_observations([5]),small_contracts,'2025-01-01',engagement_threshold=-1)
def test_more_declines_required(make_observations,small_contracts):
    x=lifecycle_as_of(make_observations([5,4,3]),small_contracts,'2025-03-31',consecutive_declines=3)
    assert x.lifecycle_stage.iloc[0]=='Active'
def test_output_all_contracts(obs,contracts):assert len(lifecycle_as_of(obs,contracts,'2025-06-30'))==10

def test_unknown_deployment(obs,contracts):
    with pytest.raises(ValueError,match='Contracts missing'):lifecycle_as_of(obs,contracts.iloc[1:],'2025-01-31')

# Activity proxy is NOT contract retention and keeps eligibility/observability explicit

def test_proxy_three_horizons(obs,contracts):assert activity_continuity_proxy(obs,contracts).month_number.tolist()==[3,6,12]
def test_proxy_label(obs,contracts):assert activity_continuity_proxy(obs,contracts).metric_type.eq('ACTIVITY_PROXY_NOT_CONTRACT_RETENTION').all()
def test_proxy_month3_denominator(obs,contracts):
    x=activity_continuity_proxy(obs,contracts);assert x.loc[x.month_number.eq(3),'observed_deployments'].iloc[0]==10
def test_proxy_cutoff(obs,contracts):
    x=activity_continuity_proxy(obs,contracts,observation_cutoff='2025-04-30')
    assert x.loc[x.month_number.eq(6),'eligible_deployments'].iloc[0]==0
def test_proxy_missing_value_excluded(obs,contracts):
    x=obs.copy();sel=x.platform.eq('Fari') & x.deployment_id.eq('FARI-001') & x.date.eq('2025-03-01')
    x.loc[sel,'daily_engagement_sessions']=np.nan
    p=activity_continuity_proxy(x,contracts,months=(3,))
    assert p.observed_deployments.iloc[0]==9 and p.missing_observations.iloc[0]==1
def test_proxy_no_future_observation(obs,contracts):
    x=obs.copy();x.loc[x.platform.eq('Fari') & x.date.eq('2025-03-01'),'daily_engagement_sessions']=0
    a=activity_continuity_proxy(x,contracts,months=(3,),observation_cutoff='2025-02-01')
    assert np.isnan(a.engagement_continuity_proxy.iloc[0])
def test_proxy_invalid_month(obs,contracts):
    with pytest.raises(ValueError,match='Invalid'):activity_continuity_proxy(obs,contracts,months=(0,))
