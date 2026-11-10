import numpy as np
import pandas as pd
import pytest
from src.churn_model import calculate_revenue_risk,select_top_k,risk_concentration

@pytest.fixture
def contracts():
 return pd.DataFrame([dict(deployment_id=f'FARI-{n:03}',assumed_contract_start_date='2025-01-01',
 assumed_contract_end_date='2025-12-31',assumed_contract_start_quarter='2025Q1',
 assumed_contract_term_months=12,assumed_monthly_contract_value_usd=1000*n,
 metadata_source='W06_ADDED_SYNTHETIC_ASSUMPTION') for n in range(1,4)])
@pytest.fixture
def scores():
 return pd.DataFrame({'deployment_id':['FARI-001','FARI-002','FARI-003'],
 'prediction_as_of':['2025-09-01']*3,'churn_probability_exploratory':[0.2,0.8,0.6]})
@pytest.fixture
def revenue(contracts,scores):
 return calculate_revenue_risk(scores,contracts,'2025-09-01')

@pytest.mark.parametrize('n,p,remaining,at_risk,ltv',[(1,.2,4000,800,3200),(2,.8,8000,6400,1600),(3,.6,12000,7200,4800)])
def test_financial_formula(revenue,n,p,remaining,at_risk,ltv):
 r=revenue.loc[revenue.deployment_id.eq(f'FARI-{n:03}')].iloc[0]
 assert r.months_remaining_including_current==4
 assert r.churn_probability_exploratory==pytest.approx(p)
 assert r.assumed_remaining_contract_value_usd==pytest.approx(remaining)
 assert r.expected_revenue_at_risk_usd==pytest.approx(at_risk)
 assert r.residual_ltv_usd==pytest.approx(ltv)

@pytest.mark.parametrize('k,expected',[(1,['FARI-003']),(2,['FARI-003','FARI-002']),(3,['FARI-003','FARI-002','FARI-001']),(9,['FARI-003','FARI-002','FARI-001'])])
def test_topk_correct(revenue,k,expected):
 assert select_top_k(revenue,k).deployment_id.tolist()==expected

@pytest.mark.parametrize('prob',[-.1,1.1,float('nan'),float('inf'),-float('inf')])
def test_reject_bad_probabilities(contracts,scores,prob):
 s=scores.copy();s.loc[0,'churn_probability_exploratory']=prob
 with pytest.raises(ValueError): calculate_revenue_risk(s,contracts,'2025-09-01')

@pytest.mark.parametrize('k',[0,-1,1.2,True])
def test_reject_bad_k(revenue,k):
 with pytest.raises(ValueError):select_top_k(revenue,k)

@pytest.mark.parametrize('date,count', [('2025-01-01',12),('2025-06-01',7),('2025-09-01',4),('2025-12-01',1),('2026-01-01',0)])
def test_months_remaining(contracts,scores,date,count):
 s=scores.drop(columns=['prediction_as_of'])
 assert set(calculate_revenue_risk(s,contracts,date).months_remaining_including_current)=={count}

def test_early_half_is_date_dependent(contracts,scores):
 s=scores.drop(columns=['prediction_as_of'])
 early=calculate_revenue_risk(s,contracts,'2025-03-01')
 late=calculate_revenue_risk(s,contracts,'2025-09-01')
 assert early.early_contract_half.all()
 assert not late.early_contract_half.any()
 assert early.early_high_value_nonrenewal_risk.sum()==1
 assert not late.early_high_value_nonrenewal_risk.any()

def test_scoring_asof_mismatch_rejected(contracts,scores):
 with pytest.raises(ValueError,match='date'):calculate_revenue_risk(scores,contracts,'2025-06-01')

def test_reject_duplicate_score(contracts,scores):
 with pytest.raises(ValueError): calculate_revenue_risk(pd.concat([scores,scores.iloc[[0]]]),contracts,'2025-09-01')

def test_reject_unknown_deployment(contracts,scores):
 s=scores.copy();s.loc[0,'deployment_id']='UNKNOWN'
 with pytest.raises(ValueError):calculate_revenue_risk(s,contracts,'2025-09-01')

def test_topk_tie_break_deployment_id(revenue):
 r=revenue.copy();r.loc[:,'expected_revenue_at_risk_usd']=100
 assert select_top_k(r,1).deployment_id.tolist()==['FARI-001']

def test_concentration_manual(revenue):
 m=risk_concentration(revenue)
 assert m['total_revenue_at_risk_usd']==pytest.approx(14400)
 assert m['early_high_value_at_risk_usd']==0
 assert m['early_high_value_risk_share']==0
 assert m['top_residual_ltv_quartile_n']==1
 assert m['top_residual_ltv_quartile_risk_share']==pytest.approx(7200/14400)

def test_no_risk_zero_denominator(revenue):
 r=revenue.copy();r['expected_revenue_at_risk_usd']=0
 result=risk_concentration(r)
 assert result['early_high_value_risk_share'] is None
 assert result['top_quartile_meets_25pct_assertion'] is None

def test_all_out_of_contract_excluded(contracts,scores):
 r=calculate_revenue_risk(scores.drop(columns=['prediction_as_of']),contracts,'2026-03-01')
 assert len(select_top_k(r,10))==0

def test_financial_identity(revenue):
 assert np.allclose(revenue.residual_ltv_usd+revenue.expected_revenue_at_risk_usd,revenue.assumed_remaining_contract_value_usd)

def test_original_input_unchanged(contracts,scores):
 original=contracts.copy(deep=True); ss=scores.copy(deep=True)
 calculate_revenue_risk(scores,contracts,'2025-09-01')
 pd.testing.assert_frame_equal(contracts,original);pd.testing.assert_frame_equal(scores,ss)

def test_amount_derived_from_assumed_metadata(revenue):
 assert (revenue.metadata_source=='W06_ADDED_SYNTHETIC_ASSUMPTION').all()

def test_probability_drives_loss_not_ltv(revenue):
 r=revenue.copy();r['churn_probability_exploratory']=[.9,.05,.05]
 assert select_top_k(r,1).deployment_id.iloc[0]=='FARI-003' # ranks by computed loss column

@pytest.mark.parametrize('threshold',[-.1,1.1])
def test_reject_bad_risk_threshold(contracts,scores,threshold):
 with pytest.raises(ValueError):calculate_revenue_risk(scores,contracts,'2025-09-01',high_risk_threshold=threshold)

def test_concentration_deployment_count(revenue):
 assert risk_concentration(revenue)['n_deployments']==3

def test_topk_has_rank(revenue):
 assert select_top_k(revenue,2).intervention_rank.tolist()==[1,2]
