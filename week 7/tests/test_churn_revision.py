import numpy as np
import pandas as pd
import pytest
from src.churn_model import retention_observability, at_risk_quartile_audit

@pytest.fixture
def contracts():
    return pd.read_csv('data/W06_Fari_Contract_Metadata_SYNTHETIC_ASSUMPTIONS.csv')

@pytest.mark.parametrize('m',[3,6,12])
def test_retention_unavailable_without_events(contracts,m):
    result=retention_observability(contracts,months=(m,))
    assert result.verified_renewal_retention_rate.isna().all()
    assert (result.timestamped_renewal_events==0).all()
    assert (result.eligible_contracts<=len(contracts)).all()

def test_bad_horizon(contracts):
    with pytest.raises(ValueError):retention_observability(contracts,months=(0,))

def test_cutoff_eligibility(contracts):
    r=retention_observability(contracts,as_of='2025-02-01',months=(12,))
    assert r.eligible_contracts.sum()==0

def test_empty_at_risk():
    df=pd.DataFrame({'deployment_id':['a'],'expected_revenue_at_risk_usd':[10.], 'residual_ltv_usd':[90.]})
    r=at_risk_quartile_audit(df,[])
    assert r['share'] is None and r['meets_25pct'] is None

def test_at_risk_scope():
    df=pd.DataFrame({'deployment_id':['a','b','c'],'expected_revenue_at_risk_usd':[2.,8.,900.], 'residual_ltv_usd':[20.,10.,1000.]})
    r=at_risk_quartile_audit(df,['a','b'])
    assert r['at_risk_pool_size']==2 and r['share']==.2 and not r['meets_25pct']
