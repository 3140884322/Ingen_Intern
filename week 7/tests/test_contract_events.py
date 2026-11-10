import pandas as pd
import pytest
from pathlib import Path
from src.contract_events import synthetic_contract_retention
P=Path(__file__).resolve().parents[1]
@pytest.fixture
def events():return pd.read_csv(P/'data/W06_Synthetic_Contract_Events_SCENARIO.csv')
@pytest.mark.parametrize('horizon', ['30d','60d','90d','3m','6m','12m'])
def test_original_cohort_complete(events,horizon):
    result=synthetic_contract_retention(events)
    r=result[result.horizon.eq(horizon)].iloc[0]
    assert r.eligible==10 and 0<=r.retention_rate<=1
@pytest.mark.parametrize('horizon,expected', [('30d',1.0),('60d',1.0),('90d',1.0),('3m',1.0),('6m',.9),('12m',.8)])
def test_stated_hypothetical_scenario(events,horizon,expected):
    actual=synthetic_contract_retention(events).set_index('horizon').loc[horizon,'retention_rate']
    assert actual==pytest.approx(expected)
def test_duplicate_event_rejected(events):
    with pytest.raises(ValueError,match='Duplicate'):synthetic_contract_retention(pd.concat([events,events.iloc[[0]]]))
def test_missing_horizon_rejected(events):
    with pytest.raises(ValueError,match='Missing'):synthetic_contract_retention(events.drop(columns='horizon'))
def test_unlabeled_event_rejected(events):
    events.loc[0,'scenario_type']='OBSERVED';
    with pytest.raises(ValueError,match='Unlabeled'):synthetic_contract_retention(events)
def test_binary_state_required(events):
    events.loc[0,'contract_active_in_scenario']=2
    with pytest.raises(ValueError,match='binary'):synthetic_contract_retention(events)
def test_synthetic_events_do_not_modify_week2_source(events):
    original=pd.read_csv(P/'data/W02_Robot_Deployment_Dataset.csv')
    assert len(original)==360 and len(events)==60
