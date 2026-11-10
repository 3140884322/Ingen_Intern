"""Clearly labeled synthetic contract-event SCENARIO analytics, never observed InGen retention."""
import pandas as pd

def synthetic_contract_retention(events: pd.DataFrame) -> pd.DataFrame:
    required={'deployment_id','assumed_start_cohort','horizon','assessment_date','contract_active_in_scenario','scenario_type'}
    if not required.issubset(events):
        raise ValueError(f'Missing event columns: {sorted(required-set(events))}')
    if events.duplicated(['deployment_id','horizon']).any():raise ValueError('Duplicate deployment/horizon')
    if not events.scenario_type.eq('ADDITIONAL_SYNTHETIC_EVENT_SCENARIO_NOT_OBSERVED').all():
        raise ValueError('Unlabeled or observed events cannot be treated as synthetic')
    if not events.contract_active_in_scenario.isin([0,1]).all():raise ValueError('Retention must be binary')
    e=events.copy();e['assessment_date']=pd.to_datetime(e.assessment_date,errors='raise')
    result=e.groupby(['assumed_start_cohort','horizon'],as_index=False).agg(eligible=('deployment_id','nunique'),retained=('contract_active_in_scenario','sum'))
    result['retention_rate']=result.retained/result.eligible
    result['metric_type']='SIMULATED_CONTRACT_RETENTION_NOT_OBSERVED'
    return result

def conditional_at_risk_quartile_assertion(revenue, at_risk_ids):
    """Report the baseline result; never force the 25-percent test to pass."""
    from src.churn_model import at_risk_quartile_audit
    return at_risk_quartile_audit(revenue,at_risk_ids)
