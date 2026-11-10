"""Week 6, Step 2: Fari contract-start cohorts and as-of lifecycle classification.

All contract dates and amounts come from explicitly SYNTHETIC assumptions.
No renewal outcome is used before contract maturity. Activity continuity is NOT renewal retention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_OBSERVATION_COLS = {
    'deployment_id', 'platform', 'date', 'daily_engagement_sessions',
    'avg_session_duration_min', 'caregiver_app_interactions',
    'health_alert_response_time_min', 'renewal_status',
}
REQUIRED_CONTRACT_COLS = {
    'deployment_id', 'assumed_contract_start_date', 'assumed_contract_end_date',
    'assumed_contract_start_quarter', 'assumed_contract_term_months',
    'assumed_monthly_contract_value_usd', 'metadata_source',
}


def prepare_fari_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Return validated Fari rows sorted by ID/date, without modifying the input."""
    missing = REQUIRED_OBSERVATION_COLS - set(observations.columns)
    if missing:
        raise ValueError(f'Missing observation columns: {sorted(missing)}')
    out = observations.loc[observations.platform.eq('Fari')].copy()
    if out.empty:
        raise ValueError('No Fari observations')
    out['date'] = pd.to_datetime(out['date'], errors='raise').dt.normalize()
    if out[['deployment_id', 'date']].isna().any().any():
        raise ValueError('Null deployment ID or observation date')
    if out.duplicated(['deployment_id', 'date']).any():
        raise ValueError('Duplicate ID/date observations')
    for col in ('daily_engagement_sessions','avg_session_duration_min',
                'caregiver_app_interactions','health_alert_response_time_min'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
        if (out[col].dropna() < 0).any():
            raise ValueError(f'Negative value in {col}')
    return out.sort_values(['deployment_id','date']).reset_index(drop=True)


def prepare_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_CONTRACT_COLS - set(contracts.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')
    out = contracts.copy()
    if out['deployment_id'].isna().any() or out['deployment_id'].duplicated().any():
        raise ValueError('Contract IDs must be non-null and unique')
    for c in ('assumed_contract_start_date','assumed_contract_end_date'):
        out[c] = pd.to_datetime(out[c], errors='raise').dt.normalize()
    if (out.assumed_contract_end_date < out.assumed_contract_start_date).any():
        raise ValueError('Contract end precedes start')
    actual_quarters = out.assumed_contract_start_date.dt.to_period('Q').astype(str)
    if not actual_quarters.eq(out.assumed_contract_start_quarter.astype(str)).all():
        raise ValueError('Contract-start quarter does not match assumed start date')
    if (pd.to_numeric(out.assumed_contract_term_months) <= 0).any():
        raise ValueError('Contract term must be positive')
    if (pd.to_numeric(out.assumed_monthly_contract_value_usd) < 0).any():
        raise ValueError('Negative contract amount')
    return out.sort_values('deployment_id').reset_index(drop=True)


def contract_cohort_summary(contracts: pd.DataFrame) -> pd.DataFrame:
    """Contracts by ASSUMED signing quarter. Not an observed acquisition-cohort result."""
    c = prepare_contracts(contracts)
    summary = c.groupby('assumed_contract_start_quarter', as_index=False).agg(
        deployment_count=('deployment_id', 'nunique'),
        assumed_monthly_contract_value_usd=('assumed_monthly_contract_value_usd', 'sum'),
    )
    return summary.rename(columns={'assumed_contract_start_quarter':'assumed_start_cohort'})


def lifecycle_as_of(observations: pd.DataFrame, contracts: pd.DataFrame,
                    as_of: str | pd.Timestamp, *, engagement_threshold: float = 1.0,
                    consecutive_declines: int = 2, use_matured_renewal: bool = True) -> pd.DataFrame:
    """Classify each deployment using only observations on/before `as_of`.

    At-Risk: daily engagement decreases across two consecutive month-to-month transitions.
    Churned: engagement below threshold for two consecutive observed months, OR explicit
    Non-Renewal when as_of is strictly AFTER assumed contract end (matured outcome only).
    Missing observations never count as inactivity or decline.
    Status precedence: Churned > At-Risk > Active; Insufficient Data if no valid readings.
    """
    if engagement_threshold < 0 or consecutive_declines < 1:
        raise ValueError('Invalid lifecycle parameters')
    date = pd.Timestamp(as_of).normalize()
    o = prepare_fari_observations(observations)
    c = prepare_contracts(contracts)
    missing_ids = set(o.deployment_id) - set(c.deployment_id)
    if missing_ids:
        raise ValueError(f'Contracts missing deployments: {sorted(missing_ids)}')
    o = o.loc[o.date.le(date)].copy()
    rows = []
    for contract in c.itertuples(index=False):
        account = o[o.deployment_id.eq(contract.deployment_id)].sort_values('date')
        valid = account.dropna(subset=['daily_engagement_sessions'])
        vals = valid['daily_engagement_sessions'].to_numpy(dtype=float)
        # A genuine consecutive monthly sequence is required, not just adjacent observed rows.
        periods = valid.date.dt.to_period('M').tolist()
        two_low = False
        downward_streak = 0
        at_risk = False
        if len(vals) >= 2:
            for j in range(1, len(vals)):
                adjacent = (periods[j] - periods[j-1]).n == 1
                downward_streak = downward_streak + 1 if adjacent and vals[j] < vals[j-1] else 0
                # Status reflects most recent observations, not any historical episode.
                at_risk = downward_streak >= consecutive_declines
                two_low = bool(adjacent and vals[j] < engagement_threshold and vals[j-1] < engagement_threshold)
        # Limit early outcome leakage: only consult renewal_status after assumed contract end.
        renewal_churn = False
        if use_matured_renewal and date > contract.assumed_contract_end_date:
            matured = account.loc[account.date.le(contract.assumed_contract_end_date)]
            if not matured.empty:
                renewal_churn = str(matured.iloc[-1]['renewal_status']).strip().lower() in {
                    'non-renewal','nonrenewal','not renewed','churned'
                }
        if len(vals) == 0:
            status = 'Insufficient Data'
        elif two_low or renewal_churn:
            status = 'Churned'
        elif at_risk:
            status = 'At-Risk'
        else:
            status = 'Active'
        rows.append({
            'deployment_id':contract.deployment_id,
            'assumed_start_cohort':contract.assumed_contract_start_quarter,
            'as_of_date':date,
            'lifecycle_stage':status,
            'observed_period_count':len(vals),
            'latest_engagement_sessions':vals[-1] if len(vals) else np.nan,
            'two_consecutive_declines_observed':at_risk,
            'two_consecutive_low_periods_observed':two_low,
            'matured_nonrenewal_observed':renewal_churn,
        })
    return pd.DataFrame(rows).sort_values('deployment_id').reset_index(drop=True)


def activity_continuity_proxy(observations: pd.DataFrame, contracts: pd.DataFrame,
                              *, months: tuple[int,...] = (3,6,12),
                              engagement_threshold: float = 1.0,
                              observation_cutoff: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Engaged-at-month proxy; NOT true contract renewal retention.

    A deployer is eligible for month m if its assumed contract start + (m-1) months
    is <= cutoff, and a matching observation exists. Missing reads are excluded from
    measured denominator and reported. This is point-month activity, not survival.
    """
    o = prepare_fari_observations(observations)
    c = prepare_contracts(contracts)
    if set(o.deployment_id) - set(c.deployment_id):
        raise ValueError('Observation deployment missing metadata')
    if engagement_threshold < 0 or any(m < 1 for m in months):
        raise ValueError('Invalid month or threshold')
    cutoff = pd.Timestamp(observation_cutoff).normalize() if observation_cutoff is not None else o.date.max()
    lookup = o.set_index(['deployment_id', 'date'])['daily_engagement_sessions']
    rows=[]
    for cohort, group in c.groupby('assumed_contract_start_quarter'):
        for month in months:
            eligible=observed=active=0
            for r in group.itertuples(index=False):
                target = (r.assumed_contract_start_date + pd.DateOffset(months=month-1)).normalize()
                if target > cutoff:
                    continue
                eligible += 1
                val = lookup.get((r.deployment_id, target), np.nan)
                if pd.notna(val):
                    observed += 1
                    active += int(float(val) >= engagement_threshold)
            rows.append({
                'assumed_start_cohort':cohort,'month_number':month,
                'eligible_deployments':eligible,'observed_deployments':observed,
                'missing_observations':eligible-observed,'engaged_deployments':active,
                'engagement_continuity_proxy':active/observed if observed else np.nan,
                'metric_type':'ACTIVITY_PROXY_NOT_CONTRACT_RETENTION',
            })
    return pd.DataFrame(rows).sort_values(['assumed_start_cohort','month_number']).reset_index(drop=True)


# ----------------- Week 6 Step 3: leakage-safe churn prediction -----------------
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

FEATURE_COLUMNS = (
    'session_duration_trend_3m', 'caregiver_interactions_per_month',
    'health_response_time_trend_3m', 'contract_tenure_months',
)


def _last_three_slope(account: pd.DataFrame, column: str) -> float:
    """Last 3 consecutive calendar-month readings; otherwise missing."""
    vals = account.dropna(subset=[column]).tail(3)
    if len(vals) < 3:
        return np.nan
    periods = vals.date.dt.to_period('M').tolist()
    if any((periods[j] - periods[j-1]).n != 1 for j in (1, 2)):
        return np.nan
    return float(np.polyfit(np.arange(3), vals[column].to_numpy(dtype=float), 1)[0])


def build_asof_features(observations: pd.DataFrame, contracts: pd.DataFrame,
                        as_of: str | pd.Timestamp) -> pd.DataFrame:
    """One row per eligible contract; NEVER read outcome labels or future observations.

    Only uses sessions/interaction/response readings dated on/before as_of.
    Contracts must have begun and not matured at as_of (strictly earlier than end).
    """
    cutoff = pd.Timestamp(as_of).normalize()
    obs = prepare_fari_observations(observations)
    meta = prepare_contracts(contracts)
    if set(obs.deployment_id) - set(meta.deployment_id):
        raise ValueError('Fari observation without contract metadata')
    rows = []
    for c in meta.itertuples(index=False):
        if not (c.assumed_contract_start_date <= cutoff < c.assumed_contract_end_date):
            continue
        account = obs.loc[(obs.deployment_id == c.deployment_id) & (obs.date <= cutoff)]
        account = account.sort_values('date')
        # trailing three available calendar months, with month contiguity checked for trends
        last_three = account.tail(3)
        caregiver = last_three.caregiver_app_interactions.mean() if not last_three.empty else np.nan
        tenure = (cutoff.year - c.assumed_contract_start_date.year)*12 + (cutoff.month - c.assumed_contract_start_date.month)
        rows.append({
            'deployment_id':c.deployment_id,
            'prediction_as_of':cutoff,
            'assumed_contract_end_date':c.assumed_contract_end_date,
            'latest_feature_date':account.date.max() if not account.empty else pd.NaT,
            'session_duration_trend_3m':_last_three_slope(account,'avg_session_duration_min'),
            'caregiver_interactions_per_month':float(caregiver) if pd.notna(caregiver) else np.nan,
            'health_response_time_trend_3m':_last_three_slope(account,'health_alert_response_time_min'),
            'contract_tenure_months':int(tenure),
        })
    return pd.DataFrame(rows, columns=['deployment_id','prediction_as_of','assumed_contract_end_date',
                                       'latest_feature_date',*FEATURE_COLUMNS]).sort_values('deployment_id').reset_index(drop=True)


def construct_matured_labels(observations: pd.DataFrame, contracts: pd.DataFrame,
                             *, outcome_as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Retrospective labels ONLY, taken at/after contract end; never a model feature.

    Source dataset repeats a single known synthetic renewal outcome at each month;
    it is therefore read only from the last observed record AT contract end.
    """
    cutoff = pd.Timestamp(outcome_as_of).normalize()
    obs = prepare_fari_observations(observations)
    meta = prepare_contracts(contracts)
    rows=[]
    for c in meta.itertuples(index=False):
        if c.assumed_contract_end_date > cutoff:
            continue
        account = obs.loc[(obs.deployment_id == c.deployment_id) &
                          (obs.date <= c.assumed_contract_end_date)].sort_values('date')
        if account.empty:
            continue
        last = account.iloc[-1]
        if last.date.to_period('M') != c.assumed_contract_end_date.to_period('M'):
            continue
        status = str(last.renewal_status).strip().lower()
        if status in ('non-renewal','nonrenewal','not renewed','churned'):
            label = 1
        elif status in ('renewal','renewed'):
            label = 0
        else:
            continue
        rows.append({'deployment_id':c.deployment_id,'churn_label':label,
                     'outcome_known_as_of':c.assumed_contract_end_date})
    return pd.DataFrame(rows,columns=['deployment_id','churn_label','outcome_known_as_of'])


def make_model() -> Pipeline:
    """Imputation and scaling are fitted inside each training fold only."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(C=1.0, class_weight='balanced',max_iter=2000,random_state=42)),
    ])


def leave_one_deployment_out(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Strict deployment-level leave-one-out retrospective evaluation.

    This 10-deployment synthetic sample yields highly unstable metrics.
    Each fold fits preprocessing and classifier on only OTHER deployments.
    """
    needed = {'deployment_id', *FEATURE_COLUMNS}
    if not needed.issubset(features):
        raise ValueError(f'Missing features {sorted(needed - set(features))}')
    if features.deployment_id.duplicated().any() or labels.deployment_id.duplicated().any():
        raise ValueError('Duplicate deployment IDs')
    if not {'deployment_id','churn_label'}.issubset(labels):
        raise ValueError('Labels need deployment_id and churn_label')
    ds = features[['deployment_id',*FEATURE_COLUMNS]].merge(
        labels[['deployment_id','churn_label']],on='deployment_id',how='inner',validate='one_to_one')
    if len(ds) < 3 or ds.churn_label.nunique() != 2 or not ds.churn_label.isin([0,1]).all():
        raise ValueError('Need at least 3 labeled deployments from both classes')
    result=[]
    for i in range(len(ds)):
        train=ds.drop(ds.index[i]); test=ds.iloc[[i]]
        if train.churn_label.nunique() < 2:
            raise ValueError('Training fold has only one outcome class; reliable LOOCV unavailable')
        m=make_model()
        m.fit(train[list(FEATURE_COLUMNS)],train.churn_label)
        p=float(m.predict_proba(test[list(FEATURE_COLUMNS)])[0, list(m.classes_).index(1)])
        result.append({'deployment_id':test.iloc[0].deployment_id,
                       'churn_label':int(test.iloc[0].churn_label),'oof_churn_probability':p,
                       'training_deployment_count':len(train)})
    return pd.DataFrame(result).sort_values('deployment_id').reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame, threshold:float=0.5,k:int=10) -> dict:
    if not (0 <= threshold <= 1) or k < 1:
        raise ValueError('Invalid threshold or K')
    y=predictions.churn_label.astype(int).to_numpy()
    p=predictions.oof_churn_probability.astype(float).to_numpy()
    if len(y)==0 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid score data')
    pred=(p>=threshold).astype(int)
    top = predictions.sort_values(['oof_churn_probability','deployment_id'],ascending=[False,True]).head(k)
    return {
        'n_deployments':len(y),'n_churned':int(y.sum()),
        'auc':float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,
        'precision_at_0_5':float(precision_score(y,pred,zero_division=0)),
        'recall_at_0_5':float(recall_score(y,pred,zero_division=0)),
        'top_k_precision':float(top.churn_label.mean()),'effective_k':len(top),
        'threshold':threshold,
    }


def train_full_and_score(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """In-sample exploratory scores, NOT unbiased validation predictions."""
    ds=features.merge(labels[['deployment_id','churn_label']],on='deployment_id',validate='one_to_one')
    if ds.churn_label.nunique()!=2:
        raise ValueError('Need both classes')
    model=make_model();model.fit(ds[list(FEATURE_COLUMNS)],ds.churn_label)
    out=features[['deployment_id','prediction_as_of']].copy()
    out['churn_probability_exploratory']=model.predict_proba(features[list(FEATURE_COLUMNS)])[:,list(model.classes_).index(1)]
    out['score_type']='IN_SAMPLE_EXPLORATORY_NOT_VALIDATION'
    return out.sort_values('churn_probability_exploratory',ascending=False).reset_index(drop=True)


# ---------------- Week 6 Step 4: Revenue, residual LTV and Top-K ----------------
def calculate_revenue_risk(scores: pd.DataFrame, contracts: pd.DataFrame,
                           as_of: str | pd.Timestamp, *,
                           probability_col: str = 'churn_probability_exploratory',
                           high_risk_threshold: float = 0.6) -> pd.DataFrame:
    """Monthly synthetic contract-value estimates, not observed revenue or real LTV.

    Monthly periods include the as_of calendar month until the contract-end month,
    inclusive. For example, September 1 to December 31 = four monthly periods.
    The probability is a conditional model SCORE; financial figures are illustrations.
    """
    if not 0 <= high_risk_threshold <= 1:
        raise ValueError('Invalid high-risk threshold')
    if not {'deployment_id', probability_col}.issubset(scores):
        raise ValueError('Missing deployment ID or probability')
    if scores.deployment_id.isna().any() or scores.deployment_id.duplicated().any():
        raise ValueError('Scores require unique non-null deployment IDs')
    c = prepare_contracts(contracts)
    d = pd.Timestamp(as_of).normalize()
    if 'prediction_as_of' in scores:
        score_dates = pd.to_datetime(scores.prediction_as_of, errors='raise').dt.normalize()
        if not score_dates.eq(d).all():
            raise ValueError('Score date and revenue valuation date differ')
    if not set(scores.deployment_id).issubset(set(c.deployment_id)):
        raise ValueError('Scores contain unknown contracts')
    out = scores[['deployment_id',probability_col]].merge(c, on='deployment_id',
                                                        validate='one_to_one', how='left')
    prob = pd.to_numeric(out[probability_col],errors='raise')
    if prob.isna().any() or (~np.isfinite(prob)).any() or ((prob<0)|(prob>1)).any():
        raise ValueError('Probabilities must be finite numbers in [0, 1]')
    starts = out.assumed_contract_start_date.dt.to_period('M')
    ends = out.assumed_contract_end_date.dt.to_period('M')
    date_month = d.to_period('M')
    if (starts > date_month).any():
        raise ValueError('Cannot value a contract before it starts')
    out['valuation_as_of'] = d
    out['months_remaining_including_current'] = [max(0,(e-date_month).n+1) for e in ends]
    out['assumed_remaining_contract_value_usd'] = (
        out.months_remaining_including_current * out.assumed_monthly_contract_value_usd)
    out['expected_revenue_at_risk_usd'] = out.assumed_remaining_contract_value_usd * prob
    out['residual_ltv_usd'] = out.assumed_remaining_contract_value_usd * (1-prob)
    # Contract first HALF is determined using elapsed months from start;
    # score date is fixed rather than retrospectively moving a score.
    out['elapsed_contract_months'] = [(date_month-s).n for s in starts]
    out['early_contract_half'] = out.elapsed_contract_months < (out.assumed_contract_term_months / 2)
    out['early_high_value_nonrenewal_risk'] = (out.early_contract_half & (prob > high_risk_threshold)
                                            & (out.months_remaining_including_current > 0))
    out['score_source'] = ('IN_SAMPLE_EXPLORATORY_NOT_VALIDATION' if probability_col ==
                           'churn_probability_exploratory' else probability_col)
    return out.sort_values('deployment_id').reset_index(drop=True)


def select_top_k(revenue: pd.DataFrame, k: int=10) -> pd.DataFrame:
    """Rank eligible contracts by expected revenue loss, NOT churn probability."""
    if not isinstance(k,int) or isinstance(k,bool) or k<1:
        raise ValueError('K must be a positive integer')
    required={'deployment_id','expected_revenue_at_risk_usd','months_remaining_including_current'}
    if not required.issubset(revenue):
        raise ValueError('Missing revenue columns')
    if revenue.deployment_id.duplicated().any() or revenue.deployment_id.isna().any():
        raise ValueError('Duplicate or null deployment ID')
    if not np.isfinite(revenue.expected_revenue_at_risk_usd).all() or (revenue.expected_revenue_at_risk_usd<0).any():
        raise ValueError('Invalid revenue at risk')
    pool=revenue.loc[revenue.months_remaining_including_current.gt(0)].copy()
    pool=pool.sort_values(['expected_revenue_at_risk_usd','deployment_id'],ascending=[False,True])
    result=pool.head(k).copy().reset_index(drop=True)
    result.insert(0,'intervention_rank',np.arange(1,len(result)+1))
    return result


def risk_concentration(revenue: pd.DataFrame) -> dict:
    """Compare empirical early-risk share with prior 30% reference, without forcing result."""
    required={'deployment_id','residual_ltv_usd','expected_revenue_at_risk_usd',
              'early_high_value_nonrenewal_risk'}
    if not required.issubset(revenue):
        raise ValueError('Missing concentration fields')
    if revenue.deployment_id.duplicated().any():
        raise ValueError('Duplicate deployment ID')
    if (revenue[['residual_ltv_usd','expected_revenue_at_risk_usd']]<0).any().any():
        raise ValueError('Negative value')
    if not np.isfinite(revenue[['residual_ltv_usd','expected_revenue_at_risk_usd']]).all().all():
        raise ValueError('Non-finite value')
    total=float(revenue.expected_revenue_at_risk_usd.sum())
    early=float(revenue.loc[revenue.early_high_value_nonrenewal_risk,
                            'expected_revenue_at_risk_usd'].sum())
    n_top=int(np.ceil(len(revenue)*0.25))
    top_by_ltv=revenue.sort_values(['residual_ltv_usd','deployment_id'],ascending=[False,True]).head(n_top)
    top_share=(float(top_by_ltv.expected_revenue_at_risk_usd.sum())/total if total>0 else None)
    return {'n_deployments':len(revenue),'total_revenue_at_risk_usd':total,
            'early_high_value_at_risk_usd':early,
            'early_high_value_risk_share': early/total if total>0 else None,
            'prior_streaming_benchmark_share':0.30,
            'top_residual_ltv_quartile_n':n_top,
            'top_residual_ltv_quartile_risk_share':top_share,
            'top_quartile_meets_25pct_assertion':top_share>=0.25 if top_share is not None else None,
            'interpretation':'Synthetic financial assumptions; exploratory in-sample risk scores; not real revenue.'}


# ---------------- Week 6 revision: honest retention and concentration audit ----------------
def retention_observability(contracts: pd.DataFrame, *, as_of='2025-12-31', months=(3,6,12)) -> pd.DataFrame:
    """Audit whether observed contract EVENTS support true renewal retention.

    There are no timestamped renewal/termination events in the Week 2 source.
    Consequently each 3/6/12-month renewal-retention rate is UNAVAILABLE rather
    than imputed from December's static renewal_status.
    """
    c=prepare_contracts(contracts)
    if any(m <= 0 for m in months): raise ValueError('Positive month horizons required')
    cutoff=pd.Timestamp(as_of).normalize()
    rows=[]
    for cohort,grp in c.groupby('assumed_contract_start_quarter'):
        for m in months:
            dates=grp.assumed_contract_start_date+pd.DateOffset(months=m)
            eligible=int((dates <= cutoff).sum())
            rows.append({'cohort':cohort,'horizon_months':m,'eligible_contracts':eligible,
                         'timestamped_renewal_events':0,'verified_renewal_retention_rate':np.nan,
                         'metric_status':'UNAVAILABLE_NO_TIMESTAMPED_CONTRACT_EVENTS'})
    return pd.DataFrame(rows)


def at_risk_quartile_audit(revenue: pd.DataFrame, at_risk_ids) -> dict:
    """Apply quartile test strictly to externally defined at-risk pool, not all deployments."""
    pool=revenue.loc[revenue.deployment_id.isin(set(at_risk_ids))].copy()
    if pool.empty:
        return {'at_risk_pool_size':0,'top_quartile_size':0,'share':None,
                'meets_25pct':None,'reason':'No at-risk contracts; test not evaluable'}
    total=float(pool.expected_revenue_at_risk_usd.sum())
    n=max(1,int(np.ceil(len(pool)*.25)))
    top=pool.nlargest(n,'residual_ltv_usd')
    share=float(top.expected_revenue_at_risk_usd.sum()/total) if total>0 else None
    return {'at_risk_pool_size':len(pool),'top_quartile_size':n,
            'share':share,'meets_25pct':share>=.25 if share is not None else None,
            'reason':'Evaluated on at-risk pool; numerator and denominator aligned'}


def model_driver_coefficients(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Standardized logistic coefficients; exploratory association, not causal drivers."""
    ds=features.merge(labels[['deployment_id','churn_label']],on='deployment_id',validate='one_to_one')
    if ds.churn_label.nunique()<2: raise ValueError('Both classes required')
    model=make_model(); model.fit(ds[list(FEATURE_COLUMNS)],ds.churn_label)
    coef=model.named_steps['logistic'].coef_[0]
    return pd.DataFrame({'feature':list(FEATURE_COLUMNS),'standardized_coefficient':coef,
                         'interpretation':'ASSOCIATION_NOT_CAUSAL'}).sort_values('standardized_coefficient')
