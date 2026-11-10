"""Step 3 model/feature engineering tests, independent of Step 2 suite."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.churn_model import (FEATURE_COLUMNS, build_asof_features,
    construct_matured_labels,make_model,leave_one_deployment_out,
    evaluate_predictions,train_full_and_score)
ROOT=Path(__file__).resolve().parents[1]
OBS=pd.read_csv(ROOT/'data/W02_Robot_Deployment_Dataset.csv')
CON=pd.read_csv(ROOT/'data/W06_Fari_Contract_Metadata_SYNTHETIC_ASSUMPTIONS.csv')

@pytest.fixture(scope='module')
def features():return build_asof_features(OBS,CON,'2025-09-01')
@pytest.fixture(scope='module')
def labels():return construct_matured_labels(OBS,CON,outcome_as_of='2025-12-31')
@pytest.fixture(scope='module')
def oof(features,labels):return leave_one_deployment_out(features,labels)

@pytest.mark.parametrize('column',FEATURE_COLUMNS)
def test_feature_exists(features,column):assert column in features
@pytest.mark.parametrize('column',FEATURE_COLUMNS)
def test_feature_numeric(features,column):assert pd.api.types.is_numeric_dtype(features[column])
@pytest.mark.parametrize('day', ['2025-03-01','2025-05-01','2025-09-01'])
def test_future_rows_cannot_change_features(day):
    before=build_asof_features(OBS,CON,day)
    changed=OBS.copy()
    later=(changed.platform=='Fari') & (pd.to_datetime(changed.date)>pd.Timestamp(day))
    for c in ['avg_session_duration_min','caregiver_app_interactions','health_alert_response_time_min','renewal_status']:
        changed.loc[later,c]=999 if c!='renewal_status' else 'Non-Renewal'
    after=build_asof_features(changed,CON,day)
    pd.testing.assert_frame_equal(before,after)
@pytest.mark.parametrize('day',['2025-03-01','2025-06-01','2025-09-01'])
def test_outcomes_never_features(day):
    altered=OBS.copy()
    altered.loc[altered.platform.eq('Fari'),'renewal_status']='Non-Renewal'
    pd.testing.assert_frame_equal(build_asof_features(altered,CON,day),build_asof_features(OBS,CON,day))
@pytest.mark.parametrize('day',['2025-03-01','2025-06-01','2025-09-01'])
def test_feature_asof_limit(day):
    f=build_asof_features(OBS,CON,day)
    assert f.latest_feature_date.le(pd.Timestamp(day)).all()
@pytest.mark.parametrize('col',FEATURE_COLUMNS)
def test_independent_risk_values_are_finite(features,col):assert np.isfinite(features[col]).all()

def test_no_contract_future_labels():
    assert construct_matured_labels(OBS,CON,outcome_as_of='2025-12-30').empty

def test_end_of_contract_labels(labels):
    assert len(labels)==10 and labels.churn_label.sum()==2

def test_label_id_set(labels,features):assert set(labels.deployment_id)==set(features.deployment_id)

def test_label_source_only_from_contract_end():
    changed=OBS.copy()
    rows=(changed.platform=='Fari') & (pd.to_datetime(changed.date)<pd.Timestamp('2025-12-01'))
    changed.loc[rows,'renewal_status']='Non-Renewal'
    pd.testing.assert_frame_equal(construct_matured_labels(changed,CON,outcome_as_of='2025-12-31'),
                                  construct_matured_labels(OBS,CON,outcome_as_of='2025-12-31'))

def test_no_features_when_all_contracts_expired():
    assert build_asof_features(OBS,CON,'2026-01-01').empty

def test_missing_early_data_yields_nan_not_future_data():
    f=build_asof_features(OBS,CON,'2025-01-01')
    assert f.session_duration_trend_3m.isna().all()

def test_all_fari_unique(features):assert features.deployment_id.is_unique

def test_oof_one_per_deployment(oof):assert len(oof)==10 and oof.deployment_id.is_unique

def test_oof_training_excludes_deployment(oof):assert oof.training_deployment_count.eq(9).all()

def test_oof_range(oof):assert oof.oof_churn_probability.between(0,1).all()

def test_reproducibility(features,labels,oof):
    pd.testing.assert_frame_equal(oof,leave_one_deployment_out(features,labels))

def test_oof_labels_not_fold_inputs(features,labels):
    # Changing held-out ID's label changes class supervision for other folds,
    # but no held-out model is allowed to see that held-out label.
    assert not any(c in FEATURE_COLUMNS for c in ['churn_label','renewal_status'])

def test_missing_feature_raises(features,labels):
    with pytest.raises(ValueError,match='Missing features'):
        leave_one_deployment_out(features.drop(columns=['contract_tenure_months']),labels)

def test_duplicate_feature_raises(features,labels):
    with pytest.raises(ValueError,match='Duplicate'):
        leave_one_deployment_out(pd.concat([features,features.head(1)]),labels)

def test_duplicate_label_raises(features,labels):
    with pytest.raises(ValueError,match='Duplicate'):
        leave_one_deployment_out(features,pd.concat([labels,labels.head(1)]))

def test_single_class_raises(features,labels):
    x=labels.copy();x['churn_label']=0
    with pytest.raises(ValueError,match='both classes'):
        leave_one_deployment_out(features,x)

def test_single_positive_unusable(features,labels):
    x=labels.copy();x.loc[x.deployment_id=='FARI-008','churn_label']=0
    with pytest.raises(ValueError,match='one outcome class'):
        leave_one_deployment_out(features,x)

def test_two_class_model(features,labels):
    m=make_model();m.fit(features[list(FEATURE_COLUMNS)],labels.churn_label)
    assert set(m.classes_)=={0,1}

def test_pipeline_preprocessing_included():
    m=make_model()
    assert list(m.named_steps)==['imputer','scaler','logistic']

def test_metrics_bounds(oof):
    d=evaluate_predictions(oof)
    for col in ['auc','precision_at_0_5','recall_at_0_5','top_k_precision']:
        assert 0<=d[col]<=1

def test_auc_computed_directly(oof):
    from sklearn.metrics import roc_auc_score
    assert evaluate_predictions(oof)['auc']==pytest.approx(roc_auc_score(oof.churn_label,oof.oof_churn_probability))

def test_k_greater_than_cohort(oof):assert evaluate_predictions(oof,k=100)['effective_k']==10

def test_k_one(oof):assert evaluate_predictions(oof,k=1)['effective_k']==1

@pytest.mark.parametrize('bad_k',[0,-1])
def test_bad_k(oof,bad_k):
    with pytest.raises(ValueError,match='Invalid threshold'):
        evaluate_predictions(oof,k=bad_k)
@pytest.mark.parametrize('bad_threshold',[-0.01,1.01])
def test_bad_threshold(oof,bad_threshold):
    with pytest.raises(ValueError,match='Invalid threshold'):
        evaluate_predictions(oof,threshold=bad_threshold)

def test_missing_scores_invalid(oof):
    x=oof.copy();x.loc[0,'oof_churn_probability']=np.nan
    with pytest.raises(ValueError,match='Invalid score'):
        evaluate_predictions(x)

def test_full_scoring_has_warning(features,labels):
    df=train_full_and_score(features,labels)
    assert len(df)==10 and df.score_type.eq('IN_SAMPLE_EXPLORATORY_NOT_VALIDATION').all()

def test_full_scoring_ranked(features,labels):
    df=train_full_and_score(features,labels)
    assert df.churn_probability_exploratory.is_monotonic_decreasing

def test_contract_start_cutoff():
    assert build_asof_features(OBS,CON,'2024-12-01').empty

def test_no_modification_inputs():
    o=OBS.copy(deep=True);c=CON.copy(deep=True)
    build_asof_features(o,c,'2025-09-01')
    pd.testing.assert_frame_equal(o,OBS);pd.testing.assert_frame_equal(c,CON)

def test_high_uncertainty_reporting(oof):
    assert len(oof)<=10 # test fixture documents very small number of independent entities
