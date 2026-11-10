"""Four-tab InGen Week 7 Streamlit app. All observations/contracts are SYNTHETIC.
Run: streamlit run dashboards/app.py
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.integration import (anomaly_bundle,forecast_bundle,churn_bundle,executive_summary,
  anomaly_dashboard_flags,anomaly_contributions,quality_checks,renewal_trend)
st.set_page_config(page_title='InGen Integrated Robot Analytics',page_icon='🤖',layout='wide')

@st.cache_data(show_spinner=False)
def load_analytics():
    return anomaly_bundle(),forecast_bundle(),churn_bundle()

def operational_overview(a,f,c):
    st.subheader('Operational Overview')
    summary=executive_summary(a,f,c)
    counts=summary['anomaly_counts']
    m1,m2,m3,m4=st.columns(4)
    m1.metric('Anomaly flags (deployment-month)',int(counts.anomaly_count.sum()) if len(counts) else 0)
    m2.metric('Next-month Aido coverage',f"{summary['patrol_forecast_mean']:.1f}%")
    m3.metric('Aido below 90%',summary['forecast_below_90_count'])
    m4.metric('Fari At-Risk',summary['fari_at_risk_count'])
    if not counts.empty:
        st.plotly_chart(px.bar(counts,x='platform',y='anomaly_count',color='platform',
             title='Synthetic deployment-month anomalies by robot platform'),use_container_width=True)
    x,y=st.columns(2)
    with x:
        st.markdown('#### Priority retention contacts')
        st.dataframe(c['top_k'][['intervention_rank','deployment_id','expected_revenue_at_risk_usd']].head(5),hide_index=True)
    with y:
        st.markdown('#### Patrol intervention alerts')
        flagged=f['next'].loc[f['next'].intervention_flag]
        if flagged.empty:st.info('No deployments currently forecast below the assumed 90% threshold.')
        else:st.dataframe(flagged[['deployment_id','forecast_date','patrol_coverage_forecast']],hide_index=True)
    st.caption('Churn scores are in-sample exploratory outputs. Fleet data, contracts and thresholds are simulated.')

def anomaly_investigation(a):
    st.subheader('Anomaly Investigation')
    platform=st.selectbox('Robot platform',sorted(a['raw'].platform.unique()),key='a_platform')
    alerts=a['alerts'].loc[a['alerts'].platform.eq(platform)]
    if alerts.empty:st.info('No anomaly events for this platform.');return
    kpi=st.selectbox('KPI',sorted(alerts.kpi_id.unique()),key='a_kpi')
    dates=sorted(alerts.loc[alerts.kpi_id.eq(kpi),'date'].unique(),reverse=True)
    date=pd.Timestamp(st.selectbox('Anomalous month',dates,key='a_date'))
    current=alerts.loc[alerts.kpi_id.eq(kpi)&alerts.date.eq(date)]
    st.dataframe(current[['deployment_id','date','kpi_id','severity','deviation_magnitude','primary_contributing_subsegment']],hide_index=True)
    contrib=anomaly_contributions(a,platform,date,kpi)
    st.plotly_chart(px.bar(contrib,x='deployment_id',y='signed_contribution',color='signed_contribution',
           title=f'{platform}: ranked deployment contributions for {kpi}'),use_container_width=True)
    st.caption('Contribution estimates are descriptive, not causal. All flags come from src/anomaly_detection.py.')
    st.markdown('#### Cross-module flags (unmodified from anomaly module)')
    flags=anomaly_dashboard_flags(a)
    st.dataframe(flags.loc[flags.platform.eq(platform)&flags.date.eq(date)],hide_index=True)
    st.markdown('#### Data quality')
    st.dataframe(quality_checks(a).loc[lambda d:d.platform.eq(platform)],hide_index=True)

def forecasting_tab(f):
    st.subheader('Forecasting — Aido Rover')
    nxt=f['next']
    a,b,c=st.columns(3)
    a.metric('Next-month mean',f"{nxt.patrol_coverage_forecast.mean():.2f}%")
    b.metric('Interventions below 90%',int(nxt.intervention_flag.sum()))
    c.metric('Historical OOS Ridge MAE',f"{f['metrics'].iloc[2]['MAE']:.2f} pp")
    chart=go.Figure()
    chart.add_trace(go.Scatter(x=nxt.deployment_id,y=nxt.patrol_coverage_forecast,mode='lines+markers',name='Next-month Ridge forecast',
       error_y=dict(type='data',symmetric=False,array=nxt.illustrative_upper-nxt.patrol_coverage_forecast,
                    arrayminus=nxt.patrol_coverage_forecast-nxt.illustrative_lower)))
    chart.add_hline(y=90,line_dash='dash',annotation_text='90% service threshold')
    chart.update_layout(title='Next-month patrol coverage vs proposed service threshold',yaxis_title='Patrol coverage (%)')
    st.plotly_chart(chart,use_container_width=True)
    st.caption('Approximate 95% prediction band uses 2.5th and 97.5th percentiles of historical OOS actual-minus-prediction errors, clipped to [0,100]. Illustrative; not formally calibrated.')
    st.dataframe(nxt[['deployment_id','forecast_date','patrol_coverage_forecast','illustrative_lower','illustrative_upper','intervention_flag']],hide_index=True)
    st.plotly_chart(px.bar(f['metrics'],x='model',y='MAE',title='Expanding-window out-of-sample MAE by model layer'),use_container_width=True)
    st.dataframe(f['metrics'],hide_index=True)

def churn_tab(c,a):
    st.subheader('Churn & Retention Risk — Fari')
    l=c['lifecycle'];risk=c['revenue'];ranked=c['top_k']
    trend=renewal_trend(a['raw'])
    x,y,z=st.columns(3)
    x.metric('At-Risk accounts',int(l.lifecycle_stage.eq('At-Risk').sum()))
    y.metric('Expected revenue at risk',f"${risk.expected_revenue_at_risk_usd.sum():,.0f}")
    z.metric('Final synthetic renewal labels',f"{trend['renewals']}/{trend['total']}")
    st.info('True historical renewal-rate trend is unavailable: Week 2 contains no timestamped renewal events. The curves below are a separate, labeled SYNTHETIC contract-event scenario.')
    retention=c['retention']
    for title,order in [('30/60/90-day retention (scenario)',['30d','60d','90d']),
                        ('3/6/12-month retention (scenario)',['3m','6m','12m'])]:
        subset=retention.set_index('horizon').reindex(order).reset_index()
        st.plotly_chart(px.line(subset,x='horizon',y='retention_rate',color='assumed_start_cohort',markers=True,
             title=title,range_y=[0,1.05]),use_container_width=True)
    st.markdown('#### Top-K quarterly outreach by expected revenue loss')
    k=st.slider('Account manager outreach capacity (K)',1,max(1,len(ranked)),min(5,len(ranked)))
    selected=ranked.head(k)
    st.plotly_chart(px.bar(selected,x='deployment_id',y='expected_revenue_at_risk_usd',
              title=f'Top {k} expected revenue-at-risk accounts'),use_container_width=True)
    st.dataframe(selected[['intervention_rank','deployment_id','churn_probability_exploratory','expected_revenue_at_risk_usd','residual_ltv_usd']],hide_index=True)
    st.markdown('#### Lifecycle classification')
    st.dataframe(l,hide_index=True)
    st.caption('Original synthetic data did not meet the 25% top-LTV-quartile risk-concentration assertion; results are not manipulated to pass.')

def main():
    st.title('🤖 InGen Dynamics | Integrated Robot Analytics Suite')
    st.warning('SYNTHETIC DEMO — NOT actual InGen operational, financial or customer data.')
    with st.spinner('Loading three analytics modules...'):
        a,f,c=load_analytics()
    tabs=st.tabs(['1 · Operational Overview','2 · Anomaly Investigation','3 · Forecasting','4 · Churn & Retention Risk'])
    with tabs[0]:operational_overview(a,f,c)
    with tabs[1]:anomaly_investigation(a)
    with tabs[2]:forecasting_tab(f)
    with tabs[3]:churn_tab(c,a)
if __name__=='__main__':main()
