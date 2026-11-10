"""Week 7 Streamlit presentation smoke tests with a no-op UI adapter.
These exercise all display paths and real Plotly chart construction without
claiming to have tested browser rendering or Streamlit hosting.
"""
from __future__ import annotations
from contextlib import nullcontext
import importlib
import sys
import types
import pandas as pd
import pytest

from src.integration import anomaly_bundle,forecast_bundle,churn_bundle

class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.calls=[]
        self.chart_calls=0
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def __getattr__(self,key):
        if key in {'spinner'}:
            return lambda *a,**kw:nullcontext()
        if key in {'columns','tabs'}:
            return lambda vals,*a,**kw:[self for _ in (range(vals) if isinstance(vals,int) else vals)]
        if key=='selectbox':
            return lambda label,options,**kw: list(options)[0]
        if key=='slider':
            return lambda label,min_value,max_value,value,**kw:value
        if key=='cache_data':
            return lambda **kw:lambda f:f
        if key=='plotly_chart':
            def plot(fig,*a,**kw):
                assert fig is not None and len(fig.data)>=1
                self.chart_calls+=1
            return plot
        return lambda *a,**kw:None

@pytest.fixture(scope='module')
def dashboard():
    stub=FakeStreamlit()
    old=sys.modules.get('streamlit')
    sys.modules['streamlit']=stub
    sys.modules.pop('dashboards.app',None)
    app=importlib.import_module('dashboards.app')
    yield app,stub
    sys.modules.pop('dashboards.app',None)
    if old is None:sys.modules.pop('streamlit',None)
    else:sys.modules['streamlit']=old

@pytest.fixture(scope='module')
def data():return anomaly_bundle(),forecast_bundle(),churn_bundle()

def test_app_syntax_and_imports(dashboard):
    app,ui=dashboard
    assert hasattr(app,'main') and callable(app.load_analytics)

def test_dashboard_loads_three_source_modules(dashboard,data):
    app,_=dashboard
    a,f,c=app.load_analytics()
    assert len(a['raw'])==len(data[0]['raw'])
    assert len(f['oos'])==len(data[1]['oos'])
    assert len(c['top_k'])==len(data[2]['top_k'])

def test_overview_renders_real_chart(dashboard,data):
    app,ui=dashboard
    before=ui.chart_calls
    app.operational_overview(*data)
    assert ui.chart_calls>before

def test_anomaly_investigation_renders_attribution(dashboard,data):
    app,ui=dashboard
    before=ui.chart_calls
    app.anomaly_investigation(data[0])
    assert ui.chart_calls>before

def test_forecast_renders_with_bands(dashboard,data):
    app,ui=dashboard
    before=ui.chart_calls
    app.forecasting_tab(data[1])
    assert ui.chart_calls>before

def test_churn_renders_retention_and_top_k(dashboard,data):
    app,ui=dashboard
    before=ui.chart_calls
    app.churn_tab(data[2],data[0])
    assert ui.chart_calls>=before+3

def test_full_main_has_four_tabs_and_renders(dashboard):
    app,ui=dashboard
    app.main()
    assert ui.chart_calls>=8

@pytest.mark.parametrize('fn',['operational_overview','anomaly_investigation','forecasting_tab','churn_tab'])
def test_four_sections_exist(dashboard,fn):
    app,_=dashboard
    assert callable(getattr(app,fn))
