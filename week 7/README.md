# InGen Dynamics — Week 7 Integrated Robot Analytics Suite

**SYNTHETIC ANALYTICS DEMONSTRATION — not real InGen operational, customer, or financial records.**

This is the Week 7 integration of the existing Week 3 KPI/anomaly-detection pipeline, Week 5 expanding-window patrol forecasting model, and Week 6 Fari enterprise churn/retention risk module. It is designed to run from the extracted repository root without access to confidential data.

## Setup (Python 3.11–3.13 recommended)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run dashboards/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). The working directory must be the project root, which contains `src`, `config`, `data`, `dashboards`, and `tests`. If using an IDE, configure it to use `.venv`.

## Four tabs

| Tab | Analytic source and information |
|---|---|
| **1. Operational Overview** | KPI anomaly counts per platform from Week 3, one-month-ahead Aido coverage and 90% threshold from Week 5, and Fari At-Risk/Top-K counts from Week 6; finding-first summary. |
| **2. Anomaly Investigation** | Interactive platform, KPI and month filters; original Week 3 anomaly flags, alert severity, and ranked signed contribution analysis; applicable-KPI missingness. |
| **3. Forecasting** | Pooled Ridge one-month-ahead Aido prediction per deployment, proposed 90% intervention threshold, approximate 95% historical-OOS-error prediction bands, and Week 5 three-layer OOS MAE comparison. |
| **4. Churn & Retention Risk** | Week 6 Fari lifecycle classification, **separately labeled simulated** 30/60/90-day and 3/6/12-month contract-event retention, sample-level renewal labels (not a time trend), revenue at risk and Top-K intervention list. |

## Analytics and method

- `src/anomaly_detection.py`: original Week 3 configurable KPI, anomaly and contribution logic; `src/integration.py` calls it directly, **without reimplementing the underlying flags**. Week 7 flags are identical to source outputs.
- `src/forecasting.py`: original Week 5 baseline, Holt, Ridge OOS forecasting with strict prior-target-date training. `src/integration.py` fits an end-of-sample Ridge using all available **past** lag features, then produces a **January 2026** one-period-forward prediction (the sample dataset ends December 2025). The Week 5 OOS results support an approximate empirical 95% prediction band using residual quantiles, not a statistically calibrated interval.
- `src/churn_model.py`, `src/contract_events.py`: original Week 6 lifecycle, risk, residual LTV and simulated contract-event retention. Dashboard reads the supplied Week 6 exploratory fitted risk outputs; these are **in-sample**, not independently validated prospective risk probabilities. Real renewal-rate **time trends cannot be inferred** from static year-end labels.
- `src/integration.py`: pure integration layer with no Streamlit requirement, so the business computations can be tested independently from the UI.

## Required project structure

```text
W07_Integrated_Pipeline/
├── dashboards/app.py
├── src/anomaly_detection.py
├── src/forecasting.py
├── src/churn_model.py
├── src/contract_events.py
├── src/integration.py
├── config/W03_KPI_Framework.xlsx
├── data/ (all required simulated robot, contract and historical forecast data)
├── outputs/ (Week 6 simulated risk and targeting output)
├── tests/test_integration.py (93 parametrized integration checks)
├── tests/test_dashboard.py (11 page-logic smoke checks; UI adapter, not browser)
├── notebooks/ (original W03, W05, W06 notebooks)
├── docs/ (source model cards, Week 7 validation notes)
├── weekly/Wk-07-Log.md
├── requirements.txt
└── README.md
```

## Testing and operating constraints

`python -m pytest -q` runs all copied Week 3, Week 5 and Week 6 analytical tests, plus 93 Week 7 integration checks and 11 Week 7 presentation smoke checks. Tests include: exact original-vs-dashboard anomaly flags for 30 deployments; reproducibility of all three modules and end-to-end outputs; no future-date forecasting training; source-vs-view KPI equivalence; null Fari activity, zero anomaly, zero renewal and no at-risk cases; correct Top-K expected-loss ranking; under-three-second **backend computation + Tab 1 scorecard data preparation** in this test environment.

**Note:** A backend timing assertion cannot measure a real browser's Streamlit render time, which depends on installation, browser and hardware. A real browser runtime should be smoke-tested after installing Streamlit. The authoring environment did not have Streamlit installed and could not download it, so the browser UI is not claimed to have been tested end-to-end. Tests instead exercise chart creation and all four view functions using a no-op Streamlit UI adapter.

## Scientific limitations

- All robots, deployments, anomalies, contracts and financial results are synthetic; simulation assumptions are unapproved operational defaults.
- Week 6 used **10** independent Fari deployments. Performance estimates cannot be generalized to real deployments.
- Year-end `renewal_status` lacks dated events. The separate synthetic contract-event scenario is not observed retention, and no observed fleet renewal-rate trend is asserted.
- The 25% top-residual-LTV-quartile revenue-risk condition was **not achieved** in the original synthetic Week 6 scenario (about 8.37% within its At-Risk pool). Do not describe it as passing.
- Dashboard risk targeting is exploratory/in-sample; do not use these outputs to make binding commercial decisions.

See `docs/W07_Test_and_Methodology_Notes.md` for the final verification and provenance details.
