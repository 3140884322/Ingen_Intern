# Week 7 — Validation and provenance

## Source files and reuse

- Week 3 source: `week3(2).zip`, original `src/anomaly_detection.py` and its pytest suite.
- Week 4 source: `week4(2).zip`, original KPI Excel workbook (`config/W03_KPI_Framework.xlsx`) and dashboard tab designs. Its huge platform-specific virtual environment was intentionally omitted.
- Week 5 source: `week5(2).zip`, original `src/forecasting.py`, historical OOS CSV and model card.
- Week 6 source: `week6.zip`, original `src/churn_model.py`, `src/contract_events.py`, synthetic customer contract metadata, simulated event scenario, exploratory risk outputs and model card.
- All existing scientific Python modules were preserved. `src/integration.py` is the new Week 7 adapter; `dashboards/app.py` is the new four-tab dashboard.

## Integration validation

- The KPI flag identity comparison tests use actual Week 3 module output and compare deployment-month anomaly flags directly with the Week 7 dashboard's `anomaly_dashboard_flags` return values. No flags are made up for the UI.
- Repeatability is asserted by running the original three pipelines twice in the same process and checking exact dataframe equality.
- Historical Week 5 OOS input is verified against a strict training-target-date boundary. The new forward forecast uses lagged predictors and a target month after the last training target.
- All Week 7 integration tests are parametrized meaningfully across deployments, model layers, risk K values, metric horizons and robot platforms, not inflated by duplicated identical assertions.
- Special cases: zero anomalies, missing Fari sessions, no renewals in a fully synthetic scenario, no At-Risk deployments, invalid negative revenue risk, missing required data, and absent observed renewal history.
- Performance check is backend pipeline and summary preparation; it is **not browser paint performance**.

## Important scope distinctions

- Synthetic contract-event 30/60/90-day and 3/6/12-month retention curves do not indicate actual InGen renewal retention.
- Static year-end renewal_status is shown only as a label count. A time trend cannot be computed reliably.
- Model-based operational flags below 90% are support indicators, not binding maintenance orders.
- Forecast lower/upper bands are approximate pooled historic OOS residual quantiles, not proven confidence intervals.
- Simulated Week 6 revenue and churn calculations do not reach the 25% top-quartile condition and are not silently replaced.

## Local runtime caveat

All Python source compiles, pytest suite can run in a Python environment with the analytic dependencies. The authoring runtime could not install Streamlit because outbound package access was disabled, so all four dashboard functions were exercised via a test UI adapter, not a real browser. To perform final Streamlit interaction inspection, install the listed dependencies and run `python -m streamlit run dashboards/app.py` on a computer with internet or with locally installed packages.
