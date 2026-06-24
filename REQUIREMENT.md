# Requirements

## 1. Multi-Source Time Series Dataset

The project must provide a multi-source time series dataset that includes:

- KPI time series data.
- Alarm and event logs.
- Ticket or text-based incident records.
- Relevant metadata for services, systems, nodes, or entities.
- Anomaly labels or weak labels for training and evaluation.
- A complete data processing pipeline, including data cleaning, alignment, normalization, feature construction, and train/validation/test splitting.

## 2. Multi-Task Foundation Model

The project must develop a foundation model for time series that supports multiple tasks using a shared backbone, including:

- Forecasting.
- Anomaly detection.
- Imputation.
- Event-aware reasoning that incorporates alarm, event, ticket, or text information into the model's predictions and decisions.

## 3. Calibrated Uncertainty Mechanism

The model must provide calibrated uncertainty estimates for its outputs. This includes:

- Prediction intervals.
- Quantile forecasts or probabilistic forecast distributions.
- Calibration methods that ensure alerts and predictions have clear confidence levels.
- Uncertainty-aware alerting so that the system can distinguish high-confidence alerts from uncertain or ambiguous cases.

## 4. Benchmark and Evaluation Metrics

The project must include a benchmark suite and evaluation protocol covering all supported tasks.

Forecasting metrics must include:

- MAE.
- RMSE.
- sMAPE.

Anomaly detection metrics must include:

- Precision.
- Recall.
- F1-score.
- AUC-ROC.
- AUC-PR.

Uncertainty evaluation metrics must include:

- Coverage.
- Interval width.
- Calibration error.
- False alarm rate.

## 5. Error Analysis Report

The project must provide an error analysis report that investigates:

- Cases where the model produces inaccurate forecasts.
- Cases with high uncertainty.
- False positive anomaly alerts.
- False negative missed anomalies.
- The impact of text, ticket, alarm, and event information on model performance and decision quality.

## 6. Intelligent Alerting System Demo

The project must include a demo dashboard for an intelligent alerting system. The dashboard should display:

- Forecasted time series values.
- Confidence intervals or uncertainty bands.
- Anomaly scores.
- Related alarms, events, tickets, or text context.
- Recommended incident priority or handling level.
