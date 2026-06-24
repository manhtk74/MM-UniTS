# UniTS-Compatible Multi-Source Data and Event-Aware Extension Plan

## 1. Goal

This plan adapts KPI time series, alarm/event logs, ticket text, metadata, anomaly labels, and weak labels into a UniTS-based multi-task foundation model.

The goal is not only to improve preprocessing. It is an architecture and data-interface extension of UniTS so that the same backbone can support:

- Forecasting.
- Anomaly detection.
- Imputation.
- Event-aware reasoning.
- Calibrated uncertainty for alerting.

## 2. Key Compatibility Constraints With Current UniTS

The current UniTS implementation has several assumptions that should be respected in the first version:

- Time series input is a regular tensor with shape `(B, T, V)`, where `B` is batch size, `T` is sequence length, and `V` is the number of variables.
- After tokenization, UniTS represents data as `(B, V, L, D)`, where `L` is the number of patch tokens and `D` is `d_model`.
- Patch embedding currently assumes non-overlapping patches, so `patch_len == stride`.
- Existing tasks use task prompts and shared backbone blocks.
- Current anomaly detection is reconstruction-based, using the forecast head to reconstruct the input window.
- Current data loaders return regular time series tensors and time marks only; they do not yet return event sequences or text embeddings.

Therefore, the first compatible version should avoid changing the core patching and forecast reconstruction logic too much.

## 3. Recommended Design Principle

Do not force alarm, event, or ticket data into KPI channels at the raw input level.

Instead:

1. Keep KPI data on a regular time grid.
2. Patch KPI data using the existing UniTS patch mechanism.
3. Encode events and tickets as a separate variable-length token sequence.
4. Fuse event tokens into KPI patch tokens before the shared UniTS backbone.
5. Preserve the original UniTS task heads where possible.

This keeps UniTS as the main backbone while adding an event-aware adapter.

## 4. Data Format

### 4.1 KPI Table

The KPI data should remain a regular multivariate time series:

```text
timestamp | entity_id | cpu_util | latency_ms | throughput_mbps | packet_loss | ...
```

Expected tensor after windowing:

```python
x_kpi.shape = (B, T, V)
```

Where:

- `T` is the input window length.
- `V` is the number of KPI variables.
- The time grid must be regular, for example every 1 minute, 5 minutes, or 15 minutes.

### 4.2 Event and Alarm Table

Events should remain irregular records:

```text
event_id | entity_id | start_time | end_time | event_type | severity | source_text
```

Each event should be encoded as one token, not one channel.

Recommended event features:

- Event type embedding.
- Severity embedding.
- Source embedding, for example `ALARM`, `LOG`, `TICKET`, `CHANGE`.
- Relative start time to the KPI window.
- Relative end time or duration.
- Optional text embedding from event description or log message.
- Optional entity metadata embedding.

Expected event tensors after batching:

```python
event_tokens.shape = (B, E_max, D)
event_mask.shape = (B, E_max)
event_start_time.shape = (B, E_max)
event_end_time.shape = (B, E_max)
```

Where:

- `E_max` is the maximum number of events in the batch after padding or top-k filtering.
- `event_mask` marks valid event tokens.

### 4.3 Ticket Text

Tickets can be treated as delayed event tokens:

```text
ticket_id | entity_id | created_at | priority | text
```

Recommended handling:

- Encode each ticket as an event-like token.
- Set `source = TICKET`.
- Use `created_at` as the event timestamp.
- If the ticket text mentions an estimated incident time, store it separately as `mentioned_time` when available.
- In real-time forecasting or alerting, do not use tickets created after the forecast origin.

### 4.4 Metadata

Metadata can be used in two ways:

- Static entity embedding: service type, node type, region, vendor, topology group.
- Event-token feature: attach entity metadata to each event token.

For the first version, metadata should be added to event tokens and optionally to KPI variables through a simple entity embedding.

## 5. Dataset Loader Output

A UniTS-compatible multi-source dataset should return a dictionary instead of only tuple outputs.

Recommended sample format:

```python
sample = {
    "x_kpi": x_kpi,                       # (T, V)
    "y_kpi": y_kpi,                       # forecasting target when needed
    "x_mark": x_mark,                     # existing UniTS time marks
    "y_mark": y_mark,
    "event_features": event_features,     # raw event feature ids or precomputed vectors
    "event_mask": event_mask,             # (E_max,)
    "event_start": event_start,           # (E_max,)
    "event_end": event_end,               # (E_max,)
    "anomaly_label": anomaly_label,       # optional, shape depends on task
    "weak_label": weak_label,             # optional
    "metadata": metadata,                 # optional
}
```

The training loop can then route the fields into UniTS and the event adapter.

For backward compatibility, old datasets can still return the original UniTS tuple format.

## 6. Model Integration Point

The safest first integration point is:

```text
x_kpi
  -> UniTS tokenize()
  -> KPI patch tokens: (B, V, L, D)
  -> EventFusionAdapter
  -> fused KPI patch tokens: (B, V, L, D)
  -> prepare_prompt()
  -> UniTS backbone()
  -> existing task heads
```

This is more compatible than changing the raw input channel format.

## 7. EventFusionAdapter

### 7.1 Input and Output

The adapter should take:

```python
kpi_tokens.shape = (B, V, L, D)
event_tokens.shape = (B, E, D)
event_mask.shape = (B, E)
patch_start.shape = (B, L)
patch_end.shape = (B, L)
event_start.shape = (B, E)
event_end.shape = (B, E)
```

It should return:

```python
fused_tokens.shape = (B, V, L, D)
```

### 7.2 Attention Direction

Use KPI patch tokens as queries and event tokens as keys/values:

```text
Q = KPI patch token
K = event token
V = event token
```

This keeps the output shape identical to UniTS token shape.

### 7.3 Handling UniTS Variable Dimension

Because UniTS keeps a separate variable dimension, there are two practical options:

#### Option A: Broadcast Event Context to All Variables

Each KPI variable attends to the same relevant event tokens.

Pros:

- Simple.
- Compatible with current UniTS tensor shape.
- Good MVP.

Cons:

- Does not know which event affects which KPI variable unless learned indirectly.

#### Option B: Variable-Aware Event Fusion

Add variable or entity embeddings so each KPI variable can attend differently to events.

Pros:

- Better for root cause and event-aware reasoning.
- More expressive.

Cons:

- Requires a mapping between KPI variables, entities, services, and events.
- More data engineering.

Recommended first version: Option A.

Recommended later version: Option B.

### 7.4 Relative Time Bias

Add a deterministic time-distance bias to attention scores:

```python
distance = distance_between_patch_and_event(patch_start, patch_end, event_start, event_end)
time_bias = -alpha * distance
attention_score = qk_score + time_bias
```

Where:

- `distance = 0` if the event overlaps the patch.
- `distance > 0` if the event is before or after the patch.
- `alpha` can be learnable or fixed.

This is important because Time2Vec or timestamp embeddings alone do not guarantee correct alignment between short events and longer KPI patches.

### 7.5 Event Filtering

Before attention, filter or mask events by:

- Time overlap with the KPI window.
- Buffer before the window for early symptoms.
- Causal constraint for forecasting and real-time alerting.
- Top-k severity or rarity when there is an alarm storm.

For the first version, use simple top-k filtering:

```text
priority score = severity weight + rarity weight + recency weight
```

## 8. Task-Specific Rules

### 8.1 Forecasting

For forecasting, prevent future leakage.

Allowed event information:

- Events that happened before or at the forecast origin.
- Tickets created before or at the forecast origin.
- Historical alarms in the input window.

Not allowed:

- Events from the prediction horizon.
- Tickets created after the forecast origin.
- Labels derived from future incidents.

Output:

- Keep the current UniTS forecast head for point forecasting.
- Add optional quantile or distribution head later for uncertainty.

### 8.2 Anomaly Detection

For historical offline anomaly detection:

- Bidirectional event context can be used if the evaluation setting allows it.

For real-time anomaly detection:

- Use causal event context only.

Output options:

- MVP: use reconstruction error from the existing UniTS anomaly detection path.
- Improved version: add an anomaly score head over fused patch tokens.

### 8.3 Imputation

For imputation:

- Event context may be bidirectional if the task is offline.
- If the task simulates real-time missing-value recovery, use only past events.

Output:

- Keep the existing UniTS imputation path.
- Use event fusion before reconstruction.

### 8.4 Event-Aware Reasoning

This task is not currently native to UniTS.

It needs a new head and labels, such as:

- Incident priority classification.
- Root-cause category classification.
- Event relevance ranking.
- Explanation text generation outside UniTS, using retrieved events and model scores.

Recommended first version:

- Add a classification head for incident priority or root-cause category.
- Do not add free-form generation inside UniTS.

## 9. Uncertainty

The least disruptive uncertainty design is post-hoc calibration.

Recommended MVP:

- Train UniTS for point forecasting.
- Compute residuals on the validation set.
- Use conformal prediction to produce prediction intervals.
- Evaluate coverage, interval width, calibration error, and false alarm rate.

Recommended later version:

- Add quantile heads for `q10`, `q50`, `q90`.
- Train with pinball loss.
- Calibrate quantiles using validation data.

For anomaly alerts:

- Combine anomaly score with calibrated confidence.
- Report alert level as `low`, `medium`, `high`, or `critical`.

## 10. Benchmark Design

The benchmark should compare:

1. KPI-only UniTS.
2. UniTS with grid-based event count features.
3. UniTS with event-token fusion.
4. UniTS with event-token fusion and relative time bias.
5. UniTS with event-token fusion, relative time bias, and uncertainty calibration.

Forecasting metrics:

- MAE.
- RMSE.
- sMAPE.

Anomaly detection metrics:

- Precision.
- Recall.
- F1-score.
- AUC-ROC.
- AUC-PR.
- False alarm rate.

Uncertainty metrics:

- Coverage.
- Average interval width.
- Calibration error.
- Risk-coverage curve.

Event-aware reasoning metrics:

- Accuracy or macro-F1 for priority/root-cause classification.
- Recall@K for related-event retrieval.
- Ablation impact of text/event features.

## 11. Error Analysis

The report should include:

- Forecasting errors during high event density.
- False positive alerts caused by noisy alarm storms.
- False negative anomalies with weak or missing event signals.
- Cases where event text improves the model.
- Cases where ticket delay causes leakage risk or stale context.
- High-uncertainty cases and whether uncertainty matches actual error.

Useful plots:

- KPI forecast with confidence interval.
- Anomaly score over time.
- Event timeline aligned with KPI patches.
- Attention or relevance score from patches to events.

## 12. Demo Dashboard

The demo should show:

- Historical KPI values.
- Forecasted KPI values.
- Prediction intervals.
- Anomaly score.
- Related events, alarms, and tickets.
- Alert confidence.
- Recommended incident priority.
- Top contributing event tokens or retrieved related events.

For the first demo, the recommendation can be rule-assisted:

```text
priority = function(anomaly_score, uncertainty, severity, event_count, affected_kpi)
```

This is easier to validate than a fully learned decision system.

## 13. Implementation Roadmap

### Phase 1: Data Interface

- Create a multi-source dataset loader.
- Keep KPI tensor compatible with UniTS.
- Add event/ticket sequence fields.
- Add event masks and relative timestamps.
- Add anomaly or weak labels.

### Phase 2: Event Encoder

- Implement event type embedding.
- Implement severity/source embedding.
- Add time features or Time2Vec.
- Add optional frozen text embeddings.
- Project all features to `d_model`.

### Phase 3: Event Fusion Adapter

- Add cross-attention from KPI patch tokens to event tokens.
- Preserve output shape `(B, V, L, D)`.
- Add event mask.
- Add relative time bias.
- Add causal masking for forecasting.

### Phase 4: UniTS Integration

- Insert fusion after `tokenize()` and before `prepare_prompt()`.
- Keep original task heads for forecasting, anomaly detection, and imputation.
- Add a simple classification head for event-aware reasoning if labels are available.

### Phase 5: Uncertainty

- Start with conformal prediction on validation residuals.
- Add quantile head only after the point forecast baseline is stable.

### Phase 6: Evaluation and Demo

- Run ablation benchmarks.
- Produce error analysis.
- Build dashboard using saved predictions, intervals, anomaly scores, and related events.

## 14. What Should Not Be Done in the First Version

- Do not use overlapping patches unless the UniTS patch and forecast reconstruction logic are refactored.
- Do not convert every event type into a KPI channel.
- Do not use future events or future tickets in forecasting.
- Do not build a free-form text generation head inside UniTS initially.
- Do not replace the UniTS backbone before testing an adapter-based approach.

## 15. Summary

The most UniTS-compatible direction is:

```text
regular KPI tensor
  -> existing UniTS patch tokenizer
  -> event-token cross-attention adapter
  -> existing UniTS prompt and backbone
  -> existing task heads plus optional event reasoning head
  -> post-hoc uncertainty calibration
```

This preserves the strengths of UniTS while adding multi-source event awareness in a controlled and testable way.
