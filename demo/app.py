"""Vietnamese Streamlit dashboard for the offline SMD monitoring demo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.alert_engine import status_at
from demo.artifact_store import ArtifactStore


KPI_NAMES = [f"KPI_{index:02d}" for index in range(1, 39)]
STATUS_COLORS = {
    "NORMAL": "#29d391",
    "WARNING": "#ffbf5b",
    "CRITICAL": "#ff5c7a",
    "RECOVERED": "#64a8ff",
}


def parse_artifact_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", type=Path, default=APP_DIR / "artifacts")
    args, _ = parser.parse_known_args()
    return args.artifacts.resolve()


@st.cache_resource
def get_store(path: str) -> ArtifactStore:
    return ArtifactStore(path)


@st.cache_data(show_spinner="Đang tải dữ liệu máy...")
def load_machine(path: str, backend: str, machine_id: str) -> dict:
    return get_store(path).load(backend, machine_id)


def contiguous_regions(labels: np.ndarray, start: int, end: int):
    visible = labels[start:end + 1].astype(bool)
    padded = np.pad(visible.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1) + start
    ends = np.flatnonzero(changes == -1) + start - 1
    return list(zip(starts, ends))


def style_figure(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=18, t=42, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,18,33,0.46)",
        font=dict(color="#dce8f6", family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.10)", title="Phút trong tập test")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.10)")
    return fig


def metric_value(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.3f}{suffix}"


def default_cursor(incidents, length):
    if incidents:
        return min(length - 1, max(96, int(incidents[0]["start"]) - 120))
    return min(length - 1, 96)


def sync_timeline_from_widget():
    """Copy a user slider change into the replay cursor before rendering."""
    st.session_state.timeline = st.session_state.timeline_widget


def render_app():
    st.set_page_config(
        page_title="UniTS Smart Monitor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .stApp {background: radial-gradient(circle at 12% 0%, #102941 0%, #071421 42%, #050c15 100%);}
          [data-testid="stSidebar"] {background: rgba(6, 18, 31, .96); border-right: 1px solid rgba(120,170,220,.14);}
          .hero {padding: 1.1rem 1.25rem; border: 1px solid rgba(105,181,255,.18); border-radius: 18px;
                 background: linear-gradient(120deg, rgba(18,52,82,.86), rgba(8,22,37,.72)); margin-bottom: 1rem;}
          .hero h1 {font-size: 1.65rem; margin: 0; color: #f3f8ff; letter-spacing: -.02em;}
          .hero p {margin: .35rem 0 0; color: #91a9c3;}
          .status-pill {display:inline-flex; align-items:center; gap:.5rem; padding:.42rem .75rem;
                        border-radius:999px; font-size:.76rem; font-weight:800; letter-spacing:.07em;}
          div[data-testid="stMetric"] {background: rgba(10,29,48,.72); border: 1px solid rgba(117,170,218,.14);
                                       padding: .8rem 1rem; border-radius: 14px;}
          div[data-testid="stMetricLabel"] {color:#8ea7c0;}
          .backend-badge {font-size:.72rem; font-weight:800; color:#66d9ff; background:rgba(24,155,201,.13);
                          padding:.3rem .55rem; border:1px solid rgba(76,201,240,.25); border-radius:8px;}
          .section-title {font-size:.82rem; text-transform:uppercase; letter-spacing:.1em; color:#7f9bb8;
                          margin:.8rem 0 .3rem; font-weight:750;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    artifact_root = parse_artifact_root()
    try:
        store = get_store(str(artifact_root))
        backends = store.backends()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python demo/generate_artifacts.py --backend baseline")
        st.stop()
    if not backends:
        st.error(f"Chưa có artifact trong {artifact_root}")
        st.code("python demo/generate_artifacts.py --backend baseline")
        st.stop()

    with st.sidebar:
        st.markdown("## ⚡ UniTS Monitor")
        st.caption("Server Machine Dataset · Offline replay")
        backend_labels = {value.upper(): value for value in backends}
        backend_label = st.selectbox("Backend", list(backend_labels))
        backend = backend_labels[backend_label]
        machine_ids = store.machines(backend)
        preferred = machine_ids.index("machine-1-1") if "machine-1-1" in machine_ids else 0
        machine_id = st.selectbox("Máy chủ", machine_ids, index=preferred)
        kpi = st.selectbox("KPI đang theo dõi", KPI_NAMES, index=8)
        speed_label = st.select_slider(
            "Tốc độ replay", options=["1×", "10×", "60×"], value="10×"
        )
        speed = int(speed_label.removesuffix("×"))
        evaluation_mode = st.toggle("Chế độ đánh giá", value=False,
                                    help="Hiển thị ground truth; nhãn không tham gia sinh cảnh báo.")
        st.divider()
        st.caption("Cửa sổ đầu vào: 96 phút")
        st.caption("Chân trời dự báo: 24 phút")
        st.caption("KPI được đánh số do SMD không công bố tên semantic.")

    selection = f"{backend}/{machine_id}"
    data = load_machine(str(artifact_root), backend, machine_id)
    actual = data["actual"]
    expected = data["expected"]
    scores = data["anomaly_score"]
    labels = data["ground_truth"]
    threshold = float(np.asarray(data["threshold"]).item())
    incidents = data["incidents"]
    metrics = data["metrics"]
    if st.session_state.get("selection") != selection:
        st.session_state.selection = selection
        st.session_state.timeline = default_cursor(incidents, len(actual))
        st.session_state.running = False

    st.markdown(
        f"""
        <div class="hero">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem">
            <div><h1>Trung tâm cảnh báo thông minh</h1>
            <p>{machine_id} · 38 KPI · dữ liệu lấy mẫu mỗi phút</p></div>
            <span class="backend-badge">{backend.upper()} ARTIFACT</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_cols = st.columns([1, 1, 1, 7])
    if control_cols[0].button("▶ Chạy", use_container_width=True):
        st.session_state.running = True
    if control_cols[1].button("⏸ Dừng", use_container_width=True):
        st.session_state.running = False
    if control_cols[2].button("↺ Reset", use_container_width=True):
        st.session_state.timeline = default_cursor(incidents, len(actual))
        st.session_state.running = False
    # Keep the replay cursor separate from the widget-owned key. Streamlit
    # forbids mutating a widget key after that widget has been instantiated.
    st.session_state.timeline_widget = st.session_state.timeline
    cursor = st.slider(
        "Dòng thời gian",
        min_value=96,
        max_value=len(actual) - 1,
        key="timeline_widget",
        on_change=sync_timeline_from_widget,
        format="T+%d phút",
    )

    current_score = float(scores[cursor])
    state = status_at(incidents, cursor, current_score, threshold)
    color = STATUS_COLORS[state]
    passed_incidents = [item for item in incidents if item["start"] <= cursor]
    anomaly_stats = metrics.get("anomaly", {})
    forecast_stats = metrics.get("forecast", {})

    status_col, score_col, incident_col, mae_col, f1_col = st.columns(5)
    status_col.markdown(
        f'<div class="section-title">Trạng thái</div><span class="status-pill" '
        f'style="color:{color};background:{color}1c;border:1px solid {color}55">● {state}</span>',
        unsafe_allow_html=True,
    )
    score_col.metric("Anomaly score", f"{current_score:.4f}",
                     delta=f"ngưỡng {threshold:.4f}", delta_color="off")
    incident_col.metric("Incident đã ghi nhận", len(passed_incidents))
    mae_col.metric("Forecast MAE", metric_value(forecast_stats.get("MAE")))
    f1_col.metric("Anomaly F1", metric_value(anomaly_stats.get("F1"), "%"))

    kpi_index = KPI_NAMES.index(kpi)
    history_start = max(0, cursor - 179)
    history_x = np.arange(history_start, cursor + 1)
    series_fig = go.Figure()
    series_fig.add_trace(go.Scatter(
        x=history_x, y=actual[history_start:cursor + 1, kpi_index],
        name="Quan sát", line=dict(color="#57c7ff", width=2.2),
    ))
    series_fig.add_trace(go.Scatter(
        x=history_x, y=expected[history_start:cursor + 1, kpi_index],
        name="Giá trị kỳ vọng", line=dict(color="#8b9bb0", width=1.5, dash="dot"),
    ))
    origins = data["forecast_origins"]
    valid_origins = np.flatnonzero(origins <= cursor)
    if len(valid_origins):
        forecast_index = int(valid_origins[-1])
        origin = int(origins[forecast_index])
        future_x = np.arange(origin + 1, origin + 1 + data["forecasts"].shape[1])
        series_fig.add_trace(go.Scatter(
            x=future_x, y=data["forecasts"][forecast_index, :, kpi_index],
            name="Dự báo 24 phút", line=dict(color="#ffbd59", width=2.4),
        ))
    if evaluation_mode:
        for region_start, region_end in contiguous_regions(labels, history_start, cursor):
            series_fig.add_vrect(x0=region_start, x1=region_end, fillcolor="#ff4f73",
                                 opacity=.12, line_width=0)
    series_fig.update_layout(title=f"{kpi}: quan sát, kỳ vọng và dự báo")
    style_figure(series_fig)

    score_fig = go.Figure()
    score_fig.add_trace(go.Scatter(
        x=history_x, y=scores[history_start:cursor + 1], name="Anomaly score",
        line=dict(color="#d17bff", width=2), fill="tozeroy", fillcolor="rgba(174,94,255,.10)",
    ))
    score_fig.add_hline(y=threshold, line_color="#ffbd59", line_dash="dash",
                        annotation_text="Threshold")
    score_fig.add_hline(y=1.5 * threshold, line_color="#ff5c7a", line_dash="dot",
                        annotation_text="Critical")
    if evaluation_mode:
        for region_start, region_end in contiguous_regions(labels, history_start, cursor):
            score_fig.add_vrect(x0=region_start, x1=region_end, fillcolor="#ff4f73",
                                opacity=.13, line_width=0)
    score_fig.update_layout(title="Điểm bất thường và ngưỡng cảnh báo")
    style_figure(score_fig, height=290)

    chart_col, alert_col = st.columns([2.15, 1])
    with chart_col:
        st.plotly_chart(series_fig, use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(score_fig, use_container_width=True, config={"displayModeBar": False})
    with alert_col:
        st.markdown('<div class="section-title">Phân tích nguyên nhân</div>', unsafe_allow_html=True)
        feature_scale = np.std(actual, axis=0) + 1e-6
        current_contribution = np.square((actual[cursor] - expected[cursor]) / feature_scale)
        top_indices = np.argsort(current_contribution)[::-1][:5]
        top_frame = pd.DataFrame({
            "KPI": [KPI_NAMES[index] for index in top_indices],
            "Đóng góp": [float(current_contribution[index]) for index in top_indices],
        })
        st.dataframe(top_frame, hide_index=True, use_container_width=True)
        if evaluation_mode:
            matching = [item for item in data["interpretation"] if item["start"] <= cursor <= item["end"]]
            if matching:
                dimensions = ", ".join(f"KPI_{value:02d}" for value in matching[0]["dimensions"])
                st.info(f"Ground truth dimensions: {dimensions}")
            elif labels[cursor]:
                st.info("Điểm này có nhãn bất thường nhưng không có interpretation dimensions.")
        st.markdown('<div class="section-title">Incident gần nhất</div>', unsafe_allow_html=True)
        recent = list(reversed(passed_incidents[-5:]))
        if recent:
            incident_frame = pd.DataFrame([{
                "Mức": item["severity"],
                "Bắt đầu": f"T+{item['start']}",
                "Thời lượng": f"{item['duration']}m",
                "KPI chính": ", ".join(item["top_kpis"][:2]),
            } for item in recent])
            st.dataframe(incident_frame, hide_index=True, use_container_width=True)
        else:
            st.caption("Chưa có incident tại thời điểm replay hiện tại.")

    heat_start = max(0, cursor - 59)
    residual = np.abs((actual[heat_start:cursor + 1] - expected[heat_start:cursor + 1]) / (
        np.std(actual, axis=0) + 1e-6
    ))
    heatmap = go.Figure(go.Heatmap(
        z=residual.T,
        x=np.arange(heat_start, cursor + 1),
        y=KPI_NAMES,
        colorscale=[[0, "#071421"], [.35, "#174b70"], [.7, "#c16af1"], [1, "#ff5c7a"]],
        colorbar=dict(title="|residual|"),
    ))
    heatmap.update_layout(title="Bản đồ residual 38 KPI · 60 phút gần nhất")
    style_figure(heatmap, height=520)
    st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})

    if st.session_state.running:
        if cursor >= len(actual) - 1:
            st.session_state.running = False
        else:
            time.sleep(0.8)
            st.session_state.timeline = min(len(actual) - 1, cursor + speed)
            st.rerun()


if __name__ == "__main__":
    render_app()
