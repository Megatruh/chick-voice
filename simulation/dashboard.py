from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from simulation.config import API_BASE_URL

st.set_page_config(
    page_title="ChickVoice Dashboard",
    page_icon="🐔",
    layout="wide",
)

st_autorefresh(interval=2000, key="chickvoice-refresh")

STATUS_LABEL = {
    "healthy": "SEHAT",
    "unhealthy": "WASPADA",
    "noise": "NOISE",
}

STATUS_COLOR = {
    "healthy": "#1f9d55",
    "unhealthy": "#d64545",
    "noise": "#d99a20",
    "empty": "#7a7a7a",
}


def get_json(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=3)
    response.raise_for_status()
    return response.json()


def zone_card(zone: str, event: dict[str, Any] | None) -> None:
    if event is None:
        status = "empty"
        label = "BELUM ADA DATA"
        confidence = "-"
        node_id = "-"
        timestamp = "-"
        audio_name = "-"
    else:
        status = event["classification"]
        label = STATUS_LABEL.get(status, status.upper())
        confidence = f'{event["confidence"]:.1%}'
        node_id = event["node_id"]
        timestamp = event["timestamp"].replace("T", " ")[:19]
        audio_name = event.get("audio_file") or "-"

    color = STATUS_COLOR.get(status, STATUS_COLOR["empty"])

    st.markdown(
        f"""
        <div style="
            border: 3px solid {color};
            border-radius: 14px;
            padding: 18px;
            min-height: 210px;
            background: rgba(255,255,255,0.03);
        ">
            <h2 style="margin: 0;">Zona {zone}</h2>
            <h3 style="color: {color}; margin: 14px 0 8px 0;">{label}</h3>
            <div><b>Confidence:</b> {confidence}</div>
            <div><b>Node:</b> {node_id}</div>
            <div><b>Waktu:</b> {timestamp}</div>
            <div style="word-break: break-word;"><b>Audio:</b> {audio_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("ChickVoice Early Warning System")
st.caption(
    "Dataset WAV → MFCC → 1D-CNN → REST API → SQLite → dashboard zona"
)

try:
    health = get_json("/health")
    st.success(f"API aktif. Database: {health['database']}")
except requests.RequestException:
    st.error(
        "API belum aktif. Jalankan `python -m uvicorn simulation.api:app --reload`."
    )
    st.stop()

latest = get_json("/latest")
events = get_json("/events?limit=100")

st.subheader("Heatmap zona kandang")
row1 = st.columns(2)
with row1[0]:
    zone_card("A", latest.get("A"))
with row1[1]:
    zone_card("B", latest.get("B"))

row2 = st.columns(2)
with row2[0]:
    zone_card("C", latest.get("C"))
with row2[1]:
    zone_card("D", latest.get("D"))

st.subheader("Ringkasan 100 event terakhir")

if events:
    df = pd.DataFrame(events)
    counts = df["classification"].value_counts()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total event", len(df))
    m2.metric("Sehat", int(counts.get("healthy", 0)))
    m3.metric("Waspada", int(counts.get("unhealthy", 0)))
    m4.metric("Noise", int(counts.get("noise", 0)))

    display_df = df[
        [
            "id",
            "node_id",
            "zone",
            "source_label",
            "classification",
            "confidence",
            "audio_file",
            "timestamp",
        ]
    ].copy()

    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.1%}")
    display_df["classification"] = display_df["classification"].map(
        lambda x: STATUS_LABEL.get(x, x)
    )
    display_df.columns = [
        "ID",
        "Node",
        "Zona",
        "Label sumber",
        "Prediksi",
        "Confidence",
        "File audio",
        "Timestamp UTC+7",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info(
        "Belum ada event. Jalankan `python -m simulation.audio_simulator "
        "--scenario outbreak --interval 2`."
    )
