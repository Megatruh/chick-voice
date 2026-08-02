from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone 
from zoneinfo import ZoneInfo
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from simulation.config import DB_PATH, VALID_ZONES


class EventIn(BaseModel):
    node_id: str = Field(min_length=1, max_length=80)
    zone: Literal["A", "B", "C", "D"]
    classification: Literal["healthy", "unhealthy", "noise"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: str | None = None
    audio_file: str | None = Field(default=None, max_length=1000)
    source_label: str | None = Field(default=None, max_length=50)


class EventOut(EventIn):
    id: int


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                zone TEXT NOT NULL,
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                audio_file TEXT,
                source_label TEXT
            )
            """
        )
        connection.commit()


app = FastAPI(
    title="ChickVoice Integrated Simulation API",
    version="2.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    init_database()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": str(DB_PATH)}


@app.post("/events", response_model=EventOut)
def create_event(event: EventIn) -> EventOut:
    timestamp = (event.timestamp or datetime.now(ZoneInfo("Asia/Jakarta")).isoformat())

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    node_id, zone, classification, confidence,
                    timestamp, audio_file, source_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.node_id,
                    event.zone,
                    event.classification,
                    event.confidence,
                    timestamp,
                    event.audio_file,
                    event.source_label,
                ),
            )
            connection.commit()
            event_id = int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EventOut(
        id=event_id,
        node_id=event.node_id,
        zone=event.zone,
        classification=event.classification,
        confidence=event.confidence,
        timestamp=timestamp,
        audio_file=event.audio_file,
        source_label=event.source_label,
    )


@app.get("/events", response_model=list[EventOut])
def list_events(
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[EventOut]:
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, node_id, zone, classification, confidence,
                   timestamp, audio_file, source_label
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [EventOut(**dict(row)) for row in rows]


@app.get("/latest", response_model=dict[str, EventOut | None])
def latest_per_zone() -> dict[str, EventOut | None]:
    result: dict[str, EventOut | None] = {}
    with closing(get_connection()) as connection:
        for zone in VALID_ZONES:
            row = connection.execute(
                """
                SELECT id, node_id, zone, classification, confidence,
                       timestamp, audio_file, source_label
                FROM events
                WHERE zone = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (zone,),
            ).fetchone()
            result[zone] = EventOut(**dict(row)) if row else None
    return result


@app.delete("/events")
def clear_events() -> dict[str, int]:
    with closing(get_connection()) as connection:
        cursor = connection.execute("DELETE FROM events")
        connection.commit()
    return {"deleted": cursor.rowcount}
