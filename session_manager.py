"""
Patient Session History Manager
Persistent storage for prediction sessions without ML modifications.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os

SESSION_DB = "data/session_history.csv"

def ensure_session_db():
    """Create session database if it doesn't exist."""
    Path("data").mkdir(exist_ok=True)
    if not os.path.exists(SESSION_DB):
        columns = [
            "session_id", "timestamp", "patient_id", "predicted_gesture",
            "confidence", "rows_analyzed", "accuracy", "session_duration_sec"
        ]
        pd.DataFrame(columns=columns).to_csv(SESSION_DB, index=False)

def save_session(
    patient_id: str,
    predicted_gesture: str,
    confidence: float,
    rows_analyzed: int,
    accuracy: float = None,
    session_duration_sec: float = None,
) -> str:
    """
    Save a prediction session to persistent storage.
    Returns session_id.
    """
    ensure_session_db()

    df = pd.read_csv(SESSION_DB)
    session_id = f"S{len(df) + 1:06d}"

    new_row = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "predicted_gesture": predicted_gesture,
        "confidence": round(confidence, 2),
        "rows_analyzed": int(rows_analyzed),
        "accuracy": round(accuracy, 4) if accuracy else None,
        "session_duration_sec": round(session_duration_sec, 2) if session_duration_sec else None,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(SESSION_DB, index=False)
    return session_id

def get_patient_sessions(patient_id: str) -> pd.DataFrame:
    """Get all sessions for a specific patient."""
    ensure_session_db()
    df = pd.read_csv(SESSION_DB)
    return df[df["patient_id"] == patient_id].copy()

def get_all_sessions() -> pd.DataFrame:
    """Get all sessions (for analytics)."""
    ensure_session_db()
    df = pd.read_csv(SESSION_DB)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_gesture_stats(patient_id: str) -> dict:
    """Get gesture performance stats for a patient."""
    sessions = get_patient_sessions(patient_id)
    if sessions.empty:
        return {}

    stats = {}
    for gesture in sessions["predicted_gesture"].unique():
        gesture_data = sessions[sessions["predicted_gesture"] == gesture]
        stats[gesture] = {
            "count": len(gesture_data),
            "avg_confidence": gesture_data["confidence"].mean(),
            "success_rate": gesture_data["accuracy"].mean() * 100 if gesture_data["accuracy"].notna().any() else None,
        }
    return stats

def get_recent_accuracy_trend(patient_id: str, days: int = 30) -> pd.DataFrame:
    """Get accuracy trend over time."""
    sessions = get_patient_sessions(patient_id)
    if sessions.empty:
        return pd.DataFrame()

    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"])
    recent = sessions[sessions["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]

    if recent.empty:
        return pd.DataFrame()

    recent["date"] = recent["timestamp"].dt.date
    trend = recent.groupby("date").agg({
        "accuracy": "mean",
        "confidence": "mean",
        "session_id": "count"
    }).rename(columns={"session_id": "session_count"})

    return trend

def get_session_count(patient_id: str) -> int:
    """Total sessions for patient."""
    return len(get_patient_sessions(patient_id))

def get_avg_accuracy(patient_id: str) -> float:
    """Average accuracy across all sessions."""
    sessions = get_patient_sessions(patient_id)
    if sessions.empty or sessions["accuracy"].isna().all():
        return 0.0
    return sessions["accuracy"].mean() * 100

def get_avg_confidence(patient_id: str) -> float:
    """Average confidence across all sessions."""
    sessions = get_patient_sessions(patient_id)
    if sessions.empty:
        return 0.0
    return sessions["confidence"].mean()
