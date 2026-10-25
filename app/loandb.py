"""Backend table for assessed applications (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

DB_PATH = Path(getattr(config, "DB_PATH", Path(__file__).with_name("database/loan_applications.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant      TEXT NOT NULL,                 -- full name
    applicant_key  TEXT NOT NULL UNIQUE,          -- normalised ID number
    id_number      TEXT NOT NULL DEFAULT '',
    submitted_at   TEXT NOT NULL,
    score          REAL NOT NULL,
    threshold      REAL NOT NULL,
    model_decision TEXT NOT NULL,
    model          TEXT,
    inputs         TEXT NOT NULL,
    final_decision TEXT NOT NULL DEFAULT 'Pending',
    admin_note     TEXT NOT NULL DEFAULT '',
    reviewed_at    TEXT
)
"""


def _key(id_number: str) -> str:
    return "".join(str(id_number).split()).casefold()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(_SCHEMA)
    # Upgrade a database created by an earlier version of the app
    columns = {row[1] for row in con.execute("PRAGMA table_info(applications)")}
    if "id_number" not in columns:
        con.execute("ALTER TABLE applications ADD COLUMN id_number TEXT NOT NULL DEFAULT ''")
        con.commit()
    return con


def exists(id_number: str) -> bool:
    with closing(_connect()) as con:
        row = con.execute("SELECT 1 FROM applications WHERE applicant_key = ?", (_key(id_number),)).fetchone()
    return row is not None


def add(id_number: str, full_name: str, score: float, threshold: float, decision: str,
        model: str, inputs: dict) -> bool:
    """Save one assessment. Returns False if this ID number was already assessed."""
    try:
        with closing(_connect()) as con, con:
            con.execute(
                "INSERT INTO applications (applicant, applicant_key, id_number, submitted_at, score, "
                "threshold, model_decision, model, inputs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (" ".join(str(full_name).split()), _key(id_number), "".join(str(id_number).split()),
                 datetime.now().strftime("%Y-%m-%d %H:%M"), float(score), float(threshold), decision,
                 model, json.dumps(inputs, default=str)),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def fetch_all() -> pd.DataFrame:
    with closing(_connect()) as con:
        return pd.read_sql_query("SELECT * FROM applications ORDER BY id DESC", con)


def update_review(app_id: int, final_decision: str, note: str) -> None:
    reviewed = None if final_decision == "Pending" else datetime.now().strftime("%Y-%m-%d %H:%M")
    with closing(_connect()) as con, con:
        con.execute(
            "UPDATE applications SET final_decision = ?, admin_note = ?, reviewed_at = ? WHERE id = ?",
            (final_decision, note, reviewed, int(app_id)),
        )


def delete(ids: list[int]) -> int:
    """Permanently remove applications by row id. Returns how many were deleted."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    with closing(_connect()) as con, con:
        cur = con.execute(f"DELETE FROM applications WHERE id IN ({marks})", ids)
        return cur.rowcount