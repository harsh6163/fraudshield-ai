
from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "fraudshield.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_reference TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                monitoring_status TEXT NOT NULL,
                review_decision TEXT NOT NULL DEFAULT 'Pending review',
                analyst_notes TEXT NOT NULL DEFAULT '',
                source_file TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS case_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_reference TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                previous_decision TEXT NOT NULL,
                new_decision TEXT NOT NULL,
                previous_notes TEXT NOT NULL DEFAULT '',
                new_notes TEXT NOT NULL DEFAULT '',
                changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()


def save_cases(results, source_file=""):
    if results is None or results.empty:
        return

    records = []

    for _, row in results.iterrows():
        records.append((
            str(row["Case Reference"]),
            str(row["Transaction ID"]),
            float(row["Fraud Risk Score"]),
            str(row["Monitoring Status"]),
            str(row["Review Decision"]),
            str(row.get("Analyst Notes", "")),
            str(source_file),
        ))

    with get_connection() as connection:
        connection.executemany("""
            INSERT INTO cases (
                case_reference,
                transaction_id,
                risk_score,
                monitoring_status,
                review_decision,
                analyst_notes,
                source_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_reference) DO UPDATE SET
                transaction_id = excluded.transaction_id,
                risk_score = excluded.risk_score,
                monitoring_status = excluded.monitoring_status,
                review_decision = excluded.review_decision,
                analyst_notes = excluded.analyst_notes,
                source_file = excluded.source_file,
                updated_at = CURRENT_TIMESTAMP
        """, records)
        connection.commit()


def update_case_decision(case_reference, decision, notes=""):
    """Update a case and log any change to its decision or notes."""
    with get_connection() as connection:
        current = connection.execute("""
            SELECT transaction_id, review_decision, analyst_notes
            FROM cases WHERE case_reference = ?
        """, (case_reference,)).fetchone()
        if current is None:
            raise ValueError(f"Case not found: {case_reference}")

        old_decision = str(current["review_decision"] or "")
        old_notes = str(current["analyst_notes"] or "")
        new_decision = str(decision or "")
        new_notes = str(notes or "")

        if old_decision != new_decision or old_notes != new_notes:
            connection.execute("""
                INSERT INTO case_audit_log (
                    case_reference, transaction_id, previous_decision,
                    new_decision, previous_notes, new_notes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (case_reference, str(current["transaction_id"]),
                  old_decision, new_decision, old_notes, new_notes))
            connection.execute("""
                UPDATE cases
                SET review_decision = ?, analyst_notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE case_reference = ?
            """, (new_decision, new_notes, case_reference))
            connection.commit()


def get_audit_logs():
    """Return audit events, newest first."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, case_reference, transaction_id, previous_decision,
                   new_decision, previous_notes, new_notes, changed_at
            FROM case_audit_log ORDER BY changed_at DESC, id DESC
        """).fetchall()
    return [dict(row) for row in rows]

def get_all_cases():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM cases
            ORDER BY updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]
