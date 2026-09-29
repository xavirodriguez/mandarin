import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.domain.ports.assessment_repository import AssessmentRepository

class SQLAssessmentRepository(AssessmentRepository):
    """
    SQL Assessment Repository persistence adapter.
    Supports SQLite database by default, or PostgreSQL when postgresql:// URL is configured.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL") or os.getenv("CAPT_DATABASE_URL") or "assessments.db"
        self.is_postgres = self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")

        if self.is_postgres:
            try:
                import psycopg2  # type: ignore # pylint: disable=import-outside-toplevel
                self._pg_module = psycopg2
            except ImportError:
                # If psycopg2 driver is not installed in local environment, fallback to SQLite
                self.is_postgres = False
                self.db_file = "assessments.db"
        else:
            if self.db_url.startswith("sqlite:///"):
                self.db_file = self.db_url.replace("sqlite:///", "")
            elif self.db_url.startswith("sqlite://"):
                self.db_file = self.db_url.replace("sqlite://", "")
            else:
                self.db_file = self.db_url

        self._init_db()

    def _get_connection(self):
        if self.is_postgres:
            conn = self._pg_module.connect(self.db_url)
            return conn
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                CREATE TABLE IF NOT EXISTS assessments (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(128),
                    target_text TEXT,
                    phonetic_score REAL,
                    tone_score REAL,
                    diagnostics_json TEXT,
                    created_at VARCHAR(64)
                );
            """
            cursor.execute(query)
            conn.commit()

    def save_assessment(self, assessment_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
        record_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        target_text = assessment_data.get("utterance", "")
        p_score = float(assessment_data.get("phonetic_assessment", {}).get("overall_score", 0.0))
        t_score = float(assessment_data.get("tone_assessment", {}).get("overall_score", 0.0))

        raw_errors = assessment_data.get("errors", [])
        ser_errors = []
        for err in raw_errors:
            if hasattr(err, "error_type"):
                ser_errors.append({
                    "category": getattr(err.category, "value", str(err.category)),
                    "error_type": getattr(err.error_type, "value", str(err.error_type)),
                    "error_subtype": getattr(err, "error_subtype", ""),
                    "severity": getattr(err, "severity", 0.0),
                    "confidence": getattr(err, "confidence", 1.0),
                    "syllable_text": getattr(err, "syllable_text", ""),
                    "pinyin": getattr(err, "pinyin", "")
                })
            elif isinstance(err, dict):
                ser_errors.append(err)

        diag_json = json.dumps(ser_errors)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            param_placeholder = "%s" if self.is_postgres else "?"
            query = f"""
                INSERT INTO assessments (id, user_id, target_text, phonetic_score, tone_score, diagnostics_json, created_at)
                VALUES ({param_placeholder}, {param_placeholder}, {param_placeholder}, {param_placeholder}, {param_placeholder}, {param_placeholder}, {param_placeholder})
            """
            cursor.execute(query, (record_id, user_id, target_text, p_score, t_score, diag_json, now_iso))
            conn.commit()

        return record_id

    def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            param_placeholder = "%s" if self.is_postgres else "?"
            cursor.execute(
                f"SELECT id, user_id, target_text, phonetic_score, tone_score, diagnostics_json, created_at FROM assessments WHERE id = {param_placeholder}",
                (assessment_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            if self.is_postgres:
                return {
                    "id": row[0],
                    "user_id": row[1],
                    "target_text": row[2],
                    "phonetic_score": row[3],
                    "tone_score": row[4],
                    "diagnostics": json.loads(row[5]) if row[5] else [],
                    "created_at": row[6]
                }
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "target_text": row["target_text"],
                "phonetic_score": row["phonetic_score"],
                "tone_score": row["tone_score"],
                "diagnostics": json.loads(row["diagnostics_json"]) if row["diagnostics_json"] else [],
                "created_at": row["created_at"]
            }
