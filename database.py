from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from platform_config import SUBJECT_COLS, WEIGHTS

@dataclass(frozen=True)
class Paths:
    clean_csv: Path = Path("data") / "clean.csv"
    db_path: Path = Path("students.db")


def _schema_sql() -> str:
    subject_cols_sql = "\n".join([f"    {c} REAL NOT NULL," for c in SUBJECT_COLS])
    return f"""
DROP TABLE IF EXISTS students;
CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    nom TEXT NOT NULL,
    filiere TEXT NOT NULL,
    classe TEXT NOT NULL,
    date TEXT NOT NULL,
{subject_cols_sql}
    absences INTEGER NOT NULL,
    moyenne_generale REAL NOT NULL,
    moyenne_ponderee REAL NOT NULL,
    score_global REAL NOT NULL,
    niveau TEXT NOT NULL,
    risk TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_students_student_date ON students(student_id, date);
""".lstrip()


def load_clean_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. Run `python etl.py` first."
        )
    return pd.read_csv(path, encoding="utf-8")


def init_db(conn: sqlite3.Connection) -> None:
    # For a demo/portfolio project, we recreate the table to keep schema consistent
    conn.executescript(_schema_sql())


def insert_students(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    cols = [
        "student_id",
        "nom",
        "filiere",
        "classe",
        "date",
        *SUBJECT_COLS,
        "absences",
        "moyenne_generale",
        "moyenne_ponderee",
        "score_global",
        "niveau",
        "risk",
    ]
    missing = set(cols).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in clean.csv: {sorted(missing)}")

    records = df[cols].to_records(index=False)
    insert_cols = ", ".join(cols)
    placeholders = ", ".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT INTO students ({insert_cols}) VALUES ({placeholders});",
        list(records),
    )
    conn.commit()
    return len(df)


def run(paths: Paths = Paths()) -> None:
    df = load_clean_csv(paths.clean_csv)
    if "moyenne_ponderee" not in df.columns:
        weighted = 0.0
        for col, w in WEIGHTS.items():
            weighted = weighted + df[col] * float(w)
        df["moyenne_ponderee"] = weighted.round(2)
    with sqlite3.connect(paths.db_path) as conn:
        init_db(conn)
        inserted = insert_students(conn, df)
    print(f"OK - Base SQLite prete: {paths.db_path} (rows inserted: {inserted})")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"ERROR - database.py echoue: {exc}", file=sys.stderr)
        raise
