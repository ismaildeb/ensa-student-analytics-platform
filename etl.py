from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from platform_config import SUBJECT_COLS, WEIGHTS

@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    raw_csv: Path = Path("data") / "raw.csv"
    clean_csv: Path = Path("data") / "clean.csv"



def _parse_number_series(series: pd.Series) -> pd.Series:
    """
    Convert strings like '12,5' or '12.5' to numeric floats.
    Non-parsable values become NaN.
    """
    s = series.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Expected file: data/raw.csv"
        )
    return pd.read_csv(path, encoding="utf-8")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "student_id",
        "nom",
        "filiere",
        "classe",
        *SUBJECT_COLS,
        "absences",
        "date",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw dataset: {sorted(missing)}")

    out = df.copy()

    out["student_id"] = out["student_id"].astype("string").str.strip()
    out["nom"] = out["nom"].astype("string").str.strip()
    out["filiere"] = out["filiere"].astype("string").str.strip()
    out["classe"] = out["classe"].astype("string").str.strip()

    for col in SUBJECT_COLS:
        out[col] = _parse_number_series(out[col]).clip(lower=0, upper=20)

    out["absences"] = _parse_number_series(out["absences"]).clip(lower=0)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["student_id", "nom", "filiere", "classe", "date"])

    # Remove exact duplicates (same student snapshot)
    out = out.drop_duplicates(subset=["student_id", "date"], keep="last")

    # Impute missing subject grades with class median (fallback to global median)
    for col in SUBJECT_COLS:
        class_median = out.groupby("classe")[col].transform("median")
        out[col] = out[col].fillna(class_median)
        out[col] = out[col].fillna(out[col].median())

    # Impute absences with 0 if missing (absence is often recorded as 0 by default)
    out["absences"] = out["absences"].fillna(0).round().astype(int)

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["moyenne_generale"] = out[SUBJECT_COLS].mean(axis=1).round(2)

    # Weighted average (ENSA-style): keeps the score explainable and closer to real grading priorities
    weighted = 0.0
    for col, w in WEIGHTS.items():
        weighted = weighted + out[col] * float(w)
    out["moyenne_ponderee"] = weighted.round(2)

    # 0-100 score: weighted performance minus a stronger penalty when absences increase
    academic = (out["moyenne_ponderee"] / 20.0) * 100.0
    absences = out["absences"].astype(float)
    penalty = absences * 1.5 + np.maximum(0.0, absences - 3.0) * 1.5
    score = academic - penalty
    out["score_global"] = score.clip(lower=0, upper=100).round(2)

    def niveau_from_avg(avg: float) -> str:
        if avg < 10:
            return "faible"
        if avg < 14:
            return "moyen"
        return "bon"

    out["niveau"] = out["moyenne_generale"].apply(niveau_from_avg).astype("string")

    def risk_from_row(avg: float, absences: int, score_global: float) -> str:
        # Explainable rule-based baseline (used as a label + operational threshold)
        if avg < 10 or absences >= 8 or score_global < 45:
            return "high"
        if avg < 12 or absences >= 5 or score_global < 60:
            return "medium"
        return "low"

    out["risk"] = [
        risk_from_row(float(a), int(x), float(s))
        for a, x, s in zip(
            out["moyenne_generale"].to_numpy(),
            out["absences"].to_numpy(),
            out["score_global"].to_numpy(),
        )
    ]

    return out


def save_clean(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.to_csv(path, index=False, encoding="utf-8")


def run_analysis(clean_df: pd.DataFrame) -> None:
    """
    Basic analytics printed to console:
    - averages per subject
    - top students (latest snapshot)
    - % of at-risk students
    - correlation absences vs average grade
    """
    df = clean_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    means = df[SUBJECT_COLS].mean().round(2)
    print("\n=== Moyenne par matiere ===")
    for k, v in means.to_dict().items():
        print(f"- {k}: {v}")

    latest = (
        df.sort_values("date")
        .groupby("student_id", as_index=False)
        .tail(1)
        .sort_values("moyenne_generale", ascending=False)
    )

    print("\n=== Top 10 etudiants (dernier releve) ===")
    top10 = latest.head(10)[["student_id", "nom", "classe", "moyenne_generale", "risk"]]
    for row in top10.itertuples(index=False):
        print(
            f"- {row.student_id} | {row.nom} | {row.classe} | moyenne={row.moyenne_generale} | risk={row.risk}"
        )

    risk_rate = (latest["risk"] == "high").mean() * 100.0
    print("\n=== Taux d'etudiants a risque (high) ===")
    print(f"- {risk_rate:.1f}% (sur {len(latest)} etudiants, dernier releve)")

    corr = df["absences"].corr(df["moyenne_generale"])
    print("\n=== Correlation absences / moyenne ===")
    print(f"- corr(absences, moyenne_generale) = {corr:.3f}")


def run(paths: Paths = Paths()) -> pd.DataFrame:
    raw = load_raw(paths.raw_csv)
    clean = clean_data(raw)
    enriched = add_features(clean)
    save_clean(enriched, paths.clean_csv)
    run_analysis(enriched)
    return enriched


if __name__ == "__main__":
    try:
        run()
        print("\nOK - ETL termine. Fichier genere:", Paths.clean_csv.as_posix())
    except Exception as exc:
        print(f"\nERROR - ETL echoue: {exc}", file=sys.stderr)
        raise
