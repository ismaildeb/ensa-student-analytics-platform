from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
from joblib import load

from platform_config import SUBJECT_COLS, WEIGHTS

@dataclass(frozen=True)
class Paths:
    clean_csv: Path = Path("data") / "clean.csv"
    model_path: Path = Path("models") / "risk_model.joblib"
    meta_path: Path = Path("models") / "risk_model_meta.json"
    out_csv: Path = Path("data") / "clean_with_predictions.csv"



def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["student_id", "date"]).sort_values(["student_id", "date"])

    out["avg_prev"] = out.groupby("student_id")["moyenne_generale"].shift(1)
    out["avgp_prev"] = out.groupby("student_id")["moyenne_ponderee"].shift(1)
    out["abs_prev"] = out.groupby("student_id")["absences"].shift(1)
    out["delta_avg"] = out["moyenne_generale"] - out["avg_prev"]
    out["delta_avgp"] = out["moyenne_ponderee"] - out["avgp_prev"]
    out["delta_abs"] = out["absences"] - out["abs_prev"]
    return out


def main(paths: Paths = Paths()) -> None:
    if not paths.clean_csv.exists():
        raise FileNotFoundError("data/clean.csv introuvable. Lance `python etl.py`.")
    if not paths.model_path.exists():
        raise FileNotFoundError("Modele introuvable. Lance `python train_model.py`.")

    model = load(paths.model_path)
    threshold = 0.5
    if paths.meta_path.exists():
        try:
            meta = json.loads(paths.meta_path.read_text(encoding="utf-8"))
            threshold = float(meta.get("threshold", 0.5))
        except Exception:
            threshold = 0.5
    df = pd.read_csv(paths.clean_csv, encoding="utf-8")
    if "moyenne_ponderee" not in df.columns:
        weighted = 0.0
        for col, w in WEIGHTS.items():
            weighted = weighted + df[col] * float(w)
        df["moyenne_ponderee"] = weighted.round(2)
    df = add_model_features(df)

    features = [
        "filiere",
        "classe",
        *SUBJECT_COLS,
        "absences",
        "moyenne_generale",
        "moyenne_ponderee",
        "score_global",
        "avg_prev",
        "avgp_prev",
        "abs_prev",
        "delta_avg",
        "delta_avgp",
        "delta_abs",
    ]

    # Predictions only where lag features exist
    can_predict = df[["avg_prev", "avgp_prev", "abs_prev"]].notna().all(axis=1)
    proba = pd.Series([None] * len(df), dtype="float64")
    if can_predict.any():
        proba.loc[can_predict] = model.predict_proba(df.loc[can_predict, features])[:, 1]

    df["pred_risk_high_next_proba"] = proba.round(4)
    df["pred_risk_high_next"] = pd.Series(
        pd.NA, index=df.index, dtype="string"
    )
    df.loc[can_predict, "pred_risk_high_next"] = (df.loc[can_predict, "pred_risk_high_next_proba"] >= threshold).map(
        {True: "high", False: "not_high"}
    )

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.to_csv(paths.out_csv, index=False, encoding="utf-8")

    print("OK - Predictions generees:")
    print(f"- {paths.out_csv}")


if __name__ == "__main__":
    main()
