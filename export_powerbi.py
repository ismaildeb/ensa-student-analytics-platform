from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from platform_config import SUBJECT_COLS


@dataclass(frozen=True)
class Paths:
    src_csv: Path = Path("data") / "clean_with_predictions.csv"
    out_dir: Path = Path("powerbi")
    out_history: Path = Path("powerbi") / "clean_with_predictions_fr.csv"
    out_latest: Path = Path("powerbi") / "students_latest_fr.csv"
    out_modules: Path = Path("powerbi") / "modules_long_fr.csv"


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "risk" in out.columns:
        out["risk_high_flag"] = (
            (out["risk"].astype("string") == "high").fillna(False).astype(int)
        )
    if "pred_risk_high_next" in out.columns:
        out["pred_risk_high_next_flag"] = (
            (out["pred_risk_high_next"].astype("string") == "high")
            .fillna(False)
            .astype(int)
        )
    return out


def build_students_latest(history: pd.DataFrame) -> pd.DataFrame:
    df = _to_datetime(history)
    df = df.dropna(subset=["student_id", "date"]).sort_values(["student_id", "date"])
    latest = (
        df.sort_values("date", ascending=False)
        .drop_duplicates(subset=["student_id"], keep="first")
        .sort_values(["filiere", "classe", "student_id"])
    )
    latest["date"] = pd.to_datetime(latest["date"]).dt.date.astype(str)
    return latest.reset_index(drop=True)


def build_modules_long(students_latest: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "student_id",
        "nom",
        "filiere",
        "classe",
        "date",
        "risk",
        "pred_risk_high_next_proba",
        "pred_risk_high_next",
        "risk_high_flag",
        "pred_risk_high_next_flag",
    ]
    keep_cols = [c for c in keep_cols if c in students_latest.columns]

    wide = students_latest[keep_cols + [c for c in SUBJECT_COLS if c in students_latest.columns]].copy()
    long = wide.melt(
        id_vars=keep_cols,
        value_vars=[c for c in SUBJECT_COLS if c in wide.columns],
        var_name="module",
        value_name="note",
    )
    return long.reset_index(drop=True)


def export_csv(
    df: pd.DataFrame, path: Path, *, sep: str, decimal: str, encoding: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, decimal=decimal, encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporte des CSV 'Power BI ready' (locale FR) pour eviter les problemes '.' vs ','."
    )
    parser.add_argument("--src", default=str(Paths.src_csv), help="CSV source (historique).")
    parser.add_argument("--outdir", default=str(Paths.out_dir), help="Dossier de sortie.")
    parser.add_argument("--sep", default=";", help="Separateur CSV (par defaut ';').")
    parser.add_argument(
        "--decimal", default=",", help="Separateur decimal (par defaut ',')."
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encodage (par defaut 'utf-8-sig' pour Excel/Power BI).",
    )
    args = parser.parse_args()

    paths = Paths(
        src_csv=Path(args.src),
        out_dir=Path(args.outdir),
        out_history=Path(args.outdir) / "clean_with_predictions_fr.csv",
        out_latest=Path(args.outdir) / "students_latest_fr.csv",
        out_modules=Path(args.outdir) / "modules_long_fr.csv",
    )

    if not paths.src_csv.exists():
        raise FileNotFoundError(
            f"{paths.src_csv} introuvable. Lance d'abord `python predict.py`."
        )

    history = pd.read_csv(paths.src_csv, encoding="utf-8")
    history = _add_flags(history)

    students_latest = build_students_latest(history)
    modules_long = build_modules_long(students_latest)

    export_csv(history, paths.out_history, sep=args.sep, decimal=args.decimal, encoding=args.encoding)
    export_csv(students_latest, paths.out_latest, sep=args.sep, decimal=args.decimal, encoding=args.encoding)
    export_csv(modules_long, paths.out_modules, sep=args.sep, decimal=args.decimal, encoding=args.encoding)

    print("OK - Exports Power BI generes:")
    print(f"- {paths.out_history}")
    print(f"- {paths.out_latest}")
    print(f"- {paths.out_modules}")


if __name__ == "__main__":
    main()
