from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression

from platform_config import SUBJECT_COLS, WEIGHTS
from ml_model import IsotonicCalibratedModel

@dataclass(frozen=True)
class Paths:
    clean_csv: Path = Path("data") / "clean.csv"
    model_dir: Path = Path("models")
    model_path: Path = Path("models") / "risk_model.joblib"
    meta_path: Path = Path("models") / "risk_model_meta.json"
    fi_path: Path = Path("models") / "feature_importance.csv"



def build_supervised_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Early warning use-case:
    Features at time t -> label whether the student will be 'high' risk at time t+1.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["student_id", "date"]).sort_values(["student_id", "date"])

    out["avg_prev"] = out.groupby("student_id")["moyenne_generale"].shift(1)
    out["avgp_prev"] = out.groupby("student_id")["moyenne_ponderee"].shift(1)
    out["abs_prev"] = out.groupby("student_id")["absences"].shift(1)
    out["delta_avg"] = out["moyenne_generale"] - out["avg_prev"]
    out["delta_avgp"] = out["moyenne_ponderee"] - out["avgp_prev"]
    out["delta_abs"] = out["absences"] - out["abs_prev"]

    out["risk_next"] = out.groupby("student_id")["risk"].shift(-1)
    out["y_high_next"] = (out["risk_next"] == "high").astype(int)

    # Keep only rows where we have a next label and at least one previous snapshot
    out = out.dropna(subset=["risk_next", "avg_prev", "avgp_prev", "abs_prev"])
    return out


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
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

    X = df[features].copy()
    y = df["y_high_next"].astype(int).to_numpy()
    groups = df["student_id"].astype(str).to_numpy()

    numeric_features = [c for c in features if c not in ("classe", "filiere")]
    categorical_features = ["filiere", "classe"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    base_pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Calibration without leakage: split train -> (train2, calib) by student_id
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    t2_idx, calib_idx = next(
        splitter2.split(X_train, y_train, groups=groups[train_idx])
    )
    X_train2, y_train2 = X_train.iloc[t2_idx], y_train[t2_idx]
    X_calib, y_calib = X_train.iloc[calib_idx], y_train[calib_idx]

    base_pipe.fit(X_train2, y_train2)

    # Isotonic calibration (manual) to avoid CalibratedClassifierCV 'prefit' incompatibilities
    calib_proba = base_pipe.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_proba, y_calib)

    calibrated = IsotonicCalibratedModel(base_pipe, iso)
    calibrated.classes_ = np.array([0, 1])

    proba = calibrated.predict_proba(X_test)[:, 1]

    # Threshold selection: prioritize recall (catch high risk), keep precision reasonable.
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_score = -1.0
    best_stats = None
    for thr in thresholds:
        pred_thr = (proba >= thr).astype(int)
        tp = int(((pred_thr == 1) & (y_test == 1)).sum())
        fp = int(((pred_thr == 1) & (y_test == 0)).sum())
        fn = int(((pred_thr == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # constraint: precision >= 0.60, optimize recall then f1
        if precision >= 0.60:
            score = recall * 10.0 + f1
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_stats = {"precision": precision, "recall": recall, "f1": f1}

    thr_used = best_thr
    pred = (proba >= thr_used).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "test_size": int(len(test_idx)),
        "train_size": int(len(train_idx)),
        "positive_rate_test": float(np.mean(y_test)),
        "threshold": float(thr_used),
        "threshold_selection": {
            "method": "precision>=0.60, optimize recall then f1",
            "stats_at_threshold": best_stats,
        },
        "features": features,
    }

    print("\n=== Evaluation (test split by student_id) ===")
    print(f"- ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"- Brier (calibration): {metrics['brier']:.3f}")
    print(f"- Positive rate (test): {metrics['positive_rate_test']:.3f}")
    print(f"- Threshold used: {metrics['threshold']:.2f}")
    print("\n=== Classification report (selected threshold) ===")
    print(classification_report(y_test, pred, digits=3))

    # Feature importance on original columns (permutation importance)
    fi = permutation_importance(
        calibrated,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )
    fi_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": fi.importances_mean,
            "importance_std": fi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return calibrated, metrics, fi_df


def main(paths: Paths = Paths()) -> None:
    if not paths.clean_csv.exists():
        raise FileNotFoundError("data/clean.csv introuvable. Lance d'abord `python etl.py`.")

    df = pd.read_csv(paths.clean_csv, encoding="utf-8")
    if "moyenne_ponderee" not in df.columns:
        weighted = 0.0
        for col, w in WEIGHTS.items():
            weighted = weighted + df[col] * float(w)
        df["moyenne_ponderee"] = weighted.round(2)
    supervised = build_supervised_dataset(df)
    if supervised.empty:
        raise ValueError("Dataset supervise vide. Verifie que tu as plusieurs dates par etudiant.")

    model, metrics, fi_df = train(supervised)

    paths.model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, paths.model_path)
    paths.meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fi_df.to_csv(paths.fi_path, index=False, encoding="utf-8")

    print("\nOK - Modele enregistre:")
    print(f"- {paths.model_path}")
    print(f"- {paths.meta_path}")
    print(f"- {paths.fi_path}")


if __name__ == "__main__":
    main()
