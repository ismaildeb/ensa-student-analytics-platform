from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from platform_config import FILIERES, SUBJECT_COLS


@dataclass(frozen=True)
class Paths:
    out_csv: Path = Path("data") / "raw.csv"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main(paths: Paths = Paths()) -> None:
    random.seed(42)

    first_names = [
        "Adam",
        "Amine",
        "Aya",
        "Badr",
        "Chaimae",
        "Dounia",
        "Hajar",
        "Hamza",
        "Imane",
        "Ismail",
        "Ilyas",
        "Khadija",
        "Lina",
        "Maha",
        "Mariam",
        "Mehdi",
        "Meryem",
        "Mohamed",
        "Nada",
        "Nour",
        "Omar",
        "Rania",
        "Salma",
        "Sara",
        "Soufiane",
        "Yassine",
        "Youssef",
        "Zineb",
        "Anas",
        "Ikram",
    ]
    last_names = [
        "El Amrani",
        "Bennani",
        "Alaoui",
        "El Idrissi",
        "Benjelloun",
        "Rahmani",
        "El Fassi",
        "Chraibi",
        "Tahiri",
        "Kabbaj",
        "Lahlou",
        "Mansouri",
        "Zerouali",
        "Slaoui",
        "Kouider",
        "Raji",
        "Bouzidi",
        "Haddad",
        "Ouhami",
        "Amrani",
    ]

    # ENSA-like structure: tronc commun puis filiere (1..3)
    classes: list[tuple[str, str]] = []
    classes.extend([("TC", "TC1"), ("TC", "TC2")])
    for filiere in FILIERES:
        for year in (1, 2, 3):
            classes.append((filiere, f"{filiere}{year}"))

    # Ensure each class has enough students (more realistic + nicer dashboards)
    students_per_class = 12
    students: list[tuple[str, str, str, str, float, float]] = []
    idx = 1
    for filiere, classe in classes:
        for _ in range(students_per_class):
            student_id = f"S{idx:04d}"
            idx += 1
            nom = f"{random.choice(first_names)} {random.choice(last_names)}"
            ability = random.betavariate(2.2, 1.8)  # 0-1
            eng_bias = random.uniform(-0.10, 0.10)  # STEM / engineering bias
            students.append((student_id, nom, filiere, classe, ability, eng_bias))

    # 5 snapshots -> 300 lignes (>= 200)
    start = date(2025, 10, 1)
    dates = [start + timedelta(days=30 * k) for k in range(5)]

    def score(base: float, shift: float) -> float:
        val = 6 + base * 12 + shift + random.gauss(0, 2.1)
        return round(clamp(val, 0, 20), 1)

    rows: list[dict] = []
    for d in dates:
        for student_id, nom, filiere, classe, ability, eng_bias in students:
            drift = random.uniform(-0.05, 0.05)
            base = clamp(ability + drift, 0.0, 1.0)

            # Absences: tends to be higher when base is lower
            lam = 2.0 + (1.0 - base) * 5.0
            absences = min(18, int(random.gammavariate(lam, 0.8)))

            # Module scores (coherent, with small correlations)
            module = {
                "analyse": score(base, eng_bias * 8),
                "algebre": score(base, eng_bias * 6),
                "physique": score(base, eng_bias * 6),
                "chimie": score(base, eng_bias * 2),
                "programmation": score(base, eng_bias * 7),
                "algorithmique": score(base, eng_bias * 7),
                "electronique": score(base, eng_bias * 5),
                "reseaux": score(base, eng_bias * 4),
                "anglais": score(base, random.uniform(-1.5, 1.5)),
            }

            # Inject a bit of missingness / messy decimals (for ETL demo)
            for k in SUBJECT_COLS:
                if random.random() < 0.03:
                    module[k] = ""
                elif random.random() < 0.02:
                    module[k] = str(module[k]).replace(".", ",")

            abs_out = absences
            if random.random() < 0.02:
                abs_out = ""

            row = {
                "student_id": student_id,
                "nom": nom,
                "filiere": filiere,
                "classe": classe,
                **{k: module[k] for k in SUBJECT_COLS},
                "absences": abs_out,
                "date": d.isoformat(),
            }
            rows.append(row)

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.out_csv.open("w", newline="", encoding="utf-8") as f:
        cols = ["student_id", "nom", "filiere", "classe", *SUBJECT_COLS, "absences", "date"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {paths.out_csv}")


if __name__ == "__main__":
    main()
