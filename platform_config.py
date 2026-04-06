from __future__ import annotations

SUBJECT_COLS = [
    # Tronc commun ENSA (exemples)
    "analyse",
    "algebre",
    "physique",
    "chimie",
    # Informatique / Genie info
    "programmation",
    "algorithmique",
    "electronique",
    "reseaux",
    # Soft skills
    "anglais",
]

# Filieres ENSA Tetouan (hors tronc commun)
FILIERES = ["BDIA", "GI", "GM", "GC", "GSTR", "SCM", "GCSE"]

# Pondérations (somme = 1.0) pour une moyenne "plus métier"
WEIGHTS = {
    "analyse": 0.18,
    "algebre": 0.12,
    "programmation": 0.18,
    "algorithmique": 0.14,
    "physique": 0.12,
    "electronique": 0.10,
    "reseaux": 0.08,
    "chimie": 0.05,
    "anglais": 0.03,
}
