# Smart Student Analytics Platform

Plateforme data end-to-end (Python) pour analyser les performances des etudiants (ENSA Tetouan), detecter les etudiants a risque et produire des insights exploitables.

## Tech stack

- Data Engineering / Analysis : Python, pandas, numpy
- Machine Learning : scikit-learn (modele "early warning" t -> t+1)
- Base de donnees : SQLite
- App : Dash (Python)
- BI : Power BI (dataset propre + predictions)

## Demo (Power BI + captures)

- Power BI (PBIX): `assets/SmartStudentAnalytics_ENSA.pbix` (a ajouter)
- Export PDF: `assets/SmartStudentAnalytics_ENSA.pdf` (a ajouter)
- Captures: `assets/screenshots/` (a ajouter)

## Modules (exemples ENSA)

- `analyse`, `algebre`, `physique`, `chimie`
- `programmation`, `algorithmique`, `electronique`, `reseaux`
- `anglais`

## Structure

```
Smart Student Analytics Platform/
data/                          (genere par `generate_data.py` / `etl.py` / `predict.py`)
powerbi/                       (genere par `export_powerbi.py`)
models/                        (genere par `train_model.py`)
app.py                         (Dash UI)
database.py                    (SQLite loader)
etl.py                         (ETL + features + analyses)
export_powerbi.py              (exports Power BI "FR ready")
generate_data.py               (synthetic ENSA dataset)
ml_model.py                    (calibration wrapper)
platform_config.py             (colonnes + poids)
predict.py                     (batch predictions)
train_model.py                 (ML training)
requirements.txt
POWERBI_GUIDE.md
Dockerfile
README.md
assets/                        (pbix/pdf/captures)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execution (ETL -> DB -> ML -> App)

0) (Optionnel mais recommande) Regenerer un dataset "ENSA-style" (>= 200 lignes):

```powershell
python generate_data.py
```

Note: le generateur cree un volume equilibre (meme nombre d'etudiants par classe) pour que les dashboards soient plus realistes.

1) ETL + features + analyses console:

```powershell
python etl.py
```

2) Charger les donnees dans SQLite:

```powershell
python database.py
```

3) Entrainement du modele ML (optionnel mais recommande):

```powershell
python train_model.py
```

4) Generer un fichier avec predictions (recommande pour Power BI):

```powershell
python predict.py
```

5) Exporter des fichiers "Power BI ready" (locale FR, evite '.' vs ','):

```powershell
python export_powerbi.py
```

6) Lancer l'application Dash:

```powershell
python app.py
```

Mode debug (affiche la barre devtools en bas):

```powershell
$env:DASH_DEBUG="1"; python app.py
```

Si le port 8050 est occupe:

```powershell
$env:PORT="8051"; python app.py
```

## Power BI

Importer de preference les exports FR (evite '.' vs ','):
- `powerbi/students_latest_fr.csv` (KPI / dernier releve)
- `powerbi/modules_long_fr.csv` (analyse modules)
- `powerbi/clean_with_predictions_fr.csv` (historique)

Guide pas-a-pas + exemples DAX: `POWERBI_GUIDE.md`.

Notes (pour partager le PBIX sur GitHub):
- Mets ton fichier `.pbix` dans `assets/` pour qu'il soit versionne.
- Si le PBIX demande de mettre a jour les chemins de source, va dans:
  Transform Data -> Data source settings -> Change source vers les CSV dans `powerbi/`.

## Docker (optionnel)

```bash
docker build -t student-analytics .
docker run -p 8050:8050 student-analytics
```
