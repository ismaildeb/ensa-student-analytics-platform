# Power BI Guide (Smart Student Analytics Platform)

## 1) Quelle source utiliser ?

Recommande (le plus simple, evite les problemes '.' vs ',' en FR):
1) Generer les predictions: `python predict.py`
2) Exporter des fichiers "Power BI ready" (CSV FR): `python export_powerbi.py`
3) Importer depuis le dossier `powerbi/`:
   - `powerbi/students_latest_fr.csv` (KPI / dernier releve)
   - `powerbi/modules_long_fr.csv` (analyse des modules)
   - `powerbi/clean_with_predictions_fr.csv` (historique complet)

Alternative:
- `data/clean_with_predictions.csv` (brut) + conversion de types via "Utiliser les parametres regionaux"
- `data/clean.csv` (sans predictions)
- `students.db` (table `students`) si tu veux une source SQL (SQLite)

## 2) Import CSV (simple)

1. Power BI Desktop -> Get Data -> Text/CSV
2. Choisir `powerbi/students_latest_fr.csv` ou `powerbi/clean_with_predictions_fr.csv`
3. Verifier les types (normalement OK si tu utilises les exports `*_fr.csv`):
   - `date`: Date
   - notes (modules) + moyennes: Decimal Number
   - `absences`: Whole Number
   - `risk`, `niveau`, `classe`: Text
   - `pred_risk_high_next_proba`: Decimal Number

Si tu importes directement `data/clean_with_predictions.csv` (decimales avec un point '.'),
il faut souvent faire:
- Transform Data -> selectionner les colonnes numeriques -> Data type -> Using Locale...
- Decimal Number + English (United States)

Modules (exemples ENSA):
- `analyse`, `algebre`, `physique`, `chimie`, `programmation`, `algorithmique`, `electronique`, `reseaux`, `anglais`
Filtres utiles:
- `filiere` (BDIA, GI, GM, GC, GSTR, SCM, GCSE)
- `classe` (TC1/TC2 puis filiere1..3)

## 3) Creer une vue "dernier releve" (important)

Le dataset est historique (plusieurs dates / etudiant). Pour des KPI "actuels", cree une table basee sur le dernier releve par etudiant.

Si tu utilises `powerbi/students_latest_fr.csv`, cette etape est deja faite.

Dans Power Query (Transform Data):
1. Dupliquer la requete `students` -> renommer `students_latest`
2. Trier par `date` decroissant
3. Group By:
   - Group by: `student_id`
   - Operation: All Rows
4. Ajouter une colonne personnalisee: `Table.FirstN([All Rows], 1)`
5. Expand et supprimer `All Rows`

Tu obtiens 1 ligne par etudiant pour KPI + classements.

## 4) Mesures DAX (exemples)

Sur `students_latest`:

```DAX
Nb Etudiants = DISTINCTCOUNT(students_latest[student_id])
```

```DAX
Moyenne Generale (Latest) = AVERAGE(students_latest[moyenne_generale])
```

```DAX
Taux Risk High (Latest) =
DIVIDE(
    CALCULATE(COUNTROWS(students_latest), students_latest[risk] = "high"),
    [Nb Etudiants]
)
```

Si tu as les predictions:

```DAX
Taux Pred High Next (Latest) =
DIVIDE(
    CALCULATE(COUNTROWS(students_latest), students_latest[pred_risk_high_next_proba] >= 0.5),
    [Nb Etudiants]
)
```

Correlation (simple via scatter + trendline):
- Visual: Scatter chart
- X: `absences`
- Y: `moyenne_generale`
- Analytics: Trend line

## 5) Dashboards recommandes (4 pages)

1) Vue globale
- Cards: Nb Etudiants, Moyenne Generale (Latest), Taux Risk High (Latest)
- Bar: moyenne par `classe`
- Donut: repartition `niveau` / `risk`

2) Analyse modules
- Bar: moyennes par module et par `classe`
- Table: top/bottom 10 par module (sur `students_latest`)

3) Detection risque
- Matrice: `classe` x `risk` (count)
- Liste: etudiants `risk=high` tries par `score_global` et `absences`
- (Bonus) filtre sur `pred_risk_high_next_proba`

4) Evolution temporelle
- Line chart: `moyenne_generale` par `date` (global + slicer classe)
- Line chart: `absences` par `date`
- Decomposition tree: `risk` -> `classe` -> `niveau`
