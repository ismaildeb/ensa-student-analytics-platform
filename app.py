from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import dash
from dash import Dash, Input, Output, dcc, html
import pandas as pd
import plotly.express as px

from platform_config import FILIERES, SUBJECT_COLS, WEIGHTS


DATA_CLEAN = Path("data") / "clean.csv"
DATA_PRED = Path("data") / "clean_with_predictions.csv"


def load_data() -> pd.DataFrame:
    path = DATA_PRED if DATA_PRED.exists() else DATA_CLEAN
    if not path.exists():
        raise FileNotFoundError(
            "Dataset introuvable. Lance d'abord `python etl.py` (et optionnellement `python predict.py`)."
        )
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "moyenne_ponderee" not in df.columns:
        weighted = 0.0
        for col, w in WEIGHTS.items():
            weighted = weighted + df[col] * float(w)
        df["moyenne_ponderee"] = weighted.round(2)

    return df


def risk_color(risk: str) -> str:
    r = (risk or "").lower()
    if r == "high":
        return "#ef4444"
    if r == "medium":
        return "#f59e0b"
    return "#22c55e"


_DF_CACHE: pd.DataFrame | None = None
_DF_SIG: tuple[str, int, int] | None = None


def get_df_all() -> pd.DataFrame:
    """
    Reload data automatically when CSV changes.
    This avoids having to restart the Dash server after running ETL/predict.
    """
    global _DF_CACHE, _DF_SIG
    path = DATA_PRED if DATA_PRED.exists() else DATA_CLEAN
    st = path.stat()
    sig = (str(path), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))
    if _DF_CACHE is None or _DF_SIG != sig:
        _DF_CACHE = load_data()
        _DF_SIG = sig
    return _DF_CACHE

app: Dash = dash.Dash(__name__)
server = app.server
APP_VERSION = "dash-ui-v2"

def latest_by_student(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby("student_id", as_index=False).tail(1)


def kpi_card(title: str, value: str, subtitle: str | None = None) -> html.Div:
    return html.Div(
        style={
            "flex": "1",
            "minWidth": "180px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "padding": "12px",
            "background": "white",
        },
        children=[
            html.Div(title, style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "900", "marginTop": "4px"}),
            html.Div(subtitle or "", style={"color": "#6b7280", "fontSize": "12px", "marginTop": "2px"}),
        ],
    )


app.layout = html.Div(
    style={"fontFamily": "system-ui, Segoe UI, Arial", "padding": "18px", "maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H2(f"ENSA Tetouan - Smart Student Analytics Platform (Dash) [{APP_VERSION}]"),
        html.Div(
            style={"color": "#6b7280", "marginBottom": "10px"},
            children="Filtres filiere/classe + detail etudiant + graphiques + (optionnel) prediction ML.",
        ),
        html.Hr(),
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    children=[
                        html.Label("Filiere"),
                        dcc.Dropdown(
                            id="filiere",
                            options=[{"label": "Toutes", "value": "ALL"}]
                            + [{"label": "TC", "value": "TC"}]
                            + [{"label": f, "value": f} for f in FILIERES],
                            value="ALL",
                            clearable=False,
                            style={"minWidth": "220px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Classe"),
                        dcc.Dropdown(id="classe", options=[{"label": "Toutes", "value": "ALL"}], value="ALL", clearable=False, style={"minWidth": "220px"}),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Etudiant"),
                        dcc.Dropdown(id="student", options=[], value=None, clearable=False, style={"minWidth": "360px"}),
                    ]
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "end", "gap": "10px"},
                    children=[
                        html.Button(
                            "Reload data",
                            id="reload_btn",
                            n_clicks=0,
                            style={
                                "height": "38px",
                                "padding": "0 12px",
                                "borderRadius": "10px",
                                "border": "1px solid #e5e7eb",
                                "background": "white",
                                "cursor": "pointer",
                                "fontWeight": "700",
                            },
                        ),
                        html.Div(id="data_status", style={"color": "#6b7280", "fontSize": "12px"}),
                    ],
                ),
            ],
        ),
        html.Br(),
        dcc.Tabs(
            id="tabs",
            value="global",
            colors={"border": "#e5e7eb", "primary": "#2563eb", "background": "#f9fafb"},
            style={"marginTop": "6px"},
            children=[
                dcc.Tab(
                    label="Vue globale",
                    value="global",
                    selected_style={"fontWeight": "800"},
                    children=[
                        html.Br(),
                        html.Div(id="kpis", style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
                        html.Br(),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                            children=[
                                html.Div(
                                    style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                                    children=[
                                        html.H4("Risque par filiere / niveau"),
                                        dcc.Graph(id="risk_heatmap", config={"displayModeBar": False}),
                                    ],
                                ),
                                html.Div(
                                    style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                                    children=[
                                        html.H4("Absences vs moyenne (click pour selectionner)"),
                                        dcc.Graph(id="scatter_abs", config={"displayModeBar": False}),
                                    ],
                                ),
                            ],
                        ),
                        html.Br(),
                        html.Div(
                            style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                            children=[
                                html.H4("Top etudiants a risque"),
                                html.Div(id="top_risk_table"),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Detail etudiant",
                    value="student",
                    selected_style={"fontWeight": "800"},
                    children=[
                        html.Br(),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1.1fr 1.4fr", "gap": "14px"},
                            children=[
                                html.Div(
                                    style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                                    children=[
                                        html.H4("Informations"),
                                        html.Div(id="student_info"),
                                        html.Div(id="student_pred", style={"marginTop": "10px"}),
                                    ],
                                ),
                                html.Div(
                                    style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                                    children=[
                                        dcc.Graph(id="trend_avg", config={"displayModeBar": False}),
                                    ],
                                ),
                            ],
                        ),
                        html.Br(),
                        html.Div(
                            style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px"},
                            children=[
                                html.H4("Notes (dernier releve)"),
                                dcc.Graph(id="bar_subjects", config={"displayModeBar": False}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"color": "#6b7280", "marginTop": "10px", "fontSize": "12px"},
            children=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ),
    ],
)


@app.callback(Output("data_status", "children"), Input("reload_btn", "n_clicks"))
def reload_data_status(_n: int):
    global _DF_CACHE, _DF_SIG
    # Force reload
    _DF_CACHE = None
    _DF_SIG = None
    df = get_df_all()
    latest = latest_by_student(df)
    src = str(DATA_PRED if DATA_PRED.exists() else DATA_CLEAN)
    return f"source={Path(src).name} | rows={len(df)} | latest_students={len(latest)}"


@app.callback(
    Output("classe", "options"),
    Output("classe", "value"),
    Input("filiere", "value"),
)
def update_classes(filiere: str):
    df = get_df_all()
    if filiere != "ALL":
        df = df[df["filiere"] == filiere]
    classes = sorted(df["classe"].unique().tolist())
    opts = [{"label": "Toutes", "value": "ALL"}] + [{"label": c, "value": c} for c in classes]
    return opts, "ALL"


@app.callback(
    Output("student", "options"),
    Output("student", "value"),
    Input("filiere", "value"),
    Input("classe", "value"),
)
def update_students(filiere: str, classe: str):
    df = get_df_all()
    if filiere != "ALL":
        df = df[df["filiere"] == filiere]
    if classe != "ALL":
        df = df[df["classe"] == classe]
    latest = latest_by_student(df)
    latest = latest.sort_values(["nom", "student_id"])
    options = [
        {"label": f"{r.student_id} - {r.nom}", "value": r.student_id}
        for r in latest.itertuples(index=False)
    ]
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("kpis", "children"),
    Output("risk_heatmap", "figure"),
    Output("scatter_abs", "figure"),
    Output("top_risk_table", "children"),
    Input("filiere", "value"),
    Input("classe", "value"),
)
def render_global(filiere: str, classe: str):
    df = get_df_all()
    if filiere != "ALL":
        df = df[df["filiere"] == filiere]
    if classe != "ALL":
        df = df[df["classe"] == classe]

    latest = latest_by_student(df)
    n = len(latest)
    avg = float(latest["moyenne_generale"].mean()) if n else 0.0
    high_pct = float((latest["risk"] == "high").mean() * 100.0) if n else 0.0

    pred_pct = None
    if "pred_risk_high_next" in latest.columns:
        pred_pct = float((latest["pred_risk_high_next"] == "high").mean() * 100.0)
    elif "pred_risk_high_next_proba" in latest.columns:
        pred_pct = float((latest["pred_risk_high_next_proba"] >= 0.5).mean() * 100.0)

    kpis = [
        kpi_card("Etudiants", str(n)),
        kpi_card("Moyenne generale", f"{avg:.2f}/20"),
        kpi_card("Risk HIGH", f"{high_pct:.1f}%", "dernier releve"),
    ]
    if pred_pct is not None:
        kpis.append(kpi_card("Pred HIGH (t+1)", f"{pred_pct:.1f}%", "seuil 0.5"))

    # Heatmap: filiere x risk count
    if n:
        heat = (
            latest.groupby(["filiere", "risk"])
            .size()
            .reset_index(name="count")
            .pivot(index="filiere", columns="risk", values="count")
            .fillna(0)
        )
        fig_heat = px.imshow(
            heat,
            aspect="auto",
            text_auto=True,
            title=None,
            color_continuous_scale="Blues",
        )
        fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    else:
        fig_heat = px.imshow([[0]])

    # Scatter: absences vs moyenne
    if n:
        fig_scatter = px.scatter(
            latest,
            x="absences",
            y="moyenne_generale",
            color="risk",
            hover_data=["student_id", "nom", "filiere", "classe"],
            color_discrete_map={"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"},
        )
        fig_scatter.update_traces(customdata=latest["student_id"])
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="risk")
        fig_scatter.update_yaxes(range=[0, 20])
    else:
        fig_scatter = px.scatter()

    # Top risk table
    top = latest.copy()
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    top["_risk_rank"] = top["risk"].map(risk_rank).fillna(9)
    top = top.sort_values(["_risk_rank", "score_global", "absences"], ascending=[True, True, False])
    top = top[top["risk"].isin(["high", "medium"])].head(12)
    rows = []
    for r in top.itertuples(index=False):
        rows.append(
            html.Tr(
                [
                    html.Td(r.student_id),
                    html.Td(r.nom),
                    html.Td(r.filiere),
                    html.Td(r.classe),
                    html.Td(f"{float(r.moyenne_generale):.2f}"),
                    html.Td(str(int(r.absences))),
                    html.Td(str(r.risk).upper()),
                ]
            )
        )
    table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("ID"),
                        html.Th("Nom"),
                        html.Th("Filiere"),
                        html.Th("Classe"),
                        html.Th("Moy"),
                        html.Th("Abs"),
                        html.Th("Risk"),
                    ],
                    style={"textAlign": "left", "color": "#6b7280"},
                )
            ),
            html.Tbody(rows if rows else [html.Tr([html.Td("Aucun", colSpan=7)])]),
        ],
    )

    return kpis, fig_heat, fig_scatter, table


@app.callback(
    Output("student", "value"),
    Input("scatter_abs", "clickData"),
    Input("student", "value"),
)
def select_student_from_scatter(click_data, current):
    if click_data and "points" in click_data and click_data["points"]:
        p = click_data["points"][0]
        student_id = p.get("customdata")
        return student_id or current
    return current


@app.callback(
    Output("student_info", "children"),
    Output("student_pred", "children"),
    Output("trend_avg", "figure"),
    Output("bar_subjects", "figure"),
    Input("student", "value"),
)
def render_student(student_id: str | None):
    if not student_id:
        empty_fig = px.line()
        return "Aucun etudiant.", "", empty_fig, empty_fig

    df = get_df_all()
    hist = df[df["student_id"] == student_id].sort_values("date")
    latest = hist.tail(1).iloc[0]

    risk = str(latest.get("risk", "")).lower()
    badge = html.Span(
        f"RISK: {risk.upper()}",
        style={
            "display": "inline-block",
            "padding": "6px 10px",
            "borderRadius": "999px",
            "background": risk_color(risk),
            "color": "white",
            "fontWeight": "800",
            "fontSize": "12px",
        },
    )

    info = html.Div(
        [
            html.Div(badge),
            html.Ul(
                [
                    html.Li(f"Nom: {latest['nom']}"),
                    html.Li(f"Filiere: {latest['filiere']}"),
                    html.Li(f"Classe: {latest['classe']}"),
                    html.Li(f"Date: {pd.to_datetime(latest['date']).date().isoformat()}"),
                    html.Li(f"Absences: {int(latest['absences'])}"),
                    html.Li(f"Moyenne generale: {float(latest['moyenne_generale']):.2f}/20"),
                    html.Li(f"Moyenne ponderee: {float(latest['moyenne_ponderee']):.2f}/20"),
                    html.Li(f"Score global: {float(latest['score_global']):.1f}/100"),
                ]
            ),
        ]
    )

    pred_block = ""
    if "pred_risk_high_next_proba" in df.columns:
        proba = latest.get("pred_risk_high_next_proba")
        if pd.notna(proba):
            pred_block = html.Div(
                [
                    html.H4("Prediction ML (t+1)", style={"marginTop": "10px"}),
                    html.Div(f"P(high risk prochain): {float(proba)*100:.1f}%"),
                ],
                style={"borderTop": "1px solid #e5e7eb", "paddingTop": "10px"},
            )

    fig_trend = px.line(
        hist,
        x="date",
        y="moyenne_generale",
        markers=True,
        title="Moyenne generale dans le temps",
        range_y=[0, 20],
    )
    fig_trend.update_layout(margin=dict(l=10, r=10, t=50, b=10))

    subj = pd.DataFrame({"matiere": SUBJECT_COLS, "note": [float(latest[c]) for c in SUBJECT_COLS]})
    fig_bar = px.bar(subj, x="matiere", y="note", range_y=[0, 20])
    fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    return info, pred_block, fig_trend, fig_bar


if __name__ == "__main__":
    # Dash devtools (bottom bar + Plotly Cloud prompt) appears when debug=True.
    debug = os.environ.get("DASH_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=debug, port=port)
