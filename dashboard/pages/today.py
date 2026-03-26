"""
Today's Predictions page.
Reads predictions from MLB_HR_Predictions.xlsx (written by the morning scheduler).
Shows featured pick-cards with player headshots + full sortable table.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from openpyxl import load_workbook

EXCEL_PATH = Path("MLB_HR_Predictions.xlsx")
PRED_COLUMNS = [
    "Date", "Player", "Team", "Opponent", "Pitcher",
    "HR_Probability", "Confidence", "Park_Factor",
    "Temp_F", "Wind_Speed_MPH", "Wind_Direction", "Is_Indoor",
    "Home_Game", "Predicted_HRs", "Actual_HRs", "Hit",
]
CONF_COLORS = {"High": "#10b981", "Medium": "#f97316", "Low": "#94a3b8"}


def layout():
    today_str = date.today().strftime("%B %d, %Y")
    return html.Div([
        html.Div([
            html.Div([
                html.H2(f"Today's HR Predictions", className="page-title"),
                html.P("Batters ranked by predicted home run probability", className="page-subtitle"),
            ]),
            html.Div(today_str, className="page-header-meta"),
        ], className="page-header"),

        # ── Filters ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Confidence"),
                dcc.Dropdown(
                    id="conf-filter",
                    options=[
                        {"label": "All", "value": "All"},
                        {"label": "High Only", "value": "High"},
                        {"label": "High + Medium", "value": "High+Medium"},
                    ],
                    value="All", clearable=False,
                    style={"width": "190px", "fontSize": "13px"},
                ),
            ]),
            html.Div([
                html.Label("Team"),
                dcc.Dropdown(
                    id="team-filter",
                    options=[{"label": "All Teams", "value": "All"}],
                    value="All", clearable=False,
                    style={"width": "175px", "fontSize": "13px"},
                ),
            ]),
            html.Div([
                html.Label("Wind"),
                dcc.Dropdown(
                    id="wind-filter",
                    options=[
                        {"label": "All", "value": "All"},
                        {"label": "Out (boost)",     "value": "out"},
                        {"label": "Calm",             "value": "calm"},
                        {"label": "Cross",            "value": "cross"},
                        {"label": "In (suppressor)", "value": "in"},
                    ],
                    value="All", clearable=False,
                    style={"width": "175px", "fontSize": "13px"},
                ),
            ]),
        ], className="filters-row"),

        # ── Summary cards ────────────────────────────────────────────────────
        html.Div(id="summary-cards", className="summary-cards"),

        # ── Featured picks (top 8 with photos) ───────────────────────────────
        html.Div([
            html.Div([
                html.H4("Featured Picks", style={"margin": 0}),
                html.Span("Top picks by HR probability",
                          style={"fontSize": "12px", "color": "#94a3b8"}),
            ], className="table-title-bar"),
            html.Div(id="picks-grid", className="picks-grid",
                     style={"padding": "16px"}),
        ], className="table-container"),

        # ── Full ranked table ─────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H4("All Predictions", style={"margin": 0}),
            ], className="table-title-bar"),
            dash_table.DataTable(
                id="predictions-table",
                columns=[
                    {"name": "#",           "id": "rank",          "type": "numeric"},
                    {"name": "Player",      "id": "Player"},
                    {"name": "Team",        "id": "Team"},
                    {"name": "Opponent",    "id": "Opponent"},
                    {"name": "vs Pitcher",  "id": "Pitcher"},
                    {"name": "HR Prob",     "id": "HR_Probability", "type": "numeric",
                     "format": {"specifier": ".1%"}},
                    {"name": "Confidence",  "id": "Confidence"},
                    {"name": "Park",        "id": "Park_Factor",   "type": "numeric"},
                    {"name": "Temp °F",     "id": "Temp_F",        "type": "numeric"},
                    {"name": "Wind",        "id": "wind_display"},
                    {"name": "Home?",       "id": "home_display"},
                ],
                data=[],
                sort_action="native",
                filter_action="native",
                page_size=30,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#0d1b2a",
                    "color": "white",
                    "fontWeight": "600",
                    "textAlign": "center",
                    "padding": "11px 10px",
                    "fontSize": "12px",
                    "letterSpacing": ".4px",
                    "textTransform": "uppercase",
                    "borderBottom": "2px solid #f59e0b",
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "9px 12px",
                    "fontFamily": "Inter, sans-serif",
                    "fontSize": "13px",
                    "border": "1px solid #f1f5f9",
                    "color": "#1e293b",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Player"}, "textAlign": "left", "fontWeight": "600"},
                    {"if": {"column_id": "rank"},   "width": "48px"},
                ],
                style_data_conditional=[
                    {"if": {"filter_query": '{Confidence} = "High"'},
                     "backgroundColor": "#f0fdf4", "color": "#065f46"},
                    {"if": {"filter_query": '{Confidence} = "Medium"'},
                     "backgroundColor": "#fffbeb", "color": "#92400e"},
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#f8fafc"},
                ],
            ),
        ], className="table-container"),

        # ── Probability distribution ─────────────────────────────────────────
        html.Div([
            html.H3("HR Probability Distribution", className="section-title"),
            dcc.Graph(id="prob-distribution-chart", config={"displayModeBar": False}),
        ], className="chart-section"),

        dcc.Store(id="predictions-store"),
        dcc.Interval(id="refresh-interval", interval=5 * 60 * 1000, n_intervals=0),
    ], className="page-container")


# ─── Data loader ──────────────────────────────────────────────────────────────

def _load_todays_predictions() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        return pd.DataFrame()
    try:
        wb = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
        ws = wb["Predictions"]
        data = list(ws.iter_rows(min_row=2, values_only=True))
        wb.close()
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=PRED_COLUMNS)
    today_str = date.today().isoformat()
    df = df[df["Date"] == today_str].copy()
    if df.empty:
        return df

    df["HR_Probability"] = pd.to_numeric(df["HR_Probability"], errors="coerce")
    df = (
        df.sort_values("HR_Probability", ascending=False)
          .drop_duplicates(subset=["Player", "Team", "Pitcher"], keep="first")
          .reset_index(drop=True)
    )
    return df


# ─── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("predictions-store", "data"),
    Output("team-filter", "options"),
    Input("refresh-interval", "n_intervals"),
)
def load_predictions(_n):
    df = _load_todays_predictions()
    if df.empty:
        return [], [{"label": "All Teams", "value": "All"}]
    teams = [{"label": "All Teams", "value": "All"}] + [
        {"label": t, "value": t} for t in sorted(df["Team"].dropna().unique())
    ]
    return df.to_dict("records"), teams


@callback(
    Output("predictions-table", "data"),
    Output("summary-cards", "children"),
    Output("prob-distribution-chart", "figure"),
    Output("picks-grid", "children"),
    Input("predictions-store", "data"),
    Input("conf-filter", "value"),
    Input("team-filter", "value"),
    Input("wind-filter", "value"),
)
def update_table(data, conf_filter, team_filter, wind_filter):
    if not data:
        return [], _empty_cards(), _empty_chart(), _empty_picks()

    df = pd.DataFrame(data)
    df["HR_Probability"] = pd.to_numeric(df["HR_Probability"], errors="coerce")

    if conf_filter == "High":
        df = df[df["Confidence"] == "High"]
    elif conf_filter == "High+Medium":
        df = df[df["Confidence"].isin(["High", "Medium"])]
    if team_filter != "All":
        df = df[df["Team"] == team_filter]
    if wind_filter != "All":
        df = df[df["Wind_Direction"] == wind_filter]

    df = df.sort_values("HR_Probability", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["wind_display"] = (
        df["Wind_Speed_MPH"].fillna(0).astype(int).astype(str) + " mph " +
        df["Wind_Direction"].fillna("calm")
    )
    df["home_display"] = df["Home_Game"].map(
        {True: "Home", False: "Away", 1: "Home", 0: "Away"}
    )

    table_data = df[[
        "rank", "Player", "Team", "Opponent", "Pitcher", "HR_Probability",
        "Confidence", "Park_Factor", "Temp_F", "wind_display", "home_display",
    ]].to_dict("records")

    picks = _build_picks_grid(df.head(8))
    return table_data, _build_cards(df), _build_dist_chart(df), picks


# ─── Pick cards ───────────────────────────────────────────────────────────────

def _build_picks_grid(df: pd.DataFrame):
    if df.empty:
        return _empty_picks()

    try:
        from utils.player_photos import batch_headshot_urls
        photo_map = batch_headshot_urls(df["Player"].tolist(), width=90)
    except Exception:
        photo_map = {}

    cards = []
    for _, row in df.iterrows():
        conf   = row.get("Confidence", "Low") or "Low"
        prob   = row.get("HR_Probability", 0) or 0
        name   = str(row.get("Player", ""))
        team   = str(row.get("Team", ""))
        opp    = str(row.get("Opponent", ""))
        pitcher = str(row.get("Pitcher", ""))
        wind   = str(row.get("Wind_Direction", "calm") or "calm")
        temp   = row.get("Temp_F", "—")
        photo  = photo_map.get(name, "")

        badge_cls = "badge badge-high" if conf == "High" else (
            "badge badge-medium" if conf == "Medium" else "badge badge-low")

        card = html.Div([
            html.Img(
                src=photo, alt=name,
                className="pick-card-photo",
                loading="lazy",
                style={"background": "#e2e8f0"},
            ),
            html.P(name, className="pick-card-name"),
            html.P(f"{team}  vs  {opp}" if opp else team, className="pick-card-team"),
            html.Div(f"{prob:.1%}", className="pick-card-prob"),
            html.P(f"vs {pitcher}", className="pick-card-matchup")
            if pitcher else None,
            html.Div([
                html.Span(conf, className=badge_cls),
                html.Span(f"🌡 {int(temp)}°" if temp and temp != "—" else "",
                          style={"fontSize": "11px", "color": "#94a3b8"}),
                html.Span(f"💨 {wind}",
                          style={"fontSize": "11px", "color": "#94a3b8"}),
            ], className="pick-card-footer"),
        ], className=f"pick-card {conf.lower()}")
        cards.append(card)

    return cards


def _empty_picks():
    return [html.Div([
        html.Div("⚾", className="empty-state-icon"),
        html.Div("No predictions yet for today", className="empty-state-msg"),
        html.Div("The scheduler runs at 9:00 AM", className="empty-state-sub"),
    ], className="empty-state")]


# ─── Summary cards ────────────────────────────────────────────────────────────

def _build_cards(df: pd.DataFrame):
    total = len(df)
    high  = int((df["Confidence"] == "High").sum())
    med   = int((df["Confidence"] == "Medium").sum())
    avg_p = float(df["HR_Probability"].mean()) if total > 0 else 0

    def card(title, value, cls=""):
        return html.Div([
            html.Div(str(value), className="summary-card-value",
                     style={"color": "#1e40af" if not cls else
                            "#10b981" if cls == "green" else
                            "#f97316" if cls == "orange" else "#f59e0b"}),
            html.Div(title, className="summary-card-label"),
        ], className=f"summary-card {cls}")

    return [
        card("Total Players", total),
        card("High Confidence", high, "green"),
        card("Medium Confidence", med, "orange"),
        card("Avg HR Probability", f"{avg_p:.1%}", "amber"),
    ]


def _empty_cards():
    return [html.Div(
        "No predictions yet for today — scheduler runs at 9:00 AM.",
        style={"color": "#94a3b8", "padding": "16px", "fontStyle": "italic", "fontSize": "14px"},
    )]


# ─── Distribution chart ───────────────────────────────────────────────────────

def _build_dist_chart(df: pd.DataFrame):
    if df.empty:
        return _empty_chart()
    fig = px.histogram(
        df, x="HR_Probability", color="Confidence",
        color_discrete_map=CONF_COLORS,
        nbins=20,
        labels={"HR_Probability": "HR Probability", "count": "Players"},
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        legend_title_text="Confidence",
        margin=dict(l=40, r=20, t=10, b=40),
        height=260, bargap=0.08,
        font={"family": "Inter, sans-serif", "size": 12},
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def _empty_chart():
    fig = go.Figure()
    fig.add_annotation(
        text="No predictions loaded yet",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font={"size": 14, "color": "#94a3b8"},
    )
    fig.update_layout(height=260, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                      margin=dict(l=40, r=20, t=10, b=40))
    return fig
