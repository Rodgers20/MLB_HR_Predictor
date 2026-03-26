"""
Player Deep-Dive page.
Shows individual player Statcast stats, trend charts, and HR probability breakdown by matchup.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from utils.data_fetcher import (
    fetch_fangraphs_batting,
    fetch_statcast_batter_leaderboard,
)
from utils.model_trainer import load_feature_importance

CURRENT_YEAR = 2025


def layout():
    return html.Div([
        html.Div([
            html.H2("Player Deep-Dive", className="page-title"),
            html.P("Individual batter Statcast analysis and HR probability breakdown",
                   className="page-subtitle"),
        ], className="page-header"),

        # Search
        html.Div([
            dcc.Dropdown(id="player-search-dropdown",
                         placeholder="Search for a player...",
                         clearable=True, searchable=True,
                         className="dropdown", style={"width": "350px"}),
            html.Button("Analyze", id="analyze-btn", className="btn-primary"),
        ], className="filters-row"),

        html.Div(id="player-content"),
        dcc.Interval(id="player-refresh", interval=60 * 60 * 1000, n_intervals=0),
    ], className="page-container")


@callback(
    Output("player-search-dropdown", "options"),
    Input("player-refresh", "n_intervals"),
)
def load_player_list(_n):
    try:
        df = fetch_fangraphs_batting(CURRENT_YEAR)
        if df is not None and not df.empty:
            names = sorted(df["Name"].dropna().unique())
            return [{"label": n, "value": n} for n in names]
    except Exception:
        pass
    return []


@callback(
    Output("player-content", "children"),
    Input("analyze-btn", "n_clicks"),
    State("player-search-dropdown", "value"),
    prevent_initial_call=True,
)
def show_player(n_clicks, player_name):
    if not player_name:
        return html.Div("Select a player to analyze", style={"color": "#888", "padding": "20px"})

    try:
        fg_df = fetch_fangraphs_batting(CURRENT_YEAR)
        player_row = fg_df[fg_df["Name"] == player_name]
        if player_row.empty:
            return html.Div(f"No data found for {player_name}")
        player = player_row.iloc[0]
    except Exception as exc:
        return html.Div(f"Error loading player data: {exc}")

    # Build stat cards
    stat_cards = _stat_cards(player)

    # Build multi-year trend
    trend_chart = _trend_chart(player_name)

    # Park factor impact table
    park_chart = _park_factor_chart(player)

    # ── Player photo header ─────────────────────────────────────────────────
    try:
        from utils.player_photos import headshot_url
        photo = headshot_url(player_name, width=120)
    except Exception:
        photo = ""

    team_name = str(player.get("Team", "")) if "Team" in player else ""

    profile = html.Div([
        html.Img(src=photo, alt=player_name, className="player-profile-photo"),
        html.Div([
            html.H3(player_name, className="player-profile-name"),
            html.Div(team_name, className="player-profile-meta"),
            html.Div([
                html.Span(f"HR: {player.get('HR', '—')}",
                          style={"marginRight": "12px", "fontWeight": "600",
                                 "color": "#f59e0b", "fontFamily": "'Fira Code', monospace"}),
                html.Span(f"wRC+: {player.get('wRC+', '—')}",
                          style={"marginRight": "12px", "color": "#475569"}),
                html.Span(f"ISO: {player.get('ISO', 0):.3f}" if player.get("ISO") else "",
                          style={"color": "#475569"}),
            ], style={"marginTop": "8px", "fontSize": "14px"}),
        ]),
    ], className="player-profile")

    return html.Div([
        profile,
        html.Div(stat_cards, className="summary-cards"),
        html.Div([
            html.H4("Season Trends", className="section-title"),
            dcc.Graph(figure=trend_chart, config={"displayModeBar": False}),
        ], className="chart-section"),
        html.Div([
            html.H4("HR Probability by Park", className="section-title"),
            dcc.Graph(figure=park_chart, config={"displayModeBar": False}),
        ], className="chart-section"),
    ])


def _stat_cards(player: pd.Series):
    stats = [
        ("HR (2025)", player.get("HR", "—")),
        ("Barrel %", f"{player.get('Barrel%', 0):.1f}%" if player.get("Barrel%") else "—"),
        ("Hard Hit %", f"{player.get('HardHit%', 0):.1f}%" if player.get("HardHit%") else "—"),
        ("ISO", f"{player.get('ISO', 0):.3f}" if player.get("ISO") else "—"),
        ("HR/FB", f"{player.get('HR/FB', 0):.1f}%" if player.get("HR/FB") else "—"),
        ("wRC+", player.get("wRC+", "—")),
        ("Pull %", f"{player.get('Pull%', 0):.1f}%" if player.get("Pull%") else "—"),
        ("FB %", f"{player.get('FB%', 0):.1f}%" if player.get("FB%") else "—"),
    ]
    cards = []
    for label, value in stats:
        cards.append(html.Div([
            html.H4(str(value), style={"margin": "0", "color": "#1F4E79", "fontSize": "22px"}),
            html.P(label, style={"margin": "4px 0 0", "fontSize": "12px", "color": "#666"}),
        ], className="summary-card"))
    return cards


def _trend_chart(player_name: str):
    """Multi-year trend for HR, Barrel%, ISO."""
    years = list(range(2021, CURRENT_YEAR + 1))
    hr_vals, barrel_vals, iso_vals = [], [], []

    for yr in years:
        try:
            df = fetch_fangraphs_batting(yr)
            row = df[df["Name"] == player_name]
            if not row.empty:
                r = row.iloc[0]
                hr_vals.append(r.get("HR", np.nan))
                barrel_vals.append(r.get("Barrel%", np.nan))
                iso_vals.append(r.get("ISO", np.nan))
            else:
                hr_vals.append(np.nan)
                barrel_vals.append(np.nan)
                iso_vals.append(np.nan)
        except Exception:
            hr_vals.append(np.nan)
            barrel_vals.append(np.nan)
            iso_vals.append(np.nan)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=hr_vals, name="Home Runs",
                         marker_color="#1F4E79", yaxis="y"))
    fig.add_trace(go.Scatter(x=years, y=barrel_vals, name="Barrel %",
                             mode="lines+markers", line={"color": "#00B050"},
                             yaxis="y2"))
    fig.add_trace(go.Scatter(x=years, y=iso_vals, name="ISO",
                             mode="lines+markers", line={"color": "#FF8C00"},
                             yaxis="y2"))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        height=300, margin=dict(l=60, r=60, t=20, b=40),
        yaxis={"title": "HR Count", "side": "left"},
        yaxis2={"title": "Rate", "side": "right", "overlaying": "y", "tickformat": ".1f"},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def _park_factor_chart(player: pd.Series):
    """Show how HR probability changes across different parks."""
    from utils.data_fetcher import load_park_factors
    park_factors = load_park_factors()

    hr_rate = float(player.get("HR", 0)) / max(float(player.get("PA", 1)), 1)
    base_hr_rate = hr_rate

    park_factors = park_factors.sort_values("hr_park_factor", ascending=False)
    adjusted = base_hr_rate * park_factors["hr_park_factor"] / 100

    fig = go.Figure(go.Bar(
        x=park_factors["team"], y=adjusted,
        marker_color=["#00B050" if v > base_hr_rate * 1.1
                      else "#dc3545" if v < base_hr_rate * 0.9
                      else "#1F4E79"
                      for v in adjusted],
        text=[f"{v:.3f}" for v in adjusted],
        textposition="outside",
    ))
    fig.add_hline(y=base_hr_rate, line_dash="dash", line_color="gray",
                  annotation_text="Season avg")
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        height=320, margin=dict(l=60, r=20, t=20, b=60),
        yaxis={"title": "Adjusted HR Rate per PA", "tickformat": ".3f"},
        xaxis={"title": "Stadium (Home Team)", "tickangle": -45},
    )
    return fig
