"""
Player Deep-Dive page.

Sections (top → bottom):
  1. Player profile header (photo, team, key FanGraphs stats)
  2. Current-season stat banner (live AVG/OBP/SLG/OPS/HR/K%/BB%/EV from Statcast)
  3. Historical summary cards (FanGraphs: Barrel%, HardHit%, ISO, HR/FB, wRC+, Pull%, FB%)
  4. At-Bat Log
       a. Result breakdown donut
       b. Exit-velocity vs launch-angle scatter (with barrel zone)
       c. Season-progress chart (cumulative HR + rolling AVG)
       d. Plate-appearance table (most recent first, color-coded by result)
  5. Career trends chart (HR / Barrel% / ISO by year)
  6. HR probability by park
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

from utils.data_fetcher import fetch_fangraphs_batting, fetch_statcast_batter_leaderboard
from utils.model_trainer import load_feature_importance

# Always use the live season year — no more hardcoding 2025
CURRENT_YEAR = date.today().year
ROSTER_YEAR  = CURRENT_YEAR


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(player_name: str | None = None, pitcher_name: str | None = None):
    """Render the player deep-dive page.

    Args:
        player_name:  When navigating from Today's picks, pre-populate the
                      dropdown and auto-trigger analysis for this player.
        pitcher_name: Today's opposing pitcher — used to show H2H career stats.
    """
    return html.Div([
        html.Div([
            html.H2("Player Deep-Dive", className="page-title"),
            html.P("Individual batter Statcast analysis, at-bat log, and HR probability breakdown",
                   className="page-subtitle"),
        ], className="page-header"),

        html.Div([
            dcc.Dropdown(
                id="player-search-dropdown",
                placeholder="Search for a player…",
                value=player_name,          # pre-filled when navigating from picks
                clearable=True, searchable=True,
                className="dropdown", style={"width": "350px"},
            ),
            html.Button("Analyze", id="analyze-btn", className="btn-primary"),
        ], className="filters-row"),

        # Stores trigger auto-analysis when navigating from Today's picks
        dcc.Store(id="player-autoload",  data=player_name),
        dcc.Store(id="pitcher-autoload", data=pitcher_name),

        dcc.Loading(
            html.Div(id="player-content"),
            type="circle",
            color="#ff6b00",
        ),
        dcc.Interval(id="player-refresh", interval=60 * 60 * 1000, n_intervals=0),
    ], className="page-container")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("player-search-dropdown", "options"),
    Input("player-refresh", "n_intervals"),
)
def load_player_list(_n):
    for yr in [CURRENT_YEAR, CURRENT_YEAR - 1]:
        try:
            df = fetch_fangraphs_batting(yr)
            if df is not None and not df.empty:
                names = sorted(df["Name"].dropna().unique())
                return [{"label": n, "value": n} for n in names]
        except Exception:
            pass
    return []


@callback(
    Output("player-content", "children"),
    Input("analyze-btn", "n_clicks"),
    Input("player-autoload", "data"),
    State("player-search-dropdown", "value"),
    State("pitcher-autoload", "data"),
    prevent_initial_call=True,
)
def show_player(_n_clicks, autoload_name, player_name, pitcher_name):
    # autoload_name is set when navigating from Today's picks; prefer it over
    # the dropdown state so the analysis fires immediately on page load.
    player_name = autoload_name or player_name
    if not player_name:
        return html.Div("Select a player to analyze",
                        style={"color": "#888", "padding": "20px"})

    # ── FanGraphs season data (current year preferred) ─────────────────────
    player_row = pd.Series()
    for yr in [CURRENT_YEAR, CURRENT_YEAR - 1]:
        try:
            fg_df = fetch_fangraphs_batting(yr)
            row = fg_df[fg_df["Name"] == player_name]
            if not row.empty:
                player_row = row.iloc[0]
                break
        except Exception:
            pass

    if player_row.empty:
        return html.Div(f"No FanGraphs data found for {player_name}",
                        style={"color": "#888", "padding": "20px"})

    # ── Live roster team ───────────────────────────────────────────────────
    fg_team = str(player_row.get("Team", "")) if "Team" in player_row else ""
    try:
        from utils.roster_fetcher import get_current_roster_map
        roster_map = get_current_roster_map(ROSTER_YEAR)
        team_name  = roster_map.get(player_name.lower(), fg_team) or fg_team
    except Exception:
        team_name = fg_team

    # ── Player photo ───────────────────────────────────────────────────────
    try:
        from utils.player_photos import headshot_url
        photo = headshot_url(player_name, width=120)
    except Exception:
        photo = ""

    profile = html.Div([
        html.Img(src=photo, alt=player_name, className="player-profile-photo") if photo
        else html.Div(player_name[:2].upper(), className="player-profile-photo",
                      style={"display": "flex", "alignItems": "center",
                             "justifyContent": "center", "fontSize": "24px",
                             "fontWeight": "900", "color": "#ff6b00",
                             "background": "#1e2023"}),
        html.Div([
            html.H3(player_name, className="player-profile-name"),
            html.Div(team_name, className="player-profile-meta"),
            html.Div([
                html.Span(f"HR: {player_row.get('HR', '—')}",
                          style={"marginRight": "12px", "fontWeight": "700",
                                 "color": "#ff6b00", "fontFamily": "Manrope, sans-serif"}),
                html.Span(f"wRC+: {player_row.get('wRC+', '—')}",
                          style={"marginRight": "12px", "color": "#8e909c"}),
                html.Span(
                    f"ISO: {float(player_row.get('ISO', 0)):.3f}" if player_row.get("ISO") else "",
                    style={"color": "#8e909c"},
                ),
            ], style={"marginTop": "8px", "fontSize": "14px"}),
        ]),
    ], className="player-profile")

    # ── H2H callout (only when navigating from Today's picks) ─────────────
    h2h_card = html.Div()
    if pitcher_name:
        try:
            from utils.ab_log_fetcher import (
                compute_h2h, fetch_player_ab_log, format_h2h_line,
            )
            ab_log_full = fetch_player_ab_log(player_name, CURRENT_YEAR)
            h2h  = compute_h2h(ab_log_full, pitcher_name)
            line = format_h2h_line(pitcher_name, h2h)
        except Exception:
            line = f"vs {pitcher_name}: career data unavailable"

        h2h_card = html.Div([
            html.Span("Today's Matchup", style={
                "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
                "textTransform": "uppercase", "color": "#ff6b00", "marginBottom": "4px",
                "display": "block",
            }),
            html.Span(line, style={
                "fontFamily": "'Manrope', sans-serif", "fontSize": "14px",
                "fontWeight": "700", "color": "#f0f4f8",
            }),
        ], style={
            "background": "rgba(255,107,0,.08)",
            "border": "1px solid rgba(255,107,0,.25)",
            "borderRadius": "10px",
            "padding": "14px 18px",
            "marginBottom": "16px",
        })

    return html.Div([
        profile,
        h2h_card,
        _current_season_banner(player_name),
        html.Div(_stat_cards(player_row), className="summary-cards"),
        _ab_log_section(player_name),
        html.Div([
            html.H4("Home Runs by Season", className="section-title"),
            html.P("Hover each bar for Barrel%, Hard Hit%, ISO, wRC+",
                   style={"color": "#8e909c", "fontSize": "12px", "marginBottom": "8px"}),
            dcc.Graph(figure=_trend_chart(player_name), config={"displayModeBar": False}),
        ], className="chart-section"),
        html.Div([
            html.H4("Park Impact on Home Runs", className="section-title"),
            html.P("Best and worst parks for this player's HR profile vs league average",
                   style={"color": "#8e909c", "fontSize": "12px", "marginBottom": "8px"}),
            dcc.Graph(figure=_park_factor_chart(player_row), config={"displayModeBar": False}),
        ], className="chart-section"),
    ])


# ---------------------------------------------------------------------------
# Current-season live banner
# ---------------------------------------------------------------------------

def _current_season_banner(player_name: str):
    """Fetch Statcast AB log and display live season stats."""
    try:
        from utils.ab_log_fetcher import fetch_player_ab_log, get_season_stats
        ab_log = fetch_player_ab_log(player_name, CURRENT_YEAR)
        stats  = get_season_stats(ab_log)
    except Exception:
        ab_log = pd.DataFrame()
        stats  = {}

    if not stats:
        return html.Div()

    pa = stats.get("PA", 0)
    sample_note = " ⚠ Small sample" if pa < 20 else ""
    season_label = f"{CURRENT_YEAR} Season · {pa} PA{sample_note}"

    def chip(label, value, highlight=False):
        return html.Div([
            html.Span(str(value),
                      style={"fontSize": "20px", "fontWeight": "800",
                             "color": "#ff6b00" if highlight else "#f2f2f2",
                             "fontFamily": "Manrope, sans-serif"}),
            html.Span(label,
                      style={"fontSize": "11px", "color": "#8e909c",
                             "textTransform": "uppercase", "letterSpacing": "1px",
                             "display": "block", "marginTop": "2px"}),
        ], style={"textAlign": "center", "padding": "12px 16px",
                  "background": "#1a1d1f", "borderRadius": "8px",
                  "border": "1px solid rgba(255,255,255,.06)"})

    def fmt(v, decimals=3):
        if v is None:
            return "—"
        return f".{int(round(float(v), decimals) * 10**decimals):0{decimals}d}" \
               if decimals == 3 else str(v)

    avg_str  = f".{int(round(stats['AVG'], 3) * 1000):03d}"  if stats.get("AVG")  else "—"
    obp_str  = f".{int(round(stats['OBP'], 3) * 1000):03d}"  if stats.get("OBP")  else "—"
    slg_str  = f".{int(round(stats['SLG'], 3) * 1000):03d}"  if stats.get("SLG")  else "—"
    ops_str  = f"{stats['OPS']:.3f}"                          if stats.get("OPS")  else "—"
    babip_str= f".{int(round(stats['BABIP'], 3) * 1000):03d}" if stats.get("BABIP") else "—"
    ev_str   = f"{stats['avg_ev']} mph"                        if stats.get("avg_ev") else "—"
    la_str   = f"{stats['avg_la']}°"                           if stats.get("avg_la") is not None else "—"
    hh_str   = f"{stats['hard_hit_pct']}%"                    if stats.get("hard_hit_pct") else "—"
    kp_str   = f"{stats['K_pct']}%"                           if stats.get("K_pct") is not None else "—"
    bbp_str  = f"{stats['BB_pct']}%"                          if stats.get("BB_pct") is not None else "—"

    return html.Div([
        html.Div([
            html.Span(season_label, style={
                "fontSize": "11px", "fontWeight": "700", "textTransform": "uppercase",
                "letterSpacing": "2px", "color": "#ff6b00",
            }),
        ], style={"marginBottom": "12px"}),
        html.Div([
            chip("AVG",        avg_str,  highlight=True),
            chip("OBP",        obp_str),
            chip("SLG",        slg_str),
            chip("OPS",        ops_str,  highlight=True),
            chip("HR",         stats.get("HR", "—"), highlight=True),
            chip("2B",         stats.get("2B", "—")),
            chip("3B",         stats.get("3B", "—")),
            chip("BB",         stats.get("BB", "—")),
            chip("K",          stats.get("K",  "—")),
            chip("K%",         kp_str),
            chip("BB%",        bbp_str),
            chip("BABIP",      babip_str),
            chip("Avg EV",     ev_str),
            chip("Avg LA",     la_str),
            chip("Hard Hit%",  hh_str),
        ], style={
            "display": "flex", "flexWrap": "wrap", "gap": "8px",
        }),
    ], style={
        "padding": "20px", "background": "#131618",
        "borderRadius": "12px", "border": "1px solid rgba(255,107,0,.25)",
        "marginBottom": "16px",
    })


# ---------------------------------------------------------------------------
# At-bat log section
# ---------------------------------------------------------------------------

def _ab_log_section(player_name: str):
    """Full current-season at-bat log: charts + table."""
    try:
        from utils.ab_log_fetcher import fetch_player_ab_log
        ab_log = fetch_player_ab_log(player_name, CURRENT_YEAR)
    except Exception as exc:
        return html.Div(f"Could not load at-bat log: {exc}",
                        style={"color": "#888", "padding": "12px"})

    if ab_log.empty:
        return html.Div([
            html.H4(f"{CURRENT_YEAR} At-Bat Log", className="section-title"),
            html.P("No at-bat data available yet for this season.",
                   style={"color": "#8e909c", "padding": "12px"}),
        ], className="chart-section")

    pa_count = len(ab_log)

    return html.Div([
        html.H4(f"{CURRENT_YEAR} At-Bat Log — {pa_count} Plate Appearances",
                className="section-title"),
        html.P(
            "Every plate appearance this season. Color coding: "
            "🟠 HR  🟢 Hit  🔵 Walk/HBP  🔴 Strikeout  ⬜ Out",
            style={"color": "#8e909c", "fontSize": "12px", "marginBottom": "16px"},
        ),

        # ── Row 1: donut + scatter ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.P("Result Breakdown", style=_subtitle_style()),
                dcc.Graph(figure=_result_donut(ab_log),
                          config={"displayModeBar": False},
                          style={"height": "360px"}),
            ], style={"flex": "0 0 38%"}),
            html.Div([
                html.P("Exit Velocity vs Launch Angle", style=_subtitle_style()),
                dcc.Graph(figure=_ev_la_scatter(ab_log),
                          config={"displayModeBar": False},
                          style={"height": "360px"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── Row 2: season progress ─────────────────────────────────────────
        html.Div([
            html.P("Season Progress", style=_subtitle_style()),
            dcc.Graph(figure=_cumulative_chart(ab_log),
                      config={"displayModeBar": False},
                      style={"height": "300px"}),
        ], style={"marginBottom": "16px"}),

        # ── Row 3: PA table ────────────────────────────────────────────────
        html.P("Plate Appearance Log (most recent first)", style=_subtitle_style()),
        _ab_log_table(ab_log),

    ], className="chart-section")


def _subtitle_style():
    return {
        "fontSize": "12px", "fontWeight": "700", "textTransform": "uppercase",
        "letterSpacing": "1.5px", "color": "#8e909c", "marginBottom": "8px",
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _result_donut(ab_log: pd.DataFrame):
    """Donut chart of AB result distribution."""
    # Group into broad categories
    cat_map = {
        "HR":   "HR",
        "3B":   "Hit", "2B": "Hit", "1B": "Hit",
        "BB":   "Walk/HBP", "IBB": "Walk/HBP", "HBP": "Walk/HBP",
        "K":    "Strikeout", "K-DP": "Strikeout",
        "GIDP": "Out", "DP": "Out", "Out": "Out",
        "FC":   "Out", "E": "Other", "SF": "Other",
        "SAC":  "Other", "CI": "Other",
    }
    cat_colors = {
        "HR":         "#ff6b00",
        "Hit":        "#22c55e",
        "Walk/HBP":   "#3b82f6",
        "Strikeout":  "#ef4444",
        "Out":        "#4b5563",
        "Other":      "#8e909c",
    }

    ab_log = ab_log.copy()
    ab_log["category"] = ab_log["result_label"].map(lambda r: cat_map.get(r, "Other"))
    counts = ab_log["category"].value_counts()

    order = ["HR", "Hit", "Walk/HBP", "Strikeout", "Out", "Other"]
    labels = [c for c in order if c in counts.index]
    values = [counts[c] for c in labels]
    colors = [cat_colors[c] for c in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont={"size": 11, "color": "#f2f2f2", "family": "Manrope, sans-serif"},
        hovertemplate="%{label}: %{value} PA (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#131618", plot_bgcolor="#131618",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        font={"family": "Manrope, sans-serif", "color": "#f2f2f2"},
    )
    return fig


def _ev_la_scatter(ab_log: pd.DataFrame):
    """Exit velocity vs launch angle scatter, colored by result."""
    # Only rows with batted-ball data
    batted = ab_log.dropna(subset=["launch_speed", "launch_angle"]).copy()

    fig = go.Figure()

    # ── Barrel zone shading ────────────────────────────────────────────────
    fig.add_shape(type="rect",
                  x0=98, x1=120, y0=26, y1=30,
                  fillcolor="rgba(255,107,0,.08)",
                  line={"color": "rgba(255,107,0,.3)", "width": 1, "dash": "dot"},
                  layer="below")
    fig.add_annotation(x=109, y=28, text="Barrel zone",
                       font={"size": 9, "color": "rgba(255,107,0,.6)"},
                       showarrow=False)

    if batted.empty:
        fig.update_layout(
            paper_bgcolor="#131618", plot_bgcolor="#131618",
            xaxis_title="Exit Velocity (mph)", yaxis_title="Launch Angle (°)",
            font={"family": "Manrope, sans-serif", "color": "#f2f2f2"},
            margin=dict(l=50, r=20, t=20, b=50),
        )
        return fig

    color_map = {
        "HR":   "#ff6b00", "3B":  "#22c55e", "2B":  "#22c55e", "1B":  "#22c55e",
        "BB":   "#3b82f6", "IBB": "#3b82f6", "HBP": "#3b82f6",
        "K":    "#ef4444", "K-DP":"#ef4444",
        "Out":  "#4b5563", "GIDP":"#4b5563", "DP":  "#4b5563", "FC":  "#4b5563",
        "E":    "#fbbf24", "SF":  "#8e909c", "SAC": "#8e909c",
    }

    for result_lbl in batted["result_label"].unique():
        subset = batted[batted["result_label"] == result_lbl]
        color  = color_map.get(result_lbl, "#8e909c")
        dist   = pd.to_numeric(subset.get("hit_distance_sc", pd.Series()), errors="coerce").fillna(0)
        size   = (dist / 15).clip(lower=6, upper=20)

        hover  = (
            "<b>" + result_lbl + "</b><br>" +
            "Date: "    + subset["game_date"].astype(str) + "<br>" +
            "vs: "      + subset["opponent"].astype(str)  + "<br>" +
            "Pitcher: " + subset["pitcher_name"].astype(str) + "<br>" +
            "EV: "      + subset["launch_speed"].round(1).astype(str) + " mph<br>" +
            "LA: "      + subset["launch_angle"].round(1).astype(str) + "°<br>" +
            "Dist: "    + dist.astype(int).astype(str) + " ft"
        )

        fig.add_trace(go.Scatter(
            x=subset["launch_speed"],
            y=subset["launch_angle"],
            mode="markers",
            name=result_lbl,
            marker={"color": color, "size": size, "opacity": 0.82,
                    "line": {"color": "rgba(255,255,255,.15)", "width": 0.5}},
            hovertemplate=hover + "<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#131618", plot_bgcolor="#101416",
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis={
            "title": "Exit Velocity (mph)", "gridcolor": "rgba(255,255,255,.05)",
            "color": "#8e909c", "range": [60, 120],
        },
        yaxis={
            "title": "Launch Angle (°)", "gridcolor": "rgba(255,255,255,.05)",
            "color": "#8e909c", "range": [-40, 60],
        },
        legend={
            "orientation": "h", "y": -0.20, "x": 0,
            "font": {"color": "#8e909c", "size": 10},
        },
        font={"family": "Manrope, sans-serif", "color": "#f2f2f2"},
    )
    return fig


def _cumulative_chart(ab_log: pd.DataFrame):
    """Cumulative HR count + rolling 10-PA batting average per game."""
    if "game_date" not in ab_log.columns or ab_log.empty:
        return go.Figure()

    ab_sorted = ab_log.sort_values("game_date", ascending=True).copy()

    # Per-game aggregates
    game_groups = ab_sorted.groupby("game_date").agg(
        pa=("events", "count"),
        hits=("events", lambda x: x.isin({"single", "double", "triple", "home_run"}).sum()),
        hrs=("events",  lambda x: (x == "home_run").sum()),
    ).reset_index().sort_values("game_date")

    game_groups["cum_hr"]  = game_groups["hrs"].cumsum()
    game_groups["cum_pa"]  = game_groups["pa"].cumsum()
    game_groups["cum_hit"] = game_groups["hits"].cumsum()

    # Rolling 10-PA avg (computed on PA-level data, then mapped back to game)
    ab_sorted["is_hit"] = ab_sorted["events"].isin(
        {"single", "double", "triple", "home_run"}
    ).astype(int)
    rolling_avg = (
        ab_sorted["is_hit"].rolling(10, min_periods=1).mean().values
    )
    # Take last value per game date
    ab_sorted["rolling_avg"] = rolling_avg
    roll_per_game = (
        ab_sorted.groupby("game_date")["rolling_avg"].last().reset_index()
    )
    game_groups = game_groups.merge(roll_per_game, on="game_date", how="left")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=game_groups["game_date"], y=game_groups["cum_hr"],
        name="Cumulative HR", mode="lines+markers",
        line={"color": "#ff6b00", "width": 2},
        marker={"size": 6, "color": "#ff6b00"},
        hovertemplate="Date: %{x}<br>HR to date: %{y}<extra></extra>",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=game_groups["game_date"], y=game_groups["rolling_avg"],
        name="Rolling 10-PA AVG", mode="lines",
        line={"color": "#22c55e", "width": 2, "dash": "dot"},
        hovertemplate="Date: %{x}<br>Rolling AVG (10 PA): %{y:.3f}<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="#131618", plot_bgcolor="#101416",
        margin=dict(l=55, r=65, t=20, b=70),
        xaxis={"color": "#8e909c", "gridcolor": "rgba(255,255,255,.05)"},
        yaxis={"title": "Cumulative HR", "color": "#8e909c",
               "gridcolor": "rgba(255,255,255,.05)"},
        yaxis2={"title": "Rolling AVG", "overlaying": "y", "side": "right",
                "tickformat": ".3f", "color": "#8e909c", "range": [0, 1]},
        legend={"orientation": "h", "y": -0.25, "x": 0,
                "font": {"color": "#8e909c", "size": 10}},
        font={"family": "Manrope, sans-serif", "color": "#f2f2f2"},
    )
    return fig


# ---------------------------------------------------------------------------
# PA table
# ---------------------------------------------------------------------------

def _ab_log_table(ab_log: pd.DataFrame):
    """Dash DataTable of plate appearances."""
    display_cols = {
        "game_date":                        "Date",
        "opponent":                         "Opp",
        "pitcher_name":                     "Pitcher",
        "p_throws":                         "Hand",
        "result_label":                     "Result",
        "count":                            "Count",
        "inning":                           "Inn",
        "outs_when_up":                     "Outs",
        "launch_speed":                     "EV (mph)",
        "launch_angle":                     "LA (°)",
        "hit_distance_sc":                  "Dist (ft)",
        "bb_type_short":                    "Type",
        "estimated_ba_using_speedangle":    "xBA",
    }

    available = {k: v for k, v in display_cols.items() if k in ab_log.columns}
    table_df  = ab_log[list(available.keys())].copy()
    table_df.columns = list(available.values())

    # Round numeric columns
    for col in ["EV (mph)", "LA (°)", "Dist (ft)", "xBA"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(1)
            table_df[col] = table_df[col].apply(lambda v: "—" if pd.isna(v) else v)

    # Result → color tag via style_data_conditional
    result_style = []
    color_rules = {
        "HR":   ("rgba(255,107,0,.18)",  "#ff6b00"),
        "3B":   ("rgba(34,197,94,.12)",  "#22c55e"),
        "2B":   ("rgba(34,197,94,.12)",  "#22c55e"),
        "1B":   ("rgba(34,197,94,.10)",  "#22c55e"),
        "BB":   ("rgba(59,130,246,.12)", "#3b82f6"),
        "IBB":  ("rgba(59,130,246,.12)", "#3b82f6"),
        "HBP":  ("rgba(59,130,246,.12)", "#3b82f6"),
        "K":    ("rgba(239,68,68,.12)",  "#ef4444"),
        "K-DP": ("rgba(239,68,68,.12)",  "#ef4444"),
    }
    for label, (bg, _fg) in color_rules.items():
        result_style.append({
            "if": {"filter_query": f'{{Result}} = "{label}"'},
            "backgroundColor": bg,
        })

    columns = [{"name": c, "id": c} for c in table_df.columns]

    return dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#131618",
            "color": "#f2f2f2",
            "border": "1px solid rgba(255,255,255,.06)",
            "fontFamily": "Manrope, Inter, sans-serif",
            "fontSize": "12px",
            "padding": "8px 12px",
            "textAlign": "center",
            "whiteSpace": "nowrap",
        },
        style_header={
            "backgroundColor": "#1a1d1f",
            "color": "#8e909c",
            "fontWeight": "700",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "border": "1px solid rgba(255,255,255,.08)",
        },
        style_data_conditional=result_style + [
            {"if": {"row_index": "odd"}, "backgroundColor": "#111314"},
        ],
        style_filter={
            "backgroundColor": "#1a1d1f",
            "color": "#f2f2f2",
            "border": "1px solid rgba(255,255,255,.08)",
        },
    )


# ---------------------------------------------------------------------------
# Historical summary cards
# ---------------------------------------------------------------------------

def _stat_cards(player: pd.Series):
    yr_label = f"{CURRENT_YEAR}"
    stats = [
        (f"HR ({yr_label})",   player.get("HR",      "—")),
        ("Barrel %",           f"{float(player.get('Barrel%', 0)):.1f}%"  if player.get("Barrel%")  else "—"),
        ("Hard Hit %",         f"{float(player.get('HardHit%', 0)):.1f}%" if player.get("HardHit%") else "—"),
        ("ISO",                f"{float(player.get('ISO', 0)):.3f}"        if player.get("ISO")       else "—"),
        ("HR/FB",              f"{float(player.get('HR/FB', 0)):.1f}%"    if player.get("HR/FB")     else "—"),
        ("wRC+",               player.get("wRC+", "—")),
        ("Pull %",             f"{float(player.get('Pull%', 0)):.1f}%"    if player.get("Pull%")     else "—"),
        ("FB %",               f"{float(player.get('FB%', 0)):.1f}%"      if player.get("FB%")       else "—"),
    ]
    return [
        html.Div([
            html.H4(str(value), style={"margin": "0", "color": "#ff6b00", "fontSize": "22px",
                                       "fontFamily": "Manrope, sans-serif", "fontWeight": "800"}),
            html.P(label, style={"margin": "4px 0 0", "fontSize": "12px", "color": "#8e909c"}),
        ], className="summary-card")
        for label, value in stats
    ]


# ---------------------------------------------------------------------------
# Career trend chart
# ---------------------------------------------------------------------------

def _trend_chart(player_name: str):
    """Year-by-year HR totals. Tooltip includes Barrel%, Hard Hit%, ISO, wRC+."""
    years = list(range(2021, CURRENT_YEAR + 1))
    rows  = []

    for yr in years:
        try:
            df  = fetch_fangraphs_batting(yr)
            row = df[df["Name"] == player_name]
            if not row.empty:
                r = row.iloc[0]
                rows.append({
                    "year":     yr,
                    "hr":       r.get("HR",       np.nan),
                    "barrel":   r.get("Barrel%",  np.nan),
                    "hard_hit": r.get("HardHit%", np.nan),
                    "iso":      r.get("ISO",       np.nan),
                    "wrc":      r.get("wRC+",      np.nan),
                    "pa":       r.get("PA",        np.nan),
                })
            else:
                rows.append({"year": yr, "hr": np.nan, "barrel": np.nan,
                             "hard_hit": np.nan, "iso": np.nan, "wrc": np.nan, "pa": np.nan})
        except Exception:
            rows.append({"year": yr, "hr": np.nan, "barrel": np.nan,
                         "hard_hit": np.nan, "iso": np.nan, "wrc": np.nan, "pa": np.nan})

    data   = pd.DataFrame(rows)
    labels = data["year"].astype(str).tolist()
    hr_vals = data["hr"].tolist()

    hover = [
        f"<b>{r['year']}</b><br>"
        f"HR: {int(r['hr']) if pd.notna(r['hr']) else '—'}<br>"
        f"Barrel%: {r['barrel']:.1f}%" if pd.notna(r.get('barrel')) else
        f"<b>{r['year']}</b><br>HR: —"
        for _, r in data.iterrows()
    ]
    # Build richer hover text
    hover = []
    for _, r in data.iterrows():
        hr_s  = str(int(r["hr"]))   if pd.notna(r["hr"])     else "—"
        bar_s = f"{r['barrel']:.1f}%"  if pd.notna(r["barrel"])   else "—"
        hh_s  = f"{r['hard_hit']:.1f}%" if pd.notna(r["hard_hit"]) else "—"
        iso_s = f"{r['iso']:.3f}"   if pd.notna(r["iso"])    else "—"
        wrc_s = str(int(r["wrc"]))  if pd.notna(r["wrc"])    else "—"
        pa_s  = str(int(r["pa"]))   if pd.notna(r["pa"])     else "—"
        hover.append(
            f"<b>{int(r['year'])}</b><br>"
            f"HR: {hr_s}  ·  PA: {pa_s}<br>"
            f"Barrel%: {bar_s}  ·  Hard Hit%: {hh_s}<br>"
            f"ISO: {iso_s}  ·  wRC+: {wrc_s}"
        )

    bar_colors = [
        "#ff6b00" if (not np.isnan(v) and v == max(x for x in hr_vals if not np.isnan(x)))
        else "#3b82f6"
        for v in hr_vals
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=hr_vals,
        marker_color=bar_colors,
        text=[str(int(v)) if pd.notna(v) else "" for v in hr_vals],
        textposition="outside",
        textfont={"color": "#f2f2f2", "size": 12},
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))
    fig.update_layout(
        paper_bgcolor="#151515", plot_bgcolor="#101416",
        height=360, margin=dict(l=50, r=20, t=50, b=50),
        font={"family": "Manrope, Inter, sans-serif", "size": 12, "color": "#f2f2f2"},
        yaxis={"title": "Home Runs", "gridcolor": "rgba(255,255,255,.06)",
               "color": "#8e909c", "dtick": 5},
        xaxis={"color": "#8e909c"},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Park factor chart
# ---------------------------------------------------------------------------

def _park_factor_chart(player: pd.Series):
    """
    Top 5 HR-friendly and bottom 5 HR-suppressing parks.
    Y-axis shows % above/below league-average HR environment (100 = average).
    """
    from utils.data_fetcher import load_park_factors
    pf = load_park_factors().copy()

    # Delta vs average (0 = league average)
    pf["delta"] = pf["hr_park_factor"] - 100
    pf = pf.sort_values("delta", ascending=False)

    top5    = pf.head(5)
    bottom5 = pf.tail(5).sort_values("delta", ascending=True)
    subset  = pd.concat([top5, bottom5]).reset_index(drop=True)

    colors = ["#22c55e" if d > 0 else "#ef4444" for d in subset["delta"]]
    labels = [f"+{int(d)}%" if d > 0 else f"{int(d)}%" for d in subset["delta"]]

    fig = go.Figure(go.Bar(
        x=subset["team"],
        y=subset["delta"],
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont={"size": 11, "color": "#f2f2f2"},
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Park factor: %{customdata}<br>"
            "vs average: %{text}<extra></extra>"
        ),
        customdata=subset["hr_park_factor"],
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,.2)", line_width=1)
    fig.update_layout(
        paper_bgcolor="#151515", plot_bgcolor="#101416",
        height=400, margin=dict(l=60, r=20, t=40, b=90),
        font={"family": "Manrope, Inter, sans-serif", "size": 12, "color": "#f2f2f2"},
        yaxis={
            "title": "% vs League Average",
            "gridcolor": "rgba(255,255,255,.06)", "color": "#8e909c",
            "ticksuffix": "%",
        },
        xaxis={"color": "#8e909c", "tickangle": -35},
        showlegend=False,
        annotations=[{
            "x": 0.5, "y": 1.02, "xref": "paper", "yref": "paper",
            "text": "🟢 Hitter-friendly  🔴 Pitcher-friendly  (top 5 / bottom 5 only)",
            "showarrow": False,
            "font": {"size": 10, "color": "#8e909c"},
        }],
    )
    return fig
