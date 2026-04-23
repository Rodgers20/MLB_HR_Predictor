"""
Hit Predictor page — /hits
Predicts players most likely to record a hit today, broken down by:
  • Any Hit
  • Single (high payout)
  • Double (higher payout)
  • Triple

Plus 15 three-leg parlays across three betting categories:
  Group A (orange): 3 players × 2+ Total Bases each
  Group B (blue):   3 players × Any Hit each
  Group C (green):  3 players × mixed Singles/Doubles
"""

import math

import pandas as pd
from dash import Input, Output, callback, dcc, html

SEASON_YEAR = 2026

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    return html.Div([
        dcc.Store(id="hit-predictions-store"),
        dcc.Interval(id="hit-interval", interval=10 * 60 * 1000, n_intervals=0),

        # ── Page header ───────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Hit Predictor", style={
                    "margin": 0, "fontSize": "24px", "fontWeight": "900",
                    "background": "linear-gradient(135deg, #fff 40%, #f97316)",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                }),
                html.P(
                    "Players most likely to record a hit today — singles, doubles & triples",
                    style={"margin": "4px 0 0", "fontSize": "13px",
                           "color": "var(--on-surface-variant)"},
                ),
            ]),
            html.Div([
                html.Span("sports_baseball", className="material-symbols-outlined",
                          style={"fontSize": "28px", "color": "#f97316", "opacity": ".6"}),
            ]),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "24px 28px 12px",
        }),

        # ── Leaderboard section ───────────────────────────────────────────────
        html.Div(id="hit-leaderboard"),

        # ── Parlays section ───────────────────────────────────────────────────
        html.Div(id="hit-parlays", style={"padding": "0 28px 40px"}),

    ], style={"padding": "0 0 40px"})


# ---------------------------------------------------------------------------
# Data loader callback
# ---------------------------------------------------------------------------

@callback(
    Output("hit-predictions-store", "data"),
    Input("hit-interval", "n_intervals"),
)
def load_hit_data(_n):
    try:
        from utils.hit_predictor import get_hit_predictions
        df = get_hit_predictions(SEASON_YEAR)
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Hit predictor load failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Render callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("hit-leaderboard", "children"),
    Output("hit-parlays", "children"),
    Input("hit-predictions-store", "data"),
)
def render_hit_page(data):
    if not data:
        return _empty_leaderboard(), _empty_parlays()

    df = pd.DataFrame(data)
    for col in ["hit_score", "single_score", "double_score", "triple_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    leaderboard = _build_leaderboard(df)
    parlays = _build_parlays_section(df)
    return leaderboard, parlays


# ---------------------------------------------------------------------------
# Leaderboard builder
# ---------------------------------------------------------------------------

def _player_row(row, rank: int, score_col: str, accent: str, label: str):
    name    = str(row.get("Player", ""))
    team    = str(row.get("Team",   ""))
    score   = float(row.get(score_col, 0))
    conf    = str(row.get("confidence", ""))
    hot     = bool(row.get("hot_streak", False))
    h_avg   = float(row.get("H_per_AB",  0))
    d_avg   = float(row.get("2B_per_AB", 0))
    t_avg   = float(row.get("3B_per_AB", 0))

    badge_color = "#22c55e" if conf == "High" else ("#f97316" if conf == "Medium" else "#6b7280")

    return html.Div([
        # Rank number
        html.Div(str(rank), style={
            "width": "28px", "height": "28px", "borderRadius": "50%",
            "background": f"{accent}18",
            "border": f"1px solid {accent}44",
            "color": accent, "fontSize": "11px", "fontWeight": "800",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "flexShrink": "0",
        }),

        # Name + team + hot badge
        html.Div([
            html.Div([
                html.Span(name, style={"fontSize": "13px", "fontWeight": "700", "color": "#f0dfd8"}),
                html.Span(" • ", style={"color": "rgba(255,255,255,.2)", "margin": "0 3px"}),
                html.Span(team, style={"fontSize": "11px", "color": "var(--on-surface-variant)"}),
                html.Span("🔥 Hot", style={
                    "marginLeft": "8px", "fontSize": "9px", "fontWeight": "700",
                    "color": "#f97316", "border": "1px solid #f97316",
                    "borderRadius": "4px", "padding": "1px 5px",
                }) if hot else None,
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            html.Div([
                html.Span(f"AVG {h_avg:.3f}", style={"fontSize": "10px", "color": "rgba(255,255,255,.35)", "marginRight": "8px"}),
                html.Span(f"2B/AB {d_avg:.3f}", style={"fontSize": "10px", "color": "rgba(255,255,255,.35)", "marginRight": "8px"}),
                html.Span(f"3B/AB {t_avg:.3f}", style={"fontSize": "10px", "color": "rgba(255,255,255,.35)"}),
            ]),
        ], style={"flex": "1", "minWidth": "0"}),

        # Score + type label + confidence
        html.Div([
            html.Span(f"{score:.1%}", style={"fontSize": "14px", "fontWeight": "900", "color": accent}),
            html.Div([
                html.Span(label, style={
                    "fontSize": "9px", "fontWeight": "700", "color": accent,
                    "border": f"1px solid {accent}55", "borderRadius": "3px", "padding": "1px 4px",
                }),
                html.Span(conf, style={
                    "fontSize": "9px", "color": badge_color,
                    "border": f"1px solid {badge_color}",
                    "borderRadius": "3px", "padding": "1px 4px", "marginLeft": "4px",
                }),
            ], style={"display": "flex", "marginTop": "2px"}),
        ], style={"flexShrink": "0", "textAlign": "right"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "12px",
        "padding": "10px 0", "borderBottom": "1px solid rgba(255,255,255,.05)",
    })


def _leaderboard_card(title: str, subtitle: str, accent: str, rows_html: list):
    return html.Div([
        html.Div([
            html.H4(title, style={"margin": "0 0 2px", "fontSize": "14px", "fontWeight": "800", "color": accent}),
            html.P(subtitle, style={"margin": 0, "fontSize": "11px", "color": "var(--on-surface-variant)"}),
        ], style={"marginBottom": "12px"}),
        html.Div(rows_html),
    ], style={
        "background": "var(--surface-container)",
        "border": f"1px solid {accent}33",
        "borderRadius": "12px", "padding": "16px 18px",
    })


def _build_leaderboard(df: pd.DataFrame):
    top_n = 12

    any_hit_rows = [
        _player_row(r, i + 1, "hit_score",    "#a78bfa", "Any Hit")
        for i, (_, r) in enumerate(df.sort_values("hit_score",    ascending=False).head(top_n).iterrows())
    ]
    single_rows = [
        _player_row(r, i + 1, "single_score", "#38bdf8", "Single")
        for i, (_, r) in enumerate(df.sort_values("single_score", ascending=False).head(top_n).iterrows())
    ]
    double_rows = [
        _player_row(r, i + 1, "double_score", "#f97316", "Double")
        for i, (_, r) in enumerate(df.sort_values("double_score", ascending=False).head(top_n).iterrows())
    ]
    triple_rows = [
        _player_row(r, i + 1, "triple_score", "#22c55e", "Triple")
        for i, (_, r) in enumerate(df.sort_values("triple_score", ascending=False).head(top_n).iterrows())
    ]

    cards = html.Div([
        _leaderboard_card("Any Hit Leaders",    "Most likely to record any type of hit", "#a78bfa", any_hit_rows),
        _leaderboard_card("Single Leaders",     "Most likely to record a single",        "#38bdf8", single_rows),
        _leaderboard_card("Double Leaders",     "Most likely to record a double",        "#f97316", double_rows),
        _leaderboard_card("Triple Leaders",     "Most likely to record a triple",        "#22c55e", triple_rows),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fill, minmax(340px, 1fr))",
        "gap": "16px",
        "padding": "16px 28px",
    })

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Today's Hit Leaders", style={"margin": 0, "fontSize": "15px", "fontWeight": "800"}),
                html.P("Ranked by model-predicted probability — hot streaks highlighted",
                       style={"margin": "2px 0 0", "fontSize": "12px",
                              "color": "var(--on-surface-variant)"}),
            ]),
        ], className="table-title-bar"),
        cards,
    ], className="table-container", style={
        "marginBottom": "24px",
        "border": "1px solid rgba(249,115,22,.2)",
        "background": "linear-gradient(135deg, rgba(249,115,22,.04) 0%, var(--surface-container) 60%)",
    })


# ---------------------------------------------------------------------------
# Parlays builder
# ---------------------------------------------------------------------------

def _parlay_leg_item(row, leg_num: int, accent: str, leg_label: str):
    name  = str(row.get("Player", ""))
    team  = str(row.get("Team",   ""))
    conf  = str(row.get("confidence", ""))
    badge_color = "#22c55e" if conf == "High" else "#f97316"

    if leg_label == "Single":
        score = float(row.get("single_score", 0))
    elif leg_label == "Double":
        score = float(row.get("double_score", 0))
    elif leg_label == "2TB":
        score = float(row.get("double_score", 0))
    else:
        score = float(row.get("hit_score", 0))

    return html.Div([
        html.Div(str(leg_num), style={
            "width": "18px", "height": "18px", "borderRadius": "50%",
            "background": f"{accent}22", "border": f"1px solid {accent}",
            "color": accent, "fontSize": "10px", "fontWeight": "700",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "flexShrink": "0",
        }),
        html.Div([
            html.Span(name, style={"fontSize": "12px", "fontWeight": "600", "color": "#f0dfd8"}),
            html.Span(f" {team}", style={"fontSize": "10px", "color": "var(--on-surface-variant)", "marginLeft": "4px"}),
        ], style={"flex": "1", "minWidth": "0", "overflow": "hidden"}),
        html.Div([
            html.Span(f"{score:.1%}", style={"fontSize": "11px", "fontWeight": "700", "color": accent}),
            html.Span(leg_label, style={
                "fontSize": "9px", "color": accent,
                "marginLeft": "5px", "border": f"1px solid {accent}55",
                "borderRadius": "3px", "padding": "1px 4px",
            }),
            html.Span(conf, style={
                "fontSize": "9px", "color": badge_color,
                "marginLeft": "4px", "border": f"1px solid {badge_color}",
                "borderRadius": "3px", "padding": "1px 4px",
            }),
        ], style={"display": "flex", "alignItems": "center", "flexShrink": "0"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "8px",
        "padding": "5px 0", "borderBottom": "1px solid rgba(255,255,255,.04)",
    })


def _parlay_card(legs: list, parlay_num: int, accent: str, leg_labels: list[str], combined: float):
    leg_items = [
        _parlay_leg_item(row, i + 1, accent, leg_labels[i])
        for i, row in enumerate(legs)
    ]
    return html.Div([
        html.Div([
            html.Span(f"Parlay {parlay_num}", style={
                "fontSize": "10px", "fontWeight": "800", "color": accent,
                "textTransform": "uppercase", "letterSpacing": "1px",
            }),
            html.Div([
                html.Span("Combined: ", style={"fontSize": "10px", "color": "var(--on-surface-variant)"}),
                html.Span(f"{combined:.2%}", style={"fontSize": "13px", "fontWeight": "800", "color": accent}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "marginBottom": "6px",
        }),
        html.Div(leg_items),
    ], style={
        "background": "var(--surface-container)",
        "border": f"1px solid {accent}33",
        "borderRadius": "10px", "padding": "12px 14px",
    })


def _parlay_group(title: str, subtitle: str, accent: str, cards: list):
    return html.Div([
        html.Div([
            html.H4(title, style={
                "margin": "0 0 4px", "fontSize": "14px", "fontWeight": "800", "color": accent,
            }),
            html.P(subtitle, style={"margin": 0, "fontSize": "11px", "color": "var(--on-surface-variant)"}),
        ], style={"marginBottom": "14px"}),
        html.Div(cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(320px, 1fr))",
            "gap": "12px",
        }),
    ])


def _score_for_label(row, label: str) -> float:
    if label == "Single":
        return float(row.get("single_score", 0))
    if label == "Double":
        return float(row.get("double_score", 0))
    if label == "2TB":
        return float(row.get("double_score", 0))
    return float(row.get("hit_score", 0))


def _combined_prob(legs: list, labels: list[str]) -> float:
    return math.prod(_score_for_label(r, l) for r, l in zip(legs, labels))


def _build_parlays_section(df: pd.DataFrame):
    from utils.hit_predictor import build_hit_parlays
    parlays = build_hit_parlays(df)

    sections = []

    # ── Group A: 2+ Total Bases ────────────────────────────────────────────
    two_base_cards = [
        _parlay_card(
            legs, i + 1, "#f97316",
            ["2TB", "2TB", "2TB"],
            _combined_prob(legs, ["2TB", "2TB", "2TB"]),
        )
        for i, legs in enumerate(parlays.get("two_base", []))
    ]
    if two_base_cards:
        sections.append(_parlay_group(
            "2+ Total Bases Parlays",
            "Each player must record a double, triple, or HR — picks from different players",
            "#f97316",
            two_base_cards,
        ))

    # ── Group B: Any Hit ──────────────────────────────────────────────────
    any_hit_cards = [
        _parlay_card(
            legs, i + 1, "#a78bfa",
            ["Hit", "Hit", "Hit"],
            _combined_prob(legs, ["Hit", "Hit", "Hit"]),
        )
        for i, legs in enumerate(parlays.get("any_hit", []))
    ]
    if any_hit_cards:
        if sections:
            sections.append(html.Div(style={"height": "24px"}))
        sections.append(_parlay_group(
            "Any Hit Parlays",
            "Each player must record any type of hit — single, double, triple, or HR",
            "#a78bfa",
            any_hit_cards,
        ))

    # ── Group C: Mixed Singles / Doubles ─────────────────────────────────
    mixed_cards = []
    for i, entry in enumerate(parlays.get("mixed", [])):
        legs   = entry["legs"]
        labels = entry["leg_type"]
        mixed_cards.append(
            _parlay_card(legs, i + 1, "#22c55e", labels, _combined_prob(legs, labels))
        )
    if mixed_cards:
        if sections:
            sections.append(html.Div(style={"height": "24px"}))
        sections.append(_parlay_group(
            "Singles & Doubles Parlays",
            "Higher-payout bets: 2 of 5 are all-single, 2 are all-double, 1 is mixed",
            "#22c55e",
            mixed_cards,
        ))

    if not sections:
        return _empty_parlays()

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Today's Hit Parlays", style={
                    "margin": 0, "fontSize": "15px", "fontWeight": "800",
                }),
                html.P(
                    "15 three-leg parlays across 3 categories — no player repeats more than twice",
                    style={"margin": "2px 0 0", "fontSize": "12px",
                           "color": "var(--on-surface-variant)"},
                ),
            ]),
        ], className="table-title-bar"),

        html.Div(sections, style={"padding": "16px"}),

        html.Div(
            "Probability estimates based on FanGraphs season stats + Statcast xBA. For entertainment only.",
            style={"fontSize": "10px", "color": "rgba(255,255,255,.2)", "padding": "8px 16px"},
        ),
    ], className="table-container", style={
        "border": "1px solid rgba(249,115,22,.2)",
        "background": "linear-gradient(135deg, rgba(249,115,22,.04) 0%, var(--surface-container) 60%)",
    })


# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------

def _empty_leaderboard():
    return html.Div(
        "Loading hit predictions…",
        style={"padding": "40px", "textAlign": "center",
               "color": "var(--on-surface-variant)", "fontSize": "14px"},
    )


def _empty_parlays():
    return html.Div(
        "Hit parlay data unavailable — check back after season stats update.",
        style={"padding": "20px", "textAlign": "center",
               "color": "var(--on-surface-variant)", "fontSize": "13px",
               "fontStyle": "italic"},
    )
