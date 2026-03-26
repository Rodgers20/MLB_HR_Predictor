"""
MLB HR Predictor — Dash Dashboard
Entry point: python dashboard/app.py
Navigate to http://localhost:8050
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import dash
from dash import Input, Output, dcc, html

from dashboard.pages import history, model_perf, player, today

BOOTSTRAP_CDN = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP_CDN],
    suppress_callback_exceptions=True,
    title="MLB HR Predictor",
)
server = app.server

PAGES = [
    {"href": "/",            "label": "Today's Picks",    "icon": "⚾"},
    {"href": "/player",      "label": "Player Analysis",  "icon": "👤"},
    {"href": "/performance", "label": "Model Performance","icon": "📈"},
    {"href": "/history",     "label": "History",          "icon": "📋"},
]


def _nav_links(pathname: str):
    return html.Div([
        dcc.Link(
            [
                html.Span(p["icon"], style={"marginRight": "9px", "fontSize": "14px"}),
                p["label"],
            ],
            href=p["href"],
            className="nav-link active" if pathname == p["href"] else "nav-link",
        )
        for p in PAGES
    ])


def sidebar(pathname: str = "/"):
    return [
        html.Div([
            html.Div([
                html.Div("⚾", className="sidebar-logo-icon"),
                html.Span("MLB HR", style={"marginLeft": "8px"}),
            ], className="sidebar-logo"),
            html.Div("Home Run Predictor", className="sidebar-subtitle"),
            html.Div("2026 Season", className="sidebar-season-badge"),
        ], className="sidebar-header"),

        html.Div("Navigation", className="nav-section-label"),
        _nav_links(pathname),

        html.Div(style={"flex": "1"}),  # push footer down
        html.Div([
            html.Div("Powered by XGBoost + Statcast",
                     style={"fontSize": "10px", "color": "rgba(255,255,255,.25)",
                            "padding": "16px 18px", "borderTop": "1px solid rgba(255,255,255,.06)"}),
        ]),
    ]


# ─── App Layout ───────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(sidebar("/"), id="sidebar"),
    html.Div(id="main-content", style={"marginLeft": "240px", "flex": "1"}),
], style={"display": "flex", "minHeight": "100vh"})


# ─── Page Routing ─────────────────────────────────────────────────────────────

@app.callback(
    Output("main-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname == "/player":
        return player.layout()
    elif pathname == "/performance":
        return model_perf.layout()
    elif pathname == "/history":
        return history.layout()
    else:
        return today.layout()


# ─── Active nav highlighting ──────────────────────────────────────────────────

@app.callback(
    Output("sidebar", "children"),
    Input("url", "pathname"),
)
def update_nav(pathname):
    return sidebar(pathname)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "0.0.0.0")
    port = int(os.getenv("DASH_PORT", 8050))
    debug = os.getenv("DASH_DEBUG", "False").lower() == "true"
    print(f"\n  MLB HR Predictor → http://localhost:{port}\n")
    app.run(host=host, port=port, debug=debug)
