"""
Model Performance page.
Shows accuracy over time, cumulative hit rate, high-confidence ROI, and feature importance.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from openpyxl import load_workbook

EXCEL_PATH = Path("MLB_HR_Predictions.xlsx")

PRED_COLUMNS = [
    "Date", "Player", "Team", "Opponent", "Pitcher",
    "HR_Probability", "Confidence", "Park_Factor",
    "Temp_F", "Wind_Speed_MPH", "Wind_Direction", "Is_Indoor",
    "Home_Game", "Predicted_HRs", "Actual_HRs", "Hit",
]

PERF_COLUMNS = [
    "Date", "Total_Predictions", "High_Conf_Count",
    "Daily_Accuracy", "Cumulative_Accuracy",
    "High_Conf_Hit_Rate", "ROI_Flat_Bet",
]


def layout():
    return html.Div([
        html.Div([
            html.H2("Model Performance", className="page-title"),
            html.P("Track prediction accuracy, hit rate, and ROI over time", className="page-subtitle"),
        ], className="page-header"),

        # KPI cards
        html.Div(id="perf-kpi-cards", className="summary-cards"),

        # Accuracy over time chart
        html.Div([
            html.H3("Cumulative Accuracy Over Time", className="section-title"),
            dcc.Graph(id="accuracy-chart", config={"displayModeBar": False}),
        ], className="chart-section"),

        # High confidence hit rate vs all predictions
        html.Div([
            html.H3("Daily Hit Rate: High Confidence vs All", className="section-title"),
            dcc.Graph(id="hit-rate-chart", config={"displayModeBar": False}),
        ], className="chart-section"),

        # ROI chart
        html.Div([
            html.H3("Cumulative ROI (flat $1 bet on High Confidence picks)", className="section-title"),
            dcc.Graph(id="roi-chart", config={"displayModeBar": False}),
        ], className="chart-section"),

        # Feature importance
        html.Div([
            html.H3("Top Feature Importances (SHAP)", className="section-title"),
            dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False}),
        ], className="chart-section"),

        dcc.Interval(id="perf-refresh", interval=10 * 60 * 1000, n_intervals=0),
    ], className="page-container")


def _load_performance() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        return pd.DataFrame(columns=PERF_COLUMNS)
    try:
        wb = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
        ws = wb["Model_Performance"]
        data = list(ws.iter_rows(min_row=2, values_only=True))
        wb.close()
        if not data:
            return pd.DataFrame(columns=PERF_COLUMNS)
        return pd.DataFrame(data, columns=PERF_COLUMNS).dropna(subset=["Date"])
    except Exception:
        return pd.DataFrame(columns=PERF_COLUMNS)


def _load_feature_importance() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        return pd.DataFrame(columns=["Date", "Feature", "Importance_Score"])
    try:
        wb = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
        ws = wb["Feature_Importance"]
        data = list(ws.iter_rows(min_row=2, values_only=True))
        wb.close()
        if not data:
            return pd.DataFrame(columns=["Date", "Feature", "Importance_Score"])
        return pd.DataFrame(data, columns=["Date", "Feature", "Importance_Score"])
    except Exception:
        return pd.DataFrame(columns=["Date", "Feature", "Importance_Score"])


@callback(
    Output("perf-kpi-cards", "children"),
    Output("accuracy-chart", "figure"),
    Output("hit-rate-chart", "figure"),
    Output("roi-chart", "figure"),
    Output("feature-importance-chart", "figure"),
    Input("perf-refresh", "n_intervals"),
)
def update_performance(_n):
    df = _load_performance()
    fi_df = _load_feature_importance()

    cards = _build_kpi_cards(df)
    acc_fig = _accuracy_chart(df)
    hit_fig = _hit_rate_chart(df)
    roi_fig = _roi_chart(df)
    feat_fig = _feature_importance_chart(fi_df)
    return cards, acc_fig, hit_fig, roi_fig, feat_fig


def _build_kpi_cards(df: pd.DataFrame):
    if df.empty:
        return [html.Div("No performance data yet. Run predictions and update results.",
                         style={"color": "#888", "padding": "20px"})]

    overall_acc = df["Cumulative_Accuracy"].iloc[-1] if "Cumulative_Accuracy" in df.columns else 0
    high_hit = df["High_Conf_Hit_Rate"].mean() if "High_Conf_Hit_Rate" in df.columns else 0
    total_roi = df["ROI_Flat_Bet"].sum() if "ROI_Flat_Bet" in df.columns else 0
    days = len(df)

    def card(title, value, color):
        return html.Div([
            html.H3(str(value), style={"color": color, "margin": "0", "fontSize": "28px"}),
            html.P(title, style={"margin": "4px 0 0", "fontSize": "13px", "color": "#666"}),
        ], className="summary-card")

    return [
        card("Overall Accuracy", f"{overall_acc:.1%}", "#1F4E79"),
        card("High Conf Hit Rate", f"{high_hit:.1%}", "#00B050"),
        card("Cumulative ROI", f"{total_roi:+.2f}u", "#FF8C00" if total_roi >= 0 else "#dc3545"),
        card("Days Tracked", days, "#6c757d"),
    ]


def _accuracy_chart(df: pd.DataFrame):
    if df.empty or "Date" not in df.columns:
        return _empty_fig("No accuracy data yet")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Cumulative_Accuracy"],
        mode="lines+markers", name="Cumulative Accuracy",
        line={"color": "#1F4E79", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Daily_Accuracy"],
        mode="lines", name="Daily Accuracy",
        line={"color": "#90caf9", "width": 1, "dash": "dot"},
    ))
    fig.update_layout(**_chart_layout(), yaxis_tickformat=".1%", height=280)
    return fig


def _hit_rate_chart(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("No data yet")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Daily_Accuracy"],
        name="All Predictions", marker_color="#90caf9",
    ))
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["High_Conf_Hit_Rate"],
        name="High Confidence", marker_color="#00B050",
    ))
    fig.update_layout(**_chart_layout(), barmode="group", yaxis_tickformat=".1%", height=280)
    return fig


def _roi_chart(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("No ROI data yet")
    cumulative_roi = df["ROI_Flat_Bet"].cumsum()
    colors = ["#00B050" if v >= 0 else "#dc3545" for v in cumulative_roi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=cumulative_roi,
        mode="lines+markers", fill="tozeroy",
        line={"color": "#00B050", "width": 2},
        fillcolor="rgba(0,176,80,0.1)",
        name="Cumulative ROI",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(**_chart_layout(), yaxis_title="Units", height=280)
    return fig


def _feature_importance_chart(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("Train model to see feature importances")
    latest = df.sort_values("Date").groupby("Feature")["Importance_Score"].last().reset_index()
    latest = latest.sort_values("Importance_Score", ascending=True).tail(15)
    fig = go.Figure(go.Bar(
        x=latest["Importance_Score"], y=latest["Feature"],
        orientation="h", marker_color="#1F4E79",
    ))
    fig.update_layout(**_chart_layout(), height=400, xaxis_title="SHAP Importance")
    return fig


def _chart_layout():
    return {
        "paper_bgcolor": "white", "plot_bgcolor": "#f8f9fa",
        "margin": dict(l=60, r=20, t=20, b=40),
        "legend": {"orientation": "h", "y": -0.15},
    }


def _empty_fig(msg: str):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font={"size": 14, "color": "#888"})
    fig.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                      margin=dict(l=40, r=20, t=20, b=40))
    return fig
