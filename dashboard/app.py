"""
Main Plotly Dash application for the Baseball Performance CPD Dashboard.

Four-view dashboard:
1. Player Overview — season-level summary
2. Change-Point Timeline — detected breakpoints on time series
3. Before/After Snapshot — dual-state metric comparison
4. Clutch vs. Core — situational performance breakdown

Usage:
    python -m dashboard.app
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Baseball CPD Dashboard",
    suppress_callback_exceptions=True,
)

server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("⚾ Baseball Performance Change-Point Dashboard"), width=12),
    ], className="my-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Player"),
            dcc.Dropdown(
                id="player-dropdown",
                placeholder="Search for a player...",
            ),
        ], width=4),
        dbc.Col([
            html.Label("CPD Algorithm"),
            dcc.Dropdown(
                id="algorithm-dropdown",
                options=[
                    {"label": "PELT", "value": "pelt"},
                    {"label": "Binary Segmentation (CUSUM)", "value": "binseg"},
                    {"label": "Bayesian Online", "value": "bayesian"},
                ],
                value="pelt",
            ),
        ], width=3),
    ], className="mb-4"),

    dbc.Tabs([
        dbc.Tab(label="Player Overview", tab_id="tab-overview", children=[
            html.Div(id="overview-content", className="p-3"),
        ]),
        dbc.Tab(label="Change-Point Timeline", tab_id="tab-timeline", children=[
            dcc.Graph(id="timeline-graph"),
        ]),
        dbc.Tab(label="Before / After Snapshot", tab_id="tab-snapshot", children=[
            dcc.Graph(id="snapshot-graph"),
        ]),
        dbc.Tab(label="Clutch vs. Core", tab_id="tab-clutch", children=[
            dcc.Graph(id="clutch-graph"),
        ]),
    ], id="tabs", active_tab="tab-overview"),

], fluid=True)


# Import callbacks (register after layout is defined)
# from dashboard.callbacks import register_callbacks
# register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
