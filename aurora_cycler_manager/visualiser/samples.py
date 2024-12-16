"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Samples tab layout and callbacks for the visualiser app.
"""
import gzip
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.analysis import _run_from_sample, combine_jobs
from aurora_cycler_manager.visualiser.funcs import smoothed_derivative

graph_template = "seaborn"
graph_margin = {"l": 50, "r": 10, "t": 50, "b": 50}

# Side menu for the samples tab
samples_menu =  html.Div(
        style = {"overflow": "scroll", "height": "100%"},
        children = [
            html.H5("Select samples to plot:"),
            dcc.Dropdown(
                id="samples-dropdown",
                options=[], # updated by callback
                value=[],
                multi=True,
            ),
            html.Div(style={"margin-top": "50px"}),
            html.H5("Time graph"),
            html.Label("X-axis:", htmlFor="samples-time-x"),
            dcc.Dropdown(
                id="samples-time-x",
                options=["Unix time","From start","From formation","From cycling"],
                value="From start",
                multi=False,
            ),
            dcc.Dropdown(
                id="samples-time-units",
                options=["Seconds","Minutes","Hours","Days"],
                value="Hours",
            ),
            html.Div(style={"margin-top": "10px"}),
            html.Label("Y-axis:", htmlFor="samples-time-y"),
            dcc.Dropdown(
                id="samples-time-y",
                options=["V (V)"],
                value="V (V)",
                multi=False,
            ),
            html.Div(style={"margin-top": "50px"}),
            html.H5("Cycles graph"),
            html.P("X-axis: Cycle"),
            html.Label("Y-axis:", htmlFor="samples-cycles-y"),
            dcc.Dropdown(
                id="samples-cycles-y",
                options=[
                    "Specific discharge capacity (mAh/g)",
                    "Efficiency (%)",
                ],
                value="Specific discharge capacity (mAh/g)",
                multi=False,
            ),
            html.Div(style={"margin-top": "50px"}),
            html.H5("One cycle graph"),
            html.Label("X-axis:", htmlFor="samples-cycle-x"),
            dcc.Dropdown(
                id="samples-cycle-x",
                options=["Q (mAh)", "V (V)", "dQdV (mAh/V)"],
                value="Q (mAh)",
            ),
            html.Div(style={"margin-top": "10px"}),
            html.Label("Y-axis:", htmlFor="samples-cycle-y"),
            dcc.Dropdown(
                id="samples-cycle-y",
                options=["Q (mAh)", "V (V)", "dQdV (mAh/V)"],
                value="V (V)",
            ),
            html.Div(style={"margin-top": "100px"}),
        ],
    )

time_graph = dcc.Graph(
    id="time-graph",
    style={"height": "100%"},
    config={"scrollZoom":True, "displaylogo":False},
    figure={
        "data": [],
        "layout": go.Layout(
            template = graph_template,
            margin = graph_margin,
            title="vs time",
            xaxis={"title": "X-axis Title"},
            yaxis={"title": "Y-axis Title"},
            showlegend=False,
        ),
    },
)

cycles_graph = dcc.Graph(
    id="cycles-graph",
    style={"height": "100%"},
    config={"scrollZoom":True, "displaylogo":False},
    figure={
        "data": [],
        "layout": go.Layout(
            template = graph_template,
            margin = graph_margin,
            title = "vs cycle",
            xaxis = {"title": "X-axis Title"},
            yaxis = {"title": "Y-axis Title"},
            showlegend=False,
        ),
    },
)

one_cycle_graph = dcc.Graph(
    id="cycle-graph",
    config={"scrollZoom":True, "displaylogo":False},
    style={"height": "100%"},
    figure={
        "data": [],
        "layout": go.Layout(
            template = graph_template,
            margin = graph_margin,
            title = "One cycle",
            xaxis = {"title": "X-axis Title"},
            yaxis = {"title": "Y-axis Title"},
            showlegend=False,
        ),
    },
)

samples_layout =  html.Div(
    style={"height": "100%"},
    children = [
        dcc.Store(
            id="samples-data-store",
            data={"data_sample_time": {}, "data_sample_cycle": {}},
        ),
        PanelGroup(
            id="samples-panel-group",
            direction="horizontal",
            style={"height": "100%"},
            children=[
                Panel(
                    id="samples-menu",
                    className="menu-panel",
                    children=samples_menu,
                    defaultSizePercentage=16,
                    collapsible=True,
                ),
                PanelResizeHandle(
                    html.Div(className="resize-handle-horizontal"),
                ),
                Panel(
                    id="samples-graphs",
                    minSizePercentage=50,
                    children=[
                        PanelGroup(
                            id="samples-graph-group",
                            direction="vertical",
                            children=[
                                Panel(
                                    time_graph,
                                    id="samples-top-graph",
                                ),
                                PanelResizeHandle(
                                    html.Div(className="resize-handle-vertical"),
                                ),
                                Panel(
                                    id="samples-bottom-graphs",
                                    children=[
                                        PanelGroup(
                                            id="samples-bottom-graph-group",
                                            direction="horizontal",
                                            children=[
                                                Panel(
                                                    cycles_graph,
                                                    id="samples-bottom-left-graph",
                                                ),
                                                PanelResizeHandle(
                                                    html.Div(className="resize-handle-horizontal"),
                                                ),
                                                Panel(
                                                    one_cycle_graph,
                                                    id="samples-bottom-right-graph",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

#--------------------------------- CALLBACKS ----------------------------------#
def register_samples_callbacks(app: Dash, config: dict) -> None:
    """Register all callbacks for the samples tab."""

    # Sample list has updated, update dropdowns
    @app.callback(
        Output("samples-dropdown", "options"),
        Input("samples-store", "data"),
    )
    def update_samples_dropdown(samples):
        return [{"label": name, "value": name} for name in samples]

    # Update the samples data store
    @app.callback(
        Output("samples-data-store", "data"),
        Output("samples-time-y", "options"),
        Output("samples-cycles-y", "options"),
        Input("samples-dropdown", "value"),
        Input("samples-data-store", "data"),
    )
    def update_sample_data(samples, data):
        # Get rid of samples that are no longer selected
        for sample in list(data["data_sample_time"].keys()):
            if sample not in samples:
                data["data_sample_time"].pop(sample)
                if sample in data["data_sample_cycle"]:
                    data["data_sample_cycle"].pop(sample)

        for sample in samples:
            # Check if already in data store
            if sample in data["data_sample_time"]:
                continue

            # Otherwise import the data
            run_id = _run_from_sample(sample)
            data_folder = config["Processed snapshots folder path"]
            file_location = os.path.join(data_folder,run_id,sample)

            # Get raw data
            try:
                files = os.listdir(file_location)
            except FileNotFoundError:
                continue
            if any(f.startswith("full") and f.endswith(".json.gz") for f in files):
                filepath = next(f for f in files if f.startswith("full") and f.endswith(".json.gz"))
                with gzip.open(f"{file_location}/{filepath}", "rb") as f:
                    data_dict = json.load(f)["data"]
                data["data_sample_time"][sample] = data_dict
            elif any(f.startswith("full") and f.endswith(".h5") for f in files):
                filepath = next(f for f in files if f.startswith("full") and f.endswith(".h5"))
                df = pd.read_hdf(f"{file_location}/{filepath}")
                data["data_sample_time"][sample] = df.to_dict(orient="list")
            else:
                cycling_files = [
                    os.path.join(file_location,f) for f in files
                    if (f.startswith("snapshot") and f.endswith(".json.gz"))
                ]
                if not cycling_files:
                    cycling_files = [
                        os.path.join(file_location,f) for f in files
                        if (f.startswith("snapshot") and f.endswith(".h5"))
                    ]
                    if not cycling_files:
                        print(f"No cycling files found in {file_location}")
                        continue
                df, metadata = combine_jobs(cycling_files)
                data["data_sample_time"][sample] = df.to_dict(orient="list")

            # Get the analysed file
            try:
                analysed_file = next(f for f in files if (f.startswith("cycles") and f.endswith(".json")))
            except StopIteration:
                continue
            with open(f"{file_location}/{analysed_file}", encoding="utf-8") as f:
                cycle_dict = json.load(f)["data"]
            if not cycle_dict or "Cycle" not in cycle_dict:
                continue
            data["data_sample_cycle"][sample] = cycle_dict

        # Update the y-axis options
        time_y_vars = {"V (V)"}
        for data_dict in data["data_sample_time"].values():
            time_y_vars.update(data_dict.keys())
        time_y_vars = list(time_y_vars)

        cycles_y_vars = {"Specific discharge capacity (mAh/g)", "Normalised discharge capacity (%)", "Efficiency (%)"}
        for data_dict in data["data_sample_cycle"].values():
            cycles_y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
        cycles_y_vars = list(cycles_y_vars)

        return data, time_y_vars, cycles_y_vars

    # Update the time graph
    @app.callback(
        Output("time-graph", "figure"),
        Input("samples-data-store", "data"),
        Input("samples-time-x", "value"),
        Input("samples-time-units", "value"),
        Input("samples-time-y", "value"),
    )
    def update_time_graph(data, xvar, xunits, yvar):
        fig = px.scatter()
        fig.update_layout(
            template = graph_template,
            margin = graph_margin,
            title = "No data...",
            xaxis_title = f"Time ({xunits.lower()})",
            yaxis_title=yvar,
            showlegend=False,
        )
        if not data["data_sample_time"] or not xvar or not yvar or not xunits:
            return fig

        multiplier = {"Seconds": 1, "Minutes": 60, "Hours": 3600, "Days": 86400}[xunits]
        for sample, data_dict in data["data_sample_time"].items():
            uts = np.array(data_dict["uts"])
            if xvar == "From start":
                offset=uts[0]
            elif xvar == "From formation":
                offset=uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x >= 1)]
            elif xvar == "From cycling":
                offset=uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x >= 4)]
            else:
                offset=0

            trace = go.Scatter(
                x=(np.array(data_dict["uts"]) - offset) / multiplier,
                y=data_dict[yvar],
                mode="lines",
                name=sample,
                hovertemplate=f"{sample}<br>Time: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig.add_trace(trace)

        fig.update_layout(
            title=f"{yvar} vs time",
        )

        return fig


    # Update the cycles graph
    @app.callback(
        Output("cycles-graph", "figure"),
        Input("samples-data-store", "data"),
        Input("samples-cycles-y", "value"),
    )
    def update_cycles_graph(data, yvar):
        fig = px.scatter()
        fig.update_layout(
            template = graph_template,
            margin = graph_margin,
            title = "No data...",
            xaxis_title = "Cycle",
            yaxis_title = yvar,
            showlegend = False,
        )
        if not data["data_sample_cycle"] or not yvar:
            return fig
        for sample, cycle_dict in data["data_sample_cycle"].items():
            trace = go.Scatter(
                x=cycle_dict["Cycle"],
                y=cycle_dict[yvar],
                mode="lines+markers",
                name=sample,
                hovertemplate=f"{sample}<br>Cycle: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig.add_trace(trace)
        fig.update_layout(title=f"{yvar} vs cycle")
        return fig

    # Update the one cycle graph
    @app.callback(
        Output("cycle-graph", "figure"),
        Input("cycles-graph", "clickData"),
        Input("samples-data-store", "data"),
        Input("samples-cycle-x", "value"),
        Input("samples-cycle-y", "value"),
    )
    def update_cycle_graph(clickData, data, xvar, yvar):
        fig = px.scatter()
        fig.update_layout(
            template = graph_template,
            margin = graph_margin,
            title = "No data...",
            xaxis_title=xvar,
            yaxis_title=yvar,
            showlegend=False,
        )
        if not data["data_sample_cycle"] or not xvar or not yvar:
            return fig
        if not clickData:
            cycle = 1
        else:
            # clickData is a dict with keys 'points' and 'event'
            # 'points' is a list of dicts with keys 'curveNumber', 'pointNumber', 'pointIndex', 'x', 'y', 'text'
            point = clickData["points"][0]
            cycle = point["x"]
        for sample, data_dict in data["data_sample_time"].items():
            # find where the cycle = cycle
            mask = np.array(data_dict["Cycle"]) == cycle
            if not any(mask):
                continue
            mask_dict = {}
            mask_dict["V (V)"] = np.array(data_dict["V (V)"])[mask]
            mask_dict["Q (mAh)"] = np.array(data_dict["dQ (mAh)"])[mask].cumsum()
            mask_dict["dQdV (mAh/V)"] = smoothed_derivative(mask_dict["V (V)"], mask_dict["Q (mAh)"])
            trace = go.Scatter(
                x=mask_dict[xvar],
                y=mask_dict[yvar],
                mode="lines",
                name=sample,
                hovertemplate=f"{sample}<br>{xvar}: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig.add_trace(trace)

        fig.update_layout(title=f"{yvar} vs {xvar} for cycle {cycle}")
        return fig
