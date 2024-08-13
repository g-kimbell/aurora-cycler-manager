import os
import sys
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import textwrap
import yaml
import numpy as np
import json
import sqlite3
import pandas as pd
from scipy import stats
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.analysis import combine_hdfs, _run_from_sample

app = dash.Dash(__name__)

graph_template = 'seaborn'

#======================================================================================================================#
#===================================================== FUNCTIONS ======================================================#
#======================================================================================================================#

def get_sample_names() -> list:
    db_path = "K:/Aurora/cucumber/database/database.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM results")
        samples = cursor.fetchall()
    return [sample[0] for sample in samples]

def get_batch_names() -> list:
    graph_config_path = "K:/Aurora/cucumber/graph_config.yml"
    with open(graph_config_path, 'r') as f:
        graph_config = yaml.safe_load(f)
    return list(graph_config.keys())

def cramers_v(x, y):
    """ Calculate Cramer's V for two categorical variables. """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def anova_test(x, y):
    """ ANOVA test between categorical and continuous variables."""
    categories = x.unique()
    groups = [y[x == category] for category in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def correlation_ratio(categories, measurements):
    """ Measure of the relationship between a categorical and numerical variable. """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def correlation_matrix(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the correlation matrix for a DataFrame including categorical columns.
    For continuous-continuous use Pearson correlation
    For continuous-categorical use correlation ratio
    For categorical-categorical use Cramer's V

    Args:
        df (pd.DataFrame): The DataFrame to calculate the correlation matrix for.
    """
    corr = pd.DataFrame(index=df.columns, columns=df.columns)
    # Calculate the correlation matrix
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return corr

def moving_average(x, npoints=11):
    if npoints % 2 == 0:
        npoints += 1  # Ensure npoints is odd for a symmetric window
    window = np.ones(npoints) / npoints
    xav = np.convolve(x, window, mode='same')
    xav[:npoints // 2] = np.nan
    xav[-npoints // 2:] = np.nan
    return xav

def deriv(x, y):
    dydx = np.zeros(len(y))
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # for any 3 points where x direction changes sign set to nan
    mask = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0
    dydx[1:-1][mask] = np.nan
    return dydx

def smoothed_derivative(x, y, npoints=21):
    x_smooth = moving_average(x, npoints)
    y_smooth = moving_average(y, npoints)
    dydx_smooth = deriv(x_smooth, y_smooth)
    dydx_smooth[deriv(x_smooth,np.arange(len(x_smooth))) < 0] *= -1
    dydx_smooth[abs(dydx_smooth) > 20] = np.nan
    return dydx_smooth

#======================================================================================================================#
#======================================================= LAYOUT =======================================================#
#======================================================================================================================#

colorscales = px.colors.named_colorscales()

app.layout = html.Div([
    html.Div(
        [
            dcc.Tabs(id="tabs", value='tab-1', children=[
                #################### SAMPLES TAB ####################
                dcc.Tab(
                    label='Samples',
                    value='tab-1',
                    children=[
                        html.Div(
                            [
                                html.H4("Select samples to plot:"),
                                dcc.Dropdown(
                                    id='samples-dropdown',
                                    options=[
                                        {'label': name, 'value': name} for name in get_sample_names()
                                    ],
                                    value=[],
                                    multi=True,
                                ),
                                html.Div(style={'margin-top': '50px'}),
                                html.H4("Time graph"),
                                html.Label('X-axis', htmlFor='samples-time-x'),
                                dcc.Dropdown(
                                    id='samples-time-x',
                                    options=['Unix time','From protection','From formation','From cycling'],
                                    value='From formation',
                                    multi=False,
                                ),
                                dcc.Dropdown(
                                    id='samples-time-units',
                                    options=['Seconds','Minutes','Hours','Days'],
                                    value='Seconds',
                                ),
                                html.Label('Y-axis', htmlFor='samples-time-y'),
                                dcc.Dropdown(
                                    id='samples-time-y',
                                    options=['V (V)'],
                                    value='V (V)',
                                    multi=False,
                                ),
                                html.Div(style={'margin-top': '50px'}),
                                html.H4("Cycles graph"),
                                html.P("X-axis: Cycle"),
                                html.Label('Y-axis', htmlFor='samples-cycles-y'),
                                dcc.Dropdown(
                                    id='samples-cycles-y',
                                    options=[
                                        'Specific discharge capacity (mAh/g)',
                                        'Normalised discharge capacity (%)',
                                        'Efficiency (%)',
                                    ],
                                    value='Specific discharge capacity (mAh/g)',
                                    multi=False,
                                ),
                                html.Div(style={'margin-top': '50px'}),
                                html.H4("One cycle graph"),
                                dcc.Dropdown(
                                    id='samples-cycle-x',
                                    options=['Q (mAh)', 'V (V)', 'dQdV (mAh/V)'],
                                    value='Q (mAh)',
                                ),
                                dcc.Dropdown(
                                    id='samples-cycle-y',
                                    options=['Q (mAh)', 'V (V)', 'dQdV (mAh/V)'],
                                    value='V (V)',
                                ),
                            ],
                        style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'horizontalAlign': 'left', 'height': '90vh'}
                        ),
                        html.Div(
                            [
                                dcc.Loading(
                                    id='loading-samples-data-store',
                                    type='circle',
                                    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                                    children=[
                                        dcc.Store(id='samples-data-store', data={'data_sample_time': {}, 'data_sample_cycle': {}}),
                                        dcc.Graph(id='time-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='vs time',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'})}, config={'scrollZoom':True, 'displaylogo':False}, style={'height': '45vh'}),
                                        # two graphs side by side
                                        html.Div(
                                            [
                                                # First graph on the left
                                                html.Div(
                                                    dcc.Graph(id='cycles-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='vs cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'})}, config={'scrollZoom':True, 'displaylogo':False}, style={'height': '45vh'}),
                                                ),
                                            ],
                                            style={'width': '50%', 'display': 'inline-block', 'height': '45vh'}
                                        ),
                                        html.Div(
                                            [
                                                # Second graph on the right
                                                html.Div(
                                                    dcc.Graph(id='cycle-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='One cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'})}, config={'scrollZoom':True, 'displaylogo':False}, style={'height': '45vh'}),
                                                ),
                                            ],
                                            style={'width': '50%', 'display': 'inline-block', 'height': '45vh'}
                                        ),
                                    ],
                                    fullscreen=False,
                                ),
                            ],
                        style={'width': '75%', 'display': 'inline-block', 'paddingLeft': '20px', 'horizontalAlign': 'right', 'height': '90vh'}
                        ),
                    ],
                ),
                #################### BATCHES TAB ####################
                dcc.Tab(
                    label='Batches',
                    value='tab-2',
                    children=[
                        dcc.Loading(
                            id='loading-page',
                            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                            type='circle',
                            fullscreen=False,
                            children=[ dcc.Store(id='batches-data-store', data={'data_batch_cycle': {}}),],
                        ),   
                        html.Div(
                            [
                                html.H4("Select batches to plot:"),
                                dcc.Dropdown(
                                    id='batches-dropdown',
                                    options=[
                                        {'label': name, 'value': name} for name in get_batch_names()
                                    ],
                                    value=[],
                                    multi=True,
                                ),
                                html.Div(style={'margin-top': '50px'}),
                                html.P("X-axis: Cycle"),
                                html.P("Y-axis"),
                                dcc.Dropdown(
                                    id='batch-cycle-y',
                                    options=['Normalised discharge capacity (%)'],
                                    value='Normalised discharge capacity (%)',
                                    multi=False,
                                ),
                                html.P("Colormap"),
                                dcc.Dropdown(
                                    id='batch-cycle-color',
                                    # TODO remove these default options, change to run ID
                                    options=[
                                        'Max voltage (V)',
                                        'Actual N:P ratio',
                                        '1/Formation C',
                                        'Electrolyte name',
                                    ],
                                    value='1/Formation C',
                                    multi=False,
                                ),
                                dcc.Dropdown(
                                    id='batch-cycle-colormap',
                                    options=colorscales,
                                    value='viridis'
                                )
                            ],
                        style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'horizontalAlign': 'left', 'height': '40vh'}
                        ),
                        html.Div(
                            [
                                # Top graph
                                dcc.Graph(
                                    id='batch-cycle-graph',
                                    figure={
                                        'data': [],
                                        'layout': go.Layout(template=graph_template, title='vs cycle', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                    }, 
                                    config={'scrollZoom': True, 'displaylogo':False,  'toImageButtonOptions': {'format': 'svg',}},
                                    style={'height': '40vh'}
                                ),
                            ],
                        style={'width': '75%', 'display': 'inline-block', 'paddingLeft': '20px', 'horizontalAlign': 'right', 'height': '40vh'}
                        ),
                        # Div for two graphs side by side
                        html.Div(
                            [
                                # First graph on the left
                                html.Div(
                                    dcc.Graph(
                                        id='batch-correlation-map',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(template=graph_template, title='Click to show correlation', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                        }, 
                                        config={'scrollZoom': False, 'displaylogo':False, 'modeBarButtonsToRemove' : ['zoom2d','pan2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d'], 'toImageButtonOptions': {'format': 'png', 'width': 1000, 'height': 800}},
                                        style={'height': '50vh'},
                                    ),
                                    style={'width': '50%', 'display': 'inline-block', 'height': '50vh'}
                                ),
                                # Second graph on the right
                                html.Div(
                                    [
                                        html.Div(
                                            html.P("Choose a colormap"),
                                            style={'width': '25%', 'display': 'inline-block', 'paddingLeft': '50px', 'verticalAlign': 'middle'},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id='batch-correlation-color',
                                                    options=[
                                                        'Max voltage (V)',
                                                        'Actual N:P ratio',
                                                        '1/Formation C',
                                                        'Electrolyte name',
                                                    ],
                                                    value='Actual N:P ratio',
                                                    multi=False,
                                                ),
                                            ],
                                            style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id='batch-correlation-colorscale',
                                                    options=colorscales,
                                                    value='viridis',
                                                    multi=False,
                                                ),
                                            ],
                                            style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                                        ),
                                        dcc.Graph(
                                            id='batch-correlation-graph',
                                            figure={
                                                'data': [],
                                                'layout': go.Layout(template=graph_template, title='params', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                            },
                                            config={'scrollZoom': True, 'displaylogo':False, 'toImageButtonOptions': {'format': 'svg'}},
                                            style={'height': '45vh'},
                                        ),
                                    ],
                                    style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}
                                ),
                            ],
                            style={'height': '50vh'}
                        ),
                    ],
                ),
            ]),
        ],
        style={'height': '90vh'},
    ),
])

#======================================================================================================================#
#===================================================== CALLBACKS ======================================================#
#======================================================================================================================#

#----------------------------- SAMPLES CALLBACKS ------------------------------#

# TODO fix errors when data does not include any Formation C values

# Update the samples data store
@app.callback(
    Output('samples-data-store', 'data'),
    Output('samples-time-y', 'options'),
    Output('samples-cycles-y', 'options'),
    Input('samples-dropdown', 'value'),
    Input('samples-data-store', 'data'),
)
def update_sample_data(samples, data):
    # Get rid of samples that are no longer selected
    for sample in list(data['data_sample_time'].keys()):
        if sample not in samples:
            data['data_sample_time'].pop(sample)
            data['data_sample_cycle'].pop(sample)

    for sample in samples:
        # Check if already in data store
        if sample in data['data_sample_time'].keys():
            continue

        # Otherwise import the data
        run_id = _run_from_sample(sample)
        data_folder = "K:/Aurora/cucumber/snapshots" # hardcoded for now
        file_location = os.path.join(data_folder,run_id,sample)

        # Get raw data
        files = os.listdir(file_location)
        cycling_files = [
            os.path.join(file_location,f) for f in files
            if (f.startswith('snapshot') and f.endswith('.h5'))
        ]
        if not cycling_files:
            print(f"No cycling files found in {file_location}")
            continue
        df, metadata = combine_hdfs(cycling_files)
        data['data_sample_time'][sample] = df.to_dict(orient='list')

        # Get the analysed file
        try:
            analysed_file = next(f for f in files if (f.startswith('cycles') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            cycle_dict = json.load(f)["data"]
        if not cycle_dict or 'Cycle' not in cycle_dict.keys():
            continue
        data['data_sample_cycle'][sample] = cycle_dict
    
    # Update the y-axis options
    time_y_vars = set(['V (V)'])
    for sample, data_dict in data['data_sample_time'].items():
        time_y_vars.update(data_dict.keys())
    time_y_vars = list(time_y_vars)

    cycles_y_vars = set(['Specific discharge capacity (mAh/g)', 'Normalised discharge capacity (%)', 'Efficiency (%)'])
    for sample, data_dict in data['data_sample_cycle'].items():
        cycles_y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
    cycles_y_vars = list(cycles_y_vars)
    
    return data, time_y_vars, cycles_y_vars

# Update the time graph
@app.callback(
    Output('time-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-time-x', 'value'),
    Input('samples-time-units', 'value'),
    Input('samples-time-y', 'value'),
)
def update_time_graph(data, xvar, xunits, yvar):
    fig = px.scatter().update_layout(title='No data...', xaxis_title=f'Time ({xunits.lower()})', yaxis_title=yvar)
    fig.update_layout(template = graph_template)
    if not data['data_sample_time']:
        return fig

    multiplier = {'Seconds': 1, 'Minutes': 60, 'Hours': 3600, 'Days': 86400}[xunits]
    for sample, data_dict in data['data_sample_time'].items():
        uts = np.array(data_dict['uts'])
        if xvar == 'From protection':
            offset=uts[0]
        elif xvar == 'From formation':
            offset=uts[next(i for i, x in enumerate(data_dict['Cycle']) if x >= 1)]
        elif xvar == 'From cycling':
            offset=uts[next(i for i, x in enumerate(data_dict['Cycle']) if x >= 4)]
        else:
            offset=0

        trace = go.Scatter(
            x=(np.array(data_dict['uts']) - offset) / multiplier,
            y=data_dict[yvar],
            mode='lines',
            name=sample
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=f'{yvar} vs time',
    )

    return fig


# Update the cycles graph
@app.callback(
    Output('cycles-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-cycles-y', 'value'),
)
def update_cycles_graph(data, yvar):
    fig = px.scatter().update_layout(title='No data...', xaxis_title='Cycle', yaxis_title=yvar)
    fig.update_layout(template = graph_template)
    if not data['data_sample_cycle']:
        return fig
    for sample, cycle_dict in data['data_sample_cycle'].items():
        trace = go.Scatter(x=cycle_dict['Cycle'], y=cycle_dict[yvar], mode='lines+markers', name=sample)
        fig.add_trace(trace)
    fig.update_layout(title=f'{yvar} vs cycle')
    return fig

# Update the one cycle graph
@app.callback(
    Output('cycle-graph', 'figure'),
    Input('cycles-graph', 'clickData'),
    Input('samples-data-store', 'data'),
    Input('samples-cycle-x', 'value'),
    Input('samples-cycle-y', 'value'),
)
def update_cycle_graph(clickData,data,xvar,yvar):

    fig = px.scatter().update_layout(title='No data...', xaxis_title=xvar, yaxis_title=yvar)
    fig.update_layout(template = graph_template)
    if not data['data_sample_cycle']:
        return fig
    
    if not clickData:
        cycle = 1
    else:
        # clickData is a dict with keys 'points' and 'event'
        # 'points' is a list of dicts with keys 'curveNumber', 'pointNumber', 'pointIndex', 'x', 'y', 'text'
        point = clickData['points'][0]
        cycle = point['x']
    traces = []
    for sample, data_dict in data['data_sample_time'].items():
        # find where the cycle = cycle
        mask = np.array(data_dict['Cycle']) == cycle
        if not any(mask):
            continue
        mask_dict = {}
        mask_dict['V (V)'] = np.array(data_dict['V (V)'])[mask]
        mask_dict['Q (mAh)'] = np.array(data_dict['dQ (mAh)'])[mask].cumsum()
        mask_dict['dQdV (mAh/V)'] = smoothed_derivative(mask_dict['V (V)'], mask_dict['Q (mAh)'])
        trace = go.Scatter(x=mask_dict[xvar], y=mask_dict[yvar], mode='lines', name=sample)
        fig.add_trace(trace)

    fig.update_layout(title=f'{yvar} vs {xvar} for cycle {cycle}')
    return fig

    


#----------------------------- BATCHES CALLBACKS ------------------------------#

# Update the batches data store
@app.callback(
    Output('batches-data-store', 'data'),
    Output('batch-cycle-y', 'options'),
    Output('batch-cycle-color', 'options'),
    Output('batch-correlation-color', 'options'),
    Input('batches-dropdown', 'value'),
    Input('batches-data-store', 'data'),
)
def update_batch_data(batches, data):
    # Remove batches no longer in dropdown
    for batch in list(data['data_batch_cycle'].keys()):
        if batch not in batches:
            data['data_batch_cycle'].pop(batch)

    data_folder = "K:/Aurora/cucumber/batches" # Hardcoded for now

    for batch in batches:
        if batch in data['data_batch_cycle'].keys():
            continue
        file_location = os.path.join(data_folder, batch)
        files = os.listdir(file_location)
        try:
            analysed_file = next(f for f in files if (f.startswith('batch') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data['data_batch_cycle'][batch] = json_data["data"]
    
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data_list

    data['data_all_samples'] = data_list
    
    # Update the y-axis options
    y_vars = set(['Normalised discharge capacity (%)'])
    for data_dict in data_list:
        y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
    y_vars = list(y_vars)

    color_vars = set(['Max voltage (V)', 'Actual N:P ratio', '1/Formation C', 'Electrolyte name'])
    for data_dict in data['data_all_samples']:
        color_vars.update([k for k,v in data_dict.items() if not isinstance(v,list)])
    color_vars = list(color_vars)

    return data, y_vars, color_vars, color_vars


# Update the batch cycle graph
@app.callback(
    Output('batch-cycle-graph', 'figure'),
    Input('batches-data-store', 'data'),
    Input('batch-cycle-y', 'value'),
    Input('batch-cycle-color', 'value'),
    Input('batch-cycle-colormap', 'value'),
)
def update_batch_cycle_graph(data, variable, color, colormap):
    if not data or not data['data_batch_cycle']:
        fig = px.scatter().update_layout(title='No data...', xaxis_title='Cycle', yaxis_title='')
        fig.update_layout(template = graph_template)
        return fig

    # data['data_batch_cycle'] is a dict with keys as batch names and values as dicts
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data_list
    df = pd.concat(pd.DataFrame(d) for d in data_list)

    if df.empty:
        return px.scatter().update_layout(title=f'{variable} vs cycle', xaxis_title='Cycle', yaxis_title=variable)
    
    # Use Plotly Express to create the scatter plot
    # TODO copy the stuff from other plots here
    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']

    fig = px.scatter(df, x='Cycle', y=variable, title=f'{variable} vs cycle', color=color, color_continuous_scale=colormap)
    fig.update_layout(scattermode="group", scattergap=0.75, template = graph_template)
    # Plotly Express returns a complete figure, so you can directly return it
    return fig

# Update the correlation map
@app.callback(
    Output('batch-correlation-map', 'figure'),
    # Output('batch-correlation-vars', 'options'),
    Input('batches-data-store', 'data'),
)
def update_correlation_map(data):
    # data is a list of dicts
    if not data['data_batch_cycle']:
        fig = px.imshow([[0]], color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template = graph_template)
        return fig
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data_list]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    if not dfs:
        fig = px.imshow([[0]], color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template = graph_template)
        return fig
    df = pd.concat(dfs, ignore_index=True)

    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']

    # remove columns where all values are the same
    df = df.loc[:, df.nunique() > 1]

    # remove other unnecessary columns
    columns_not_needed = [
        'Sample ID',
        'Last efficiency (%)',
        'Last specific discharge capacity (mAh/g)',
        'Capacity loss (%)',
    ]
    df = df.drop(columns=columns_not_needed)

    def customwrap(s,width=30):
        return "<br>".join(textwrap.wrap(s,width=width))

    df.columns = [customwrap(col) for col in df.columns]

    # Calculate the correlation matrix
    corr = correlation_matrix(df)

    # Use Plotly Express to create the heatmap
    fig = px.imshow(corr, color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)

    fig.update_layout(
        coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=0, b=0),
        template = graph_template,
    )
    return fig

@app.callback(
    Output('batch-correlation-graph', 'figure'),
    Input('batch-correlation-map', 'clickData'),
    Input('batches-data-store', 'data'),
    Input('batch-correlation-color', 'value'),
    Input('batch-correlation-colorscale', 'value'),
)
def update_correlation_graph(clickData, data, color, colormap):
    if not clickData:
        fig = px.scatter().update_layout(xaxis_title='X-axis Title', yaxis_title='Y-axis Title')
        fig.update_layout(template = graph_template)
        return fig
    # clickData is a dict with keys 'points' and 'event'
    # 'points' is a list of dicts with keys 'curveNumber', 'pointNumber', 'pointIndex', 'x', 'y', 'text'
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data_list]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    if not dfs:
        fig = px.scatter().update_layout(xaxis_title='X-axis Title', yaxis_title='Y-axis Title')
        fig.update_layout(template = graph_template)
        return fig
    df = pd.concat(dfs, ignore_index=True)
    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']

    point = clickData['points'][0]
    xvar = point['x']
    yvar = point['y']

    xvar = xvar.replace('<br>', ' ')
    yvar = yvar.replace('<br>', ' ')

    hover_columns = [
        'Sample ID',
        'Formation C',
    ]
    # remove columns which are not in the data
    hover_columns = [col for col in hover_columns if col in df.columns]

    fig = px.scatter(
        df,
        x=xvar,
        y=yvar,
        color=color,
        color_continuous_scale=colormap,
        custom_data=df[hover_columns],
        hover_name="Sample ID",
        hover_data={
            'Formation C': True,
        },
    )
    fig.update_traces(
        marker=dict(size=10,line=dict(color='black', width=1)),
    )
    fig.update_layout(
        xaxis_title=xvar,
        yaxis_title=yvar,
        template = graph_template,
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)