#!/usr/bin/env python
# coding: utf-8

# In[21]:


import dash
import dash_table
from dash import dcc, html, Input, Output, State
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import plotly.express as px
import webbrowser

# Function to fetch stock data for a given symbol and time frame
def fetch_stock_data(symbol, years):
    stock_data = yf.download(symbol, period=f"1y", auto_adjust=True)
    return stock_data

# Function to fetch and forward-fill stock data with automated filling of missing values
def fetch_and_auto_forward_fill_stock_data(symbol, years):
    stock_data = yf.download(symbol, period=f"1y", auto_adjust=True)
    
    # Check for missing dates
    missing_dates = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max()).difference(stock_data.index)
    
    if not missing_dates.empty:
        # Fill missing dates with data from the previous available date
        for missing_date in missing_dates:
            previous_date = missing_date - pd.DateOffset(days=1)
            if previous_date in stock_data.index:
                stock_data.loc[missing_date] = stock_data.loc[previous_date]
    
    stock_data = stock_data.sort_index()  # Sort by date
    return stock_data

# Define stock symbols
stock_symbols = [
    'AAPL', 'PFE', 'DIS', 'INTC', 'NKE', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'NFLX', 'TSLA', 'KO', 'JPM', 'IBM'
]

# Create a dictionary to hold stock data for each symbol
stock_data_dict = {symbol: fetch_and_auto_forward_fill_stock_data(symbol, 1) for symbol in stock_symbols}

# Create a Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout of the app
app.layout = html.Div(style={
    'backgroundColor': '#f2f2f2',
    'color': '#000000',
    'textAlign': 'center',
}, children=[
    html.H1("Stock Investment Risk Analysis Dashboard", style={'margin': '20px'}),

    # Input for adding new stock symbol
    dcc.Input(id='new-stock-input', type='text', placeholder='Enter New Stock Symbol', style={'width': '30%', 'margin': '10px'}),

    # Input for selecting number of years
    dcc.Input(id='years-input', type='number', placeholder='Enter Number of Years', style={'width': '20%', 'margin': '10px'}),

    # Button to add new stock
    html.Button('Add New Stock', id='add-new-stock-button', style={'margin': '10px'}),

    # Dropdown for selecting multiple stocks
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in stock_symbols],
        value=[stock_symbols[0]],
        multi=True,
        style={'width': '60%', 'margin': '10px auto', 'alignItems': 'center'}
    ),

    # Add an empty div to display min-max close prices
    html.Div(id='min-max-close-prices', style={'margin': '20px'}),

    # Tabs for organizing different charts and tables
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Stock Performance', value='tab-1'),
        dcc.Tab(label='Model Comparison', value='tab-2'),
        dcc.Tab(label='Model R-squared', value='tab-3'),
        dcc.Tab(label='Risk Metrics', value='tab-4'),
    ],style={'margin': '0 auto', 'width': '50%'}),

    # Tab content
    html.Div(id='tab-content'),

    # Conclusion and disclaimer sections
    html.Div([
        # Conclusion section
        html.Div([
            html.H2("Key Insights", style={'width': '80%', 'margin': '20px'}),
            html.P("Based on the analysis of stock performance, risk metrics, and model simulations, here are some key insights:", style={'width': '70%', 'margin': '20px'}),
            html.Ul([
                html.Li("Consider stocks with lower volatility if seeking lower-risk investments."),
                html.Li("High R-squared values indicate good model fit, but evaluate models holistically."),
                html.Li("Diversification across sectors can help manage risk."),
                html.Li("Use the dashboard as a starting point; conduct thorough research before investing."),
            ], style={'width': '80%', 'text-align': 'left'}),
        ], style={'font-size': '12px', 'width': '80%', 'margin': '0 auto', 'padding': '20px'}),
        
        # Disclaimer section
        html.Div([
            html.H2("Disclaimer", style={'margin': '20px', 'font-size': '12px'}),
            html.P("This dashboard provides analysis and insights based on historical stock data, risk metrics, and model simulations. The information presented should not be considered as financial advice. Investing in the stock market involves risks, and past performance does not guarantee future results. Users should conduct their own research, seek professional advice, and understand that any investment decision is their own responsibility."),
        ], style={'width': '80%', 'margin': '0 auto', 'padding': '10px'}),
    ], style={'font-size': '10px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'space-between', 'alignItems': 'center', 'text-align': 'center'}),
])

# Callback to fetch data for a new stock
@app.callback(
    Output('stock-dropdown', 'options'),
    Output('stock-dropdown', 'value'),
    [Input('add-new-stock-button', 'n_clicks')],
    [State('new-stock-input', 'value'),
     State('years-input', 'value'),
     State('stock-dropdown', 'options'),
     State('stock-dropdown', 'value')]
)
def add_new_stock(n_clicks, new_stock_symbol, years, existing_options, selected_stocks):
    if n_clicks is None:
        return existing_options, selected_stocks

    # Check if the new stock symbol is already in the dropdown options
    if any(new_stock_symbol == option['value'] for option in existing_options):
        return existing_options, selected_stocks

    # Fetch data for the new stock and add it to the dictionary
    new_stock_data = fetch_and_auto_forward_fill_stock_data(new_stock_symbol, years)
    stock_data_dict[new_stock_symbol] = new_stock_data

    # Update the dropdown options and add the new stock to the selected stocks
    new_options = existing_options + [{'label': new_stock_symbol, 'value': new_stock_symbol}]
    selected_stocks.append(new_stock_symbol)

    return new_options, selected_stocks

# Callback to calculate and display min-max close prices
@app.callback(
    Output('min-max-close-prices', 'children'),
    [Input('stock-dropdown', 'value')]
)
def calculate_min_max_close_prices(selected_stocks):
    min_max_prices_text = []

    for selected_stock in selected_stocks:
        selected_stock_data = stock_data_dict[selected_stock]
        min_close_price = selected_stock_data['Close'].min()
        max_close_price = selected_stock_data['Close'].max()

        min_max_prices_text.append(html.P(f"Stock: {selected_stock}, Min Close Price: {min_close_price:.2f}, Max Close Price: {max_close_price:.2f}"))

    return min_max_prices_text

# Callback to update the tab content based on the selected tab
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tab_content(tab_value):
    if tab_value == 'tab-1':
        # Stock performance chart
        return dcc.Graph(id='stock-performance-chart', style={'height': '300px', 'margin': '0 auto', 'width': '50%'})
    elif tab_value == 'tab-2':
        # Model comparison chart
        return dcc.Graph(id='model-comparison-chart', style={'height': '300px', 'margin': '0 auto', 'width': '50%'})
    elif tab_value == 'tab-3':
        # Model R-squared tables for all three models
        return [
            html.Div([
                html.H2(f"Model R-squared - {model_name}", style={'margin': '5px'}),
                dash_table.DataTable(
                    id=f'model-r2-table-{model_name}',
                    style_table={'margin': '10px', 'border-collapse': 'collapse'},
                    style_cell={'font-weight': 'bold'},
                ),
            ], style={'margin': '0 auto', 'width': '50%'})
            for model_name in ['Linear Regression', 'Random Forest Regression', 'Support Vector Machine']
        ]
    elif tab_value == 'tab-4':
        # Risk metrics table
        return dash_table.DataTable(
            id='risk-metrics-table',
            columns=[{'name': 'Stock', 'id': 'Stock'}, {'name': 'Volatility', 'id': 'Volatility'}],
            style_table={'margin': '0 auto', 'width': '50%', 'border-collapse': 'collapse'},
            style_cell={'font-weight': 'bold'},
        )

# Callback to update the stock performance chart
@app.callback(
    Output('stock-performance-chart', 'figure'),
    [Input('stock-dropdown', 'value')]
)
def update_stock_performance_chart(selected_stocks):
    # Fetch selected stock data from the dictionary
    selected_stock_data = {symbol: stock_data_dict[symbol] for symbol in selected_stocks}

    # Create performance chart
    performance_chart = {
        'data': [{'x': stock.index, 'y': stock['Close'], 'type': 'line', 'name': symbol} for symbol, stock in selected_stock_data.items()],
        'layout': {
            'title': 'Stock Performance',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Close Price'}
        }
    }

    return performance_chart

# Callback to update the model comparison chart
@app.callback(
    Output('model-comparison-chart', 'figure'),
    [Input('stock-dropdown', 'value')]
)
def update_model_comparison_chart(selected_stocks):
    # Fetch selected stock data from the dictionary
    selected_stock_data = {symbol: stock_data_dict[symbol] for symbol in selected_stocks}

    # Perform the Monte Carlo simulation for selected stocks
    results = []

    for selected_stock in selected_stocks:
        for model_name, model in [('Linear Regression', LinearRegression()),
                                  ('Random Forest Regression', RandomForestRegressor()),
                                  ('Support Vector Machine', SVR())]:
            mse_values = []
            r2_values = []
            # Define the number of simulations
            num_simulations = 100

            for i in range(num_simulations):
                # Smooth the percentage changes of the selected stock using a moving average
                window_size = 5
                selected_stock_pct_change = selected_stock_data[selected_stock]['Close'].pct_change()
                smoothed_data = selected_stock_pct_change.rolling(window=window_size).mean()

                # Remove NaN values from the smoothed data
                valid_indices = ~np.isnan(smoothed_data)
                smoothed_data = smoothed_data[valid_indices]

                X_perturbed = {
                    symbol: data['Close'][valid_indices] if symbol != selected_stock else data['Close'][valid_indices] + np.random.normal(0, 0.05, size=len(smoothed_data))
                    for symbol, data in selected_stock_data.items()
                }
                y_actual = smoothed_data

                model.fit(pd.DataFrame(X_perturbed), y_actual)
                y_pred = model.predict(pd.DataFrame(X_perturbed))

                mse_values.append(mean_squared_error(y_actual, y_pred))
                r2_values.append(r2_score(y_actual, y_pred))

            results.append({
                'Stock': selected_stock,
                'Model': model_name,
                'Mean Squared Error': np.mean(mse_values),
                'Mean R-squared': np.mean(r2_values),
                'Smoothed Data - Close': smoothed_data,  # Store smoothed data
                'Original Data - Close': selected_stock_data[selected_stock]['Close'].values[valid_indices],  # Store original data
            })

            # Print the difference between original and smoothed data
            print(f"Stock: {selected_stock}, Model: {model_name}")
            print("Difference (Original - Smoothed):")
            print(selected_stock_data[selected_stock]['Close'].values[valid_indices] - smoothed_data)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Create the bar plot for simulation results comparison
    comparison_plot = px.bar(df, x='Stock', y='Mean R-squared', color='Model', barmode='group',
                             title='Model Comparison',
                             labels={'Mean R-squared': 'Mean R-squared'})

    # Set y-axis range to be between 0 and 1
    comparison_plot.update_yaxes(range=[0, 1])

    return comparison_plot

# Callback to update the model R-squared tables for all three models
@app.callback(
    Output('model-r2-table-Linear Regression', 'data'),
    Output('model-r2-table-Random Forest Regression', 'data'),
    Output('model-r2-table-Support Vector Machine', 'data'),
    [Input('stock-dropdown', 'value')]
)
def update_model_r2_tables(selected_stocks):
    # Fetch selected stock data from the dictionary
    selected_stock_data = {symbol: stock_data_dict[symbol] for symbol in selected_stocks}

    # Calculate R-squared values for all three models
    r2_data = {model_name: [] for model_name in ['Linear Regression', 'Random Forest Regression', 'Support Vector Machine']}

    for selected_stock in selected_stocks:
        for model_name, model in [('Linear Regression', LinearRegression()),
                                  ('Random Forest Regression', RandomForestRegressor()),
                                  ('Support Vector Machine', SVR())]:
            r2_values = []
            # Define the number of simulations
            num_simulations = 100
            
            for i in range(num_simulations):
                # Smooth the percentage changes of the selected stock using a moving average
                window_size = 5
                selected_stock_pct_change = selected_stock_data[selected_stock]['Close'].pct_change()
                smoothed_data = selected_stock_pct_change.rolling(window=window_size).mean()

                # Remove NaN values from the smoothed data
                valid_indices = ~np.isnan(smoothed_data)
                smoothed_data = smoothed_data[valid_indices]

                # Create a perturbed copy of the smoothed data
                perturbed_data = smoothed_data + np.random.normal(0, 0.05, size=len(smoothed_data))

                X_perturbed = {
                    symbol: data['Close'][valid_indices] if symbol != selected_stock else perturbed_data
                    for symbol, data in selected_stock_data.items()
                }
                y_actual = smoothed_data

                model.fit(pd.DataFrame(X_perturbed), y_actual)
                y_pred = model.predict(pd.DataFrame(X_perturbed))

                r2_values.append(r2_score(y_actual, y_pred))
            
            r2_data[model_name].append({
                'Stock': selected_stock,
                'Mean R-squared': np.mean(r2_values)
            })

    return r2_data['Linear Regression'], r2_data['Random Forest Regression'], r2_data['Support Vector Machine']

# Callback to update the risk metrics table
@app.callback(
    Output('risk-metrics-table', 'data'),
    [Input('stock-dropdown', 'value')]
)
def update_risk_metrics_table(selected_stocks):
    # Fetch selected stock data from the dictionary
    selected_stock_data = {symbol: stock_data_dict[symbol] for symbol in selected_stocks}

    # Calculate risk metrics
    risk_metrics = {stock: np.std(selected_stock_data[stock]['Close'].pct_change()) for stock in selected_stocks}

    # Create risk metrics table data
    risk_metrics_data = [{'Stock': stock, 'Volatility': volatility} for stock, volatility in risk_metrics.items()]

    return risk_metrics_data

# Run the app
if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:8051/')
    app.run_server(debug=True, port=8051)

