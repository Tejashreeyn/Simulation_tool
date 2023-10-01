#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch stock data for a given symbol and time frame
def fetch_stock_data(symbol, years):
    stock_data = yf.download( symbol , period=f"1y", auto_adjust=True)
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

# Function to calculate and plot the comparison between original log returns and smoothed log returns
def compare_original_vs_smoothed_log_returns(selected_stock, window_size=5):
    selected_stock_data = stock_data_dict[selected_stock]
    selected_stock_close = selected_stock_data['Close']
    
    # Calculate log returns
    log_returns = np.log(selected_stock_close / selected_stock_close.shift(1))
    
    # Calculate the smoothed log returns using a moving average
    smoothed_log_returns = log_returns.rolling(window=window_size).mean()

    # Remove NaN values from the smoothed log returns
    valid_indices = ~np.isnan(smoothed_log_returns)
    smoothed_log_returns = smoothed_log_returns[valid_indices]

    # Plot the comparison using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(log_returns[valid_indices], label='Original Log Returns', marker='o')
    plt.plot(smoothed_log_returns, label=f'Log Returns (Window Size {window_size})', marker='o')
    plt.title(f"Original Log Returns vs. Smoothed Log Returns for {selected_stock} by moving average")
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file (must be before plt.show())
    plt.savefig(f'{selected_stock}_Smoothing by moving average.jpg', format='jpg')

    # Display the plot
    plt.show()

# Function to perform Logarithmic Transformation
def logarithmic_transformation(selected_stock):
    selected_stock_data = stock_data_dict[selected_stock]
    selected_stock_close = selected_stock_data['Close']

    # Calculate log returns
    log_returns = np.log(selected_stock_close / selected_stock_close.shift(1))

    # Plot the log returns
    plt.figure(figsize=(10, 6))
    plt.plot(log_returns, label='Log Returns', marker='o')
    plt.title(f"Logarithmic Transformation for {selected_stock}")
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file (must be before plt.show())
    plt.savefig(f'{selected_stock}_log_transformation_plot.jpg', format='jpg')

    # Display the plot
    plt.show()

# Function to perform Z-Score Standardization
def z_score_standardization(selected_stock):
    selected_stock_data = stock_data_dict[selected_stock]
    selected_stock_close = selected_stock_data['Close']

    # Calculate log returns
    log_returns = np.log(selected_stock_close / selected_stock_close.shift(1))

    # Calculate z-scores
    z_scores = (log_returns - np.mean(log_returns)) / np.std(log_returns)

    # Plot the z-scores
    plt.figure(figsize=(10, 6))
    plt.plot(z_scores, label='Z-Scores', marker='o')
    plt.title(f"Z-Score Standardization for {selected_stock}")
    plt.xlabel('Date')
    plt.ylabel('Z-Scores')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file (must be before plt.show())
    plt.savefig(f'{selected_stock}_z_score_standardization_plot.jpg', format='jpg')

    # Display the plot
    plt.show()

# Define stock symbols
stock_symbols = [
    'AAPL', 'PFE', 'DIS', 'INTC', 'NKE', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'NFLX', 'TSLA', 'KO', 'JPM', 'IBM'
]

# Create a dictionary to hold stock data for each symbol
stock_data_dict = {}

# Fetch and store stock data, and print original and forward-filled data
for symbol in stock_symbols:
    stock_data = fetch_stock_data(symbol, 1)  # Fetch data for 1 year
    forward_filled_data = fetch_and_auto_forward_fill_stock_data(symbol, 1)  # Fetch forward-filled data

    stock_data_dict[symbol] = stock_data
    stock_data_dict[f"{symbol}_forward_filled"] = forward_filled_data  # Store forward-filled data with a modified key
    
    # Combine original data and forward-filled data side by side
    combined_data = pd.concat([stock_data['Close'].tail(10), forward_filled_data['Close'].tail(10)], axis=1)
    combined_data.columns = ['Original Data- Close', 'Forward-Filled Data- Close']

    # Print the combined data
    print(f"{'='*40}\n{symbol} - Last 10 days:\n{'='*40}")
    print(combined_data)

    # Continue with the rest of your code for transformations and visualizations
    # Calculate and plot the comparison for all stocks using forward-filled data
    compare_original_vs_smoothed_log_returns(f"{symbol}_forward_filled")

    # Calculate and plot logarithmic transformation for all stocks using forward-filled data
    logarithmic_transformation(f"{symbol}_forward_filled")

    # Calculate and plot z-score standardization for all stocks using forward-filled data
    z_score_standardization(f"{symbol}_forward_filled")

# Calculate SD and mean for all stocks using forward-filled data
sd_mean_data = []
for symbol in stock_symbols:
    selected_stock_data = stock_data_dict[f"{symbol}_forward_filled"]
    selected_stock_close = selected_stock_data['Close']

    # Calculate log returns using forward-filled data
    log_returns = np.log(selected_stock_close / selected_stock_close.shift(1))

    # Calculate SD and mean using forward-filled data
    sd = log_returns.std()
    mean = log_returns.mean()

    sd_mean_data.append({
        'Stock': symbol,
        'Standard Deviation': sd,
        'Mean': mean
    })

# Create a DataFrame from the results
df_sd_mean = pd.DataFrame(sd_mean_data)

# Print the table
print("Standard Deviation and Mean for Log Returns (Using Forward-Filled Data):")
print(df_sd_mean)


# In[ ]:




