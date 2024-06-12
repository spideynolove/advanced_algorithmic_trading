import os
import numpy as np
import pandas as pd
import yfinance as yf

def obtain_plot_prices_dataframe(ticker, max_hist, req_hist):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    max_start, max_end = max_hist.split(" : ")
    req_start, req_end = req_hist.split(" : ")
    
    csv_file = f"data/{ticker}.csv"
    
    if not os.path.isfile(csv_file):
        print(f"Downloading {ticker} maximum history data...")
        ticker_data = yf.Ticker(ticker)
        data = ticker_data.history(period="1d", start=max_start, end=max_end)
        data.to_csv(csv_file)
    else:
        print(f"Reading {ticker} data from file...")
        data = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
        last_date_in_file = data.index[-1].strftime('%Y-%m-%d')
        
        if req_end > last_date_in_file:
            print(f"Updating {ticker} data to cover up to {req_end}...")
            ticker_data = yf.Ticker(ticker)
            new_data = ticker_data.history(period="1d", start=last_date_in_file, end=req_end)
            new_data.index.name = "Date"  # Ensure the index is named 'Date'
            new_data.to_csv(csv_file, mode='a', header=False)
            data = pd.read_csv(csv_file, index_col="Date", parse_dates=True)

    data.index = pd.to_datetime(data.index, utc=True)
    req_start = pd.to_datetime(req_start, utc=True)
    req_end = pd.to_datetime(req_end, utc=True)
    data = data.loc[req_start:req_end]
    
    data["returns"] = data["Close"] / data["Close"].shift(1)
    data.dropna(inplace=True)
    data["log_returns"] = np.log(data["returns"])
    
    return data

# Example usage
ticker = "AMZN"
max_hist = "1997-05-15 : 2024-05-01"
req_hist = "2022-05-15 : 2024-06-11"
data = obtain_plot_prices_dataframe(ticker, max_hist, req_hist)
print(data)
