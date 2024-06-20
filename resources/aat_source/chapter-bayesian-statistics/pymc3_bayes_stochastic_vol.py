# stochastic-volatility-model: pymc3_bayes_stochastic_vol.py
import os

import datetime
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pandas_datareader as pdr
import yfinance as yf

import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk
import seaborn as sns


def obtain_plot_amazon_prices_dataframe(start_date, end_date):
    """
    Download, calculate and plot the AMZN logarithmic returns.
    """
    print("Downloading and plotting AMZN log returns...")
    
    # amzn = pdr.get_data_yahoo("AMZN", start_date, end_date) # ???
    amzn_ticker = yf.Ticker("AMZN")
    amzn = amzn_ticker.history(period="1d", start="1997-05-15", end="2024-06-11")
    # amzn.to_csv("amzn.csv")

    amzn["returns"] = amzn["Adj Close"]/amzn["Adj Close"].shift(1)
    amzn.dropna(inplace=True)
    amzn["log_returns"] = np.log(amzn["returns"])
    amzn["log_returns"].plot(linewidth=0.5)
    plt.ylabel("AMZN daily percentage returns")
    plt.show()  
    return amzn


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

def configure_sample_stoch_vol_model(log_returns, samples):
    """
    Configure the stochastic volatility model using PyMC3
    in a 'with' context. Then sample from the model using
    the No-U-Turn-Sampler (NUTS).

    Plot the logarithmic volatility process and then the
    absolute returns overlaid with the estimated vol.
    """
    print("Configuring stochastic volatility with PyMC3...")
    model = pm.Model()
    with model:
        sigma = pm.Exponential('sigma', 50.0, testval=0.1)
        nu = pm.Exponential('nu', 0.1)
        s = GaussianRandomWalk('s', sigma**-2, shape=len(log_returns))
        logrets = pm.StudentT(
            'logrets', nu,
            lam=pm.math.exp(-2.0*s),
            observed=log_returns
        )

    print("Fitting the stochastic volatility model...")
    with model:
        trace = pm.sample(samples, cores=20)
    print(model.vars)   # [sigma_log__ ~ TransformedDistribution, nu_log__ ~ TransformedDistribution, s ~ GaussianRandomWalk]
    print("-----------------")
    pm.traceplot(trace, model.vars[:-1])
    plt.show()

    print("Plotting the log volatility...")
    k = 10
    opacity = 0.03
    plt.plot(trace[s][::k].T, 'b', alpha=opacity)
    plt.xlabel('Time')
    plt.ylabel('Log Volatility')
    plt.show()

    print("Plotting the absolute returns overlaid with vol...")
    plt.plot(np.abs(np.exp(log_returns))-1.0, linewidth=0.5)
    plt.plot(np.exp(trace[s][::k].T), 'r', alpha=opacity)
    plt.xlabel("Trading Days")
    plt.ylabel("Absolute Returns/Volatility")
    plt.show()


if __name__ == "__main__":
    # # State the starting and ending dates of the AMZN returns
    # start_date = "2006-01-01"
    # end_date = "2015-12-31"

    # # Obtain and plot the logarithmic returns of Amazon prices
    # amzn_df = obtain_plot_amazon_prices_dataframe(start_date, end_date)
    # log_returns = np.array(amzn_df["log_returns"])

    # ----------------- new code -----------------
    ticker = "MSFT" # MSFT, AAPL, AMZN, GOOGL, FB
    max_hist = "1997-05-15 : 2024-06-11"
    req_hist = "2018-05-15 : 2023-05-15"
    df = obtain_plot_prices_dataframe(ticker, max_hist, req_hist)

    # # Plot the log returns
    # df["log_returns"].plot(linewidth=0.5)
    # plt.title(f"Log Returns of {ticker}")
    # plt.show()

    log_returns = np.array(df["log_returns"])

    # Configure the stochastic volatility model and carry out MCMC sampling using NUTS, plotting the trace
    samples = 2000
    configure_sample_stoch_vol_model(log_returns, samples)