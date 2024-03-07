# Mean Reversion Trading
# https://raposa.trade/how-to-build-your-first-mean-reversion-trading-strategy-in-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import investpy as iv
from datetime import date

'''
# price is too high or low compared to the SMA
+ too low -> price will go higher, towards the MA -> LONG
+ too high -> price will go lower, towards the MA -> SHORT
'''


def SMAMeanReversion(ticker, sma, threshhold,
                     country='united states', shorts=False,
                     start_date='01/01/2010', end_date='31/12/2020'):
    '''
    # download data
    df = iv.stocks.get_stock_historical_data(
        ticker, country, start_date, end_date)
    df.to_csv(f'data/{ticker}.csv')
    '''
    # read data
    df = pd.read_csv(f'data/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df['SMA'] = df['Close'].rolling(sma).mean()
    df['extension'] = (df['Close']-df['SMA'])/df['SMA']
    df['position'] = np.nan
    df['position'] = np.where(df['extension'] < -threshhold,
                              1, df['position'])
    if shorts:
        df['position'] = np.where(df['extension'] > threshold,
                                  -1, df['position'])
    df['position'] = np.where(np.abs(df['extension']) < 0.01,
                              0, df['position'])
    df['position'] = df['position'].ffill().fillna(0)
    # Calculate returns and statistics
    df['returns'] = df['Close'] / df['Close'].shift(1)
    df['log_returns'] = np.log(df['returns'])
    df['strat_returns'] = df['position'].shift(1) * df['returns']
    df['strat_log_returns'] = df['position'].shift(1) * df['log_returns']
    df['cum_returns'] = np.exp(df['log_returns'].cumsum())
    df['strat_cum_returns'] = np.exp(df['strat_log_returns'].cumsum())
    df['peak'] = df['cum_returns'].cummax()
    df['strat_peak'] = df['strat_cum_returns'].cummax()
    return df.dropna()


def getStratStats(data, risk_free_rate=0.02):
    sma_strat, buy_hold_strat = {}, {}

    # Total Returns
    sma_strat['tot_returns'] = np.exp(data['strat_log_returns'].sum()) - 1
    buy_hold_strat['tot_returns'] = np.exp(data['log_returns'].sum()) - 1

    # Mean Annual Returns
    sma_strat['annual_returns'] = np.exp(
        data['strat_log_returns'].mean() * 252) - 1
    buy_hold_strat['annual_returns'] = np.exp(
        data['log_returns'].mean() * 252) - 1

    # Annual Volatility
    sma_strat['annual_volatility'] = data['strat_log_returns'].std() * \
        np.sqrt(252)
    buy_hold_strat['annual_volatility'] = data['log_returns'].std() * \
        np.sqrt(252)

    # Sharpe Ratio
    sma_strat['sharpe_ratio'] = (
        sma_strat['annual_returns'] - risk_free_rate) \
        / sma_strat['annual_volatility']
    buy_hold_strat['sharpe_ratio'] = (
        buy_hold_strat['annual_returns'] - risk_free_rate) \
        / buy_hold_strat['annual_volatility']

    # Max Drawdown
    _strat_dd = data['strat_peak'] - data['strat_cum_returns']
    _buy_hold_dd = data['peak'] - data['cum_returns']
    sma_strat['max_drawdown'] = _strat_dd.max()
    buy_hold_strat['max_drawdown'] = _buy_hold_dd.max()

    # Max Drawdown Duration
    strat_dd = _strat_dd[_strat_dd == 0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days, (_strat_dd.index[-1]
                                               - strat_dd.index[-1]).days])

    buy_hold_dd = _buy_hold_dd[_buy_hold_dd == 0]
    buy_hold_diff = buy_hold_dd.index[1:] - buy_hold_dd.index[:-1]
    buy_hold_days = buy_hold_diff.map(lambda x: x.days).values
    buy_hold_days = np.hstack([buy_hold_days, (_buy_hold_dd.index[-1]
                                               - buy_hold_dd.index[-1]).days])
    sma_strat['max_drawdown_duration'] = strat_dd_days.max()
    buy_hold_strat['max_drawdown_duration'] = buy_hold_days.max()

    stats_dict = {'strat_stats': sma_strat,
                  'base_stats': buy_hold_strat}

    return stats_dict


ticker = 'TSLA'
SMA = 50
threshold = 0.1
# shorts = False
today = date.today().strftime("%d/%m/%Y")
data = SMAMeanReversion(ticker, SMA, threshold, end_date=today)
# print(data.tail())

stats_dict = getStratStats(data)
df_stats = pd.DataFrame(stats_dict).round(3)
print(df_stats)
