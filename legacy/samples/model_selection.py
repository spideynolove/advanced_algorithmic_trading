from __future__ import print_function
import datetime as dt
import pprint
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold


# def create_lagged_series(symbol, start, end, lags=3):
def create_lagged_series(start_date, symbol='amzn', lags=3):
    # ------------- old code -------------
    # ts = web.DataReader(symbol, 'yahoo', start, end)

    ts = pd.read_csv(f'data/{symbol}.csv')
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts.set_index('Date', inplace=True)

    tslag = pd.DataFrame(index=ts.index)
    tslag['Today'] = ts['Adj Close']
    tslag['Volume'] = ts['Volume']

    for i in range(0, lags):
        tslag[f'Lag{str(i+1)}'] = ts['Adj Close'].shift(i+1)

    tsret = pd.DataFrame(index=tslag.index)
    tsret['Volume'] = tslag['Volume']
    tsret['Today'] = tslag['Today'].pct_change()*100.0

    for i, x in enumerate(tsret['Today']):
        if(abs(x) < 0.0001):
            tsret['Today'][i] = 0.0001

    for i in range(0, lags):
        tsret[f'Lag{str(i+1)}'] = tslag[f'Lag{str(i+1)}'].pct_change()*100.0

    tsret['Direction'] = np.sign(tsret['Today'])

    tsret = tsret[tsret.index >= start_date]
    return tsret


def validation_set_poly(random_seed, degree, X, y):
    sample_dict = dict([(f'seed_{i}', []) for i in range(1, random_seed+1)])

    pass


if __name__ == "__main__":
    start_dt = dt.datetime(2011, 6, 14)
    lags = create_lagged_series(start_date=start_dt)
