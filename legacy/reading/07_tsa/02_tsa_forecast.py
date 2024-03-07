# ----------------- import some libraries ----------------
import warnings
from sklearn.metrics import median_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.metrics import r2_score, median_absolute_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
warnings.filterwarnings('ignore')


# Exponential smoothing -----------------
def exponential_smoothing(series, alpha):
    ret = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        ret.append(alpha*series[n]+(1-alpha)*ret[n-1])
    return ret


def plot_exponential_smoothing(series, alphas):
    plt.figure(figsize=(15, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label=f'Alpha {alpha}')
    plt.plot(series.values, 'c', label="Actual")
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Exponential smoothing')


# Double exponential smoothing -----------------
def double_exponential_smoothing(series, alpha, beta):
    ret = [series[0]]  # first value is same as series
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1]-series[0]
        # forecasting
        if n >= len(series):
            value = ret[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value+(1-alpha)*(level+trend)
        trend = beta*(level-last_level)+(1-beta)*trend
        ret.append(level+trend)
    return ret


def plot_double_exponential_smoothing(series, alphas, betas):
    plt.figure(figsize=(15, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta),
                     label=f'Alpha {alpha}, beta {beta}')
    plt.plot(series.values, 'c', label="Actual")
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Double Exponential smoothing')


# SMA------------------------------
def plot_ma(series, window, plot_intervals=False, scale=1.96):
    roll_mean = series.rolling(window=window).mean()
    plt.figure(figsize=(15, 8))
    plt.title(f'Moving average\n window size = {window}')
    plt.plot(roll_mean, 'g', label='Rolling mean trend')
    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], roll_mean[window:])
        std = np.std(series[window:] - roll_mean[window:])
        lower_bound = roll_mean - (mae + scale*std)
        upper_bound = roll_mean + (mae + scale*std)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound)
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc='best')
    plt.grid(False)


# add MAPE -------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


PATH = 'data/stock_prices_sample.csv'

df = pd.read_csv(PATH, index_col=['DATE'], parse_dates=['DATE'])

# Clean the data: remove unwanted columns -------------
df = df[df.TICKER != 'GEF']
df = df[df.TYPE != 'Intraday']
drop_cols = ['SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 'ADJ_VOLUME',
             'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 'ADJ_OPEN', 'VOLUME',
             'FREQUENCY', 'TYPE', 'FIGI']
df.drop(drop_cols, axis=1, inplace=True)
# print(df.tail())

# Exploratory Data Analysis (EDA) -------------
'''
# Plot closing price ----------------
plt.figure(figsize=(15, 8))
plt.plot(df.CLOSE)
plt.title('Closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()
'''
# moving average plot ----------------
'''
# Smooth by the previous 5 days (by 1 week)
plot_ma(df.CLOSE, 5)
# Smooth by the previous month (30 days)
plot_ma(df.CLOSE, 30)
# Smooth by previous quarter (90 days)
plot_ma(df.CLOSE, 90, plot_intervals=True)
'''
# plot_exponential_smoothing ----------------
'''
plot_exponential_smoothing(df.CLOSE, [0.05, 0.3])
'''
# plot_exponential_smoothing ----------------
'''
plot_double_exponential_smoothing(df.CLOSE, alphas=[0.05, 0.3],
                                  betas=[0.9, 0.02])
'''
# plt.show()

# Dickey-Fuller test and stationary process -----------------


def tsplot(y, lags=None, figsize=(12, 7)):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title(
            f'Time Series Analysis Plots\n Dickey-Fuller: p={p_value:.5f}')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# Preparing before Modelling --------------------
tsplot(df.CLOSE, lags=30)   # non-stationary, no clear seasonality

# Take the first difference to remove to make the process stationary
df_diff = df.CLOSE - df.CLOSE.shift(1)
tsplot(df_diff[1:], lags=30)    # stationary

# plt.show()

# SARIMA (ps, qs , d), (P, D, Qs, s) ----------------------------
ps = range(0, 5)
qs = range(0, 5)
d = 1
Ps = range(0, 5)
Qs = range(0, 5)
D = 1
s = 5

# Create a list with all possible combinations of parameters ----
params = product(ps, qs, Ps, Qs)
params_list = list(params)

# Train many SARIMA models to find the best set of parameters


def optimize_SARIMA(params_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    rets = []
    best_aic = float('inf')
    for param in tqdm_notebook(params_list):
        try:
            model = sm.tsa.statespace.SARIMAX(df.CLOSE, order=(
                param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except Exception as e:
            print(f'Error: {e}')
            continue
        aic = model.aic
        # save the best model, AIC and params
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        rets.append([param, aic])
    ret_table = pd.DataFrame(rets)
    ret_table.columns = ['parameters', 'AIC']
    # sort in ascending order, to find the lowest AIC
    ret_table = ret_table.sort_values(
        by='AIC', ascending=True).reset_index(drop=True)
    return ret_table


'''
# optimize SARIMA then write data
ret_table = optimize_SARIMA(params_list, d, D, s)
ret_table.to_csv('data/aic_sarima.csv')
'''
# speedup by read data
ret_table = pd.read_csv('data/aic_sarima.csv')
# Set parameters that give the lowest AIC (Akaike Information Criteria)
parameters = [int(i) for i in ret_table.parameters[0][1:-1].split(',')]
p, q, P, Q = parameters
best_model = sm.tsa.statespace.SARIMAX(df.CLOSE, order=(
    p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())
