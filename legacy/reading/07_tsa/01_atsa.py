from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from itertools import product
warnings.filterwarnings('ignore')
# ###########################################################################
# ------------------------------- ARMA(p,q) Model --------------------------
'''
# https://towardsdatascience.com/advanced-time-series-analysis-with-arma-and-arima-a7d9b589ed6d

Concept: 2 ways to model time series
# 1. MA: moving average (have order and noise)
# 2. AR: autoregressive (have order and noise)
So ARMA(p,q) is simply the combination of 2 above models -
relationship (random noise + itself at a previous step)
'''
# ------------ simulate ARMA(1,1) process (order 1) ---------------
''' Needed functions:
plot_pacf / plot_acf / ArmaProcess / acorr_ljungbox / SARIMAX
adfuller / pacf / acf / tqdm_notebook
'''
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])

sim_ARMA_data = ArmaProcess(ar1, ma1).generate_sample(nsample=10000)
# ------------- visualize generated time series (200 points) ----------
'''
plt.figure(figsize=[15, 7.5])
plt.plot(sim_ARMA_data)
plt.title('Simulated ARMA(1,1) process')
plt.xlim([0, 200])  # used to get or set the x-limits
'''
# ------------- ACF and PACF plots ----------
'''
plot_acf(sim_ARMA_data)
plot_pacf(sim_ARMA_data)
'''
# ------------ simulate ARMA(2, 2) process (order 2) ---------------
'''
# what is order ???
# how to get the function???
'''
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
sim_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)

# ------------- visualize generated time series (200 points) ----------
'''
plt.figure(figsize=[15, 7.5])
plt.plot(sim_ARMA2_data)
plt.title('Simulated ARMA(2,2) process')
plt.xlim([0, 200])
'''
# ------------- ACF and PACF plots ----------
'''
plot_acf(sim_ARMA_data)
plot_pacf(sim_ARMA_data)
'''
# ###########################################################################
# ------------------------------- ARIMA(p,d,q)  Model -----------------------
'''
1. differencing
2. integration is the opposite of differencing
3. "Differencing is useful to remove the trend in a
time series and make it stationary"
4. Simplest way: subtracting a point a t-1 from time t
5. ACF and PACF cannot be used except using ARIMA(p,d,0) process
6.
'''
df = pd.read_csv('data/jj.csv')  # quarterly EPS Johnson&Johnson
# ------------- dataset plotting ----------------
'''
plt.figure(figsize=[15, 7.5])  # Set dimensions for figure
plt.scatter(df['date'], df['data'])
plt.title('Quarterly EPS for Johnson & Johnson')
plt.ylabel('EPS per share ($)')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
'''
# ------------- ACF and PACF plots ----------
'''
# how to read the plot???
plot_acf(df['data'])
plot_pacf(df['data'])
'''
# ------------- Augmented Dickey-Fuller test ----------
'''
ad_fuller_ret = adfuller(df['data'])
print(f'ADF Statistic: {ad_fuller_ret[0]}')
print(f'p-value: {ad_fuller_ret[1]}')   # p-value > 0.05->ts is non-stationary
'''
# plt.show()
# ------ transformation/ differencing: remove trend/change in variance -----
df['data'] = np.log(df['data'])
df['data'] = df.data.diff()
df = df.drop(df.index[0])
# -------------- plot the new transformed data -----------------
'''
plt.figure(figsize=[15, 7.5])  # Set dimensions for figure
plt.plot(df['data'])
plt.title("Log Difference of Quarterly EPS for Johnson & Johnson")
'''
# ------------- Augmented Dickey-Fuller test ----------
'''
ad_fuller_ret = adfuller(df['data'])
print(f'ADF Statistic: {ad_fuller_ret[0]}')
print(f'p-value: {ad_fuller_ret[1]}')   # p-value < 0.0005->ts is stationary
'''
# ------------- ACF and PACF plots ----------
'''
# how determine order???
plot_acf(df['data'])
plot_pacf(df['data'])
'''
# plt.show()
# --------------- Akaike’s Information Criterion (AIC) ----------
'''
# 1. express: likelihood + number of parameters
# 2. cannot use order of differencing ??? because differencing change
# data's likelihood
3. consider lowest AIC, at least parameters numbers
4. Main purpose: ARIMA best order
'''


# ------------- optimize ARIMA functions ----------
def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        - order_list - list with (p, d, q) tuples
        - exog - the exogenous variable
    """
    rets = []
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except Exception as e:
            print(f"Error: {e}")
            continue
        aic = model.aic
        rets.append([order, modelaic])
    ret_df = pd.DataFrame(rets)
    ret_df.columns = ['(p, d, q)', 'AIC']
    # Sort in ascending order, lower AIC is better
    ret_df = ret_df.sort_values(
        by='AIC', ascending=True).reset_index(drop=True)
    return ret_df


# ------------- find optimize params ----------
# combinations with orders (p,q) ranging from 0 to 8
ps = range(0, 8, 1)
qs = range(0, 8, 1)
params = product(ps, qs)
# keep the differencing order equal to 1
d = 1
params_list = list(params)
order_list = []
'''
for each in params_list:
    each = list(each)
    each.insert(1, 1)   # ??? add differencing
    each = tuple(each)
    order_list.append(each)
ret_df = optimize_ARIMA(order_list, exog=df['data'])
ret_df.to_csv('data/aic_arima.csv')
'''
ret_df = pd.read_csv('data/aic_arima.csv')
optimize_order = ret_df['(p, d, q)'].iloc[0]
# print(optimize_order)  # lowest AIC->ARIMA params

# ------------- best model summary ----------
'''
optimize_order = [int(i) for i in optimize_order[1:-1].split(',')]
best_model = SARIMAX(df['data'], order=optimize_order).fit()
# print(best_model.summary())  # consider ar, ma, sigma
'''

# ------------- Ljung-Box test----------
'''
ljung_box, p_value = acorr_ljungbox(best_model.resid)   # residuals
print(f'Ljung-Box test: {ljung_box[:10]}')
print(f'p-value: {p_value[:10]}')
# 1. ??? null hypothesis
# 2. no autocorrelation, all values > 0.05
'''
# ------------- ACF and PACF plots ----------
'''
# ??? white noise, how to read from ACF and PACF plot
plot_pacf(best_model.resid)
plot_acf(best_model.resid)
plt.show()
'''
# ----------------------- Conclusion ------------------------
''' # General Modelling Procedure
1. Plot data and identify unsual observations. Understand data pattern
2. Transormation or differencing, remove trend and stabilize the variance
3. Stationarity test. If not apply another transformation or differencing
4. ACF and PACF plotting, estimate order of the MA or AR process
5. find lowest AIC
6. Ljung-Box test to check residuals (the best is white noise)
7. Calculate forecasts
'''
