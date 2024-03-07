'''
import libs part
read more:
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.html
# https://nguyenthieublog.wordpress.com/2018/12/26/lam-sao-de-biet-time-series-data-la-stationary-hay-non-stationary/
'''
import pandas as pd  # use for read_csv and datetime
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_process import ArmaProcess

import datetime as dt
from scipy import stats

if __name__ == "__main__":
    # ARIMA Models: Autoregressive Integrated Moving Average --------
    # machinelearningmastery-------------

    def parse(time):
        return dt.datetime.strptime('190'+time, '%Y-%m')

    series = pd.read_csv('data/shampoo.csv', header=0,
                         parse_dates=[0], index_col=0,
                         squeeze=True, date_parser=parse)
    series.index = series.index.to_period('M')

    # # test
    # series = pd.read_csv('data/Canada 10Y_spread_6_Monthly.csv')
    '''
    Shampoo Sales Dataset and autocorrelation plot of the time series
    '''
    # ---------------------------------------- # ------------------------
    # # first version
    # series.plot()

    # # use auto
    # autocorrelation_plot(series)
    # plt.show()
    # ---------------------------------------- # ------------------------
    '''
    # fit model
    This sets the lag value to 5 for autoregression,
    uses a difference order of 1 to make the time series stationary,
    and uses a moving average model of 0
    '''
    # model = ARIMA(series, order=(5, 1, 0))
    # model_fit = model.fit()

    # # # summary model
    # # print(model_fit.summary())

    # # line plot of residuals
    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # # plt.show()

    # # residuals density plot
    # residuals.plot(kind='kde')
    # # plt.show()

    # # # summary stats of residuals
    # # print(residuals.describe())
    # ---------------------------------------- # ------------------------
    '''
    # Rolling Forecast ARIMA Model
    forecast future time steps (index of the time steps)
    predict() vs forecast()
    split -> fit -> generate prediction
    rolling - dependence history observations
    '''
    # # split into train and test sets
    # X = series.values
    # size = int(len(X)*2/3)
    # train, test = X[0:size], X[size:len(X)]
    # history = [x for x in train]
    # predictions = list()
    # # walk forward validation
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(5, 1, 0))
    #     model_fit = model.fit()
    #     output = model_fit.forecast()

    #     yhat = output[0]
    #     predictions.append(yhat)

    #     obs = test[t]
    #     history.append(obs)

    #     print(f'Predicted: {yhat}\tExpected: {obs}')
    # # evaluate forecasts: root mean squared error (RMSE) vs MSE
    # rmse = sqrt(mean_squared_error(test, predictions))
    # print(f'Test RMSE: {rmse}')
    # # plot forecast
    # plt.plot(test)
    # plt.plot(predictions, color='r')
    # plt.show()

# ---------------------------------------- # ------------------------

''' statsmodels example
'''
# # print(sm.datasets.sunspots.NOTE)
# dat = sm.datasets.sunspots.load_pandas().data
# # write data
# dat.to_csv('data/sunspots.csv', index=False)

# read data
dat = pd.read_csv('data/sunspots.csv')
dat.index = pd.Index(
    sm.tsa.datetools.dates_from_range('1700', '2008'), freq='Y')
del dat['YEAR']
# # eg plot
# dat.plot(figsize=(12, 8))

# '''''''''''''''''''''''''''''''''''''''
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# # autocorrelation function plotting
# fig = sm.graphics.tsa.plot_acf(dat.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# # Partial Autocorrelation plotting
# fig = sm.graphics.tsa.plot_pacf(dat, lags=40, ax=ax2)
# # plt.show()

# '''''''''''''''''''''''''''''''''''''''

arma_mod20 = ARIMA(dat, order=(2, 0, 0)).fit()
arma_mod30 = ARIMA(dat, order=(3, 0, 0)).fit()
'''
for more information
https://www.statsmodels.org/v0.12.2/generated/statsmodels.tsa.arima.model.ARIMAResults.hqic.html
https://stats.stackexchange.com/questions/44992/what-are-the-values-p-d-q-in-arima
'''
# print(arma_mod20.params)
# print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
# print(arma_mod30.params)
# print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)

# what is good number for durbin_watson ???
# print(sm.stats.durbin_watson(arma_mod30.resid.values))
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax = arma_mod30.resid.plot(ax=ax)

# plt.show()

# '''''''''''''' qqplot '''''''''''''''''''''''''
resid = arma_mod30.resid
# # print(stats.normaltest(resid))
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

# ''''''''''' acf and pacf ''''''''''''''''''''''''''''

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# # autocorrelation function plotting
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# # Partial Autocorrelation plotting
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

# ''''''''''' predict_sunspots ''''''''''''''''''''''''''''

# r, q, p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
# # print(table.set_index('lag'))

# # How good does our model do?
# predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
# # print(predict_sunspots)


# def mean_forecast_err(y, yhat):
#     return y.sub(yhat).mean()


# print(mean_forecast_err(dat.SUNACTIVITY, predict_sunspots))

# ''''''''''' ARMA(4,1) ''''''''''''''''''''''''''''
np.random.seed(1234)

# ------------ test 1 -------------
# # autoregressive param
# arparams = np.array([1, .75, -.65, -.55, .9])
# # moving avg params
# maparams = np.array([1, .65])

# arma_t = ArmaProcess(arparams, maparams)
# # print(arma_t.isinvertible)
# # print(arma_t.isstationary)
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax.plot(arma_t.generate_sample(nsample=50))
# plt.show()
# ------------ end test 1 -------------

arparams = np.array([1, .35, -.15, .55, .1])
maparams = np.array([1, .65])
arma_t = ArmaProcess(arparams, maparams)
# print(arma_t.isstationary)    # why True

arma_rvs = arma_t.generate_sample(nsample=500, burnin=250, scale=2.5)

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)
# # plt.show()

# ''''''''''' ARMA(1,1) and ARMA(4,1)'''''''''''''''''''
# # arma11 = ARIMA(arma_rvs, order=(1, 0, 1)).fit()
# arma41 = ARIMA(arma_rvs, order=(4, 0, 1)).fit()
# # resid = arma11.resid
# resid = arma41.resid
# r, q, p = sm.tsa.acf(resid, fft=True, qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# # print(table.set_index('lag'))

# ''''''''''' CPI example '''''''''''''''''''
# macrodta = sm.datasets.macrodata.load_pandas().data
# macrodta.index = pd.Index(
#     sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'), freq='3M')
# print(macrodta.tail())
# macrodta.to_csv('data/macrodata.csv', index=False)

macrodta = pd.read_csv('data/macrodata.csv')
cpi = macrodta["cpi"]
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = cpi.plot(ax=ax)
ax.legend()
# plt.show()

# Augmented Dickey-Fuller unit root test.
'''
# https://www.google.com/search?q=unit+root+test+purpose&sxsrf=ALeKk03_05_XoS6dV7nNcAg0lpmvJapVcg%3A1623667484616&ei=HDPHYPyXJc34hwOuw5yAAw&oq=unit+root+test+purpose&gs_lcp=Cgdnd3Mtd2l6EAMyBggAEBYQHjoHCCMQsAMQJzoHCAAQRxCwAzoECCMQJzoECAAQQzoFCAAQywE6AggAOgUIIRCgAToICAAQCBANEB5QywlYkSFghiJoBHACeACAAYMBiAGgCZIBBDEwLjKYAQCgAQGqAQdnd3Mtd2l6yAEJwAEB&sclient=gws-wiz&ved=0ahUKEwj8lZ3d-JbxAhVN_GEKHa4hBzAQ4dUDCA4&uact=5
'''
print(sm.tsa.adfuller(cpi)[1])

# -----------------------------------------------------------------------


# ---------------------------------------- # ------------------------
''' statsmodels intro
provides classes and functions for the estimation of
many different statistical models, conducting statistical
tests, and statistical data exploration

'''
# # Load dataset
# dat = sm.datasets.get_rdataset('Guerry', 'HistData').data
# dat.to_csv('data/Guerry.csv')

# -------------------------------------------

# # read downloaded data
# dat = pd.read_csv('data/Guerry.csv')
# # Fit regression model using natural log: ols some data columns then fit them
# results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
# print(results.summary())

# -------------------------------------------

# # Generate artificial data instead of Load dataset
# nobs = 100
# X = np.random.random((nobs, 2))
# X = sm.add_constant(X)
# beta = [1, .1, .5]
# e = np.random.random(nobs)
# y = np.dot(X, beta) + e
# results = sm.OLS(y, X).fit()
# print(results.summary())
