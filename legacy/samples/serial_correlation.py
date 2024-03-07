import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import investpy as iv
from datetime import date, datetime
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Huge Stock Examples ---------------------
'''
https://www.kaggle.com/jeegarmaru/timeseriesanalysis-autocorrelation/notebook
'''
# ------------------------------------------
accurate = 4
DF_COLUMNS = ['Open', 'High', 'Low', 'Close']
ADD_COLUMNS = ['mean', 'var', 'std']


def get_stock_data(symbol="AAPL"):
    # start_ = '01/01/2010'
    # today = date.today().strftime("%d/%m/%Y")
    # df = iv.stocks.get_stock_historical_data(
    #     symbol, 'united states', start_, today)
    # df.to_csv(f"data/{symbol}.csv")

    # ------------------- Read data -------------------
    df = pd.read_csv(f"data/{symbol}.csv", index_col='Date',
                     parse_dates=True, na_values='nan',
                     usecols=['Date', 'Close'])
    # print(df.tail())
    return df


def huge_stock_ds(*args, **kwargs):
    tsla = get_stock_data('TSLA')

    # tsla.plot(figsize=(15, 5))      # common plot
    # lag_plot(tsla['Close'])     # lag_plot with default lag=1

    # plt.clf()
    fig, ax = plt.subplots(figsize=(15, 5))

    # # try pandas autocorrelation_plot --------
    # autocorrelation_plot(tsla['Close'], ax=ax)

    # # # try statsmodels ACF plot --------
    # plot_acf(tsla['Close'], lags=300, use_vlines=False, ax=ax)
    plot_pacf(tsla['Close'], lags=30, use_vlines=False, ax=ax)
    plt.show()
    return

# End Huge Stock Examples ---------------------


def get_df(name='GBPUSD', periods=20):
    ''' Calculate "E"xpectation, "V"ariance and "C"ovariance'''
    today = datetime.today().strftime("%d/%m/%Y")

    # --------- get multiple file ---------
    # quotes = ["GBP/USD", "GBP/JPY"]
    # for quote in quotes:
    #     df = iv.currency_crosses.get_currency_cross_historical_data(quote, '01/01/2020', today)
    #     quote = quote.replace('/', '')
    #     df.to_csv(f'data\\{quote}.csv')

    # --------- read one file ---------
    df = pd.read_csv(f'data\\{name}.csv', index_col=0)
    df.drop('Currency', axis=1, inplace=True)
    # print(df.info())        # describe
    return df[-periods:]


def autocovariance(Xi, N, k, Xs):
    autoCov = 0
    for i in np.arange(0, N-k):
        autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
    return (1/(N-1))*autoCov


def calc_stats(df):
    '''mean std var'''
    df['mean'] = df.mean(axis=1).round(accurate)
    df['std'] = df.std(axis=1).round(accurate)
    df['var'] = df.var(axis=1).round(accurate)
    return df


def calc_stats_mean(df):
    '''mean of total mean rows'''
    for column in ADD_COLUMNS:
        yield f'Mean of {column}', df[column].mean().round(accurate)


def filter_where(joined_df):
    # --------- OHLC price ---------
    corr_joined_df = joined_df.corr().to_numpy()
    corr_joined_np = np.where(corr_joined_df > 0.8, corr_joined_df, 0)
    print(corr_joined_np)


    # print(corr_joined_np)
def autocorrelation(option=1):
    # -------------------- GBPUSD | GBPJPY pair --------------------
    gu_df = get_df()
    gj_df = get_df(name='GBPJPY')
    joined_df = gu_df.join(gj_df, lsuffix='_gu', rsuffix='_gj')
    # --------- Only Close price ---------
    closed_dfs = joined_df[['Close_gu', 'Close_gj']]
    
    if option == 1:
        ''' acf and pacf '''
        # print("Example: I say Hello world")
        huge_stock_ds()

    elif option == 2:
        ''' calc_stats Expectation, Variance '''
        mod_df = calc_stats(gu_df)
        print(mod_df)
        print(dict(calc_stats_mean(mod_df)))

    elif option == 3:
        ''' Covariance '''
        print(joined_df.cov())
        print(closed_dfs.cov())
        
    elif option == 4:
        ''' Correlation '''
        print(joined_df.corr())
        print(closed_dfs.corr()) 

    elif option == 5:
        ''' Flatten test '''
        corr_joined_np_ravel = np.ravel(corr_joined_np)
        # print(corr_joined_np_ravel)

        N = np.size(corr_joined_np_ravel)
        k = 5           # why choose
        # print(N)      # 64
        corr_joined_nps = np.average(corr_joined_np_ravel)    # average nums
        # print(corr_joined_nps)
        print("Autocovariance joined_nps:", autocovariance(
            corr_joined_np_ravel, N, k, corr_joined_nps))
        print("Autocorrelation: ", autocovariance(
            corr_joined_np_ravel, N, k, corr_joined_nps) / autocovariance(corr_joined_np_ravel, N, 0, corr_joined_nps))
        # print("Numpy built-in: ", np.corrcoef(corr_joined_np))

        # how to check Fixed Linear Trend ---------------------------

        # how to check Repeated Sequence --------------------------- 
    elif option == 6:
        ''' TBD '''
        pass


def main():
    autocorrelation(int(sys.argv[1]))


if __name__ == "__main__":
    main()
