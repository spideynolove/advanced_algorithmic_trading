from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# import pandas_datareader as web
import investpy as iv
from pykalman import KalmanFilter


# Scatterplot of ETF Prices
def draw_plot(etfs, prices):
    # create color map with 2 main colors: yellow, red
    plen = len(prices)
    color_map = plt.cm.get_cmap('YlOrRd')
    colors = np.linspace(0.1, 1, plen)

    # create scatterplot objects
    scatterplot = plt.scatter(prices[etfs[0]], prices[etfs[1]],
                              s=30, c=colors, edgecolors='k', alpha=0.8)
    # add color bar for color date then set axis tick labels
    colorbar = plt.colorbar(scatterplot)
    colorbar.ax.set_yticklabels([p for p in prices[::plen//9].Date])
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()


# Time-Varying Slope and Intercept
def calc_slope_intercept(etfs, prices):
    delta = 1e-5  # delta
    trans_cov = delta/(1-delta)*np.eye(2)   # trans_cov
    obs_mat = np.vstack([prices[etfs[0]], np.ones(
        prices[etfs[0]].shape)]).T[:, np.newaxis]  # obs_mat
    # KalmanFilter
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=trans_cov)
    # return state_means, state_covs
    return kf.filter(prices[etfs[1]].values)


def draw_slope_intercept_changes(prices, state_means):
    pd.DataFrame(dict(slope=state_means[:, 0],
                      intercept=state_means[:, 1]),
                 index=prices.index).plot(subplots=True)
    plt.show()


if __name__ == "__main__":
    etfs = ['iShares 20+ Year Treasury Bond', 'iShares 3-7 Year Treasury Bond']
    # etfs = ['TLT', 'IEI']
    start_date = '01/07/2015'
    end_date = '01/07/2021'

    '''
    # Get ETFs: iShares 20+ Year Treasury Bond/ iShares 3-7 Year Treasury Bond
    df1 = iv.etfs.get_etf_historical_data(
        etfs[0], 'united states', from_date=start_date,
        to_date=end_date, interval='Daily')
    df1.to_csv(f'data/{etfs[0]}.csv')
    df2 = iv.etfs.get_etf_historical_data(
        etfs[1], 'united states', from_date=start_date,
        to_date=end_date, interval='Daily')
    df2.to_csv(f'data/{etfs[1]}.csv')
    # with open('data/us_etfs.json', 'w', encoding='utf-8') as f:
    #     f.write(str(df))
    '''

    df1 = pd.read_csv(f'data/{etfs[0]}.csv')
    df2 = pd.read_csv(f'data/{etfs[1]}.csv')
    prices = pd.DataFrame(index=df2.index)  # df2 have less index
    prices['Date'] = df2['Date']
    prices[etfs[0]] = df1['Close']
    prices[etfs[1]] = df2['Close']
    # print(prices.tail())
    # draw_plot(etfs, prices)
    state_means, state_covs = calc_slope_intercept(etfs, prices)
    # print(state_means, state_covs)
    # draw_slope_intercept_changes(prices, state_means)
