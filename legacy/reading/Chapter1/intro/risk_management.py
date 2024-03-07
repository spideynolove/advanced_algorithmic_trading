from scipy.stats import norm
import pandas_datareader as web
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------------- ----------------------
# risk, profitability: https://github.com/topics/risk-management
# risk management builtin in some libs
'''medium example
# https://www.quantstart.com/articles/Value-at-Risk-VaR-for-Algorithmic-Trading-Risk-Management-Part-I/
# Variance-Covariance
'''
if __name__ == "__main__":

    def var_cov_var(P, c, mu, sigma):
        # Percent point function (inverse of cdf) at q(=1-p) of the given RV
        alpha = norm.ppf(1-c, mu, sigma)
        return P - P*(alpha+1)

    def rms():
        # https://haohanwang.medium.com/risk-management-strategy-for-algorithmic-trading-1-749baa0a6086
        '''
        start = dt(2021, 1, 1)
        end = dt(2021, 6, 17)
        tsla = web.DataReader('TSLA', 'yahoo', start, end)
        tsla['rets'] = tsla['Adj Close'].pct_change()
        tsla.to_csv('data/tsla.csv')
        '''
        tsla = pd.read_csv('data/tsla.csv')
        P = 1e5
        c = 0.95
        mu = np.mean(tsla['rets'])
        sigma = np.std(tsla['rets'])
        var = var_cov_var(P, c, mu, sigma)
        print(f"Value-at-Risk: {var:.2f}")

    # rms()

# ---------------------- ----------------------
    def test_graph():
        ax = plt.figure(figsize=(2, 1))
        plt.show()

    # test_graph()
'''
# https://campus.datacamp.com/courses/quantitative-risk-management-in-python/risk-and-return-recap?ex=1
'''
