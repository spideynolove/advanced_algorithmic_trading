# Step by step:
# 1. select priors for model parameters: std vs degrees of freedom
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
if __name__ == "__main__":
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (8, 4)})
# -----------------------------------------------------
    # linspace use much
    x = np.linspace(0.0, 5.0, 100)
    # print(x)
    # lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 4.0]
    lambdas = [0.5, 1.0, 2.0]
    for lamb in lambdas:
        y = lamb*np.exp(-lamb*x)
        ax = plt.plot(x, y, label="$\\lambda=%s$" % lamb)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend(title="Parameters")
    plt.show()
# 2. Student’s t-distribution: mean zero, exponential variance
# kurtosis??
    x = np.linspace(-5.0, 5.0, 100)
    nus = [1.0, 2.0, 5.0, 50.0]
    for nu in nus:
        # t stand for student_t
        y = t.pdf(x, nu)
        ax = plt.plot(x, y, label="$\\nu=%s$" % nu)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend(title="Parameters")
    plt.show()
    '''

# PyMC3 Implementation-----------------------------

import datetime
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


# Obtaining the Price History ------------------------
def obtain_plot_amz_dataset(start_date, end_date):
    # download data
    '''
    amzn = web.get_data_yahoo('AMZN', start_date, end_date)
    # amzn = data.DataReader('AMZN', data_source='yahoo',
    #                        start=start_date, end=end_date)
    amzn.to_csv('data/amzn1.csv')
    '''
    # amzn = pd.read_csv('data/amzn1.csv')
    amzn = pd.read_csv('data/amzn.csv')
    # print(amzn.index)
    print("Loading and plotting AMZN log returns...")
    amzn['returns'] = amzn['Adj Close']/amzn['Adj Close'].shift(1)
    amzn.dropna(inplace=True)
    amzn['log_returns'] = np.log(amzn['returns'])
    amzn['log_returns'].plot(linewidth=0.5)
    plt.ylabel('AMZN daily percentage returns')
    plt.show()
    return amzn
# obtain_plot_amz_dataset('2011-06-06', '2021-06-06')


# Model Specification in PyMC3 ------------------------
def configure_sample_stoch_vol_model(log_returns, samples):
    print("Configuring stochastic volatility with PyMC3...")
    model = pm.Model()
    with model:
        # page 63
        sigma = pm.Exponential('sigma', 50.0, testval=0.1)
        nu = pm.Exponential('nu', 0.1)
        s = GaussianRandomWalk('s', sigma=(sigma**-2), shape=len(log_returns))
        logrets = pm.StudentT(
            'logrets', nu, lam=pm.math.exp(-2.0*s), observed=log_returns)
    # Fitting the Model with NUTS
    print('Fitting the stochastic volatility model')
    with model:
        # trace = pm.sample(samples)
        # with open('trace2.pickle', 'wb') as f:
        #     pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('trace.pickle', 'rb') as f:
        #     trace = pickle.load(f)
        with open('trace2.pickle', 'rb') as f:
            trace = pickle.load(f)
    '''
    # print(model.vars[:-1])
    # error when call traceplot
    pm.traceplot(trace, model.vars)
    plt.show()
    '''
    print("Plotting the log volatility...")
    # step-size is set to k = 10
    k = 10
    opacity = 0.03
    # why transpose : 'b' -> blue
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
    # 2011-06-06', '2021-06-06
    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2015, 12, 31)
    # Obtain and plot the logarithmic returns of amz prices
    df = obtain_plot_amz_dataset(start_date, end_date)
    log_returns = np.array(df['log_returns'])
    # MCMC sampling using NUTS ??? why 2000
    samples = 2000
    configure_sample_stoch_vol_model(log_returns, samples)
    pass
