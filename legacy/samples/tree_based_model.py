from numba import jit, cuda
import code

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import sklearn
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle
# code.interact(local=locals)


def create_lagged_series(symbol, start, end, lags=3):
    ts = web.DataReader(symbol, 'yahoo', start, end)
    tslag = pd.DataFrame(index=ts.index)
    tslag['Today'] = ts['Adj Close']
    tslag['Volume'] = ts['Volume']

    for i in range(0, lags):
        tslag[f'Lag{str(i+1)}'] = ts['Adj Close'].shift(i+1)
    tsret = pd.DataFrame(index=tslag.index)
    tsret['Volume'] = tslag['Volume']
    tsret['Today'] = tslag['Today'].pct_change()*100.0

    for i in range(0, lags):
        tsret[f'Lag{str(i+1)}'] = tslag[f'Lag{str(i+1)}'].pct_change()*100.0
    tsret = tsret[tsret.index >= start]
    return tsret


# @jit
def main():
    # n_estimators defines the total number of estimators
    #  to use in the graph of the MSE
    random_state = 42
    n_jobs = 1
    # n_estimators = 1000
    n_estimators = 100
    step_factor = 10
    axis_step = int(n_estimators/step_factor)
    # Download ten years worth of Tesla adjusted closing prices
    start = datetime.datetime(2011, 6, 14)
    end = datetime.datetime(2021, 6, 14)
    tsla = create_lagged_series('TSLA', start, end, lags=3)
    # print(tsla.tail(10))
    # '''
    tsla.dropna(inplace=True)
    # Use the first three daily lags
    # scale the data to lie within -1 and +1
    X = tsla[['Lag1', 'Lag2', 'Lag3']]  # feature
    y = tsla['Today']   # label
    X = scale(X)
    y = scale(y)
    # training-testing split with 70/30 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)
    # MSE array each ensemble method
    estimators = np.zeros(axis_step)
    bagging_mse = np.zeros(axis_step)
    rf_mse = np.zeros(axis_step)
    boosting_mse = np.zeros(axis_step)
    # Estimate the Bagging MSE
    for i in range(0, axis_step):
        print(f'Bagging estimator: {step_factor*(i+1)} of {n_estimators}')
        bagging = BaggingRegressor(DecisionTreeRegressor(),
                                   n_estimators=step_factor*(i+1),
                                   n_jobs=n_jobs,
                                   random_state=random_state)
        bagging.fit(X_train, y_train)
        # # saving to pickle
        # with open('data/bagging.pickle', 'wb') as f:
        #     pickle.dump(bagging, f)
        mse = mean_squared_error(y_test, bagging.predict(X_test))
        estimators[i] = step_factor*(i+1)
        bagging_mse[i] = mse

    # Estimate the rf MSE
    for i in range(0, axis_step):
        print(
            f'Random forest estimator: {step_factor*(i+1)} of {n_estimators}')
        rf = RandomForestRegressor(n_estimators=step_factor*(i+1),
                                   n_jobs=n_jobs,
                                   random_state=random_state)
        rf.fit(X_train, y_train)
        # # saving to pickle
        # with open('data/randomforest.pickle', 'wb') as f:
        #     pickle.dump(rf, f)
        mse = mean_squared_error(y_test, rf.predict(X_test))
        estimators[i] = step_factor*(i+1)
        rf_mse[i] = mse

    # AdaBoost boosting algorithm
    for i in range(0, axis_step):
        print(f'Boost estimator: {step_factor*(i+1)} of {n_estimators}')
        boosting = AdaBoostRegressor(DecisionTreeRegressor(),
                                     n_estimators=step_factor*(i+1),
                                     random_state=random_state,
                                     learning_rate=0.01)
        boosting.fit(X_train, y_train)
        # # saving to pickle
        # with open('data/adaBoost.pickle', 'wb') as f:
        #     pickle.dump(boosting, f)
        mse = mean_squared_error(y_test, boosting.predict(X_test))
        estimators[i] = step_factor*(i+1)
        boosting_mse[i] = mse
    plt.figure(figsize=(8, 8))
    plt.title('Bagging, Random Forest and Boosting comparison')
    plt.plot(estimators, bagging_mse, 'b-', color='black', label='Bagging')
    plt.plot(estimators, rf_mse, 'b-', color='blue', label='Random Forest')
    plt.plot(estimators, boosting_mse, 'b-', color='red', label='AdaBoost')
    plt.legend(loc='upper right')
    plt.xlabel('Estimators')
    plt.ylabel('Mean Squared Error')
    plt.show()
    # '''


if __name__ == "__main__":
    # print(cuda.gpus)
    main()
    pass
