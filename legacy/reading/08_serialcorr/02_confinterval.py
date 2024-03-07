# confidence-interval
'''
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
'''
import numpy as np
import scipy.stats
import statsmodels.stats.api as sms
import pandas as pd
# ------------------------------------------------------
'''
Way 1
'''


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


df = pd.read_csv('data/Gold_Daily.csv')
# print(df.info())
df_close = df.Close[2900:2989]
print(mean_confidence_interval(df_close))
# df_low = df.Low[2900:2989]
# print(mean_confidence_interval(df_low))
# df_vol = df.Volume[2900:2989]
# print(mean_confidence_interval(df_vol))
# ------------------------------------------------------
'''
Way 2
'''
