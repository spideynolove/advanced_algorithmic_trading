import pandas as pd
from random import randrange
import numpy as np
# Serial Correlation
'''
Example
'''
tmp = [randrange(10) for _ in range(15)]
print(tmp)

tmp2 = [randrange(10) for _ in range(15)]
print(tmp2)

s = pd.Series(tmp)
s2 = pd.Series(tmp2)

'''
# autocorrelation in period lag
print(s.autocorr())
print(s.autocorr(lag=3))
'''


def histogram_intersection(a, b):
    return np.minimum(a, b).sum().round(decimals=1)


# correlation with other Series
print(s.corr(s2, method=histogram_intersection))
