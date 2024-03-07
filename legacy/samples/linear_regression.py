import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
from sklearn import linear_model
'''
# a statistical technique

# multiple linear regression

# statistical technique: maximum likelihood estimation (MLE)

# contain: error, normally distributed (mean and variance)

# Probabilistic Interpretation
# 1. Think as joint probability model
# conditional of feature vector x + others model parameters
# conditional probability density (CPD) model
'''
# -------------------------------------------------------
'''
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 20, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)    # [] -> [[]]
    # define CPD
    beta0 = -5.0
    beta1 = 0.5
    Z = norm.pdf(Y, beta0 + beta1*X, 1.0)
    # Plot the surface with the "coolwarm"
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P(Y|X)')

    ax.view_init(elev=30, azim=50.0)
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.show()
'''
# -------------------------------------------------------
# Notice:
# 1. peak of distribution
# 2. variance does not increase with feature increase
# -------------------------------------------------------
'''
# Maximum Likelihood Estimation (MLE)
# 1. Likelihood and Negative Log Likelihood
Residual Sum of Squares also known as Sum of Squared Errors (SSE)

'''
# Simulated Data Example ----------------------------

# N values, with 80% used for training
N = 500
split = int(0.8*N)
# intercept and slope
alpha = 2.0
beta = 3.0
# mean and variance of the randomly distributed noise
eps_mu = 0.0
eps_sigma = 30.0
# mean and variance of the X data
X_mu = 0.0
X_sigma = 10.0
# Create the error/noise, X and y data
eps = np.random.normal(loc=eps_mu, scale=eps_sigma, size=N)
X = np.random.normal(loc=X_mu, scale=X_sigma, size=N)
y = alpha + beta*X + eps
X = X.reshape(-1, 1)

# Split up the features, X, and responses, y
# to training and test arrays
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

lr_model = linear_model.LinearRegression()
lr_model.fit(X_train, y_train)

print(
    f"Estimated intercept: {lr_model.intercept_: .2f},\
    slope: {lr_model.coef_[0]}")
# Create a scatterplot, plotting the estimated line
# of best fit from OLS
plt.scatter(X_test, y_test)
plt.plot(X_test, lr_model.predict(X_test), color='r', linewidth=1.0)

plt.xlabel('X')
plt.ylabel('y')
plt.show()
