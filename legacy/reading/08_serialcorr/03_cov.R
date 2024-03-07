# # Covariance
# set.seed(1)
# x <- seq(1, 100) + 20.0*rnorm(1:100)
# set.seed(2)
# y <- seq(1, 100) + 20.0*rnorm(1:100)
# # plot(x, y)
# z <- cov(x, y)
# print(z)

# ---------------------------------------
# Correlogram

set.seed(1)

# # normally distributed random variables ------
# w <- rnorm(100)
# print(w)

# # Linear Trend ------
# w <- seq(1, 100)

# Repeated 10 times - Sequence (1:10) ------
w <- rep(1:10, 10)

# autocorrelation function ------
acf(w)
