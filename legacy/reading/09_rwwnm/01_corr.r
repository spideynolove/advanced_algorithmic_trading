# set.seed(1)

# # autocorrelation function with 1000 samples -------
# acf(rnorm(1000))

# # estimate the variance White Noise -------
# print(var(rnorm(1000, mean=0, sd=1)))

# # autocorrelation of a random walk -------
# set.seed(4)
# x <- w <- rnorm(1000)
# for (t in 2:1000) {
#    x[t] <- x[t-1] + w[t]
# }
# layout(1:3)
# plot(x, type='l')
# acf(x)
# # for difference (đạo hàm) -------
# acf(diff(x))

# ------- read csv data -------
tsla_dat <- read.csv('data/tsla.csv')
print(tsla_dat)
acf(diff(tsla_dat$Price), na.action=na.omit)
