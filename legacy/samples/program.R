# # correlation example
# set.seed(1)
# x <- seq(1,100) + 20.0*rnorm(1:100)
# set.seed(2)
# y <- seq(1,100) + 20.0*rnorm(1:100)
# plot(x,y)
# --------------------------------------------
# plot(1:10)
# --------------------------------------------

# # normally distributed random variables
# set.seed(1)
# w <- rnorm(100)
# acf(w)
# --------------------------------------------

# # Fixed Linear Trend: autocorrelation
# w <- seq(1, 80, 10)
# acf(w)
# --------------------------------------------

# # Repeated Sequence
# w <- rep(1:10, 10)
# # print(length(w))
# acf(w)
# --------------------------------------------

# # DWN
# # Random Number Generators
# set.seed(1)
# acf(rnorm(100))
# --------------------------------------------

# # DWN has only 1 param: variance = 1 (std=1) and mean=0
# set.seed(1)
# x <- var(rnorm(1000, mean=0, sd=1))
# print(x)
# # acf(rnorm(100))
# --------------------------------------------

# # Correlogram
# set.seed(4)
# x <- w <- rnorm(1000)
# for (t in 2:1000) {
#    x[t] <- x[t-1] + w[t]
# }
# plot(x, type='l')
# # acf(x)
# acf(diff(x))
# --------------------------------------------

# Fitting to Financial Data
# require('quantmod')
# getSymbols('MSFT', src='yahoo')
# MSFT

if (FALSE) {
# NOT RUN {
setSymbolLookup(QQQ='yahoo',SPY='yahoo')

# loads QQQQ from yahoo (set with setSymbolLookup)
# loads SPY from MySQL (set with setSymbolLookup)
getSymbols(c('QQQ','SPY'))

# loads Ford market data from yahoo (the formal default)
getSymbols('F')

# loads symbol from MySQL database (set with setDefaults)
# getSymbols('DIA', verbose=TRUE, src='MySQL')

# loads Ford as time series class ts
getSymbols('F',src='yahoo',return.class='ts')

# load into a new environment
# data.env <- new.env()
# getSymbols("YHOO", env=data.env)
# ls.str(data.env)

# constrain to local scope
try(local( {
  getSymbols("AAPL")  # or getSymbols("AAPL", env=environment())
  str(AAPL)
  }))

exists("AAPL")  # FALSE

# assign into an attached environment
attach(NULL, name="DATA.ENV")
getSymbols("AAPL", env=as.environment("DATA.ENV"))
ls("DATA.ENV")
detach("DATA.ENV")

# directly return to caller
str( getSymbols("AAPL", env=NULL) )
str( getSymbols("AAPL", auto.assign=FALSE) )  # same

# }
}

