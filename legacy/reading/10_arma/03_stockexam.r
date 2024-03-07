# library(glue)
# # ------- Test read Multi csv data -------
# # quotes <- list('tsla', 'aapl', 'amzn')
# quotes <- list('tsla', 'aapl')

# for (quote in quotes) {
#    print(glue('{quote}'))
#     dat <- read.csv(glue('data/{quote}.csv'))
#     # print(tsla_dat$Close)

#     # ------- plotting -------
#     layout(1:2)
#     plot(dat$Close, type='l')

#     # # acf plot ----------
#     # acf(diff(tsla_dat$Close), na.action=na.omit)

#     # differenced log returns
#     dat_logdiff = diff(log(dat$Close))
#     plot(dat_logdiff, type='l')
# }

# # --------------------------------------------- #
# # Autoregressive (AR) Models of order p
# # --------------------------------------------- #
# # read data -----
# tsla_dat <- read.csv('data/tsla.csv')
# # print(tsla_dat$Close)
# # # ------- plotting -------
# layout(1:3)
# plot(tsla_dat$Close, type='l')

# # # acf plot ----------
# # acf(diff(tsla_dat$Close), na.action=na.omit)

# # differenced log returns ----------
# tsla_logdiff = diff(log(tsla_dat$Close))
# plot(tsla_logdiff, type='l')

# # acf plot ----------
# acf(tsla_logdiff, na.action=na.omit)
# tsla_logdiff.ar <- ar(tsla_logdiff, na.action=na.omit)
# print(tsla_logdiff.ar$order)
# print(tsla_logdiff.ar$ar)

# # # standard error, variance, 95% confidence intervals
# # print(tsla_logdiff.ar$ar+c(-1.96, 1.96)*c(sqrt(tsla_logdiff.ar$asy.var)))

# # --------------------------------------------- #
# # AMZN example
# # Graph not showing up:
# # https://stackoverflow.com/questions/38068774/rstudio-suddenly-stopped-showing-plots-in-the-plot-pane
# # --------------------------------------------- #
# # download data -----
# require(quantmod)
# getSymbols("AMZN")
# # write.csv(AMZN, file = 'data/dl_amzn.csv')
# layout(1:3)
# print(plot(Cl(AMZN), type='l'))
# amznrt = diff(log(Cl(AMZN)))
# print(plot(amznrt, type='l'))
# acf(amznrt, na.action=na.omit)
# amznrt.ar <- ar(amznrt, na.action=na.omit)
# print(amznrt.ar$order)
# print(amznrt.ar$ar)
# print(amznrt.ar$asy.var)

# # # standard error, variance, 95% confidence intervals
# # print(amznrt.ar$ar+c(-1.96, 1.96)*c(sqrt(amznrt.ar$asy.var)))


# # --------------------------------------------- #
# # SP 500 example
# # --------------------------------------------- #
# require(quantmod)
# getSymbols("^GSPC")
# # print(plot(Cl(GSPC)))

# # # first order difference of the log (rt~return) ----
# # layout(1:2)
# gspcrt = diff(log(Cl(GSPC)))
# # print(plot(gspcrt))   # it show largest vol in 2008
# # acf(gspcrt, na.action=na.omit)

# # autoregressive model
# gspcrt.ar <- ar(gspcrt, na.action=na.omit)
# print(gspcrt.ar$order)
# print(gspcrt.ar$ar)

# --------------------------------------------- #
# Moving Average (MA) Models of order q
# --------------------------------------------- #
set.seed(1)
x <- w <- rnorm(100)
for(t in 2:100) x[t] <- 0.6*x[t-1] + w[t]
layout(1:2)
plot(x, type='l')
acf(x)
