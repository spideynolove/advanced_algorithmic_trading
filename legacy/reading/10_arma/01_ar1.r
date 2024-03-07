# model ----------
set.seed(1)
x <- w <- rnorm(100)

# # use alpha = 0.6 or -0.6
# for(t in 2:100) x[t] <- 0.6*x[t-1] + w[t]
for(t in 2:100) x[t] <- -0.6*x[t-1] + w[t]

# # ---------- AR(1) ----------
# # graph ----------
# layout(1:2)
# plot(x, type='l')
# acf(x)

# fit ar model to above model using maximum likelihood estimation -----
x.ar <- ar(x, method='mle')

# AR(1) mean order = 1 so it fit...
print(x.ar$order)

# ar param
print(x.ar$ar)

# standard error, variance, 95% confidence intervals
print(x.ar$ar+c(-1.96, 1.96)*c(sqrt(x.ar$asy.var)))

# --------------------------------------------------------------


