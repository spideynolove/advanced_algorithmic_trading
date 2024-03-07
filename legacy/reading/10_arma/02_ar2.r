# common declare
set.seed(1)
x <- w <- rnorm(100)
for(t in 3:100) x[t] <- 0.666*x[t-1] - 0.333*x[t-2] + w[t]

# # ---------- AR(2) ----------
# # graph ----------
# layout(1:2)
# plot(x, type='l')
# acf(x)

# fit ar model to above model using maximum likelihood estimation -----
x.ar <- ar(x, method='mle')

# AR(2) mean order = 2 so it fit...
print(x.ar$order)

# ar param: 0.6961005 -0.3946280 : not too far off the true parameter
print(x.ar$ar)
