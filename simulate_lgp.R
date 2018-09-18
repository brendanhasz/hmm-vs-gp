
setwd("~/Code/hmm-vs-gp")

# Packages
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Parameters
N = 500
x = seq(0, 5, l=N)
rho = 0.5
alpha = 2
sigma = 0.3

# Simulate
sim_data = list(N=N, x=x, rho=rho, alpha=alpha, sigma=sigma)
sim_fit = stan(file='simulate_lgp.stan', data=sim_data, iter=1, 
               chains=1, seed=723456, algorithm="Fixed_param")

# Extract simulated output
f = extract(sim_fit)$f[1,]
y = extract(sim_fit)$y[1,]

# Plot
plot(x, f, type="l", lwd=2, ylim=c(0,1))
points(x, y, col="darkorange", pch=16, cex=0.5)
title(main='Simulated Gaussian Process')